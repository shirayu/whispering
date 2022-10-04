#!/usr/bin/env python3

from logging import getLogger
from typing import Final, Iterator, Optional, Union

import numpy as np
import torch
from whisper import Whisper, load_model
from whisper.audio import (
    CHUNK_LENGTH,
    HOP_LENGTH,
    N_FRAMES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import get_tokenizer
from whisper.utils import exact_div

from schema import Context, ParsedChunk, WhisperConfig
from vad import VAD

logger = getLogger(__name__)


class WhisperStreamingTranscriber:
    def _set_dtype(self, fp16: bool):
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32
        if self.model.device == torch.device("cpu"):
            if torch.cuda.is_available():
                logger.warning("Performing inference on CPU when CUDA is available")
            if self.dtype == torch.float16:
                logger.warning("FP16 is not supported on CPU; using FP32 instead")
                self.dtype = torch.float32

        if self.dtype == torch.float32:
            self.fp16 = False

    def __init__(self, *, config: WhisperConfig):
        self.config: Final[WhisperConfig] = config
        self.model: Final[Whisper] = load_model(config.model_name, device=config.device)
        # language specified
        if config.language != "multilanguage":
            self.tokenizer = get_tokenizer(
                self.model.is_multilingual,
                language=config.language,
                task="transcribe",
            )
        # Mulilanguage transcripts
        else: 
            self.tokenizer = get_tokenizer(
                self.model.is_multilingual,
                task="transcribe",
            )
        self._set_dtype(config.fp16)
        self.input_stride: Final[int] = exact_div(
            N_FRAMES, self.model.dims.n_audio_ctx
        )  # mel frames per output token: 2
        self.time_precision: Final[float] = (
            self.input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)
        self.duration_pre_one_mel: Final[float] = CHUNK_LENGTH / HOP_LENGTH
        self.vad = VAD()

    def _get_decoding_options(
        self,
        *,
        t,
        prompt,
        beam_size: Optional[int],
        patience: Optional[float],
        best_of: Optional[int],
    ) -> DecodingOptions:
        if self.config.language != "multilanguage":
            return DecodingOptions(
                task="transcribe",
                language=self.config.language,
                temperature=t,
                sample_len=None,
                best_of=best_of,
                beam_size=beam_size,
                patience=patience,
                length_penalty=None,
                prompt=prompt,
                prefix=None,
                suppress_blank=True,
                suppress_tokens="-1",
                without_timestamps=False,
                max_initial_timestamp=1.0,
                fp16=self.fp16,
            )
        else:
            return DecodingOptions(
                task="transcribe",
                temperature=t,
                sample_len=None,
                best_of=best_of,
                beam_size=beam_size,
                patience=patience,
                length_penalty=None,
                prompt=prompt,
                prefix=None,
                suppress_blank=True,
                suppress_tokens="-1",
                without_timestamps=False,
                max_initial_timestamp=1.0,
                fp16=self.fp16,
            )

    def _decode_with_fallback(
        self,
        *,
        segment: torch.Tensor,
        ctx: Context,
    ) -> DecodingResult:
        assert len(ctx.temperatures) >= 1
        decode_result: Optional[DecodingResult] = None

        for t in ctx.temperatures:
            _decode_options: DecodingOptions = self._get_decoding_options(
                t=t,
                prompt=ctx.buffer_tokens,
                beam_size=ctx.beam_size if t <= 0 else None,
                patience=ctx.patience if t <= 0 else None,
                best_of=ctx.best_of if t < 0 else None,
            )
            logger.debug(f"DecodeOptions: {_decode_options}")
            decode_result = self.model.decode(
                segment,
                _decode_options,
            )  # type: ignore
            assert decode_result is not None
            needs_fallback: bool = False
            if (
                ctx.compression_ratio_threshold is not None
                and decode_result.compression_ratio > ctx.compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                ctx.logprob_threshold is not None
                and decode_result.avg_logprob < ctx.logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        assert isinstance(decode_result, DecodingResult)
        return decode_result

    def _get_chunk(
        self,
        *,
        start: float,
        end: float,
        text_tokens: torch.Tensor,
        result: DecodingResult,
    ) -> Optional[ParsedChunk]:
        text = self.tokenizer.decode(
            [token for token in text_tokens if token < self.tokenizer.eot]  # type: ignore
        )
        if len(text.strip()) == 0:  # skip empty text output
            return

        return ParsedChunk(
            start=start,
            end=end,
            text=text,
            tokens=result.tokens,
            temperature=result.temperature,
            avg_logprob=result.avg_logprob,
            compression_ratio=result.compression_ratio,
            no_speech_prob=result.no_speech_prob,
        )

    def _deal_timestamp(
        self,
        *,
        result,
        segment_duration,
        ctx: Context,
    ) -> Iterator[Union[ParsedChunk, int]]:
        tokens = torch.tensor(result.tokens)
        timestamp_tokens: torch.Tensor = tokens.ge(self.tokenizer.timestamp_begin)

        consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(
            1
        )

        if (
            len(consecutive) > 0
        ):  # if the output contains two consecutive timestamp tokens
            logger.debug(f"Length of consecutive: {len(consecutive)}")
            last_slice = 0
            for current_slice in consecutive:
                sliced_tokens = tokens[last_slice:current_slice]
                logger.debug(f" last_slice={last_slice}, current_slice={current_slice}")
                start_timestamp_position = (
                    sliced_tokens[0].item() - self.tokenizer.timestamp_begin
                )
                end_timestamp_position = (
                    sliced_tokens[-1].item() - self.tokenizer.timestamp_begin
                )
                chunk = self._get_chunk(
                    start=ctx.timestamp
                    + start_timestamp_position * self.time_precision,
                    end=ctx.timestamp + end_timestamp_position * self.time_precision,
                    text_tokens=sliced_tokens[1:-1],
                    result=result,
                )
                if chunk is not None:
                    yield chunk
                last_slice = current_slice
            last_timestamp_position0: int = (
                tokens[last_slice - 1].item()
                - self.tokenizer.timestamp_begin  # type:ignore
            )
            ctx.buffer_tokens.extend(tokens[: last_slice + 1].tolist())
            ctx.timestamp += last_timestamp_position0 * self.time_precision
            yield last_timestamp_position0
        else:
            duration = segment_duration
            timestamps = tokens[timestamp_tokens.nonzero().flatten()]
            logger.debug(f"Length of consecutive: 0, timestamps: {timestamps}")
            if (
                len(timestamps) > 0
                and timestamps[-1].item() != self.tokenizer.timestamp_begin
            ):
                # no consecutive timestamps but it has a timestamp; use the last one.
                # single timestamp at the end means no speech after the last timestamp.
                last_timestamp_position = (
                    timestamps[-1].item() - self.tokenizer.timestamp_begin
                )
                duration = last_timestamp_position * self.time_precision
            logger.debug(f"segment_duration: {segment_duration}, Duration: {duration}")
            chunk = self._get_chunk(
                start=ctx.timestamp,
                end=ctx.timestamp + duration,
                text_tokens=tokens,
                result=result,
            )
            if chunk is not None:
                yield chunk
            ctx.timestamp += duration

        if result.temperature > ctx.buffer_threshold:
            # do not feed the prompt tokens if a high temperature was used
            del ctx.buffer_tokens
            ctx.buffer_tokens = []
        logger.debug(f"Length of buffer: {len(ctx.buffer_tokens)}")

    def transcribe(
        self,
        *,
        audio: np.ndarray,
        ctx: Context,
    ) -> Iterator[ParsedChunk]:
        logger.debug(f"{len(audio)}")

        if ctx.vad:
            x = [
                v
                for v in self.vad(
                    audio=audio,
                    total_block_number=1,
                    threshold=ctx.vad_threshold,
                )
            ]
            if len(x) == 0:  # No speech
                logger.debug("No speech")
                ctx.timestamp += len(audio) / N_FRAMES * self.duration_pre_one_mel
                return

        new_mel = log_mel_spectrogram(audio=audio)
        logger.debug(f"Incoming new_mel.shape: {new_mel.shape}")
        if ctx.buffer_mel is None:
            mel = new_mel
        else:
            logger.debug(f"buffer_mel.shape: {ctx.buffer_mel.shape}")
            mel = torch.cat([ctx.buffer_mel, new_mel], dim=-1)
            ctx.buffer_mel = None
        logger.debug(f"mel.shape: {mel.shape}")

        seek: int = 0
        while seek < mel.shape[-1]:
            logger.debug(f"seek: {seek}")
            if mel.shape[-1] - seek < N_FRAMES:
                logger.debug(
                    f"mel.shape ({mel.shape[-1]}) - seek ({seek}) < N_FRAMES ({N_FRAMES})"
                )
                if ctx.allow_padding:
                    logger.warning("Padding is not expected while speaking")
                else:
                    logger.debug("No padding")
                    break

            segment: torch.Tensor = (
                pad_or_trim(mel[:, seek:], N_FRAMES)
                .to(self.model.device)  # type: ignore
                .to(self.dtype)
            )

            logger.debug(
                f"seek={seek}, timestamp={ctx.timestamp}, "
                f"mel.shape: {mel.shape}, segment.shape: {segment.shape}"
            )
            result = self._decode_with_fallback(
                segment=segment,
                ctx=ctx,
            )
            logger.debug(
                f"Result: temperature={result.temperature:.2f}, no_speech_prob={result.no_speech_prob:.2f}, "
                f"avg_logprob={result.avg_logprob:.2f}"
            )

            if ctx.no_speech_threshold is not None:
                if (result.no_speech_prob > ctx.no_speech_threshold) and not (
                    ctx.logprob_threshold is not None
                    and result.avg_logprob > ctx.logprob_threshold
                ):
                    seek += segment.shape[-1]
                    logger.debug(
                        f"Skip: {segment.shape[-1]}, new seek={seek}, mel.shape: {mel.shape}"
                    )
                    continue

            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE
            last_timestamp_position: Optional[int] = None
            for v in self._deal_timestamp(
                result=result,
                segment_duration=segment_duration,
                ctx=ctx,
            ):
                if isinstance(v, int):
                    last_timestamp_position = v
                else:
                    yield v
            if last_timestamp_position is None:
                seek += segment.shape[-1]
            else:
                seek += last_timestamp_position * self.input_stride
            logger.debug(f"new seek={seek}, mel.shape: {mel.shape}")

        if mel.shape[-1] - seek <= 0:
            logger.debug(f"ctx.buffer_mel is None ({mel.shape}, {seek})")
            return
        ctx.buffer_mel = mel[:, seek:]
        assert ctx.buffer_mel is not None
        logger.debug(f"ctx.buffer_mel.shape: {ctx.buffer_mel.shape}")
        del mel
