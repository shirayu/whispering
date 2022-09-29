#!/usr/bin/env python3

from logging import getLogger
from typing import Final, Iterator, List, Optional, Union

import numpy as np
import torch
from whisper import Whisper, load_model
from whisper.audio import (
    HOP_LENGTH,
    N_FRAMES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import get_tokenizer
from whisper.utils import exact_div

from whispering.schema import Context, ParsedChunk, WhisperConfig

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
        self.tokenizer = get_tokenizer(
            self.model.is_multilingual,
            language=config.language,
            task="transcribe",
        )
        self._set_dtype(config.fp16)
        self.input_stride: Final[int] = exact_div(
            N_FRAMES, self.model.dims.n_audio_ctx
        )  # mel frames per output token: 2
        self.time_precision: Final[float] = (
            self.input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)

    def _get_decoding_options(
        self,
        *,
        t,
        prompt,
        beam_size: Optional[int],
        patience: Optional[float],
        best_of: Optional[int],
    ) -> DecodingOptions:
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
            max_initial_timestamp=0.0,
            fp16=self.fp16,
        )

    def _decode_with_fallback(
        self,
        *,
        segment: np.ndarray,
        ctx: Context,
    ) -> List[DecodingResult]:
        assert len(ctx.temperatures) >= 1
        t = ctx.temperatures[0]
        logger.debug(f"temperature: {t}")

        _decode_options1: DecodingOptions = self._get_decoding_options(
            t=t,
            prompt=ctx.buffer_tokens,
            beam_size=ctx.beam_size,
            patience=None,
            best_of=None,
        )
        results: List[DecodingResult] = self.model.decode(segment, _decode_options1)  # type: ignore

        for t in ctx.temperatures[1:]:
            needs_fallback = [
                ctx.compression_ratio_threshold is not None
                and result.compression_ratio > ctx.compression_ratio_threshold
                or ctx.logprob_threshold is not None
                and result.avg_logprob < ctx.logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                logger.debug(
                    f"Fall back with temperature: {t}, needs_fallback: {needs_fallback}"
                )
                _decode_options2: DecodingOptions = self._get_decoding_options(
                    t=t,
                    prompt=ctx.buffer_tokens,
                    beam_size=None,
                    patience=ctx.patience,
                    best_of=ctx.best_of,
                )
                retries: List[DecodingResult] = self.model.decode(
                    segment[needs_fallback], _decode_options2  # type: ignore
                )
                for retry_index, original_index in enumerate(
                    np.nonzero(needs_fallback)[0]
                ):
                    results[original_index] = retries[retry_index]
            else:
                break
        logger.debug(f"# of results: {len(results)}")
        return results

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
            if len(timestamps) > 0:
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
        segment: np.ndarray,
        ctx: Context,
    ) -> Iterator[ParsedChunk]:
        new_mel = log_mel_spectrogram(audio=segment).unsqueeze(0)
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
            segment = (
                pad_or_trim(mel[:, :, seek:], N_FRAMES)
                .to(self.model.device)  # type: ignore
                .to(self.dtype)
            )
            if not ctx.allow_padding and segment.shape[-1] > mel.shape[-1]:
                logger.warning("Padding is not expected while speaking")

            logger.debug(
                f"seek={seek}, timestamp={ctx.timestamp}, "
                f"mel.shape: {mel.shape}, segment.shape: {segment.shape}"
            )
            results = self._decode_with_fallback(
                segment=segment,
                ctx=ctx,
            )
            result = results[0]
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

            if (not ctx.allow_padding) and (mel.shape[-1] - seek < N_FRAMES):
                break

        if mel.shape[-1] - seek <= 0:
            logger.debug(f"ctx.buffer_mel is None ({mel.shape}, {seek})")
            return
        ctx.buffer_mel = mel[:, :, seek:]
        assert ctx.buffer_mel is not None
        logger.debug(f"ctx.buffer_mel.shape: {ctx.buffer_mel.shape}")
        del mel
