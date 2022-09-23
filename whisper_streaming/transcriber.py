#!/usr/bin/env python3

from logging import getLogger
from typing import Iterator, List, Optional, Union

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

from whisper_streaming.schema import ParsedChunk, WhisperConfig

logger = getLogger(__name__)


class WhisperStreamingTranscriber:
    def __init__(self, *, config: WhisperConfig):
        self.config: WhisperConfig = config
        self.model: Whisper = load_model(config.model_name, device=config.device)
        self.tokenizer = get_tokenizer(
            self.model.is_multilingual,
            language=config.language,
            task="transcribe",
        )
        self.dtype = torch.float16
        self.timestamp: float = 0.0
        self.input_stride = exact_div(
            N_FRAMES, self.model.dims.n_audio_ctx
        )  # mel frames per output token: 2
        self.time_precision = (
            self.input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)

        self.buffer_tokens = []
        self.buffer_mel = None

    def _get_decoding_options(
        self,
        *,
        t,
        prompt,
        beam_size: Optional[int],
        patience: float,
        best_of: Optional[int],
    ) -> DecodingOptions:
        return DecodingOptions(
            task="transcribe",
            language=None,
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
            fp16=True,
        )

    def _decode_with_fallback(
        self,
        *,
        segment: np.ndarray,
    ) -> List[DecodingResult]:
        assert len(self.config.temperatures) >= 1
        t = self.config.temperatures[0]

        _decode_options1: DecodingOptions = self._get_decoding_options(
            t=t,
            prompt=self.buffer_tokens,
            beam_size=self.config.beam_size,
            patience=0.0,
            best_of=None,
        )
        results: List[DecodingResult] = self.model.decode(segment, _decode_options1)  # type: ignore

        for t in self.config.temperatures[1:]:
            needs_fallback = [
                self.config.compression_ratio_threshold is not None
                and result.compression_ratio > self.config.compression_ratio_threshold
                or self.config.logprob_threshold is not None
                and result.avg_logprob < self.config.logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                _decode_options2: DecodingOptions = self._get_decoding_options(
                    t=t,
                    prompt=self.buffer_tokens,
                    beam_size=None,
                    patience=0.0,
                    best_of=self.config.best_of,
                )
                retries: List[DecodingResult] = self.model.decode(
                    segment[needs_fallback], _decode_options2  # type: ignore
                )
                for retry_index, original_index in enumerate(
                    np.nonzero(needs_fallback)[0]
                ):
                    results[original_index] = retries[retry_index]
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
        self, *, result, segment_duration
    ) -> Iterator[Union[ParsedChunk, int]]:
        tokens = torch.tensor(result.tokens)
        timestamp_tokens: torch.Tensor = tokens.ge(self.tokenizer.timestamp_begin)

        consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(
            1
        )

        if (
            len(consecutive) > 0
        ):  # if the output contains two consecutive timestamp tokens
            last_slice = 0
            for current_slice in consecutive:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = (
                    sliced_tokens[0].item() - self.tokenizer.timestamp_begin
                )
                end_timestamp_position = (
                    sliced_tokens[-1].item() - self.tokenizer.timestamp_begin
                )
                chunk = self._get_chunk(
                    start=self.timestamp
                    + start_timestamp_position * self.time_precision,
                    end=self.timestamp + end_timestamp_position * self.time_precision,
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
            self.buffer_tokens.extend(tokens[: last_slice + 1].tolist())
            self.timestamp += last_timestamp_position0 * self.time_precision
            yield last_timestamp_position0
        else:
            duration = segment_duration
            timestamps = tokens[timestamp_tokens.nonzero().flatten()]
            if len(timestamps) > 0:
                # no consecutive timestamps but it has a timestamp; use the last one.
                # single timestamp at the end means no speech after the last timestamp.
                last_timestamp_position = (
                    timestamps[-1].item() - self.tokenizer.timestamp_begin
                )
                duration = last_timestamp_position * self.time_precision
            chunk = self._get_chunk(
                start=self.timestamp,
                end=self.timestamp + duration,
                text_tokens=tokens,
                result=result,
            )
            if chunk is not None:
                yield chunk
            self.timestamp += duration

        if result.temperature > 0.5:
            # do not feed the prompt tokens if a high temperature was used
            del self.buffer_tokens
            self.buffer_tokens = []

    def transcribe(
        self,
        *,
        segment: np.ndarray,
    ) -> Iterator[ParsedChunk]:
        new_mel = log_mel_spectrogram(audio=segment).unsqueeze(0)
        if self.buffer_mel is None:
            mel = new_mel
        else:
            mel = torch.cat([self.buffer_mel, new_mel], dim=-1)
            self.buffer_mel = None

        seek: int = 0
        rest_start: Optional[int] = None
        while seek < mel.shape[-1]:
            logger.debug(
                f"seek={seek}, timestamp={self.timestamp}, rest_start={rest_start}"
            )
            segment = (
                pad_or_trim(mel[:, :, seek:], N_FRAMES)
                .to(self.model.device)  # type: ignore
                .to(self.dtype)
            )

            results = self._decode_with_fallback(
                segment=segment,
            )
            result = results[0]

            if self.config.no_speech_threshold is not None:
                if (result.no_speech_prob > self.config.no_speech_threshold) and not (
                    self.config.logprob_threshold is not None
                    and result.avg_logprob > self.config.logprob_threshold
                ):
                    seek += segment.shape[-1]
                    rest_start = None
                    continue

            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE
            last_timestamp_position: Optional[int] = None
            for v in self._deal_timestamp(
                result=result, segment_duration=segment_duration
            ):
                if isinstance(v, int):
                    last_timestamp_position = v
                else:
                    yield v
            if last_timestamp_position is None:
                seek += segment.shape[-1]
                rest_start = None
            else:
                seek += last_timestamp_position
                rest_start = seek

        logger.debug(f"Last rest_start={rest_start}")
        if rest_start is None:
            return

        self.buffer_mel = mel[:, :, rest_start:]
        del mel
