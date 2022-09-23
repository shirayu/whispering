#!/usr/bin/env python3

from typing import List, Optional

import numpy as np
import torch
from whisper import Whisper, load_model
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import get_tokenizer

from whisper_streaming.schema import WhisperConfig


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

    def _get_decoding_options(
        self,
        *,
        t,
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
            prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens="-1",
            without_timestamps=False,
            max_initial_timestamp=0.0,
            fp16=True,
        )

    def _decode_with_fallback(self, *, segment: np.ndarray) -> List[DecodingResult]:
        assert len(self.config.temperatures) >= 1
        t = self.config.temperatures[0]

        _decode_options1: DecodingOptions = self._get_decoding_options(
            t=t,
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

    def transcribe(
        self,
        *,
        segment: np.ndarray,
    ) -> Optional[DecodingResult]:
        log_spec = log_mel_spectrogram(audio=segment).unsqueeze(0)
        segment = (
            pad_or_trim(log_spec, N_FRAMES)
            .to(self.model.device)  # type:ignore
            .to(self.dtype)  # type:ignore
        )
        results = self._decode_with_fallback(segment=segment)
        result = results[0]

        if self.config.no_speech_threshold is not None:
            if (result.no_speech_prob > self.config.no_speech_threshold) and not (
                self.config.logprob_threshold is not None
                and result.avg_logprob > self.config.logprob_threshold
            ):
                return

        # FIXME: work with timestamp

        return result
