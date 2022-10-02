#!/usr/bin/env python3

from typing import Iterator

import torch
from whisper.audio import N_FRAMES, SAMPLE_RATE

from whispering.schema import SpeechSegment


class VAD:
    def __init__(
        self,
    ):
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
        )

    def __call__(
        self,
        *,
        segment: torch.Tensor,
        thredhold: float = 0.5,
    ) -> Iterator[SpeechSegment]:
        # segment.shape should be multiple of (N_FRAMES,)

        def my_ret(
            *,
            start_block_idx: int,
            idx: int,
        ) -> SpeechSegment:
            return SpeechSegment(
                start_block_idx=start_block_idx,
                end_block_idx=idx,
                segment=segment[N_FRAMES * start_block_idx : N_FRAMES * idx],
            )

        block_size: int = int(segment.shape[0] / N_FRAMES)

        start_block_idx = None
        for idx in range(block_size):
            start: int = N_FRAMES * idx
            end: int = N_FRAMES * (idx + 1)
            vad_prob = self.vad_model(
                torch.from_numpy(segment[start:end]),
                SAMPLE_RATE,
            ).item()
            if vad_prob > thredhold:
                if start_block_idx is None:
                    start_block_idx = idx
            else:
                if start_block_idx is not None:
                    yield my_ret(
                        start_block_idx=start_block_idx,
                        idx=idx,
                    )
                    start_block_idx = None
        if start_block_idx is not None:
            yield my_ret(
                start_block_idx=start_block_idx,
                idx=block_size,
            )
