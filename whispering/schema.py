#!/usr/bin/env python3

import sys
from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel, root_validator


class WhisperConfig(BaseModel):
    model_name: str
    device: str
    language: str
    fp16: bool = True

    @root_validator
    def validate_model_name(cls, values):
        if values["model_name"].endswith(".en") and values["language"] not in {
            "en",
            "English",
        }:
            raise ValueError("English only model")
        return values


class Context(BaseModel, arbitrary_types_allowed=True):
    timestamp: float = 0.0
    buffer_tokens: List[torch.Tensor] = []
    buffer_mel: Optional[torch.Tensor] = None
    vad: bool = True

    temperatures: List[float]
    allow_padding: bool = False
    patience: Optional[float] = None
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_captions_threshold: Optional[float] = 0.6
    best_of: int = 5
    beam_size: Optional[int] = None
    no_speech_threshold: Optional[float] = 0.6
    logprob_threshold: Optional[float] = -1.0
    compression_ratio_threshold: Optional[float] = 2.4
    buffer_threshold: Optional[float] = 0.5
    vad_threshold: float = 0.5


class ParsedChunk(BaseModel):
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class SpeechSegment(BaseModel, arbitrary_types_allowed=True):
    start_block_idx: int
    end_block_idx: int
    audio: np.ndarray


class StdoutWriter:
    def open(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        sys.stdout.flush()

    def write(self, text, *args, **kwargs):
        sys.stdout.write(text)
