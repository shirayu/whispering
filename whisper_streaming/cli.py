#!/usr/bin/env python3

import argparse
import queue
from logging import DEBUG, INFO, basicConfig, getLogger
from typing import Optional, Union

import sounddevice as sd
import torch
from whisper import available_models
from whisper.audio import N_FRAMES, SAMPLE_RATE
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

from whisper_streaming.schema import WhisperConfig
from whisper_streaming.transcriber import WhisperStreamingTranscriber

logger = getLogger(__name__)


def transcribe_from_mic(
    *,
    config: WhisperConfig,
    sd_device: Optional[Union[int, str]],
    num_block: int,
) -> None:
    wsp = WhisperStreamingTranscriber(config=config)
    q = queue.Queue()

    def sd_callback(indata, frames, time, status):
        if status:
            logger.warning(status)
        q.put(indata.ravel())

    logger.info("Ready to transcribe")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=N_FRAMES * num_block,
        device=sd_device,
        dtype="float32",
        channels=1,
        callback=sd_callback,
    ):
        idx: int = 0
        while True:
            logger.debug(f"Segment: {idx}")
            segment = q.get()
            for chunk in wsp.transcribe(segment=segment):
                print(f"{chunk.start:.2f}->{chunk.end:.2f}\t{chunk.text}")
            idx += 1


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=available_models(),
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for PyTorch inference",
    )
    parser.add_argument(
        "--beam_size",
        "-b",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_block",
        "-n",
        type=int,
        default=20,
        help="Number of operation unit. Larger values can improve accuracy but consume more memory.",
    )
    parser.add_argument(
        "--mic",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    opts = get_opts()
    basicConfig(
        level=DEBUG if opts.debug else INFO,
        format="[%(asctime)s] %(module)s.%(funcName)s %(levelname)s -> %(message)s",
    )

    if opts.beam_size <= 0:
        opts.beam_size = None
    try:
        opts.mic = int(opts.mic)
    except Exception:
        pass

    config = WhisperConfig(
        model_name=opts.model,
        language=opts.language,
        device=opts.device,
        beam_size=opts.beam_size,
    )
    transcribe_from_mic(
        config=config,
        sd_device=opts.mic,
        num_block=opts.num_block,
    )


if __name__ == "__main__":
    main()
