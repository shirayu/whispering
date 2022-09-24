#!/usr/bin/env python3
import argparse
import asyncio
import json
from logging import DEBUG, INFO, basicConfig, getLogger
from typing import Optional, Union

import sounddevice as sd
import websockets
from whisper.audio import N_FRAMES, SAMPLE_RATE

from whisper_streaming.schema import ParsedChunk

logger = getLogger(__name__)


def sd_callback(indata, frames, time, status):
    if status:
        logger.warning(status)
    loop.call_soon_threadsafe(q.put_nowait, indata.ravel().tobytes())


async def transcribe_from_mic_and_send(
    *,
    sd_device: Optional[Union[int, str]],
    num_block: int,
    host: str,
    port: int,
) -> None:
    uri = f"ws://{host}:{port}"

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=N_FRAMES * num_block,
        device=sd_device,
        dtype="float32",
        channels=1,
        callback=sd_callback,
    ):
        async with websockets.connect(uri, max_size=999999999) as ws:  # type:ignore
            idx: int = 0
            while True:

                async def g():
                    return await q.get()

                logger.debug(f"Segment #: {idx}")
                try:
                    segment = await asyncio.wait_for(g(), timeout=3.0)
                except asyncio.TimeoutError:
                    await ws.send("dummy")
                    continue

                logger.debug(f"Segment size: {len(segment)}")
                await ws.send(segment)
                logger.debug("Sent")

                d = await ws.recv()
                for c in json.loads(d):
                    chunk = ParsedChunk.parse_obj(c)
                    print(f"{chunk.start:.2f}->{chunk.end:.2f}\t{chunk.text}")
                idx += 1


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        required=True,
        help="host of websocker server",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port number of websocker server",
    )

    parser.add_argument(
        "--mic",
    )
    parser.add_argument(
        "--num_block",
        "-n",
        type=int,
        default=160,
        help="Number of operation unit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


async def main() -> None:
    opts = get_opts()
    basicConfig(
        level=DEBUG if opts.debug else INFO,
        format="[%(asctime)s] %(module)s.%(funcName)s:%(lineno)d %(levelname)s -> %(message)s",
    )
    global q
    global loop
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    await transcribe_from_mic_and_send(
        sd_device=opts.mic,
        num_block=opts.num_block,
        host=opts.host,
        port=opts.port,
    )


if __name__ == "__main__":
    asyncio.run(main())
