#!/usr/bin/env python3
import asyncio
from logging import getLogger
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

                logger.debug(f"Loop #: {idx}")
                segment = None
                try:
                    segment = await asyncio.wait_for(g(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                if segment is not None:
                    logger.debug(f"Segment size: {len(segment)}")
                    await ws.send(segment)
                    logger.debug("Sent")

                async def recv():
                    return await ws.recv()

                while True:
                    try:
                        c = await asyncio.wait_for(recv(), timeout=0.5)
                        chunk = ParsedChunk.parse_raw(c)
                        print(f"{chunk.start:.2f}->{chunk.end:.2f}\t{chunk.text}")
                    except asyncio.TimeoutError:
                        break

                idx += 1


async def run_websocket_client(*, opts) -> None:
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
