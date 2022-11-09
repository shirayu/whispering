#!/usr/bin/env python3
import asyncio
import json
import sys
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import sounddevice as sd
import websockets
from whisper.audio import N_FRAMES, SAMPLE_RATE

from whispering.schema import ParsedChunk, StdoutWriter
from whispering.transcriber import Context

logger = getLogger(__name__)


def sd_callback(indata, frames, time, status):
    if status:
        logger.warning(status)
    logger.debug(f"Have indata: {len(indata)} {type(indata)}")
    ravelled = indata.ravel().tobytes()
    logger.debug(f"Have ravelled: {len(ravelled)} {type(ravelled)}")
    loop.call_soon_threadsafe(q.put_nowait, indata.ravel().tobytes())


async def transcribe_from_mic_and_send(
    *,
    sd_device: Optional[Union[int, str]],
    num_block: int,
    host: str,
    port: int,
    ctx: Context,
    path_out: Union[Path, StdoutWriter],
) -> None:
    uri = f"ws://{host}:{port}"

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=N_FRAMES * num_block,
        device=sd_device,
        dtype="float32",
        channels=1,
        callback=sd_callback,
    ), path_out.open("w") as outf:
        async with websockets.connect(uri, max_size=999999999) as ws:  # type:ignore
            logger.debug("Sent context")
            v: str = ctx.json()
            msg = """{"context": """ + v + """}"""
            logger.debug(f"Sending context\n{msg}")
            await ws.send(msg)

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
                    logger.debug(f"Segment size: {len(segment)} {type(segment)}")
                    await ws.send(segment)
                    logger.debug("Sent")

                async def recv():
                    return await ws.recv()

                while True:
                    try:
                        c = await asyncio.wait_for(recv(), timeout=0.5)
                        c_json = json.loads(c)
                        if (err := c_json.get("error")) is not None:
                            sys.stderr.write(f"Error: {err}\n")
                            sys.exit(1)
                        chunk = ParsedChunk.parse_obj(c_json)
                        outf.write(
                            f"{chunk.start:.2f}->{chunk.end:.2f}\t{chunk.text}\n"
                        )
                        outf.flush()
                    except asyncio.TimeoutError:
                        break
                idx += 1


async def run_websocket_client(
    *,
    sd_device: Optional[Union[int, str]],
    num_block: int,
    host: str,
    port: int,
    ctx: Context,
    no_progress: bool,
    path_out: Union[Path, StdoutWriter],
) -> None:
    global q
    global loop
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    await transcribe_from_mic_and_send(
        sd_device=sd_device,
        num_block=num_block,
        host=host,
        port=port,
        ctx=ctx,
        path_out=path_out,
    )
