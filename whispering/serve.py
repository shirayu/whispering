#!/usr/bin/env python3

import asyncio
from logging import getLogger

import numpy as np
import websockets

from whispering.transcriber import WhisperStreamingTranscriber

logger = getLogger(__name__)


async def serve_with_websocket_main(websocket):
    global g_wsp
    idx: int = 0

    while True:
        logger.debug(f"Segment #: {idx}")
        message = await websocket.recv()

        if isinstance(message, str):
            logger.debug(f"Got str: {message}")
            continue

        logger.debug(f"Message size: {len(message)}")
        segment = np.frombuffer(message, dtype=np.float32)
        for chunk in g_wsp.transcribe(segment=segment):
            await websocket.send(chunk.json())
        idx += 1


async def serve_with_websocket(
    *,
    wsp: WhisperStreamingTranscriber,
    host: str,
    port: int,
):
    logger.info(f"Serve at {host}:{port}")
    logger.info("Make secure with your responsibility!")
    global g_wsp
    g_wsp = wsp

    try:
        async with websockets.serve(  # type: ignore
            serve_with_websocket_main,
            host=host,
            port=port,
            max_size=999999999,
        ):
            await asyncio.Future()
    except KeyboardInterrupt:
        pass
