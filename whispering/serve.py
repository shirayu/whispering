#!/usr/bin/env python3

import asyncio
import json
from logging import getLogger
from typing import Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedOK

from whispering.transcriber import Context, WhisperStreamingTranscriber

logger = getLogger(__name__)


async def serve_with_websocket_main(websocket):
    global g_wsp
    idx: int = 0
    ctx: Optional[Context] = None

    while True:
        logger.debug(f"Audio #: {idx}")
        try:
            message = await websocket.recv()
        except ConnectionClosedOK:
            break

        if isinstance(message, str):
            logger.debug(f"Got str: {message}")
            d = json.loads(message)
            v = d.get("context")
            if v is not None:
                ctx = Context.parse_obj(v)
            else:
                await websocket.send(
                    json.dumps(
                        {
                            "error": "unsupported message",
                        }
                    )
                )
                return
            continue

        logger.debug(f"Message size: {len(message)}")
        audio = np.frombuffer(message, dtype=np.float32)
        if ctx is None:
            await websocket.send(
                json.dumps(
                    {
                        "error": "no context",
                    }
                )
            )
            return
        for chunk in g_wsp.transcribe(
            audio=audio,  # type: ignore
            ctx=ctx,
        ):
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
