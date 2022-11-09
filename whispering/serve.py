#!/usr/bin/env python3

import asyncio
import base64
import json
from logging import getLogger
from typing import Final, Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedOK

from whispering.schema import CURRENT_PROTOCOL_VERSION, Context
from whispering.transcriber import WhisperStreamingTranscriber

logger = getLogger(__name__)

MIN_PROTOCOL_VERSION: Final[int] = int("000_006_000")
MAX_PROTOCOL_VERSION: Final[int] = CURRENT_PROTOCOL_VERSION


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

        force_padding = False

        if isinstance(message, str) and not ctx:
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

            if ctx.protocol_version < MIN_PROTOCOL_VERSION:
                await websocket.send(
                    json.dumps(
                        {
                            "error": f"protocol_version is older than {MIN_PROTOCOL_VERSION}"
                        }
                    )
                )
            elif ctx.protocol_version > MAX_PROTOCOL_VERSION:
                await websocket.send(
                    json.dumps(
                        {
                            "error": f"protocol_version is newer than {MAX_PROTOCOL_VERSION}"
                        }
                    )
                )
                return

            continue

        elif isinstance(message, str) and ctx:
            d = json.loads(message)
            logger.warning(f"Received last message")
            if "last_message" in d:
                # b64-decode message
                message = base64.b64decode(d["last_message"])

            force_padding = True

        logger.debug(f"Message size: {len(message)}")
        if ctx is None:
            await websocket.send(
                json.dumps(
                    {
                        "error": "no context",
                    }
                )
            )
            return
        logger.debug(f"Have message of size {len(message)} data type {ctx.data_type}")
        # bytes to np array
        # audio = np.frombuffer(message, dtype=ctx.data_type)

        audio = np.frombuffer(message, dtype=np.dtype(ctx.data_type)).astype(np.float32)
        logger.warning(f"Have audio shape {audio.shape}")

        for chunk in g_wsp.transcribe(
            audio=audio, ctx=ctx, force_padding=force_padding  # type: ignore
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
