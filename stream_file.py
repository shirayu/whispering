import argparse
import asyncio
import base64
import datetime
import json
import logging
import os
import time
from pathlib import Path

import coloredlogs
import soundfile as sf
import uuid
import websockets

logger = logging.getLogger(__name__)
coloredlogs.install(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)s %(message)s",
)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="file to send")
parser.add_argument(
    "-u",
    "--url",
    help="url to connect to",
    default="ws://ec2-3-71-72-35.eu-central-1.compute.amazonaws.com",
)
parser.add_argument(
    "-c",
    "--chunk-size",
    help="chunk size in bytes",
    default=100000,
    type=int,
)

initial_context = {
    "context": {
        "protocol_version": 6002,
        "timestamp": 0.0,
        "buffer_tokens": [],
        "buffer_mel": None,
        "nosoeech_skip_count": None,
        "temperatures": [0.2],
        "patience": None,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_captions_threshold": 0.6,
        "best_of": 5,
        "beam_size": 5,
        "no_speech_threshold": 0.6,
        "buffer_threshold": 0.5,
        "vad_threshold": 0.5,
        "max_nospeech_skip": 16,
        "data_type": "float32",
    }
}


def wave_generator():
    for data in sf.blocks(_tfile, blocksize=args.chunk_size, dtype="float32"):
        yield data.tobytes()


async def send_receive():
    logger.info(f"Connecting websocket to url ${args.url}")
    wave_gen = wave_generator()
    start_time = time.time()
    async with websockets.connect(
        args.url,
        ping_interval=5,
        ping_timeout=20,
    ) as _ws:
        await asyncio.sleep(0.1)
        # print("Receiving SessionBegins ...")
        logger.info(f"Sending initial context")
        await _ws.send(json.dumps(initial_context))
        # session_begins = await _ws.recv()
        # print(session_begins)
        logger.info("Sending messages ...")

        async def send():
            i = 1
            while True:
                try:
                    # print("Trying to send")
                    segment = next(wave_gen)
                    logger.info(f"Sending segment {i} of length {len(segment)}")
                    if len(segment) == args.chunk_size * 4:
                        msg = segment

                    else:
                        # b64-encode bytes
                        payload = base64.b64encode(segment).decode("utf-8")
                        msg = json.dumps({"last_message": payload})

                    i += 1

                    await _ws.send(msg)
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(e)
                    assert e.code == 4008
                    break
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Caught send exception {type(e)}")
                    logger.error(e)
                    break
                await asyncio.sleep(1e-5)

            return True

        async def receive():
            i = 0
            full_text = ""
            while True:
                try:
                    result_str = await _ws.recv()
                    rjs = json.loads(result_str)
                    if "close_connection" in rjs:
                        logger.info("Closing connection")
                        break

                    res = rjs["text"]
                    logger.info(f"Result {i}:\t{res.strip()}")
                    full_text += res
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(e)
                    assert e.code == 4008
                    break
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Caught exception {type(e)}")
                    break

                i += 1

            audio_duration = sf.info(_tfile).duration
            transcription_time = time.time() - start_time

            logger.info(f"Full text:\n---\n{full_text.strip()}\n---")
            logger.info(f"Performed inference on {args.url}")
            logger.info(
                f"Duration of audio file: {audio_duration:.3f} s - "
                f"transcription time: {transcription_time:.3f} s - "
                f"transcription speed: {audio_duration / transcription_time:.3f} x real time"
            )

        send_result, receive_result = await asyncio.gather(send(), receive())


def convert_to_wav(tfile_name):
    os.system(
        f"ffmpeg -hide_banner -loglevel error "
        f"-i {args.file} -acodec pcm_s16le -ac 1 -ar 16000 {tfile_name}"
    )
    logger.info(f"Converted file to {tfile_name}")


if __name__ == "__main__":
    args = parser.parse_args()

    _tfile = f"tmp_{args.file}_{uuid.uuid4()}.wav"

    try:
        convert_to_wav(_tfile)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(send_receive())
    finally:
        Path(_tfile).unlink()
