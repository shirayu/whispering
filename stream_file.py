import sys
import wave

import numpy
import websockets
import asyncio
import base64
import json
import logging
import soundfile as sf

logging.basicConfig(level=logging.INFO)

# URL = "ws://ec2-3-71-72-35.eu-central-1.compute.amazonaws.com"

# URL = "ws://localhost:1111/ws"

URL = "ws://localhost:8000"

FNAME = sys.argv[1]


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

# # read binary file in chunks
# def wave_generator(chunk_size=50000):
#     with open(FNAME, "rb") as f:
#         while True:
#             data = f.read(chunk_size)
#             if not data:
#                 break
#             yield data


# def wave_generator():
#     with wave.open(FNAME, "rb") as f:
#         while True:
#             data = f.readframes(100000)
#             if not data:
#                 break
#             yield data

CHUNK_SIZE = 100000


def wave_generator(chunk_size: int = CHUNK_SIZE):
    for data in sf.blocks(FNAME, blocksize=chunk_size, dtype="float32"):
        yield data.tobytes()


# def wave_generator(chunk_size=60000):
#     from scipy.io.wavfile import read
#
#     a = read(FNAME)
#     data = numpy.array(a[1], dtype=numpy.float32)
#     print(f"Data: {data.shape}")
#     for i in range(0, data.shape[0], chunk_size):
#         print(f"From {i} to {i+chunk_size}")
#         yield data[i : i + chunk_size]


async def send_receive():
    print(f"Connecting websocket to url ${URL}")
    wave_gen = wave_generator()
    async with websockets.connect(
        URL,
        ping_interval=5,
        ping_timeout=20,
    ) as _ws:
        await asyncio.sleep(0.1)
        # print("Receiving SessionBegins ...")
        await _ws.send(json.dumps(initial_context))
        # session_begins = await _ws.recv()
        # print(session_begins)
        print("Sending messages ...")

        async def send():
            print("Gonna send")
            while True:
                try:
                    # print("Trying to send")
                    segment = next(wave_gen)
                    print(f"Sending segment of length {len(segment)}")
                    if len(segment) == CHUNK_SIZE * 4:
                        msg = segment

                    else:
                        # b64-encode bytes
                        payload = base64.b64encode(segment).decode("utf-8")
                        msg = json.dumps({"last_message": payload})

                    await _ws.send(msg)
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    print(f"Caught send exception {type(e)}")
                    print(e)
                    break
                await asyncio.sleep(1)

            return True

        async def receive():
            while True:
                try:
                    result_str = await _ws.recv()
                    # print(f"Result: {result_str}")
                    rjs = json.loads(result_str)
                    print(f"Result: '{rjs['text']}'")
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    print(f"Caught exception {type(e)}")
                    break

        send_result, receive_result = await asyncio.gather(send(), receive())


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_receive())
