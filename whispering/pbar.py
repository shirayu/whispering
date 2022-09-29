#!/usr/bin/env python3
import threading
import time

from tqdm import tqdm
from whisper.audio import CHUNK_LENGTH, HOP_LENGTH


class ProgressBar(threading.Thread):
    def __init__(self, *, num_block: int):
        super().__init__()
        self.started = threading.Event()
        self.alive = True
        self.start()
        self.started.set()  # start
        self.num_block = num_block

    def __del__(self):
        self.kill()

    def kill(self):
        self.started.set()
        self.alive = False
        self.join()

    def end(self):
        self.started.clear()

    def run(self):
        self.started.wait()
        with tqdm(
            total=self.num_block,
            leave=False,
            bar_format="Listening (Elapsed: {elapsed}, Estimated remaining: {remaining})",
        ) as t:
            for _ in range(self.num_block):
                time.sleep(CHUNK_LENGTH / HOP_LENGTH)
                t.update(1)
                if not self.alive:
                    break
