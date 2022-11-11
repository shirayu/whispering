import argparse
from pathlib import Path

import whisper

download_path = str(Path("~/.cache/whisper").expanduser())

print(f"Downloading whisper models to {download_path}")

# get models to download from argparse
parser = argparse.ArgumentParser()

# model is a list of strings
parser.add_argument("--models", nargs="+", default=[])

args = parser.parse_args()

print(f"Downloading models: {args.models}")

for m in args.models:
    print(f"Downloading model {m} to {download_path}")
    whisper._download(whisper._MODELS[m], download_path, False)
