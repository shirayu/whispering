import argparse
from pathlib import Path

import whisper

download_path = str(Path("~/.cache/whisper").expanduser())

# get models to download from argparse
parser = argparse.ArgumentParser()

# model is a list of strings
parser.add_argument("--models", nargs="+", default=[])

args = parser.parse_args()

for m in args.models:
    whisper._download(whisper._MODELS[m], download_path, False)
