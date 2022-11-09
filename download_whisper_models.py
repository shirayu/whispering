from pathlib import Path

import whisper

download_path = str(Path("~/.cache/whisper").expanduser())

for m in ["tiny", "base", "small", "medium", "large"]:
    whisper._download(whisper._MODELS[m], download_path, False)
