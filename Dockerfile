FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# install python and pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev

# install dependencies
RUN apt -y install portaudio19-dev git ffmpeg curl

# torch
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install ipython

# Copy entire repo
COPY . /app

WORKDIR /app

RUN pip install .

# install whisper models
RUN python3 -c 'import whisper; whisper._download(whisper._MODELS["tiny"], "~/.cache/whisper", False)'
RUN python3 -c 'import whisper; whisper._download(whisper._MODELS["small"], "~/.cache/whisper", False)'
RUN python3 -c 'import whisper; whisper._download(whisper._MODELS["base"], "~/.cache/whisper", False)'
RUN python3 -c 'import whisper; whisper._download(whisper._MODELS["medium"], "~/.cache/whisper", False)'
RUN python3 -c 'import whisper; whisper._download(whisper._MODELS["large"], "~/.cache/whisper", False)'

# open bash
CMD ["/bin/bash"]
