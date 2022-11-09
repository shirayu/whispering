FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# install python and pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev

# install dependencies
RUN apt -y install portaudio19-dev git ffmpeg curl

# torch
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install ipython

COPY . /app

WORKDIR /app

RUN pip install .

RUN python3 download_whisper_models.py --models tiny small base medium large

# open bash
CMD ["/bin/bash"]
