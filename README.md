
# Whispering

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](LICENSE)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

[![CI](https://github.com/shirayu/whispering/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/whispering/actions/workflows/ci.yml)
[![CodeQL](https://github.com/shirayu/whispering/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/shirayu/whispering/actions/workflows/codeql-analysis.yml)
[![Typos](https://github.com/shirayu/whispering/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/whispering/actions/workflows/typos.yml)

Streaming transcriber with [whisper](https://github.com/openai/whisper).
Enough machine power is needed to transcribe in real time.

## Setup

```bash
pip install -U git+https://github.com/shirayu/whispering.git@v0.5.1

# If you use GPU, install proper torch and torchaudio
# Check https://pytorch.org/get-started/locally/
# Example : torch for CUDA 11.6
pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

If you get ``OSError: PortAudio library not found`` in Linux, install "PortAudio".

```bash
sudo apt -y install portaudio19-dev
```

## Example of microphone

```bash
# Run in English
whispering --language en --model tiny
```

- ``--help`` shows full options
- ``--model`` sets the [model name](https://github.com/openai/whisper#available-models-and-languages) to use. Larger models will be more accurate, but may not be able to transcribe in real time.
- ``--language`` sets the language to transcribe. The list of languages are shown with ``whispering -h``
- ``--no-progress`` disables the progress message
- ``-t`` sets temperatures to decode. You can set several like ``-t 0.0 -t 0.1 -t 0.5``, but too many temperatures exhaust decoding time
- ``--debug`` outputs logs for debug
- ``--vad`` sets VAD (Voice Activity Detection) threshold. 0 disables VAD and forces whisper to analyze non-voice activity sound period
- ``--output`` sets output file (Default: Standard output)

### Parse interval

By default, whispering performs VAD for every 3.75 second.
This interval is determined by the value of ``-n`` and its default is ``20``.
When an interval is predicted as "silence", it will not be passed to whisper.
If you want to disable VAD, please use ``--no-vad`` option.

By default, Whisper does not perform analysis until the total length of the segments determined by VAD to have speech exceeds 30 seconds.
This is because Whisper is trained to make predictions for 30-second intervals.
Nevertheless, if you want to force Whisper to perform analysis even if a segment is less than 30 seconds, please use ``--allow-padding`` option like this.

```bash
whispering --language en --model tiny -n 20 --allow-padding
```

This forces Whisper to analyze every 3.75 seconds speech segment.
Using ``--allow-padding`` may sacrifice the accuracy, while you can get quick response.
The smaller value of ``-n`` with ``--allow-padding`` is, the worse the accuracy becomes.

## Example of web socket

⚠  **No security mechanism. Please make secure with your responsibility.**

Run with ``--host`` and ``--port``.

### Host

```bash
whispering --language en --model tiny --host 0.0.0.0 --port 8000
```

### Client

```bash
whispering --host ADDRESS_OF_HOST --port 8000 --mode client
```

You can set ``-n``, ``--allow-padding`` and other options.

## License

- [MIT License](LICENSE)
- Some codes are ported from the original whisper. Its license is also [MIT License](LICENSE.whisper)
