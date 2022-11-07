
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
pip install -U git+https://github.com/shirayu/whispering.git@v0.6.3

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
- ``--vad`` sets VAD (Voice Activity Detection) threshold. The default is ``0.5``. ``0`` disables VAD and forces whisper to analyze non-voice activity sound period. Try ``--vad 0`` if VAD prevents transcription.
- ``--output`` sets output file (Default: Standard output)

### Parse interval

By default, whispering performs VAD for every 3.75 second.
This interval is determined by the value of ``-n`` and its default is ``20``.
When an interval is predicted as "silence", it will not be passed to whisper.
If you want to disable VAD, please make VAD threshold 0 by adding ``--vad 0``.

By default, Whisper does not perform analysis until the total length of the segments determined by VAD to have speech exceeds 30 seconds.
However, if silence segments appear 16 times (the default value of ``--max_nospeech_skip``) after speech is detected, the analysis is performed.

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

You can set ``-n`` and other options.

## For Developers

1. Install [Python](https://www.python.org/) and [Node.js](https://nodejs.org/)
2. [Install poetry](https://python-poetry.org/docs/) to use ``poetry`` command
3. Clone and install libraries

    ```console
    # Clone
    git clone https://github.com/shirayu/whispering.git

    # With poetry
    poetry config virtualenvs.in-project true
    poetry install --all-extras
    poetry run pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

    # With npm
    npm install
    ```

4. Run test and check that no errors occur

    ```bash
    poetry run make -j4
    ```

5. Make fancy updates
6. Make style

    ```bash
    poetry run make style
    ```

7. Run test again and check that no errors occur

    ```bash
    poetry run make -j4
    ```

8. Check typos by using [typos](https://github.com/crate-ci/typos). Just run ``typos`` command in the root directory.

    ```bash
    typos
    ```

9. Send Pull requests!

## License

- [MIT License](LICENSE)
- Some codes are ported from the original whisper. Its license is also [MIT License](LICENSE.whisper)
