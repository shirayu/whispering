
# Whispering

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](LICENSE)
![Python Versions](https://img.shields.io/badge/Python-3.8%20--%203.10-blue)

[![CI](https://github.com/shirayu/whispering/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/whispering/actions/workflows/ci.yml)
[![CodeQL](https://github.com/shirayu/whispering/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/shirayu/whispering/actions/workflows/codeql-analysis.yml)
[![Typos](https://github.com/shirayu/whispering/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/whispering/actions/workflows/typos.yml)

Streaming transcriber with [whisper](https://github.com/openai/whisper).
Enough machine power is needed to transcribe in real time.

## Setup

```bash
pip install -U git+https://github.com/shirayu/whispering.git

# If you use GPU, install proper torch and torchaudio
# Example : torch for CUDA 11.6
pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Example of microphone

```bash
# Run in English
whispering --language en --model tiny
```

- ``--help`` shows full options
- ``--model`` set the [model name](https://github.com/openai/whisper#available-models-and-languages) to use. Larger models will be more accurate, but may not be able to transcribe in real time.
- ``--language`` sets the language to transcribe. The list of languages are shown with ``whispering -h``
- ``--no-progress`` disables the progress message
- ``-t`` sets temperatures to decode. You can set several like (``-t 0.0 -t 0.1 -t 0.5``), but too many temperatures exhaust decoding time
- ``--debug`` outputs logs for debug

### Parse interval

If you want quick response, set small ``-n`` and add ``--allow-padding``.
However, this may sacrifice the accuracy.

```bash
whispering --language en --model tiny -n 20 --allow-padding
```

## Example of web socket

⚠  **No security mechanism. Please make secure with your responsibility.**

Run with ``--host`` and ``--port``.

### Host

```bash
whispering --language en --model tiny --host 0.0.0.0 --port 8000
```

You can set ``--allow-padding`` and other options.

### Client

```bash
whispering --host ADDRESS_OF_HOST --port 8000 --mode client
```

You can set ``-n`` and other options.

## Tips

## PortAudio Error

If you get ``OSError: PortAudio library not found``: Install ``portaudio``

```bash
# Ubuntu
sudo apt-get install portaudio19-dev
```

## License

- [MIT License](LICENSE)
- Some codes are ported from the original whisper. Its license is also [MIT License](LICENSE.whisper)
