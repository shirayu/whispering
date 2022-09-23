
# whisper_streaming

[![CI](https://github.com/shirayu/whisper_streaming/actions/workflows/ci.yml/badge.svg)](https://github.com/shirayu/whisper_streaming/actions/workflows/ci.yml)
[![CodeQL](https://github.com/shirayu/whisper_streaming/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/shirayu/whisper_streaming/actions/workflows/codeql-analysis.yml)
[![Typos](https://github.com/shirayu/whisper_streaming/actions/workflows/typos.yml/badge.svg)](https://github.com/shirayu/whisper_streaming/actions/workflows/typos.yml)

Streaming transcriber with [whisper](https://github.com/openai/whisper)

## Example

```bash
git clone https://github.com/shirayu/whisper_streaming.git
cd whisper_streaming
poetry install --only main
poetry run whisper_streaming --language ja --model base
```

## Tips

If you get ``OSError: PortAudio library not found``: Install ``portaudio``

```bash
# Ubuntu
sudo apt-get install portaudio19-dev
```

## License

- [MIT License](LICENSE)
- Some codes are ported from the original whisper. Its license is also [MIT License](LICENSE.whisper)
