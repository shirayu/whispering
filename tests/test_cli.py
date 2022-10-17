#!/usr/bin/env python3


from pydantic import BaseModel

from whispering.cli import Mode, is_valid_arg


class ArgExample(BaseModel):
    mode: Mode
    cmd: str
    ok: bool


def test_options():

    exs = [
        ArgExample(mode=Mode.server, cmd="--mic 0", ok=False),
        ArgExample(mode=Mode.server, cmd="--mic 1", ok=False),
        ArgExample(
            mode=Mode.server,
            cmd="--host 0.0.0.0 --port 8000",
            ok=True,
        ),
        ArgExample(
            mode=Mode.server,
            cmd="--language en --model tiny --host 0.0.0.0 --port 8000",
            ok=True,
        ),
        ArgExample(mode=Mode.server, cmd="--beam_size 3", ok=False),
        ArgExample(mode=Mode.server, cmd="--temperature 0", ok=False),
        ArgExample(mode=Mode.server, cmd="--num_block 3", ok=False),
        ArgExample(mode=Mode.mic, cmd="--host 0.0.0.0", ok=False),
        ArgExample(mode=Mode.mic, cmd="--port 8000", ok=False),
    ]

    for ex in exs:
        ok = is_valid_arg(
            mode=ex.mode.value,
            args=ex.cmd.split(),
        )
        assert ok is ex.ok, f"{ex.cmd} should be {ex.ok}"
