#!/usr/bin/env python3


import sys
from unittest.mock import patch

from pydantic import BaseModel

from whispering.cli import get_opts, is_valid_arg


class ArgExample(BaseModel):
    cmd: str
    ok: bool


def test_options():

    exs = [
        ArgExample(cmd="--mode server --mic 0", ok=False),
        ArgExample(cmd="--mode server --mic 1", ok=False),
        ArgExample(cmd="--mode server --beam_size 3", ok=False),
        ArgExample(cmd="--mode server --temperature 0", ok=False),
        ArgExample(cmd="--mode server --num_block 3", ok=False),
        ArgExample(cmd="--mode mic --host 0.0.0.0", ok=False),
        ArgExample(cmd="--mode mic --port 8000", ok=False),
    ]

    for ex in exs:
        with patch.object(sys, "argv", [""] + ex.cmd.split()):
            opts = get_opts()
            ok = is_valid_arg(opts)
            assert ok is ex.ok, f"{ex.cmd} should be invalid"
