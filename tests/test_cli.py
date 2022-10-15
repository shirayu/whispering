#!/usr/bin/env python3


import sys
from unittest.mock import patch

from whispering.cli import get_opts, is_valid_arg


def test_options():

    invalid_args = [
        "--mode server --mic 0",
        "--mode server --mic 1",
        "--mode server --beam_size 3",
        "--mode server --temperature 0",
        "--mode server --num_block 3",
        "--mode mic --host 0.0.0.0",
        "--mode mic --port 8000",
    ]

    for invalid_arg in invalid_args:
        with patch.object(sys, "argv", [""] + invalid_arg.split()):
            opts = get_opts()
            ok = is_valid_arg(opts)
            assert ok is False, f"{invalid_arg} should be invalid"
