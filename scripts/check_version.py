#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Final, Tuple

import toml


def get_stable_version(obj) -> str:
    stable_version: Final[str] = obj["misc"]["stable_version"]
    return f"v{stable_version}"


def check_version(path_in: Path, path_pyproject_toml: Path) -> Tuple[bool, str]:
    with path_pyproject_toml.open() as f:
        obj = toml.load(f)
        stable_version: Final[str] = get_stable_version(obj)
        repository: Final[str] = obj["tool"]["poetry"]["repository"]
        cmd: Final[str] = f"pip install -U git+{repository}@{stable_version}"

    with path_in.open() as f:
        for line in f:
            if line.strip() == cmd:
                return True, stable_version

    sys.stderr.write(f"The following command not found in {path_in}: {cmd}\n")
    return False, stable_version


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path)
    oparser.add_argument("--toml", "-t", type=Path, required=True)
    oparser.add_argument("--tags", type=Path)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()

    assert opts.input is not None
    ok, stable_version = check_version(opts.input, opts.toml)
    if not ok:
        sys.exit(1)

    if opts.tags:
        tags = []
        if str(opts.tags) == "/dev/stdin":
            for line in sys.stdin:
                tags.append(line[:-1])
        else:
            with opts.tags.open() as f:
                for line in f:
                    tags.append(line[:-1])

        if stable_version not in tags:
            sys.stderr.write(f"Tag {stable_version} not in git tags: {tags}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
