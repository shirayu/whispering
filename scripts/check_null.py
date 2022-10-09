#!/usr/bin/env python3

import sys


def main() -> None:
    data = sys.stdin.read()
    if len(data) != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
