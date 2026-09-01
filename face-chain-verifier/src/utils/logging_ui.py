"""Tiny terminal presentation helpers so the CLI reads well on a screen recording."""

from __future__ import annotations

import sys

WIDTH = 60


def banner(title: str) -> None:
    print("=" * WIDTH)
    print(title.upper())
    print("=" * WIDTH)


def step(msg: str) -> None:
    print(f"  {msg}")


def ok(msg: str) -> None:
    print(f"  {msg} \u2713")


def warn(msg: str) -> None:
    print(f"  ! {msg}")


def fail(msg: str) -> None:
    print(f"  \u2717 {msg}", file=sys.stderr)


def kv(key: str, value: object) -> None:
    print(f"  {key:<20} {value}")
