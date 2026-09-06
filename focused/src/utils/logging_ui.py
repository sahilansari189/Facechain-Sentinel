"""Tiny terminal presentation helpers so the CLI reads well on a screen recording."""

from __future__ import annotations

import sys

# Reconfigure stdout/stderr for Windows console encodings
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WIDTH = 60


def banner(title: str) -> None:
    print("=" * WIDTH)
    print(title.upper())
    print("=" * WIDTH)


def step(msg: str) -> None:
    print(f"  {msg}")


def ok(msg: str) -> None:
    try:
        print(f"  {msg} \u2713")
    except UnicodeEncodeError:
        print(f"  {msg} [OK]")


def warn(msg: str) -> None:
    print(f"  ! {msg}")


def fail(msg: str) -> None:
    try:
        print(f"  \u2717 {msg}", file=sys.stderr)
    except UnicodeEncodeError:
        print(f"  [FAIL] {msg}", file=sys.stderr)


def kv(key: str, value: object) -> None:
    print(f"  {key:<20} {value}")
