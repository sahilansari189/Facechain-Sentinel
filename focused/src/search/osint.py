"""Public-only query and identity-pivot helpers."""

from __future__ import annotations

import re
from typing import List


def split_handle(handle: str) -> str:
    """Turn a username or CamelCase slug into a readable search phrase."""
    value = re.sub(r"[_-]+", " ", (handle or "").strip())
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return " ".join(value.split())


def contextual_queries(handle: str, context: str = "") -> List[str]:
    """Build deduplicated public-web dorks from runtime inputs."""
    name = split_handle(handle)
    if not name:
        return []
    queries = [f'"{name}"']
    if context.strip():
        queries.append(f'"{name}" {context.strip()}')
    queries.extend([
        f'site:instagram.com "{name}"',
        f'site:x.com "{name}"',
        f'site:linkedin.com/posts "{name}"',
    ])
    return list(dict.fromkeys(queries))


def linkedin_post_query(handle: str, context: str = "") -> str:
    """Return a guest-visible LinkedIn post query, never a profile query."""
    name = split_handle(handle)
    suffix = f" {context.strip()}" if context.strip() else ""
    return f'site:linkedin.com/posts "{name}"{suffix}'.strip()