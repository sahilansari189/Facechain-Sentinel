"""Public Instagram post and reel harvester.

This module only uses guest-visible HTML and optional browser-rendered HTML.
It never logs in or attempts to bypass an Instagram access wall.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

from src.search.free_search import Candidate

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_IMAGE_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>\\]+(?:cdninstagram\.com|fbcdn\.net)[^\s\"'<>\\]*",
    re.IGNORECASE,
)
_POST_PATTERN = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:p|reel)/[A-Za-z0-9_-]+",
    re.IGNORECASE,
)


@dataclass
class InstagramPostContent:
    """Metadata and ranked image URLs extracted from a public post page."""

    source_url: str = ""
    title: str = ""
    text: str = ""
    username: str = ""
    photos: list[str] = field(default_factory=list)
    raw_html_len: int = 0


def is_instagram_post_url(url: str) -> bool:
    """Return whether *url* identifies an Instagram post or reel."""
    return bool(_POST_PATTERN.match((url or "").split("?", 1)[0]))


def _add_url(urls: list[str], value: object) -> None:
    if not isinstance(value, str):
        return
    value = html_lib.unescape(value).replace("\\u0026", "&").strip()
    if value.startswith("http") and value not in urls:
        urls.append(value)


def _collect_json_images(value: object, urls: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"image", "images", "display_url", "thumbnail_url", "thumbnail_src"}:
                if isinstance(item, list):
                    for entry in item:
                        _collect_json_images(entry, urls)
                else:
                    _add_url(urls, item)
            else:
                _collect_json_images(item, urls)
    elif isinstance(value, list):
        for item in value:
            _collect_json_images(item, urls)
    elif isinstance(value, str) and ("cdninstagram.com" in value or "fbcdn.net" in value):
        _add_url(urls, value)


def extract_instagram_content(html: str, source_url: str) -> InstagramPostContent:
    """Parse guest HTML into metadata and deduplicated image URLs."""
    content = InstagramPostContent(source_url=source_url, raw_html_len=len(html or ""))
    if not html:
        return content

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
    description = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
    if title and title.get("content"):
        content.title = " ".join(str(title["content"]).split())[:240]
    if description and description.get("content"):
        content.text = " ".join(str(description["content"]).split())[:600]

    urls: list[str] = []
    for tag in soup.find_all("meta"):
        key = str(tag.get("property") or tag.get("name") or "").lower()
        if key in {"og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src"}:
            _add_url(urls, tag.get("content"))

    for match in _IMAGE_URL_PATTERN.findall(html):
        _add_url(urls, match)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            _collect_json_images(json.loads(script.string or ""), urls)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    content.photos = [url for url in urls if not url.lower().endswith(".svg")]
    username_match = re.search(r"instagram\.com/([A-Za-z0-9_.-]+)", source_url, re.IGNORECASE)
    if username_match and username_match.group(1).lower() not in {"p", "reel"}:
        content.username = username_match.group(1)
    return content


def harvest_instagram_post(
    post_url: str,
    session: Optional[requests.Session] = None,
    timeout: float = 12.0,
    browser_fetcher: Optional[Callable[[str, float], str]] = None,
    max_photos: int = 8,
) -> list[Candidate]:
    """Fetch one public Instagram post/reel and return image candidates."""
    if not is_instagram_post_url(post_url):
        return []

    owns_session = session is None
    session = session or requests.Session()
    html = ""
    try:
        response = session.get(
            post_url,
            timeout=timeout,
            headers={"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
            allow_redirects=True,
        )
        if response.status_code == 200 and response.text:
            html = response.text
    except Exception as exc:
        logger.debug("Instagram post fetch failed: %s", exc)
    finally:
        if owns_session:
            session.close()

    if not html and browser_fetcher is not None:
        try:
            html = browser_fetcher(post_url, timeout)
        except Exception as exc:
            logger.debug("Instagram browser escalation failed: %s", exc)

    content = extract_instagram_content(html, post_url)
    label = content.title or "Instagram Post"
    return [
        Candidate(
            image_url=image_url,
            source_url=post_url,
            title=label,
            domain="instagram.com",
        )
        for image_url in content.photos[:max_photos]
    ]
