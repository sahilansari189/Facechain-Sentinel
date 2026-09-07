"""Public X/Twitter status harvester.

The harvester reads public OpenGraph/Twitter Card metadata and media URLs.
It does not authenticate, use the official API, or bypass protected posts.
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
_STATUS_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:x|twitter)\.com/[A-Za-z0-9_]+/status/[0-9]+",
    re.IGNORECASE,
)
_MEDIA_PATTERN = re.compile(
    r"https?://pbs\.twimg\.com/(?:media|ext_tw_video)/[^\s\"'<>\\]+",
    re.IGNORECASE,
)


@dataclass
class XPostContent:
    """Metadata and media URLs extracted from a public X status."""

    source_url: str = ""
    title: str = ""
    text: str = ""
    username: str = ""
    photos: list[str] = field(default_factory=list)
    raw_html_len: int = 0


def is_x_post_url(url: str) -> bool:
    """Return whether *url* identifies an X/Twitter status."""
    return bool(_STATUS_PATTERN.match((url or "").split("?", 1)[0]))


def _add_url(urls: list[str], value: object) -> None:
    if not isinstance(value, str):
        return
    value = html_lib.unescape(value).replace("\\u0026", "&").strip()
    if value.startswith("http") and value not in urls:
        urls.append(value)


def _collect_media(value: object, urls: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"url", "media_url", "media_url_https", "image", "images"}:
                if isinstance(item, list):
                    for entry in item:
                        _collect_media(entry, urls)
                else:
                    _add_url(urls, item)
            else:
                _collect_media(item, urls)
    elif isinstance(value, list):
        for item in value:
            _collect_media(item, urls)
    elif isinstance(value, str) and "pbs.twimg.com" in value:
        _add_url(urls, value)


def extract_x_content(html: str, source_url: str) -> XPostContent:
    """Parse public status HTML into metadata and deduplicated media URLs."""
    content = XPostContent(source_url=source_url, raw_html_len=len(html or ""))
    if not html:
        return content

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
    description = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "twitter:description"})
    if title and title.get("content"):
        content.title = " ".join(str(title["content"]).split())[:240]
    if description and description.get("content"):
        content.text = " ".join(str(description["content"]).split())[:600]

    urls: list[str] = []
    for tag in soup.find_all("meta"):
        key = str(tag.get("property") or tag.get("name") or "").lower()
        if key in {"og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src"}:
            _add_url(urls, tag.get("content"))

    for match in _MEDIA_PATTERN.findall(html):
        _add_url(urls, match)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            _collect_media(json.loads(script.string or ""), urls)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    content.photos = [url for url in urls if "pbs.twimg.com" in url]
    username_match = re.search(r"(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/", source_url, re.IGNORECASE)
    if username_match:
        content.username = username_match.group(1)
    return content


def _fxtwitter_url(status_url: str) -> str:
    match = _STATUS_PATTERN.match(status_url.split("?", 1)[0])
    if not match:
        return ""
    parts = match.group(0).rstrip("/").split("/")
    return f"https://api.fxtwitter.com/{parts[-3]}/status/{parts[-1]}"


def harvest_x_post(
    status_url: str,
    session: Optional[requests.Session] = None,
    timeout: float = 12.0,
    browser_fetcher: Optional[Callable[[str, float], str]] = None,
    max_photos: int = 8,
) -> list[Candidate]:
    """Fetch one public X/Twitter status and return image candidates."""
    if not is_x_post_url(status_url):
        return []

    owns_session = session is None
    session = session or requests.Session()
    html = ""
    try:
        response = session.get(
            status_url,
            timeout=timeout,
            headers={"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
            allow_redirects=True,
        )
        if response.status_code == 200 and response.text:
            html = response.text
    except Exception as exc:
        logger.debug("X status fetch failed: %s", exc)

    content = extract_x_content(html, status_url)
    if not content.photos:
        try:
            fx_url = _fxtwitter_url(status_url)
            if fx_url:
                response = session.get(fx_url, timeout=min(timeout, 6.0), headers={"User-Agent": _BROWSER_UA})
                if response.status_code == 200:
                    content = extract_x_content(response.text, status_url)
        except Exception as exc:
            logger.debug("FxTwitter fallback failed: %s", exc)

    if not html and browser_fetcher is not None:
        try:
            content = extract_x_content(browser_fetcher(status_url, timeout), status_url)
        except Exception as exc:
            logger.debug("X browser escalation failed: %s", exc)

    if owns_session:
        session.close()

    label = content.title or "X Post"
    return [
        Candidate(
            image_url=image_url,
            source_url=status_url,
            title=label,
            domain="x.com",
        )
        for image_url in content.photos[:max_photos]
    ]
