"""Public LinkedIn post media harvester.

Uses only guest-visible post HTML. It does not authenticate or bypass
LinkedIn login and anti-bot protections.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

from .free_search import Candidate

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_MEDIA_PATTERN = re.compile(
    r"https://media\.licdn\.com/dms/image/[^\s\"'<>\\]+",
    re.IGNORECASE,
)
_POST_PATTERN = re.compile(
    r"linkedin\.com/posts/[A-Za-z0-9_-]{3,120}",
    re.IGNORECASE,
)
_PROFILE_PATTERN = re.compile(
    r"https://[a-z]{2,4}\.linkedin\.com/in/([A-Za-z0-9_-]{3,60})",
    re.IGNORECASE,
)
_FEEDSHARE = "feedshare"
_DISPLAYPHOTO = "profile-displayphoto"


@dataclass
class LinkedInPostContent:
    source_url: str = ""
    title: str = ""
    text: str = ""
    photos: list[str] = field(default_factory=list)
    profile_slugs: list[str] = field(default_factory=list)
    raw_html_len: int = 0


def is_linkedin_post_url(url: str) -> bool:
    return bool(_POST_PATTERN.search(url or ""))


def extract_linkedin_content(html: str, source_url: str = "") -> LinkedInPostContent:
    """Extract post metadata and media URLs from public LinkedIn HTML."""
    content = LinkedInPostContent(source_url=source_url, raw_html_len=len(html or ""))
    if not html:
        return content

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("meta", property="og:title") or soup.find("title")
    description = soup.find("meta", property="og:description") or soup.find(
        "meta", attrs={"name": "description"}
    )
    if title:
        value = title.get("content") if title.name == "meta" else title.get_text(" ", strip=True)
        content.title = " ".join(str(value or "").split())[:240]
    if description and description.get("content"):
        content.text = " ".join(str(description["content"]).split())[:600]

    feedshare: list[str] = []
    displayphoto: list[str] = []
    other: list[str] = []
    seen_urls: set[str] = set()
    for raw in _MEDIA_PATTERN.findall(html):
        url = html_lib.unescape(raw).replace("\\u0026", "&")
        key = url.split("?", 1)[0]
        if key in seen_urls or key.lower().endswith(".svg") or "framing" in key.lower():
            continue
        seen_urls.add(key)
        if _FEEDSHARE in url.lower():
            feedshare.append(url)
        elif _DISPLAYPHOTO in url.lower():
            displayphoto.append(url)
        else:
            other.append(url)

    # Real post media must precede profile/avatar media.
    content.photos = feedshare + displayphoto + other
    content.profile_slugs = list(dict.fromkeys(
        slug.lower() for slug in _PROFILE_PATTERN.findall(html)
    ))
    return content


def harvest_linkedin_post(
    post_url: str,
    session: Optional[requests.Session] = None,
    timeout: float = 12.0,
    browser_fetcher: Optional[Callable[[str, float], str]] = None,
    max_photos: int = 4,
) -> list[Candidate]:
    """Fetch a public LinkedIn post and return its post-image candidates."""
    if not is_linkedin_post_url(post_url):
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
        logger.debug("LinkedIn post fetch failed: %s", exc)
    finally:
        if owns_session:
            session.close()

    if not html and browser_fetcher is not None:
        try:
            html = browser_fetcher(post_url, timeout)
        except Exception as exc:
            logger.debug("LinkedIn browser fallback failed: %s", exc)

    content = extract_linkedin_content(html, post_url)
    return [
        Candidate(
            image_url=image_url,
            source_url=post_url,
            title=content.title or "LinkedIn Post Image",
            domain="linkedin.com",
        )
        for image_url in content.photos[:max_photos]
    ]


def harvest_associate_slugs(
    post_url: str,
    session: Optional[requests.Session] = None,
    timeout: float = 12.0,
    exclude: Optional[set[str]] = None,
) -> list[str]:
    """Return public member slugs found in a LinkedIn post page."""
    owns_session = session is None
    session = session or requests.Session()
    try:
        response = session.get(
            post_url,
            timeout=timeout,
            headers={"User-Agent": _BROWSER_UA},
            allow_redirects=True,
        )
        if response.status_code != 200:
            return []
        slugs = extract_linkedin_content(response.text, post_url).profile_slugs
        excluded = {item.lower() for item in (exclude or set())}
        return [slug for slug in slugs if slug not in excluded]
    except Exception:
        return []
    finally:
        if owns_session:
            session.close()
