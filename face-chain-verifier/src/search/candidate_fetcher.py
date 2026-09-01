"""Safe candidate-image downloading + public page metadata extraction.

This module is intentionally network-safe:

* Only public HTTP/HTTPS resources are allowed.
* Private, loopback, link-local, multicast and reserved IPs are blocked.
* Candidate images are downloaded with a hard byte limit.
* Image validity is checked using the actual image bytes, not only
  Content-Type.
* Redirect destinations are validated before following them.
* Page metadata is best-effort and never bypasses authentication,
  CAPTCHA, robots restrictions, or platform access controls.

The reverse-image-search provider normally gives us both:

    candidate.url
        The public page/post URL.

    candidate.image_url
        The actual image/CDN URL returned by the search provider.

For face verification we should prefer candidate.image_url because
candidate.url is usually an HTML page, not an image.
"""

from __future__ import annotations

import ipaddress
import socket
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests


# ============================================================
# Constants
# ============================================================

USER_AGENT = (
    "face-chain-verifier/1.0 "
    "(+research; respects robots and rate limits)"
)

DEFAULT_TIMEOUT = 15

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
    "image/tiff",
    "image/avif",
}

# Some CDNs occasionally return generic/broken MIME types.
GENERIC_IMAGE_TYPES = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
}

MAX_METADATA_BYTES = 1_000_000


# ============================================================
# Errors
# ============================================================


class FetchError(RuntimeError):
    """Raised when an image/page cannot safely be fetched."""


# ============================================================
# URL safety
# ============================================================


def _resolve_hostname(hostname: str) -> list[str]:
    """Resolve a hostname to IP addresses.

    Raises FetchError when DNS resolution fails.
    """

    try:
        infos = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise FetchError(
            f"DNS resolution failed for {hostname}"
        ) from exc

    addresses: list[str] = []

    for info in infos:
        try:
            address = info[4][0]
        except (IndexError, TypeError):
            continue

        if address not in addresses:
            addresses.append(address)

    if not addresses:
        raise FetchError(
            f"hostname {hostname} resolved to no addresses"
        )

    return addresses


def _is_public_ip(address: str) -> bool:
    """Return True only for publicly routable IP addresses."""

    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False

    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_safe_url(url: str) -> bool:
    """Allow only public HTTP(S) URLs.

    This prevents the candidate fetcher from becoming an SSRF primitive.

    Examples rejected:

        http://127.0.0.1
        http://localhost
        http://10.0.0.1
        http://192.168.1.1
        http://172.16.0.1
        file://...
        ftp://...
    """

    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    if not parsed.hostname:
        return False

    hostname = parsed.hostname.strip()

    # Direct IP hostname.
    try:
        ip = ipaddress.ip_address(hostname)
        return _is_public_ip(str(ip))
    except ValueError:
        pass

    # Hostnames.
    try:
        addresses = _resolve_hostname(hostname)
    except FetchError:
        return False

    return all(_is_public_ip(address) for address in addresses)


# ============================================================
# URL helpers
# ============================================================


def normalize_url(base_url: str, value: str) -> str:
    """Resolve a possibly-relative URL against a page URL."""

    if not value:
        return ""

    value = value.strip()

    if value.startswith("//"):
        parsed = urlparse(base_url)
        return f"{parsed.scheme}:{value}"

    return urljoin(base_url, value)


def _content_type(response: requests.Response) -> str:
    """Extract normalized Content-Type."""

    return (
        (response.headers.get("Content-Type") or "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )


def _looks_like_image_url(url: str) -> bool:
    """Cheap URL-level image detection.

    This is only a hint. Actual image bytes are verified separately.
    """

    if not url:
        return False

    path = urlparse(url).path.lower()

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".avif",
    )

    return path.endswith(image_extensions)


# ============================================================
# Image byte validation
# ============================================================


def _validate_image_bytes(data: bytes) -> bool:
    """Validate that bytes represent a real image.

    Pillow is deliberately imported lazily so that modules that only
    need metadata do not necessarily require Pillow at import time.
    """

    if not data:
        return False

    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            image.verify()

        return True

    except Exception:
        return False


def _image_format_from_bytes(data: bytes) -> str:
    """Return the detected image format, or an empty string."""

    if not data:
        return ""

    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            return (image.format or "").upper()

    except Exception:
        return ""


# ============================================================
# Image download
# ============================================================


def download_image(
    url: str,
    max_bytes: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> bytes:
    """Safely download an image.

    Important:

    A reverse-image-search result often contains a page URL and a
    separate image/CDN URL. This function should normally receive the
    latter.

    The server's Content-Type is treated as a hint rather than the
    sole source of truth because real-world CDNs frequently return:

        application/octet-stream
        binary/octet-stream
        missing Content-Type

    Actual image bytes are verified with Pillow.
    """

    if not url:
        raise FetchError("empty image URL")

    if max_bytes <= 0:
        raise FetchError("invalid image size limit")

    if not is_safe_url(url):
        raise FetchError("unsafe or invalid image URL")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "image/avif,image/webp,image/apng,"
            "image/svg+xml,image/*,*/*;q=0.8"
        ),
    }

    current_url = url
    redirect_limit = 5

    for _ in range(redirect_limit + 1):
        if not is_safe_url(current_url):
            raise FetchError(
                "redirect destination is unsafe or invalid"
            )

        try:
            with requests.get(
                current_url,
                timeout=timeout,
                stream=True,
                headers=headers,
                allow_redirects=False,
            ) as response:

                # ------------------------------------------------
                # Manual redirect handling.
                # ------------------------------------------------

                if response.status_code in (
                    301,
                    302,
                    303,
                    307,
                    308,
                ):
                    location = response.headers.get("Location")

                    if not location:
                        raise FetchError(
                            "redirect response has no Location header"
                        )

                    current_url = normalize_url(
                        current_url,
                        location,
                    )

                    continue

                # ------------------------------------------------
                # HTTP errors.
                # ------------------------------------------------

                if response.status_code == 429:
                    raise FetchError(
                        "rate limited by host (HTTP 429)"
                    )

                if response.status_code == 403:
                    raise FetchError(
                        "access denied by image host (HTTP 403)"
                    )

                if response.status_code >= 400:
                    raise FetchError(
                        f"HTTP {response.status_code}"
                    )

                ctype = _content_type(response)

                declared = response.headers.get(
                    "Content-Length"
                )

                if (
                    declared
                    and declared.isdigit()
                    and int(declared) > max_bytes
                ):
                    raise FetchError(
                        "image exceeds size limit"
                    )

                # ------------------------------------------------
                # Stream the response.
                # ------------------------------------------------

                buffer = bytearray()

                for chunk in response.iter_content(
                    chunk_size=64 * 1024
                ):
                    if not chunk:
                        continue

                    buffer.extend(chunk)

                    if len(buffer) > max_bytes:
                        raise FetchError(
                            "image exceeds size limit"
                        )

                data = bytes(buffer)

                if not data:
                    raise FetchError(
                        "empty image response"
                    )

                # ------------------------------------------------
                # MIME validation.
                #
                # Don't reject immediately when the MIME type is
                # generic/missing. Many CDNs mislabel images.
                # ------------------------------------------------

                if ctype and ctype not in ALLOWED_IMAGE_TYPES:
                    if ctype not in GENERIC_IMAGE_TYPES:
                        # It may still be a valid image with an
                        # incorrect server MIME type. Verify bytes.
                        if not _validate_image_bytes(data):
                            raise FetchError(
                                "not an image "
                                f"(content-type {ctype})"
                            )
                    else:
                        if not _validate_image_bytes(data):
                            raise FetchError(
                                "response is not a valid image"
                            )
                else:
                    if not _validate_image_bytes(data):
                        raise FetchError(
                            "downloaded bytes are not a valid image"
                        )

                return data

        except FetchError:
            raise

        except requests.RequestException as exc:
            raise FetchError(
                f"download failed: {exc}"
            ) from exc

    raise FetchError(
        f"too many redirects while fetching image: {url}"
    )


# ============================================================
# Candidate image URL selection
# ============================================================


def candidate_image_url(candidate) -> str:
    """Return the best available image URL from a Candidate.

    Priority:

        1. candidate.image_url
        2. candidate.url only if it itself looks like an image

    Never blindly treat an Instagram/LinkedIn/YouTube page as an
    image URL.
    """

    image_url = getattr(
        candidate,
        "image_url",
        "",
    ) or ""

    if image_url and is_safe_url(image_url):
        return image_url

    page_url = getattr(
        candidate,
        "url",
        "",
    ) or ""

    if page_url and _looks_like_image_url(page_url):
        if is_safe_url(page_url):
            return page_url

    return ""


def download_candidate_image(
    candidate,
    max_bytes: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> bytes:
    """Download the actual candidate image.

    Raises FetchError with a useful reason when no usable image URL
    is available.
    """

    image_url = candidate_image_url(candidate)

    if not image_url:
        raise FetchError(
            "candidate has no usable image URL"
        )

    return download_image(
        image_url,
        max_bytes=max_bytes,
        timeout=timeout,
    )


# ============================================================
# HTML metadata helpers
# ============================================================


def _meta(
    soup,
    *,
    prop: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Read a meta tag safely."""

    if prop:
        tag = soup.find(
            "meta",
            attrs={"property": prop},
        )
    elif name:
        tag = soup.find(
            "meta",
            attrs={"name": name},
        )
    else:
        return ""

    if not tag:
        return ""

    value = tag.get("content")

    if value is None:
        return ""

    return str(value).strip()


def _first_meta(
    soup,
    properties: tuple[str, ...] = (),
    names: tuple[str, ...] = (),
) -> str:
    """Return the first non-empty meta value."""

    for prop in properties:
        value = _meta(
            soup,
            prop=prop,
        )
        if value:
            return value

    for name in names:
        value = _meta(
            soup,
            name=name,
        )
        if value:
            return value

    return ""


# ============================================================
# Page metadata
# ============================================================


def fetch_page_metadata(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Dict[str, str], Optional[str]]:
    """Best-effort public page metadata extraction.

    Returns:

        (metadata, None)

    or:

        ({}, error)

    This function does not attempt to bypass:

    * login walls
    * CAPTCHA
    * anti-bot protection
    * private content
    * authentication
    """

    if not is_safe_url(url):
        return {}, "unsafe or invalid page URL"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
    }

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers=headers,
            allow_redirects=True,
        )

    except requests.RequestException as exc:
        return {}, f"page unreachable: {exc}"

    # Validate final redirect destination.
    final_url = response.url

    if not is_safe_url(final_url):
        return {}, "page redirected to an unsafe URL"

    if response.status_code in (401, 403):
        return (
            {},
            "access restricted by the platform "
            f"(HTTP {response.status_code}) - not bypassed",
        )

    if response.status_code == 429:
        return (
            {},
            "rate limited by the platform (HTTP 429)",
        )

    if response.status_code >= 400:
        return {}, f"HTTP {response.status_code}"

    ctype = _content_type(response)

    if ctype and "html" not in ctype and "xml" not in ctype:
        return {}, f"page is not HTML (content-type {ctype})"

    # Don't parse unlimited response bodies.
    raw = response.content[:MAX_METADATA_BYTES]

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            raw,
            "html.parser",
        )

    except Exception as exc:
        return {}, f"could not parse page: {exc}"

    # ------------------------------------------------------------
    # Title.
    # ------------------------------------------------------------

    title = _first_meta(
        soup,
        properties=("og:title",),
    )

    if not title and soup.title:
        title = (
            soup.title.get_text(
                " ",
                strip=True,
            )
            or ""
        )

    # ------------------------------------------------------------
    # Description.
    # ------------------------------------------------------------

    description = _first_meta(
        soup,
        properties=("og:description",),
        names=("description",),
    )

    # ------------------------------------------------------------
    # Image.
    # ------------------------------------------------------------

    og_image = _first_meta(
        soup,
        properties=(
            "og:image",
            "og:image:url",
            "og:image:secure_url",
        ),
    )

    if og_image:
        og_image = normalize_url(
            final_url,
            og_image,
        )

    # Twitter image fallback.
    if not og_image:
        og_image = _first_meta(
            soup,
            properties=("twitter:image",),
            names=("twitter:image",),
        )

        if og_image:
            og_image = normalize_url(
                final_url,
                og_image,
            )

    # ------------------------------------------------------------
    # Site name.
    # ------------------------------------------------------------

    site_name = _first_meta(
        soup,
        properties=("og:site_name",),
    )

    # ------------------------------------------------------------
    # Published time.
    # ------------------------------------------------------------

    published_time = _first_meta(
        soup,
        properties=(
            "article:published_time",
            "og:published_time",
        ),
        names=(
            "publish-date",
            "date",
            "article:published_time",
        ),
    )

    # ------------------------------------------------------------
    # Author.
    # ------------------------------------------------------------

    author = _first_meta(
        soup,
        properties=("article:author",),
        names=("author",),
    )

    # ------------------------------------------------------------
    # Canonical URL.
    # ------------------------------------------------------------

    canonical_url = ""

    canonical = soup.find(
        "link",
        rel=lambda value: (
            value
            and (
                "canonical" in value
                if isinstance(value, list)
                else value == "canonical"
            )
        ),
    )

    if canonical:
        canonical_url = normalize_url(
            final_url,
            canonical.get("href", ""),
        )

    # ------------------------------------------------------------
    # Build metadata.
    # ------------------------------------------------------------

    metadata = {
        "title": title,
        "description": description,
        "og_image": og_image,
        "site_name": site_name,
        "published_time": published_time,
        "author": author,
        "canonical_url": canonical_url,
        "final_url": final_url,
    }

    return (
        {
            key: value
            for key, value in metadata.items()
            if value
        },
        None,
    )


# ============================================================
# Platform identification
# ============================================================


def platform_name(url: str) -> str:
    """Return a friendly platform name from a URL."""

    try:
        host = (
            urlparse(url)
            .netloc
            .lower()
            .removeprefix("www.")
        )
    except Exception:
        return "unknown"

    known = {
        "instagram.com": "Instagram",
        "x.com": "X",
        "twitter.com": "X",
        "facebook.com": "Facebook",
        "linkedin.com": "LinkedIn",
        "youtube.com": "YouTube",
        "youtu.be": "YouTube",
        "reddit.com": "Reddit",
        "github.com": "GitHub",
        "wikipedia.org": "Wikipedia",
        "behance.net": "Behance",
        "hackster.io": "Hackster.io",
        "hackerearth.com": "HackerEarth",
        "findinfluencer.in": "Findinfluencer",
        "tring.co.in": "Tring",
        "tring.com": "Tring",
        "soundcloud.com": "SoundCloud",
        "medium.com": "Medium",
        "dev.to": "Dev.to",
    }

    for domain, label in known.items():
        if host == domain or host.endswith(
            "." + domain
        ):
            return label

    return host or "unknown"


# ============================================================
# Post record
# ============================================================


def build_post_record(
    candidate,
    page_meta: Dict[str, str],
) -> Dict[str, object]:
    """Build the normalized post representation.

    Prefer the canonical source URL discovered from the actual page.
    This prevents Google/Lens redirect URLs from being stored as the
    source of the verified result.
    """

    # Prefer the actual canonical page URL over a search-engine redirect.
    source_url = (
        page_meta.get("canonical_url")
        or page_meta.get("final_url")
        or getattr(candidate, "url", "")
        or ""
    ).strip()

    # Prefer the image that was actually returned by the search provider.
    image_url = (
        getattr(candidate, "image_url", "")
        or page_meta.get("og_image", "")
        or ""
    ).strip()

    title = (
        page_meta.get("title")
        or getattr(candidate, "title", "")
        or ""
    )

    record = {
        "source_url": source_url,
        "platform": platform_name(source_url),
        "title": title,
        "description": page_meta.get("description", ""),
        "image_url": image_url,
        "site_name": page_meta.get("site_name", ""),
        "published_time": page_meta.get("published_time", ""),
        "author": page_meta.get("author", ""),
        "canonical_url": page_meta.get("canonical_url", ""),
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),
    }

    # Preserve exact-match information when available.
    if hasattr(candidate, "is_exact_match"):
        record["is_exact_match"] = bool(
            getattr(candidate, "is_exact_match", False)
        )

    if hasattr(candidate, "search_type"):
        record["search_type"] = (
            getattr(candidate, "search_type", "")
            or ""
        )

    return record