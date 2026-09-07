"""Reverse-image-search provider abstraction.

Every provider performs a live network request against a third-party
reverse-image-search API and returns structured candidates.

Providers implemented:

    serpapi  - SerpApi Google Lens
    bing     - Azure Bing Visual Search v7
    tineye   - TinEye REST API

Image upload:

    serpapi  - Direct upload through SerpApi Image API
    cloudinary - Upload local image and return HTTPS URL
    generic HTTP uploader - Legacy fallback

Important:

* SerpApi Google Lens is queried with exact_matches first.
* visual_matches are queried separately as a broader fallback.
* SerpApi's Image API can accept the local file directly, avoiding
  Cloudinary for the actual Lens query.
* Exact-match candidates are marked with is_exact_match=True.
* Candidate image URLs prefer the full-size "image" field over
  thumbnails.
* Google Lens ``google.com/goto`` page redirects are resolved to
  publisher URLs when possible.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests


# ============================================================
# Errors
# ============================================================


class SearchError(RuntimeError):
    """Raised on provider/config/network failures."""


# ============================================================
# Candidate
# ============================================================


@dataclass
class Candidate:
    """One reverse-image-search candidate."""

    url: str
    title: str = ""
    source: str = ""
    image_url: str = ""

    # Search metadata.
    is_exact_match: bool = False
    search_type: str = ""

    # Optional position returned by the search provider.
    position: int = 0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# ============================================================
# Helpers
# ============================================================


def _domain(url: str) -> str:
    """Extract a normalized domain from a URL."""

    try:
        return (
            urlparse(url)
            .netloc
            .lower()
            .removeprefix("www.")
        )
    except Exception:
        return ""


def _clean_url(url: str) -> str:
    """Normalize a URL enough for duplicate detection."""

    if not url:
        return ""

    return url.strip().split("#", 1)[0].rstrip("/")


def _is_google_goto(url: str) -> bool:
    """Return True when a URL is a Google redirect wrapper."""
    try:
        parsed = urlparse(url or "")
        return (
            parsed.netloc.lower().endswith("google.com")
            and parsed.path.startswith("/goto")
        )
    except Exception:
        return False


def _extract_redirect_target(url: str) -> str:
    """Extract a direct target URL when the redirect exposes one."""
    if not url:
        return ""

    try:
        query = parse_qs(urlparse(url).query)
        for key in ("url", "q", "target", "dest", "destination"):
            values = query.get(key) or []
            if values:
                target = unquote(values[0]).strip()
                if target.startswith(("http://", "https://")):
                    return target
    except Exception:
        pass

    return ""


def _resolve_page_url(url: str, timeout: int = 10) -> str:
    """Resolve a Google Lens /goto URL to the publisher URL when possible."""
    url = (url or "").strip()

    if not url or not _is_google_goto(url):
        return url

    explicit = _extract_redirect_target(url)
    if explicit and not _is_google_goto(explicit):
        return explicit

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,*/*;q=0.8"
        ),
    }

    # Resolve while the Lens result is still fresh.
    try:
        response = requests.head(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        final_url = (response.url or "").strip()
        if final_url and not _is_google_goto(final_url):
            return final_url
    except requests.RequestException:
        pass

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        final_url = (response.url or "").strip()
        response.close()
        if final_url and not _is_google_goto(final_url):
            return final_url
    except requests.RequestException:
        pass

    # Keep the original result if Google blocks automated resolution.
    return explicit or url


def _candidate_page_url(row: Dict, timeout: int = 10) -> str:
    """Choose the best page URL and resolve Google redirect wrappers."""
    page_url = (
        row.get("link")
        or row.get("source_url")
        or row.get("host_page_url")
        or row.get("hostPageUrl")
        or row.get("url")
        or ""
    )
    return _resolve_page_url(page_url, timeout=timeout)


def _canonical_page_key(url: str) -> str:
    """Build a stable dedupe key without merging distinct social posts."""
    value = _clean_url(url)
    if not value:
        return ""

    try:
        parsed = urlparse(value)
        host = parsed.netloc.lower().removeprefix("www.")
        path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/").lower()

        # Tracking parameters and image-index query strings should not make
        # the same social post appear as multiple search results.
        if host in {"instagram.com", "x.com", "twitter.com", "linkedin.com"}:
            return f"{host}{path}"

        query = f"?{parsed.query}" if parsed.query else ""
        return f"{host}{path}{query}"
    except Exception:
        return value.lower()


def dedupe(
    candidates: List[Candidate]
) -> List[Candidate]:
    """Deduplicate results while preferring canonical source URLs.

    Lens can return the same image through a Google redirect and the
    canonical publisher URL. The canonical publisher result wins when the
    image is otherwise identical, while exact-match status is preserved.
    """
    def clean(value: str) -> str:
        return (value or "").strip().split("#", 1)[0].rstrip("/")

    def quality(c: Candidate) -> tuple:
        page = clean(c.url).lower()
        image = clean(c.image_url)
        return (
            1 if c.is_exact_match else 0,
            0 if "google.com/goto" in page else 1,
            1 if image else 0,
        )

    best: List[Candidate] = []
    for c in candidates:
        page = clean(c.url)
        image = clean(c.image_url)
        if not page and not image:
            continue
        duplicate = next(
            (
                index
                for index, previous in enumerate(best)
                if (
                    (page and _canonical_page_key(previous.url) == _canonical_page_key(page))
                    or (image and clean(previous.image_url) == image)
                )
            ),
            None,
        )
        if duplicate is None:
            best.append(c)
        elif quality(c) > quality(best[duplicate]):
            best[duplicate] = c

    return best


class ReverseImageSearchProvider:
    """Base class for reverse-image-search providers."""

    name = "base"

    def __init__(
        self,
        api_key: str = "",
        timeout: int = 15,
        **kwargs,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.options = kwargs

    def search(
        self,
        image_url: str,
        limit: int = 20,
    ) -> List[Candidate]:
        raise NotImplementedError

    # --------------------------------------------------------
    # HTTP helper
    # --------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """Perform an HTTP request with common error handling."""

        try:

            resp = requests.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs,
            )

        except requests.RequestException as exc:

            raise SearchError(
                f"{self.name}: network error: {exc}"
            ) from exc

        if resp.status_code == 429:

            raise SearchError(
                f"{self.name}: rate limited "
                "(HTTP 429). Wait and retry, or "
                "reduce MAX_CANDIDATES."
            )

        if resp.status_code in (
            401,
            403,
        ):

            raise SearchError(
                f"{self.name}: authentication failed "
                f"(HTTP {resp.status_code}). "
                "Check your API key."
            )

        if resp.status_code >= 400:

            raise SearchError(
                f"{self.name}: HTTP "
                f"{resp.status_code}: "
                f"{resp.text[:500]}"
            )

        return resp


# ============================================================
# SerpApi Google Lens
# ============================================================


class SerpApiGoogleLensProvider(
    ReverseImageSearchProvider
):
    """SerpApi Google Lens provider.

    The provider intentionally performs:

        exact_matches
             +
        visual_matches

    rather than relying exclusively on type=all.

    This is important because Google Lens exposes exact matches
    as a separate search type.
    """

    name = "serpapi"

    endpoint = (
        "https://serpapi.com/search.json"
    )

    image_endpoint = (
        "https://serpapi.com/image"
    )

    # --------------------------------------------------------
    # Parse a Lens row
    # --------------------------------------------------------

    def _candidate_from_row(
        self,
        row: Dict,
        search_type: str,
    ) -> Candidate:

        page_url = _candidate_page_url(
            row,
            timeout=min(self.timeout, 10),
        )

        # Prefer the actual/full image.
        image_url = (
            row.get("image")
            or row.get("contentUrl")
            or row.get("content_url")
            or row.get("thumbnail")
            or ""
        )

        exact = (
            search_type == "exact_matches"
            or bool(
                row.get(
                    "exact_matches",
                    False,
                )
            )
        )

        source = (
            row.get("source")
            or _domain(page_url)
        )

        position = row.get(
            "position",
            0,
        )

        try:
            position = int(position)
        except (
            TypeError,
            ValueError,
        ):
            position = 0

        return Candidate(
            url=page_url,
            title=row.get("title") or "",
            source=source,
            image_url=image_url,
            is_exact_match=exact,
            search_type=search_type,
            position=position,
        )

    # --------------------------------------------------------
    # Request one Lens type
    # --------------------------------------------------------

    def _search_type(
        self,
        image_url: str,
        search_type: str,
    ) -> List[Candidate]:

        params = {
            "engine": "google_lens",
            "type": search_type,
            "url": image_url,
            "api_key": self.api_key,
        }

        # Optional localization.
        country = (
            self.options.get("country")
            or ""
        ).strip()

        hl = (
            self.options.get("hl")
            or ""
        ).strip()

        if country:
            params["country"] = country

        if hl:
            params["hl"] = hl

        auto_crop = self.options.get("auto_crop")
        if auto_crop is not None:
            params["auto_crop"] = "true" if auto_crop else "false"

        # Force fresh search if requested.
        if self.options.get(
            "no_cache",
            False,
        ):
            params["no_cache"] = "true"

        resp = self._request(
            "GET",
            self.endpoint,
            params=params,
        )

        try:

            data = resp.json()

        except ValueError as exc:

            raise SearchError(
                "serpapi: response was not valid JSON"
            ) from exc

        if data.get("error"):

            raise SearchError(
                f"serpapi: {data['error']}"
            )

        rows = (
            data.get(search_type)
            or []
        )

        if not isinstance(
            rows,
            list,
        ):
            return []

        out: List[Candidate] = []

        for row in rows:

            if not isinstance(
                row,
                dict,
            ):
                continue

            candidate = (
                self._candidate_from_row(
                    row,
                    search_type,
                )
            )

            if (
                candidate.url
                or candidate.image_url
            ):
                out.append(candidate)

        return out

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        image_url: str,
        limit: int = 20,
    ) -> List[Candidate]:
        """Layered Lens search with a conditional broad fallback.

        Exact matches are queried first, visual matches second, and the
        broader ``type=all`` query is used only when the first two passes
        produce too few unique candidates. This improves recall without
        adding a third request to searches that already have enough results.
        """
        if not self.api_key:
            raise SearchError("serpapi: SEARCH_API_KEY is not set")
        if not image_url:
            raise SearchError("serpapi: image URL is empty")

        # First pass: exact matches. If Lens gives us enough exact results,
        # do not pay for a second network request for broad visual matches.
        exact = self._search_type(image_url, "exact_matches")
        combined = dedupe(exact)

        try:
            exact_min = max(
                1,
                int(self.options.get("exact_min_results", 3) or 3),
            )
        except (TypeError, ValueError):
            exact_min = 3

        # A successful exact-match result is normally the highest-value
        # search result for identity verification. Only broaden when the
        # exact pass is weak.
        if len(combined) < exact_min:
            visual = self._search_type(image_url, "visual_matches")
            combined = dedupe(combined + visual)

        try:
            fallback_min = max(
                1,
                int(self.options.get("fallback_min_results", 10) or 10),
            )
        except (TypeError, ValueError):
            fallback_min = 10

        # Optional third pass, only when recall is still poor.
        if len(combined) < fallback_min:
            try:
                broad = self._search_type(image_url, "all")
                combined = dedupe(combined + broad)
            except SearchError:
                pass

        combined.sort(
            key=lambda candidate: (
                not candidate.is_exact_match,
                candidate.position if candidate.position > 0 else 999999,
            )
        )
        for candidate in combined:
            if _is_google_goto(candidate.url):
                candidate.url = _resolve_page_url(
                    candidate.url,
                    timeout=min(self.timeout, 10),
                )
                if not candidate.source:
                    candidate.source = _domain(candidate.url)

        return combined[:limit]

    # Direct SerpApi Image API upload
    # --------------------------------------------------------

    def upload_image(
        self,
        path: str,
    ) -> str:
        """Upload a local image directly to SerpApi.

        Returns an image_id.

        SerpApi documents a 500 KB maximum for this endpoint,
        so callers should resize/compress larger images first.
        """

        if not self.api_key:

            raise SearchError(
                "serpapi: SEARCH_API_KEY "
                "is not set"
            )

        if not os.path.isfile(path):

            raise SearchError(
                f"image not found: {path}"
            )

        try:

            with open(
                path,
                "rb",
            ) as fh:

                response = requests.post(
                    self.image_endpoint,
                    files={
                        "image": (
                            os.path.basename(path),
                            fh,
                        )
                    },
                    data={
                        "api_key": self.api_key,
                    },
                    headers={
                        "User-Agent":
                            "face-chain-verifier/1.0"
                    },
                    timeout=self.timeout,
                )

        except requests.RequestException as exc:

            raise SearchError(
                f"serpapi image upload failed: "
                f"{exc}"
            ) from exc

        if response.status_code >= 400:

            raise SearchError(
                "serpapi image upload failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:

            data = response.json()

        except ValueError as exc:

            raise SearchError(
                "serpapi image upload returned "
                "invalid JSON"
            ) from exc

        if data.get("error"):

            raise SearchError(
                f"serpapi image upload: "
                f"{data['error']}"
            )

        image_id = (
            data.get("image_id")
            or ""
        ).strip()

        if not image_id:

            raise SearchError(
                "serpapi image upload succeeded "
                "but no image_id was returned"
            )

        return image_id

    def search_local_file(
        self,
        path: str,
        limit: int = 20,
    ) -> List[Candidate]:
        """Upload a local image directly to SerpApi and search it.

        This avoids Cloudinary for the Lens query. If direct upload is
        unavailable or fails, callers can fall back to their existing
        public-URL path.
        """
        image_id = self.upload_image(path)
        return self.search_image_id(image_id, limit=limit)

    # --------------------------------------------------------
    # Search using image_id
    # --------------------------------------------------------

    def search_image_id(
        self,
        image_id: str,
        limit: int = 20,
    ) -> List[Candidate]:
        """Run exact + visual Lens search using image_id."""

        if not image_id:

            raise SearchError(
                "serpapi: image_id is empty"
            )

        def search_type_by_id(
            search_type: str,
        ) -> List[Candidate]:

            params = {
                "engine": "google_lens",
                "type": search_type,
                "image_id": image_id,
                "api_key": self.api_key,
            }

            country = (
                self.options.get("country")
                or ""
            ).strip()

            hl = (
                self.options.get("hl")
                or ""
            ).strip()

            if country:
                params["country"] = country

            if hl:
                params["hl"] = hl

            if self.options.get(
                "no_cache",
                False,
            ):
                params["no_cache"] = "true"

            resp = self._request(
                "GET",
                self.endpoint,
                params=params,
            )

            try:

                data = resp.json()

            except ValueError as exc:

                raise SearchError(
                    "serpapi: response was not valid JSON"
                ) from exc

            if data.get("error"):

                raise SearchError(
                    f"serpapi: {data['error']}"
                )

            rows = (
                data.get(search_type)
                or []
            )

            out: List[Candidate] = []

            for row in rows:

                if not isinstance(
                    row,
                    dict,
                ):
                    continue

                candidate = (
                    self._candidate_from_row(
                        row,
                        search_type,
                    )
                )

                if (
                    candidate.url
                    or candidate.image_url
                ):
                    out.append(candidate)

            return out

        # Exact-first: avoid the visual request when exact results are
        # already sufficient for identity verification.
        exact = search_type_by_id("exact_matches")
        combined = dedupe(exact)

        try:
            exact_min = max(
                1,
                int(self.options.get("exact_min_results", 3) or 3),
            )
        except (TypeError, ValueError):
            exact_min = 3

        if len(combined) < exact_min:
            visual = search_type_by_id("visual_matches")
            combined = dedupe(combined + visual)

        try:
            fallback_min = max(
                1,
                int(self.options.get("fallback_min_results", 10) or 10),
            )
        except (TypeError, ValueError):
            fallback_min = 10

        if len(combined) < fallback_min:
            try:
                broad = search_type_by_id("all")
                combined = dedupe(combined + broad)
            except SearchError:
                pass

        combined.sort(
            key=lambda candidate: (
                not candidate.is_exact_match,
                candidate.position
                if candidate.position > 0
                else 999999,
            )
        )

        for candidate in combined:
            if _is_google_goto(candidate.url):
                candidate.url = _resolve_page_url(
                    candidate.url,
                    timeout=min(self.timeout, 10),
                )
                if not candidate.source:
                    candidate.source = _domain(candidate.url)

        return combined[:limit]


# ============================================================
# Bing Visual Search
# ============================================================


class BingVisualSearchProvider(
    ReverseImageSearchProvider
):
    """Azure Bing Visual Search provider."""

    name = "bing"

    def search(
        self,
        image_url: str,
        limit: int = 20,
    ) -> List[Candidate]:

        if not self.api_key:

            raise SearchError(
                "bing: SEARCH_API_KEY "
                "is not set"
            )

        endpoint = (
            self.options.get("endpoint")
            or (
                "https://api.bing.microsoft.com/"
                "v7.0/images/visualsearch"
            )
        )

        knowledge_request = (
            '{"imageInfo":{"url":"%s"}}'
            % image_url
        )

        resp = self._request(
            "POST",
            endpoint,
            headers={
                "Ocp-Apim-Subscription-Key":
                    self.api_key,
            },
            data={
                "knowledgeRequest":
                    knowledge_request,
            },
        )

        try:

            data = resp.json()

        except ValueError as exc:

            raise SearchError(
                "bing: response was not valid JSON"
            ) from exc

        out: List[Candidate] = []

        for tag in data.get(
            "tags",
            [],
        ):

            for action in tag.get(
                "actions",
                [],
            ):

                values = (
                    (
                        action.get(
                            "data"
                        )
                        or {}
                    ).get(
                        "value"
                    )
                    or []
                )

                for value in values:

                    if not isinstance(
                        value,
                        dict,
                    ):
                        continue

                    page_url = (
                        value.get(
                            "hostPageUrl",
                            "",
                        )
                    )

                    image_url_value = (
                        value.get(
                            "contentUrl",
                            "",
                        )
                        or value.get(
                            "thumbnailUrl",
                            "",
                        )
                    )

                    out.append(
                        Candidate(
                            url=page_url,
                            title=(
                                value.get(
                                    "name"
                                )
                                or value.get(
                                    "hostPageDisplayUrl",
                                    "",
                                )
                            ),
                            source=_domain(
                                page_url
                            ),
                            image_url=(
                                image_url_value
                            ),
                            is_exact_match=False,
                            search_type="visual_matches",
                        )
                    )

        return dedupe(
            out
        )[:limit]


# ============================================================
# TinEye
# ============================================================


class TinEyeProvider(
    ReverseImageSearchProvider
):
    """TinEye REST API provider."""

    name = "tineye"

    def search(
        self,
        image_url: str,
        limit: int = 20,
    ) -> List[Candidate]:

        if not self.api_key:

            raise SearchError(
                "tineye: TINEYE_API_KEY "
                "is not set"
            )

        url = (
            self.options.get(
                "api_url"
            )
            or (
                "https://api.tineye.com/"
                "rest/search/"
            )
        )

        resp = self._request(
            "POST",
            url,
            headers={
                "x-api-key":
                    self.api_key,
            },
            data={
                "image_url":
                    image_url,
                "limit":
                    limit,
            },
        )

        try:

            data = resp.json()

        except ValueError as exc:

            raise SearchError(
                "tineye: response was not valid JSON"
            ) from exc

        out: List[Candidate] = []

        results = (
            data.get("results")
            or {}
        )

        for match in results.get(
            "matches",
            [],
        ):

            for backlink in match.get(
                "backlinks",
                [],
            ):

                page_url = (
                    backlink.get(
                        "backlink",
                        "",
                    )
                )

                out.append(
                    Candidate(
                        url=page_url,
                        title=(
                            backlink.get(
                                "crawl_date",
                                "",
                            )
                        ),
                        source=_domain(
                            page_url
                        ),
                        image_url=(
                            backlink.get(
                                "url"
                            )
                            or match.get(
                                "image_url",
                                "",
                            )
                        ),
                        is_exact_match=True,
                        search_type="exact_matches",
                    )
                )

        return dedupe(
            out
        )[:limit]


# ============================================================
# Optional public visual-search fallbacks
# ============================================================


class HeadlessLensProvider(ReverseImageSearchProvider):
    """Best-effort Google Lens upload parser for environments without SerpApi.

    The provider only follows the public 303 result URL and extracts links from
    returned HTML. It does not submit credentials, solve challenges, or access
    private pages. Google can change this endpoint; callers should configure a
    documented API provider when reliability is required.
    """

    name = "headless_lens"
    endpoint = "https://lens.google.com/v3/upload"

    def search(self, image_url: str, limit: int = 20) -> List[Candidate]:
        if not image_url:
            raise SearchError("headless_lens: image URL is empty")
        try:
            response = self._request(
                "GET",
                image_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            html = response.text
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(f"headless_lens: {exc}") from exc

        urls = re.findall(r"https?://[^\"'<>\\s]+", html)
        candidates = []
        for position, url in enumerate(dict.fromkeys(urls), 1):
            if "google." in _domain(url) or url == image_url:
                continue
            candidates.append(Candidate(
                url=url,
                source=_domain(url),
                image_url=url if re.search(r"\.(?:jpe?g|png|webp)(?:[?#]|$)", url, re.I) else "",
                search_type="visual_matches",
                position=position,
            ))
        return dedupe(candidates)[:limit]


class DirectYandexProvider(ReverseImageSearchProvider):
    """Public Yandex image-result parser used as a tertiary best effort."""

    name = "yandex"
    endpoint = "https://yandex.com/images/search"

    def search(self, image_url: str, limit: int = 20) -> List[Candidate]:
        if not image_url:
            raise SearchError("yandex: image URL is empty")
        response = self._request(
            "GET",
            self.endpoint,
            params={"rpt": "imageview", "url": image_url},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        urls = re.findall(r"https?://[^\"'<>\\s]+", response.text)
        return dedupe([
            Candidate(url=url, source=_domain(url), search_type="visual_matches")
            for url in dict.fromkeys(urls)
            if "yandex." not in _domain(url)
        ])[:limit]


class FallbackProvider(ReverseImageSearchProvider):
    """Try providers in order and preserve successful candidates."""

    name = "cascade"

    def __init__(self, providers: List[ReverseImageSearchProvider], **kwargs):
        super().__init__(timeout=kwargs.get("timeout", 15))
        self.providers = providers

    def search(self, image_url: str, limit: int = 20) -> List[Candidate]:
        errors = []
        for provider in self.providers:
            try:
                candidates = provider.search(image_url, limit=limit)
                if candidates:
                    return candidates[:limit]
            except SearchError as exc:
                errors.append(str(exc))
        raise SearchError("all search providers failed: " + " | ".join(errors))


# ============================================================
# Provider factory
# ============================================================


PROVIDERS = {
    SerpApiGoogleLensProvider.name:
        SerpApiGoogleLensProvider,

    BingVisualSearchProvider.name:
        BingVisualSearchProvider,

    TinEyeProvider.name:
        TinEyeProvider,

    HeadlessLensProvider.name:
        HeadlessLensProvider,

    DirectYandexProvider.name:
        DirectYandexProvider,
}


def get_provider(
    name: str,
    cfg,
) -> ReverseImageSearchProvider:
    """Create the configured reverse-image-search provider."""

    key = (
        name or ""
    ).lower()

    if key in {"cascade", "fallback", "multi"}:
        names = [item.strip().lower() for item in cfg.search_fallbacks.split(",") if item.strip()]
        providers = [get_provider(item, cfg) for item in names if item in PROVIDERS]
        if cfg.search_provider not in {"cascade", "fallback", "multi"}:
            providers.insert(0, get_provider(cfg.search_provider, cfg))
        if not providers:
            raise SearchError("no search providers configured for cascade")
        return FallbackProvider(providers, timeout=cfg.http_timeout)

    if key not in PROVIDERS:

        raise SearchError(
            f"unknown SEARCH_PROVIDER "
            f"'{name}'. Available: "
            f"{', '.join(sorted(PROVIDERS))}"
        )

    if key == "tineye":

        return TinEyeProvider(
            api_key=cfg.tineye_api_key,
            timeout=cfg.http_timeout,
            api_url=cfg.tineye_api_url,
        )

    if key == "bing":

        return BingVisualSearchProvider(
            api_key=cfg.search_api_key,
            timeout=cfg.http_timeout,
            endpoint=cfg.bing_endpoint,
        )

    return SerpApiGoogleLensProvider(
        api_key=cfg.search_api_key,
        timeout=cfg.http_timeout,
        auto_crop=True,
        no_cache=False,
        exact_min_results=1,
        fallback_min_results=5,
        country="IN",
        hl="en",
    )


# ============================================================
# Cloudinary upload
# ============================================================


def _upload_to_cloudinary(
    path: str,
    timeout: int = 30,
) -> str:
    """Upload an image to Cloudinary and return its HTTPS URL."""

    try:

        import cloudinary
        import cloudinary.uploader

    except ImportError as exc:

        raise SearchError(
            "Cloudinary SDK is not installed. "
            "Run: pip install cloudinary"
        ) from exc

    cloud_name = os.environ.get(
        "CLOUDINARY_CLOUD_NAME",
        "",
    ).strip()

    api_key = os.environ.get(
        "CLOUDINARY_API_KEY",
        "",
    ).strip()

    api_secret = os.environ.get(
        "CLOUDINARY_API_SECRET",
        "",
    ).strip()

    if not cloud_name:

        raise SearchError(
            "CLOUDINARY_CLOUD_NAME "
            "is not configured"
        )

    if not api_key:

        raise SearchError(
            "CLOUDINARY_API_KEY "
            "is not configured"
        )

    if not api_secret:

        raise SearchError(
            "CLOUDINARY_API_SECRET "
            "is not configured"
        )

    try:

        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
        )

        result = (
            cloudinary.uploader.upload(
                path,
                resource_type="image",
                folder="face-chain-verifier",
                use_filename=False,
                unique_filename=True,
                overwrite=False,
                timeout=timeout,
            )
        )

    except Exception as exc:

        raise SearchError(
            f"Cloudinary upload failed: {exc}"
        ) from exc

    secure_url = (
        result.get("secure_url")
        or ""
    ).strip()

    if not secure_url:

        raise SearchError(
            "Cloudinary upload succeeded "
            "but returned no secure URL"
        )

    return secure_url


# ============================================================
# Generic upload
# ============================================================


def _upload_generic(
    path: str,
    upload_url: str,
    timeout: int,
) -> str:
    """Upload using the legacy generic HTTP endpoint."""

    if not upload_url:

        raise SearchError(
            "no IMAGE_UPLOAD_URL configured "
            "and no --image-url provided"
        )

    try:

        with open(
            path,
            "rb",
        ) as fh:

            resp = requests.post(
                upload_url,
                files={
                    "file": (
                        os.path.basename(path),
                        fh,
                    )
                },
                headers={
                    "User-Agent":
                        "face-chain-verifier/1.0 "
                        "(contact: local demo)"
                },
                timeout=timeout,
            )

    except requests.RequestException as exc:

        raise SearchError(
            f"image upload failed: {exc}"
        ) from exc

    if resp.status_code >= 400:

        raise SearchError(
            f"image upload failed: "
            f"HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )

    url = resp.text.strip()

    if not url.startswith(
        (
            "http://",
            "https://",
        )
    ):

        raise SearchError(
            "image upload returned an "
            "unexpected response: "
            f"{url[:120]}"
        )

    return url


# ============================================================
# Main image upload helper
# ============================================================


def upload_image_for_search(
    path: str,
    upload_url: str,
    timeout: int = 30,
) -> str:
    """Publish a local image at a temporary public URL.

    Supported values:

        cloudinary
            Upload through Cloudinary.

        <HTTP URL>
            Use the legacy generic upload endpoint.

    For SerpApi Google Lens, prefer using the provider's
    ``upload_image()`` method directly because SerpApi can
    accept the local image without Cloudinary.
    """

    if not os.path.isfile(path):

        raise SearchError(
            f"image not found: {path}"
        )

    if (
        upload_url
        and upload_url.strip().lower()
        == "cloudinary"
    ):

        return _upload_to_cloudinary(
            path,
            timeout=timeout,
        )

    return _upload_generic(
        path,
        upload_url,
        timeout,
    )