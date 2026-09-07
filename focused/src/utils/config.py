"""Central configuration loaded from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional at import time so unit tests never require the package
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:  # pragma: no cover
    pass


def _get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass
class Config:
    # search
    search_provider: str = field(default_factory=lambda: _get("SEARCH_PROVIDER", "serpapi").lower())
    search_api_key: str = field(default_factory=lambda: _get("SEARCH_API_KEY") or _get("SERPAPI_KEY"))
    bing_endpoint: str = field(
        default_factory=lambda: _get(
            "BING_VISUAL_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/images/visualsearch"
        )
    )
    tineye_api_url: str = field(default_factory=lambda: _get("TINEYE_API_URL", "https://api.tineye.com/rest/search/"))
    tineye_api_key: str = field(default_factory=lambda: _get("TINEYE_API_KEY"))
    image_upload_url: str = field(default_factory=lambda: _get("IMAGE_UPLOAD_URL", "https://0x0.st"))

    # face
    face_model: str = field(default_factory=lambda: _get("FACE_MODEL", "buffalo_l"))
    face_det_size: int = field(default_factory=lambda: _get_int("FACE_DET_SIZE", 640))
    match_threshold: float = field(default_factory=lambda: _get_float("MATCH_THRESHOLD", 0.45))
    max_candidates: int = field(default_factory=lambda: _get_int("MAX_CANDIDATES", 20))
    max_image_bytes: int = field(default_factory=lambda: _get_int("MAX_IMAGE_BYTES", 8 * 1024 * 1024))
    http_timeout: int = field(default_factory=lambda: _get_int("HTTP_TIMEOUT", 15))

    # Optional public-only OSINT extensions
    search_fallbacks: str = field(default_factory=lambda: _get("SEARCH_FALLBACKS", "headless_lens,yandex"))
    search_context: str = field(default_factory=lambda: _get("SEARCH_CONTEXT"))
    pivot_enabled: bool = field(default_factory=lambda: _get("PIVOT_ENABLED", "true").lower() not in {"0", "false", "no"})
    pivot_max_sites: int = field(default_factory=lambda: _get_int("PIVOT_MAX_SITES", 300))
    pivot_timeout: float = field(default_factory=lambda: _get_float("PIVOT_TIMEOUT", 8.0))
    pivot_max_workers: int = field(default_factory=lambda: _get_int("PIVOT_MAX_WORKERS", 12))
    pivot_max_accounts: int = field(default_factory=lambda: _get_int("PIVOT_MAX_ACCOUNTS", 25))
    pivot_max_candidates: int = field(default_factory=lambda: _get_int("PIVOT_MAX_CANDIDATES", 50))
    pivot_browser_fallback: bool = field(default_factory=lambda: _get("PIVOT_BROWSER_FALLBACK", "false").lower() in {"1", "true", "yes"})
    pivot_sweep_timeout: float = field(default_factory=lambda: _get_float("PIVOT_SWEEP_TIMEOUT", 30.0))
    pivot_exhaustive: bool = field(default_factory=lambda: _get("PIVOT_EXHAUSTIVE", "false").lower() in {"1", "true", "yes"})

    # batch processing
    batch_workers: int = field(default_factory=lambda: _get_int("BATCH_WORKERS", 8))
    face_pool_size: int = field(default_factory=lambda: _get_int("FACE_POOL_SIZE", 0))
    batch_chunk_size: int = field(default_factory=lambda: _get_int("BATCH_CHUNK_SIZE", 25))


    # blockchain
    rpc_url: str = field(default_factory=lambda: _get("RPC_URL"))
    chain_id: int = field(default_factory=lambda: _get_int("CHAIN_ID", 11155111))
    private_key: str = field(default_factory=lambda: _get("PRIVATE_KEY"))
    contract_address: str = field(default_factory=lambda: _get("CONTRACT_ADDRESS"))
    explorer_base: str = field(default_factory=lambda: _get("EXPLORER_BASE", "https://sepolia.etherscan.io"))

    def require_search(self) -> None:
        if self.search_provider in {"cascade", "fallback", "multi"}:
            return
        if self.search_provider == "serpapi" and not self.search_api_key:
            raise ConfigError("SEARCH_API_KEY is required for SEARCH_PROVIDER=serpapi (get one at serpapi.com).")
        if self.search_provider == "bing" and not self.search_api_key:
            raise ConfigError("SEARCH_API_KEY is required for SEARCH_PROVIDER=bing (Azure Bing Search v7 key).")
        if self.search_provider == "tineye" and not self.tineye_api_key:
            raise ConfigError("TINEYE_API_KEY is required for SEARCH_PROVIDER=tineye.")

    def require_chain(self, need_key: bool = True) -> None:
        if not self.rpc_url:
            raise ConfigError("RPC_URL is not set (e.g. an Ethereum Sepolia RPC endpoint).")
        if need_key and not self.private_key:
            raise ConfigError("PRIVATE_KEY is not set. Use a throwaway testnet key only.")
        if need_key and not self.private_key.startswith("0x"):
            raise ConfigError("PRIVATE_KEY must be a 0x-prefixed hex string.")


def load_config() -> Config:
    return Config()


# Shared data location for optional public-only OSINT extensions.  Keeping it
# relative to the repository preserves the existing deployment configuration.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = DATA_DIR / "results"
CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"

SERPAPI_KEY = _get("SERPAPI_KEY") or _get("SEARCH_API_KEY")
RPC_URL = _get("RPC_URL")
PRIVATE_KEY = _get("PRIVATE_KEY")
CONTRACT_ADDRESS = _get("CONTRACT_ADDRESS")
PIVOT_BROWSER_FALLBACK = _get("PIVOT_BROWSER_FALLBACK", "false").lower() in {"1", "true", "yes"}
PIVOT_EXHAUSTIVE = _get("PIVOT_EXHAUSTIVE", "false").lower() in {"1", "true", "yes"}
PIVOT_MAX_ACCOUNTS = _get_int("PIVOT_MAX_ACCOUNTS", 25)
PIVOT_MAX_SITES = _get_int("PIVOT_MAX_SITES", 300)
PIVOT_MAX_WORKERS = _get_int("PIVOT_MAX_WORKERS", 12)
PIVOT_SWEEP_TIMEOUT = _get_float("PIVOT_SWEEP_TIMEOUT", 30.0)
PIVOT_TIMEOUT = _get_float("PIVOT_TIMEOUT", 8.0)
PIVOT_MAX_CANDIDATES = _get_int("PIVOT_MAX_CANDIDATES", 50)


def require_env(name: str, value: str) -> str:
    """Return *value* if non-empty and not a placeholder, otherwise exit with a clear message."""
    placeholders = ("your_", "YOUR_", "0x_deployed", "_here")
    if not value or any(p in value for p in placeholders):
        import sys
        print(f"\n  ✗ Missing or unconfigured environment variable: {name}")
        print(f"    Current value: '{value}'")
        print(f"    Set a valid value in your .env file or export it in your shell.\n")
        sys.exit(1)
    return value


def require_search_config(optional: bool = True) -> str:
    placeholders = ("your_", "YOUR_", "_here")
    key = SERPAPI_KEY or _get("SEARCH_API_KEY")
    if not key or any(p in key for p in placeholders):
        if optional:
            return ""
        return require_env("SERPAPI_KEY", key)
    return key


def require_blockchain_config() -> tuple[str, str, str]:
    rpc = require_env("RPC_URL", RPC_URL or _get("RPC_URL"))
    pk = require_env("PRIVATE_KEY", PRIVATE_KEY or _get("PRIVATE_KEY"))
    ca = require_env("CONTRACT_ADDRESS", CONTRACT_ADDRESS or _get("CONTRACT_ADDRESS"))
    return rpc, pk, ca
