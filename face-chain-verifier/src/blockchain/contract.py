"""Solidity compilation + contract binding helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import solcx

SOLC_VERSION = "0.8.20"

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "FaceVerification.sol"
BUILD_PATH = ROOT / "build" / "FaceVerification.json"

SOLC_PATH = Path.home() / ".solcx" / "solc-v0.8.20.exe"


class ChainError(RuntimeError):
    """Any recoverable blockchain problem."""


def compile_contract(source_path: Path = CONTRACT_PATH) -> Tuple[list, str]:
    """Compile the contract, returning (abi, bytecode). Caches to build/."""

    if not source_path.is_file():
        raise ChainError(f"contract source not found: {source_path}")

    if not SOLC_PATH.is_file():
        raise ChainError(
            f"local solc compiler not found: {SOLC_PATH}"
        )

    try:
        compiled = solcx.compile_files(
            [str(source_path)],
            output_values=["abi", "bin"],
            solc_binary=str(SOLC_PATH),
            optimize=True,
        )
    except Exception as exc:
        raise ChainError(f"solidity compilation failed: {exc}") from exc

    key = next(
        (k for k in compiled if k.endswith(":FaceVerification")),
        None,
    )

    if key is None:
        raise ChainError("FaceVerification not found in compiler output")

    abi = compiled[key]["abi"]
    bytecode = "0x" + compiled[key]["bin"].removeprefix("0x")

    BUILD_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUILD_PATH.write_text(
        json.dumps(
            {
                "abi": abi,
                "bytecode": bytecode,
            },
            indent=2,
        )
    )

    return abi, bytecode


def load_artifact() -> Dict[str, Any]:
    """Load a previously compiled artifact, compiling on demand."""

    if BUILD_PATH.is_file():
        try:
            return json.loads(BUILD_PATH.read_text())
        except json.JSONDecodeError:
            pass

    abi, bytecode = compile_contract()
    return {"abi": abi, "bytecode": bytecode}


def load_abi() -> list:
    return load_artifact()["abi"]


def connect(rpc_url: str):
    """Return a connected Web3 instance or raise ChainError."""

    try:
        from web3 import Web3
    except ImportError as exc:
        raise ChainError(
            "web3 is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if not rpc_url:
        raise ChainError("RPC_URL is not configured")

    w3 = Web3(
        Web3.HTTPProvider(
            rpc_url,
            request_kwargs={"timeout": 30},
        )
    )

    try:
        connected = w3.is_connected()
    except Exception as exc:
        raise ChainError(
            f"cannot reach RPC endpoint: {exc}"
        ) from exc

    if not connected:
        raise ChainError(
            f"blockchain RPC unavailable at {rpc_url}"
        )

    return w3


def get_contract(w3, address: str):
    """Bind to a deployed contract, validating the address and that code exists."""

    from web3 import Web3

    if not address:
        raise ChainError(
            "CONTRACT_ADDRESS is not set. "
            "Deploy first: python scripts/deploy.py"
        )

    if not Web3.is_address(address):
        raise ChainError(f"invalid contract address: {address}")

    checksum = Web3.to_checksum_address(address)

    if w3.eth.get_code(checksum) in (b"", b"0x"):
        raise ChainError(
            f"no contract code at {checksum} on this network - "
            "wrong address or wrong chain"
        )

    return w3.eth.contract(
        address=checksum,
        abi=load_abi(),
    )


def account_from_key(w3, private_key: str):
    if not private_key:
        raise ChainError("PRIVATE_KEY is not set")

    try:
        return w3.eth.account.from_key(private_key)
    except Exception as exc:
        raise ChainError(f"invalid PRIVATE_KEY: {exc}") from exc


def env_or(name: str, fallback: str = "") -> str:
    return (os.environ.get(name) or fallback).strip()