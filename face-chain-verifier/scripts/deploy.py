"""Deploy FaceVerification.sol to the configured network (default: Sepolia).

Usage:
    python scripts/deploy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blockchain.contract import ChainError, account_from_key, compile_contract, connect  # noqa: E402
from src.utils.config import ConfigError, load_config  # noqa: E402
from src.utils.logging_ui import banner, fail, kv, ok, step  # noqa: E402


def main() -> int:
    cfg = load_config()
    banner("deploy FaceVerification")
    try:
        cfg.require_chain(need_key=True)
        step("Compiling contract...")
        abi, bytecode = compile_contract()
        ok("Compiled")

        w3 = connect(cfg.rpc_url)
        acct = account_from_key(w3, cfg.private_key)
        kv("Network chainId", w3.eth.chain_id)
        kv("Deployer", acct.address)
        balance = w3.eth.get_balance(acct.address)
        kv("Balance (ETH)", w3.from_wei(balance, "ether"))
        if balance == 0:
            raise ChainError("insufficient testnet funds - use a Sepolia faucet first")

        contract = w3.eth.contract(abi=abi, bytecode=bytecode)
        tx = contract.constructor().build_transaction(
            {
                "from": acct.address,
                "nonce": w3.eth.get_transaction_count(acct.address),
                "chainId": cfg.chain_id,
                "gas": 1_500_000,
                "maxFeePerGas": w3.eth.gas_price * 2,
                "maxPriorityFeePerGas": w3.to_wei(1.5, "gwei"),
            }
        )
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        step("Submitting deployment transaction...")
        tx_hash = w3.eth.send_raw_transaction(raw)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        if receipt.status != 1:
            raise ChainError("deployment transaction reverted")

        banner("deployed")
        kv("Contract", receipt.contractAddress)
        kv("Tx", "0x" + tx_hash.hex().removeprefix("0x"))
        kv("Explorer", f"{cfg.explorer_base.rstrip('/')}/address/{receipt.contractAddress}")
        print()
        print(f"  Add this to your .env:\n\n  CONTRACT_ADDRESS={receipt.contractAddress}\n")
        return 0
    except (ChainError, ConfigError) as exc:
        fail(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
