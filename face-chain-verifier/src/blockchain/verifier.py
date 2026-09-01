"""Submit fingerprints on-chain and verify them by reading the record back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .contract import ChainError, account_from_key, connect, get_contract
from ..utils.hashing import from_bytes32, to_bytes32


@dataclass
class OnChainRecord:
    data_hash: str
    image_hash: str
    source_url: str
    timestamp: int
    submitter: str
    record_id: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_hash": self.data_hash,
            "image_hash": self.image_hash,
            "source_url": self.source_url,
            "timestamp": self.timestamp,
            "submitter": self.submitter,
            "record_id": self.record_id,
        }


def parse_record(raw, data_hash: str = "") -> OnChainRecord:
    """Parse a tuple returned by ``getRecord``/``getLatestByHash``.

    ``getRecord``      -> (dataHash, imageHash, sourceUrl, timestamp, submitter)
    ``getLatestByHash`` -> (imageHash, sourceUrl, timestamp, submitter, recordId)
    """
    if raw is None:
        raise ChainError("empty on-chain response")
    items = list(raw)
    if len(items) == 5 and isinstance(items[4], int):
        image_hash, source_url, timestamp, submitter, record_id = items
        return OnChainRecord(
            data_hash=data_hash.removeprefix("0x"),
            image_hash=from_bytes32(bytes(image_hash)),
            source_url=source_url,
            timestamp=int(timestamp),
            submitter=str(submitter),
            record_id=int(record_id),
        )
    if len(items) == 5:
        dh, image_hash, source_url, timestamp, submitter = items
        return OnChainRecord(
            data_hash=from_bytes32(bytes(dh)),
            image_hash=from_bytes32(bytes(image_hash)),
            source_url=source_url,
            timestamp=int(timestamp),
            submitter=str(submitter),
            record_id=-1,
        )
    raise ChainError(f"unexpected on-chain record shape: {len(items)} fields")


class BlockchainVerifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.w3 = connect(cfg.rpc_url)
        self.contract = get_contract(self.w3, cfg.contract_address)

    # --------------------------------------------------------------- #

    def store(self, data_hash: str, image_hash: str, source_url: str) -> Dict[str, Any]:
        """Send the storeRecord transaction. Returns tx info."""
        acct = account_from_key(self.w3, self.cfg.private_key)
        balance = self.w3.eth.get_balance(acct.address)
        if balance == 0:
            raise ChainError(
                f"insufficient testnet funds: {acct.address} has 0 ETH. "
                "Fund it from a Sepolia faucet before submitting."
            )

        fn = self.contract.functions.storeRecord(
            to_bytes32(data_hash),
            to_bytes32(image_hash) if image_hash else b"\x00" * 32,
            source_url,
        )
        try:
            gas = fn.estimate_gas({"from": acct.address})
        except Exception as exc:
            raise ChainError(f"gas estimation failed (transaction would revert): {exc}") from exc

        tx = fn.build_transaction(
            {
                "from": acct.address,
                "nonce": self.w3.eth.get_transaction_count(acct.address),
                "chainId": self.cfg.chain_id,
                "gas": int(gas * 1.2),
                "maxFeePerGas": self.w3.eth.gas_price * 2,
                "maxPriorityFeePerGas": self.w3.to_wei(1.5, "gwei"),
            }
        )
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        try:
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise ChainError(f"transaction submission failed: {exc}") from exc

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        if receipt.status != 1:
            raise ChainError(f"transaction reverted on-chain: {tx_hash.hex()}")

        record_id: Optional[int] = None
        try:
            logs = self.contract.events.RecordStored().process_receipt(receipt)
            if logs:
                record_id = int(logs[0]["args"]["recordId"])
        except Exception:
            record_id = None

        tx_hex = tx_hash.hex()
        if not tx_hex.startswith("0x"):
            tx_hex = "0x" + tx_hex
        return {
            "tx_hash": tx_hex,
            "block_number": receipt.blockNumber,
            "gas_used": receipt.gasUsed,
            "record_id": record_id,
            "submitter": acct.address,
            "explorer_url": f"{self.cfg.explorer_base.rstrip('/')}/tx/{tx_hex}",
        }

    def read_back(self, data_hash: str, record_id: Optional[int] = None) -> OnChainRecord:
        try:
            if record_id is not None and record_id >= 0:
                raw = self.contract.functions.getRecord(record_id).call()
                return parse_record(raw)
            raw = self.contract.functions.getLatestByHash(to_bytes32(data_hash)).call()
            return parse_record(raw, data_hash=data_hash)
        except ChainError:
            raise
        except Exception as exc:
            raise ChainError(f"could not read the record back from chain: {exc}") from exc

    @staticmethod
    def compare(local_hash: str, onchain_hash: str) -> bool:
        return local_hash.removeprefix("0x").lower() == onchain_hash.removeprefix("0x").lower()


# --------------------------------------------------------------------- #
# Batch anchoring
# --------------------------------------------------------------------- #

@dataclass
class BatchReceipt:
    """Result of anchoring one chunk of records in a single transaction."""

    tx_hash: str
    block_number: int
    gas_used: int
    first_record_id: int
    batch_id: int
    count: int
    batch_root: str
    explorer_url: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "gas_used": self.gas_used,
            "gas_per_record": round(self.gas_used / self.count, 1) if self.count else 0,
            "first_record_id": self.first_record_id,
            "batch_id": self.batch_id,
            "count": self.count,
            "batch_root": self.batch_root,
            "explorer_url": self.explorer_url,
        }


def _chunk(items, size):
    for i in range(0, len(items), size):
        yield i, items[i : i + size]


class BatchBlockchainVerifier(BlockchainVerifier):
    """Anchors many fingerprints per transaction and verifies them in bulk."""

    MAX_CHUNK = 200

    def store_batch(self, records, batch_root: str = "", chunk_size: int = 25):
        """Anchor ``records`` (list of ``(data_hash, image_hash, source_url)``).

        Records are split into chunks so a single transaction stays well under
        the block gas limit; each chunk is one transaction. Returns a list of
        :class:`BatchReceipt`.
        """
        from ..utils.merkle import merkle_root

        records = list(records)
        if not records:
            raise ChainError("nothing to anchor: the batch contains zero verified records")
        chunk_size = max(1, min(int(chunk_size), self.MAX_CHUNK))

        acct = account_from_key(self.w3, self.cfg.private_key)
        if self.w3.eth.get_balance(acct.address) == 0:
            raise ChainError(
                f"insufficient testnet funds: {acct.address} has 0 ETH. Fund it from a Sepolia faucet."
            )

        root = batch_root or merkle_root([r[0] for r in records])
        receipts = []
        nonce = self.w3.eth.get_transaction_count(acct.address)

        for _offset, chunk in _chunk(records, chunk_size):
            data_hashes = [to_bytes32(r[0]) for r in chunk]
            image_hashes = [to_bytes32(r[1]) if r[1] else b"\x00" * 32 for r in chunk]
            source_urls = [r[2] for r in chunk]

            fn = self.contract.functions.storeBatch(to_bytes32(root), data_hashes, image_hashes, source_urls)
            try:
                gas = fn.estimate_gas({"from": acct.address})
            except Exception as exc:
                raise ChainError(
                    f"gas estimation failed for a batch of {len(chunk)} records "
                    f"(try a smaller --chunk-size): {exc}"
                ) from exc

            tx = fn.build_transaction(
                {
                    "from": acct.address,
                    "nonce": nonce,
                    "chainId": self.cfg.chain_id,
                    "gas": int(gas * 1.2),
                    "maxFeePerGas": self.w3.eth.gas_price * 2,
                    "maxPriorityFeePerGas": self.w3.to_wei(1.5, "gwei"),
                }
            )
            signed = acct.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            try:
                tx_hash = self.w3.eth.send_raw_transaction(raw)
            except Exception as exc:
                raise ChainError(f"batch transaction submission failed: {exc}") from exc
            nonce += 1

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
            if receipt.status != 1:
                raise ChainError(f"batch transaction reverted on-chain: {tx_hash.hex()}")

            first_id, batch_id = -1, -1
            try:
                logs = self.contract.events.BatchStored().process_receipt(receipt)
                if logs:
                    first_id = int(logs[0]["args"]["firstRecordId"])
                    batch_id = int(logs[0]["args"]["batchId"])
            except Exception:
                pass

            tx_hex = tx_hash.hex()
            if not tx_hex.startswith("0x"):
                tx_hex = "0x" + tx_hex
            receipts.append(
                BatchReceipt(
                    tx_hash=tx_hex,
                    block_number=receipt.blockNumber,
                    gas_used=receipt.gasUsed,
                    first_record_id=first_id,
                    batch_id=batch_id,
                    count=len(chunk),
                    batch_root=root,
                    explorer_url=f"{self.cfg.explorer_base.rstrip('/')}/tx/{tx_hex}",
                )
            )
        return receipts

    def read_back_range(self, start: int, count: int):
        """Bulk read-back of stored dataHashes: one RPC call for the whole chunk."""
        try:
            raw = self.contract.functions.getDataHashes(int(start), int(count)).call()
        except Exception as exc:
            raise ChainError(f"bulk read-back failed: {exc}") from exc
        return [from_bytes32(bytes(h)) for h in raw]

    def verify_batch(self, receipts, local_hashes):
        """Read every anchored hash back and compare it with the local recomputation.

        Returns ``(all_ok, rows)`` where each row is
        ``{"index", "local", "onchain", "verified"}``.
        """
        rows, cursor, all_ok = [], 0, True
        for r in receipts:
            expected = local_hashes[cursor : cursor + r.count]
            if r.first_record_id < 0:
                onchain = []
                for i in range(r.count):
                    onchain.append(self.read_back(expected[i]).data_hash)
            else:
                onchain = self.read_back_range(r.first_record_id, r.count)
            for i, (local, stored) in enumerate(zip(expected, onchain)):
                verified = self.compare(local, stored)
                all_ok = all_ok and verified
                rows.append(
                    {
                        "index": cursor + i,
                        "record_id": (r.first_record_id + i) if r.first_record_id >= 0 else -1,
                        "local": local.removeprefix("0x"),
                        "onchain": stored.removeprefix("0x"),
                        "verified": verified,
                    }
                )
            cursor += r.count
        return all_ok, rows

    def verify_batch_root(self, batch_id: int, local_hashes) -> bool:
        """Recompute the merkle root locally and compare it with the on-chain anchor."""
        from ..utils.merkle import merkle_root

        try:
            root, _first, _count, _ts, _submitter = self.contract.functions.getBatch(int(batch_id)).call()
        except Exception as exc:
            raise ChainError(f"could not read batch anchor {batch_id}: {exc}") from exc
        return self.compare(merkle_root(list(local_hashes)), from_bytes32(bytes(root)))
