# face-chain-verifier

**Face Identification & Blockchain Verification Pipeline** — HH Goa 2026 Shortlisting Task 3.

A Python CLI that takes a face photo, finds where that face actually appears on the public web
(through a **live** reverse-image-search API), independently re-verifies every candidate with
ArcFace embeddings, fingerprints the confirmed match with SHA-256, anchors that fingerprint on
the **Ethereum Sepolia** testnet, and then **reads the record back from chain** to prove integrity.

No website, no database, no fake data. One terminal command.

---

## Project Overview

```
python src/main.py --image examples/input.jpg
```

1. **Face detection & embedding** — InsightFace (`buffalo_l`, RetinaFace detector + ArcFace
   recognition head) produces a 512-D L2-normalised embedding of the primary face.
2. **Live web search** — the image is submitted to a real reverse-image-search provider
   (SerpApi Google Lens by default) and structured candidates come back.
3. **Independent verification** — every candidate thumbnail is downloaded, re-detected and
   embedded with the *same* pipeline, then compared to the reference by cosine similarity.
4. **Match selection** — candidates are ranked; the strongest one above a configurable
   threshold wins. If nothing clears the bar, the run fails honestly.
5. **Post extraction** — public OpenGraph/HTML metadata of the matching page is normalised.
6. **Fingerprinting** — SHA-256 over the canonical JSON of that record (and over the raw
   matched image bytes).
7. **Blockchain anchor** — `storeRecord(dataHash, imageHash, sourceUrl)` on Sepolia.
8. **Read-back verification** — the record is fetched from chain, the hash is recomputed
   locally, and the two are compared. ✅ VERIFIED / ❌ FAILED.

## Architecture

```text
Face Input
   ↓
Face Detection            (src/face/detector.py — InsightFace / RetinaFace)
   ↓
Face Embedding            (ArcFace, 512-D, L2-normalised)
   ↓
Reverse Image Search      (src/search/reverse_search.py — pluggable provider)
   ↓
Candidate Images          (src/search/candidate_fetcher.py — safe, size-capped download)
   ↓
Face Similarity Matching  (src/face/matcher.py — cosine similarity + ranking)
   ↓
Matching Post             (normalised public metadata)
   ↓
SHA-256 Fingerprint       (src/utils/hashing.py — canonical JSON)
   ↓
Blockchain                (contracts/FaceVerification.sol on Sepolia)
   ↓
Verification              (src/blockchain/verifier.py — on-chain read-back + compare)
```

### Layout

```text
face-chain-verifier/
├── README.md  requirements.txt  .env.example  .gitignore
├── contracts/FaceVerification.sol
├── scripts/deploy.py
├── src/
│   ├── main.py       CLI: single-image and batch modes
│   ├── pipeline.py   reusable per-image discovery pipeline + candidate cache
│   ├── batch.py      concurrent batch runner, checkpointing, JSON/CSV reports
│   ├── face/       detector.py (+DetectorPool)  matcher.py (vectorised)
│   ├── search/     reverse_search.py  candidate_fetcher.py
│   ├── blockchain/ contract.py  verifier.py (+BatchBlockchainVerifier)
│   └── utils/      hashing.py  merkle.py  config.py  logging_ui.py
├── tests/  test_hashing.py  test_face_matching.py  test_blockchain.py
│           test_search.py  test_batch.py  test_merkle.py
└── examples/
```

## Features

- InsightFace/ArcFace detection with bounding boxes, detection scores and embedding dimensions printed.
- Multi-face handling: largest face auto-selected, `--face-index N` to override.
- Pluggable reverse-image-search layer: **SerpApi (Google Lens)**, **Bing Visual Search**, **TinEye**.
  Adding a provider = one subclass + one registry entry.
- Deduplication of candidates by page URL and image URL.
- SSRF-safe candidate downloads: http(s) only, public IPs only, content-type allow-list,
  hard byte cap, request timeouts.
- Independent face re-verification of every candidate — search results are never trusted blindly.
- Configurable cosine-similarity threshold (env, or `--threshold`).
- Public post metadata extraction (OpenGraph/title/description/canonical) — never bypasses
  logins, CAPTCHAs or private content.
- Deterministic canonical-JSON SHA-256 fingerprinting plus raw image-bytes SHA-256.
- Minimal Solidity contract with an indexed `RecordStored` event and read-back getters.
- Mandatory on-chain read-back verification with explicit VERIFIED / FAILED output.
- `--save-results` debug snapshots (clearly labelled; never used as input).
- `--no-chain` mode for demoing discovery + hashing without spending testnet gas.
- **Batch mode for 50-100+ images**: thread-pooled workers, a pooled face model,
  a batch-wide candidate cache, input de-duplication, live progress with ETA,
  resumable checkpoints and JSON/CSV reports.
- **Batched anchoring**: `storeBatch` writes up to 200 fingerprints per transaction
  under one merkle root, with bulk read-back verification of every record.
- 51 unit tests, fully mocked, no network or chain required.

## Tech Stack

| Technology | Why |
|---|---|
| **InsightFace (`buffalo_l`)** | State-of-the-art open-source RetinaFace + ArcFace; runs on CPU, gives comparable normalised embeddings suited to cosine similarity. |
| **ONNX Runtime** | CPU inference for the InsightFace models, no GPU/CUDA required. |
| **OpenCV (headless)** | Fast, dependency-light image decoding for local files and downloaded bytes. |
| **SerpApi Google Lens** | Legitimate, documented API over Google's visual match index — the broadest real-world coverage available without scraping. Swappable for Bing/TinEye. |
| **requests + BeautifulSoup** | Simple, well-understood HTTP and public HTML metadata parsing. |
| **hashlib SHA-256** | Standard, deterministic, collision-resistant fingerprint; 32 bytes fits `bytes32` exactly. |
| **Solidity 0.8.20** | Built-in overflow checks and custom errors; only fingerprints are stored on-chain. |
| **web3.py + py-solc-x** | Pure-Python compile, deploy, transact and read-back — no Node/Hardhat toolchain needed. |
| **Ethereum Sepolia** | The current, well-supported Ethereum testnet with free faucets and Etherscan support. |
| **pytest** | Mock-based unit tests that never require a live API or a real transaction. |

## Installation

```bash
git clone <repo>
cd face-chain-verifier

# Windows PowerShell (Python 3.12 recommended)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

The InsightFace model pack (~300 MB) downloads automatically to `~/.insightface` on first run.

### Windows installation troubleshooting

The project requires `insightface==1.0.1`, whose official PyPI wheel does not
compile native code during installation. If pip still downloads
`insightface-0.7.3.tar.gz` or reports that Microsoft Visual C++ 14 is required,
your checkout still has the older requirements file. Confirm and reinstall:

```powershell
Select-String insightface requirements.txt  # must show insightface==1.0.1
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -c "from insightface.app import FaceAnalysis; print('InsightFace ready')"
```

## Environment Variables

| Variable | Meaning |
|---|---|
| `SEARCH_PROVIDER` | `serpapi` (default), `bing`, or `tineye`. |
| `SEARCH_API_KEY` | API key for the selected provider (SerpApi or Bing). |
| `BING_VISUAL_SEARCH_ENDPOINT` | Azure Bing Visual Search endpoint (Bing only). |
| `TINEYE_API_URL` / `TINEYE_API_KEY` | TinEye REST credentials (TinEye only). |
| `IMAGE_UPLOAD_URL` | Temporary public host for the query image so the search API can fetch it. Skipped when you pass `--image-url`. |
| `FACE_MODEL` | InsightFace model pack. Default `buffalo_l`. |
| `FACE_DET_SIZE` | Detector input size (default `640`). |
| `MATCH_THRESHOLD` | Cosine-similarity cut-off. Default `0.45`. **Empirical** — see Limitations. |
| `MAX_CANDIDATES` | How many search candidates to evaluate (default `20`). |
| `MAX_IMAGE_BYTES` | Hard download cap per candidate image (default 8 MiB). |
| `HTTP_TIMEOUT` | Per-request timeout in seconds. |
| `RPC_URL` | Sepolia JSON-RPC endpoint. |
| `CHAIN_ID` | `11155111` for Sepolia. |
| `PRIVATE_KEY` | 0x-prefixed key of a **throwaway testnet** account. Never a real one. |
| `CONTRACT_ADDRESS` | Address printed by `scripts/deploy.py`. |
| `EXPLORER_BASE` | Block explorer base URL, default `https://sepolia.etherscan.io`. |

### Credentials you need to obtain

1. **SerpApi key** — sign up at <https://serpapi.com/manage-api-key>.
   Free tier: **100 searches/month**, no card required. Put it in `.env` as `SEARCH_API_KEY`.
   *Alternative:* Azure Bing Visual Search (Bing Search v7 resource, free F1 tier ≈ 1 000
   transactions/month, 3 per second) → set `SEARCH_PROVIDER=bing` and the same `SEARCH_API_KEY`.
   *Alternative:* TinEye REST API (paid search bundles) → `SEARCH_PROVIDER=tineye`, `TINEYE_API_KEY`.
2. **Sepolia RPC URL** — the public default works; for reliability use Alchemy or Infura
   (free tiers) and put the HTTPS URL in `RPC_URL`.
3. **Sepolia test ETH** — create a fresh wallet, export its private key into `PRIVATE_KEY`,
   and fund it from <https://sepoliafaucet.com>, <https://www.alchemy.com/faucets/ethereum-sepolia>,
   or the Google Cloud Sepolia faucet. ~0.01 ETH is plenty.

All credentials live in `.env`, which is git-ignored. Nothing is ever printed to the terminal.

## Smart Contract Deployment

```bash
python scripts/deploy.py
```

This compiles `contracts/FaceVerification.sol` with solc 0.8.20 (installed automatically on
first run), deploys it from `PRIVATE_KEY` to `RPC_URL`, and prints:

```text
Contract   0xAbC...
Tx         0x...
Explorer   https://sepolia.etherscan.io/address/0xAbC...
```

Copy the address into `.env` as `CONTRACT_ADDRESS`. The ABI is cached at `build/FaceVerification.json`.

## Running

```bash
python src/main.py --image examples/input.jpg
```

Useful flags:

```bash
--image-url https://...      # skip the temporary upload; use an image you already host
--threshold 0.5              # override MATCH_THRESHOLD for this run
--max-candidates 30          # evaluate more search results
--provider bing              # switch reverse-image-search provider
--face-index 1               # choose a specific face when several are detected
--save-results               # write debug/last_run.json (live data, labelled as a snapshot)
--no-chain                   # discovery + hashing only, no transaction
```

Exit codes: `0` verified · `2` face stage failed · `3` search stage failed ·
`4` no candidate above threshold · `5` blockchain submission failed · `6` verification failed.

### Sample output

```text
============================================================
FACE IDENTIFICATION
============================================================
  Loading image: examples/input.jpg
  Image size           1024x1024
  Faces detected: 1 ✓
    face[0]            bbox=(311,208)-(688,701) score=0.884 dim=512
  Embedding generated ✓
  Embedding dims       512
============================================================
WEB SEARCH
============================================================
  Performing LIVE reverse image search (provider: serpapi)...
  Found 18 unique candidates ✓
============================================================
FACE MATCHING
============================================================
  Candidate  1 (example.com) -> similarity 0.324
  Candidate  2 (news.test)   -> no face in candidate image
  Candidate  3 (site.test)   -> similarity 0.781 ← MATCH
  MATCH FOUND ✓
============================================================
DATA FINGERPRINTING
============================================================
  SHA-256 (metadata):
  9f8c1e...
============================================================
BLOCKCHAIN VERIFICATION
============================================================
  Stored hash:      9f8c1e...
  Recomputed hash:  9f8c1e...
  Result:
  ✅ VERIFIED - DATA INTEGRITY CONFIRMED
```

## Batch mode (50-100+ images)

Batch mode runs the exact same genuine pipeline for every image, then anchors all
verified fingerprints together.

```bash
# a whole folder (recurses by default)
python src/main.py --batch data/faces --workers 12 --out reports

# explicit files or globs, capped at 100 images
python src/main.py --batch "data/**/*.jpg" --limit 100

# discovery + hashing only, no chain
python src/main.py --batch data/faces --no-chain

# continue an interrupted run - already-processed images are skipped
python src/main.py --batch data/faces --resume
```

### Batch options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--batch PATH...` | - | files, directories and/or globs |
| `--workers N` | `BATCH_WORKERS=8` | parallel worker threads |
| `--pool-size N` | auto | concurrent face-model sessions (~300 MB each) |
| `--chunk-size N` | `BATCH_CHUNK_SIZE=25` | verified records anchored per transaction |
| `--out DIR` | `reports` | JSON + CSV report destination |
| `--checkpoint FILE` | `reports/checkpoint.jsonl` | resumable progress log |
| `--resume` | off | skip images already in the checkpoint |
| `--limit N` | all | process at most N images |
| `--no-recursive` | off | do not descend into sub-directories |

### How it stays fast

| Technique | Effect |
| --- | --- |
| Single model load + session pool (`DetectorPool`) | the ~300 MB ArcFace model is loaded once, not per image |
| Thread pool over network-bound stages | search, downloads and metadata fetches overlap |
| Batch-wide candidate cache | each candidate URL is downloaded and embedded exactly once, even when it appears for 40 different inputs |
| Input de-duplication by SHA-256 | byte-identical inputs are processed once and fanned back out into the report |
| Vectorised cosine similarity | one matrix product per candidate instead of a Python loop |
| `storeBatch` (one tx per 25 records) | amortises the 21k base gas cost; ~5-10x cheaper and far fewer confirmations to wait for |
| Bulk `getDataHashes` read-back | verifies an entire chunk in one RPC call |
| Merkle root per batch | a single 32-byte anchor proves membership of any record with a log2(N) proof |

### Batch outputs

* `reports/batch_results.json` - per-image results plus a summary block containing the
  merkle root, per-transaction anchoring info and the full read-back verification table.
* `reports/batch_results.csv` - spreadsheet-friendly one row per image.
* `reports/checkpoint.jsonl` - append-only progress log used by `--resume`.

### Batch verification

Every anchored fingerprint is read back from chain and compared with a local
recomputation, and the merkle root stored in the batch anchor is recomputed from
the local hashes as well. The run exits non-zero unless **all** records and the
root verify.

```
BLOCKCHAIN VERIFICATION
  Bulk read-back of every anchored fingerprint...
  ✅ record    0  local 4f2a91c0d3b7e5a8...  on-chain 4f2a91c0d3b7e5a8...
  ✅ record    1  local 9b18e77c22ad4f01...  on-chain 9b18e77c22ad4f01...
  ...                  62 more records verified the same way
  Verified             64/64
  Merkle root match    yes

  Result:
  ✅ ALL RECORDS VERIFIED - DATA INTEGRITY CONFIRMED
```

### Throughput notes

Wall-clock time is dominated by the reverse-image-search API, not by the face model.
With 12 workers a 100-image batch typically completes in a few minutes; the printed
summary reports actual throughput, average time per image and how many candidate
downloads the cache avoided. **Watch your provider's rate limit** - SerpApi's free
tier allows 100 searches/month, so a 100-image batch consumes it in one run. Lower
`--workers` if you see HTTP 429 responses.

## What exactly is hashed

- **`dataHash`** = `sha256(canonical_json(post_record))` where `post_record` contains
  `source_url`, `platform`, `title`, `description`, `image_url`, `site_name`,
  `published_time`, `retrieved_at`, `similarity`, `search_provider`.
  Canonical JSON = UTF-8, **sorted keys**, `(",", ":")` separators — so the identical record
  always yields the identical digest, which is what makes read-back comparison meaningful.
- **`imageHash`** = `sha256(raw bytes of the matched candidate image)`.

Note that `retrieved_at` is part of the record, so the fingerprint is bound to a specific
observation of the post at a specific time — that is the point of the anchor.

## Blockchain Verification

After the transaction is mined the CLI prints the transaction hash and an Etherscan link:

```
https://sepolia.etherscan.io/tx/0x<txhash>
```

Open it and look at the **Logs** tab: the `RecordStored` event shows the indexed `dataHash`,
the `sourceUrl`, the block timestamp and the submitter. The CLI then calls
`getRecord(recordId)` (or `getLatestByHash(dataHash)`) via `eth_call`, recomputes the SHA-256
locally from the discovered data, and compares the two strings. It reports success **only**
when they match byte for byte — a mined transaction alone is never treated as verification.

## Testing

```bash
pytest -q
```

29 tests covering SHA-256 / canonical JSON / `bytes32` conversion, cosine similarity, candidate
ranking and threshold selection, search-response parsing and error handling (rate limits, API
errors, missing keys), URL safety, and on-chain record parsing. Every external service is
mocked — no API key, no network and no real transaction is required.

## Limitations

- **Reverse-search coverage.** Providers index what they index. A face that exists only on a
  private profile, in a video, or behind a login simply will not be found. SerpApi's free tier
  is 100 searches/month; Bing F1 is rate-limited to a few requests per second.
- **Query image must be reachable.** Search APIs fetch the image by URL, so the CLI uploads it
  to a temporary public host unless you pass `--image-url`. Only run this with images you are
  authorised to publish.
- **Thumbnails, not originals.** Many providers return low-resolution, cropped or CDN-proxied
  thumbnails. Small or heavily compressed faces reduce embedding quality and similarity scores.
- **The threshold is empirical.** `0.45` is a reasonable starting point for ArcFace `buffalo_l`
  cosine similarity on reasonable-quality images. There is **no universally correct value**:
  it depends on the model pack, image resolution, pose, lighting, age gap and your tolerance
  for false positives versus false negatives. Tune it on your own data.
- **False positives and negatives are real.** Look-alikes, siblings, heavy makeup, masks,
  extreme angles and generated images all degrade accuracy. A match is evidence, not proof.
- **Social platforms restrict automation.** Instagram, Facebook and X frequently return 401/403
  or serve login walls to non-authenticated clients. The tool reports this and moves on; it does
  not attempt to bypass authentication, CAPTCHAs or access controls.
- **Testnet caveats.** Sepolia is a test network: faucets are rate-limited, blocks and state
  are not guaranteed permanent, and an anchor there has no legal or financial weight. It
  demonstrates the mechanism, not a production notarisation service.
- **On-chain anchoring proves integrity, not truth.** It proves *this exact record was recorded
  at this time by this address* — not that the underlying identification is correct.

## Privacy / Ethics

This project processes biometric data. Use it only:

- on images you own or are explicitly authorised to process;
- against **publicly accessible** material;
- for legitimate purposes such as verifying your own online presence, consented research,
  or the hackathon demonstration it was built for.

Do **not** use it to identify, track, profile or dox private individuals, to build a face
database without consent, or to circumvent authentication, CAPTCHAs, robots directives, rate
limits or any other access control. Biometric processing is regulated in many jurisdictions
(GDPR Art. 9, BIPA and others) — you are responsible for compliance. Face recognition is
probabilistic and can be wrong; never treat an output of this tool as a definitive identification.

## License

MIT.
