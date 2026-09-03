
# FaceChain Sentinel

FaceChain Sentinel is a face-identification and provenance-verification project. It accepts an image containing a face, searches public web sources for possible appearances of that face, independently checks each candidate with ArcFace embeddings, fingerprints the confirmed result with SHA-256, and optionally anchors the fingerprint on a blockchain.

The repository contains:

- A React/TanStack web interface for uploading an image and viewing progress and results.
- A FastAPI service that queues verification jobs for the web interface.
- A Python CLI and reusable pipeline under `face-chain-verifier/`.
- A Solidity contract that stores verification fingerprints and source metadata.

The detailed CLI documentation is in [face-chain-verifier/README.md](face-chain-verifier/README.md).

## How It Works

1. InsightFace detects faces and creates a 512-dimensional ArcFace embedding.
2. A reverse-image-search provider returns public candidate pages and images.
3. Candidate images are downloaded and re-embedded locally; cosine similarity is used to select a match above the configured threshold.
4. Public metadata for the matching page is normalized into a record.
5. The record and matched image bytes are hashed with SHA-256.
6. The metadata hash, image hash, source URL, timestamp, and submitter are stored on-chain when blockchain mode is enabled.
7. The pipeline reads the record back from chain and compares the on-chain hash with a local recomputation.

Search results are treated as leads, not proof: a result is only accepted after independent face matching.

## Requirements

- Python 3.12 recommended for the verifier.
- Node.js and npm for the web interface.
- A reverse-image-search API key: SerpApi Google Lens is the default; Bing Visual Search and TinEye are also supported.
- An Ethereum Sepolia RPC endpoint and test ETH for blockchain mode.

The InsightFace model pack (`buffalo_l`, about 300 MB) downloads automatically on first use.

## Setup

### Python verifier and API

From the repository root, create and activate the virtual environment:

```powershell
cd face-chain-verifier
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn python-multipart
Copy-Item .env.example .env
```

Edit `face-chain-verifier/.env` and provide at least `SEARCH_API_KEY`. For blockchain mode also provide `RPC_URL` and a throwaway Sepolia `PRIVATE_KEY`; `CONTRACT_ADDRESS` is filled in after deployment.

On macOS/Linux, use `python3 -m venv .venv`, `source .venv/bin/activate`, and `cp .env.example .env` instead.

### Web interface

From the repository root, install the JavaScript dependencies:

```powershell
npm install
```

The UI defaults to `http://localhost:8000` for the API. Set `VITE_API_URL` before starting Vite if the API is hosted elsewhere.

## Run It

Start the FastAPI service in one terminal:

```powershell
cd face-chain-verifier
.venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Start the web interface in a second terminal from the repository root:

```powershell
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`. Upload a JPG, PNG, or WEBP image up to 15 MB. The UI submits the image to `POST /api/verify` and polls `GET /api/verify/{job_id}` until the job completes.

The Python CLI can be run without the web interface:

```powershell
cd face-chain-verifier
.venv\Scripts\Activate.ps1
python src/main.py --image examples/input.webp
```

Use `--no-chain` to run discovery, matching, and hashing without a blockchain transaction:

```powershell
python src/main.py --image examples/input.webp --no-chain
```

Useful options include `--provider serpapi|bing|tineye`, `--threshold 0.5`, `--face-index 1`, `--max-candidates 30`, and `--save-results`. Batch processing is available with `--batch PATH...`; see the [verifier README](face-chain-verifier/README.md#batch-mode-50-100-images) for checkpointing, reports, and Merkle-root anchoring.

## Evidence Re-verification

Normal API verification runs the blockchain step and creates an evidence bundle for every successful match under `evidence/<evidence-id>/`:

```text
evidence/<evidence-id>/
├── metadata.json
└── matched_image.bin       # present when an image was available
```

The metadata hash is computed from the saved `record` object, and the image hash is computed from the saved matched-image bytes. Blockchain transaction details are added to `metadata.json` without changing the hashed `record` object. This keeps the original evidence independently reproducible while retaining its transaction reference.

Re-verify an evidence bundle from the CLI with:

```powershell
python src/main.py --reverify evidence/<evidence-id>
```

Re-verification loads `metadata.json` and `matched_image.bin` from disk, recomputes both SHA-256 fingerprints, reads the anchored record from Ethereum, and compares:

- The metadata hash.
- The image hash, when an image was saved.
- The source URL.

The web API exposes the same operation:

```http
POST /api/reverify
Content-Type: application/json

{"evidence_id":"<evidence-id>"}
```

The web interface includes a **Re-verify evidence** button and displays separate match or mismatch result cards for the metadata hash, image hash, and source URL.

### Blockchain read-back

After the transaction is mined, the CLI reads the blockchain record back and compares the saved evidence with the on-chain values:

```text
Local Metadata Hash       -> On-chain Data Hash
Local Image Hash          -> On-chain Image Hash
Local Source URL          -> On-chain Source URL
```

A successful read-back reports:

```text
ON-CHAIN DATA HASH     MATCH
ON-CHAIN IMAGE HASH    MATCH
SOURCE URL             MATCH

FINAL RESULT
BLOCKCHAIN EVIDENCE VERIFIED
```

### Independent re-verification

Saved evidence can be checked later without repeating the face search. From the `face-chain-verifier` directory, run:

```powershell
python src/main.py --reverify "evidence\<EVIDENCE_ID>"
```

For example:

```powershell
python src/main.py --reverify "evidence\8f3a7c91e2b4"
```

The CLI loads `metadata.json` and `matched_image.bin`, independently recalculates their SHA-256 fingerprints, reads the original Ethereum record, and reports `MATCH` or `MISMATCH` for the metadata hash, image hash, and source URL.

The expected successful exit code is `0`. In PowerShell, inspect it with:

```powershell
$LASTEXITCODE
```

The verification flow is:

```text
Saved Evidence -> Recalculate SHA-256 -> Read Ethereum Record -> Compare -> MATCH / MISMATCH
```

### Tamper demonstration

1. Run a normal verification and note the evidence bundle ID.
2. Edit the `record` object in `evidence/<evidence-id>/metadata.json`.
3. Click **Re-verify evidence** or run the CLI command above.
4. The metadata hash and source URL should report `MISMATCH`, while the blockchain record remains unchanged.

Expected result:

```text
ON-CHAIN DATA HASH     MISMATCH
ON-CHAIN IMAGE HASH    MATCH
SOURCE URL             MISMATCH

FINAL RESULT
BLOCKCHAIN EVIDENCE TAMPERED
```

Evidence tampering returns exit code `6`.

For an image tamper test, modify `evidence/<evidence-id>/matched_image.bin`. The image hash should then report `MISMATCH`.

Expected image-tamper result:

```text
ON-CHAIN DATA HASH     MATCH
ON-CHAIN IMAGE HASH    MISMATCH
SOURCE URL             MATCH

FINAL RESULT
BLOCKCHAIN EVIDENCE TAMPERED
```

### Verification example

Run a normal verification:

```powershell
python src/main.py --image "C:\images\test.jpg"
```

After it completes, list the generated bundles and independently verify one:

```powershell
Get-ChildItem .\evidence\
python src/main.py --reverify "evidence\8f3a7c91e2b4"
```

Expected result:

```text
BLOCKCHAIN EVIDENCE VERIFIED
```

### Platform image URL fallback

The original image link may not always be retrievable from platforms such as **X, LinkedIn, and Instagram** because of privacy settings, authentication requirements, anti-bot protection, and restricted media access. In these cases, the system may receive only a proxy or cached image URL from the reverse-search provider instead of the original platform CDN link.

This does not mean verification failed. The system can still:

- Verify the downloaded image.
- Calculate and store its image hash.
- Preserve the original source post or profile URL.
- Anchor the evidence hash on the blockchain.
- Re-verify the locally stored evidence later.

For transparency, result metadata distinguishes between these URLs and the image source:

```text
source_url: Original X/LinkedIn/Instagram post or profile
verified_image_url: Image actually used for face verification
original_image_url: Unavailable due to platform restrictions
image_source: Reverse-search provider fallback
```

## Blockchain Used

The project uses the **Ethereum Sepolia testnet**:

```text
Chain ID: 11155111
Contract: 0x7801548f318dB4517103035A9c129F2d4EdF2dCD
```

The contract is implemented in Solidity `0.8.20` at [face-chain-verifier/contracts/FaceVerification.sol](face-chain-verifier/contracts/FaceVerification.sol).

Only compact verification data is stored on-chain:

- `dataHash`: SHA-256 of canonical JSON metadata.
- `imageHash`: SHA-256 of the matched image bytes.
- `sourceUrl`, block timestamp, and submitting address.

The raw image and face embedding are not stored on-chain. Deploy a new instance with:

```powershell
cd face-chain-verifier
.venv\Scripts\Activate.ps1
python scripts/deploy.py
```

Copy the printed contract address into `.env` as `CONTRACT_ADDRESS`. Sepolia test ETH is required to submit transactions; `--no-chain` avoids this requirement.

## Testing

The Python tests are mocked and do not require API credentials, network access, or a real blockchain transaction:

```powershell
cd face-chain-verifier
.venv\Scripts\Activate.ps1
pytest -q
```

The web project also provides `npm run lint` and `npm run build`.

## CLI Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Evidence verified successfully. |
| `6` | Evidence tampering detected during re-verification. |
| Other non-zero codes | Verification, configuration, search, or blockchain error. |

## Known Limitations

- Reverse search only finds material indexed by the selected provider. Private profiles, videos, login-protected pages, and unindexed content are out of scope.
- The query image must be publicly reachable by the search provider. By default, the CLI uses the configured temporary upload endpoint; use `--image-url` when you already have an authorized public URL.
- Candidate images may be thumbnails, cropped, compressed, or unavailable. These conditions reduce matching accuracy.
- The default ArcFace cosine threshold (`0.45`) is empirical, not a guarantee. Lighting, pose, resolution, age, makeup, masks, look-alikes, and generated images can produce false positives or false negatives.
- Social platforms often return `401`, `403`, or login pages. The pipeline skips inaccessible material and does not bypass authentication, CAPTCHAs, robots directives, rate limits, or other access controls.
- Provider quotas apply. For example, SerpApi's free tier allows 100 searches per month, so large batches can consume the allowance quickly.
- Sepolia is a testnet. Testnet state, faucets, and confirmations are not guaranteed permanently, and a testnet anchor has no legal or financial weight.
- Blockchain anchoring proves that a particular fingerprint was recorded by an address at a particular time; it does not prove that the face identification or source metadata is true.
- The API keeps jobs in process memory and launches the CLI locally. It is intended for local demonstration and does not provide durable job storage, authentication, or production worker isolation.

## Privacy and Responsible Use

This project processes biometric data. Use it only with images you own or are authorized to process, against publicly accessible material, and for legitimate purposes. Do not use it to identify, track, profile, or dox private individuals or to build an unconsented face database. Face recognition is probabilistic; never treat its output as definitive identification. Check applicable privacy and biometric-data laws before use.

## License

MIT
