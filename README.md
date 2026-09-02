
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

## Blockchain Used

The project uses the **Ethereum Sepolia testnet** (`chain ID 11155111`) and a Solidity `0.8.20` contract at [face-chain-verifier/contracts/FaceVerification.sol](face-chain-verifier/contracts/FaceVerification.sol).

Only compact verification data is stored on-chain:

- `dataHash`: SHA-256 of canonical JSON metadata.
- `imageHash`: SHA-256 of the matched image bytes.
- `sourceUrl`, block timestamp, and submitting address.

Images, face embeddings, API credentials, and full page contents are not stored on-chain. Deploy the contract with:

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
