import { createFileRoute } from "@tanstack/react-router";
import {
  Upload,
  ShieldCheck,
  Search,
  Link2,
  Fingerprint,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ExternalLink,
  X,
  Database,
  ScanFace,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

export const Route = createFileRoute("/")({
  component: Index,
});

const API_BASE =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

const MAX_FILE_SIZE = 15 * 1024 * 1024;
const POLL_INTERVAL = 1000;
const MAX_POLL_TIME = 180000;

type Candidate = {
  rank: number;
  platform: string;
  similarity: number;
  faces: number;
  source: string;
};

type VerifyResult = {
  face?: {
    detected?: boolean;
    count?: number;
    confidence?: number;
    embedding_dimensions?: number;
  };

  match?: {
    found?: boolean;
    similarity?: number;
    threshold?: number;
    platform?: string;
    source_url?: string;
    verified_image_url?: string;
    image_source?: string;
  };

  candidates?: Candidate[];

  record?: {
    source_url?: string;
    platform?: string;
    title?: string;
    description?: string;
    image_url?: string;
    site_name?: string;
    published_time?: string;
    author?: string;
    canonical_url?: string;
    retrieved_at?: string;
    is_exact_match?: boolean;
    search_type?: string;
    similarity?: number;
    match_threshold?: number;
    search_provider?: string;
    verified_image_url?: string;
    verified_image_source?: string;
  };

  fingerprints?: {
    metadata_sha256?: string;
    image_sha256?: string;
  };

  blockchain?: {
    anchored?: boolean;
    verified?: boolean;
    transaction?: string | null;
    block?: number | null;
    explorer?: string | null;
  };

  status?: string;
};

type VerifyResponse = {
  success?: boolean;
  job_id?: string;
  status?: string;
  stage?: string;
  progress?: number;
  attempts?: number;
  filename?: string;
  message?: string;
  result?: VerifyResult | null;
  output?: string;
  warnings?: string;
  error?: string;
  detail?: string;
};

function formatSimilarity(value?: number) {
  if (typeof value !== "number") return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function shortenHash(value?: string | null) {
  if (!value) return "—";
  if (value.length <= 24) return value;
  return `${value.slice(0, 14)}…${value.slice(-10)}`;
}

function getStageLabel(response: VerifyResponse | null) {
  if (!response) return "Waiting for verification";

  if (response.stage) return response.stage;

  switch (response.status) {
    case "queued":
      return "Queued";
    case "processing":
      return "Processing verification";
    case "completed":
      return "Verification complete";
    case "error":
      return "Verification failed";
    case "timeout":
      return "Pipeline timeout";
    default:
      return "Processing";
  }
}

function getProgress(response: VerifyResponse | null) {
  if (!response) return 0;

  if (typeof response.progress === "number") {
    return Math.max(0, Math.min(100, response.progress));
  }

  switch (response.status) {
    case "queued":
      return 5;
    case "processing":
      return 25;
    case "completed":
      return 100;
    default:
      return 0;
  }
}

function Index() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [isVerifying, setIsVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [response, setResponse] = useState<VerifyResponse | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const result = response?.result;

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;

      if (pollTimerRef.current !== null) {
        window.clearTimeout(pollTimerRef.current);
      }

      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const selectFile = (file: File | null) => {
    setError(null);
    setResponse(null);
    setJobId(null);

    if (!file) {
      setSelectedFile(null);
      setPreviewUrl(null);
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    const allowedTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
      "image/webp",
    ];

    if (!allowedTypes.includes(file.type)) {
      setError("Only JPG, PNG and WEBP images are supported.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError("Image is too large. Maximum size is 15 MB.");
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    selectFile(event.target.files?.[0] ?? null);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    selectFile(event.dataTransfer.files?.[0] ?? null);
  };

  const clearFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setPreviewUrl(null);
    setResponse(null);
    setError(null);
    setJobId(null);
    setIsVerifying(false);

    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const pollJob = async (id: string) => {
    const startedAt = Date.now();

    const poll = async (): Promise<void> => {
      if (!mountedRef.current) return;

      if (Date.now() - startedAt > MAX_POLL_TIME) {
        if (mountedRef.current) {
          setIsVerifying(false);
          setError(
            "Verification timed out after 3 minutes. The backend may still be processing the job.",
          );
        }

        return;
      }

      try {
        const apiResponse = await fetch(
          `${API_BASE}/api/verify/${encodeURIComponent(id)}`,
          {
            method: "GET",
            headers: {
              Accept: "application/json",
            },
            cache: "no-store",
          },
        );

        const contentType =
          apiResponse.headers.get("content-type") || "";

        let data: VerifyResponse;

        if (contentType.includes("application/json")) {
          data = await apiResponse.json();
        } else {
          const text = await apiResponse.text();

          throw new Error(
            text ||
              `Status request failed with HTTP ${apiResponse.status}.`,
          );
        }

        if (!apiResponse.ok) {
          throw new Error(
            data.detail ||
              data.error ||
              `Status request failed with HTTP ${apiResponse.status}.`,
          );
        }

        if (!mountedRef.current) return;

        setResponse(data);

        /*
         * The important part:
         *
         * queued / processing
         *        ↓
         *      poll
         *
         * completed / error / timeout
         *        ↓
         *      stop
         */
        if (
          data.status === "completed" ||
          data.status === "error" ||
          data.status === "timeout"
        ) {
          setIsVerifying(false);
          return;
        }

        pollTimerRef.current = window.setTimeout(
          () => {
            void poll();
          },
          POLL_INTERVAL,
        );
      } catch (err) {
        if (!mountedRef.current) return;

        setIsVerifying(false);

        if (err instanceof TypeError) {
          setError(
            `Could not connect to FaceChain Sentinel API at ${API_BASE}.`,
          );
        } else {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to retrieve verification status.",
          );
        }
      }
    };

    await poll();
  };

  const startVerification = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    if (isVerifying) return;

    setIsVerifying(true);
    setError(null);
    setResponse(null);
    setJobId(null);

    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    try {
      const formData = new FormData();

      formData.append(
        "file",
        selectedFile,
        selectedFile.name,
      );

      /*
       * Step 1:
       * Start the backend job.
       *
       * POST /api/verify returns immediately with:
       * {
       *   success: true,
       *   job_id: "...",
       *   status: "queued"
       * }
       */
      const apiResponse = await fetch(
        `${API_BASE}/api/verify`,
        {
          method: "POST",
          body: formData,
          headers: {
            Accept: "application/json",
          },
        },
      );

      const contentType =
        apiResponse.headers.get("content-type") || "";

      let data: VerifyResponse;

      if (contentType.includes("application/json")) {
        data = await apiResponse.json();
      } else {
        const text = await apiResponse.text();

        throw new Error(
          text ||
            `Verification request failed with HTTP ${apiResponse.status}.`,
        );
      }

      if (!apiResponse.ok) {
        throw new Error(
          data.detail ||
            data.error ||
            `Verification failed with HTTP ${apiResponse.status}.`,
        );
      }

      if (!data.job_id) {
        throw new Error(
          "The API started the request but did not return a job ID.",
        );
      }

      if (data.success === false) {
        throw new Error(
          data.error ||
            data.detail ||
            "The verification pipeline could not be started.",
        );
      }

      if (!mountedRef.current) return;

      setJobId(data.job_id);
      setResponse(data);

      /*
       * Step 2:
       * Poll the job until it completes.
       */
      await pollJob(data.job_id);
    } catch (err) {
      if (!mountedRef.current) return;

      setIsVerifying(false);

      if (err instanceof TypeError) {
        setError(
          `Could not connect to FaceChain Sentinel API at ${API_BASE}. Make sure the FastAPI server is running.`,
        );
      } else {
        setError(
          err instanceof Error
            ? err.message
            : "An unexpected verification error occurred.",
        );
      }
    }
  };

  const progress = getProgress(response);
  const stage = getStageLabel(response);

  return (
    <main className="min-h-screen bg-[#fcfbf8] text-slate-950">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-950 text-white">
              <ShieldCheck className="h-5 w-5" />
            </div>

            <div>
              <h1 className="text-lg font-bold tracking-tight">
                FaceChain Sentinel
              </h1>

              <p className="text-xs text-slate-500">
                Face identification & verification
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 sm:flex">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            API Ready
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-10">
        {/* Hero */}
        <section className="mb-10">
          <div className="max-w-3xl">
            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
              Verification Pipeline
            </p>

            <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
              Verify a face against
              <span className="block text-slate-500">
                public-source matches.
              </span>
            </h2>

            <p className="mt-5 max-w-2xl text-base leading-7 text-slate-600">
              Upload an image to run face detection, reverse-image
              discovery, similarity matching, metadata fingerprinting,
              and verification.
            </p>
          </div>
        </section>

        {/* Pipeline */}
        <section className="mb-8 grid gap-3 sm:grid-cols-4">
          {[
            {
              icon: Upload,
              title: "Input",
              text: "Upload image",
            },
            {
              icon: Search,
              title: "Discovery",
              text: "Search public sources",
            },
            {
              icon: Fingerprint,
              title: "Fingerprint",
              text: "Generate SHA-256",
            },
            {
              icon: Link2,
              title: "Verification",
              text: "Integrity check",
            },
          ].map((step, index) => {
            const Icon = step.icon;

            return (
              <div
                key={step.title}
                className="relative rounded-2xl border border-slate-200 bg-white p-5"
              >
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100">
                  <Icon className="h-5 w-5 text-slate-700" />
                </div>

                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  0{index + 1}
                </p>

                <h3 className="mt-1 font-semibold">
                  {step.title}
                </h3>

                <p className="mt-1 text-sm text-slate-500">
                  {step.text}
                </p>
              </div>
            );
          })}
        </section>

        {/* Main */}
        <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
          {/* Upload */}
          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5">
              <h3 className="text-lg font-semibold">
                Input image
              </h3>

              <p className="mt-1 text-sm text-slate-500">
                JPG, PNG, WEBP up to 15 MB.
              </p>
            </div>

            {!selectedFile ? (
              <div
                onDragOver={(event) => {
                  event.preventDefault();
                }}
                onDrop={handleDrop}
                onClick={() =>
                  fileInputRef.current?.click()
                }
                className="flex min-h-[330px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 px-6 text-center transition hover:border-slate-500 hover:bg-slate-100"
              >
                <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-white shadow-sm">
                  <Upload className="h-7 w-7 text-slate-600" />
                </div>

                <p className="font-semibold">
                  Drop an image here
                </p>

                <p className="mt-2 text-sm text-slate-500">
                  or click to browse
                </p>

                <p className="mt-2 text-xs text-slate-400">
                  JPG, PNG or WEBP
                </p>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </div>
            ) : (
              <div className="overflow-hidden rounded-2xl border border-slate-200">
                <div className="relative bg-slate-100">
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Selected verification input"
                      className="mx-auto max-h-[430px] w-full object-contain"
                    />
                  )}

                  <button
                    type="button"
                    onClick={clearFile}
                    disabled={isVerifying}
                    className="absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-full bg-white/95 text-slate-700 shadow-md hover:bg-white disabled:opacity-50"
                    aria-label="Remove image"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>

                <div className="flex items-center justify-between gap-4 p-4">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold">
                      {selectedFile.name}
                    </p>

                    <p className="mt-1 text-xs text-slate-500">
                      {(selectedFile.size / 1024 / 1024).toFixed(
                        2,
                      )}{" "}
                      MB
                    </p>
                  </div>

                  <button
                    type="button"
                    disabled={isVerifying}
                    onClick={() =>
                      fileInputRef.current?.click()
                    }
                    className="shrink-0 rounded-lg border border-slate-300 px-3 py-2 text-xs font-medium hover:bg-slate-50 disabled:opacity-50"
                  >
                    Change
                  </button>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </div>
              </div>
            )}

            {error && (
              <div className="mt-4 flex gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />

                <div>
                  <p className="font-semibold">
                    Verification error
                  </p>

                  <p className="mt-1 break-words">
                    {error}
                  </p>
                </div>
              </div>
            )}

            {/* Live progress */}
            {isVerifying && (
              <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <Loader2 className="h-5 w-5 animate-spin text-slate-700" />

                    <div>
                      <p className="text-sm font-semibold">
                        {stage}
                      </p>

                      <p className="mt-0.5 text-xs text-slate-500">
                        Verification job is running
                      </p>
                    </div>
                  </div>

                  <span className="text-sm font-bold">
                    {progress}%
                  </span>
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full bg-slate-950 transition-all duration-500"
                    style={{
                      width: `${Math.max(
                        3,
                        progress,
                      )}%`,
                    }}
                  />
                </div>

                {jobId && (
                  <p className="mt-3 truncate font-mono text-[10px] text-slate-400">
                    Job: {jobId}
                  </p>
                )}
              </div>
            )}

            <button
              type="button"
              disabled={!selectedFile || isVerifying}
              onClick={startVerification}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-slate-950 px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isVerifying ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Verifying…
                </>
              ) : (
                <>
                  <ShieldCheck className="h-4 w-4" />
                  Verify Image
                </>
              )}
            </button>

            <p className="mt-3 text-center text-xs text-slate-400">
              API endpoint: {API_BASE}/api/verify
            </p>
          </div>

          {/* Results */}
          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5">
              <h3 className="text-lg font-semibold">
                Verification result
              </h3>

              <p className="mt-1 text-sm text-slate-500">
                Results returned directly from the Sentinel API.
              </p>
            </div>

            {!response && !isVerifying && (
              <div className="flex min-h-[330px] items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-8 text-center">
                <div>
                  <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-white">
                    <Search className="h-6 w-6 text-slate-400" />
                  </div>

                  <p className="font-medium text-slate-700">
                    No verification yet
                  </p>

                  <p className="mt-2 text-sm text-slate-500">
                    Upload an image and click Verify Image.
                  </p>
                </div>
              </div>
            )}

            {/* Queued / processing */}
            {response &&
              isVerifying &&
              !result && (
                <div className="space-y-5">
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                    <div className="flex items-center gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-white">
                        <Loader2 className="h-6 w-6 animate-spin text-slate-700" />
                      </div>

                      <div>
                        <p className="font-semibold">
                          {stage}
                        </p>

                        <p className="mt-1 text-sm text-slate-500">
                          The API job has started. Waiting for the
                          verification pipeline to finish.
                        </p>
                      </div>
                    </div>

                    <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full rounded-full bg-slate-950 transition-all duration-500"
                        style={{
                          width: `${Math.max(
                            3,
                            progress,
                          )}%`,
                        }}
                      />
                    </div>

                    <div className="mt-3 flex justify-between text-xs text-slate-500">
                      <span>{stage}</span>
                      <span>{progress}%</span>
                    </div>
                  </div>

                  {jobId && (
                    <div className="rounded-xl bg-slate-50 px-4 py-3 text-xs text-slate-500">
                      Job ID:{" "}
                      <span className="font-mono text-slate-700">
                        {jobId}
                      </span>
                    </div>
                  )}
                </div>
              )}

            {/* Final result */}
            {response &&
              !isVerifying &&
              result && (
                <div className="space-y-5">
                  {/* Status */}
                  <div
                    className={`flex items-center gap-3 rounded-2xl border p-4 ${
                      result.match?.found
                        ? "border-emerald-200 bg-emerald-50"
                        : "border-amber-200 bg-amber-50"
                    }`}
                  >
                    {result.match?.found ? (
                      <CheckCircle2 className="h-6 w-6 text-emerald-600" />
                    ) : (
                      <AlertCircle className="h-6 w-6 text-amber-600" />
                    )}

                    <div>
                      <p className="font-semibold">
                        {result.match?.found
                          ? "Match found"
                          : "No match found"}
                      </p>

                      <p className="mt-0.5 text-sm text-slate-600">
                        {result.match?.found
                          ? `${
                              result.match.platform ??
                              "Source"
                            } · ${formatSimilarity(
                              result.match.similarity,
                            )} similarity`
                          : "No matching public-source result exceeded the configured threshold."}
                      </p>
                    </div>
                  </div>

                  {/* Face detection */}
                  <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Face detection
                    </p>

                    <div className="grid grid-cols-3 gap-3">
                      <Metric
                        label="Detected"
                        value={
                          result.face?.detected
                            ? "Yes"
                            : "No"
                        }
                      />

                      <Metric
                        label="Confidence"
                        value={
                          typeof result.face?.confidence ===
                          "number"
                            ? `${(
                                result.face.confidence * 100
                              ).toFixed(1)}%`
                            : "—"
                        }
                      />

                      <Metric
                        label="Embedding"
                        value={
                          result.face?.embedding_dimensions
                            ? `${result.face.embedding_dimensions}D`
                            : "—"
                        }
                      />
                    </div>
                  </div>

                  {/* Best match */}
                  {result.match && (
                    <div>
                      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Best match
                      </p>

                      <div className="rounded-2xl border border-slate-200 p-4">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="font-semibold">
                              {result.match.platform ??
                                "Unknown platform"}
                            </p>

                            <p className="mt-1 text-sm text-slate-500">
                              Similarity{" "}
                              <span className="font-semibold text-slate-800">
                                {formatSimilarity(
                                  result.match.similarity,
                                )}
                              </span>
                            </p>

                            {result.match.threshold !==
                              undefined && (
                              <p className="mt-1 text-xs text-slate-400">
                                Threshold:{" "}
                                {formatSimilarity(
                                  result.match.threshold,
                                )}
                              </p>
                            )}
                          </div>

                          {result.match.source_url && (
                            <a
                              href={result.match.source_url}
                              target="_blank"
                              rel="noreferrer"
                              className="flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-950"
                            >
                              Open source
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          )}
                        </div>

                        {result.record?.title && (
                          <p className="mt-4 text-sm font-medium leading-6">
                            {result.record.title}
                          </p>
                        )}

                        {result.record?.description && (
                          <p className="mt-2 text-sm leading-6 text-slate-500">
                            {result.record.description}
                          </p>
                        )}

                        {result.record?.canonical_url && (
                          <a
                            href={
                              result.record.canonical_url
                            }
                            target="_blank"
                            rel="noreferrer"
                            className="mt-3 block break-all text-xs text-slate-500 underline hover:text-slate-950"
                          >
                            {result.record.canonical_url}
                          </a>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Candidates */}
                  {result.candidates &&
                    result.candidates.length > 0 && (
                      <div>
                        <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                          Candidates
                        </p>

                        <div className="overflow-hidden rounded-2xl border border-slate-200">
                          {result.candidates.map(
                            (candidate) => (
                              <div
                                key={`${candidate.rank}-${candidate.platform}`}
                                className="flex items-center justify-between border-b border-slate-100 px-4 py-3 last:border-b-0"
                              >
                                <div className="flex items-center gap-3">
                                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold">
                                    #{candidate.rank}
                                  </span>

                                  <div>
                                    <p className="text-sm font-medium">
                                      {candidate.platform}
                                    </p>

                                    <p className="text-xs text-slate-500">
                                      {candidate.faces} face
                                      {candidate.faces === 1
                                        ? ""
                                        : "s"}{" "}
                                      · {candidate.source}
                                    </p>
                                  </div>
                                </div>

                                <p className="text-sm font-semibold">
                                  {formatSimilarity(
                                    candidate.similarity,
                                  )}
                                </p>
                              </div>
                            ),
                          )}
                        </div>
                      </div>
                    )}

                  {/* Evidence metadata */}
                  {result.record && (
                    <div>
                      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Evidence
                      </p>

                      <div className="grid gap-3 sm:grid-cols-2">
                        <InfoCard
                          label="Source"
                          value={
                            result.record.site_name ??
                            result.record.platform ??
                            "—"
                          }
                        />

                        <InfoCard
                          label="Search provider"
                          value={
                            result.record.search_provider ??
                            "—"
                          }
                        />

                        <InfoCard
                          label="Exact match"
                          value={
                            result.record.is_exact_match
                              ? "Yes"
                              : "No"
                          }
                        />

                        <InfoCard
                          label="Retrieved"
                          value={
                            result.record.retrieved_at ??
                            "—"
                          }
                        />
                      </div>
                    </div>
                  )}

                  {/* Fingerprints */}
                  {result.fingerprints && (
                    <div>
                      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Data fingerprints
                      </p>

                      <div className="space-y-3 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                        <HashRow
                          icon={
                            <Database className="h-4 w-4" />
                          }
                          label="Metadata SHA-256"
                          value={
                            result.fingerprints
                              .metadata_sha256
                          }
                        />

                        <HashRow
                          icon={
                            <Fingerprint className="h-4 w-4" />
                          }
                          label="Matched image SHA-256"
                          value={
                            result.fingerprints.image_sha256
                          }
                        />
                      </div>
                    </div>
                  )}

                  {/* Blockchain */}
                  {result.blockchain && (
                    <div>
                      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Blockchain verification
                      </p>

                      <div className="rounded-2xl border border-slate-200 p-4">
                        <div className="flex items-center gap-3">
                          {result.blockchain.verified ? (
                            <CheckCircle2 className="h-6 w-6 text-emerald-600" />
                          ) : (
                            <Link2 className="h-6 w-6 text-slate-400" />
                          )}

                          <div className="min-w-0">
                            <p className="text-sm font-semibold">
                              {result.blockchain.verified
                                ? "Blockchain verified"
                                : result.blockchain.anchored
                                  ? "Anchored on-chain"
                                  : "Blockchain anchoring skipped"}
                            </p>

                            <p className="mt-1 text-xs text-slate-500">
                              {result.blockchain.verified
                                ? "The verification record was independently verified."
                                : result.blockchain.anchored
                                  ? "A blockchain transaction was created for this record."
                                  : "API integration currently runs with --no-chain."}
                            </p>
                          </div>
                        </div>

                        {result.blockchain.transaction && (
                          <div className="mt-4 rounded-xl bg-slate-50 p-3">
                            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                              Transaction
                            </p>

                            <p className="mt-1 break-all font-mono text-xs text-slate-700">
                              {
                                result.blockchain
                                  .transaction
                              }
                            </p>
                          </div>
                        )}

                        {result.blockchain.block !==
                          null &&
                          result.blockchain.block !==
                            undefined && (
                            <p className="mt-3 text-xs text-slate-500">
                              Block:{" "}
                              <span className="font-semibold text-slate-700">
                                {result.blockchain.block}
                              </span>
                            </p>
                          )}

                        {result.blockchain.explorer && (
                          <a
                            href={
                              result.blockchain.explorer
                            }
                            target="_blank"
                            rel="noreferrer"
                            className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-950"
                          >
                            View blockchain explorer
                            <ExternalLink className="h-3 w-3" />
                          </a>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Job information */}
                  <div className="rounded-xl bg-slate-50 px-4 py-3 text-xs text-slate-500">
                    <div className="flex items-center gap-2">
                      <ScanFace className="h-4 w-4" />

                      <span>
                        Verification completed
                      </span>
                    </div>

                    {response.job_id && (
                      <p className="mt-2 break-all font-mono text-[10px] text-slate-400">
                        Job ID: {response.job_id}
                      </p>
                    )}

                    {response.attempts !==
                      undefined && (
                      <p className="mt-1 text-[10px] text-slate-400">
                        Attempts: {response.attempts}
                      </p>
                    )}
                  </div>
                </div>
              )}

            {/* Completed/error without structured result */}
            {response &&
              !isVerifying &&
              !result && (
                <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5">
                  <div className="flex gap-3">
                    <AlertCircle className="h-5 w-5 shrink-0 text-amber-600" />

                    <div className="min-w-0">
                      <p className="font-semibold text-amber-900">
                        {response.status === "error"
                          ? "Verification failed"
                          : "Verification completed without structured result"}
                      </p>

                      <p className="mt-1 text-sm text-amber-800">
                        {response.message ||
                          response.error ||
                          "The API did not return a structured verification result."}
                      </p>

                      {response.job_id && (
                        <p className="mt-3 break-all font-mono text-xs text-amber-700">
                          Job ID: {response.job_id}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-12 border-t border-slate-200 pt-6 text-center">
          <p className="text-xs text-slate-400">
            FaceChain Sentinel · Public-source verification
            system
          </p>
        </footer>
      </div>
    </main>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-sm font-semibold text-slate-800">
        {value}
      </p>
    </div>
  );
}

function InfoCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </p>

      <p className="mt-1 break-words text-sm font-medium text-slate-700">
        {value}
      </p>
    </div>
  );
}

function HashRow({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value?: string | null;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        {icon}
        {label}
      </div>

      <p
        title={value ?? undefined}
        className="mt-1 break-all font-mono text-xs text-slate-700"
      >
        {value ?? "—"}
      </p>

      {value && (
        <p className="mt-1 font-mono text-[10px] text-slate-400">
          {shortenHash(value)}
        </p>
      )}
    </div>
  );
}