"""Application configuration with system capacity guardrails."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. All byte/size limits enforce system capacity constraints."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="INGEST_", extra="ignore")

    # --- Service identity ---
    service_name: str = "multimodal-ingestion-engine"
    environment: str = "development"

    # --- Capacity guardrails (sized for an 8GB host; clamped at runtime by app/resources.py) ---
    max_file_size_bytes: int = 100 * 1024 * 1024  # strict 100MB per-file ceiling
    # Deployment invariant: the sum of every container cap below must stay within
    # this envelope so macOS + the Docker daemon keep ~1.75GB of the 8GB host.
    total_container_budget_mb: int = 6400  # 6.25 GiB stack envelope
    # Per-Celery-child memory limit; the worker's total footprint is
    # concurrency * worker_child_memory_limit_mb and must stay within
    # worker_memory_budget_mb so the OS, Redis and the API keep headroom.
    # The ingestion worker is capped at 2g (2 children x 1GB), so the budget is
    # 2048MB - the old 4096MB default exceeded the container cap and could never
    # be honoured. Values are additionally clamped to the detected host RAM.
    worker_child_memory_limit_mb: int = 1024
    worker_memory_budget_mb: int = 2048
    worker_max_tasks_per_child: int = 20  # proactive child recycling (ingestion)
    media_max_tasks_per_child: int = 10   # media children hold OCR/ASR models: recycle sooner
    stream_chunk_size_bytes: int = 1024 * 1024  # 1MB stream chunks for uploads/IO
    min_free_disk_mb: int = 2048  # refuse new uploads below 2GB free disk

    # --- Per-queue concurrency (single source of truth; docker-compose consumes these) ---
    # Worst-case footprint per queue = concurrency * that queue's child memory limit.
    ingestion_concurrency: int = 2
    index_concurrency: int = 1     # 1 x 768MB child inside the 1g cap
    reasoning_concurrency: int = 1  # cloud I/O bound; 1 child inside the 512m cap

    # --- Per-queue child memory limits (each worker container overrides its own via env) ---
    media_child_memory_limit_mb: int = 384  # whisper tiny int8 + 512px frame buffers
    index_child_memory_limit_mb: int = 768  # MiniLM ~90MB + polars batch buffers
    reasoning_child_memory_limit_mb: int = 512  # pinned to the 512m cap: cloud I/O bound

    # --- Container caps: used at startup to validate the plan against the budget ---
    ingestion_container_cap_mb: int = 2048   # worker: 2 children x 1GB
    media_container_cap_mb: int = 1024       # media-worker: 2 children x 384MB + parent
    index_container_cap_mb: int = 1024       # index-worker: 1 child x 768MB + parent
    reasoning_container_cap_mb: int = 512    # reasoning-worker: cloud I/O bound
    api_container_cap_mb: int = 512          # FastAPI streams uploads to disk
    qdrant_container_cap_mb: int = 1024      # dedicated vector DB
    redis_container_cap_mb: int = 256        # broker + result backend

    # --- Storage ---
    upload_dir: Path = Path("/tmp/multimodal-ingestion/uploads")
    result_dir: Path = Path("/tmp/multimodal-ingestion/results")

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    celery_task_queue: str = "ingestion"
    celery_task_soft_time_limit_s: int = 900
    celery_task_time_limit_s: int = 1200

    # --- Chunking ---
    chunk_target_tokens: int = 512
    chunk_overlap_fraction: float = 0.10
    chars_per_token: float = 4.0  # heuristic for token <-> char conversion

    # --- Phase 2: vision / audio processing node ---
    media_task_queue: str = "media"
    max_image_edge_px: int = 512            # bilinear downsample guardrail
    audio_target_sample_rate_hz: int = 16000
    audio_window_seconds: float = 30.0      # strict window blocks before transcription
    video_sample_fps: float = 1.0           # strict 1 frame per second
    whisper_model_size: str = "tiny"        # int8 quantized; keep footprint low on 8GB host
    ocr_confidence_threshold: float = 0.35  # gates OCR text attachment
    media_task_timeout_s: int = 300         # per-item processing timeout
    ffmpeg_timeout_s: int = 120             # subprocess timeout for MP3/other containers
    media_concurrency: int = 2              # media queue children (capacity plan input)

    # --- Phase 3: tabular / telemetry / vector indexer ---
    index_task_queue: str = "index"         # dedicated Phase 3 Celery queue
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384                # MiniLM output size
    # "auto" picks the best accelerator the host can actually afford: CUDA first
    # (dedicated VRAM), then MPS only when host RAM clears the threshold below
    # (unified memory shares the pool the container budget is carved from).
    # Force a device with INGEST_EMBEDDING_DEVICE=cpu|mps|cuda; unavailable
    # requests fall back to CPU with a warning. Torch is required for mps/cuda.
    embedding_device: str = "auto"
    accelerator_min_host_memory_mb: int = 16384  # MPS allowed only on 16GB+ hosts
    upsert_batch_size: int = 64             # vectors per batch upsert (per spec)
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout_s: float = 10.0
    qdrant_max_retries: int = 5             # retry attempts w/ exponential backoff
    qdrant_retry_base_delay_s: float = 0.5
    max_embed_batch_rows: int = 256         # bound RAM: rows per embedding call
    spool_dir_name: str = "spool"           # offline fallback when Qdrant unreachable

    # --- Phase 4: reasoning core & output optimization ---
    reasoning_task_queue: str = "reasoning"  # dedicated Phase 4 Celery queue
    reasoning_backend: str = "auto"          # auto | local | openai | anthropic
    local_token_budget: int = 8000           # safe local context threshold
    cloud_token_budget: int = 32000          # cloud context threshold
    max_context_bytes: int = 2 * 1024 * 1024  # hard cap: never send >2MB context
    local_llm_enabled: bool = False          # GGUF local backbone opt-in
    local_llm_model_path: str = ""           # path to quantized .gguf model
    openai_api_key: str = ""                 # env-injected; empty = disabled
    anthropic_api_key: str = ""              # env-injected; empty = disabled
    reasoning_cloud_model: str = "gpt-4o"
    breaker_failure_threshold: int = 3       # failures before OPEN
    breaker_cooldown_s: float = 30.0         # OPEN -> HALF_OPEN cooldown
    reasoning_request_timeout_s: float = 60.0

    # --- PDF layout heuristics ---
    header_footer_margin_fraction: float = 0.08  # top/bottom 8% of page height


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
