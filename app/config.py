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
    # Per-Celery-child memory limit; the worker's total footprint is
    # concurrency * worker_child_memory_limit_mb and must stay within
    # worker_memory_budget_mb (~50% of host RAM) so the OS, Redis and the
    # API keep headroom.
    worker_child_memory_limit_mb: int = 1024
    worker_memory_budget_mb: int = 4096
    worker_max_tasks_per_child: int = 20  # proactive child recycling
    stream_chunk_size_bytes: int = 1024 * 1024  # 1MB stream chunks for uploads/IO
    min_free_disk_mb: int = 2048  # refuse new uploads below 2GB free disk

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
    media_concurrency: int = 2              # keep worker footprint modest on 8GB host

    # --- Phase 3: tabular / telemetry / vector indexer ---
    index_task_queue: str = "index"         # dedicated Phase 3 Celery queue
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384                # MiniLM output size
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
