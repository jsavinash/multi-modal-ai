# Phase 2 — Technical Documentation

Vision, Audio & OCR Processing Node of the Multimodal AI Ingestion Engine.
Inherits Phase 1's media fragments (`extracted_media_references`) and adds
standalone image/video/audio ingestion.

- Runtime: Python 3.10+ (container pins 3.12); **CPU-first** inference, optional Apple MPS
- Stack: OpenCV (headless), Pillow, numpy, faster-whisper (int8), EasyOCR (optional)
- Requirements traceability & prompt adaptations: [`docs/PHASE2_SPEC.md`](PHASE2_SPEC.md)

---

## 1. Architecture

```
   Phase 1 results (media fragments)      HTTP uploads (image/video/audio)
                 │                                      │
                 ▼                                      ▼
        ┌───────────────────────────┐      ┌────────────────────────────┐
        │ process_media (pipeline)  │◄─────│ /api/v1/media/ingest       │
        │  detect_media_format()    │      │ magic-byte sniff + storage │
        └──────┬─────────┬──────────┘      └─────────────┬──────────────┘
               │         │                               │ Celery queue "media"
       ┌───────▼──┐   ┌──▼───────────┐        ┌──────────▼───────────┐
       │ vision   │   │ audio        │        │ media-worker (x2)    │
       │ 512px DS │   │ 16kHz mono   │        │ process_media_task   │
       │ OCR route│   │ 30s windows  │        └──────────┬───────────┘
       └───────┬──┘   └──┬───────────┘                   │
               │         │ faster-whisper int8           │
               ▼         ▼                               ▼
        UnifiedMediaPayload[] ──► results/media_*.json  (Pydantic-validated)
```

**Flow:** upload → magic-byte media detection → streamed persistence (same 100 MB
ceiling / disk floor as Phase 1) → `media.process` task → modality pipeline →
`UnifiedMediaPayload[]` persisted to `results/`.

## 2. Module Reference

| Module | Lines | Responsibility |
|---|---|---|
| `app/media/pipeline.py` | 177 | Media magic-byte detection, modality routing, timeout enforcement, payload assembly |
| `app/media/vision.py` | 110 | 512px bilinear downsample; `OcrRouter` (lazy EasyOCR, confidence gating, graceful degradation) |
| `app/media/audio.py` | 161 | WAV parsing, 16kHz mono resample, 30s window slicing, `AudioTranscriber` (faster-whisper int8, lazy) |
| `app/media/video.py` | 64 | OpenCV 1 fps frame sampler (decoder released in `finally`) |
| `app/media/device_manager.py` | 61 | Device resolution (cpu/mps/cuda) + `inference_session()` cache-cleanup context manager |
| `app/schemas_media.py` | 37 | `UnifiedMediaPayload` / `MediaMetadata` / `TemporalAnchors` contract |
| `app/tasks_media.py` | 61 | Celery task `media.process` on the `media` queue |

## 3. API Reference

### `POST /api/v1/media/ingest` → `202`
Multipart field: `file` (required), query `parent_document_id` (optional, defaults to
`"standalone"`). Supported media (validated by magic bytes, never by Content-Type):

| Format family | Signatures | Pipeline |
|---|---|---|
| Image | JPEG (`FF D8 FF`), PNG (`89 50 4E 47`), WebP (`RIFF…WEBP`), BMP (`BM`) | downsample → OCR |
| Video | MP4 (`ftyp`), AVI (`RIFF…AVI `), MKV (`1A 45 DF A3`) | 1 fps sampling → per-frame OCR |
| Audio | WAV (`RIFF…WAVE`), MP3 (`ID3` / MPEG sync) | 16 kHz mono → 30 s windows → ASR |

Error codes: `413` (size ceiling), `415` (unknown/empty media), `507` (disk floor), `500`
(storage failure), `503` (broker down).

## 4. Guardrails (as executed)

| Guardrail | Implementation | Verified by |
|---|---|---|
| Image edge ≤ 512 px, bilinear | `downsample()` — only resizes when the longest edge exceeds the cap; original resolution preserved in metadata | `test_downsample_respects_512px_guardrail` |
| Audio 16 kHz mono | `load_audio()` — mono mixdown + numpy linear-interpolation resample | `test_audio_windows_and_resample` |
| Strict 30 s audio windows | `[(0,30),(30,60),(60,75)]` slicing of the PCM stream | `test_audio_windows_and_resample` |
| 1 fps video sampling | `sample_frames()` — `interval = round(fps / 1.0)` | smoke (`scripts/smoke_media.py`) |
| VRAM/cache cleanup | `DeviceManager.inference_session()` (`try/finally` → MPS/CUDA `empty_cache` + `gc.collect()`) around every model batch | code path in OCR + ASR |
| Explicit timeouts | ffmpeg subprocess 120 s (`ffmpeg_timeout_s`); per-item 300 s via bounded `ThreadPoolExecutor` future; Celery soft/hard limits derived from `media_task_timeout_s` | pipeline + task decorators |
| Memory footprint | one model resident per worker process (lazy singletons); whisper `tiny` int8 on CPU; media queue concurrency 2 | `docker-compose.yml` |

## 5. Graceful Degradation

Heavy model stacks are **optional at runtime** — the pipeline never crashes when they are
absent; instead the payload carries a `notes[]` marker:

| Condition | Note | Behaviour |
|---|---|---|
| EasyOCR not installed / init failed | `ocr_unavailable` / `ocr_failed` | vision payload with empty text |
| faster-whisper not installed / init failed | `asr_unavailable` / `asr_failed` | audio payload with empty text |
| Undecodable image | `image_decode_failed` | payload skipped, warning logged |
| ffmpeg missing (MP3 input) | `AudioDecodeError` → task failure | logged, temp file cleaned |

## 6. Unified Payload Contract

`app.schemas_media.UnifiedMediaPayload` (Pydantic v2, strict):

- `parent_document_id: str` — links to the Phase 1 `CanonicalDocument.document_id`.
- `media_id: str` — UUID hex for vision items; the file stem for audio items.
- `modality: "vision" | "audio"` (enum-validated).
- `processed_text_content: str` — OCR text or merged transcript.
- `media_metadata`:
  - `original_resolution: [width, height]` (pre-resize; `[0, 0]` for audio).
  - `sampling_rate_hz`: `1` (vision frames = 1 fps sampling) / `16000` (audio).
  - `duration_seconds`: `0.0` for still images, `1.0` for a video frame, audio duration.
  - `temporal_anchors: {start, end}` — frame timestamp window / full audio span.
- `notes: list[str]` — degradation markers (extra field, additive).

## 7. Configuration (`INGEST_` prefix)

| Variable | Default | Purpose |
|---|---|---|
| `INGEST_MEDIA_TASK_QUEUE` | `media` | Dedicated Phase 2 Celery queue |
| `INGEST_MAX_IMAGE_EDGE_PX` | 512 | Image/frame downsample guardrail |
| `INGEST_AUDIO_TARGET_SAMPLE_RATE_HZ` | 16000 | Audio resample target |
| `INGEST_AUDIO_WINDOW_SECONDS` | 30.0 | Strict transcription window |
| `INGEST_VIDEO_SAMPLE_FPS` | 1.0 | Frame sampling rate |
| `INGEST_WHISPER_MODEL_SIZE` | `tiny` | int8 quantized ASR footprint |
| `INGEST_OCR_CONFIDENCE_THRESHOLD` | 0.35 | OCR text attachment gate |
| `INGEST_MEDIA_TASK_TIMEOUT_S` | 300 | Per-item processing timeout |
| `INGEST_FFMPEG_TIMEOUT_S` | 120 | ffmpeg subprocess timeout |
| `INGEST_MEDIA_CONCURRENCY` | 2 | Media worker child processes |

## 8. Observability

- JSON logs carry structured Phase 2 extras: `format`, `detections`,
  `original_resolution`, `duration_seconds`, `sample_rate_hz`, `window_count`,
  `sampled_frames`, `parent_document_id`, `payload_count`.
- Device resolution is logged once per process (`Compute device resolved` with
  `device`/`backend`).
- No `print` statements; all output via the shared `JsonFormatter`.

## 9. Test Coverage (Phase 2, in `tests/test_ingestion.py`)

| Test | Covers |
|---|---|
| `test_detect_media_formats` | Image/video/audio magic-byte routing |
| `test_detect_media_rejects_unknown` | 415 path for non-media bytes |
| `test_downsample_respects_512px_guardrail` | 2048×1024 → longest edge ≤ 512, original kept |
| `test_ocr_router_degrades_without_engine` | `ocr_unavailable` degradation |
| `test_audio_windows_and_resample` | 8 kHz→16 kHz mono; 75 s → 3 windows `(0-30, 30-60, 60-75)` |
| `test_transcriber_degrades_without_model` | `asr_unavailable` degradation |
| `test_process_media_image_payload_contract` | Full image → `UnifiedMediaPayload` contract |
| `test_media_payload_schema_roundtrip` | Schema serialization contract |

## 10. Running the Service

```bash
docker compose up --build   # redis + api + ingestion-worker + media-worker

# dedicated media worker (local dev)
celery -A app.celery_app.celery_app worker -l INFO --queues=media --concurrency=2

curl -F "file=@slide.png" "localhost:8000/api/v1/media/ingest?parent_document_id=doc-123"

# validation
pytest -q                                    # 33 tests (all phases)
PYTHONPATH=. python scripts/smoke_media.py   # real PNG + WAV through the pipeline
```

## 11. Known Limitations & Phase 3 Hooks

- **OCR/ASR models lazy-loaded per process** — first media task on a worker pays the
  model load; consider a warm-up task or model preloading in Phase 3.
- **EasyOCR not installed in the default image** — install `easyocr` (plus torch) to
  enable real OCR; the pipeline degrades gracefully otherwise.
- **`original_resolution` for audio is `[0, 0]`** — schema-compatible placeholder.
- **Frame-OCR for video is sequential** — parallelize frame batches behind the media queue.
- **No word-level timestamps / diarization** — per-window anchors only.
- **Phase 3 hook**: merge `UnifiedMediaPayload` streams with Phase 1 `CanonicalDocument`
  chunks into a unified multimodal index (embeddings out of scope here).

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 2.0 | Initial Phase 2 implementation: media ingestion API, OCR router, 1 fps video sampler, 16 kHz/30 s audio transcriber, unified payload schema, dedicated `media` queue, guardrail timeouts, 8 new tests (22 total), smoke script. |
| 2026-09-21 | 2.2 | Doc refresh: corrected module line counts, test counts updated for the 3-phase suite (33 total), cross-links to [`docs/INDEX.md`](INDEX.md). |
