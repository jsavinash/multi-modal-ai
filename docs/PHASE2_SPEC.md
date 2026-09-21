# Phase 2 — Vision, Audio & OCR Processing Node (Adapted Specification)

> Phase 2 of the Multimodal AI Ingestion Engine inherits the media fragments emitted by
> Phase 1 (`extracted_media_references` → `media_id` / image bytes) and adds standalone
> image, video and audio ingestion.
>
> **This spec is the original Phase 2 engineering prompt adapted to the actual host system**
> (analysis below). Deviations from the original prompt are marked **[ADAPTED]**.

## 0. System Analysis → Prompt Modifications

| Original prompt constraint | Actual system finding | Adaptation |
|---|---|---|
| Shared 1x Enterprise GPU; low VRAM | **No discrete GPU.** Apple M1 (ARM64), 8 GB unified RAM, 8 cores, macOS 26 | **[ADAPTED]** CPU-first inference with optional Apple MPS acceleration when PyTorch is present; "VRAM cleanup" generalized to device-cache cleanup (MPS/CUDA cache + `gc.collect()`) inside `try...finally` blocks. Model footprints capped at ~1 GB to respect the Phase 1 worker budget (4 x 1 GB children on an 8 GB host) |
| EasyOCR / quantized Florence-2 on GPU | EasyOCR drags in PyTorch (~2 GB wheel); Florence-2 exceeds the 8 GB unified-memory envelope for a shared worker | **[ADAPTED]** EasyOCR is the primary OCR engine, **lazy-imported and optional** — if the model stack is absent the pipeline degrades gracefully (payload emitted with empty text + `ocr_unavailable` note) instead of crashing. Florence-2 explicitly out of scope |
| faster-whisper int8 | Works well on CPU int8 | Kept, **[ADAPTED]** default model `tiny` (configurable up to `base`), processed per 30 s window so only one window's activations live at a time |
| 512x512 image edge cap (bilinear) | Fine — also reduces EasyOCR CPU cost | Kept as-is (Pillow, bilinear) |
| 16 kHz mono audio, 30 s windows | ffmpeg not guaranteed on host | **[ADAPTED]** PCM WAV decoded natively via the stdlib `wave` module with numpy linear-interpolation resampling; MP3/other containers go through an **ffmpeg subprocess with a hard 120 s timeout** and a clear failure if ffmpeg is missing |
| 1 fps video frame sampling | CPU-bound, fine | Kept as-is (OpenCV headless); frames bypassed straight to the OCR router at 512x512 |

## 1. System Capacity Constraints (adapted guardrails)

- **Compute budget:** CPU-first inference on the shared 8-core Apple M1; optional MPS when available. **[ADAPTED]**
- **Image guardrail:** every image / video frame is downsampled so its longest edge is ≤ **512 px**, bilinear interpolation, before model ingestion.
- **Audio guardrail:** downsample to **16 kHz mono**; strict **30-second** window blocks before transcription.
- **Memory:** models load lazily per modality and are released in `try...finally` blocks; never more than one model resident per worker child at a time.

## 2. Functional Requirements

1. **Intelligent OCR Routing:** read raw image payloads (and 1 fps video frames). If an
   image contains structural text, route through the OCR engine. Detection heuristic:
   route every raster image through OCR (cheap at 512 px); EasyOCR's confidence
   scores gate whether text is attached to the payload.
2. **Video Frame Sampling Engine:** process MP4/AVI with OpenCV at a **strict 1 fps**
   sampling rate. Each sampled frame becomes a vision payload anchored at its timestamp.
3. **Audio Transcriber:** process WAV/MP3. Localized **faster-whisper**, `int8`
   quantization, transcripts carry explicit `start_time` / `end_time` timestamps per
   30 s window.
4. **Payload Standardization:** unified multi-modal payload:

```json
{
  "parent_document_id": "str",
  "media_id": "str",
  "modality": "vision|audio",
  "processed_text_content": "str",
  "media_metadata": {
    "original_resolution": [int, int],
    "sampling_rate_hz": int,
    "duration_seconds": float,
    "temporal_anchors": {"start": float, "end": float}
  }
}
```

- vision payloads: `original_resolution` = pre-resize W×H; `sampling_rate_hz` = frame
  sampling rate (1); `temporal_anchors` = frame timestamp window.
- audio payloads: `sampling_rate_hz` = 16000; `temporal_anchors` = the 30 s window bounds.

## 3. Coding Standards

- `try...finally` resource-cleanup blocks around every model invocation: free the device
  cache (MPS/CUDA) and garbage-collect after each batch.
- Explicit timeout constraints: ffmpeg subprocess 120 s; per-item model calls bounded by a
  configurable processing timeout (default 300 s) enforced via worker futures.

## 4. Out of Scope / Deferred

- GPU batch scheduling across a fleet (single-node, shared-queue design).
- Florence-2 / VLM captioning; scene classification.
- Speaker diarization or word-level timestamps.

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 2.0 | Original prompt adapted to actual host (no discrete GPU; 8 GB unified RAM; ffmpeg optional) and approved for execution. |
| 2026-09-21 | 2.1 | Executed. Technical implementation documented in [`docs/PHASE2_DOCUMENTATION.md`](PHASE2_DOCUMENTATION.md); all guardrails verified by tests/smoke. |
| 2026-09-21 | 3.0 | Phase 3 vector indexer added; Qdrant collections now include `vision` / `audio` / `document_text` / `tabular_telemetry` (see [`docs/PHASE3_DOCUMENTATION.md`](PHASE3_DOCUMENTATION.md)). |
| 2026-09-21 | 4.0 | Phase 4 reasoning core added; `ReasoningResult` consumes all prior phases' outputs (see [`docs/PHASE4_DOCUMENTATION.md`](PHASE4_DOCUMENTATION.md)). |
