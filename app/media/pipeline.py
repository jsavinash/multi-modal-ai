"""Phase 2 media processing pipeline: detection, routing, payload assembly."""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.config import settings
from app.media.audio import AudioDecodeError, AudioTranscriber, load_audio
from app.media.video import sample_frames
from app.media.vision import OcrRouter, downsample
from app.schemas_media import (
    MediaMetadata,
    MediaModality,
    TemporalAnchors,
    UnifiedMediaPayload,
)

logger = logging.getLogger(__name__)


class MediaFormat(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO_WAV = "audio_wav"
    AUDIO_MP3 = "audio_mp3"


class UnsupportedMediaError(ValueError):
    """Raised when media magic bytes cannot be validated."""


class MediaProcessingTimeout(RuntimeError):
    """Raised when a media item exceeds the per-item processing timeout."""


@dataclass(frozen=True)
class MediaSignature:
    fmt: MediaFormat
    mime_type: str


def detect_media_format(head: bytes, filename: str) -> MediaSignature:
    """Magic-byte detection for image/video/audio media uploads."""
    if head.startswith(b"\xff\xd8\xff"):
        return MediaSignature(MediaFormat.IMAGE, "image/jpeg")
    if head.startswith(b"\x89PNG"):
        return MediaSignature(MediaFormat.IMAGE, "image/png")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return MediaSignature(MediaFormat.IMAGE, "image/webp")
    if head.startswith(b"BM"):
        return MediaSignature(MediaFormat.IMAGE, "image/bmp")
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return MediaSignature(MediaFormat.AUDIO_WAV, "audio/wav")
    if head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return MediaSignature(MediaFormat.AUDIO_MP3, "audio/mpeg")
    if head[4:8] == b"ftyp":
        return MediaSignature(MediaFormat.VIDEO, "video/mp4")
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return MediaSignature(MediaFormat.VIDEO, "video/x-msvideo")
    if head.startswith(b"\x1aE\xdf\xa3"):
        return MediaSignature(MediaFormat.VIDEO, "video/x-matroska")
    raise UnsupportedMediaError(f"File '{filename}' is not a supported media format.")


def _with_timeout(fn, *args):
    """Enforce the explicit per-item processing timeout constraint."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        try:
            return future.result(timeout=settings.media_task_timeout_s)
        except FutureTimeout as exc:
            raise MediaProcessingTimeout(
                f"Media processing exceeded the {settings.media_task_timeout_s}s limit."
            ) from exc


def process_media(
    path: Path, signature: MediaSignature, parent_document_id: str
) -> list[UnifiedMediaPayload]:
    """Process one media file into unified multi-modal payloads."""
    file_size = path.stat().st_size
    logger.info("Processing media", extra={
        "format": signature.fmt.value, "file_size_bytes": file_size,
        "parent_document_id": parent_document_id,
    })

    if signature.fmt is MediaFormat.VIDEO:
        return _process_video(path, parent_document_id)
    if signature.fmt is MediaFormat.IMAGE:
        return _process_image_bytes(path.read_bytes(), path.name, parent_document_id)
    return _process_audio(path, parent_document_id)


def _process_image_bytes(
    image_bytes: bytes, name: str, parent_id: str, timestamp: float | None = None
) -> list[UnifiedMediaPayload]:
    """OCR route a single raster image (or a sampled video frame)."""
    from app.media.vision import new_media_id

    def _run() -> tuple:
        img, original = downsample(image_bytes)
        return img, original

    try:
        _, original = _with_timeout(_run)
    except MediaProcessingTimeout:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image preprocess failed", extra={"file_name": name, "reason": str(exc)})
        return []

    ocr = _with_timeout(ocr_router.extract_text, image_bytes)
    width, height = original
    start = timestamp if timestamp is not None else 0.0
    payload = UnifiedMediaPayload(
        parent_document_id=parent_id,
        media_id=new_media_id(),
        modality=MediaModality.VISION,
        processed_text_content=ocr.text,
        media_metadata=MediaMetadata(
            original_resolution=[int(width), int(height)],
            sampling_rate_hz=1,  # vision frames are point-in-time samples
            duration_seconds=0.0 if timestamp is None else 1.0 / settings.video_sample_fps,
            temporal_anchors=TemporalAnchors(start=start, end=start + (0.0 if timestamp is None else 1.0 / settings.video_sample_fps)),
        ),
        notes=list(ocr.notes),
    )
    return [payload]


def _process_video(path: Path, parent_id: str) -> list[UnifiedMediaPayload]:
    """1fps sampling; each frame is OCR-routed at the 512px guardrail."""
    frames = _with_timeout(sample_frames, str(path))
    payloads: list[UnifiedMediaPayload] = []
    for frame in frames:
        payloads.extend(
            _process_image_bytes(
                frame.image_bytes_jpeg, path.name, parent_id,
                timestamp=frame.timestamp_seconds,
            )
        )
    return payloads


def _process_audio(path: Path, parent_id: str) -> list[UnifiedMediaPayload]:
    """16kHz mono + strict 30s windows through faster-whisper int8."""
    try:
        audio = _with_timeout(load_audio, path)
    except AudioDecodeError as exc:
        logger.warning("Audio decode failed", extra={"path": str(path), "reason": str(exc)})
        return []
    segments, notes = _with_timeout(audio_transcriber.transcribe, audio)
    merged_text = " ".join(text for _s, _e, text in segments).strip()
    payload = UnifiedMediaPayload(
        parent_document_id=parent_id,
        media_id=path.stem,  # deterministic for audio files
        modality=MediaModality.AUDIO,
        processed_text_content=merged_text,
        media_metadata=MediaMetadata(
            original_resolution=[0, 0],  # not applicable to audio
            sampling_rate_hz=settings.audio_target_sample_rate_hz,
            duration_seconds=audio.duration_seconds,
            temporal_anchors=TemporalAnchors(start=0.0, end=audio.duration_seconds),
        ),
        notes=notes,
    )
    return [payload]


# Lazily-instantiated shared model handles (one per worker process).
ocr_router = OcrRouter()
audio_transcriber = AudioTranscriber()
