"""Unified multi-modal payload schemas for the Phase 2 processing node."""

from enum import Enum

from pydantic import BaseModel, Field


class MediaModality(str, Enum):
    VISION = "vision"
    AUDIO = "audio"


class TemporalAnchors(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)


class MediaMetadata(BaseModel):
    original_resolution: list[int] = Field(min_length=2, max_length=2)  # [width, height]
    sampling_rate_hz: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    temporal_anchors: TemporalAnchors


class UnifiedMediaPayload(BaseModel):
    parent_document_id: str
    media_id: str
    modality: MediaModality
    processed_text_content: str = ""
    media_metadata: MediaMetadata
    notes: list[str] = Field(default_factory=list)  # e.g. "ocr_unavailable"


class MediaProcessingResult(UnifiedMediaPayload):
    """Adds processing diagnostics on top of the wire contract."""
    ocr_detections: int = Field(default=0, ge=0)
    avg_ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
