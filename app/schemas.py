"""Canonical Pydantic schemas — the strictly validated output contract of the ingestion engine."""

from enum import Enum

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    RAW_IMAGE = "raw_image"


class DocumentMetadata(BaseModel):
    filename: str
    file_size_bytes: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class TextPayload(BaseModel):
    chunk_id: int = Field(ge=0)
    page_number: int = Field(ge=1)
    modality: Modality = Modality.TEXT
    content: str
    structural_tag: str


class MediaReference(BaseModel):
    media_id: str
    page_number: int = Field(ge=1)
    spatial_bounding_box: list[float] = Field(min_length=4, max_length=4)  # [x0, y0, x1, y1]
    modality: Modality = Modality.RAW_IMAGE


class CanonicalDocument(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    payloads: list[TextPayload] = Field(default_factory=list)
    extracted_media_references: list[MediaReference] = Field(default_factory=list)
