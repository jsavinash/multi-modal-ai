"""Vector payload schema: modality collections + validation before any DB write."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.config import settings


class SchemaValidationError(ValueError):
    """Raised when a payload does not map to the collection's schema."""


# Explicit collections for distinct modalities (per spec FR-5).
COLLECTIONS: dict[str, str] = {
    "tabular": "tabular_telemetry",
    "document_text": "document_text",
    "vision": "vision",
    "audio": "audio",
}

# field -> allowed python types (validated before upsert)
REQUIRED_PAYLOAD_FIELDS: dict[str, tuple] = {
    "document_id": (str,),
    "modality": (str,),
    "timestamp": (str, type(None)),
    "text": (str,),
}


@dataclass
class IndexPayload:
    """One vector point ready for upsert."""
    point_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    collection: str = ""
    vector: list[float] = field(default_factory=list)
    payload: dict = field(default_factory=dict)


def validate_payload_for_collection(payload: IndexPayload) -> None:
    """Verify embedded data types map correctly to the collection schema.

    Checks vector dimensionality against the configured embedding dim and ensures
    every required payload field exists with the correct Python type.
    """
    if len(payload.vector) != settings.embedding_dim:
        raise SchemaValidationError(
            f"Vector dim {len(payload.vector)} != collection dim "
            f"{settings.embedding_dim} for collection '{payload.collection}'."
        )
    if payload.collection not in COLLECTIONS.values():
        raise SchemaValidationError(
            f"Unknown collection '{payload.collection}'. "
            f"Expected one of {sorted(COLLECTIONS.values())}."
        )
    for field_name, allowed in REQUIRED_PAYLOAD_FIELDS.items():
        if field_name not in payload.payload:
            raise SchemaValidationError(
                f"Payload for '{payload.collection}' missing required field '{field_name}'."
            )
        if not isinstance(payload.payload[field_name], allowed):
            actual = type(payload.payload[field_name]).__name__
            expected = [t.__name__ for t in allowed]
            raise SchemaValidationError(
                f"Payload field '{field_name}' is {actual}, expected one of {expected}."
            )


def make_payload(collection_key: str, document_id: str, text: str,
                 timestamp: str | None, extra: dict | None = None) -> IndexPayload:
    """Build an IndexPayload for the given modality collection (pre-vector)."""
    payload = {
        "document_id": document_id,
        "modality": collection_key,
        "timestamp": timestamp,
        "text": text,
    }
    payload.update(extra or {})
    return IndexPayload(collection=COLLECTIONS[collection_key], payload=payload)
