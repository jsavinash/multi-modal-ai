"""Magic-byte / MIME sniffing for upload validation.

Never trust the client-supplied Content-Type; validate the actual byte signature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


@dataclass(frozen=True)
class FormatSignature:
    fmt: SupportedFormat
    mime_type: str


_SIGNATURES: dict[SupportedFormat, FormatSignature] = {
    SupportedFormat.PDF: FormatSignature(SupportedFormat.PDF, "application/pdf"),
    SupportedFormat.DOCX: FormatSignature(
        SupportedFormat.DOCX, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}


class UnsupportedFormatError(ValueError):
    """Raised when a file does not match any supported magic-byte signature."""


def _is_zip(docx_bytes: bytes) -> bool:
    # DOCX is an OPC package: PK zip header + stored [Content_Types].xml
    return docx_bytes.startswith(b"PK\x03\x04") and b"[Content_Types].xml" in docx_bytes[:4096]


def detect_format(head_bytes: bytes, filename: str) -> FormatSignature:
    """Identify the true file format from magic bytes, falling back to text heuristics.

    Args:
        head_bytes: The first >= 4KB of the file.
        filename: Original filename, used only to disambiguate TXT vs MD.

    Raises:
        UnsupportedFormatError: If the format cannot be validated.
    """
    if head_bytes.startswith(b"%PDF-"):
        return _SIGNATURES[SupportedFormat.PDF]
    if _is_zip(head_bytes):
        return _SIGNATURES[SupportedFormat.DOCX]

    # Plain-text family: valid UTF-8 (or Latin-1 fallback) without binary NULs.
    if b"\x00" not in head_bytes:
        try:
            head_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                head_bytes.decode("latin-1")
            except UnicodeDecodeError:
                raise UnsupportedFormatError(
                    f"File '{filename}' is neither a known binary signature nor decodable text."
                ) from None
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        fmt = SupportedFormat.MD if suffix in ("md", "markdown") else SupportedFormat.TXT
        return FormatSignature(fmt, "text/markdown" if fmt is SupportedFormat.MD else "text/plain")

    raise UnsupportedFormatError(f"File '{filename}' has an unrecognized or corrupted header.")


def sniff(stream_read_header) -> FormatSignature:
    """Convenience wrapper: takes a callable returning the first 4KB of bytes."""
    return detect_format(stream_read_header(4096), "")
