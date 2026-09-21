"""Recursive character text splitter with token-based sizing and overlap.

Target: 512 tokens per chunk with a 10% overlap factor, implemented without
heavy external dependencies so chunking stays deterministic and testable.
"""

import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _token_len(text: str) -> int:
    """Approximate token count (whitespace/word-boundary aware heuristic)."""
    if not text:
        return 0
    return max(1, round(len(text) / settings.chars_per_token))


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        # Hard split on character windows as the final fallback.
        return list(text)
    parts = text.split(separator)
    return [p for p in parts if p.strip()]


def recursive_split(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Recursively split `text` into chunks of ~`target_tokens` with `overlap_tokens` carry-over."""
    if _token_len(text) <= target_tokens:
        return [text.strip()] if text.strip() else []

    for separator in _DEFAULT_SEPARATORS:
        if separator == "" or re.search(re.escape(separator), text):
            segments = _split_on_separator(text, separator)
            if len(segments) == 1 and separator != "":
                continue  # separator ineffective at this level; descend

            chunks: list[str] = []
            current: list[str] = []
            current_tokens = 0

            for segment in segments:
                seg_tokens = _token_len(segment)
                if seg_tokens > target_tokens and separator != "":
                    # Segment itself too large: flush, then recurse deeper.
                    if current:
                        chunks.append(separator.join(current).strip())
                        current, current_tokens = [], 0
                    chunks.extend(
                        recursive_split(segment, target_tokens, overlap_tokens)
                    )
                    continue

                if current_tokens + seg_tokens > target_tokens and current:
                    chunks.append(separator.join(current).strip())
                    # Overlap: retain tail segments totalling ~overlap_tokens.
                    tail: list[str] = []
                    tail_tokens = 0
                    for prev in reversed(current):
                        pt = _token_len(prev)
                        if tail_tokens + pt > overlap_tokens:
                            break
                        tail.insert(0, prev)
                        tail_tokens += pt
                    current, current_tokens = tail, tail_tokens

                current.append(segment)
                current_tokens += seg_tokens

            if current:
                remainder = separator.join(current).strip()
                if remainder:
                    chunks.append(remainder)
            return chunks

    return [text.strip()] if text.strip() else []


def chunk_text(text: str) -> list[str]:
    """Public API: normalize text and split into overlapping chunks at configured size."""
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        return []
    target = settings.chunk_target_tokens
    overlap = max(1, int(target * settings.chunk_overlap_fraction))
    return recursive_split(normalized, target, overlap)
