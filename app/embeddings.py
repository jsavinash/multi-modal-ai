"""Embedding Model Pipeline: local all-MiniLM-L6-v2 with a deterministic fallback."""

from __future__ import annotations

import hashlib
import logging
import math

from app.config import settings
from app.media.device_manager import device_manager

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Lazy singleton for all-MiniLM-L6-v2; falls back to a hash embedder when absent.

    The fallback is deterministic (same text -> same vector) so tests and offline
    runs stay reproducible without the ~80MB model download.
    """

    def __init__(self) -> None:
        self._model = None
        self._available: bool | None = None  # None = not probed yet

    def _get_model(self):
        if self._available is False:
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # heavy, optional

                with device_manager.inference_session():
                    self._model = SentenceTransformer(settings.embedding_model_name, device="cpu")
                self._available = True
                logger.info("Embedding model loaded", extra={
                    "model": settings.embedding_model_name, "dim": settings.embedding_dim,
                })
            except ImportError:
                logger.warning("sentence-transformers not installed; using hash fallback")
                self._available = False
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding model init failed", extra={"reason": str(exc)})
                self._available = False
        return self._model

    @property
    def available(self) -> bool:
        self._get_model()
        return bool(self._available)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed up to max_embed_batch_rows texts; normalized 384-dim vectors."""
        if not texts:
            return []
        if len(texts) > settings.max_embed_batch_rows:
            raise ValueError(
                f"Batch of {len(texts)} exceeds the {settings.max_embed_batch_rows}-row RAM cap."
            )
        model = self._get_model()
        if model is not None:
            try:
                with device_manager.inference_session():
                    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
                return [v.tolist() for v in vectors]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model encode failed; falling back", extra={"reason": str(exc)})
        return [self._hash_embed(t) for t in texts]

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        """Deterministic fallback: seeded SHA-256 stream projected to the target dim."""
        dim = settings.embedding_dim
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        out: list[float] = []
        counter = 0
        while len(out) < dim:
            block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(block), 2):
                if len(out) >= dim:
                    break
                value = int.from_bytes(block[i:i + 2], "big") / 65535.0
                out.append(value * 2.0 - 1.0)
            counter += 1
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [round(v / norm, 6) for v in out]


embedding_engine = EmbeddingEngine()
