"""Vision pipeline: 512px bilinear downsampling + intelligent OCR routing."""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass

from PIL import Image

from app.config import settings
from app.media.device_manager import device_manager

logger = logging.getLogger(__name__)


@dataclass
class OcrResult:
    text: str
    detections: int
    avg_confidence: float
    notes: list[str]


def ocr_gpu_enabled() -> bool:
    """Whether EasyOCR may be constructed with ``gpu=True``.

    EasyOCR only accelerates on CUDA/Metal *via torch's CUDA path*; its Metal
    (MPS) support is not available, so on an Apple-Silicon host the reader must
    be built on CPU even when MPS was detected as the compute device.
    """
    return device_manager.device == "cuda"


def downsample(image_bytes: bytes) -> tuple[Image.Image, tuple[int, int]]:
    """Downsample an image so its longest edge is <= max_image_edge_px (bilinear).

    Returns (resized_image, original_resolution[width, height]).
    """
    with device_manager.inference_session():
        img = Image.open(io.BytesIO(image_bytes))
        original = (img.width, img.height)
        edge = settings.max_image_edge_px
        if max(original) > edge:
            scale = edge / max(original)
            new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.convert("RGB").resize(new_size, Image.BILINEAR)
        else:
            img = img.convert("RGB")
        return img, original


class OcrRouter:
    """Routes text-bearing images through EasyOCR; degrades gracefully if absent."""

    def __init__(self) -> None:
        self._reader = None  # lazy: only load the model on first use

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr  # optional heavy dependency

                self._reader = easyocr.Reader(["en"], gpu=ocr_gpu_enabled(), verbose=False)
            except ImportError:
                logger.warning("EasyOCR not installed; OCR disabled for this run")
                self._reader = False  # sentinel: permanently unavailable
            except Exception as exc:  # noqa: BLE001
                logger.warning("EasyOCR init failed; OCR disabled", extra={"reason": str(exc)})
                self._reader = False
        return self._reader or None

    def extract_text(self, image_bytes: bytes) -> OcrResult:
        """OCR a raw image. Degrades to an empty result when the engine is unavailable."""
        notes: list[str] = []
        try:
            img, original = downsample(image_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image decode/downsample failed", extra={"reason": str(exc)})
            return OcrResult("", 0, 0.0, ["image_decode_failed"])

        reader = self._get_reader()
        if reader is None:
            notes.append("ocr_unavailable")
            return OcrResult("", 0, 0.0, notes)

        try:
            with device_manager.inference_session():
                results = reader.readtext(numpy_array(img))
            threshold = settings.ocr_confidence_threshold
            texts, confs = [], []
            for _bbox, text, conf in results:
                if float(conf) >= threshold:
                    texts.append(text)
                    confs.append(float(conf))
            logger.info("OCR pass complete", extra={
                "detections": len(texts),
                "original_resolution": list(original),
            })
            return OcrResult(
                " ".join(texts).strip(), len(texts),
                sum(confs) / len(confs) if confs else 0.0, notes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR inference failed", extra={"reason": str(exc)})
            notes.append("ocr_failed")
            return OcrResult("", 0, 0.0, notes)


def numpy_array(img: Image.Image):
    import numpy as np

    return np.asarray(img)


def new_media_id() -> str:
    return uuid.uuid4().hex
