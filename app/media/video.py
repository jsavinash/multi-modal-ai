"""Video frame sampling engine: strict 1 frame per second via OpenCV."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    frame_index: int
    timestamp_seconds: float
    image_bytes_jpeg: bytes
    original_resolution: tuple[int, int]


def sample_frames(path: str) -> list[SampledFrame]:
    """Sample one frame per second (strict rate) from a video file.

    Frames are returned as JPEG-encoded bytes at their ORIGINAL resolution;
    the 512px downsample happens later in the vision pipeline (single guardrail).
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video stream: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (total_frames / fps) if fps > 0 else 0.0
        interval = max(1, round(fps / settings.video_sample_fps))

        frames: list[SampledFrame] = []
        index = 0
        ok, frame = capture.read()
        while ok:
            if index % interval == 0:
                success, jpeg = cv2.imencode(".jpg", frame)
                if success:
                    h, w = frame.shape[:2]
                    frames.append(SampledFrame(
                        frame_index=index,
                        timestamp_seconds=round(index / fps, 3),
                        image_bytes_jpeg=jpeg.tobytes(),
                        original_resolution=(int(w), int(h)),
                    ))
            ok, frame = capture.read()
            index += 1

        logger.info("Frame sampling complete", extra={
            "sampled_frames": len(frames), "video_duration_seconds": round(duration, 3),
            "sample_fps": settings.video_sample_fps,
        })
        if not frames:
            logger.warning("Video produced no sampleable frames", extra={"path": path})
        return frames
    finally:
        capture.release()  # try/finally: always release the decoder handle
