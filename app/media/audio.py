"""Audio pipeline: 16kHz mono downsampling, 30s windows, faster-whisper int8 transcription."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings
from app.media.device_manager import device_manager

logger = logging.getLogger(__name__)

_TARGET_HZ = settings.audio_target_sample_rate_hz
_WINDOW_S = settings.audio_window_seconds


class AudioDecodeError(ValueError):
    """Raised when audio cannot be decoded or resampled."""


@dataclass
class AudioWindows:
    pcm: np.ndarray                      # float32 mono at _TARGET_HZ
    windows: list[tuple[float, float]]   # (start_s, end_s) anchors
    duration_seconds: float


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise AudioDecodeError(f"Unsupported WAV sample width: {width * 8} bits")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)  # mono mixdown
    return pcm, rate


def _resample_linear(pcm: np.ndarray, source_hz: int) -> np.ndarray:
    """Linear-interpolation resample to the target rate (numpy only, no scipy)."""
    if source_hz == _TARGET_HZ or pcm.size == 0:
        return pcm.astype(np.float32)
    duration = pcm.shape[0] / source_hz
    target_len = max(1, int(round(duration * _TARGET_HZ)))
    x_src = np.linspace(0.0, duration, num=pcm.shape[0], endpoint=False)
    x_dst = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_dst, x_src, pcm).astype(np.float32)


def _decode_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Fallback for MP3/other containers: ffmpeg -> 16k mono WAV, with hard timeout."""
    tmp_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", str(_TARGET_HZ), "-ac", "1",
             "-f", "wav", str(tmp_path)],
            capture_output=True, timeout=settings.ffmpeg_timeout_s, check=True,
        )
        return _read_wav(tmp_path)
    except FileNotFoundError as exc:
        raise AudioDecodeError(
            "ffmpeg is not installed; MP3/other containers cannot be decoded."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(
            f"ffmpeg exceeded the {settings.ffmpeg_timeout_s}s timeout."
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def load_audio(path: Path) -> AudioWindows:
    """Decode to 16kHz mono and slice into strict 30s windows."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".wav":
            pcm, rate = _read_wav(path)
        else:
            pcm, rate = _decode_via_ffmpeg(path)
    except wave.Error as exc:
        raise AudioDecodeError(f"Corrupt WAV header in '{path.name}': {exc}") from exc

    pcm = _resample_linear(pcm, rate)
    duration = pcm.shape[0] / _TARGET_HZ if _TARGET_HZ else 0.0
    windows = [
        (round(start, 3), round(min(start + _WINDOW_S, duration), 3))
        for start in np.arange(0.0, duration, _WINDOW_S)
    ] or [(0.0, 0.0)]
    logger.info("Audio loaded", extra={
        "duration_seconds": round(duration, 3),
        "sample_rate_hz": _TARGET_HZ, "window_count": len(windows),
    })
    return AudioWindows(pcm=pcm, windows=windows, duration_seconds=round(duration, 3))


class AudioTranscriber:
    """faster-whisper (int8) transcriber; lazy model load, per-window inference."""

    def __init__(self) -> None:
        self._model = None
        self._available: bool | None = None  # None = not probed yet

    def _get_model(self):
        if self._available is False:
            return None
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # optional heavy dependency

                # int8 quantization keeps the footprint low on the 8GB host.
                self._model = WhisperModel(
                    settings.whisper_model_size, device="cpu", compute_type="int8",
                )
                self._available = True
            except ImportError:
                logger.warning("faster-whisper not installed; transcription disabled")
                self._available = False
            except Exception as exc:  # noqa: BLE001
                logger.warning("Whisper model init failed", extra={"reason": str(exc)})
                self._available = False
        return self._model

    def transcribe(self, audio: AudioWindows) -> tuple[list[tuple[float, float, str]], list[str]]:
        """Return [(start, end, text)] per window plus degradation notes."""
        model = self._get_model()
        if model is None:
            return [], ["asr_unavailable"]
        segments_out: list[tuple[float, float, str]] = []
        notes: list[str] = []
        try:
            with device_manager.inference_session():
                for start, end in audio.windows:
                    # Slice the exact 30s window from the decoded PCM stream.
                    slice_pcm = audio.pcm[int(start * _TARGET_HZ):int(end * _TARGET_HZ)]
                    if slice_pcm.size == 0:
                        continue
                    segments, _info = model.transcribe(slice_pcm, language="en", beam_size=1)
                    for seg in segments:
                        segments_out.append((
                            round(start + seg.start, 3),
                            round(start + min(seg.end, end - start), 3),
                            seg.text.strip(),
                        ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transcription failed", extra={"reason": str(exc)})
            notes.append("asr_failed")
        return segments_out, notes
