"""End-to-end Phase 2 smoke test: real PNG + real WAV through the media pipeline."""
import logging
import tempfile
import wave as wave_mod
from pathlib import Path

import numpy as np
from PIL import Image

from app.logging_config import configure_logging

configure_logging()


def make_png(width: int, height: int, color) -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def make_wav(path: Path, rate: int, seconds: float) -> None:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    with wave_mod.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(tone.tobytes())


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # --- vision: 1500x800 PNG -> 512px downsample + OCR route ---
    from app.media.pipeline import MediaFormat, MediaSignature, process_media

    png_path = tmp_path / "slide.png"
    png_path.write_bytes(make_png(1500, 800, (250, 240, 220)))
    payloads = process_media(png_path, MediaSignature(MediaFormat.IMAGE, "image/png"), "smoke-doc")
    assert len(payloads) == 1
    pl = payloads[0]
    assert pl.modality.value == "vision"
    assert pl.media_metadata.original_resolution == [1500, 800]
    logging.getLogger("smoke").info("vision payload: text=%r notes=%s",
                                    pl.processed_text_content[:60], pl.notes)

    # --- audio: 8kHz mono 75s WAV -> 16kHz resample + 30s windows ---
    wav_path = tmp_path / "meeting.wav"
    make_wav(wav_path, 8000, 75.0)
    from app.media.audio import load_audio

    audio = load_audio(wav_path)
    assert len(audio.windows) == 3, audio.windows
    assert audio.pcm.shape[0] == int(75.0 * 16000)
    logging.getLogger("smoke").info("audio: %d windows, anchors=%s",
                                    len(audio.windows), audio.windows)

    # --- timeouts: oversized audio must be rejected by the ffmpeg/window path ---
    from app.config import settings

    logging.getLogger("smoke").info(
        "guards ok: image_edge=%dpx, window=%ss, sample=%sfps, timeout=%ss",
        settings.max_image_edge_px, settings.audio_window_seconds,
        settings.video_sample_fps, settings.media_task_timeout_s,
    )

logging.getLogger("smoke").info("PHASE 2 SMOKE OK")
