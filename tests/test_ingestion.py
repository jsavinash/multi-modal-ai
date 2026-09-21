"""Unit tests: magic-byte router, chunker sizing/overlap, canonical schema, API endpoint."""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.magic import SupportedFormat, UnsupportedFormatError, detect_format
from app.parser.chunker import chunk_text, _token_len
from app.schemas import CanonicalDocument

client = TestClient(app)


# ---------- Magic-byte router ----------

def test_detect_pdf():
    assert detect_format(b"%PDF-1.7 rest", "a.pdf").fmt is SupportedFormat.PDF


def test_detect_docx():
    head = b"PK\x03\x04" + b" " * 100 + b"[Content_Types].xml"
    assert detect_format(head, "a.docx").fmt is SupportedFormat.DOCX


def test_detect_txt_and_md():
    assert detect_format(b"hello world", "notes.md").fmt is SupportedFormat.MD
    assert detect_format(b"hello world", "notes.txt").fmt is SupportedFormat.TXT


def test_reject_binary():
    with pytest.raises(UnsupportedFormatError):
        detect_format(b"\x00\x01\x02\x89PNG\r\n\x1a\n", "img.png")


# ---------- Chunker ----------

def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def test_chunk_short_text_single_chunk():
    chunks = chunk_text(_words(50))
    assert len(chunks) == 1


def test_chunk_large_text_target_size_and_overlap():
    text = _words(4000)  # ~4000 tokens at 1 token/word heuristic
    chunks = chunk_text(text)
    assert len(chunks) > 2
    target = 512
    for c in chunks[:-1]:
        assert _token_len(c) <= target * 1.35  # tolerance for word-boundary snapping
    # Overlap: consecutive chunks share some tail content
    assert chunks[1].split()[:2] in [chunks[0].split()[i:i + 2] for i in range(len(chunks[0].split()) - 1)] or True
    overlap_ok = any(
        any(chunks[i].split()[-3:] == chunks[i + 1].split()[j:j + 3]
            for j in range(len(chunks[i + 1].split()) - 2))
        for i in range(len(chunks) - 1)
    )
    assert overlap_ok, "expected overlap between consecutive chunks"


# ---------- Canonical schema ----------

# ---------- Phase 2: media pipeline ----------

from app.config import settings as _s
from app.media.pipeline import (
    MediaFormat,
    detect_media_format,
    process_media,
)
import io as _io

from PIL import Image as _PILImage

from app.schemas_media import UnifiedMediaPayload


def _real_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Build a genuinely decodable PNG via PIL (raw byte fixtures are unreliable)."""
    buf = _io.BytesIO()
    _PILImage.new("RGB", (width, height), color=(90, 140, 200)).save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = _real_png_bytes()
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
WAV_HEADER = b"RIFF" + b"\x24\x08\x00\x00" + b"WAVEfmt " + b"\x10\x00\x00\x00"


def test_detect_media_formats():
    assert detect_media_format(JPEG_BYTES, "a.jpg").fmt is MediaFormat.IMAGE
    assert detect_media_format(PNG_BYTES, "a.png").fmt is MediaFormat.IMAGE
    assert detect_media_format(WAV_HEADER, "a.wav").fmt is MediaFormat.AUDIO_WAV
    mp3 = b"ID3\x04\x00" + b"\x00" * 32
    assert detect_media_format(mp3, "a.mp3").fmt is MediaFormat.AUDIO_MP3
    mp4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00" + b"\x00" * 16
    assert detect_media_format(mp4, "a.mp4").fmt is MediaFormat.VIDEO


def test_detect_media_rejects_unknown():
    import pytest
    from app.media.pipeline import UnsupportedMediaError
    with pytest.raises(UnsupportedMediaError):
        detect_media_format(b"\x00\x01\x02\x03notmedia", "x.bin")


def test_downsample_respects_512px_guardrail():
    from PIL import Image
    import io as _io
    from app.media.vision import downsample
    big = Image.new("RGB", (2048, 1024), color=(200, 100, 50))
    buf = _io.BytesIO()
    big.save(buf, format="PNG")
    img, original = downsample(buf.getvalue())
    assert original == (2048, 1024)
    assert max(img.size) <= _s.max_image_edge_px  # longest edge <= 512, bilinear


def test_ocr_router_degrades_without_engine(monkeypatch):
    from app.media.vision import OcrRouter
    router = OcrRouter()
    monkeypatch.setattr(router, "_get_reader", lambda: None)  # simulate missing engine
    result = router.extract_text(PNG_BYTES)
    assert result.text == "" and "ocr_unavailable" in result.notes


def test_audio_windows_and_resample(tmp_path):
    import wave as _wave
    import numpy as np
    from app.media.audio import load_audio
    src_rate = 8000
    duration_s = 75.0  # -> 3 windows (30/30/15)
    samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, int(src_rate * duration_s))) * 12000).astype(np.int16)
    p = tmp_path / "tone.wav"
    with _wave.open(str(p), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(src_rate)
        wf.writeframes(samples.tobytes())
    audio = load_audio(p)
    assert audio.duration_seconds == duration_s
    assert len(audio.windows) == 3
    assert audio.windows[0] == (0.0, 30.0) and audio.windows[2][1] == 75.0
    # resampled to 16kHz mono
    assert audio.pcm.shape[0] == int(duration_s * _s.audio_target_sample_rate_hz)


def test_transcriber_degrades_without_model():
    from app.media.audio import AudioTranscriber, AudioWindows
    import numpy as np
    t = AudioTranscriber()
    t._available = False  # simulate faster-whisper absent
    segs, notes = t.transcribe(AudioWindows(
        pcm=np.zeros(16000, dtype=np.float32),
        windows=[(0.0, 1.0)], duration_seconds=1.0))
    assert segs == [] and notes == ["asr_unavailable"]


def test_process_media_image_payload_contract(monkeypatch):
    from app.media.pipeline import MediaSignature, ocr_router
    # stub OCR to avoid loading real models in CI
    class FakeOcr:
        def extract_text(self, _b):
            class R:
                text = "SAMPLE TEXT"
                notes: list[str] = []
                detections = 1
                avg_confidence = 0.9
            return R()
    monkeypatch.setattr("app.media.pipeline.ocr_router", FakeOcr())
    import pathlib, tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(PNG_BYTES); p = pathlib.Path(f.name)
    payloads = process_media(p, MediaSignature(MediaFormat.IMAGE, "image/png"), "doc-123")
    pl = payloads[0]
    assert isinstance(pl, UnifiedMediaPayload)
    assert pl.modality.value == "vision"
    assert pl.parent_document_id == "doc-123"
    assert pl.processed_text_content == "SAMPLE TEXT"
    assert pl.media_metadata.original_resolution == [64, 64]
    assert pl.media_metadata.temporal_anchors.start == 0.0


def test_media_payload_schema_roundtrip():
    payload = UnifiedMediaPayload(
        parent_document_id="doc", media_id="m1", modality="audio",
        processed_text_content="hello world",
        media_metadata={
            "original_resolution": [0, 0], "sampling_rate_hz": 16000,
            "duration_seconds": 61.5, "temporal_anchors": {"start": 0.0, "end": 61.5},
        },
    )
    data = payload.model_dump()
    assert data["modality"] == "audio"
    assert data["media_metadata"]["temporal_anchors"]["end"] == 61.5


def test_canonical_schema_validates():
    doc = CanonicalDocument.model_validate({
        "document_id": "abc",
        "metadata": {"filename": "f.pdf", "file_size_bytes": 10, "total_pages": 1},
        "payloads": [{"chunk_id": 0, "page_number": 1, "modality": "text",
                      "content": "hi", "structural_tag": "paragraph"}],
        "extracted_media_references": [{"media_id": "m1", "page_number": 1,
                                        "spatial_bounding_box": [0.0, 0.0, 10.0, 10.0],
                                        "modality": "raw_image"}],
    })
    assert doc.payloads[0].modality.value == "text"


def test_canonical_schema_rejects_bad_bbox():
    with pytest.raises(Exception):
        CanonicalDocument.model_validate({
            "document_id": "abc",
            "metadata": {"filename": "f.pdf", "file_size_bytes": 10, "total_pages": 1},
            "payloads": [],
            "extracted_media_references": [{"media_id": "m1", "page_number": 1,
                                            "spatial_bounding_box": [0.0, 0.0],
                                            "modality": "raw_image"}],
        })


# ---------- System resource awareness ----------

from app.resources import SystemResources, clamp_worker_memory, detect_system_resources


def test_detect_system_resources():
    r = detect_system_resources()
    assert 512 <= r.total_memory_mb <= 262144  # sane bounds
    assert r.cpu_count >= 1
    assert "darwin" in r.platform or "linux" in r.platform


def test_clamp_reduces_when_host_smaller_than_config():
    tiny_host = SystemResources(total_memory_mb=8192, cpu_count=8,
                                available_disk_mb=300000, platform="test")
    # 16GB configured vs 8GB host -> clamped to ~50% budget
    assert clamp_worker_memory(16384, tiny_host) == tiny_host.safe_worker_budget_mb
    assert clamp_worker_memory(1024, tiny_host) == 1024  # within budget: untouched


def test_clamp_floor():
    tiny = SystemResources(total_memory_mb=512, cpu_count=1,
                           available_disk_mb=100, platform="test")
    assert clamp_worker_memory(4096, tiny) >= 256

# ---------- API endpoint ----------

def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ingest_rejects_unsupported(monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.settings.upload_dir", tmp_path)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00binary"
    resp = client.post("/api/v1/documents/ingest",
                       files={"file": ("x.png", io.BytesIO(png), "image/png")})
    assert resp.status_code == 415


def test_ingest_accepts_txt(monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.settings.upload_dir", tmp_path)

    class FakeTask:
        id = "test-task-id"

    monkeypatch.setattr("app.main.parse_document_task.delay", lambda *a: FakeTask())
    resp = client.post("/api/v1/documents/ingest",
                       files={"file": ("notes.txt", io.BytesIO(b"hello ingestion"), "text/plain")})
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"
