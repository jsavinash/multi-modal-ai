"""End-to-end smoke test: craft a minimal PDF, run the full parse pipeline (no Redis needed)."""
import json
import logging
import pathlib

from app.logging_config import configure_logging

configure_logging()

# --- craft a minimal valid single-page PDF with header/body/footer text ---
content = (b"BT /F1 12 Tf 72 740 Td (Document Header Title) Tj "
           b"0 -600 Td (The quick brown fox jumps over the lazy dog repeatedly for hours. ) Tj "
           b"0 -80 Td (Page Footer 1) Tj ET")
objs = [
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
    b"/Resources<</Font<</F1 5 0 R>>>>>>endobj",
    b"4 0 obj<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"\nendstream endobj",
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj",
]
out = b"%PDF-1.4\n"
offsets: list[int] = []
for obj in objs:
    offsets.append(len(out))
    out += obj + b"\n"
xref_pos = len(out)
out += b"xref\n0 6\n0000000000 65535 f \n"
out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
out += b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"
p = pathlib.Path("/tmp/smoke.pdf")
p.write_bytes(out)

from app.magic import SupportedFormat, detect_format  # noqa: E402
from app.tasks import _parse_document  # noqa: E402

head = p.open("rb").read(4096)
sig = detect_format(head, "smoke.pdf")
assert sig.fmt is SupportedFormat.PDF, sig

doc = _parse_document(p, SupportedFormat.PDF, "smoke-123")
rendered = json.loads(doc.model_dump_json())
logging.getLogger("smoke").info("canonical output: %s", json.dumps(rendered, indent=2))
assert rendered["metadata"]["total_pages"] == 1
tags = {pl["structural_tag"] for pl in rendered["payloads"]}
assert "header" in tags and "footer" in tags and "paragraph" in tags, tags
logging.getLogger("smoke").info("SMOKE OK: %d payloads, tags=%s", len(rendered["payloads"]), tags)
