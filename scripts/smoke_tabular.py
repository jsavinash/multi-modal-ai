"""End-to-end Phase 3 smoke: CSV telemetry -> serialization -> scaling -> embed -> spool."""
import logging
import tempfile
from pathlib import Path

from app.logging_config import configure_logging

configure_logging()

from app.tabular.indexing import run_indexing  # noqa: E402
from app.tabular.parsers import TabularFormat  # noqa: E402

CSV = (
    "Timestamp,Sensor,Value\n"
    "2026-09-21T10:00:00,Temp,23.5\n"
    "2026-09-21T10:01:00,Temp,71.2\n"
    "2026-09-21T10:02:00,Temp,12.9\n"
    "2026-09-21T10:03:00,Temp,55.0\n"
)

GPS = (
    "Timestamp,Latitude,Longitude\n"
    "2026-09-21T10:00:00,52.5200,13.4050\n"
    "2026-09-21T10:01:00,48.1351,11.5820\n"
)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # 1. Telemetry CSV: semantic serialization + min-max scaling + embedding + spool
    p = tmp_path / "telemetry.csv"
    p.write_text(CSV, encoding="utf-8")
    summary = run_indexing(p, TabularFormat.CSV, "telemetry.csv")
    logging.getLogger("smoke").info("telemetry summary: %s", summary)
    assert summary["rows_indexed"] == 4
    assert summary["table_shape"] == {"rows": 4, "columns": 3}
    assert summary["store_mode"] == "spooled"  # no Qdrant on this host
    assert summary["scaling_ranges"]["Value"][0] == 12.9

    # 2. GPS array: gps_record flag + per-column scaling
    g = tmp_path / "gps.csv"
    g.write_text(GPS, encoding="utf-8")
    gps_summary = run_indexing(g, TabularFormat.CSV, "gps.csv")
    assert gps_summary["is_gps"] is True
    logging.getLogger("smoke").info("gps summary: rows=%d gps=%s",
                                    gps_summary["rows_indexed"], gps_summary["is_gps"])

logging.getLogger("smoke").info("PHASE 3 SMOKE OK")
