"""Phase 3 tabular engine: CSV/XLSX/JSON parsing with Polars + magic-byte detection."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import polars as pl

from app.config import settings

logger = logging.getLogger(__name__)


class TabularFormat(str, Enum):
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"


class UnsupportedTabularError(ValueError):
    """Raised when a file's magic bytes don't match CSV/XLSX/JSON."""


class TabularParseError(ValueError):
    """Raised when a table cannot be parsed (malformed content)."""


@dataclass
class Table:
    """A parsed table."""
    frame: pl.DataFrame
    fmt: TabularFormat
    source_filename: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame.shape


def detect_tabular_format(head: bytes, filename: str) -> TabularFormat:
    """Magic-byte detection for tabular uploads (CSV / XLSX / JSON)."""
    if head.startswith(b"PK\x03\x04"):
        # XLSX is an OPC zip; its [Content_Types].xml references spreadsheetml.
        if b"spreadsheetml" in head[:4096] or b"xl/" in head[:4096]:
            return TabularFormat.XLSX
        raise UnsupportedTabularError(f"File '{filename}' is a zip container, not XLSX.")
    try:
        text_head = head.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedTabularError(
            f"File '{filename}' is neither XLSX nor decodable tabular text."
        ) from exc
    stripped = text_head.lstrip()
    if stripped.startswith(("{", "[")):
        return TabularFormat.JSON
    if any(sep in text_head for sep in (",", ";", "\t", "|")) and "\n" in text_head:
        return TabularFormat.CSV
    raise UnsupportedTabularError(f"File '{filename}' does not look like CSV/JSON/XLSX.")


def parse_table(path: Path, fmt: TabularFormat, filename: str) -> Table:
    """Parse a tabular file into a Polars DataFrame (streaming-friendly)."""
    try:
        if fmt is TabularFormat.CSV:
            frame = pl.read_csv(path, infer_schema_length=1000, truncate_ragged_lines=True)
        elif fmt is TabularFormat.XLSX:
            frame = _read_xlsx(path)
        elif fmt is TabularFormat.JSON:
            frame = _read_json_records(path)
        else:  # pragma: no cover - guarded by detection
            raise TabularParseError(f"Unsupported tabular format '{fmt}'.")
    except (pl.exceptions.PolarsError, ValueError, OSError) as exc:
        raise TabularParseError(f"Failed to parse '{filename}': {exc}") from exc

    if frame.is_empty():
        raise TabularParseError(f"Table '{filename}' contains no data rows.")
    logger.info("Table parsed", extra={
        "file_name": filename, "format": fmt.value,
        "rows": frame.height, "columns": frame.width,
    })
    return Table(frame=frame, fmt=fmt, source_filename=filename)


def _read_xlsx(path: Path) -> pl.DataFrame:
    result = pl.read_excel(path, sheet_id=1)
    if isinstance(result, dict):
        return next(iter(result.values()))
    return result


def _read_json_records(path: Path) -> pl.DataFrame:
    """Accept a top-level records array and {"rows"/"data": [...]} wrappers."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("rows", "data", "records", "items"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            records = [data]
    else:
        raise TabularParseError("JSON must be an object or an array of records.")
    if not records:
        raise TabularParseError("JSON contains no records.")
    return pl.DataFrame(records)


@dataclass
class SerializedRow:
    row_index: int
    text: str
    timestamp: str | None
    numeric_values: dict[str, float] = field(default_factory=dict)
    point_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def serialize_rows(table: Table) -> list[SerializedRow]:
    """Semantic Matrix Serialization: rows -> declarative sentences.

    Telemetry-shaped rows (timestamp + sensor column + numeric value) render as:
      "At timestamp <ts>, the <sensor> sensor recorded a telemetry value of <v> units"
    All other rows render as "k1 is v1; k2 is v2; ...". Raw rows are never embedded.
    """
    frame = table.frame
    lower_cols = {c.lower(): c for c in frame.columns}
    ts_col = next((lower_cols[c] for c in ("timestamp", "time", "datetime") if c in lower_cols), None)
    sensor_col = next((lower_cols[c] for c in ("sensor", "sensor_id", "device", "name") if c in lower_cols), None)
    value_col = next((lower_cols[c] for c in ("value", "reading", "measurement") if c in lower_cols), None)

    rows: list[SerializedRow] = []
    for idx, row in enumerate(frame.iter_rows(named=True)):
        if ts_col and sensor_col and value_col is not None:
            ts = _stringify(row.get(ts_col))
            sensor = _stringify(row.get(sensor_col)) or "unknown"
            raw_value = row.get(value_col)
            value = _to_float(raw_value)
            text = (
                f"At timestamp {ts}, the {sensor} sensor recorded "
                f"a telemetry value of {_stringify(raw_value)} units"
            )
            numeric = {str(value_col): value} if value is not None else {}
            rows.append(SerializedRow(idx, text, ts, numeric))
        else:
            parts = [
                f"{_stringify(col)} is {_stringify(row[col])}"
                for col in frame.columns if row[col] is not None
            ]
            ts = _stringify(row.get(ts_col)) if ts_col else None
            text = "; ".join(parts) or "(empty row)"
            rows.append(SerializedRow(idx, text, ts))
    return rows


def _stringify(value) -> str:
    if value is None:
        return "null"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value).strip()


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def new_source_document_id(path: Path) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{path.stat().st_size}").hex
