"""Telemetry Stream Processor: min-max scaling for time-series and GPS vectors."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.tabular.parsers import SerializedRow, Table

logger = logging.getLogger(__name__)


@dataclass
class NormalizedRow:
    """A serialized row with min-max scaled numeric values."""
    row: SerializedRow
    scaled_values: dict[str, float]


@dataclass
class TelemetryBatch:
    rows: list[NormalizedRow]
    scaling_ranges: dict[str, tuple[float, float]]  # col -> (min, max) pre-scaling
    is_gps: bool


def _looks_like_gps(columns: list[str]) -> bool:
    lower = {c.lower() for c in columns}
    has_lat = bool({"lat", "latitude"} & lower)
    has_lon = bool({"lon", "lng", "longitude"} & lower)
    return has_lat and has_lon


def min_max_scale(value: float, lo: float, hi: float) -> float:
    """Scale `value` into [0, 1] using the column's observed min/max."""
    if hi - lo == 0.0:
        return 0.0
    scaled = (value - lo) / (hi - lo)
    return max(0.0, min(1.0, scaled))


def process_telemetry(table: Table, rows: list[SerializedRow]) -> TelemetryBatch:
    """Normalize numeric vectors with min-max scaling (per numeric column)."""
    numeric_cols = [
        c for c, dtype in table.frame.schema.items()
        if dtype.is_numeric()
    ]
    is_gps = _looks_like_gps(table.frame.columns)

    ranges: dict[str, tuple[float, float]] = {}
    for col in numeric_cols:
        series = table.frame[col].drop_nulls()
        if series.len() == 0:
            continue
        lo = float(series.min())
        hi = float(series.max())
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        ranges[col] = (lo, hi)

    normalized: list[NormalizedRow] = []
    for row in rows:
        scaled: dict[str, float] = {}
        for col, value in row.numeric_values.items():
            if col in ranges and value is not None and math.isfinite(value):
                lo, hi = ranges[col]
                scaled[col] = round(min_max_scale(value, lo, hi), 6)
        normalized.append(NormalizedRow(row=row, scaled_values=scaled))

    logger.info("Telemetry normalized", extra={
        "rows": len(normalized), "numeric_columns": len(ranges), "is_gps": is_gps,
    })
    return TelemetryBatch(rows=normalized, scaling_ranges=ranges, is_gps=is_gps)
