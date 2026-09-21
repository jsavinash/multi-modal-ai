"""Phase 3 tests: tabular parsing, serialization, telemetry scaling, embeddings."""

import pytest

CSV_TELEMETRY = (
    "Timestamp,Sensor,Value\n"
    "2026-09-21T10:00:00,Temp,23.5\n"
    "2026-09-21T10:01:00,Temp,50.0\n"
    "2026-09-21T10:02:00,Pressure,101.3\n"
)
CSV_GPS = (
    "Timestamp,Latitude,Longitude\n"
    "2026-09-21T10:00:00,52.5200,13.4050\n"
    "2026-09-21T10:01:00,48.1351,11.5820\n"
)


def _write_csv(tmp_path, text: str):
    p = tmp_path / "telemetry.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_detect_tabular_formats(tmp_path):
    from app.tabular.parsers import TabularFormat, detect_tabular_format
    assert detect_tabular_format(CSV_TELEMETRY.encode(), "t.csv") is TabularFormat.CSV
    assert detect_tabular_format(b'[{"a": 1}]', "t.json") is TabularFormat.JSON
    xlsx_head = b"PK\x03\x04" + b" " * 100 + b"xl/worksheets"
    assert detect_tabular_format(xlsx_head, "t.xlsx") is TabularFormat.XLSX
    from app.tabular.parsers import UnsupportedTabularError
    with pytest.raises(UnsupportedTabularError):
        detect_tabular_format(b"\x00\x01\x02binary\x00", "x.bin")


def test_parse_and_serialize_telemetry(tmp_path):
    from app.tabular.parsers import TabularFormat, parse_table, serialize_rows
    p = _write_csv(tmp_path, CSV_TELEMETRY)
    table = parse_table(p, TabularFormat.CSV, "telemetry.csv")
    assert table.shape == (3, 3)
    rows = serialize_rows(table)
    assert rows[0].text == (
        "At timestamp 2026-09-21T10:00:00, the Temp sensor recorded "
        "a telemetry value of 23.5 units"
    )
    assert rows[0].timestamp == "2026-09-21T10:00:00"
    assert rows[0].numeric_values == {"Value": 23.5}


def test_serialize_generic_rows(tmp_path):
    from app.tabular.parsers import TabularFormat, parse_table, serialize_rows
    p = _write_csv(tmp_path, "Event,Region,Score\nlogin,eu-west,0.98\n")
    rows = serialize_rows(parse_table(p, TabularFormat.CSV, "events.csv"))
    assert rows[0].text == "Event is login; Region is eu-west; Score is 0.98"


def test_min_max_scaling(tmp_path):
    from app.tabular.parsers import TabularFormat, parse_table, serialize_rows
    from app.tabular.telemetry import min_max_scale, process_telemetry
    assert min_max_scale(25.0, 20.0, 30.0) == 0.5
    assert min_max_scale(20.0, 20.0, 30.0) == 0.0
    assert min_max_scale(30.0, 20.0, 30.0) == 1.0
    assert min_max_scale(5.0, 5.0, 5.0) == 0.0  # degenerate range
    csv = "Timestamp,Sensor,Value\n" + "".join(
        f"2026-09-21T10:0{i}:00,Temp,{v}\n" for i, v in enumerate((10.0, 20.0, 30.0))
    )
    p = _write_csv(tmp_path, csv)
    table = parse_table(p, TabularFormat.CSV, "t.csv")
    batch = process_telemetry(table, serialize_rows(table))
    assert [r.scaled_values["Value"] for r in batch.rows] == [0.0, 0.5, 1.0]
    assert batch.scaling_ranges["Value"] == (10.0, 30.0)


def test_gps_detection(tmp_path):
    from app.tabular.parsers import TabularFormat, parse_table, serialize_rows
    from app.tabular.telemetry import process_telemetry
    p = _write_csv(tmp_path, CSV_GPS)
    batch = process_telemetry(parse_table(p, TabularFormat.CSV, "gps.csv"),
                              serialize_rows(parse_table(p, TabularFormat.CSV, "gps.csv")))
    assert batch.is_gps is True


def test_embedding_fallback_deterministic():
    from app.embeddings import embedding_engine
    text = "At timestamp 2026, the Temp sensor recorded a telemetry value of 23.5 units"
    v1 = embedding_engine.embed_batch([text])
    v2 = embedding_engine.embed_batch([text])
    assert len(v1[0]) == 384
    assert v1 == v2  # deterministic
    norm = sum(x * x for x in v1[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3  # normalized


def test_vector_schema_validation():
    from app.vector_schema import (
        COLLECTIONS, IndexPayload, SchemaValidationError, make_payload,
        validate_payload_for_collection,
    )
    import random
    vector = [round(random.uniform(-1, 1), 4) for _ in range(384)]

    ok = make_payload("tabular", "doc-1", "text", "2026-09-21T10:00:00")
    ok.vector = vector
    validate_payload_for_collection(ok)  # should pass

    bad_dim = IndexPayload(collection=COLLECTIONS["tabular"], vector=[0.0] * 7, payload={
        "document_id": "d", "modality": "tabular", "timestamp": None, "text": "x"})
    with pytest.raises(SchemaValidationError):
        validate_payload_for_collection(bad_dim)

    bad_type = IndexPayload(collection=COLLECTIONS["tabular"], vector=vector, payload={
        "document_id": 123, "modality": "tabular", "timestamp": None, "text": "x"})
    with pytest.raises(SchemaValidationError):
        validate_payload_for_collection(bad_type)

    missing = IndexPayload(collection=COLLECTIONS["tabular"], vector=vector, payload={
        "modality": "tabular", "timestamp": None, "text": "x"})
    with pytest.raises(SchemaValidationError):
        validate_payload_for_collection(missing)

    unknown = IndexPayload(collection="nope", vector=vector, payload={})
    with pytest.raises(SchemaValidationError):
        validate_payload_for_collection(unknown)


def test_batch_upsert_splits_at_64(monkeypatch):
    from app import vector_store
    from app.vector_schema import make_payload
    import random
    vector = [round(random.uniform(-1, 1), 4) for _ in range(384)]
    payloads = []
    for i in range(130):  # 130 -> batches of 64, 64, 2
        p = make_payload("tabular", "doc-batch", f"row {i}", None)
        p.vector = vector
        payloads.append(p)

    calls: list[int] = []

    class FakeClient:
        def upsert(self, collection_name, points, wait):
            calls.append(len(points))

    monkeypatch.setattr(vector_store, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(vector_store, "_QDRANT_IMPORT_OK", True)
    written = vector_store.upsert_batch(payloads, "doc-batch")
    assert written == 130
    assert calls == [64, 64, 2]


def test_retry_logic_backoff(monkeypatch):
    from app import vector_store
    from app.vector_store import VectorStoreUnavailableError
    monkeypatch.setattr(vector_store.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def always_fails():
        attempts["n"] += 1
        raise ConnectionError("network down")

    with pytest.raises(VectorStoreUnavailableError):
        vector_store._with_retries("op", always_fails)
    assert attempts["n"] == 5  # qdrant_max_retries

    recovered = {"n": 0}

    def flaky_then_ok():
        recovered["n"] += 1
        if recovered["n"] < 3:
            raise ConnectionError("blip")
        return "ok"

    assert vector_store._with_retries("op", flaky_then_ok) == "ok"


def test_offline_spool(monkeypatch, tmp_path):
    from app import vector_store
    from app.vector_schema import make_payload
    import random
    monkeypatch.setattr("app.config.settings.result_dir", tmp_path)
    vector = [round(random.uniform(-1, 1), 4) for _ in range(384)]
    p = make_payload("tabular", "doc-off", "row", None)
    p.vector = vector
    dest = vector_store.spool_offline([p], "doc-off")
    assert dest.exists() and dest.suffix == ".json"


def test_run_indexing_end_to_end_spooled(monkeypatch, tmp_path):
    """Full Phase 3 pipeline with Qdrant unreachable -> offline spool."""
    from app import vector_store
    from app.tabular.indexing import run_indexing
    from app.tabular.parsers import TabularFormat
    monkeypatch.setattr("app.config.settings.result_dir", tmp_path)

    def unreachable():
        raise vector_store.VectorStoreUnavailableError("no qdrant in test")
    monkeypatch.setattr(vector_store, "ensure_collections", unreachable)

    p = _write_csv(tmp_path, CSV_TELEMETRY)
    summary = run_indexing(p, TabularFormat.CSV, "telemetry.csv")
    assert summary["rows_indexed"] == 3
    assert summary["store_mode"] == "spooled"
    assert summary["spool_path"] is not None
    assert summary["embedding_backend"] in ("minilm", "hash_fallback")
    assert summary["table_shape"] == {"rows": 3, "columns": 3}
