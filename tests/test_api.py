"""HTTP-level tests for the decode API."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_decode_unique():
    ref = "ACGTACGAT"
    body = {
        "reference": ref,
        "read": ref * 4,
        "copies": 4,
        "max_edits": 0,
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unique"
    assert data["objective"] == {"total_edits": 0, "max_segment_edits": 0}
    assert data["rotated_reference"] == ref
    seg = data["witness"]["segments"][0]
    assert seg["cigar"] == "9M"
    # replay rows are present and consistent
    assert seg["aligned_reference"] == seg["aligned_read"] == ref


def test_decode_with_indel_and_substitution_smoke():
    # substitution + deletion + insertion across three copies (30 nt total)
    ref = "ACGTACGATC"  # 10 nt
    read = (
        "ATGTACGATC"    # C -> T substitution, 10 nt
        + "ACGTACGAT"   # trailing C deleted, 9 nt
        + "ACGTACGATCA"  # A inserted, 11 nt
    )
    assert len(read) == 30
    body = {"reference": ref, "read": read, "copies": 3, "max_edits": 1}
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objective"] == {"total_edits": 3, "max_segment_edits": 1}
    witness = data.get("witness") or data["witnesses"][0]
    lengths = [e - b for b, e in witness["boundaries"]]
    assert lengths == [10, 9, 11]
    kinds = {op for s in witness["segments"] for op in s["cigar"] if op.isalpha()}
    assert {"M", "D", "I"} <= kinds


def test_decode_ambiguous_returns_two_witnesses():
    body = {
        "reference": "AAAAAAAAAA",
        "read": "A" * 30,
        "copies": 3,
        "max_edits": 0,
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ambiguous"
    assert len(data["witnesses"]) == 2
    assert data["witnesses"][0]["shift"] < data["witnesses"][1]["shift"]


def test_decode_infeasible_422_with_location():
    body = {
        "reference": "ACGTACGATC",
        # first copy carries two substitutions (A->T, C->T)
        "read": "TTGTACGATC" + "ACGTACGATC" * 2,
        "copies": 3,
        "max_edits": 1,
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "infeasible"
    assert data["error"] == "constraint_failed"
    bad = data["nearest"]["violating_segments"]
    assert bad and bad[0]["required_edits"] == 2


def test_validation_reference_length():
    body = {"reference": "ACGT", "read": "A" * 30, "copies": 3, "max_edits": 0}
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_validation_read_length():
    body = {"reference": "ACGTACGT", "read": "A" * 20, "copies": 3, "max_edits": 0}
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_validation_copies_range():
    body = {
        "reference": "ACGTACGT",
        "read": "A" * 30,
        "copies": 9,
        "max_edits": 0,
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_validation_edits_range():
    body = {
        "reference": "ACGTACGT",
        "read": "A" * 30,
        "copies": 3,
        "max_edits": 4,
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_validation_bad_characters():
    body = {
        "reference": "ACGTACGN",
        "read": "A" * 30,
        "copies": 3,
        "max_edits": 0,
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_lowercase_is_normalized():
    ref = "acgtacgatc"
    body = {"reference": ref, "read": ref.upper() * 3, "copies": 3, "max_edits": 0}
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200
    assert resp.json()["status"] == "unique"
