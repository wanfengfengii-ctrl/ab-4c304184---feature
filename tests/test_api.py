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


# ------------------------------------------------------------- partial mode


def test_decode_partial_unique_exposes_terminal_ranges():
    ref = "ACGTACGATT"  # 10 nt
    # Extreme cuts, copies=5 -> read is 32 nt (API needs >= 30).
    read = ref[9:] + ref * 3 + ref[:1]
    body = {
        "reference": ref,
        "read": read,
        "copies": 5,
        "max_edits": 0,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["terminal_mode"] == "partial"
    assert data["status"] == "unique"
    assert data["request"]["terminal_mode"] == "partial"
    witness = data["witness"]
    assert witness["terminal_ranges"] == {
        "first_suffix": [9, 10],
        "last_prefix": [0, 1],
    }
    first = witness["segments"][0]
    last = witness["segments"][-1]
    assert first["role"] == "suffix"
    assert first["reference"] == ref[9:]
    assert first["cigar"] == "1M"
    assert last["role"] == "prefix"
    assert last["reference"] == ref[:1]
    # CIGARs remain replayable.
    for seg in witness["segments"]:
        assert seg["aligned_reference"].replace("-", "") == seg["reference"]
        assert seg["aligned_read"].replace("-", "") == seg["read"]


def test_decode_partial_indel_sub_noise():
    body = {
        "reference": "GTCGAGCGACGG",
        "read": "GAGCGACAGGTCGAGCGACGGGTCAGCGACGGGTCGAGACGA",
        "copies": 4,
        "max_edits": 1,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objective"] == {"total_edits": 3, "max_segment_edits": 1}
    witnesses = data.get("witnesses") or [data["witness"]]
    kinds = set()
    for w in witnesses:
        for seg in w["segments"]:
            kinds.update(op for op in seg["cigar"] if op.isalpha())
    assert {"M", "D", "I"} <= kinds


def test_decode_partial_ambiguous_returns_two_witnesses():
    body = {
        "reference": "AAAAAAAAAA",
        "read": "A" * 30,
        "copies": 4,
        "max_edits": 0,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ambiguous"
    assert len(data["witnesses"]) == 2
    assert data["more_witnesses"] is True


def test_decode_partial_rescues_legacy_failure():
    ref = "ACGTACGATT"
    read = ref[9:] + ref * 3 + ref[:1]
    body = {
        "reference": ref,
        "read": read,
        "copies": 5,
        "max_edits": 0,
    }
    # Legacy full mode rejects the truncated read.
    resp_full = client.post("/api/concatemers/decode", json=body)
    assert resp_full.status_code == 422
    # Partial mode decodes it.
    body["terminal_mode"] = "partial"
    resp_partial = client.post("/api/concatemers/decode", json=body)
    assert resp_partial.status_code == 200
    assert resp_partial.json()["status"] == "unique"


def test_decode_partial_terminal_range_failure_is_locatable():
    body = {
        "reference": "ACGTACGT",
        "read": "A" * 30,  # beyond budgeted partial geometry
        "copies": 3,
        "max_edits": 1,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "infeasible"
    assert data["constraint"]["name"] == "terminal_range"
    blockers = [
        v
        for v in data["nearest"]["violating_segments"]
        if v["geometric_terminal_failure"]
    ]
    assert blockers
    assert blockers[0]["role"] in ("suffix", "prefix")


def test_decode_partial_noise_failure_is_budget_not_geometry():
    body = {
        "reference": "ACGTACGTTAA",  # 11 nt
        # 33 nt: legal-length partial termini but the first is pure noise.
        "read": "GGGGG" + "ACGTACGTTAA" * 2 + "ACGT",
        "copies": 4,
        "max_edits": 1,
        "terminal_mode": "partial",
    }
    assert 30 <= len(body["read"]) <= 160
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422
    data = resp.json()
    assert data["constraint"]["name"] == "per_segment_edit_budget"
    assert data["nearest"]["violating_segments"]


def test_validation_terminal_mode():
    body = {
        "reference": "ACGTACGAT",
        "read": "A" * 30,
        "copies": 3,
        "max_edits": 0,
        "terminal_mode": "sideways",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_terminal_mode_default_is_full():
    ref = "ACGTACGATC"  # 10 nt -> 30 nt read, API-valid
    body = {"reference": ref, "read": ref * 3, "copies": 3, "max_edits": 0}
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200
    assert resp.json()["request"]["terminal_mode"] == "full"
