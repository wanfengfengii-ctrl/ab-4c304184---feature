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


# ------------------------------------------------------ terminal_mode=partial


REF20 = "ACGTACGATCGTACGATCAT"


def test_decode_partial_unique_with_terminal_ranges():
    ref = REF20  # 20 nt; one-base framing of 4 pieces gives a 42-nt read
    read = ref[19:] + ref + ref + ref[:1]
    body = {
        "reference": ref,
        "read": read,
        "copies": 4,
        "max_edits": 0,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["request"]["terminal_mode"] == "partial"
    assert data["status"] == "unique"
    w = data["witness"]
    assert w["terminal_mode"] == "partial"
    assert w["terminal_ranges"] == {"first": [19, 20], "last": [0, 1]}
    assert w["boundaries"] == [[0, 1], [1, 21], [21, 41], [41, 42]]
    for seg in w["segments"]:
        assert "cigar" in seg and "aligned_reference" in seg
        assert seg["aligned_read"].replace("-", "") == seg["read"]
        assert seg["aligned_reference"].replace("-", "") == seg["reference"]


def test_decode_partial_indel_sub_noise():
    ref = REF20
    clean = ref[14:] + ref + ref[:8]  # 6 + 20 + 8 = 34 nt
    head, mid, tail = clean[:6], clean[6:26], clean[26:]
    head = "T" + head[1:]              # substitution
    mid = mid[:10] + mid[11:]          # deletion
    tail = tail[:4] + "G" + tail[4:]   # insertion
    body = {
        "reference": ref,
        "read": head + mid + tail,
        "copies": 3,
        "max_edits": 1,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objective"] == {"total_edits": 3, "max_segment_edits": 1}
    witness = data.get("witness") or data["witnesses"][0]
    ops = {
        op for seg in witness["segments"] for op in seg["cigar"] if op.isalpha()
    }
    assert {"M", "D", "I"} <= ops


def test_decode_partial_allows_two_copies():
    ref = REF20
    body = {
        "reference": ref,
        "read": ref[6:] + ref[:16],  # 14 + 16 = 30 nt, two terminal pieces
        "copies": 2,
        "max_edits": 0,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] in ("unique", "ambiguous")
    witness = data.get("witness") or data["witnesses"][0]
    assert [s.get("role") for s in witness["segments"]] == ["first", "last"]


def test_decode_full_mode_still_rejects_two_copies():
    body = {
        "reference": REF20,
        "read": "A" * 30,
        "copies": 2,
        "max_edits": 0,
    }
    # Legacy behaviour without terminal_mode must be unchanged.
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422
    body["terminal_mode"] = "full"
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_decode_omitting_terminal_mode_defaults_to_full():
    ref = "ACGTACGATC"
    body = {"reference": ref, "read": ref * 3, "copies": 3, "max_edits": 0}
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request"]["terminal_mode"] == "full"
    assert "terminal_ranges" not in data["witness"]


def test_decode_partial_mode_is_case_insensitive():
    ref = REF20
    body = {
        "reference": ref,
        "read": ref[19:] + ref + ref + ref[:1],
        "copies": 4,
        "max_edits": 0,
        "terminal_mode": "PARTIAL",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 200
    assert resp.json()["request"]["terminal_mode"] == "partial"


def test_decode_rejects_unknown_terminal_mode():
    body = {
        "reference": "ACGTACGATC",
        "read": "A" * 30,
        "copies": 3,
        "max_edits": 0,
        "terminal_mode": "halfway",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422


def test_decode_partial_budget_failure_is_locatable():
    ref = REF20
    read = ref + ref[:19]  # 39 nt forces a full-span (20) terminal piece
    body = {
        "reference": ref,
        "read": read,
        "copies": 2,
        "max_edits": 0,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "infeasible"
    assert data["terminal_mode"] == "partial"
    assert data["constraint"]["name"] == "per_segment_edit_budget"
    bad = data["nearest"]["violating_segments"]
    assert bad and bad[0]["required_edits"] == 1
    assert bad[0]["role"] in ("first", "last")


def test_decode_partial_structural_failure_locatable():
    body = {
        "reference": "ACGTACGATCGTACGATCAT",
        "read": "A" * 119,
        "copies": 3,
        "max_edits": 3,
        "terminal_mode": "partial",
    }
    resp = client.post("/api/concatemers/decode", json=body)
    assert resp.status_code == 422
    data = resp.json()
    assert data["constraint"]["name"] == "terminal_prefix_suffix"
    assert data["nearest"] is None
