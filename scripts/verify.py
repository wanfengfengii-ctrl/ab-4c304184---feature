#!/usr/bin/env python3
"""One-shot verification run by the ``verify`` compose service.

It performs, in order, exiting non-zero on the first failure:

1. wait for the API ``/health`` endpoint to report healthy,
2. run the code test suite (pytest),
3. sanity-check the running build (importable app + dependency versions),
4. run decode smoke tests against the live API:
   - legacy full-mode regression (unique / ambiguous / infeasible),
   - partial-terminal decoding of a window that starts and ends mid-unit,
     including a case with insertion + deletion + substitution noise,
   - partial-terminal geometric truncation failure (``terminal_range``),
   - a clean re-run that confirms deterministic, reproducible output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = os.environ.get("API_BASE", "http://api:8000").rstrip("/")
HEALTH_URL = f"{API}/health"
DECODE_URL = f"{API}/api/concatemers/decode"

# Repository root (this file lives in <root>/scripts/verify.py).  In the
# image the tree is copied under /srv, so the same derivation holds there.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(title: str) -> None:
    print(f"\n=== verify: {title} ===", flush=True)


def wait_for_health(timeout: float = 60.0) -> bool:
    step(f"waiting for API health at {HEALTH_URL}")
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode())
                    if body.get("status") == "ok":
                        print("health ok:", body)
                        return True
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
        time.sleep(1.0)
    print(f"health check timed out after {timeout:.0f}s: {last_error}")
    return False


def run_tests() -> bool:
    step("running code tests (pytest)")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
    )
    if proc.returncode != 0:
        print("pytest failed")
        return False
    print("pytest passed")
    return True


def check_build() -> bool:
    step("sanity-checking the running build")
    code = (
        "import fastapi, uvicorn, pydantic; "
        "from app.main import app; "
        "print('fastapi', fastapi.__version__, "
        "'uvicorn', uvicorn.__version__, "
        "'pydantic', pydantic.VERSION, "
        "'routes', len(app.routes))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    if proc.returncode != 0:
        print("build sanity check failed")
        return False
    print("build sanity check passed")
    return True


def post(payload: dict):
    req = urllib.request.Request(
        DECODE_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def smoke_decode() -> bool:
    step("decode smoke against live API (legacy full mode)")

    # 10 nt reference; three copies with one edit of each kind.
    ref = "ACGTACGATC"
    read = (
        "ATGTACGATC"     # substitution (C -> T)
        + "ACGTACGAT"    # deletion of trailing C
        + "ACGTACGATCA"  # insertion of A
    )
    assert len(read) == 30
    status, data = post(
        {"reference": ref, "read": read, "copies": 3, "max_edits": 1}
    )
    print("mixed-noise status:", status)
    print(json.dumps(data, indent=2)[:1600])
    if status != 200:
        print("FAIL: expected HTTP 200")
        return False
    if data["objective"] != {"total_edits": 3, "max_segment_edits": 1}:
        print("FAIL: unexpected objective", data["objective"])
        return False
    witness = data.get("witness") or data["witnesses"][0]
    ops = {
        op
        for seg in witness["segments"]
        for op in seg["cigar"]
        if op.isalpha()
    }
    if not {"M", "D", "I"} <= ops:
        print("FAIL: CIGAR must contain M, D and I; got", ops)
        return False
    # The replay rows must reconstruct both original strings.
    for seg in witness["segments"]:
        if seg["aligned_reference"].replace("-", "") != seg["reference"]:
            print("FAIL: reference replay mismatch")
            return False
        if seg["aligned_read"].replace("-", "") != seg["read"]:
            print("FAIL: read replay mismatch")
            return False
    print("mixed-noise smoke passed (substitution + deletion + insertion)")

    # Unique clean case.
    status, data = post(
        {"reference": ref, "read": ref * 3, "copies": 3, "max_edits": 0}
    )
    if status != 200 or data["status"] != "unique" or data["objective"][
        "total_edits"
    ] != 0:
        print("FAIL: clean unique case", status, data.get("status"))
        return False
    print("unique case passed")

    # Ambiguous case (homopolymer shifts).
    status, data = post(
        {
            "reference": "AAAAAAAAAA",
            "read": "A" * 30,
            "copies": 3,
            "max_edits": 0,
        }
    )
    if (
        status != 200
        or data["status"] != "ambiguous"
        or len(data["witnesses"]) != 2
    ):
        print("FAIL: ambiguous case", status, data.get("status"))
        return False
    print("ambiguous case passed (two witnesses returned)")

    # Infeasible case must be locatable (422 + violating segment).
    status, data = post(
        {
            "reference": ref,
            "read": "TTGTACGATC" + ref * 2,
            "copies": 3,
            "max_edits": 1,
        }
    )
    if status != 422 or data.get("status") != "infeasible":
        print("FAIL: infeasible case", status, data.get("status"))
        return False
    bad = data["nearest"]["violating_segments"]
    if not bad or bad[0]["required_edits"] != 2:
        print("FAIL: infeasible case not locatable", data.get("nearest"))
        return False
    print("infeasible case passed (violating segment located)")

    # Regression: omitting terminal_mode is exactly full mode.
    status_default, data_default = post(
        {"reference": ref, "read": ref * 3, "copies": 3, "max_edits": 0}
    )
    status_explicit, data_explicit = post(
        {
            "reference": ref,
            "read": ref * 3,
            "copies": 3,
            "max_edits": 0,
            "terminal_mode": "full",
        }
    )
    if (
        status_default != 200
        or data_default["request"]["terminal_mode"] != "full"
        or data_default["objective"] != data_explicit["objective"]
    ):
        print("FAIL: omitted terminal_mode is not equivalent to 'full'")
        return False
    print("legacy default compatibility passed")
    return True


def smoke_partial() -> bool:
    step("partial-terminal decode smoke (window starts/ends mid-unit)")

    # Clean truncated read rescued only by partial mode: extreme cuts so the
    # interpretation is unique.  ref[9:] (1 nt) + 3 full units + ref[:1].
    ref = "ACGTACGATT"  # 10 nt
    read = ref[9:] + ref * 3 + ref[:1]  # 32 nt
    payload = {
        "reference": ref,
        "read": read,
        "copies": 5,
        "max_edits": 0,
    }

    # Legacy full mode must still reject it (backward compatibility).
    status, data = post(payload)
    if status != 422 or data.get("status") != "infeasible":
        print("FAIL: legacy mode should reject truncated read", status)
        return False

    payload["terminal_mode"] = "partial"
    status, data = post(payload)
    print("partial clean status:", status)
    if status != 200 or data["status"] != "unique":
        print("FAIL: partial clean case", status, data.get("status"))
        return False
    clean_body = data
    witness = data["witness"]
    if witness["terminal_ranges"] != {
        "first_suffix": [9, 10],
        "last_prefix": [0, 1],
    }:
        print("FAIL: unexpected terminal ranges", witness.get("terminal_ranges"))
        return False
    first, last = witness["segments"][0], witness["segments"][-1]
    if first["role"] != "suffix" or last["role"] != "prefix":
        print("FAIL: terminal segment roles missing")
        return False
    for seg in witness["segments"]:
        if seg["aligned_reference"].replace("-", "") != seg["reference"]:
            print("FAIL: partial reference replay mismatch")
            return False
        if seg["aligned_read"].replace("-", "") != seg["read"]:
            print("FAIL: partial read replay mismatch")
            return False
    print("partial clean case passed (terminal ranges + replayable CIGARs)")

    # Partial-terminal read with substitution + deletion + insertion noise;
    # objective (3, 1) and the witnesses collectively use M/D/I.
    noisy = {
        "reference": "GTCGAGCGACGG",
        "read": "GAGCGACAGGTCGAGCGACGGGTCAGCGACGGGTCGAGACGA",
        "copies": 4,
        "max_edits": 1,
        "terminal_mode": "partial",
    }
    status, data = post(noisy)
    print("partial noisy status:", status)
    if status != 200:
        print("FAIL: partial noisy case", status)
        return False
    if data["objective"] != {"total_edits": 3, "max_segment_edits": 1}:
        print("FAIL: partial noisy objective", data["objective"])
        return False
    witnesses = data.get("witnesses") or [data["witness"]]
    ops = {
        op
        for w in witnesses
        for seg in w["segments"]
        for op in seg["cigar"]
        if op.isalpha()
    }
    if not {"M", "D", "I"} <= ops:
        print("FAIL: partial CIGAR must contain M, D and I; got", ops)
        return False
    for w in witnesses:
        assert "first_suffix" in w["terminal_ranges"]
        assert "last_prefix" in w["terminal_ranges"]
    print("partial indel/substitution case passed")

    # Ambiguous partial case: stable first two witnesses are returned.
    status, data = post(
        {
            "reference": "AAAAAAAAAA",
            "read": "A" * 30,
            "copies": 4,
            "max_edits": 0,
            "terminal_mode": "partial",
        }
    )
    if (
        status != 200
        or data["status"] != "ambiguous"
        or len(data["witnesses"]) != 2
    ):
        print("FAIL: partial ambiguous case", status, data.get("status"))
        return False
    print("partial ambiguous case passed (two stable witnesses)")

    # Truncation geometry failure: read longer than any budgeted partial
    # cover, but structurally conceivable cap-free -> terminal_range.
    status, data = post(
        {
            "reference": "ACGTACGT",
            "read": "A" * 30,
            "copies": 3,
            "max_edits": 1,
            "terminal_mode": "partial",
        }
    )
    print("partial geometry status:", status)
    if status != 422 or data["constraint"]["name"] != "terminal_range":
        print(
            "FAIL: expected terminal_range failure, got",
            status,
            data.get("constraint", {}).get("name"),
        )
        return False
    blockers = [
        v
        for v in data["nearest"]["violating_segments"]
        if v["geometric_terminal_failure"]
    ]
    if not blockers or blockers[0]["role"] not in ("suffix", "prefix"):
        print("FAIL: terminal_range failure not locatable")
        return False
    print("terminal_range failure passed (distinguished from noise)")

    # Noise failure on a geometrically legal read stays a budget failure.
    status, data = post(
        {
            "reference": "ACGTACGTTAA",
            "read": "GGGGG" + "ACGTACGTTAA" * 2 + "ACGT",
            "copies": 4,
            "max_edits": 1,
            "terminal_mode": "partial",
        }
    )
    if status != 422 or data["constraint"]["name"] != "per_segment_edit_budget":
        print(
            "FAIL: expected per_segment_edit_budget, got",
            status,
            data.get("constraint", {}).get("name"),
        )
        return False
    print("budget failure case passed (noise distinguished from truncation)")

    # Deterministic clean re-run: same clean payload returns an identical
    # response body.
    status2, data2 = post(payload)
    if status2 != 200 or data2 != clean_body:
        print("FAIL: partial re-run is not deterministic")
        return False
    print("clean re-run passed (deterministic)")
    return True


def main() -> int:
    if not wait_for_health():
        return 1
    if not run_tests():
        return 1
    if not check_build():
        return 1
    if not smoke_decode():
        return 1
    if not smoke_partial():
        return 1
    print("\n=== verify: ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
