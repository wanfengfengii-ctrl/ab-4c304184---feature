#!/usr/bin/env python3
"""One-shot verification run by the ``verify`` compose service.

It performs, in order, exiting non-zero on the first failure:

1. wait for the API ``/health`` endpoint to report healthy,
2. run the code test suite (pytest),
3. sanity-check the running build (importable app + dependency versions),
4. run a decode smoke test against the live API that exercises
   insertions, deletions and substitutions together, plus unique,
   ambiguous and infeasible responses;
5. repeat the decode smoke for terminal_mode=partial (proper suffix /
   prefix terminals with indel+substitution noise, locatable budget
   failure, and framing failure), and assert the legacy default keeps
   full-copy behaviour.
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
    step("decode smoke against live API (insertion + deletion + substitution)")

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
    return smoke_decode_partial()


def smoke_decode_partial() -> bool:
    """Live checks for terminal_mode=partial (truncated acquisition window)."""
    step("partial-terminal smoke against live API")

    # 20 nt reference; one-base terminal framing makes the cut point unique.
    ref = "ACGTACGATCGTACGATCAT"
    assert len(ref) == 20
    read = ref[19:] + ref + ref + ref[:1]  # 1 + 20 + 20 + 1 = 42 nt
    status, data = post(
        {
            "reference": ref,
            "read": read,
            "copies": 4,
            "max_edits": 0,
            "terminal_mode": "partial",
        }
    )
    print("partial clean status:", status)
    if status != 200 or data.get("status") != "unique":
        print("FAIL: partial clean case", status, data.get("status"))
        return False
    witness = data["witness"]
    if witness["terminal_ranges"] != {"first": [19, 20], "last": [0, 1]}:
        print(
            "FAIL: unexpected terminal ranges", witness["terminal_ranges"]
        )
        return False
    if witness["boundaries"] != [[0, 1], [1, 21], [21, 41], [41, 42]]:
        print("FAIL: unexpected boundaries", witness["boundaries"])
        return False
    # Replayable CIGARs must reconstruct each piece and its reference range.
    for seg in witness["segments"]:
        if seg["aligned_reference"].replace("-", "") != seg["reference"]:
            print("FAIL: terminal reference replay mismatch")
            return False
        if seg["aligned_read"].replace("-", "") != seg["read"]:
            print("FAIL: terminal read replay mismatch")
            return False
    print("partial clean case passed (proper suffix/prefix framing)")

    # Truncated ends carrying substitution + deletion + insertion.
    clean = ref[14:] + ref + ref[:8]  # 6 + 20 + 8 = 34 nt
    head, mid, tail = clean[:6], clean[6:26], clean[26:]
    head = "T" + head[1:]            # substitution in the suffix head
    mid = mid[:10] + mid[11:]        # deletion in the middle copy
    tail = tail[:4] + "G" + tail[4:]  # insertion in the prefix tail
    status, data = post(
        {
            "reference": ref,
            "read": head + mid + tail,
            "copies": 3,
            "max_edits": 1,
            "terminal_mode": "partial",
        }
    )
    if status != 200:
        print("FAIL: partial mixed-noise case", status)
        return False
    if data["objective"] != {"total_edits": 3, "max_segment_edits": 1}:
        print("FAIL: partial mixed-noise objective", data["objective"])
        return False
    witness = data.get("witness") or data["witnesses"][0]
    ops = {
        op for seg in witness["segments"] for op in seg["cigar"] if op.isalpha()
    }
    if not {"M", "D", "I"} <= ops:
        print("FAIL: partial CIGAR must contain M, D and I; got", ops)
        return False
    print(
        "partial mixed-noise case passed (substitution + deletion + "
        "insertion at truncated ends)"
    )

    # Budget failure must remain locatable (noise overflow, not truncation).
    status, data = post(
        {
            "reference": ref,
            "read": ref + ref[:19],  # 39 nt forces a 20-nt terminal piece
            "copies": 2,
            "max_edits": 0,
            "terminal_mode": "partial",
        }
    )
    if status != 422 or data.get("constraint", {}).get("name") != (
        "per_segment_edit_budget"
    ):
        print("FAIL: partial budget-failure case", status)
        return False
    bad = data["nearest"]["violating_segments"]
    if not bad or bad[0].get("required_edits") != 1:
        print("FAIL: partial budget failure not locatable")
        return False
    print("partial budget failure passed (located, distinguishable from noise)")

    # Framing impossibility: the window cannot form legal prefix/suffix.
    status, data = post(
        {
            "reference": ref,
            "read": "A" * 119,
            "copies": 3,
            "max_edits": 3,
            "terminal_mode": "partial",
        }
    )
    if status != 422 or data.get("constraint", {}).get("name") != (
        "terminal_prefix_suffix"
    ):
        print("FAIL: partial structural-failure case", status)
        return False
    print(
        "partial structural failure passed "
        "(terminal_prefix_suffix, distinct from budget overflow)"
    )

    # Legacy regression: omitting terminal_mode must behave exactly as full.
    status, data = post(
        {
            "reference": ref,
            "read": ref * 3,
            "copies": 3,
            "max_edits": 0,
        }
    )
    if (
        status != 200
        or data.get("request", {}).get("terminal_mode") != "full"
        or "terminal_ranges" in data.get("witness", {})
    ):
        print("FAIL: default terminal_mode regression", status)
        return False
    print("legacy default regression passed (terminal_mode defaults to full)")
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
    print("\n=== verify: ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
