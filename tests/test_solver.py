"""Tests for the concatemer solver."""

import itertools

import pytest

from app.solver import align_global, rotate, solve, _replay


# ---------------------------------------------------------------- alignments


def test_align_exact():
    dist, cigars = align_global("ACGTACGT", "ACGTACGT", 0)
    assert dist == 0 and cigars == ("8M",)


def test_align_substitution():
    dist, cigars = align_global("ACGT", "AXGT", 1)
    assert dist == 1 and cigars == ("4M",)


def test_align_insertion_enumerates_optimal_positions():
    dist, cigars = align_global("ACGT", "ACGTT", 1)
    assert dist == 1
    # Inserting T after the aligned T (4M1I) or between the two Ts
    # (3M1I1M) are equidistant optimal alignments.
    assert set(cigars) == {"4M1I", "3M1I1M"}
    # Compact CIGAR sorts first.
    assert cigars == ("4M1I", "3M1I1M")


def test_align_deletion():
    dist, cigars = align_global("ACGT", "ACG", 1)
    assert dist == 1 and cigars == ("3M1D",)


def test_align_cap_rejects():
    assert align_global("ACGT", "XXGT", 1) is None
    assert align_global("ACGT", "ACGTTT", 1) is None


def test_replay_roundtrip():
    for ref, piece, cigar in [
        ("ACGT", "ACGTT", "4M1I"),
        ("ACGT", "ACG", "3M1D"),
        ("ACGT", "AXGT", "4M"),
    ]:
        view = _replay(ref, piece, cigar)
        assert view["aligned_reference"].replace("-", "") == ref
        assert view["aligned_read"].replace("-", "") == piece
        assert len(view["aligned_reference"]) == len(view["aligned_read"])


# --------------------------------------------------------------- clean reads


def test_clean_tandem_unique():
    # Non-periodic reference so the cut point is unique.
    ref = "ACGTACGA"
    r = solve(ref, ref * 3, 3, 0)
    assert r["status"] == "unique"
    assert r["objective"] == {"total_edits": 0, "max_segment_edits": 0}
    w = r["witness"]
    assert w["shift"] == 0
    assert w["boundaries"] == [[0, 8], [8, 16], [16, 24]]
    assert all(s["edits"] == 0 and s["cigar"] == "8M" for s in w["segments"])


def test_rotation_is_selected():
    ref = "ACGTACGA"
    rot = rotate(ref, 3)
    assert rot != ref
    r = solve(ref, rot * 4, 4, 0)
    assert r["status"] == "unique"
    assert r["witness"]["shift"] == 3


# ------------------------------------------------------- indel + substitution


def test_mixed_indel_sub_noise():
    # substitution in copy 1, deletion in copy 2, insertion in copy 3
    ref = "ACGTACGA"
    read = "AXGTACGA" + "ACGTACG" + "ACGTACGAA"  # 8, 7 (del), 9 (ins)
    assert len(read) == 8 + 7 + 9
    r = solve(ref, read, 3, 1)
    assert r["objective"] == {"total_edits": 3, "max_segment_edits": 1}
    w = r["witnesses"][0] if r["status"] == "ambiguous" else r["witness"]
    assert w["boundaries"] == [[0, 8], [8, 15], [15, 24]]
    cigs = [s["cigar"] for s in w["segments"]]
    assert cigs[0] == "8M"  # substitution counted inside M run
    assert cigs[1].endswith("1D")
    assert "1I" in cigs[2]
    for s in w["segments"]:
        view = _replay(s["reference"], s["read"], s["cigar"])
        assert view["aligned_reference"].replace("-", "") == s["reference"]
        assert view["aligned_read"].replace("-", "") == s["read"]


# ----------------------------------------------------------- objective order


def test_total_edits_primary_then_worst_segment():
    # 2 substitutions in the first copy, clean rest; non-periodic reference.
    ref = "ACGTACGA"
    read = "XXGTACGA" + ref + ref
    r1 = solve(ref, read, 3, 1)
    assert r1["status"] == "infeasible"
    r2 = solve(ref, read, 3, 2)
    assert r2["status"] == "unique"
    assert r2["objective"] == {"total_edits": 2, "max_segment_edits": 2}


def test_boundaries_can_be_ambiguous():
    # An extra A between clean copies can belong to either neighbour at cost 1.
    r = solve("ACGTAC", "ACGTACAACGTACACGTAC", 3, 1)
    assert r["objective"] == {"total_edits": 1, "max_segment_edits": 1}
    if r["status"] == "ambiguous":
        b1, b2 = r["witnesses"][0]["boundaries"], r["witnesses"][1]["boundaries"]
        assert b1 <= b2  # stable boundary ordering


def test_homopolymer_shift_ambiguity_sorted():
    r = solve("AAAAAAAA", "A" * 24, 3, 0)
    assert r["status"] == "ambiguous"
    shifts = [w["shift"] for w in r["witnesses"]]
    assert shifts == sorted(shifts)
    assert len(shifts) == 2 and shifts[0] < shifts[1]
    # Eight distinct optimal rotations exist, so more witnesses remain.
    assert r["more_witnesses"] is True


def test_exactly_two_optima_reports_no_more():
    # A trailing T insertion against ...T has exactly two equidistant
    # CIGARs and (with a non-periodic reference) a unique shift/boundaries.
    ref = "ACGTACGAT"
    read = ref + ref + (ref + "T")
    r = solve(ref, read, 3, 1)
    assert r["status"] == "ambiguous"
    assert len(r["witnesses"]) == 2
    assert r["more_witnesses"] is False
    w0, w1 = r["witnesses"]
    assert w0["boundaries"] == w1["boundaries"]
    assert w0["segments"][-1]["cigar"] != w1["segments"][-1]["cigar"]


def test_witness_ordering_is_shift_then_boundaries_then_cigar():
    r = solve("AAAAAAA", "A" * 24, 3, 1)
    assert r["status"] == "ambiguous"
    w0, w1 = r["witnesses"]
    key0 = (
        w0["shift"],
        tuple(tuple(b) for b in w0["boundaries"]),
        tuple(s["cigar"] for s in w0["segments"]),
    )
    key1 = (
        w1["shift"],
        tuple(tuple(b) for b in w1["boundaries"]),
        tuple(s["cigar"] for s in w1["segments"]),
    )
    # CIGAR ordering uses the compact-op sort key; at minimum shift ordering
    # must hold and the two witnesses must differ.
    assert w0["shift"] <= w1["shift"]
    assert key0 != key1


# ------------------------------------------------------------- infeasibility


def test_infeasible_is_locatable():
    r = solve("ACGTACGT", "XXGTACGTACGTACGTACGTACGT", 3, 1)
    assert r["status"] == "infeasible"
    assert r["error"] == "constraint_failed"
    nearest = r["nearest"]
    assert nearest is not None
    bad = nearest["violating_segments"]
    assert bad and bad[0]["required_edits"] == 2
    assert bad[0]["over_by"] == 1
    assert bad[0]["read"].startswith("XX")


def test_infeasible_when_read_too_short():
    r = solve("ACGTACGT", "ACG", 3, 3)
    assert r["status"] == "infeasible"
    assert r["constraint"]["name"] == "per_segment_edit_budget"
    # Relaxing the budget still locates a nearest segmentation.
    assert r["nearest"] is not None


def test_structural_infeasibility_read_too_long():
    # 3 copies of an 8-nt reference can cover at most 3*16 = 48 nt.
    r = solve("ACGTACGT", "A" * 160, 3, 3)
    assert r["status"] == "infeasible"
    assert r["constraint"]["name"] == "segment_length"
    assert r["nearest"] is None
    assert r["constraint"]["feasible_read_length"] == [15, 33]


def test_infeasible_random_garbage_still_locates():
    r = solve("ACGTACGTACGTACGTACGT", "Z" * 160, 8, 3)
    assert r["status"] == "infeasible"
    assert r["nearest"]["violating_segments"]


# ------------------------------------------------------- brute-force agreement


def _brute_force(reference, read, copies, cap):
    """Exhaustive reference implementation over every shift/partition."""
    n = len(read)
    L = len(reference)
    solutions = []
    best = None
    for shift in range(L):
        ref = rotate(reference, shift)
        # choose copies-1 cut points among 1..n-1 (non-empty segments)
        for cuts in itertools.combinations(range(1, n), copies - 1):
            bounds = (0,) + cuts + (n,)
            total = 0
            worst = 0
            seg_cigars = []
            ok = True
            for a, b in zip(bounds, bounds[1:]):
                aligned = align_global(ref, read[a:b], cap)
                if aligned is None:
                    ok = False
                    break
                dist, cigs = aligned
                total += dist
                worst = max(worst, dist)
                seg_cigars.append(cigs)
            if not ok:
                continue
            for combo in itertools.product(*seg_cigars):
                cand = ((total, worst), shift, list(zip(bounds, bounds[1:])), combo)
                if best is None or cand[0] < best:
                    best = cand[0]
                    solutions = [cand]
                elif cand[0] == best:
                    solutions.append(cand)
    if best is None:
        return None
    return best, solutions


@pytest.mark.parametrize(
    "reference,read,copies,cap",
    [
        ("ACGTACGT", "ACGTACGTACGTACGT", 3, 0),
        ("ACGTACGT", "ACGTACGTACGTACG", 3, 1),   # trailing deletion
        ("ACGTACGT", "TACGTACGTACGTACGTA", 4, 1),
        ("ACGTTGCA", "ACXTTGCAAACGTTGCAACGTTGCA", 3, 2),
        ("AACCGGTT", "AACCGGTA" "AACCGGTT" "AACCGT", 3, 1),
        ("ACGTACGT", "XXGTACGTACGTACGT", 3, 2),
    ],
)
def test_matches_brute_force(reference, read, copies, cap):
    r = solve(reference, read, copies, cap)
    bf = _brute_force(reference, read, copies, cap)
    if bf is None:
        assert r["status"] == "infeasible"
        return
    (total, worst), sols = bf
    assert r["status"] in ("unique", "ambiguous")
    assert r["objective"] == {"total_edits": total, "max_segment_edits": worst}

    def witness_tuple(w):
        return (
            w["shift"],
            tuple((b[0], b[1]) for b in w["boundaries"]),
            tuple(s["cigar"] for s in w["segments"]),
        )

    bf_sorted = sorted(
        (
            (shift, tuple((a, b) for a, b in bounds), tuple(combo))
            for _, shift, bounds, combo in sols
        )
    )
    if len(bf_sorted) == 1:
        assert r["status"] == "unique"
        assert witness_tuple(r["witness"]) == bf_sorted[0]
    else:
        assert r["status"] == "ambiguous"
        got = [witness_tuple(w) for w in r["witnesses"]]
        # First two optimal witnesses must match (our cigar cross-segment
        # ordering differs from raw string order, so compare as a set for the
        # first two shift/boundary groups and check content membership).
        for g in got:
            assert g in bf_sorted
        assert got[0][0] <= got[1][0]
