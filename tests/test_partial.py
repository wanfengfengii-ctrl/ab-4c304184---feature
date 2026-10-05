"""Tests for terminal_mode="partial" (collection window truncates units)."""

import itertools

import pytest

from app.solver import (
    align_global,
    align_terminal,
    rotate,
    solve,
    _replay,
    _cigar_sort_key,
)


# --------------------------------------------------------- terminal alignment


def test_align_terminal_prefix_picks_proper_prefix():
    ref = "ACGTACGAT"
    # Exact match against the proper prefix R[:5].
    dist, options = align_terminal(ref, "ACGTA", 0, prefix=True)
    assert dist == 0
    assert options == (("5M", 5),)


def test_align_terminal_suffix_picks_proper_suffix():
    ref = "ACGTACGAT"
    dist, options = align_terminal(ref, "CGAT", 0, prefix=False)
    assert dist == 0
    assert options == (("4M", 5),)  # R[5:] == "CGAT"


def test_align_terminal_excludes_full_unit_and_empty():
    ref = "ACGTACGAT"
    # The full unit must not be admissible even when it matches exactly.
    assert align_terminal(ref, ref, 0, prefix=True) is None
    assert align_terminal(ref, ref, 0, prefix=False) is None
    # An observation longer than L - 1 + cap cannot be a proper range.
    assert align_terminal(ref, "A" * 10, 1, prefix=True) is None


def test_align_terminal_enumerates_cut_ties_sorted():
    ref = "AAAAAAAA"  # homopolymer: many cuts tie
    dist, options = align_terminal(ref, "AAA", 0, prefix=True)
    assert dist == 0
    # Only span == 3 matches exactly (insertions/deletions cost), cut == 3.
    assert ("3M", 3) in options
    # Options are sorted by (cigar key, cut).
    cuts = [cut for _, cut in options]
    assert cuts == sorted(cuts)


def test_align_terminal_indel_options():
    ref = "ACGTACGAT"
    # One deletion against R[:6] = "ACGTAC": query "ACTAC" drops the G.
    dist, options = align_terminal(ref, "ACTAC", 1, prefix=True)
    assert dist == 1
    assert options == (("2M1D3M", 6),)


# ----------------------------------------------------------------- clean case


def test_partial_clean_unique():
    ref = "ACGTACGAT"  # 9 nt, non-periodic
    # Extreme cuts (a = L-1, b = 1) make the cyclic interpretation unique:
    # only one observed fragment of each terminal length is available.
    read = ref[8:] + ref + ref[:1]
    r = solve(ref, read, 3, 0, terminal_mode="partial")
    assert r["status"] == "unique"
    assert r["objective"] == {"total_edits": 0, "max_segment_edits": 0}
    w = r["witness"]
    assert w["shift"] == 0
    assert w["boundaries"] == [[0, 1], [1, 10], [10, 11]]
    assert w["terminal_ranges"] == {
        "first_suffix": [8, 9],
        "last_prefix": [0, 1],
    }
    first, middle, last = w["segments"]
    assert first["role"] == "suffix"
    assert first["reference"] == ref[8:]
    assert first["terminal_range"] == [8, 9]
    assert middle["reference"] == ref
    assert "role" not in middle
    assert last["role"] == "prefix"
    assert last["reference"] == ref[:1]
    assert last["terminal_range"] == [0, 1]


def test_partial_rotation_is_jointly_chosen():
    ref = "ACGTACGAT"
    rot = rotate(ref, 4)
    read = rot[8:] + rot + rot[:1]
    r = solve(ref, read, 3, 0, terminal_mode="partial")
    assert r["status"] == "unique"
    assert r["witness"]["shift"] == 4
    assert r["witness"]["terminal_ranges"]["first_suffix"] == [8, 9]
    assert r["witness"]["terminal_ranges"]["last_prefix"] == [0, 1]


def test_partial_cigars_are_replayable():
    ref = "CAGATTTTCA"  # 10
    # Objective (3,1); optimal witnesses together use D and I ops.
    read = "CACAGAGTTCACAGATTTTCACAGAG"
    r = solve(ref, read, 3, 1, terminal_mode="partial")
    assert r["status"] in ("unique", "ambiguous")
    assert r["objective"] == {"total_edits": 3, "max_segment_edits": 1}
    ws = r["witnesses"] if r["status"] == "ambiguous" else [r["witness"]]
    for w in ws:
        for seg in w["segments"]:
            view = _replay(seg["reference"], seg["read"], seg["cigar"])
            assert view["aligned_reference"].replace("-", "") == seg["reference"]
            assert view["aligned_read"].replace("-", "") == seg["read"]
        tr = w["terminal_ranges"]
        assert 1 <= tr["first_suffix"][0] < 10
        assert tr["last_prefix"][0] == 0 and 1 <= tr["last_prefix"][1] < 10


def test_partial_mixed_indel_sub():
    ref = "CAGATTTTCA"  # 10
    read = "CACAGAGTTCACAGATTTTCACAGAG"
    r = solve(ref, read, 3, 1, terminal_mode="partial")
    assert r["status"] in ("unique", "ambiguous")
    assert r["objective"] == {"total_edits": 3, "max_segment_edits": 1}
    witness = r.get("witness") or r["witnesses"][0]
    kinds = {
        op
        for s in witness["segments"]
        for op in s["cigar"]
        if op.isalpha()
    }
    # Substitutions count inside M; indels appear across the two optimal
    # witnesses, so inspect both when ambiguous.
    for w in r.get("witnesses", [witness]):
        kinds |= {
            op
            for s in w["segments"]
            for op in s["cigar"]
            if op.isalpha()
        }
    assert {"M", "D", "I"} <= kinds


# --------------------------------------------------------- stable tie ordering


def test_partial_ambiguous_returns_first_two_with_cut_ordering():
    # Homopolymer: every shift and many cuts are equivalent.
    r = solve("AAAAAAAAAA", "A" * 20, 3, 0, terminal_mode="partial")
    assert r["status"] == "ambiguous"
    assert len(r["witnesses"]) == 2
    assert r["more_witnesses"] is True
    w0, w1 = r["witnesses"]

    def key(w):
        return (
            w["shift"],
            tuple(tuple(b) for b in w["boundaries"]),
            tuple(
                _cigar_sort_key(s["cigar"]) + (s.get("reference_cut", -1),)
                for s in w["segments"]
            ),
        )

    assert key(w0) <= key(w1)
    # Terminal cuts are exposed and lie strictly inside the unit.
    for w in (w0, w1):
        a, _ = w["terminal_ranges"]["first_suffix"]
        _, b = w["terminal_ranges"]["last_prefix"]
        assert 1 <= a <= 9 and 1 <= b <= 9


def test_partial_ties_order_boundaries_before_cigars():
    # Regression: the second witness must be the next CIGAR option of the
    # smallest boundary (same boundaries), not a larger boundary.  This exact
    # read has two optimal middle-copy CIGARs within one boundary; a naive
    # alignment DFS skipped the second to a different boundary.
    ref = "CTTCGTGG"
    read = "GTGGCTTCGTGGCTCGTGGCTTCGGTGG"
    r = solve(ref, read, 5, 1, terminal_mode="partial")
    assert r["status"] == "ambiguous"
    assert r["objective"] == {"total_edits": 2, "max_segment_edits": 1}
    w0, w1 = r["witnesses"]
    assert w0["boundaries"] == w1["boundaries"]
    assert w0["boundaries"] == [
        [0, 1],
        [1, 9],
        [9, 16],
        [16, 24],
        [24, 28],
    ]
    c0 = [s["cigar"] for s in w0["segments"]]
    c1 = [s["cigar"] for s in w1["segments"]]
    assert c0 != c1
    assert c0[2] != c1[2]  # the differing middle-copy CIGAR


def test_partial_optimal_cut_is_reported_in_each_ambiguous_witness():
    # When a terminal has two optimal cuts, both appear and stay ordered.
    ref = "AAAAAAAA"  # L=8
    # First fragment of 3 A's matches R[5:] (span 3) exactly; nothing else at
    # distance 0 except span 3, but homopolymer middle/shift produce multiple
    # global witnesses.  Mainly assert each witness exposes legal cuts.
    read = "AAA" + "A" * 8 + "AAA"
    r = solve(ref, read, 3, 0, terminal_mode="partial")
    assert r["status"] in ("unique", "ambiguous")
    ws = r["witnesses"] if r["status"] == "ambiguous" else [r["witness"]]
    for w in ws:
        a, _ = w["terminal_ranges"]["first_suffix"]
        _, b = w["terminal_ranges"]["last_prefix"]
        assert 1 <= a <= 7 and 1 <= b <= 7


# ------------------------------------------------------------- compatibility


def test_default_mode_matches_explicit_full():
    ref = "ACGTACGAT"
    a = solve(ref, ref * 3, 3, 0)
    b = solve(ref, ref * 3, 3, 0, terminal_mode="full")
    assert a["status"] == b["status"] == "unique"
    assert a["objective"] == b["objective"]


def test_partial_read_that_needs_full_copies_is_infeasible_in_full_mode():
    ref = "ACGTACGAT"
    read = ref[8:] + ref + ref[:1]
    # partial decodes it...
    assert solve(ref, read, 3, 0, "partial")["status"] == "unique"
    # ...but legacy full mode must keep rejecting it (compatibility).
    legacy = solve(ref, read, 3, 0)
    assert legacy["status"] == "infeasible"
    assert legacy["constraint"]["name"] == "per_segment_edit_budget"


# ------------------------------------------------------------- infeasibility


def test_partial_noise_over_budget_is_locatable():
    ref = "ACGTACGTT"  # 9
    # First partial suffix is a legal-length fragment but pure noise.
    read = "GGGGG" + ref + ref[:5]
    r = solve(ref, read, 3, 1, terminal_mode="partial")
    assert r["status"] == "infeasible"
    assert r["error"] == "constraint_failed"
    assert r["constraint"]["name"] == "per_segment_edit_budget"
    nearest = r["nearest"]
    assert nearest is not None
    assert nearest["violating_segments"]


def test_partial_geometric_truncation_failure():
    # Read longer than the budgeted geometry (proper termini + full middles)
    # but still within the cap-free structural envelope, so a nearest
    # witness exists and a terminal segment is geometrically impossible.
    ref = "ACGTACGT"  # L=8, cap=1
    # budgeted max = 8 + 9 + 8 = 25; cap-free max = 14 + 16 + 14 = 44
    read = "A" * 30
    r = solve(ref, read, 3, 1, terminal_mode="partial")
    assert r["status"] == "infeasible"
    assert r["constraint"]["name"] == "terminal_range"
    assert r["constraint"]["terminal_observed_length"] == [1, 8]
    assert r["constraint"]["terminal_reference_span"] == [1, 7]
    blockers = [
        v
        for v in r["nearest"]["violating_segments"]
        if v["geometric_terminal_failure"]
    ]
    assert blockers
    assert blockers[0]["role"] in ("suffix", "prefix")
    assert "terminal_range" in blockers[0]


def test_partial_structural_too_long():
    ref = "ACGTACGT"  # cap-free max 44 for 3 partial copies
    r = solve(ref, "A" * 46, 3, 1, terminal_mode="partial")
    assert r["status"] == "infeasible"
    assert r["constraint"]["name"] == "segment_length"
    assert r["nearest"] is None


def test_partial_structural_too_short():
    r = solve("ACGTACGT", "AC", 3, 1, terminal_mode="partial")
    assert r["status"] == "infeasible"
    assert r["constraint"]["name"] == "segment_length"
    assert r["nearest"] is None


# ------------------------------------------------------- brute-force agreement


def _brute_partial(reference, read, copies, cap):
    """Exhaustive reference for partial termini."""
    n = len(read)
    L = len(reference)
    solutions = []
    best = None
    for shift in range(L):
        ref = rotate(reference, shift)
        for cuts in itertools.combinations(range(1, n), copies - 1):
            bounds = (0,) + cuts + (n,)
            total = 0
            worst = 0
            seg_options = []
            ok = True
            for idx, (a, b) in enumerate(zip(bounds, bounds[1:])):
                piece = read[a:b]
                cands = []
                if idx == 0:
                    for cut in range(1, L):
                        aligned = align_global(ref[cut:], piece, cap)
                        if aligned:
                            cands += [
                                (aligned[0], cigar, cut)
                                for cigar in aligned[1]
                            ]
                elif idx == copies - 1:
                    for cut in range(1, L):
                        aligned = align_global(ref[:cut], piece, cap)
                        if aligned:
                            cands += [
                                (aligned[0], cigar, cut)
                                for cigar in aligned[1]
                            ]
                else:
                    aligned = align_global(ref, piece, cap)
                    if aligned:
                        cands += [
                            (aligned[0], cigar, None) for cigar in aligned[1]
                        ]
                if not cands:
                    ok = False
                    break
                dist = min(c[0] for c in cands)
                cands = [c for c in cands if c[0] == dist]
                total += dist
                worst = max(worst, dist)
                seg_options.append(cands)
            if not ok:
                continue
            for combo in itertools.product(*seg_options):
                cand = (
                    (total, worst),
                    shift,
                    tuple((a, b) for a, b in zip(bounds, bounds[1:])),
                    tuple((c[1], c[2]) for c in combo),
                )
                if best is None or cand[0] < best:
                    best = cand[0]
                    solutions = [cand]
                elif cand[0] == best:
                    solutions.append(cand)
    return best, solutions


@pytest.mark.parametrize(
    "reference,read,copies,cap",
    [
        ("ACGTACGAT", "TACGAT" + "ACGTACGAT" + "ACGTA", 3, 0),
        ("ACGTACGTT", "TGACGAT" + "ACGTACGT" + "ACGTA", 3, 1),
        ("ACGTTGCA", "TTGCA" + "ACGTTGCA" + "ACGT", 3, 1),
        ("AACCGGTT", "CGGTTAACCGGTTAACC", 3, 1),
        ("ACGTACGT", "GTAC" + "ACGTACGT" + "ACGTAC" + "TACG", 4, 1),
        ("ACGTACGA", "CGTA" + "ACGTACGA" + "ACGTACGA" + "AC", 4, 2),
    ],
)
def test_partial_matches_brute_force(reference, read, copies, cap):
    r = solve(reference, read, copies, cap, terminal_mode="partial")
    best, sols = _brute_partial(reference, read, copies, cap)
    if best is None:
        assert r["status"] == "infeasible"
        return
    assert r["status"] in ("unique", "ambiguous")
    assert r["objective"] == {"total_edits": best[0], "max_segment_edits": best[1]}

    def witness_tuple(w):
        return (
            w["shift"],
            tuple((b[0], b[1]) for b in w["boundaries"]),
            tuple((s["cigar"], s.get("reference_cut")) for s in w["segments"]),
        )

    all_sorted = sorted(
        {(sh, bd, combo) for _, sh, bd, combo in sols},
        key=lambda z: (
            z[0],
            z[1],
            tuple(
                _cigar_sort_key(c) + ((ct if ct is not None else -1),)
                for c, ct in z[2]
            ),
        ),
    )
    got = r.get("witnesses")
    if got is None:
        assert witness_tuple(r["witness"]) == all_sorted[0]
    else:
        expected = all_sorted[: len(got)]
        assert [witness_tuple(w) for w in got] == expected
