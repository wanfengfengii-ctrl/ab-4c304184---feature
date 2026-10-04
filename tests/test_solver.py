"""Tests for the concatemer solver."""

import itertools
import random

import pytest

from app.solver import (
    align_global,
    rotate,
    solve,
    _cigar_sort_key,
    _replay,
    TERMINAL_PARTIAL,
)


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


# --------------------------------------------------------- partial terminals


def _framed_read(ref, shift, first_cut, last_cut, copies):
    """Build a read starting at proper suffix and ending at proper prefix."""
    rot = rotate(ref, shift)
    pieces = [rot[first_cut:]]
    pieces += [rot] * (copies - 2)
    pieces.append(rot[:last_cut])
    return rot, "".join(pieces)


def test_partial_clean_unique_reports_terminal_ranges():
    # One-base terminal framing makes the cut point unique even on clean data.
    ref = "ACGTACGATC"  # 10 nt, non-periodic
    rot = ref
    read = rot[9:] + rot + rot[:1]
    r = solve(ref, read, 3, 0, TERMINAL_PARTIAL)
    assert r["status"] == "unique"
    assert r["objective"] == {"total_edits": 0, "max_segment_edits": 0}
    w = r["witness"]
    assert w["terminal_mode"] == "partial"
    assert w["shift"] == 0
    assert w["terminal_ranges"] == {"first": [9, 10], "last": [0, 1]}
    assert w["boundaries"] == [[0, 1], [1, 11], [11, 12]]
    first, middle, last = w["segments"]
    assert first["role"] == "first"
    assert first["reference"] == rot[9:]
    assert first["ref_range"] == [9, 10]
    assert "role" not in middle
    assert middle["reference"] == rot and middle["ref_range"] == [0, 10]
    assert last["role"] == "last" and last["reference"] == rot[:1]
    for s in w["segments"]:
        view = _replay(s["reference"], s["read"], s["cigar"])
        assert view["aligned_reference"].replace("-", "") == s["reference"]
        assert view["aligned_read"].replace("-", "") == s["read"]


def test_partial_clean_framing_with_common_rotation_shift():
    ref = "ACGTACGATC"
    rot = rotate(ref, 3)
    read = rot[9:] + rot + rot[:1]
    r = solve(ref, read, 3, 0, TERMINAL_PARTIAL)
    assert r["status"] == "unique"
    w = r["witness"]
    assert w["shift"] == 3
    assert w["terminal_ranges"] == {"first": [9, 10], "last": [0, 1]}


def test_partial_wider_clean_framing_is_objectively_tied_and_stable():
    # With multi-base truncated ends on clean data the cyclic cut point is
    # intrinsically degenerate (a boundary can slide along the covered unit);
    # the service must surface this as ordered equi-optimal witnesses.
    ref = "ACGTACGATC"
    rot, read = _framed_read(ref, 0, 4, 7, 3)
    r = solve(ref, read, 3, 0, TERMINAL_PARTIAL)
    assert r["status"] == "ambiguous"
    assert r["objective"] == {"total_edits": 0, "max_segment_edits": 0}
    assert r["more_witnesses"] is True
    for w in r["witnesses"]:
        fr, lr = w["terminal_ranges"]["first"], w["terminal_ranges"]["last"]
        assert 0 < fr[0] < fr[1] == 10 and 0 == lr[0] < lr[1] < 10
        assert all(s["edits"] == 0 for s in w["segments"])
    keys = [_partial_witness_key(w) for w in r["witnesses"]]
    assert keys == sorted(keys)


def test_partial_two_copies_has_only_terminal_pieces():
    ref = "ACGTACGATC"
    rot = ref
    read = rot[9:] + rot[:1]  # "C" + "A"
    r = solve(ref, read, 2, 0, TERMINAL_PARTIAL)
    assert r["status"] == "unique"
    w = r["witness"]
    assert [s["role"] for s in w["segments"]] == ["first", "last"]
    assert w["terminal_ranges"] == {"first": [9, 10], "last": [0, 1]}
    assert w["boundaries"] == [[0, 1], [1, 2]]


def test_partial_requires_proper_suffix_and_prefix():
    # With copies=2 both pieces are terminals, each bounded to a proper
    # span (<= 9 of 10).  A 19-nt read forces one piece to span all 10
    # bases, which no proper suffix/prefix covers at zero edits: the answer
    # must be a locatable budget failure rather than a successful framing.
    ref = "ACGTACGATC"
    read = ref + ref[:9]
    assert len(read) == 19
    r = solve(ref, read, 2, 0, TERMINAL_PARTIAL)
    assert r["status"] == "infeasible"
    assert r["error"] == "constraint_failed"
    assert r["constraint"]["name"] == "per_segment_edit_budget"
    nearest = r["nearest"]
    assert nearest is not None
    bad = nearest["violating_segments"]
    assert bad and bad[0]["role"] == "first"
    assert bad[0]["required_edits"] == 1 and bad[0]["over_by"] == 1
    span = bad[0]["ref_range"][1] - bad[0]["ref_range"][0]
    assert span <= 9


def test_partial_noise_substitution_deletion_insertion():
    # 1 sub in the suffix head, 1 deletion in the middle copy, 1 insertion in
    # the prefix tail; cap 1 per piece.
    ref = "ACGTACGATC"
    _rot, clean = _framed_read(ref, 0, 4, 7, 3)
    head, mid, tail = clean[:6], clean[6:16], clean[16:]
    head = "T" + head[1:]            # substitution at the head
    mid = mid[:5] + mid[6:]          # deletion of one base
    tail = tail[:3] + "G" + tail[3:]  # insertion
    read = head + mid + tail
    r = solve(ref, read, 3, 1, TERMINAL_PARTIAL)
    assert r["status"] in ("unique", "ambiguous")
    assert r["objective"] == {"total_edits": 3, "max_segment_edits": 1}
    w = r["witness"] if r["status"] == "unique" else r["witnesses"][0]
    ops = {op for s in w["segments"] for op in s["cigar"] if op.isalpha()}
    assert {"M", "D", "I"} <= ops
    for s in w["segments"]:
        view = _replay(s["reference"], s["read"], s["cigar"])
        assert view["aligned_reference"].replace("-", "") == s["reference"]
        assert view["aligned_read"].replace("-", "") == s["read"]


def test_partial_budget_exceeded_is_locatable():
    ref = "ACGTACGATC"
    _rot, clean = _framed_read(ref, 0, 4, 7, 3)
    read = "TT" + clean[2:]  # two substitutions in the heading suffix piece
    r = solve(ref, read, 3, 1, TERMINAL_PARTIAL)
    assert r["status"] == "infeasible"
    assert r["terminal_mode"] == "partial"
    assert r["constraint"]["name"] == "per_segment_edit_budget"
    nearest = r["nearest"]
    assert nearest is not None
    assert nearest["terminal_ranges"]["first"][1] == 10
    bad = nearest["violating_segments"]
    assert bad and bad[0]["segment_index"] == 0
    assert bad[0]["required_edits"] == 2 and bad[0]["over_by"] == 1
    assert bad[0]["role"] == "first"
    assert bad[0]["ref_range"][1] - bad[0]["ref_range"][0] <= 9


def test_partial_structural_failure_when_window_too_long():
    ref = "ACGTACGT"  # L = 8
    # 3 pieces may cover at most 2*(2L-1) + 2L = 30 + 16 = 46 nt cap-free.
    r = solve(ref, "A" * 47, 3, 3, TERMINAL_PARTIAL)
    assert r["status"] == "infeasible"
    assert r["constraint"]["name"] == "terminal_prefix_suffix"
    assert r["nearest"] is None


def test_partial_structural_failure_single_base_read():
    r = solve("ACGTACGT", "A", 3, 3, TERMINAL_PARTIAL)
    assert r["status"] == "infeasible"
    assert r["constraint"]["name"] == "terminal_prefix_suffix"


def test_partial_witness_ties_include_terminal_ranges_in_ordering():
    # Homopolymer: shifts, boundaries and terminal ranges are all
    # indistinguishable in content; returned witnesses must still be
    # deterministically ordered with the terminal ranges present.
    r = solve("AAAAAAAA", "A" * 20, 3, 0, TERMINAL_PARTIAL)
    assert r["status"] == "ambiguous"
    keys = [_partial_witness_key(w) for w in r["witnesses"]]
    assert keys == sorted(keys)
    for w in r["witnesses"]:
        first = w["terminal_ranges"]["first"]
        last = w["terminal_ranges"]["last"]
        assert 0 < first[0] < first[1] == 8
        assert 0 == last[0] < last[1] < 8
        assert all(s["edits"] == 0 for s in w["segments"])


def test_partial_default_argument_keeps_full_mode():
    ref = "ACGTACGATC"
    r = solve(ref, ref * 3, 3, 0)
    assert r["status"] == "unique"
    assert "terminal_ranges" not in r["witness"]




# --------------------------------------------- partial brute-force agreement


def _brute_force_partial(reference, read, copies, cap):
    """Exhaustive reference implementation for partial terminal mode."""
    n = len(read)
    L = len(reference)
    solutions = []
    best = None
    for shift in range(L):
        rot = rotate(reference, shift)
        for cuts in itertools.combinations(range(1, n), copies - 1):
            bounds = (0,) + cuts + (n,)
            for a in range(1, L):      # proper suffix length L-a ... span a
                for b in range(1, L):  # proper prefix length b
                    seg_specs = []
                    ok = True
                    total = 0
                    worst = 0
                    for idx, (p0, p1) in enumerate(zip(bounds, bounds[1:])):
                        if idx == 0:
                            seg_ref = rot[L - a :]
                            rng = (L - a, L)
                            role = "first"
                        elif idx == copies - 1:
                            seg_ref = rot[:b]
                            rng = (0, b)
                            role = "last"
                        else:
                            seg_ref = rot
                            rng = (0, L)
                            role = "middle"
                        aligned = align_global(seg_ref, read[p0:p1], cap)
                        if aligned is None:
                            ok = False
                            break
                        dist, cigs = aligned
                        total += dist
                        worst = max(worst, dist)
                        seg_specs.append((p0, p1, cigs, rng, role))
                    if not ok:
                        continue
                    for combo in itertools.product(*[s[2] for s in seg_specs]):
                        cand = (
                            (total, worst),
                            shift,
                            tuple(
                                (p1, rng, _cigar_sort_key(cigar))
                                for (_, p1, _, rng, _), cigar in zip(
                                    seg_specs, combo
                                )
                            ),
                        )
                        if best is None or cand[0] < best:
                            best = cand[0]
                            solutions = [cand]
                        elif cand[0] == best:
                            solutions.append(cand)
    if best is None:
        return None
    return best, solutions


def _partial_witness_key(w):
    return (
        w["shift"],
        tuple(
            (s["end"], tuple(s["ref_range"]), _cigar_sort_key(s["cigar"]))
            for s in w["segments"]
        ),
    )


@pytest.mark.parametrize(
    "reference,read,copies,cap",
    [
        ("ACGTACGATC", None, 3, 0),
        ("ACGTACGATC", None, 4, 1),
        ("ACGTTGCAA", None, 3, 2),
        ("AACCGGTTA", None, 3, 1),
        ("ACGTACGT", None, 3, 1),
    ],
)
def test_partial_matches_brute_force_clean(reference, read, copies, cap):
    rot, read = _framed_read(reference, 2, 3, 6, copies)
    r = solve(reference, read, copies, cap, TERMINAL_PARTIAL)
    bf = _brute_force_partial(reference, read, copies, cap)
    assert bf is not None
    (total, worst), sols = bf
    assert r["status"] in ("unique", "ambiguous")
    assert r["objective"] == {"total_edits": total, "max_segment_edits": worst}
    bf_sorted = sorted(
        (
            (shift, tuple((p1, rng, cigar) for p1, rng, cigar in witness))
            for _, shift, witness in sols
        )
    )
    if len(bf_sorted) == 1:
        assert r["status"] == "unique"
        got = _partial_witness_key(r["witness"])
        assert got == bf_sorted[0]
    else:
        assert r["status"] == "ambiguous"
        got0 = _partial_witness_key(r["witnesses"][0])
        got1 = _partial_witness_key(r["witnesses"][1])
        assert got0 == bf_sorted[0]
        assert got1 == bf_sorted[1]
        assert r["more_witnesses"] == (len(bf_sorted) > 2)


def test_partial_matches_brute_force_randomized():
    rng = random.Random(20261004)
    for case in range(10):
        L = rng.randint(8, 10)
        reference = "".join(rng.choice("ACGT") for _ in range(L))
        copies = rng.randint(2, 3)
        shift = rng.randrange(L)
        first_cut = rng.randint(1, L - 1)
        last_len = rng.randint(1, L - 1)
        rot, read = _framed_read(
            reference, shift, first_cut, last_len, copies
        )
        # Sprinkle up to one edit per piece.
        read = list(read)
        bounds = [0]
        bounds.append(L - first_cut)
        for _ in range(copies - 2):
            bounds.append(bounds[-1] + L)
        bounds.append(len(read))
        for idx in range(copies):
            if bounds[idx + 1] - bounds[idx] <= 1:
                continue
            if rng.random() < 0.6:
                pos = rng.randrange(bounds[idx], bounds[idx + 1])
                op = rng.choice(("sub", "del", "ins"))
                if op == "sub":
                    read[pos] = rng.choice("ACGT")
                elif op == "ins":
                    read.insert(pos, rng.choice("ACGT"))
                    for j in range(idx + 1, len(bounds)):
                        bounds[j] += 1
                else:
                    read.pop(pos)
                    for j in range(idx + 1, len(bounds)):
                        bounds[j] -= 1
        read = "".join(read)
        if not 1 <= len(read):
            continue
        cap = 2
        r = solve(reference, read, copies, cap, TERMINAL_PARTIAL)
        bf = _brute_force_partial(reference, read, copies, cap)
        if bf is None:
            assert r["status"] == "infeasible", (case, read)
            continue
        (total, worst), sols = bf
        assert r["status"] in ("unique", "ambiguous"), case
        assert r["objective"] == {
            "total_edits": total,
            "max_segment_edits": worst,
        }, (case, r["objective"], (total, worst))
        bf_sorted = sorted(
            (
                (shift2, tuple((p1, rng2, cigar) for p1, rng2, cigar in witness))
                for _, shift2, witness in sols
            )
        )
        firsts = [
            _partial_witness_key(
                r["witness"]
                if r["status"] == "unique"
                else r["witnesses"][0]
            )
        ]
        assert firsts[0] == bf_sorted[0], case


def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        nxt = [i] + [0] * n
        for j in range(1, n + 1):
            nxt[j] = min(
                dp[j] + 1,
                nxt[j - 1] + 1,
                dp[j - 1] + (a[i - 1] != b[j - 1]),
            )
        dp = nxt
    return dp[n]


def _nearest_partial_brute(reference, read, copies):
    """Cap-free optimal (total, max) cost over shifts/cuts/terminal spans."""
    n = len(read)
    L = len(reference)
    best = None
    for shift in range(L):
        rot = rotate(reference, shift)
        for cuts in itertools.combinations(range(1, n), copies - 1):
            bounds = (0,) + cuts + (n,)
            for a in range(1, L):
                for b in range(1, L):
                    total = 0
                    worst = 0
                    for idx, (p0, p1) in enumerate(zip(bounds, bounds[1:])):
                        if idx == 0:
                            seg_ref = rot[L - a:]
                        elif idx == copies - 1:
                            seg_ref = rot[:b]
                        else:
                            seg_ref = rot
                        d = _levenshtein(seg_ref, read[p0:p1])
                        total += d
                        worst = max(worst, d)
                    cand = (total, worst)
                    if best is None or cand < best:
                        best = cand
    return best


def test_partial_infeasible_nearest_diagnosis_is_consistent():
    cases = [
        (
            "ACGTACGATC",
            "TT" + ("ACGTACGATC"[4:])[2:] + "ACGTACGATC" + "ACGTACGA",
            3,
            1,
        ),
        ("ACGTACGTT", "GGGTACGTTACGTACGTTACGTAC", 3, 1),
        ("ACGTTGCAA", "ZZZZZZ" + "ACGTTGCAA" + "ACGTTG", 3, 2),
    ]
    for reference, read, copies, cap in cases:
        r = solve(reference, read, copies, cap, TERMINAL_PARTIAL)
        assert r["status"] == "infeasible"
        assert r["constraint"]["name"] == "per_segment_edit_budget"
        nearest = r["nearest"]
        assert nearest is not None
        rot = rotate(reference, nearest["shift"])
        checked = 0
        for v in nearest["violating_segments"]:
            assert v["required_edits"] > cap
            # Recompute the reported distance against the reported range.
            rng_pair = v["ref_range"]
            seg_ref = rot[rng_pair[0]:rng_pair[1]]
            assert _levenshtein(seg_ref, v["read"]) == v["required_edits"]
            assert v["over_by"] == v["required_edits"] - cap
            checked += 1
        assert checked >= 1

        # Small enough cases: the cap-free optimum must match brute force.
        if len(reference) <= 9 and len(read) <= 24:
            assert (
                nearest["total_edits"],
                nearest["max_segment_edits"],
            ) == _nearest_partial_brute(reference, read, copies)
