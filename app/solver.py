"""Core concatemer decoding logic.

Given a circular reference, a noisy tandem-read, a copy number ``k`` and a
per-copy edit budget ``cap``, jointly choose:

1. a cyclic shift (rotation) of the reference,
2. exactly ``k`` consecutive, non-empty segments covering the whole read,
3. a global (Needleman-Wunsch, unit indel/sub cost) alignment for every
   segment,

so that total edit count is minimized, ties broken by the maximum edit count
of any single segment.  Witnesses are stably ordered by
``(shift, boundaries, cigars)``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# Safety bound on the number of distinct optimal CIGARs enumerated for one
# (reference, read) pair.  With edit distance <= 3 and lengths <= 20 + 3 this
# is never reached in practice; it only guards pathological homopolymers.
_CIGAR_ENUM_LIMIT = 20_000

# Sam-style op rank used only for deterministic traversal.
_OP_RANK = {"M": 0, "D": 1, "I": 2}


def _build_cigar(steps: str) -> str:
    """Turn a raw step string ("MMDMIM") into a merged CIGAR ("3M1D1M1I1M")."""
    parts: List[str] = []
    run_op = ""
    run_len = 0
    for op in steps:
        if op == run_op:
            run_len += 1
        else:
            if run_op:
                parts.append(f"{run_len}{run_op}")
            run_op = op
            run_len = 1
    if run_op:
        parts.append(f"{run_len}{run_op}")
    return "".join(parts)


@lru_cache(maxsize=40_000)
def align_global(ref: str, query: str, cap: int) -> Optional[Tuple[int, Tuple[str, ...]]]:
    """Globally align ``query`` against ``ref`` with unit edit costs.

    Returns ``(edit_distance, tuple_of_all_optimal_cigars)`` or ``None`` when
    the distance exceeds ``cap``.  CIGARs use Sam semantics:

    * ``M`` - reference/query consumed together (match or substitution),
    * ``D`` - reference base deleted from the query,
    * ``I`` - query base inserted relative to the reference.
    """
    m = len(ref)
    n = len(query)
    # A cheap necessary condition: length difference alone exceeds the cap.
    if abs(m - n) > cap:
        return None

    inf = m + n + 1
    # Banded DP: any path staying within ``cap`` edits never leaves the
    # diagonal band |i - j| <= cap, and any cell with value > cap is dead.
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(m + 1):
        j_lo = max(0, i - cap)
        j_hi = min(n, i + cap)
        for j in range(j_lo, j_hi + 1):
            if i == 0 and j == 0:
                continue
            best = inf
            if i and j and abs((i - 1) - (j - 1)) <= cap:
                cost = dp[i - 1][j - 1] + (0 if ref[i - 1] == query[j - 1] else 1)
                if cost < best:
                    best = cost
            if i and abs((i - 1) - j) <= cap:
                cost = dp[i - 1][j] + 1
                if cost < best:
                    best = cost
            if j and abs(i - (j - 1)) <= cap:
                cost = dp[i][j - 1] + 1
                if cost < best:
                    best = cost
            dp[i][j] = best if best <= cap else inf

    distance = dp[m][n]
    if distance > cap:
        return None

    # Enumerate every optimal traceback.  Iterative DFS; state holds
    # (i, j, steps-so-far).  Predecessor order is fixed for determinism.
    cigars: set[str] = set()
    stack: List[Tuple[int, int, str]] = [(m, n, "")]
    while stack:
        i, j, steps = stack.pop()
        if i == 0 and j == 0:
            cigars.add(_build_cigar(steps))
            if len(cigars) >= _CIGAR_ENUM_LIMIT:
                break
            continue
        val = dp[i][j]
        if i and j and dp[i - 1][j - 1] + (
            0 if ref[i - 1] == query[j - 1] else 1
        ) == val:
            stack.append((i - 1, j - 1, "M" + steps))
        if j and dp[i][j - 1] + 1 == val:
            stack.append((i, j - 1, "I" + steps))
        if i and dp[i - 1][j] + 1 == val:
            stack.append((i - 1, j, "D" + steps))

    ordered = tuple(sorted(cigars, key=lambda c: _cigar_sort_key(c)))
    return distance, ordered


def _cigar_sort_key(cigar: str) -> Tuple:
    """Deterministic ordering of CIGAR strings: op count, then ops/amounts."""
    ops: List[Tuple[int, int]] = []
    num = ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            ops.append((_OP_RANK[ch], int(num)))
            num = ""
    return (len(ops), tuple(ops), cigar)


def rotate(reference: str, shift: int) -> str:
    shift %= len(reference)
    return reference[shift:] + reference[:shift]


def _witness_key(witness: dict) -> Tuple:
    return (
        witness["shift"],
        tuple((s["start"], s["end"]) for s in witness["segments"]),
        tuple(_cigar_sort_key(s["cigar"]) for s in witness["segments"]),
    )


def solve(reference: str, read: str, copies: int, max_edits: int) -> dict:
    """Run the full joint optimization.

    Returns a result envelope with ``status`` of ``unique`` / ``ambiguous`` /
    ``infeasible``.
    """
    length = len(reference)
    n = len(read)
    min_seg_len = max(1, length - max_edits)
    max_seg_len = length + max_edits

    # Per-shift optimum (total_edits, max_segment_edits); None == infeasible.
    shift_best: List[Optional[Tuple[int, int]]] = []
    shift_tables: List[Optional[tuple]] = []

    for shift in range(length):
        ref = rotate(reference, shift)
        pre, suff = _build_tables(
            ref, read, copies, max_edits, min_seg_len, max_seg_len
        )
        shift_best.append(pre[copies].get(n))
        shift_tables.append((pre, suff))

    feasible = [b for b in shift_best if b is not None]
    if not feasible:
        return _infeasible_envelope(
            reference, read, copies, max_edits, shift_best, shift_tables
        )

    optimum = min(feasible)

    # Recover up to three smallest witnesses per optimal shift.  Three is
    # enough to return the first two while knowing whether more exist; the
    # shift is the leading sort key.
    candidates: List[dict] = []
    for shift, best in enumerate(shift_best):
        if best != optimum:
            continue
        ref = rotate(reference, shift)
        pre, suff = shift_tables[shift]
        witnesses = _recover_witnesses(
            shift,
            ref,
            read,
            copies,
            max_edits,
            min_seg_len,
            max_seg_len,
            pre,
            suff,
            optimum,
            limit=3,
        )
        candidates.extend(witnesses)

    candidates.sort(key=_witness_key)
    objective = {"total_edits": optimum[0], "max_segment_edits": optimum[1]}

    if len(candidates) >= 2:
        return {
            "status": "ambiguous",
            "objective": objective,
            "witnesses": candidates[:2],
            "more_witnesses": len(candidates) > 2,
        }
    return {"status": "unique", "objective": objective, "witness": candidates[0]}


def _build_tables(
    ref: str,
    read: str,
    copies: int,
    cap: int,
    min_len: int,
    max_len: int,
) -> Tuple[List[Dict[int, Tuple[int, int]]], List[Dict[int, Tuple[int, int]]]]:
    """Prefix and suffix best-cost tables over (segment index, read offset).

    Cost is the lexicographic pair (total edits, max per-segment edits).
    """
    n = len(read)
    # pre[seg][p] = best cost covering read[:p] with exactly seg segments.
    pre: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(copies + 1)]
    pre[0][0] = (0, 0)
    for seg in range(copies):
        remaining_after = copies - seg - 1
        for p, (total, worst) in pre[seg].items():
            lo = max(min_len, n - p - remaining_after * max_len)
            hi = min(max_len, n - p - remaining_after * min_len)
            for seg_len in range(lo, hi + 1):
                aligned = align_global(ref, read[p : p + seg_len], cap)
                if aligned is None:
                    continue
                dist = aligned[0]
                cand = (total + dist, max(worst, dist))
                q = p + seg_len
                old = pre[seg + 1].get(q)
                if old is None or cand < old:
                    pre[seg + 1][q] = cand

    # suff[seg][p] = best cost covering read[p:] with segments
    # seg .. copies-1 (i.e. copies-seg segments).
    suff: List[Dict[int, Tuple[int, int]]] = [
        dict() for _ in range(copies + 1)
    ]
    suff[copies][n] = (0, 0)
    for seg in range(copies - 1, -1, -1):
        remaining_after = copies - seg - 1
        for p in range(0, n + 1):
            best: Optional[Tuple[int, int]] = None
            lo = max(min_len, n - p - remaining_after * max_len)
            hi = min(max_len, n - p - remaining_after * min_len)
            for seg_len in range(lo, hi + 1):
                q = p + seg_len
                tail = suff[seg + 1].get(q)
                if tail is None:
                    continue
                aligned = align_global(ref, read[p:q], cap)
                if aligned is None:
                    continue
                dist = aligned[0]
                cand = (tail[0] + dist, max(tail[1], dist))
                if best is None or cand < best:
                    best = cand
            if best is not None:
                suff[seg][p] = best
    return pre, suff


def _recover_witnesses(
    shift: int,
    ref: str,
    read: str,
    copies: int,
    cap: int,
    min_len: int,
    max_len: int,
    pre: List[Dict[int, Tuple[int, int]]],
    suff: List[Dict[int, Tuple[int, int]]],
    optimum: Tuple[int, int],
    limit: int,
) -> List[dict]:
    """Depth-first reconstruction of optimal witnesses in stable order.

    Segment lengths are tried ascending and CIGARs are already sorted, so the
    first emitted solutions are the smallest by (boundaries, cigars).
    """
    n = len(read)
    found: List[dict] = []
    chosen: List[Tuple[int, int, int, str]] = []  # (start, end, edits, cigar)

    def dfs(seg: int, p: int, total: int, worst: int) -> None:
        if len(found) >= limit:
            return
        if seg == copies:
            if p == n and (total, worst) == optimum:
                found.append(_build_witness(shift, ref, read, chosen))
            return
        remaining_after = copies - seg - 1
        lo = max(min_len, n - p - remaining_after * max_len)
        hi = min(max_len, n - p - remaining_after * min_len)
        for seg_len in range(lo, hi + 1):
            q = p + seg_len
            tail = suff[seg + 1].get(q)
            if tail is None:
                continue
            aligned = align_global(ref, read[p:q], cap)
            if aligned is None:
                continue
            dist, cigars = aligned
            cand_total = total + dist + tail[0]
            cand_worst = max(worst, dist, tail[1])
            if (cand_total, cand_worst) != optimum:
                continue
            for cigar in cigars:
                chosen.append((p, q, dist, cigar))
                dfs(seg + 1, q, total + dist, max(worst, dist))
                chosen.pop()
                if len(found) >= limit:
                    return

    dfs(0, 0, 0, 0)
    return found


def _diagnose_shift(
    ref: str, read: str, copies: int, length: int
) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
    """Cap-free nearest segmentation for a single rotation.

    Computes the edit distance of ``ref`` to every relevant read substring
    (one full Needleman-Wunsch matrix per start position, lengths from 1 to
    ``2 * length``), then runs a segmentation DP minimizing
    (total edits, max per-segment edits).  Returns the cost and boundaries
    of the lexicographically smallest optimal segmentation, or ``None`` when
    the read cannot hold ``copies`` non-empty segments.
    """
    n = len(read)
    m = length
    max_len = 2 * m
    if copies > n:
        return None

    inf = m * 2 + n + 1
    # costs[p][q] = edit distance between ref and read[p:q].
    costs: List[Dict[int, int]] = [dict() for _ in range(n)]
    for p in range(n):
        hi = min(n, p + max_len)
        piece = read[p:hi]
        t = len(piece)
        dp = [[inf] * (t + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(t + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, t + 1):
                sub = dp[i - 1][j - 1] + (
                    0 if ref[i - 1] == piece[j - 1] else 1
                )
                dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
        for j in range(1, t + 1):
            costs[p][p + j] = dp[m][j]

    # Segmentation DP with boundary predecessors (smallest boundary on ties).
    best: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(copies + 1)]
    prev: List[Dict[int, int]] = [dict() for _ in range(copies + 1)]
    best[0][0] = (0, 0)
    for seg in range(copies):
        remaining_after = copies - seg - 1
        for p, (total, worst) in best[seg].items():
            lo = max(1, n - p - remaining_after * max_len)
            hi = min(max_len, n - p - remaining_after)
            for seg_len in range(lo, hi + 1):
                q = p + seg_len
                dist = costs[p].get(q)
                if dist is None:
                    continue
                cand = (total + dist, max(worst, dist))
                old = best[seg + 1].get(q)
                if old is None or cand < old:
                    best[seg + 1][q] = cand
                    prev[seg + 1][q] = p
    final = best[copies].get(n)
    if final is None:
        return None
    boundaries: List[Tuple[int, int]] = []
    q = n
    for seg in range(copies, 0, -1):
        p = prev[seg][q]
        boundaries.append((p, q))
        q = p
    boundaries.reverse()
    return final, boundaries


def _parse_cigar(cigar: str) -> List[Tuple[int, str]]:
    ops: List[Tuple[int, str]] = []
    num = ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            ops.append((int(num), ch))
            num = ""
    return ops


def _replay(ref: str, piece: str, cigar: str) -> dict:
    """Replay a CIGAR into aligned reference/read strings.

    Gaps are ``-``.  ``alignment`` has the reference row, a marker row where
    spaces mark matches and ``^`` marks differences, and the read row.
    """
    ref_row: List[str] = []
    read_row: List[str] = []
    mark_row: List[str] = []
    i = j = 0
    for length, op in _parse_cigar(cigar):
        if op == "M":
            for _ in range(length):
                a, b = ref[i], piece[j]
                ref_row.append(a)
                read_row.append(b)
                mark_row.append(" " if a == b else "^")
                i += 1
                j += 1
        elif op == "D":
            for _ in range(length):
                ref_row.append(ref[i])
                read_row.append("-")
                mark_row.append("^")
                i += 1
        else:  # I
            for _ in range(length):
                ref_row.append("-")
                read_row.append(piece[j])
                mark_row.append("^")
                j += 1
    return {
        "aligned_reference": "".join(ref_row),
        "marker": "".join(mark_row),
        "aligned_read": "".join(read_row),
    }


def _build_witness(shift: int, ref: str, read: str, chosen) -> dict:
    segments = []
    for start, end, edits, cigar in chosen:
        piece = read[start:end]
        segment = {
            "start": start,
            "end": end,
            "reference": ref,
            "read": piece,
            "edits": edits,
            "cigar": cigar,
        }
        segment.update(_replay(ref, piece, cigar))
        segments.append(segment)
    return {
        "shift": shift,
        "boundaries": [[s["start"], s["end"]] for s in segments],
        "segments": segments,
    }


def _infeasible_envelope(
    reference: str,
    read: str,
    copies: int,
    max_edits: int,
    shift_best: List[Optional[Tuple[int, int]]],
    shift_tables: List[Optional[tuple]],
) -> dict:
    """Build a locatable constraint-failure envelope.

    The failure is always the per-segment edit budget.  To localize it we
    compute, per rotation, the edit distance from the reference to *every*
    read substring in one band-free DP per start position, then run a single
    segmentation DP without any per-segment cap.  That yields the globally
    nearest witness; the segments exceeding the requested budget are reported
    with the edit counts they actually needed.
    """
    length = len(reference)
    n = len(read)
    min_total = copies * max(1, length - max_edits)
    max_total = copies * (length + max_edits)
    # Structural bounds ignoring the edit budget: k non-empty global
    # alignments can cover any per-segment length from 1 to 2L.
    structurally_possible = copies <= n <= copies * 2 * length

    best = None  # (cost, shift, boundaries)
    if structurally_possible:
        for shift in range(length):
            ref = rotate(reference, shift)
            found = _diagnose_shift(ref, read, copies, length)
            if found is not None and (best is None or found[0] < best[0]):
                best = (found[0], shift, found[1])

    nearest = None
    if best is not None:
        cost, shift, boundaries = best
        ref = rotate(reference, shift)
        segments = []
        violating = []
        for idx, (start, end) in enumerate(boundaries):
            piece = read[start:end]
            aligned = align_global(ref, piece, length)
            dist = aligned[0]
            cigar = aligned[1][0]
            segments.append((start, end, dist, cigar))
            if dist > max_edits:
                violating.append(
                    {
                        "segment_index": idx,
                        "start": start,
                        "end": end,
                        "read": piece,
                        "required_edits": dist,
                        "budget": max_edits,
                        "over_by": dist - max_edits,
                        "cigar": cigar,
                    }
                )
        witness = _build_witness(shift, ref, read, segments)
        nearest = {
            "shift": shift,
            "total_edits": cost[0],
            "max_segment_edits": cost[1],
            "boundaries": witness["boundaries"],
            "violating_segments": violating,
        }

    if not structurally_possible:
        constraint_name = "segment_length"
        reason = (
            f"Read length {n} cannot be covered by {copies} non-empty "
            f"global alignments of the {length}-nt reference (each aligned "
            f"segment is 1..{2 * length} nt); this is structural, not a "
            "budget issue."
        )
    else:
        constraint_name = "per_segment_edit_budget"
        reason = (
            "No rotation admits exactly {k} non-empty consecutive segments "
            "with each segment within {e} edit(s).".format(
                k=copies, e=max_edits
            )
        )

    return {
        "status": "infeasible",
        "error": "constraint_failed",
        "message": reason,
        "constraint": {
            "name": constraint_name,
            "copies": copies,
            "max_edits_per_segment": max_edits,
            "reference_length": len(reference),
            "read_length": n,
            "feasible_read_length": [min_total, max_total],
        },
        "feasible_shifts": [s for s, b in enumerate(shift_best) if b is not None],
        "nearest": nearest,
    }
