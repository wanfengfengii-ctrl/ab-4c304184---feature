"""Core concatemer decoding logic.

Given a circular reference, a noisy tandem-read, a copy number ``k`` and a
per-copy edit budget ``cap``, jointly choose:

1. a cyclic shift (rotation) of the reference,
2. exactly ``k`` consecutive, non-empty segments covering the whole read,
3. a global (Needleman-Wunsch, unit indel/sub cost) alignment for every
   segment,

so that total edit count is minimized, ties broken by the maximum edit count
of any single segment.  Witnesses are stably ordered by
``(shift, boundaries, cigars)`` (full mode) and additionally the two terminal
reference ranges in partial terminal mode.

Terminal modes
--------------

``full`` (default, backwards compatible)
    Every segment is aligned globally against the whole rotated reference;
    each observed piece is one complete copy.

``partial``
    ``copies`` still counts consecutive observed pieces, but the acquisition
    window may start/end in the middle of a repeat unit:

    * the first piece must be a non-empty **proper suffix** of the rotated
      reference (reference range ``[L - a, L)`` for some 1 <= a <= L - 1);
    * the last piece must be a non-empty **proper prefix** (range
      ``[0, b)`` for some 1 <= b <= L - 1);
    * the ``copies - 2`` middle pieces are aligned against the full
      rotated reference as in full mode.
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

TERMINAL_FULL = "full"
TERMINAL_PARTIAL = "partial"


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


@lru_cache(maxsize=80_000)
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


# ------------------------------------------------------------- segment kinds


def _segment_kinds(copies: int, partial: bool) -> Tuple[str, ...]:
    """Kinds of the ``copies`` segments, in read order.

    ``middle`` segments align against the full rotated reference; ``first`` /
    ``last`` only exist in partial terminal mode and align against a proper
    suffix / prefix of the rotated reference.
    """
    if not partial:
        return ("middle",) * copies
    return ("first",) + ("middle",) * (copies - 2) + ("last",)


def _segment_reference(ref: str, kind: str, span: int) -> Tuple[str, Tuple[int, int]]:
    """The reference substring a segment aligns against, plus its range.

    ``span`` is the reference span length ``L - a`` (first) or ``b`` (last);
    middle segments always use the full reference and range ``[0, L)``.
    """
    length = len(ref)
    if kind == "first":
        return ref[length - span :], (length - span, length)
    if kind == "last":
        return ref[:span], (0, span)
    return ref, (0, length)


def _kind_extrema(kind: str, length: int, cap: int) -> Tuple[int, int]:
    """(min, max) observed piece length for a segment kind under the cap."""
    if kind == "middle":
        return max(1, length - cap), length + cap
    # Proper suffix/prefix: reference span 1..L-1, each within ``cap`` edits.
    return 1, length - 1 + cap


def _iter_aligned_pieces(
    ref: str,
    read: str,
    p: int,
    seg_len_lo: int,
    seg_len_hi: int,
    kind: str,
    cap: int,
):
    """Yield ``(q, dist, cigars, seg_ref, ref_range, span)`` for feasible pieces.

    Pieces are emitted with ascending end ``q`` and ascending reference span,
    which is the order witness reconstruction needs for stable output.
    """
    n = len(read)
    length = len(ref)
    seg_len_hi = min(seg_len_hi, n - p)
    if kind == "middle":
        for seg_len in range(seg_len_lo, seg_len_hi + 1):
            q = p + seg_len
            aligned = align_global(ref, read[p:q], cap)
            if aligned is not None:
                dist, cigars = aligned
                yield q, dist, cigars, ref, (0, length), length
        return

    # Terminal segment: jointly choose the observed length and the proper
    # suffix/prefix span; both must be within the per-segment edit cap.
    # The span iteration direction matches the stable witness ordering of
    # the resulting reference ranges (ranges sorted ascending): for the
    # first segment ascending ranges mean *descending* spans
    # (range [L-span, L)), for the last segment they mean ascending spans
    # (range [0, span)).  Emission order then agrees with the final sort,
    # which matters for the per-shift witness truncation.
    if kind == "first":
        spans = range(length - 1, 0, -1)
    else:
        spans = range(1, length)
    for seg_len in range(seg_len_lo, seg_len_hi + 1):
        q = p + seg_len
        piece = read[p:q]
        for span in spans:
            if abs(span - seg_len) > cap:
                continue
            seg_ref, ref_range = _segment_reference(ref, kind, span)
            aligned = align_global(seg_ref, piece, cap)
            if aligned is not None:
                dist, cigars = aligned
                yield q, dist, cigars, seg_ref, ref_range, span


# --------------------------------------------------------------------- solve


def _witness_key(witness: dict) -> Tuple:
    shift = witness["shift"]
    if witness.get("terminal_mode") == TERMINAL_PARTIAL:
        # Mirrors the reconstruction DFS order exactly (segment end, then the
        # aligned reference range, then the CIGAR, segment by segment), so the
        # first witnesses emitted per shift are the first witnesses after the
        # global stable sort.  Terminal ranges participate on equal footing
        # with the existing boundary/CIGAR elements.
        return (
            shift,
            tuple(
                (
                    s["end"],
                    tuple(s["ref_range"]),
                    _cigar_sort_key(s["cigar"]),
                )
                for s in witness["segments"]
            ),
        )
    return (
        shift,
        tuple((s["start"], s["end"]) for s in witness["segments"]),
        tuple(_cigar_sort_key(s["cigar"]) for s in witness["segments"]),
    )


def solve(
    reference: str,
    read: str,
    copies: int,
    max_edits: int,
    terminal_mode: str = TERMINAL_FULL,
) -> dict:
    """Run the full joint optimization.

    Returns a result envelope with ``status`` of ``unique`` / ``ambiguous`` /
    ``infeasible``.  ``terminal_mode`` is ``"full"`` (complete copies only) or
    ``"partial"`` (terminal pieces are a proper suffix/prefix pair).
    """
    partial = terminal_mode == TERMINAL_PARTIAL
    length = len(reference)
    n = len(read)
    if partial and copies < 2:
        raise ValueError("partial terminal mode requires at least 2 copies")
    kinds = _segment_kinds(copies, partial)

    # Per-shift optimum (total_edits, max_segment_edits); None == infeasible.
    shift_best: List[Optional[Tuple[int, int]]] = []
    shift_tables: List[Optional[tuple]] = []

    for shift in range(length):
        ref = rotate(reference, shift)
        pre, suff = _build_tables(ref, read, kinds, max_edits, length)
        shift_best.append(pre[copies].get(n))
        shift_tables.append((pre, suff))

    feasible = [b for b in shift_best if b is not None]
    if not feasible:
        return _infeasible_envelope(
            reference,
            read,
            copies,
            max_edits,
            shift_best,
            partial,
            kinds,
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
            kinds,
            max_edits,
            length,
            pre,
            suff,
            optimum,
            partial,
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
    kinds: Tuple[str, ...],
    cap: int,
    length: int,
) -> Tuple[List[Dict[int, Tuple[int, int]]], List[Dict[int, Tuple[int, int]]]]:
    """Prefix and suffix best-cost tables over (segment index, read offset).

    Cost is the lexicographic pair (total edits, max per-segment edits).
    """
    n = len(read)
    copies = len(kinds)
    extrema = [_kind_extrema(kind, length, cap) for kind in kinds]

    # pre[seg][p] = best cost covering read[:p] with exactly seg segments.
    pre: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(copies + 1)]
    pre[0][0] = (0, 0)
    for seg, kind in enumerate(kinds):
        rem_min = sum(mn for mn, _ in extrema[seg + 1 :])
        rem_max = sum(mx for _, mx in extrema[seg + 1 :])
        cmin, cmax = extrema[seg]
        for p, (total, worst) in pre[seg].items():
            lo = max(cmin, n - p - rem_max)
            hi = min(cmax, n - p - rem_min)
            if lo > hi:
                continue
            for q, dist, _cigars, _ref, _range, _span in _iter_aligned_pieces(
                ref, read, p, lo, hi, kind, cap
            ):
                cand = (total + dist, max(worst, dist))
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
        kind = kinds[seg]
        rem_min = sum(mn for mn, _ in extrema[seg + 1 :])
        rem_max = sum(mx for _, mx in extrema[seg + 1 :])
        cmin, cmax = extrema[seg]
        for p in range(0, n + 1):
            best: Optional[Tuple[int, int]] = None
            lo = max(cmin, n - p - rem_max)
            hi = min(cmax, n - p - rem_min)
            if lo > hi:
                continue
            for q, dist, _cigars, _ref, _range, _span in _iter_aligned_pieces(
                ref, read, p, lo, hi, kind, cap
            ):
                tail = suff[seg + 1].get(q)
                if tail is None:
                    continue
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
    kinds: Tuple[str, ...],
    cap: int,
    length: int,
    pre: List[Dict[int, Tuple[int, int]]],
    suff: List[Dict[int, Tuple[int, int]]],
    optimum: Tuple[int, int],
    partial: bool,
    limit: int,
) -> List[dict]:
    """Depth-first reconstruction of optimal witnesses in stable order.

    Segment lengths are tried ascending (terminal spans ascending within a
    fixed length) and CIGARs are already sorted, so the first emitted
    solutions are the smallest by (boundaries, cigars[, terminal ranges]).
    """
    n = len(read)
    copies = len(kinds)
    extrema = [_kind_extrema(kind, length, cap) for kind in kinds]
    found: List[dict] = []
    chosen: List[Tuple[int, int, int, str, str, Tuple[int, int]]] = []

    def dfs(seg: int, p: int, total: int, worst: int) -> None:
        if len(found) >= limit:
            return
        if seg == copies:
            if p == n and (total, worst) == optimum:
                found.append(
                    _build_witness(shift, ref, read, chosen, kinds, partial)
                )
            return
        kind = kinds[seg]
        rem_min = sum(mn for mn, _ in extrema[seg + 1 :])
        rem_max = sum(mx for _, mx in extrema[seg + 1 :])
        cmin, cmax = extrema[seg]
        lo = max(cmin, n - p - rem_max)
        hi = min(cmax, n - p - rem_min)
        if lo > hi:
            return
        for q, dist, cigars, seg_ref, ref_range, _span in _iter_aligned_pieces(
            ref, read, p, lo, hi, kind, cap
        ):
            tail = suff[seg + 1].get(q)
            if tail is None:
                continue
            cand_total = total + dist + tail[0]
            cand_worst = max(worst, dist, tail[1])
            if (cand_total, cand_worst) != optimum:
                continue
            for cigar in cigars:
                chosen.append((p, q, dist, cigar, seg_ref, ref_range))
                dfs(seg + 1, q, total + dist, max(worst, dist))
                chosen.pop()
                if len(found) >= limit:
                    return

    dfs(0, 0, 0, 0)
    return found


# -------------------------------------------------------- failure diagnostics


def _nw_matrix(ref: str, query: str) -> List[List[int]]:
    """Full (unbanded) Needleman-Wunsch distance matrix, unit edit costs."""
    m = len(ref)
    t = len(query)
    dp = [[0] * (t + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(t + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, t + 1):
            sub = dp[i - 1][j - 1] + (0 if ref[i - 1] == query[j - 1] else 1)
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    return dp


def _diagnose_shift(
    ref: str, read: str, copies: int, length: int
) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
    """Cap-free nearest segmentation for a single rotation (full mode).

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

    # costs[p][q] = edit distance between ref and read[p:q].
    costs: List[Dict[int, int]] = [dict() for _ in range(n)]
    for p in range(n):
        hi = min(n, p + max_len)
        piece = read[p:hi]
        t = len(piece)
        dp = _nw_matrix(ref, piece)
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


def _diagnose_shift_partial(
    ref: str, read: str, kinds: Tuple[str, ...], length: int
):
    """Cap-free nearest segmentation honouring proper suffix/prefix terminals.

    Unlike :func:`_diagnose_shift`, terminal pieces may be aligned against any
    non-empty proper suffix (first) / prefix (last) of the rotated reference,
    so the DP jointly chooses the reference span per terminal.  Returns
    ``(cost, choices)`` with one ``(start, end, kind, span)`` per segment, or
    ``None`` when even without an edit budget no legal segmentation exists.
    """
    n = len(read)
    copies = len(kinds)
    max_mid_len = 2 * length
    max_end_len = 2 * length - 1
    # Largest piece any segment may present, cap-free.
    if not copies <= n <= (copies - 2) * max_mid_len + 2 * max_end_len:
        return None

    # Middle costs: one full NW matrix per start position (as in full mode).
    mid_costs: List[Dict[int, int]] = [dict() for _ in range(n)]
    for p in range(n):
        hi = min(n, p + max_mid_len)
        dp = _nw_matrix(ref, read[p:hi])
        for j in range(1, hi - p + 1):
            mid_costs[p][p + j] = dp[length][j]

    # First-segment costs: distance(ref[L-span:], read[:q]) for every span.
    # One matrix per span, columns traversing the read forward, answers all q.
    first_costs: Dict[int, Dict[int, int]] = {}
    for span in range(1, length):
        dp = _nw_matrix(ref[length - span :], read)
        first_costs[span] = {q: dp[span][q] for q in range(1, n + 1)}

    # Last-segment costs: distance(ref[:span], read[p:q]).  One standard NW
    # matrix per start position p (reference vs read[p:]) answers every
    # proper-prefix span at row ``span`` and column q - p.
    last_costs: List[Dict[Tuple[int, int], int]] = [
        dict() for _ in range(n)
    ]
    for p in range(n):
        piece = read[p : p + max_end_len]
        dp = _nw_matrix(ref, piece)
        for span in range(1, length):
            for j in range(1, len(piece) + 1):
                last_costs[p][(span, p + j)] = dp[span][j]

    max_len_by_kind = {
        "first": max_end_len,
        "middle": max_mid_len,
        "last": max_end_len,
    }

    best: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(copies + 1)]
    # Predecessor: end -> (start, reference span of this segment).
    prev: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(copies + 1)]
    best[0][0] = (0, 0)
    for seg, kind in enumerate(kinds):
        remaining_after = copies - seg - 1
        max_len = max_len_by_kind[kind]
        min_len = 1
        for p, (total, worst) in best[seg].items():
            lo = max(min_len, n - p - remaining_after * max_mid_len)
            hi = min(max_len, n - p - remaining_after * min_len)
            for seg_len in range(lo, hi + 1):
                q = p + seg_len
                if kind == "middle":
                    dist = mid_costs[p].get(q)
                    if dist is None:
                        continue
                    span = length
                else:
                    # Pick the cheapest proper suffix/prefix span; on ties the
                    # smallest span gives a deterministic, locatable witness.
                    dist = None
                    span = None
                    if kind == "first":
                        for cand_span in range(1, length):
                            d = first_costs[cand_span][q]
                            if dist is None or d < dist:
                                dist, span = d, cand_span
                    else:
                        for cand_span in range(1, length):
                            d = last_costs[p].get((cand_span, q))
                            if d is None:
                                continue
                            if dist is None or d < dist:
                                dist, span = d, cand_span
                cand = (total + dist, max(worst, dist))
                old = best[seg + 1].get(q)
                if old is None or cand < old:
                    best[seg + 1][q] = cand
                    prev[seg + 1][q] = (p, span)
    final = best[copies].get(n)
    if final is None:
        return None

    choices: List[Tuple[int, int, str, int]] = []
    q = n
    for seg in range(copies, 0, -1):
        p, span = prev[seg][q]
        choices.append((p, q, kinds[seg - 1], span))
        q = p
    choices.reverse()
    return final, choices


# ----------------------------------------------------------------- rendering


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


def _build_witness(
    shift: int, ref: str, read: str, chosen, kinds: Tuple[str, ...], partial: bool
) -> dict:
    segments = []
    for idx, (start, end, edits, cigar, seg_ref, ref_range) in enumerate(chosen):
        piece = read[start:end]
        segment = {
            "start": start,
            "end": end,
            "reference": seg_ref,
            "read": piece,
            "edits": edits,
            "cigar": cigar,
        }
        if partial:
            segment["ref_range"] = [ref_range[0], ref_range[1]]
            if kinds[idx] != "middle":
                segment["role"] = kinds[idx]
        segment.update(_replay(seg_ref, piece, cigar))
        segments.append(segment)
    witness = {
        "shift": shift,
        "boundaries": [[s["start"], s["end"]] for s in segments],
        "segments": segments,
    }
    if partial:
        witness["terminal_mode"] = TERMINAL_PARTIAL
        witness["terminal_ranges"] = {
            "first": [segments[0]["ref_range"][0], segments[0]["ref_range"][1]],
            "last": [segments[-1]["ref_range"][0], segments[-1]["ref_range"][1]],
        }
    return witness


def _infeasible_envelope(
    reference: str,
    read: str,
    copies: int,
    max_edits: int,
    shift_best: List[Optional[Tuple[int, int]]],
    partial: bool,
    kinds: Tuple[str, ...],
) -> dict:
    """Build a locatable constraint-failure envelope.

    Full mode is documented below; partial mode additionally allows the
    terminal pieces to align against proper suffix/prefix spans when looking
    for the nearest witness.  Two distinct structural failures exist there:
    the generic length impossibility and ``terminal_prefix_suffix`` (no
    non-empty proper suffix/prefix can cover the ends).
    """
    length = len(reference)
    n = len(read)

    if not partial:
        return _infeasible_envelope_full(
            reference, read, copies, max_edits, shift_best
        )

    min_total = 2 + (copies - 2) * max(1, length - max_edits)
    max_total = 2 * (length - 1 + max_edits) + (copies - 2) * (length + max_edits)
    # Structural bounds ignoring the edit budget.
    structurally_possible = (
        copies
        <= n
        <= (copies - 2) * 2 * length + 2 * (2 * length - 1)
    )

    best = None  # (cost, shift, choices)
    if structurally_possible:
        for shift in range(length):
            ref = rotate(reference, shift)
            found = _diagnose_shift_partial(ref, read, kinds, length)
            if found is not None and (best is None or found[0] < best[0]):
                best = (found[0], shift, found[1])

    nearest = None
    if best is not None:
        cost, shift, choices = best
        ref = rotate(reference, shift)
        segments = []
        violating = []
        for idx, (start, end, kind, span) in enumerate(choices):
            piece = read[start:end]
            seg_ref, ref_range = _segment_reference(ref, kind, span)
            # Any piece shorter than 2L against a reference of <= L has
            # distance below 2L; this cap only disables the band.
            aligned = align_global(seg_ref, piece, 2 * length)
            dist = aligned[0]
            cigar = aligned[1][0]
            segments.append((start, end, dist, cigar, seg_ref, ref_range))
            if dist > max_edits:
                violation = {
                    "segment_index": idx,
                    "start": start,
                    "end": end,
                    "read": piece,
                    "required_edits": dist,
                    "budget": max_edits,
                    "over_by": dist - max_edits,
                    "cigar": cigar,
                    "ref_range": [ref_range[0], ref_range[1]],
                    "role": "middle" if kind == "middle" else kind,
                }
                violating.append(violation)
        witness = _build_witness(
            shift,
            ref,
            read,
            [(s, e, d, c, sr, rr) for s, e, d, c, sr, rr in segments],
            kinds,
            True,
        )
        nearest = {
            "shift": shift,
            "total_edits": cost[0],
            "max_segment_edits": cost[1],
            "boundaries": witness["boundaries"],
            "terminal_ranges": witness["terminal_ranges"],
            "violating_segments": violating,
        }

    if not structurally_possible:
        constraint_name = "terminal_prefix_suffix"
        reason = (
            f"Read length {n} cannot be covered by {copies} consecutive "
            f"observed pieces of the {length}-nt reference in partial "
            "terminal mode: the first piece must be a non-empty proper "
            "suffix and the last a non-empty proper prefix (terminal "
            f"pieces are 1..{2 * length - 1} nt, middle pieces "
            f"1..{2 * length} nt). The acquisition window cannot frame "
            f"{copies} pieces; this is collection truncation/framing, not "
            "a noise-budget issue."
        )
    else:
        constraint_name = "per_segment_edit_budget"
        reason = (
            "No rotation admits the requested suffix-headed, "
            "prefixed-tailed {k}-piece cover with every piece within {e} "
            "edit(s).".format(k=copies, e=max_edits)
        )

    return {
        "status": "infeasible",
        "error": "constraint_failed",
        "message": reason,
        "terminal_mode": TERMINAL_PARTIAL,
        "constraint": {
            "name": constraint_name,
            "copies": copies,
            "max_edits_per_segment": max_edits,
            "reference_length": length,
            "read_length": n,
            "feasible_read_length": [min_total, max_total],
            "terminal_requirement": {
                "first": "non-empty proper suffix of the rotated reference",
                "last": "non-empty proper prefix of the rotated reference",
            },
        },
        "feasible_shifts": [s for s, b in enumerate(shift_best) if b is not None],
        "nearest": nearest,
    }


def _infeasible_envelope_full(
    reference: str,
    read: str,
    copies: int,
    max_edits: int,
    shift_best: List[Optional[Tuple[int, int]]],
) -> dict:
    """Build a locatable constraint-failure envelope for full terminal mode.

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
        witness = _build_witness(
            shift,
            ref,
            read,
            [(s, e, d, c, ref, (0, length)) for s, e, d, c in segments],
            ("middle",) * copies,
            False,
        )
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
