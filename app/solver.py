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

With ``terminal_mode="partial"`` the first segment aligns to a non-empty
proper suffix ``R[a:]`` (1 <= a < L) and the last segment to a non-empty
proper prefix ``R[:b]`` (1 <= b < L) of the same rotated reference; the
middle segments still align to the full rotation.  The terminal cuts ``a``
and ``b`` are part of the joint choice and participate in the stable
tie-breaking immediately after their segment's CIGAR.
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


# ------------------------------------------------------------ partial termini


def align_terminal(
    ref: str, query: str, cap: int, prefix: bool
) -> Optional[Tuple[int, Tuple[Tuple[str, int], ...]]]:
    """Align ``query`` against the best non-empty proper terminal range.

    ``prefix=True``  considers every proper prefix ``R[:b]`` (1 <= b < L);
    ``prefix=False`` considers every proper suffix ``R[a:]`` (1 <= a < L).
    Each candidate is an ordinary global alignment (so CIGARs and replays
    share the full-copy semantics); the cut ``a``/``b`` is part of the joint
    choice and is returned beside every optimal CIGAR.

    Returns ``(edit_distance, ((cigar, cut), ...))`` with all optimal
    (CIGAR, cut) pairs sorted by ``(cigar_key, cut)``, or ``None`` when no
    admissible range aligns within ``cap`` edits.
    """
    m = len(ref)
    n = len(query)
    best: Optional[int] = None
    options: List[Tuple[str, int]] = []
    # Only spans within the cap-length band can be optimal; |span - n| > cap
    # is rejected inside align_global anyway, but narrowing the cuts here
    # keeps the call count small.
    lo_span = max(1, n - cap)
    hi_span = min(m - 1, n + cap)
    for span in range(lo_span, hi_span + 1):
        cut = span if prefix else m - span
        region = ref[:span] if prefix else ref[cut:]
        aligned = align_global(region, query, cap)
        if aligned is None:
            continue
        dist, cigars = aligned
        if best is None or dist < best:
            best = dist
            options = [(cigar, cut) for cigar in cigars]
        elif dist == best:
            options.extend((cigar, cut) for cigar in cigars)
    if best is None:
        return None
    ordered = tuple(
        sorted(set(options), key=lambda cc: (_cigar_sort_key(cc[0]), cc[1]))
    )
    return best, ordered


def _terminal_length_domain(length: int, cap: int) -> Tuple[int, int]:
    """Budgeted observable-length interval for a proper terminal range.

    A terminal range has reference span ``s`` with ``1 <= s <= L - 1``; within
    ``cap`` edits the observed segment lies in ``[max(1, s - cap), s + cap]``.
    The union over every admissible span is ``[1, L - 1 + cap]``.
    """
    return 1, length - 1 + cap


def _witness_key(witness: dict) -> Tuple:
    # Full copies carry no cut (constant -1); partial termini carry their
    # cut, so ties are broken by the terminal ranges right after the CIGARs.
    return (
        witness["shift"],
        tuple((s["start"], s["end"]) for s in witness["segments"]),
        tuple(
            _cigar_sort_key(s["cigar"]) + (s.get("reference_cut", -1),)
            for s in witness["segments"]
        ),
    )


def solve(
    reference: str,
    read: str,
    copies: int,
    max_edits: int,
    terminal_mode: str = "full",
) -> dict:
    """Run the full joint optimization.

    ``terminal_mode`` is ``"full"`` (every copy aligns to a complete rotated
    unit; the legacy behaviour) or ``"partial"`` (the first copy aligns to a
    proper suffix and the last to a proper prefix of the same rotation).

    Returns a result envelope with ``status`` of ``unique`` / ``ambiguous`` /
    ``infeasible``.
    """
    if terminal_mode not in ("full", "partial"):
        raise ValueError(f"unknown terminal_mode: {terminal_mode!r}")
    partial = terminal_mode == "partial"
    length = len(reference)
    n = len(read)
    if partial and copies < 2:
        raise ValueError("terminal_mode='partial' requires at least 2 copies")

    # Per-segment observable-length bounds, indexed by segment role.
    full_lo, full_hi = max(1, length - max_edits), length + max_edits
    term_lo, term_hi = _terminal_length_domain(length, max_edits)
    min_lens, max_lens = [], []
    for seg in range(copies):
        lo, hi = full_lo, full_hi
        if partial and (seg == 0 or seg == copies - 1):
            lo, hi = term_lo, term_hi
        min_lens.append(lo)
        max_lens.append(hi)

    # Per-shift optimum (total_edits, max_segment_edits); None == infeasible.
    shift_best: List[Optional[Tuple[int, int]]] = []
    shift_tables: List[Optional[tuple]] = []

    for shift in range(length):
        ref = rotate(reference, shift)
        pre, suff = _build_tables(
            ref, read, copies, max_edits, partial, min_lens, max_lens
        )
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
            shift_tables,
            partial,
            min_lens,
            max_lens,
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
            partial,
            min_lens,
            max_lens,
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
            "terminal_mode": terminal_mode,
            "objective": objective,
            "witnesses": candidates[:2],
            "more_witnesses": len(candidates) > 2,
        }
    return {
        "status": "unique",
        "terminal_mode": terminal_mode,
        "objective": objective,
        "witness": candidates[0],
    }


def _segment_role(seg: int, copies: int, partial: bool) -> str:
    """``"suffix"`` (first, partial), ``"prefix"`` (last, partial), else full."""
    if partial and seg == 0:
        return "suffix"
    if partial and seg == copies - 1:
        return "prefix"
    return "full"


def _align_segment(
    role: str, ref: str, piece: str, cap: int
) -> Optional[Tuple[int, List[Tuple[str, Optional[int]]]]]:
    """Best distance and all optimal ``(cigar, cut)`` options for one segment.

    ``cut`` is ``None`` for full copies and the terminal range cut
    (suffix start ``a`` / prefix end ``b``) otherwise.
    """
    if role == "full":
        aligned = align_global(ref, piece, cap)
        if aligned is None:
            return None
        dist, cigars = aligned
        return dist, [(cigar, None) for cigar in cigars]
    aligned = align_terminal(
        ref, piece, cap, prefix=(role == "prefix")
    )
    if aligned is None:
        return None
    dist, options = aligned
    return dist, list(options)


def _build_tables(
    ref: str,
    read: str,
    copies: int,
    cap: int,
    partial: bool,
    min_lens: List[int],
    max_lens: List[int],
) -> Tuple[List[Dict[int, Tuple[int, int]]], List[Dict[int, Tuple[int, int]]]]:
    """Prefix and suffix best-cost tables over (segment index, read offset).

    Cost is the lexicographic pair (total edits, max per-segment edits).
    """
    n = len(read)
    # Suffix aggregates of per-role length bounds, used to prune segment
    # lengths that cannot leave a coverable tail.
    suff_min = [0] * (copies + 1)
    suff_max = [0] * (copies + 1)
    for seg in range(copies - 1, -1, -1):
        suff_min[seg] = suff_min[seg + 1] + min_lens[seg]
        suff_max[seg] = suff_max[seg + 1] + max_lens[seg]

    # pre[seg][p] = best cost covering read[:p] with exactly seg segments.
    pre: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(copies + 1)]
    pre[0][0] = (0, 0)
    for seg in range(copies):
        role = _segment_role(seg, copies, partial)
        for p, (total, worst) in pre[seg].items():
            lo = max(min_lens[seg], n - p - suff_max[seg + 1])
            hi = min(max_lens[seg], n - p - suff_min[seg + 1])
            for seg_len in range(lo, hi + 1):
                aligned = _align_segment(role, ref, read[p : p + seg_len], cap)
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
        role = _segment_role(seg, copies, partial)
        for p in range(0, n + 1):
            best: Optional[Tuple[int, int]] = None
            lo = max(min_lens[seg], n - p - suff_max[seg + 1])
            hi = min(max_lens[seg], n - p - suff_min[seg + 1])
            for seg_len in range(lo, hi + 1):
                q = p + seg_len
                tail = suff[seg + 1].get(q)
                if tail is None:
                    continue
                aligned = _align_segment(role, ref, read[p:q], cap)
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
    partial: bool,
    min_lens: List[int],
    max_lens: List[int],
    pre: List[Dict[int, Tuple[int, int]]],
    suff: List[Dict[int, Tuple[int, int]]],
    optimum: Tuple[int, int],
    limit: Optional[int] = None,
) -> List[dict]:
    return list(
        _iter_witnesses(
            shift,
            ref,
            read,
            copies,
            cap,
            partial,
            min_lens,
            max_lens,
            pre,
            suff,
            optimum,
            limit,
        )
    )


def _iter_witnesses(
    shift: int,
    ref: str,
    read: str,
    copies: int,
    cap: int,
    partial: bool,
    min_lens: List[int],
    max_lens: List[int],
    pre: List[Dict[int, Tuple[int, int]]],
    suff: List[Dict[int, Tuple[int, int]]],
    optimum: Tuple[int, int],
    limit: Optional[int],
):
    """Yield optimal witnesses in the true stable order.

    The sort key is ``(boundaries, per-segment (CIGAR, terminal cut))``, so
    boundaries must be enumerated in ascending order *first*; only within one
    fixed boundary do (CIGAR, cut) combinations vary.  A plain alignment DFS
    interleaves earlier CIGAR choices with later boundaries and therefore
    does not emit that order -- here the two levels are separated:

    1. enumerate distinct optimal boundaries ascending (segment lengths
       ascending), pruning every edge that cannot still reach ``optimum``;
    2. for each boundary, enumerate its per-segment (CIGAR, cut) options in
       lexicographic order.
    """
    n = len(read)
    suff_min = [0] * (copies + 1)
    suff_max = [0] * (copies + 1)
    for seg in range(copies - 1, -1, -1):
        suff_min[seg] = suff_min[seg + 1] + min_lens[seg]
        suff_max[seg] = suff_max[seg + 1] + max_lens[seg]

    emitted = 0
    boundary_path: List[int] = []  # segment lengths

    def emit_boundary(bounds: List[int]):
        nonlocal emitted
        offsets = [0]
        for length in bounds:
            offsets.append(offsets[-1] + length)
        per_seg_options: List[List[Tuple[str, Optional[int]]]] = []
        dists: List[int] = []
        total = 0
        worst = 0
        for seg, length in enumerate(bounds):
            p, q = offsets[seg], offsets[seg + 1]
            role = _segment_role(seg, copies, partial)
            aligned = _align_segment(role, ref, read[p:q], cap)
            assert aligned is not None  # boundary DFS guaranteed feasibility
            dist, options = aligned
            dists.append(dist)
            total += dist
            worst = max(worst, dist)
            per_seg_options.append(options)
        if (total, worst) != optimum:
            return
        combo: List[Tuple[str, Optional[int]]] = []

        def walk_combos(seg: int):
            nonlocal emitted
            if limit is not None and emitted >= limit:
                return
            if seg == copies:
                chosen = []
                for idx, length in enumerate(bounds):
                    p, q = offsets[idx], offsets[idx + 1]
                    cigar, cut = combo[idx]
                    role = _segment_role(idx, copies, partial)
                    chosen.append((p, q, dists[idx], cigar, cut, role))
                emitted += 1
                produced.append(
                    _build_witness(shift, ref, read, chosen, partial)
                )
                return
            for option in per_seg_options[seg]:
                combo.append(option)
                walk_combos(seg + 1)
                combo.pop()
                if limit is not None and emitted >= limit:
                    return

        produced: List[dict] = []
        walk_combos(0)
        yield from produced

    def boundary_dfs(seg: int, p: int, total: int, worst: int):
        if limit is not None and emitted >= limit:
            return
        if seg == copies:
            if p == n and (total, worst) == optimum:
                yield from emit_boundary(list(boundary_path))
            return
        role = _segment_role(seg, copies, partial)
        lo = max(min_lens[seg], n - p - suff_max[seg + 1])
        hi = min(max_lens[seg], n - p - suff_min[seg + 1])
        for seg_len in range(lo, hi + 1):
            q = p + seg_len
            tail = suff[seg + 1].get(q)
            if tail is None:
                continue
            aligned = _align_segment(role, ref, read[p:q], cap)
            if aligned is None:
                continue
            dist = aligned[0]
            cand_total = total + dist + tail[0]
            cand_worst = max(worst, dist, tail[1])
            if (cand_total, cand_worst) != optimum:
                continue
            boundary_path.append(seg_len)
            yield from boundary_dfs(seg + 1, q, total + dist, max(worst, dist))
            boundary_path.pop()
            if limit is not None and emitted >= limit:
                return

    yield from boundary_dfs(0, 0, 0, 0)


def _diagnose_shift(
    ref: str, read: str, copies: int, length: int, partial: bool = False
) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
    """Cap-free nearest segmentation for a single rotation.

    Computes the edit distance from the reference to every relevant read
    substring (one full Needleman-Wunsch matrix per start position), then a
    segmentation DP minimizing (total edits, max per-segment edits).  In
    partial mode the first segment is scored against any non-empty proper
    suffix and the last against any non-empty proper prefix (an extra
    free-start matrix per start position); middle segments use the full unit.

    Returns the cost and boundaries of the lexicographically smallest
    optimal segmentation, or ``None`` when the read cannot hold ``copies``
    non-empty segments under the role-specific length caps.
    """
    n = len(read)
    m = length
    full_max = 2 * m
    term_max = 2 * (m - 1) if partial else full_max

    def role_of(seg: int) -> str:
        return _segment_role(seg, copies, partial)

    def seg_max_len(seg: int) -> int:
        return term_max if role_of(seg) in ("suffix", "prefix") else full_max

    min_total_len = copies  # every segment non-empty
    max_total_len = sum(seg_max_len(seg) for seg in range(copies))
    if not min_total_len <= n <= max_total_len:
        return None

    inf = m * 2 + n + 1

    def full_matrix(piece: str):
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
        return dp

    def suffix_matrix(piece: str):
        """Free-start NW that may begin only on a proper start 1 <= a < m."""
        t = len(piece)
        dp = [[inf] * (t + 1) for _ in range(m + 1)]
        for i in range(1, m):  # legal proper-suffix starts; a=0 and a=m barred
            dp[i][0] = 0
        for j in range(1, t + 1):
            for i in range(1, m + 1):
                sub = dp[i - 1][j - 1] + (
                    0 if ref[i - 1] == piece[j - 1] else 1
                )
                dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
        return dp

    # costs[p][q] = best edit distance for a segment read[p:q] of each role.
    costs_full: List[Dict[int, int]] = [dict() for _ in range(n)]
    costs_pre: List[Dict[int, int]] = [dict() for _ in range(n)]
    costs_suf: List[Dict[int, int]] = [dict() for _ in range(n)]
    for p in range(n):
        # Middle copies may span up to full_max even when termini are capped.
        hi = min(n, p + full_max)
        piece = read[p:hi]
        dp = full_matrix(piece)
        for j in range(1, len(piece) + 1):
            q = p + j
            costs_full[p][q] = dp[m][j]
            if partial and j <= term_max:
                costs_pre[p][q] = min(dp[b][j] for b in range(1, m))
        if partial:
            sdp = suffix_matrix(piece)
            for j in range(1, min(len(piece), term_max) + 1):
                costs_suf[p][p + j] = sdp[m][j]

    def cost_for(seg: int, p: int, q: int) -> Optional[int]:
        role = role_of(seg)
        table = {
            "full": costs_full,
            "prefix": costs_pre,
            "suffix": costs_suf,
        }[role]
        return table[p].get(q)

    # Segmentation DP with boundary predecessors (smallest boundary on ties).
    best: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(copies + 1)]
    prev: List[Dict[int, int]] = [dict() for _ in range(copies + 1)]
    best[0][0] = (0, 0)
    for seg in range(copies):
        role_max = seg_max_len(seg)
        tail_min = copies - seg - 1
        tail_max = sum(seg_max_len(s) for s in range(seg + 1, copies))
        for p, (total, worst) in best[seg].items():
            lo = max(1, n - p - tail_max)
            hi = min(role_max, n - p - tail_min)
            for seg_len in range(lo, hi + 1):
                q = p + seg_len
                dist = cost_for(seg, p, q)
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


def _build_witness(
    shift: int, ref: str, read: str, chosen, partial: bool = False
) -> dict:
    segments = []
    length = len(ref)
    for item in chosen:
        if len(item) == 4:  # legacy tuple shape
            start, end, edits, cigar = item
            cut, role = None, "full"
        else:
            start, end, edits, cigar, cut, role = item
        piece = read[start:end]
        if role == "suffix":
            region = ref[cut:]
            terminal_range = [cut, length]
        elif role == "prefix":
            region = ref[:cut]
            terminal_range = [0, cut]
        else:
            region = ref
            terminal_range = None
        segment = {
            "start": start,
            "end": end,
            "reference": region,
            "read": piece,
            "edits": edits,
            "cigar": cigar,
        }
        if terminal_range is not None:
            segment["role"] = role
            segment["reference_cut"] = cut
            segment["terminal_range"] = terminal_range
        segment.update(_replay(region, piece, cigar))
        segments.append(segment)
    witness = {
        "shift": shift,
        "boundaries": [[s["start"], s["end"]] for s in segments],
        "segments": segments,
    }
    if partial:
        witness["terminal_ranges"] = {
            # Rotated-reference coordinates of the two observed partial units.
            "first_suffix": segments[0]["terminal_range"],
            "last_prefix": segments[-1]["terminal_range"],
        }
    return witness


def _infeasible_envelope(
    reference: str,
    read: str,
    copies: int,
    max_edits: int,
    shift_best: List[Optional[Tuple[int, int]]],
    shift_tables: List[Optional[tuple]],
    partial: bool = False,
    min_lens: Optional[List[int]] = None,
    max_lens: Optional[List[int]] = None,
) -> dict:
    """Build a locatable constraint-failure envelope.

    The failure is the per-segment edit budget or (partial mode) the legal
    terminal-range geometry.  To localize it we compute, per rotation, the
    edit distance from the reference to every relevant read substring, then a
    segmentation DP without any per-segment cap.  That yields the globally
    nearest witness; the segments exceeding the requested budget are reported
    with the edit counts they actually needed.

    Constraint names:
      * ``segment_length``          - structural: no role-valid non-empty
        cover exists regardless of budget;
      * ``terminal_range``          - partial mode only: a terminal
        observation is too long to be any non-empty proper suffix/prefix
        within the edit budget (collection truncation geometry, not noise);
      * ``per_segment_edit_budget`` - noise: a legal cover exists but some
        segment needs more than ``max_edits`` edits.
    """
    length = len(reference)
    n = len(read)
    if min_lens is None:
        min_lens = [max(1, length - max_edits)] * copies
    if max_lens is None:
        max_lens = [length + max_edits] * copies
    min_total = sum(min_lens)
    max_total = sum(max_lens)
    term_lo, term_hi = _terminal_length_domain(length, max_edits)

    # Structural bounds ignoring the edit budget, per role.
    full_max = 2 * length
    term_max = 2 * (length - 1) if partial else full_max
    diag_max = sum(
        term_max
        if partial and (seg == 0 or seg == copies - 1)
        else full_max
        for seg in range(copies)
    )
    structurally_possible = copies <= n <= diag_max

    best = None  # (cost, shift, boundaries)
    if structurally_possible:
        for shift in range(length):
            ref = rotate(reference, shift)
            found = _diagnose_shift(ref, read, copies, length, partial)
            if found is None:
                continue
            if best is None or found[0] < best[0]:
                best = (found[0], shift, found[1])

    nearest = None
    terminal_blockers: List[dict] = []
    if best is not None:
        cost, shift, boundaries = best
        ref = rotate(reference, shift)
        # The diagnosis is deliberately budget-free, so re-align with a cap
        # that can never reject: unit indel/sub distance never exceeds the
        # sum of the two sequence lengths (<= 2L for middles, <= ~2L for a
        # 2(L-1)-long terminal observation).
        diag_cap = 2 * length + n
        segments = []
        violating = []
        for idx, (start, end) in enumerate(boundaries):
            piece = read[start:end]
            role = _segment_role(idx, copies, partial)
            cut: Optional[int] = None
            region = ref
            if role == "suffix":
                aligned = align_terminal(ref, piece, diag_cap, prefix=False)
                dist, options = aligned
                cigar, cut = options[0]
                region = ref[cut:]
            elif role == "prefix":
                aligned = align_terminal(ref, piece, diag_cap, prefix=True)
                dist, options = aligned
                cigar, cut = options[0]
                region = ref[:cut]
            else:
                aligned = align_global(ref, piece, diag_cap)
                dist = aligned[0]
                cigar = aligned[1][0]
            segments.append((start, end, dist, cigar, cut, role))
            if dist > max_edits:
                # A terminal segment whose observed length cannot be reached
                # by any proper range within the budget is a truncation
                # geometry failure rather than excess noise.
                geometric = role in ("suffix", "prefix") and not (
                    term_lo <= (end - start) <= term_hi
                )
                entry = {
                    "segment_index": idx,
                    "role": role,
                    "start": start,
                    "end": end,
                    "read": piece,
                    "required_edits": dist,
                    "budget": max_edits,
                    "over_by": dist - max_edits,
                    "cigar": cigar,
                    "geometric_terminal_failure": geometric,
                }
                if cut is not None:
                    entry["reference_cut"] = cut
                    if role == "suffix":
                        entry["terminal_range"] = [cut, length]
                    else:
                        entry["terminal_range"] = [0, cut]
                violating.append(entry)
                if geometric:
                    terminal_blockers.append(entry)
        witness = _build_witness(shift, ref, read, segments, partial)
        nearest = {
            "shift": shift,
            "total_edits": cost[0],
            "max_segment_edits": cost[1],
            "boundaries": witness["boundaries"],
            "violating_segments": violating,
        }
        if partial:
            nearest["terminal_ranges"] = witness["terminal_ranges"]

    if not structurally_possible:
        constraint_name = "segment_length"
        reason = (
            f"Read length {n} cannot be covered by {copies} non-empty "
            f"global alignments of the {length}-nt reference under "
            f"terminal_mode={'partial' if partial else 'full'}; this is "
            "structural, not a budget issue."
        )
    elif terminal_blockers:
        constraint_name = "terminal_range"
        blk = terminal_blockers[0]
        reason = (
            "The {role} observation read[{start}:{end}] ({ln} nt) cannot be "
            "any non-empty proper {kind} of a {length}-nt unit within "
            "{e} edit(s): budgeted terminal lengths are {lo}..{hi} nt "
            "(reference spans 1..{maxspan}). This indicates the collection "
            "window boundary is not a valid cyclic truncation.".format(
                role="first" if blk["role"] == "suffix" else "last",
                start=blk["start"],
                end=blk["end"],
                ln=blk["end"] - blk["start"],
                kind="suffix" if blk["role"] == "suffix" else "prefix",
                length=length,
                e=max_edits,
                lo=term_lo,
                hi=term_hi,
                maxspan=length - 1,
            )
        )
    else:
        constraint_name = "per_segment_edit_budget"
        reason = (
            "No rotation admits exactly {k} non-empty consecutive segments "
            "with each segment within {e} edit(s){mode}.".format(
                k=copies,
                e=max_edits,
                mode=" (proper terminal suffix/prefix)" if partial else "",
            )
        )

    constraint = {
        "name": constraint_name,
        "copies": copies,
        "max_edits_per_segment": max_edits,
        "reference_length": len(reference),
        "read_length": n,
        "feasible_read_length": [min_total, max_total],
    }
    if partial:
        constraint["terminal_mode"] = "partial"
        constraint["terminal_observed_length"] = [term_lo, term_hi]
        constraint["terminal_reference_span"] = [1, length - 1]

    return {
        "status": "infeasible",
        "error": "constraint_failed",
        "terminal_mode": "partial" if partial else "full",
        "message": reason,
        "constraint": constraint,
        "feasible_shifts": [s for s, b in enumerate(shift_best) if b is not None],
        "nearest": nearest,
    }
