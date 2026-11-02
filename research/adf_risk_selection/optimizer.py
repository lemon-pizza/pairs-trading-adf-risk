"""ADF-score selection with the source's selection-dependent capital weights.

Only pair selection is changed.  For a support S, the historical risk is the
sample variance of sum(PnL[:, S]) / sum(capital[:, S]).  In particular this is
NOT a fixed-weight covariance quadratic and no convex covariance cuts are used.

The entire supplied graph is feasible.  Small problems are enumerated exactly;
large problems use an optional full-graph structural MILP followed by bounded
one-edge and alternating four-cycle searches.  Search limits are disclosed and
are never represented as global optimality certificates.  The source matching
is retained as a feasible incumbent.  Successive degree limits may warm-start
from the previous result without reducing its achieved ADF score.
"""
from __future__ import annotations

from itertools import combinations
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


def _ids(values, n, name):
    raw = np.asarray(values)
    if (raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.number)
            or not np.isfinite(raw).all() or np.any(raw != np.floor(raw))):
        raise ValueError(f"{name} must be a nonempty integer vector")
    ids = np.sort(raw.astype(int))
    if ids[0] < 0 or ids[-1] >= n or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Invalid {name}")
    return ids


def ratio_returns(pnl, capital):
    """Ratio of aggregate source outputs; no-capital days are zero for risk."""
    p, c = np.asarray(pnl, float), np.asarray(capital, float)
    if p.shape != c.shape or p.ndim not in (1, 2):
        raise ValueError("PnL/capital shapes must agree and have one or two axes")
    if not np.isfinite(p).all() or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("Risk inputs must be finite, with nonnegative capital")
    if np.any((c == 0) & (p != 0)):
        raise ValueError("Nonzero PnL without capital cannot be treated as cash")
    out = np.zeros_like(p)
    np.divide(p, c, out=out, where=c > 0)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite aggregate ratio return")
    return out


def support_state(oracle, ids):
    """Use direct sums for final acceptance; no incremental risk approximation."""
    p, c = oracle.arrays(np.asarray(ids, int))
    p, c = np.asarray(p, float), np.asarray(c, float)
    if p.ndim != 2 or p.shape != c.shape or p.shape[1] != len(ids) or len(p) < 2:
        raise ValueError("Oracle must return (T, number of IDs) arrays, T >= 2")
    if not np.isfinite(p).all() or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("Invalid oracle arrays")
    pn, ca = p.sum(axis=1), c.sum(axis=1)
    r = ratio_returns(pn, ca)
    return pn, ca, r, float(np.var(r, ddof=1))


def solve(edges, oracle, baseline_indices, cap=2, initial_indices=None,
          time_limit=8.0, max_moves=40, variance_rtol=1e-10,
          variance_atol=1e-16, structural_time_limit=None,
          exact_combinations=20000, proposal_limit=1024,
          scan_limit=20000, removal_choices=8, batch_size=64):
    """Return selected row indices and an auditable feasibility/search receipt.

    time_limit includes feature construction requested by this call, but a
    currently executing oracle/MILP operation may overrun it.  Every candidate
    adopted is checked by its actual ratio-return path.  variance_rtol/atol are
    numerical tolerances, not an economic risk-budget relaxation.
    """
    begun = time.monotonic()
    if isinstance(cap, bool) or int(cap) != cap or cap < 1:
        raise ValueError("cap must be a positive integer")
    cap = int(cap)
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    for value, name in [(variance_rtol, "variance_rtol"), (variance_atol, "variance_atol")]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for value, name in [(max_moves, "max_moves"), (exact_combinations, "exact_combinations"),
                        (proposal_limit, "proposal_limit"), (scan_limit, "scan_limit"),
                        (removal_choices, "removal_choices"), (batch_size, "batch_size")]:
        if isinstance(value, bool) or int(value) != value or value < (0 if name in ("max_moves", "exact_combinations") else 1):
            raise ValueError(f"Invalid {name}")
    pairs = list(edges[["ASSET_1", "ASSET_2"]].itertuples(index=False, name=None))
    n = len(pairs)
    score = -edges.T_STAT.to_numpy(float)
    if not n or not np.isfinite(score).all():
        raise ValueError("Expected a nonempty graph with finite ADF scores")
    unordered = [frozenset(p) for p in pairs]
    if len(set(unordered)) != n or any(len(p) != 2 for p in unordered):
        raise ValueError("Use a simple graph with one orientation per unordered pair")
    nodes = sorted({x for pair in pairs for x in pair})
    node_map = {x: j for j, x in enumerate(nodes)}
    left = np.array([node_map[a] for a, b in pairs], int)
    right = np.array([node_map[b] for a, b in pairs], int)
    lookup = {frozenset((a, b)): i for i, (a, b) in enumerate(zip(left, right))}
    baseline = _ids(baseline_indices, n, "baseline_indices")
    k = len(baseline)

    def degrees(ids):
        return np.bincount(np.r_[left[ids], right[ids]], minlength=len(nodes))

    if degrees(baseline).max() > 1:
        raise ValueError("The risk anchor must be a nonoverlapping matching")
    base_p, base_c, base_r, budget = support_state(oracle, baseline)
    tolerance = variance_atol + variance_rtol * budget
    base_score = float(score[baseline].sum())
    incumbent = baseline.copy()
    pn, ca, returns, variance, objective = base_p, base_c, base_r, budget, base_score
    if initial_indices is not None:
        start = _ids(initial_indices, n, "initial_indices")
        if len(start) != k or degrees(start).max() > cap:
            raise ValueError("Initial support violates cardinality or degree cap")
        sp, sc, sr, sv = support_state(oracle, start)
        if sv > budget + tolerance:
            raise ValueError("Initial support exceeds the common source risk budget")
        so = float(score[start].sum())
        if so > objective:
            incumbent, pn, ca, returns, variance, objective = start, sp, sc, sr, sv, so
    initial_objective = objective
    gain_tolerance = 64 * np.finfo(float).eps * max(1.0, np.abs(score).sum())
    # Ignoring structural/risk constraints gives a valid upper bound.
    upper_bound = float(np.sort(score)[-k:].sum())
    receipt = {
        "solver": "exact_enumeration_or_bounded_ratio_risk_search",
        "objective_definition": "sum(-T_STAT) over selected pairs; no expected-return objective",
        "risk_definition": "sample variance ddof=1 of sum(source_pnl)/sum(source_capital)",
        "zero_capital_risk_rule": "zero return only when aggregate PnL is also zero",
        "candidate_edges": n, "candidate_nodes": len(nodes), "candidate_truncation": False,
        "K": k, "cap": cap, "baseline_score": base_score,
        "initial_score": initial_objective, "baseline_variance": budget,
        "variance_rtol": variance_rtol, "variance_atol": variance_atol,
        "risk_numerical_tolerance": tolerance,
        "time_limit": float(time_limit), "max_moves": int(max_moves),
        "search_proposal_limit_per_move": int(proposal_limit),
        "search_edge_scan_limit_per_move": int(scan_limit),
        "search_unconstrained_removal_choices": int(removal_choices),
        "search_limits_are_not_feasible_set_restrictions": True,
        "evaluated_supports": 0, "risk_feasible_supports": 0,
        "accepted_single_swaps": 0, "accepted_four_cycles": 0,
        "structural_milp": None, "certified_optimal": False,
    }
    history = [{"kind": "initial", "score": objective, "variance": variance}]
    deadline = begun + time_limit
    stop = "time_limit"

    def accept(ids, kind):
        nonlocal incumbent, pn, ca, returns, variance, objective
        ids = np.sort(np.asarray(ids, int))
        if len(ids) != k or len(set(ids)) != k or degrees(ids).max() > cap:
            raise RuntimeError("Internal proposal violated structural constraints")
        obj = float(score[ids].sum())
        if obj <= objective + gain_tolerance:
            return False
        pp, cc, rr, vv = support_state(oracle, ids)
        receipt["evaluated_supports"] += 1
        if vv > budget + tolerance:
            return False
        receipt["risk_feasible_supports"] += 1
        incumbent, pn, ca, returns, variance, objective = ids, pp, cc, rr, vv, obj
        history.append({"kind": kind, "score": objective, "variance": variance})
        return True

    # Exhaustive mode is useful for small applications and independent tests.
    exact_count = math.comb(n, k)
    if exact_count <= exact_combinations:
        oracle.ensure(np.arange(n))
        complete = True
        for ids in combinations(range(n), k):
            if time.monotonic() >= deadline:
                complete = False
                break
            ids = np.asarray(ids, int)
            if degrees(ids).max() <= cap:
                accept(ids, "enumeration")
        receipt["enumeration_combinations"] = exact_count
        receipt["certified_optimal"] = complete
        stop = "exact_enumeration_complete" if complete else "enumeration_time_limit"
        if complete:
            upper_bound = objective
    else:
        # This linear model relaxes only the nonlinear risk constraint.  Its
        # incumbent is adopted only after exact risk validation.  Its dual bound
        # is a valid upper bound for the risk-constrained maximization problem.
        structural_seconds = min(2.0, time_limit * .25) if structural_time_limit is None else float(structural_time_limit)
        if not np.isfinite(structural_seconds) or structural_seconds < 0:
            raise ValueError("structural_time_limit must be finite and nonnegative")
        remaining = deadline - time.monotonic()
        if structural_seconds > 0 and remaining > .05:
            incident = coo_matrix((np.ones(2*n), (np.r_[left, right], np.tile(np.arange(n), 2))),
                                  shape=(len(nodes), n)).tocsc()
            cardinal = coo_matrix((np.ones(n), (np.zeros(n, int), np.arange(n))), shape=(1, n))
            matrix = vstack([incident, cardinal], format="csc")
            result = milp(-score, integrality=np.ones(n, np.uint8), bounds=Bounds(0., 1.),
                          constraints=LinearConstraint(matrix, np.r_[np.zeros(len(nodes)), k],
                                                       np.r_[np.full(len(nodes), cap), k]),
                          options={"time_limit": min(structural_seconds, remaining), "mip_rel_gap": 0.0})
            dual = getattr(result, "mip_dual_bound", None)
            if dual is not None and np.isfinite(dual):
                candidate_bound = -float(dual)
                if candidate_bound >= objective - gain_tolerance:
                    upper_bound = min(upper_bound, max(candidate_bound, objective))
            item = {"status": int(result.status), "message": str(result.message),
                    "dual_score_upper_bound": None if dual is None or not np.isfinite(dual) else -float(dual),
                    "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
                    "risk_feasible_incumbent": False}
            if result.x is not None and np.max(np.abs(result.x - np.round(result.x))) <= 1e-6:
                mids = np.flatnonzero(result.x > .5)
                if len(mids) == k and degrees(mids).max() <= cap:
                    item["incumbent_score"] = float(score[mids].sum())
                    item["adopted"] = accept(mids, "structural_milp")
                    if np.array_equal(mids, incumbent):
                        item["risk_feasible_incumbent"] = True
            receipt["structural_milp"] = item

        order = np.argsort(-score, kind="stable")
        for move in range(max_moves):
            if upper_bound - objective <= gain_tolerance:
                receipt["certified_optimal"] = True
                stop = "valid_upper_bound_closed"
                break
            if time.monotonic() >= deadline:
                break
            mask = np.zeros(n, bool)
            mask[incumbent] = True
            deg = degrees(incumbent)
            incident_ids = [[] for _ in nodes]
            for i in incumbent:
                incident_ids[left[i]].append(int(i))
                incident_ids[right[i]].append(int(i))
            inexpensive = incumbent[np.argsort(score[incumbent], kind="stable")[:removal_choices]]
            # Candidate proposals are ranked by the actual ADF objective gain.
            # Their risk is never approximated by ADF or individual variance.
            proposals, seen = [], set()
            scanned = 0

            def propose(remove, add, kind):
                remove, add = tuple(sorted(remove)), tuple(sorted(add))
                if len(set(remove)) != len(remove) or len(set(add)) != len(add):
                    return
                key = (remove, add)
                if key in seen or any(mask[j] for j in add):
                    return
                gain = float(score[list(add)].sum() - score[list(remove)].sum())
                if gain > gain_tolerance:
                    seen.add(key)
                    proposals.append((gain, remove, add, kind))

            for add in order:
                if mask[add]:
                    continue
                scanned += 1
                if scanned > scan_limit or (scanned % 128 == 0 and time.monotonic() >= deadline):
                    break
                a, b = left[add], right[add]
                saturated_a, saturated_b = deg[a] >= cap, deg[b] >= cap
                if saturated_a and saturated_b:
                    removals = set(incident_ids[a]).intersection(incident_ids[b])
                elif saturated_a:
                    removals = incident_ids[a]
                elif saturated_b:
                    removals = incident_ids[b]
                else:
                    removals = inexpensive
                for remove in removals:
                    propose((int(remove),), (int(add),), "single_swap")
                # Removing (a,c),(b,d) and adding (a,b),(c,d) preserves every
                # node's degree, including saturated degree-one matchings.
                for i in incident_ids[a]:
                    c = right[i] if left[i] == a else left[i]
                    for j in incident_ids[b]:
                        d = right[j] if left[j] == b else left[j]
                        if len({int(a), int(b), int(c), int(d)}) != 4:
                            continue
                        other = lookup.get(frozenset((int(c), int(d))))
                        if other is not None:
                            propose((i, j), (int(add), other), "four_cycle")
            receipt["last_scanned_unselected_edges"] = min(scanned, scan_limit)
            proposals.sort(key=lambda x: (-x[0], x[1], x[2]))
            total_proposals = len(proposals)
            proposals = proposals[:proposal_limit]
            receipt["last_improving_proposals_generated"] = total_proposals
            if not proposals:
                stop = "no_improving_proposal_in_bounded_search"
                break
            adopted = False
            for offset in range(0, len(proposals), batch_size):
                if time.monotonic() >= deadline:
                    break
                batch = proposals[offset:offset+batch_size]
                union = np.array(sorted({i for _, remove, add, _ in batch for i in (*remove, *add)}), int)
                pp, cc = oracle.arrays(union)
                column = {i: j for j, i in enumerate(union)}
                p = np.repeat(pn[:, None], len(batch), axis=1)
                c = np.repeat(ca[:, None], len(batch), axis=1)
                for j, (_, remove, add, _) in enumerate(batch):
                    p[:, j] += pp[:, [column[i] for i in add]].sum(axis=1) - pp[:, [column[i] for i in remove]].sum(axis=1)
                    c[:, j] += cc[:, [column[i] for i in add]].sum(axis=1) - cc[:, [column[i] for i in remove]].sum(axis=1)
                # Incremental zero-capital cancellation can have rounding noise.
                # Such columns are sent to a direct sum instead of silently
                # changing the zero/near-zero denominator of a return.
                uncertain = np.any(c <= np.finfo(float).eps * np.maximum(ca[:, None], 1.) * 64, axis=0)
                valid = ~uncertain
                variances = np.full(len(batch), np.inf)
                if valid.any():
                    rr = p[:, valid] / c[:, valid]
                    variances[valid] = np.var(rr, axis=0, ddof=1)
                receipt["evaluated_supports"] += int(valid.sum())
                for j, (_, remove, add, kind) in enumerate(batch):
                    if uncertain[j] or variances[j] <= budget + tolerance:
                        ids = np.array(sorted((set(incumbent) - set(remove)) | set(add)), int)
                        if accept(ids, kind):
                            receipt["accepted_single_swaps" if kind == "single_swap" else "accepted_four_cycles"] += 1
                            adopted = True
                            break
                if adopted:
                    break
            if not adopted:
                stop = "time_limit" if time.monotonic() >= deadline else "no_feasible_improvement_in_bounded_search"
                break
        else:
            stop = "max_moves"

    # Independent direct re-evaluation is mandatory even if search found no move.
    final_p, final_c, final_r, final_variance = support_state(oracle, incumbent)
    final_degrees = degrees(incumbent)
    if final_variance > budget + tolerance or final_degrees.max() > cap or len(incumbent) != k:
        raise RuntimeError("Final support failed feasibility validation")
    if objective < initial_objective - gain_tolerance:
        raise RuntimeError("Search worsened its feasible incumbent")
    if upper_bound - objective <= gain_tolerance:
        receipt["certified_optimal"] = True
    receipt.update({
        "selected_score": objective, "selected_variance": final_variance,
        "variance_ratio": None if budget == 0 else final_variance / budget,
        "risk_slack": budget - final_variance,
        "max_degree": int(final_degrees.max()), "selected_nodes": int(np.count_nonzero(final_degrees)),
        "selected_edges": len(incumbent), "risk_train_days": len(final_r),
        "zero_capital_train_days": int(np.count_nonzero(final_c == 0)),
        "objective_upper_bound": max(upper_bound, objective),
        "objective_gap_upper_bound": max(upper_bound-objective, 0.),
        "feasible": True, "stop_reason": stop,
        "seconds": time.monotonic()-begun, "history": history,
        "qualification": "Global optimality only when certified_optimal is true; otherwise a feasible bounded-search incumbent.",
    })
    return incumbent, receipt
