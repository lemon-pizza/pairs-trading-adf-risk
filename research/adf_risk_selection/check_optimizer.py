"""Independent finite-graph and original-month checks for ratio-risk selection."""
from itertools import combinations
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from optimizer import solve, ratio_returns


class ArrayOracle:
    def __init__(self, pnl, capital):
        self.pnl, self.capital = np.asarray(pnl), np.asarray(capital)

    def ensure(self, ids):
        return None

    def arrays(self, ids):
        return self.pnl[:, ids], self.capital[:, ids]


def reference_risk(pnl, capital, ids):
    # Deliberately independent scalar sums and sample-variance arithmetic.
    path = []
    for t in range(len(pnl)):
        numerator = math.fsum(float(pnl[t, i]) for i in ids)
        denominator = math.fsum(float(capital[t, i]) for i in ids)
        path.append(numerator / denominator if denominator else 0.)
    mean = math.fsum(path) / len(path)
    return math.fsum((r-mean)**2 for r in path)/(len(path)-1)


def exhaustive(edges, oracle, baseline, cap):
    pairs = list(edges[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))
    budget = reference_risk(oracle.pnl, oracle.capital, baseline)
    best = -np.inf
    for ids in combinations(range(len(edges)), len(baseline)):
        counts = {}
        for i in ids:
            for a in pairs[i]:
                counts[a] = counts.get(a, 0) + 1
        if max(counts.values()) > cap:
            continue
        risk = reference_risk(oracle.pnl, oracle.capital, ids)
        if risk <= budget + 1e-16 + 1e-10*budget:
            best = max(best, -float(edges.iloc[list(ids)].T_STAT.sum()))
    return best, budget


def run(real_month=False):
    rng = np.random.default_rng(831)
    records = []
    for repetition in range(12):
        pairs = list(combinations(list('ABCDEF'), 2))
        edges = pd.DataFrame(pairs, columns=['ASSET_1', 'ASSET_2'])
        edges['T_STAT'] = -rng.uniform(1, 6, len(edges))
        capital = rng.lognormal(4, 1, size=(24, len(edges)))
        returns = rng.normal(0, .02, size=capital.shape)
        # One whole day with no holdings tests all-cash normalization.
        capital[3] = 0.
        oracle = ArrayOracle(returns*capital, capital)
        baseline = np.array([pairs.index(('A', 'B')), pairs.index(('C', 'D')), pairs.index(('E', 'F'))])
        previous = None
        for cap in (1, 2, 3):
            selected, meta = solve(edges, oracle, baseline, cap=cap, initial_indices=previous,
                                   time_limit=15, exact_combinations=10000)
            optimum, budget = exhaustive(edges, oracle, baseline, cap)
            np.testing.assert_allclose(meta['selected_score'], optimum, atol=1e-12, rtol=0)
            assert meta['certified_optimal'] and meta['feasible']
            assert reference_risk(oracle.pnl, oracle.capital, selected) <= budget + 1e-16 + 1e-10*budget
            assert meta['selected_score'] >= meta['initial_score']-1e-12
            previous = selected
            records.append({'repetition': repetition, 'cap': cap, 'score': meta['selected_score'],
                            'variance': meta['selected_variance'], 'exact_optimum': optimum})

    # Exercise the actual bounded-search path, including saturated degree-one
    # matching exchanges, rather than testing only the enumeration fallback.
    pairs = list(combinations(list('ABCDEFGH'), 2))
    edges = pd.DataFrame(pairs, columns=['ASSET_1', 'ASSET_2'])
    edges['T_STAT'] = -rng.uniform(1, 7, len(edges))
    capital = rng.lognormal(4, .8, size=(80, len(edges)))
    oracle = ArrayOracle(rng.normal(0, .02, capital.shape)*capital, capital)
    baseline = np.array([pairs.index(x) for x in [('A','B'), ('C','D'), ('E','F'), ('G','H')]])
    previous = None
    heuristic = []
    for cap in (1, 2, 3):
        selected, meta = solve(edges, oracle, baseline, cap=cap, initial_indices=previous,
                               time_limit=10, exact_combinations=0, structural_time_limit=0,
                               max_moves=30, scan_limit=1000, proposal_limit=1000)
        optimum, budget = exhaustive(edges, oracle, baseline, cap)
        assert meta['selected_score'] <= optimum+1e-12
        assert meta['selected_score'] >= meta['initial_score']-1e-12
        assert reference_risk(oracle.pnl, oracle.capital, selected) <= budget+1e-16+1e-10*budget
        heuristic.append(meta)
        previous = selected
    # A deterministic graph guarantees an improving degree-preserving exchange;
    # random risk constraints can legitimately reject every four-cycle.
    cycle_pairs = list(combinations(list('ABCD'), 2))
    cycle_edges = pd.DataFrame(cycle_pairs, columns=['ASSET_1', 'ASSET_2'])
    cycle_edges['T_STAT'] = [-1., -5., -2., -2., -5., -1.]
    cycle_capital = np.full((20, 6), 100.)
    cycle_returns = np.repeat(np.linspace(-.01, .01, 20)[:,None], 6, axis=1)
    cycle_oracle = ArrayOracle(cycle_returns*cycle_capital, cycle_capital)
    cycle_selected, cycle_meta = solve(cycle_edges, cycle_oracle, [0,5], cap=1,
                                       time_limit=5, exact_combinations=0, structural_time_limit=0)
    assert cycle_meta['accepted_four_cycles'] == 1
    assert cycle_meta['selected_score'] == 10.
    # Exercise structural MILP incumbent acceptance and its rigorous upper bound.
    selected, mip_meta = solve(edges, oracle, baseline, cap=2, time_limit=10,
                               exact_combinations=0, structural_time_limit=2, max_moves=0)
    optimum, _ = exhaustive(edges, oracle, baseline, 2)
    assert mip_meta['objective_upper_bound'] >= optimum-1e-10
    assert mip_meta['selected_score'] >= mip_meta['baseline_score']-1e-12

    invalid = 0
    for pnl, capital in [(np.array([1.]), np.array([0.])), (np.array([0.]), np.array([-1.])),
                         (np.array([np.inf]), np.array([1.]))]:
        try:
            ratio_returns(pnl, capital)
        except ValueError:
            invalid += 1
        else:
            raise AssertionError('Invalid aggregate should raise')
    np.testing.assert_array_equal(ratio_returns(np.zeros(3), np.zeros(3)), np.zeros(3))
    result = {'passed': True, 'exact_cases': records, 'heuristic_cases': heuristic,
              'structural_milp_case': mip_meta, 'deterministic_four_cycle': cycle_meta,
              'invalid_cases_rejected': invalid}
    if real_month:
        from features import prepare_month, HERE
        edges, baseline, oracle, metadata = prepare_month('20161230', HERE/'output')
        previous = None
        real = []
        for cap in (1, 2, 3):
            selected, receipt = solve(edges, oracle, baseline, cap=cap, initial_indices=previous,
                                      time_limit=8, max_moves=40)
            real.append(receipt)
            previous = selected
        result['real_month'] = real
    path = Path(__file__).with_name('optimizer_checks.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'exact_cases': len(records),
                      'heuristic_moves': [[m['accepted_single_swaps'],m['accepted_four_cycles']] for m in heuristic],
                      'real_month': [{k:m[k] for k in ['cap','baseline_score','selected_score','variance_ratio',
                                                     'certified_optimal','seconds','stop_reason']}
                                     for m in result.get('real_month', [])]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-month', action='store_true')
    run(parser.parse_args().real_month)
