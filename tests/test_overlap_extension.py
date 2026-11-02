"""Meaningful boundaries for the unlimited-overlap extension."""
from itertools import combinations
import unittest

import numpy as np
import pandas as pd

from release.extend_overlap import ROOT, cap_for_model, selection_ids, writable_output
from optimizer import solve


class ArrayOracle:
    def __init__(self, edges):
        self.capital = np.ones((6, edges))
        self.pnl = np.tile(np.array([.01, -.01, .02, -.02, .01, -.01])[:, None], (1, edges))

    def ensure(self, ids):
        pass

    def arrays(self, ids):
        return self.pnl[:, ids], self.capital[:, ids]


class ExtensionChecks(unittest.TestCase):
    def test_unlimited_cap_is_redundant_and_exact_on_star_optimum(self):
        baseline_pairs = [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G', 'H'), ('I', 'J'), ('K', 'L')]
        pairs = baseline_pairs + [('A', node) for node in 'CEGIK']
        edges = pd.DataFrame(pairs, columns=['ASSET_1', 'ASSET_2'])
        edges['T_STAT'] = [-20. if x == 'A' else -1. for x, y in pairs]
        baseline = np.arange(len(baseline_pairs))
        k = len(baseline)
        for support in combinations(range(len(pairs)), k):
            degree = pd.Series([x for i in support for x in pairs[i]]).value_counts()
            self.assertLessEqual(degree.max(), cap_for_model('ADF_RISK_UNLIMITED', k))
        answer, info = solve(edges, ArrayOracle(len(pairs)), baseline,
                             cap=cap_for_model('ADF_RISK_UNLIMITED', k), time_limit=10.,
                             variance_rtol=1e-8, variance_atol=1e-16)
        self.assertTrue(info['certified_optimal'])
        self.assertAlmostEqual(-edges.iloc[answer].T_STAT.sum(), 120.)
        self.assertEqual(info['max_degree'], 6)
        capped, capped_info = solve(edges, ArrayOracle(len(pairs)), baseline, cap=5,
                                    time_limit=10., variance_rtol=1e-8, variance_atol=1e-16)
        self.assertTrue(capped_info['certified_optimal'])
        self.assertAlmostEqual(-edges.iloc[capped].T_STAT.sum(), 101.)
        self.assertLess(capped_info['selected_score'], info['selected_score'])

    def test_frozen_orientations_and_row_ids_cannot_silently_drift(self):
        edges = pd.DataFrame({'ASSET_1': ['A', 'B'], 'ASSET_2': ['C', 'D'], 'T_STAT': [-3., -4.]})
        selected = edges.iloc[[1, 0]].copy()
        selected['CANDIDATE_INDEX'] = [1, 0]
        np.testing.assert_array_equal(selection_ids(edges, selected), [1, 0])
        selected.loc[1, 'ASSET_1'], selected.loc[1, 'ASSET_2'] = 'D', 'B'
        with self.assertRaises(ValueError):
            selection_ids(edges, selected)

    def test_extension_cannot_overwrite_published_strategy(self):
        with self.assertRaises(ValueError):
            writable_output(ROOT / 'research/overlap_extension/output')
        self.assertEqual(writable_output(ROOT / 'work/overlap_extension'), ROOT / 'work/overlap_extension')
        self.assertEqual(cap_for_model('ADF_RISK4', 215), 4)
        self.assertEqual(cap_for_model('ADF_RISK5', 215), 5)
        self.assertEqual(cap_for_model('ADF_RISK_UNLIMITED', 215), 215)


if __name__ == '__main__':
    unittest.main()
