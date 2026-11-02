"""Economic-accounting edge cases for the source-cost overlay."""
import unittest

import numpy as np
import pandas as pd

from release.costs import after_cost, aggregate, order_notional


class CostAccountingTests(unittest.TestCase):
    def test_full_round_trip_charges_both_directions(self):
        entry, _, _ = order_notional([2, -1], [0, 0], [10, 20], [10, 20])
        exit_, _, _ = order_notional([0, 0], [2, -1], [10, 20], [10, 20])
        self.assertEqual(entry, 40)
        self.assertEqual(exit_, 40)
        fee, _, net = after_cost(0., 40., entry + exit_, 5.)
        self.assertAlmostEqual(float(fee), .04)
        self.assertAlmostEqual(float(net), -.001)

    def test_stock_netting_cancels_pair_transfers(self):
        # Pair A requests +3 shares and pair B -3 of the same stock.
        self.assertEqual(order_notional([3-3], [0], [10], [10])[0], 0.)

    def test_price_change_does_not_trade_unchanged_raw_shares(self):
        self.assertEqual(order_notional([2, -1], [2, -1], [12, 23], [12, 23])[0], 0.)

    def test_month_handoff_is_direct_difference(self):
        # Shared unchanged stock incurs no fee; replacing a leg does.
        total, _, count = order_notional([2, 0, -1], [2, -1, 0], [10, 20, 25], [10, 20, 25])
        self.assertEqual(total, 45.)
        self.assertEqual(count, 2)

    def test_missing_current_quote_requires_prior_mark(self):
        self.assertEqual(order_notional([0], [2], [np.nan], [10]), (20., 20., 1))
        with self.assertRaises(ValueError):
            order_notional([0], [2], [np.nan], [np.nan])

    def test_no_zero_capital_fallback(self):
        with self.assertRaises(ValueError):
            after_cost(0., 0., 10., 5.)

    def test_zero_fee_and_fee_monotonicity(self):
        pnl, capital, orders = np.array([1., -2.]), np.array([100., 200.]), np.array([40., 300.])
        np.testing.assert_array_equal(after_cost(pnl, capital, orders, 0)[2], pnl/capital)
        values = [np.prod(1 + after_cost(pnl, capital, orders, bp)[2])-1 for bp in (0, 1, 5, 10)]
        self.assertTrue((np.diff(values) < 0).all())

    def test_year_partition_does_not_recharge_entry(self):
        # Net ledger already includes entry on the first day and terminal on the last.
        dates = pd.to_datetime(['2019-12-30', '2019-12-31', '2020-01-02', '2020-01-03'])
        frame = pd.DataFrame({'date': dates, 'formation_cutoff': dates, 'model': 'TEST',
                             'cost_bps': 5., 'net_return': [.01, -.02, .03, -.01],
                             'fee_return': [.001, 0, 0, .001], 'order_notional': [1, 0, 0, 0],
                             'terminal_notional': [0, 0, 0, 1], 'stale_notional': 0.})
        result = aggregate(frame)
        annual = result['yearly']
        full = result['metrics'].iloc[0]
        self.assertAlmostEqual(np.prod(1 + annual.cumulative_return)-1, full.cumulative_return)
        self.assertEqual(float(annual.fee_return_sum.sum()), .002)


if __name__ == '__main__':
    unittest.main()
