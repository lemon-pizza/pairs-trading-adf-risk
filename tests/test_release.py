"""Independent accounting and publication-boundary regression checks."""
import unittest

import pandas as pd

from release.validate import independent_metrics, validate_cost_rows
from release.optimize import ROOT, writable_output


class ReleaseChecks(unittest.TestCase):
    def test_drawdown_includes_initial_capital(self):
        result = independent_metrics([-.1, .05])
        self.assertAlmostEqual(result['cumulative_return'], -.055)
        self.assertAlmostEqual(result['max_drawdown'], -.1)

    def test_optimizer_cannot_replace_published_results(self):
        for path in [ROOT, ROOT / 'work', ROOT / 'results', ROOT / 'research/adf_risk_selection/output',
                     ROOT / 'work/../results']:
            with self.assertRaises(ValueError):
                writable_output(path)
        self.assertEqual(writable_output(ROOT / 'work/rerun'), ROOT / 'work/rerun')

    def test_one_way_fee_is_charged_to_each_order(self):
        # Enter 100, later reverse 100 to -100 (200 notional), then close 100.
        rows = []
        for bps in (0, 1, 5, 10):
            for event, (pnl, order, close_order) in enumerate([(2., 100., 0.), (-1., 200., 100.)]):
                fee = bps / 10000 * (order + close_order)
                rows.append(dict(model='test', event_id=event, cost_bps=bps, pnl=pnl,
                                 capital=100., gross_return=pnl / 100., order_notional=order,
                                 terminal_notional=close_order, fee=fee, fee_return=fee / 100.,
                                 net_return=(pnl - fee) / 100.))
        data = pd.DataFrame(rows)
        validate_cost_rows(data)
        five = data[data.cost_bps.eq(5)]
        self.assertAlmostEqual(five.fee.sum(), .2)
        bad = data.copy()
        bad.loc[bad.cost_bps.eq(5), 'fee'] *= .5
        with self.assertRaises(AssertionError):
            validate_cost_rows(bad)

    def test_missing_liquidation_cost_is_detected(self):
        rows = []
        for bps in (0, 1, 5, 10):
            fee = bps / 10000 * 200
            rows.append(dict(model='test', event_id=0, cost_bps=bps, pnl=0., capital=100.,
                             gross_return=0., order_notional=100., terminal_notional=100.,
                             fee=fee, fee_return=fee / 100., net_return=-fee / 100.))
        data = pd.DataFrame(rows)
        validate_cost_rows(data)
        data.loc[data.cost_bps.eq(10), 'net_return'] = -.001
        with self.assertRaises(AssertionError):
            validate_cost_rows(data)


if __name__ == '__main__':
    unittest.main()
