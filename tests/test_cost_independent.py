"""Independent scalar OLS/Q reconstruction of published raw-share order costs.

This reference deliberately imports neither release.costs nor src.trading.
It checks actual first entry, month-end handoff, missing-quote exit and final
liquidation rather than just rechecking fee multiplication on a supplied ledger.
"""
from functools import lru_cache
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('SOURCE_M', 'ADF_RISK1', 'ADF_RISK2', 'ADF_RISK3')


class IndependentCostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        daily_path = ROOT / 'results/costs/daily.csv'
        if not daily_path.exists():
            raise unittest.SkipTest('Generate published costs with python -m release.costs first')
        cls.daily = pd.read_csv(daily_path, parse_dates=['date', 'formation_cutoff'])
        cls.prices = pd.read_parquet(ROOT / 'data/prices.parquet.gz')
        cls.logs = np.log(cls.prices.to_numpy(float))
        cls.columns = {name: i for i, name in enumerate(cls.prices.columns)}
        cls.marks = cls.prices.where(cls.prices.gt(0)).ffill()

    @classmethod
    @lru_cache(maxsize=None)
    def regression(cls, x, y, day):
        sample = cls.logs[day-503:day+1, [cls.columns[x], cls.columns[y]]]
        if len(sample) != 504 or not np.isfinite(sample).all():
            return np.nan, np.nan
        centered = sample - sample.mean(axis=0)
        variance = float(centered[:, 0] @ centered[:, 0])
        if variance < 1e-12:
            return np.nan, np.nan
        beta = float(centered[:, 0] @ centered[:, 1]) / variance
        residual = float(centered[-1, 1] - beta * centered[-1, 0])
        return beta, residual

    @classmethod
    @lru_cache(maxsize=None)
    def pair_shares(cls, x, y, day):
        # Q_t uses the 20 endpoint residuals ending t-1, with the exact original
        # Q25-Q75 sign and NumPy/Pandas nearest-even round convention.
        history = np.array([cls.regression(x, y, d)[1] for d in range(day-20, day)])
        beta = cls.regression(x, y, day)[0]
        if not np.isfinite(history).all() or not np.isfinite(beta):
            return 0., 0.
        q25, median, q75 = np.quantile(history, [.25, .5, .75])
        if q25 == q75:
            raise AssertionError('Unexpected zero Q denominator in scalar reference')
        score = (history[-1] - median)/(q25-q75)
        q = float(np.sign(score) * np.round(abs(score)))
        return -q*beta, q

    @classmethod
    @lru_cache(maxsize=None)
    def holdings(cls, model, formation, date):
        path = ROOT / 'research/adf_risk_selection/output/months' / (
            f'{pd.Timestamp(formation):%Y%m%d}_{model}_selection.parquet')
        selection = pd.read_parquet(path)
        day = int(cls.prices.index.get_loc(pd.Timestamp(date)))
        shares = np.zeros(len(cls.prices.columns))
        for x, y in selection[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None):
            sx, sy = cls.pair_shares(x, y, day)
            shares[cls.columns[x]] += sx
            shares[cls.columns[y]] += sy
        return shares

    def test_actual_entry_boundary_missing_quote_and_terminal_orders(self):
        checked = 0
        max_error = 0.
        for model in MODELS:
            ledger = self.daily.loc[self.daily.model.eq(model) & self.daily.cost_bps.eq(5)].reset_index(drop=True)
            indices = sorted(set([0, len(ledger)-1] + ledger.index[
                ledger.date.isin(pd.to_datetime(['2017-01-31', '2018-01-02']))].tolist()))
            for idx in indices:
                with self.subTest(model=model, event_id=idx):
                    row = ledger.iloc[idx]
                    current = self.holdings(model, row.formation_cutoff, row.date)
                    if idx:
                        old = ledger.iloc[idx-1]
                        previous = self.holdings(model, old.formation_cutoff, old.date)
                    else:
                        previous = np.zeros_like(current)
                    mark = self.marks.loc[row.date].to_numpy(float)
                    raw = self.prices.loc[row.date].to_numpy(float)
                    order_values = np.abs(current-previous)*np.nan_to_num(mark, nan=0.)
                    expected_order = float(order_values.sum())
                    expected_terminal = float((np.abs(current)*np.nan_to_num(mark, nan=0.)).sum()) if idx == len(ledger)-1 else 0.
                    expected_stale = float(order_values[~np.isfinite(raw)].sum())
                    if idx == len(ledger)-1:
                        expected_stale += float((np.abs(current)*np.nan_to_num(mark, nan=0.))[~np.isfinite(raw)].sum())
                    np.testing.assert_allclose(
                        [row.order_notional, row.terminal_notional, row.stale_notional],
                        [expected_order, expected_terminal, expected_stale], rtol=1e-10, atol=2e-6)
                    expected_fee = (expected_order+expected_terminal)*.0005
                    self.assertAlmostEqual(row.fee, expected_fee, delta=2e-9)
                    self.assertAlmostEqual(row.net_return, (row.pnl-expected_fee)/row.capital, delta=1e-12)
                    max_error = max(max_error, abs(row.order_notional-expected_order),
                                    abs(row.terminal_notional-expected_terminal), abs(row.stale_notional-expected_stale))
                    checked += 1
        print(f'Independent scalar OLS/Q orders: {checked} actual events; maximum notional error {max_error:.3g}')


if __name__ == '__main__':
    unittest.main()
