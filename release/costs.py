"""Add explicit transaction fees to the preserved source return index.

This is NOT a self-financing NAV backtest. The original P&L, current-capital
denominator, same-day beta and inclusive duplicate month-end events remain.
Fees use changes in ORIGINAL RAW stock shares, netted across pairs, rather
than the unrelated fixed-budget normalization used in earlier diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trading import WINDOW, _bounds, _pair_list, public_arithmetic, rolling_endpoint

MODELS = ('SOURCE_M', 'ADF_RISK1', 'ADF_RISK2', 'ADF_RISK3')
COST_BPS = (0., 1., 5., 10.)
START, END = pd.Timestamp('2017-01-03'), pd.Timestamp('2023-05-31')
PRICE_SHA256 = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_notional(current, previous, raw_prices, marks):
    """Full buys-plus-sells notional after stock-level netting; no /2."""
    current, previous, raw, marks = [np.asarray(x, float) for x in
                                     (current, previous, raw_prices, marks)]
    if current.ndim != 1 or any(x.shape != current.shape for x in (previous, raw, marks)):
        raise ValueError('Orders and prices must be matching one-dimensional vectors')
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError('Raw share positions must be finite')
    change = current - previous
    known = np.isfinite(raw) & (raw > 0)
    marked = np.isfinite(marks) & (marks > 0)
    if np.any((change != 0) & ~marked):
        raise ValueError('No current or past positive quote to value a stock order')
    dollars = np.zeros_like(change)
    np.multiply(np.abs(change), marks, out=dollars, where=marked)
    return float(dollars.sum()), float(dollars[~known].sum()), int((dollars > 1e-10).sum())


def after_cost(pnl, capital, notional, cost_bps):
    """Source-unit fee subtraction, retaining the original denominator."""
    pnl, capital, notional = np.broadcast_arrays(np.asarray(pnl, float),
                                               np.asarray(capital, float),
                                               np.asarray(notional, float))
    if not all(np.isfinite(x).all() for x in (pnl, capital, notional)):
        raise ValueError('Nonfinite source accounting input')
    if np.any(capital <= 0):
        raise ValueError('Zero/negative source capital: no invented fee denominator')
    if np.any(notional < 0) or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('Cost rate and order notional must be nonnegative')
    fee = notional * float(cost_bps) / 10000.
    net = (pnl - fee) / capital
    if np.any(net <= -1):
        raise ValueError('Source return index reaches or crosses zero')
    return fee, fee / capital, net


def reconstruct_month(prices, selections, start, end):
    """Use unchanged original kernels, verify their totals against frozen rows."""
    normalized = {name: _pair_list(frame) for name, frame in selections.items()}
    pairs = sorted({edge for values in normalized.values() for edge in values})
    lookup = {edge: i for i, edge in enumerate(pairs)}
    asset_index = {name: i for i, name in enumerate(prices.columns)}
    xi = np.array([asset_index[x] for x, _ in pairs], int)
    yi = np.array([asset_index[y] for _, y in pairs], int)
    warm, left, right = _bounds(prices, start, end)
    data = prices.iloc[warm:right].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(data)
    trim = min(WINDOW - 1, len(data))
    offset_start = left - warm - trim
    if offset_start < 0:
        raise ValueError('Insufficient original signal history')
    dates = prices.index[left:right]
    shape = (len(dates), len(pairs))
    pn, ca, sx, sy, q = [np.zeros(shape) for _ in range(5)]
    invalid = np.zeros(shape, bool)
    for lo in range(0, len(pairs), 128):
        hi = min(lo + 128, len(pairs))
        x, y = logs[:, xi[lo:hi]], logs[:, yi[lo:hi]]
        beta, spread = rolling_endpoint(x, y)
        pnl, capital, position, _ = public_arithmetic(x[trim:], y[trim:], beta[trim:], spread[trim:])
        pnl, capital, position = [a[offset_start:] for a in (pnl, capital, position)]
        hedge = beta[trim + offset_start:]
        if any(np.isinf(a).any() for a in (pnl, capital, position, hedge)):
            raise ValueError('Infinite source position or accounting input')
        px, py = data[left-warm:, xi[lo:hi]], data[left-warm:, yi[lo:hi]]
        valid = (np.isfinite(position) & np.isfinite(hedge) & np.isfinite(px)
                 & np.isfinite(py) & (px > 0) & (py > 0) & np.isfinite(capital)
                 & (capital > 0))
        pn[:, lo:hi] = np.nan_to_num(pnl, nan=0.)
        ca[:, lo:hi] = np.nan_to_num(capital, nan=0.)
        q[:, lo:hi] = np.nan_to_num(position, nan=0.)
        sx[:, lo:hi] = np.where(valid, -position * hedge, 0.)
        sy[:, lo:hi] = np.where(valid, position, 0.)
        invalid[:, lo:hi] = np.isfinite(position) & (position != 0) & ~valid
        # Positive capital must equal the gross value of these raw pair shares.
        share_cap = np.abs(sx[:, lo:hi]) * np.where(np.isfinite(px), px, 0.)
        share_cap += np.abs(sy[:, lo:hi]) * np.where(np.isfinite(py), py, 0.)
        np.testing.assert_allclose(share_cap, ca[:, lo:hi], rtol=1e-11, atol=1e-8)
    states = {}
    for model, selected in normalized.items():
        ids = np.array([lookup[edge] for edge in selected], int)
        shares = np.zeros((len(dates), len(prices.columns)))
        rows = np.arange(len(dates))[:, None]
        np.add.at(shares, (rows, xi[ids][None, :]), sx[:, ids])
        np.add.at(shares, (rows, yi[ids][None, :]), sy[:, ids])
        states[model] = {'shares': shares, 'pnl': pn[:, ids].sum(axis=1),
                         'capital': ca[:, ids].sum(axis=1),
                         'active_pairs': np.count_nonzero(q[:, ids], axis=1),
                         'invalid_active': invalid[:, ids].sum(axis=1)}
    return dates, states


def summarize(frame, period):
    r = frame.net_return.to_numpy(float)
    wealth = np.r_[1., np.cumprod(1. + r)]
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    elapsed = (pd.Timestamp(frame.date.max()) - pd.Timestamp(frame.date.min())).days
    return {'model': frame.model.iloc[0], 'cost_bps': float(frame.cost_bps.iloc[0]),
            'period': period, 'start': str(frame.date.min())[:10], 'end': str(frame.date.max())[:10],
            'event_rows': len(frame), 'unique_dates': frame.date.nunique(),
            'cumulative_return': float(wealth[-1]-1.), 'annual_vol': sd*np.sqrt(252),
            'sharpe': float(np.mean(r)/sd*np.sqrt(252)) if sd > 0 else np.nan,
            'max_drawdown': float(np.min(wealth/np.maximum.accumulate(wealth)-1.)),
            'index_cagr_calendar': float(wealth[-1] ** (365.25/elapsed)-1.) if elapsed > 0 else np.nan,
            'fee_return_sum': float(frame.fee_return.sum()),
            'order_notional_sum': float(frame.order_notional.sum()),
            'terminal_notional_sum': float(frame.terminal_notional.sum()),
            'stale_notional_sum': float(frame.stale_notional.sum())}


def aggregate(daily):
    """Periods are slices of one continuous full-run fee ledger, never restarts."""
    metrics, yearly, monthly, periods = [], [], [], []
    for (model, bps), frame in daily.groupby(['model', 'cost_bps'], sort=False):
        metrics.append(summarize(frame, 'extended_2017'))
        for year, part in frame.groupby(frame.date.dt.year):
            yearly.append(summarize(part, str(year)))
        for month, part in frame.groupby(frame.date.dt.to_period('M')):
            monthly.append(summarize(part, str(month)))
        for label, start, end in [('2017_2019_continuation', '2017-01-03', '2019-12-31'),
                                 ('2020_2023_continuation', '2020-01-01', '2023-05-31'),
                                 ('2020_source_default_continuation', '2020-01-31', '2023-05-31')]:
            part = frame.loc[frame.date.between(start, end)]
            if label == '2020_source_default_continuation':
                # Match the original source-default formation subset exactly.
                # Its first holding is carried from the full-run event ledger.
                part = part.loc[part.formation_cutoff >= pd.Timestamp('2020-01-01')]
            if len(part):
                periods.append(summarize(part, label))
    return {name: pd.DataFrame(rows) for name, rows in
            [('metrics', metrics), ('yearly', yearly), ('monthly', monthly), ('periods', periods)]}


def run(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output else root / 'results/costs'
    frozen = root / 'research/adf_risk_selection/output'
    execution = json.loads((frozen / 'execution.json').read_text())
    if not execution.get('complete') or len(execution['months']) != 77:
        raise ValueError('Require the complete frozen pre-cap20 study')
    price_path = root / 'data/prices.parquet.gz'
    if digest(price_path) != PRICE_SHA256:
        raise ValueError('Original price hash differs')
    prices = pd.read_parquet(price_path)
    marks = prices.where(np.isfinite(prices) & prices.gt(0)).ffill()
    previous = {m: np.zeros(len(prices.columns)) for m in MODELS}
    records = {m: [] for m in MODELS}
    sources = {str(price_path.relative_to(root)): digest(price_path),
               'src/trading.py': digest(root/'src/trading.py')}
    max_error = {'pnl': 0., 'capital': 0., 'returns': 0.}
    for key in execution['months']:
        receipt_path = frozen / 'months' / f'{key}.json'
        receipt = json.loads(receipt_path.read_text())
        sources[str(receipt_path.relative_to(root))] = digest(receipt_path)
        selections, reference = {}, {}
        for model in MODELS:
            item = receipt['models'][model]
            for filename in (item['selection_file'], item['daily_file']):
                path = frozen / 'months' / filename
                if digest(path) != receipt['output_sha256'][filename]:
                    raise ValueError('Frozen output hash mismatch: '+filename)
                sources[str(path.relative_to(root))] = digest(path)
            selections[model] = pd.read_parquet(frozen/'months'/item['selection_file'])
            reference[model] = pd.read_parquet(frozen/'months'/item['daily_file'])
        dates, states = reconstruct_month(prices, selections, receipt['cutoff'], receipt['end'])
        for model in MODELS:
            state, ref = states[model], reference[model]
            pd.testing.assert_index_equal(dates, ref.index, check_names=False)
            for field in ('pnl', 'capital'):
                np.testing.assert_allclose(state[field], ref[field], rtol=1e-11, atol=1e-8)
                max_error[field] = max(max_error[field], float(np.max(np.abs(state[field]-ref[field]))))
            np.testing.assert_array_equal(state['active_pairs'], ref.active_pairs)
            with np.errstate(divide='ignore', invalid='ignore'):
                rr = state['pnl']/state['capital']
            np.testing.assert_allclose(rr, ref.returns, rtol=1e-11, atol=1e-12, equal_nan=True)
            finite = np.isfinite(rr) & np.isfinite(ref.returns)
            max_error['returns'] = max(max_error['returns'], float(np.max(np.abs(rr[finite]-ref.returns.to_numpy()[finite]))))
            for row in np.flatnonzero((dates >= START) & (dates <= END)):
                date = dates[row]
                current = state['shares'][row]
                notional, stale, count = order_notional(current, previous[model], prices.loc[date], marks.loc[date])
                records[model].append({'event_id': len(records[model]), 'date': date,
                    'formation_cutoff': pd.Timestamp(receipt['cutoff']), 'model': model,
                    'pnl': float(ref.pnl.iloc[row]), 'capital': float(ref.capital.iloc[row]),
                    'gross_return': float(ref.returns.iloc[row]), 'order_notional': notional,
                    'terminal_notional': 0., 'stale_notional': stale, 'stock_order_count': count,
                    'terminal_order_count': 0, 'invalid_active_pairs': int(state['invalid_active'][row])})
                previous[model] = current.copy()
        print(json.dumps({'stage': 'source_costs', 'month': key}), flush=True)
    frames = []
    for model, rows in records.items():
        final = rows[-1]
        terminal, stale, count = order_notional(np.zeros_like(previous[model]), previous[model],
                                               prices.loc[final['date']], marks.loc[final['date']])
        final.update(terminal_notional=terminal, stale_notional=final['stale_notional']+stale,
                     terminal_order_count=count)
        base = pd.DataFrame(rows)
        if list(base[['date','formation_cutoff']].itertuples(index=False, name=None)) != sorted(
                base[['date','formation_cutoff']].itertuples(index=False, name=None)):
            raise AssertionError('Source event order must be date then old-to-new formation')
        for bps in COST_BPS:
            frame = base.copy()
            frame['cost_bps'] = bps
            fee, fee_return, net = after_cost(frame.pnl, frame.capital,
                                               frame.order_notional+frame.terminal_notional, bps)
            frame['fee'], frame['fee_return'], frame['net_return'] = fee, fee_return, net
            if bps == 0:
                np.testing.assert_array_equal(net, frame.gross_return)
            frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    tables = aggregate(daily)
    original_metrics = pd.read_csv(frozen/'analysis/metrics.csv')
    for model in MODELS:
        expected = original_metrics.loc[(original_metrics.scope == 'extended_2017') & (original_metrics.model == model)].iloc[0]
        actual = tables['metrics'].loc[(tables['metrics'].model == model) & tables['metrics'].cost_bps.eq(0)].iloc[0]
        np.testing.assert_allclose(actual.cumulative_return, expected.cumulative_return, atol=1e-12, rtol=0)
        source_expected = original_metrics.loc[(original_metrics.scope == 'source_default') & (original_metrics.model == model)].iloc[0]
        source_actual = tables['periods'].loc[(tables['periods'].model == model) & tables['periods'].cost_bps.eq(0)
                           & tables['periods'].period.eq('2020_source_default_continuation')].iloc[0]
        np.testing.assert_allclose(source_actual.cumulative_return, source_expected.cumulative_return, atol=1e-12, rtol=0)
        net_returns = tables['metrics'].loc[tables['metrics'].model.eq(model)].sort_values('cost_bps').cumulative_return
        if not (np.diff(net_returns.to_numpy()) <= 0).all():
            raise AssertionError('Higher nonnegative fee must not improve return')
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output/'daily.csv', index=False)
    for name, table in tables.items():
        table.to_csv(output/f'{name}.csv', index=False)
    if any(digest(root/path) != sha for path, sha in sources.items()):
        raise AssertionError('Source input mutated during cost calculation')
    audit = {'passed': True, 'completed_at': datetime.now(timezone.utc).isoformat(),
        'models': list(MODELS), 'cost_bps_one_way': list(COST_BPS), 'months': 77,
        'events_per_model': len(records[MODELS[0]]),
        'unique_dates_per_model': len({r['date'] for r in records[MODELS[0]]}),
        'duplicate_dates_retained': True, 'zero_cost_matches_original': True,
        'source_default_zero_cost_matches_original': True,
        'reconstruction_max_absolute_errors': max_error,
        'source_inputs_unchanged': True, 'input_sha256': sources,
        'cost_code_sha256': digest(__file__),
        'output_sha256': {p.name: digest(p) for p in sorted(output.glob('*.csv'))},
        'accounting': 'Source return index with raw-share fees; NOT self-financing NAV',
        'formula': 'net_k=(source_pnl_k-c*sum_i(mark_i,k*abs(h_i,k-h_i,k-1)))/source_capital_k; terminal liquidation added to last event',
        'event_order': 'date ascending, formation cutoff ascending; both original monthly boundary rows retained',
        'stock_netting': 'Aggregate raw shares across pairs before differencing; no 1/2 turnover factor',
        'initial_final': 'Start cash at first 2017 event; pay initial entry and final liquidation; no monthly forced flat reset',
        'period_accounting': 'All annual/monthly/subperiod statistics slice the continuing full-run ledger; no reset or new entry fees at subperiod starts',
        'missing_positions': 'Original invalid/zero-capital pair observations have zero valued shares; exits may be valued at last prior positive quote',
        'missing_order_prices': 'Forward-filled last observed positive prices for valuation only; never future backfill',
        'stale_events_by_model': {model: [str(r['date'].date()) for r in rows if r['stale_notional'] > 0]
                                  for model, rows in records.items()},
        'warnings': ['Same-day beta, current-price denominator, source universe and duplicate dates remain.',
            'Cost-adjusted compounded source indices are not executable or self-financing NAV returns.',
            'Fees do not affect signals, selections, position size or subsequent source capital.',
            'Borrow costs, financing, bid-ask spreads, impact and minimum order fees are excluded.',
            '5 basis points is 0.05% per direction, not round trip.',
            'Annual volatility and Sharpe use 252 source events, including duplicate dates; CAGR is calendar index growth.']}
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(tables['metrics'].to_string(index=False), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    run(output=args.output)


if __name__ == '__main__':
    main()
