"""Independently validate the shipped full-size experiment and cost arithmetic.

No full candidate cache is needed. Every frozen selection is re-traded through
the unchanged original kernel, including the 252-date risk history. This checks
feasibility and accounting; it does not certify a global optimization optimum.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/adf_risk_selection/output'
sys.path.insert(0, str(ROOT / 'src'))
from trading import trade_interval

MODELS = ('SOURCE_M', 'ADF_RISK1', 'ADF_RISK2', 'ADF_RISK3')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, message, atol=1e-10, rtol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               equal_nan=True, err_msg=message)


def verify_frozen_inputs(root=ROOT):
    manifest = json.loads((root / 'checks/frozen_inputs.json').read_text())
    files = manifest['files']
    check(bool(files), 'Frozen manifest has no files')
    for name, expected in files.items():
        path = root / name
        check(path.is_file(), 'Missing frozen input: ' + name)
        check(digest(path) == expected, 'Frozen input changed: ' + name)
    return len(files)


def independent_metrics(returns):
    r = pd.Series(returns, dtype=float)
    wealth = np.r_[1., np.cumprod(1 + r.fillna(0).to_numpy())]
    return {'cumulative_return': float(wealth[-1] - 1),
            'annual_vol': float(r.std(ddof=1) * np.sqrt(252)),
            'sharpe': float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
            'max_drawdown': float(np.min(wealth / np.maximum.accumulate(wealth) - 1))}


def validate_month(key, prices):
    record = json.loads((STUDY / 'months' / (key + '.json')).read_text())
    cutoff, end = pd.Timestamp(record['cutoff']), pd.Timestamp(record['end'])
    ci = prices.index.get_loc(cutoff)
    start = prices.index[ci - 251]
    check(record['train_days'] == 252 and pd.Timestamp(record['train_start']) == start,
          f'{key}: training start differs')
    check(pd.Timestamp(record['train_end']) == cutoff, f'{key}: training must stop at cutoff')
    source = pd.read_parquet(ROOT / 'data/selection' / (key + '.parquet'))
    source = source[source.METHOD.eq('MAX_WEIGHT')].sort_values('POSITION')
    source_pairs = list(source[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))
    check(record['K'] == len(source_pairs), f'{key}: original pair count changed')
    reference_train, _ = trade_interval(prices, source_pairs, start, cutoff)
    budget = float(reference_train.returns.fillna(0).var(ddof=1))
    close(record['baseline_variance'], budget, f'{key}: baseline risk', atol=1e-14, rtol=1e-8)
    result, frames, previous_score = {}, {}, None
    for model in MODELS:
        model_record = record['models'][model]
        selection_path = STUDY / 'months' / model_record['selection_file']
        daily_path = STUDY / 'months' / model_record['daily_file']
        check(digest(selection_path) == record['output_sha256'][selection_path.name], 'Selection receipt hash')
        check(digest(daily_path) == record['output_sha256'][daily_path.name], 'Daily receipt hash')
        selected = pd.read_parquet(selection_path)
        pairs = list(selected[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))
        check(len(pairs) == record['K'] and len({frozenset(p) for p in pairs}) == len(pairs),
              f'{key}/{model}: cardinality or reverse duplicate')
        check(all(x != y for x, y in pairs), f'{key}/{model}: self pair')
        degree = pd.Series([asset for pair in pairs for asset in pair]).value_counts()
        cap = 1 if model == 'SOURCE_M' else int(model[-1])
        check(degree.max() <= cap, f'{key}/{model}: stock degree')
        score = float(-selected.T_STAT.sum())
        close(score, model_record['solver']['score_sum'], f'{key}/{model}: score', atol=1e-8)
        if previous_score is not None:
            check(score >= previous_score - 1e-8, f'{key}/{model}: incumbent score worsened')
        previous_score = score
        if model == 'SOURCE_M':
            check(pairs == source_pairs, f'{key}: source direction/order changed')
            close(selected.T_STAT, source.T_STAT, 'Source score changed')
            training = reference_train
        else:
            training, _ = trade_interval(prices, pairs, start, cutoff)
        check(len(training) == 252, f'{key}/{model}: training rows')
        risk = float(training.returns.fillna(0).var(ddof=1))
        check(risk <= budget * (1 + 1e-8) + 1e-14, f'{key}/{model}: risk budget exceeded')
        close(risk, model_record['solver']['variance'], f'{key}/{model}: saved risk', atol=1e-14, rtol=1e-8)
        expected, _ = trade_interval(prices, pairs, cutoff, end)
        stored = pd.read_parquet(daily_path)
        pd.testing.assert_index_equal(expected.index, stored.index)
        for col in ('pnl', 'capital', 'returns', 'active_pairs'):
            close(stored[col], expected[col], f'{key}/{model}: {col}',
                  atol=1e-10, rtol=0 if col == 'returns' else 1e-11)
        check(pd.to_datetime(stored.formation_cutoff).eq(cutoff).all(), 'Formation labels')
        if model == 'SOURCE_M':
            old = pd.read_parquet(ROOT / 'results/intervals' / (key + '_MAX_WEIGHT.parquet'))
            close(stored.returns, old.returns, 'Original frozen interval', atol=1e-12, rtol=0)
        frames[model] = stored
        result[model] = {'pairs': len(pairs), 'max_degree': int(degree.max()),
                         'score': score, 'training_variance': risk, 'risk_budget': budget,
                         'gross_rows': len(stored), 'max_return_error': float(np.nanmax(np.abs(stored.returns - expected.returns)))}
    return result, frames


def validate_cost_rows(daily):
    """Fee equations, rate sensitivity and event identity, without production code."""
    required = {'model', 'event_id', 'cost_bps', 'pnl', 'capital', 'gross_return',
                'order_notional', 'terminal_notional', 'fee', 'fee_return', 'net_return'}
    check(required.issubset(daily.columns), 'Missing cost-ledger fields')
    check(not daily[list(required - {'model'})].isna().any().any(), 'Missing numeric cost value')
    check((daily.capital > 0).all(), 'Cost replay requires positive original capital')
    check((daily[['order_notional', 'terminal_notional', 'fee']] >= 0).all().all(), 'Negative order/fee')
    check(set(daily.cost_bps.unique()) == {0, 1, 5, 10}, 'Cost scenarios differ')
    close(daily.gross_return, daily.pnl / daily.capital, 'Gross numerator/denominator')
    close(daily.fee, daily.cost_bps / 10000 * (daily.order_notional + daily.terminal_notional), 'One-way fee arithmetic')
    close(daily.fee_return, daily.fee / daily.capital, 'Fee normalization')
    close(daily.net_return, (daily.pnl - daily.fee) / daily.capital, 'Net replay arithmetic')
    check(not daily.duplicated(['model', 'event_id', 'cost_bps']).any(), 'Duplicate cost event identity')
    for model, part in daily.groupby('model', sort=False):
        zero = part[part.cost_bps.eq(0)].sort_values('event_id')
        check(np.array_equal(zero.event_id, np.arange(len(zero))), f'{model}: event ids')
        for bps in (1, 5, 10):
            other = part[part.cost_bps.eq(bps)].sort_values('event_id')
            check(np.array_equal(other.event_id, zero.event_id), 'Different cost event dates')
            for col in ('pnl', 'capital', 'gross_return', 'order_notional', 'terminal_notional'):
                close(other[col], zero[col], 'Rate-dependent gross event: ' + col)
            check((other.net_return.to_numpy() <= zero.net_return.to_numpy() + 1e-15).all(), 'Fees improve a daily return')
    return {'rows': len(daily), 'models': sorted(daily.model.unique()), 'cost_bps': [0, 1, 5, 10]}


def validate_cost_outputs(frames, folder):
    daily = pd.read_csv(folder / 'daily.csv', parse_dates=['date', 'formation_cutoff'])
    info = validate_cost_rows(daily)
    check(set(daily.model) == set(MODELS), 'Missing cost strategy')
    for model in MODELS:
        reference = pd.concat(frames[model]).sort_values(['date', 'formation_cutoff']) if 'date' in frames[model][0].columns else pd.concat(frames[model]).reset_index().sort_values(['date', 'formation_cutoff'])
        reference = reference[reference.date.ge('2017-01-03') & reference.date.le('2023-05-31')]
        actual = daily[daily.model.eq(model) & daily.cost_bps.eq(0)].sort_values('event_id')
        np.testing.assert_array_equal(actual.date.to_numpy(), reference.date.to_numpy())
        np.testing.assert_array_equal(actual.formation_cutoff.to_numpy(), reference.formation_cutoff.to_numpy())
        close(actual.gross_return, reference.returns, 'Cost/gross frozen event linkage', atol=1e-12, rtol=0)
        close(actual.pnl, reference.pnl, 'Cost/gross PnL linkage')
        close(actual.capital, reference.capital, 'Cost/gross capital linkage')
    checked = 0
    for name in ('metrics.csv', 'yearly.csv', 'monthly.csv', 'periods.csv'):
        path = folder / name
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        for row in summary.itertuples():
            part = daily[daily.model.eq(row.model) & daily.cost_bps.eq(row.cost_bps)
                         & daily.date.ge(row.start) & daily.date.le(row.end)]
            if row.period == '2020_source_default_continuation':
                part = part[part.formation_cutoff.ge('2020-01-01')]
            check(len(part) == row.event_rows, f'{name}: period event count')
            part = part.sort_values('event_id')
            measures = independent_metrics(part.net_return)
            for metric, value in measures.items():
                close(getattr(row, metric), value, f'{name}: {metric}', atol=1e-10)
            check(part.date.nunique() == row.unique_dates, f'{name}: unique dates')
            for output, source in [('fee_return_sum', 'fee_return'), ('order_notional_sum', 'order_notional'), ('terminal_notional_sum', 'terminal_notional')]:
                close(getattr(row, output), part[source].sum(), f'{name}: {output}', atol=1e-8)
            if hasattr(row, 'stale_notional_sum'):
                close(row.stale_notional_sum, part.stale_notional.sum(), f'{name}: stale notional')
            elapsed = (part.date.max() - part.date.min()).days
            calendar_cagr = (1 + measures['cumulative_return']) ** (365.25 / elapsed) - 1 if elapsed else np.nan
            close(row.index_cagr_calendar, calendar_cagr, f'{name}: calendar index CAGR')
            checked += 1
    info['checked_summary_rows'] = checked
    return info


def validate(months=None, check_costs=True, destination=None):
    protected = verify_frozen_inputs()
    keys = sorted(p.stem for p in (STUDY / 'months').glob('*.json'))
    check(len(keys) == 77, 'Expected all 77 frozen months')
    if months is not None:
        check(months > 0, 'months must be positive')
        keys = keys[:months]
    prices = pd.read_parquet(ROOT / 'data/prices.parquet.gz')
    frames = {model: [] for model in MODELS}
    results = {}
    for index, key in enumerate(keys, 1):
        result, monthly_frames = validate_month(key, prices)
        results[key] = result
        for model, frame in monthly_frames.items():
            frames[model].append(frame)
        if index % 10 == 0 or index == len(keys):
            print(json.dumps({'validated_months': index, 'total': len(keys)}), flush=True)
    complete = len(keys) == 77
    report = {'passed': True, 'complete': complete, 'validated_months': len(keys),
              'protected_files': protected, 'models': list(MODELS), 'months': results,
              'scope': 'Frozen selection feasibility and exact original trade replay; no global optimality certificate.',
              'checked_at': datetime.now(timezone.utc).isoformat()}
    if check_costs and complete:
        report['costs'] = validate_cost_outputs(frames, ROOT / 'results/costs')
    elif check_costs:
        report['costs'] = {'skipped': 'Partial smoke run'}
    verify_frozen_inputs()
    target = Path(destination) if destination else ROOT / 'checks/release_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--months', type=int, help='Explicit partial smoke validation')
    parser.add_argument('--skip-costs', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    result = validate(args.months, not args.skip_costs, args.output)
    print(json.dumps({key: result[key] for key in ('passed', 'complete', 'validated_months', 'protected_files')}))


if __name__ == '__main__':
    main()
