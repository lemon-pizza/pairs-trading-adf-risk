"""Change selection only; execute every portfolio with the untouched replay.

Run from any directory using the project's Python environment. Original input
files and original results are read-only. Separate output files are resumable
only when code, settings and input hashes match.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Must precede NumPy/SciPy imports in both parent and spawned workers.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
from trading import trade_interval, summarize_original_style

MODELS = ('SOURCE_M', 'ADF_RISK1', 'ADF_RISK2', 'ADF_RISK3')
SCOPES = {'extended_2017': ('2017-01-03', '2023-05-31'),
          'source_default': ('2020-01-31', '2023-05-31')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def immutable_snapshot():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    return {name: digest(ROOT / name) for name in names if name and (ROOT / name).is_file()}


def code_identity():
    names = ['run.py', 'features.py', 'optimizer.py', 'protocol.json']
    return {name: digest(HERE / name) for name in names} | {
        'src/trading.py': digest(ROOT / 'src/trading.py'),
        'src/selection.py': digest(ROOT / 'src/selection.py')}


def month(key, end, output, settings):
    from features import prepare_month
    from optimizer import solve
    begun = time.monotonic()
    output = Path(output)
    folder = output / 'months'
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / f'{key}.json'
    selection_path = ROOT / 'data/selection' / f'{key}.parquet'
    identity = {'key': key, 'end': end, 'settings': settings, 'code': code_identity(),
                'prices': digest(ROOT / 'data/prices.parquet.gz'),
                'source_selection': digest(selection_path)}
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get('identity') == identity and all(
            (folder / name).is_file() and digest(folder / name) == value
            for name, value in old.get('output_sha256', {}).items()
        ) and len(old.get('output_sha256', {})) == 8:
            return {'key': key, 'seconds': time.monotonic() - begun, 'reused': True,
                    'scores': {m: old['models'][m]['solver']['score_sum'] for m in MODELS}}

    edges, baseline, oracle, meta = prepare_month(key, output, train_days=settings['train_days'])
    baseline = np.asarray(baseline, dtype=int)
    pnl, capital = oracle.arrays(baseline)
    total_capital = capital.sum(axis=1)
    train_return = np.divide(pnl.sum(axis=1), total_capital,
                             out=np.zeros(len(pnl)), where=total_capital > 0)
    budget = float(np.var(train_return, ddof=1))
    source = pd.read_parquet(selection_path)
    source = source[source.METHOD.eq('MAX_WEIGHT')].sort_values('POSITION').copy()
    original_pairs = list(source[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))
    candidate_pairs = list(edges.iloc[baseline][['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))
    if set(candidate_pairs) != set(original_pairs) or len(baseline) != len(source):
        raise AssertionError('Baseline pairs/orientation/cardinality were changed')

    models = {}
    chosen = baseline
    baseline_score = float(-edges.iloc[baseline].T_STAT.sum())
    selections = {'SOURCE_M': (baseline, {'score_sum': baseline_score,
                  'variance': budget, 'risk_budget': budget, 'max_degree': 1,
                  'certified_optimal': False, 'status': 'unchanged_source_reference'})}
    for cap in (1, 2, 3):
        ids, info = solve(edges, oracle, baseline, cap=cap, initial_indices=chosen,
                          time_limit=settings['time_limit'], variance_rtol=1e-8,
                          variance_atol=0., structural_time_limit=0., max_moves=1000)
        if (info['evaluated_supports'] == 0 and not info['certified_optimal']
                and info['stop_reason'] in ('time_limit', 'enumeration_time_limit')):
            raise RuntimeError('Search budget ended before any candidate risk evaluation')
        ids = np.asarray(ids, int)
        chosen = ids
        info['score_sum'] = float(-edges.iloc[ids].T_STAT.sum())
        pn, ca = oracle.arrays(ids)
        denom = ca.sum(axis=1)
        rr = np.divide(pn.sum(axis=1), denom, out=np.zeros(len(pn)), where=denom > 0)
        info['variance'] = float(np.var(rr, ddof=1))
        info['risk_budget'] = budget
        if info['variance'] > budget * (1 + 1e-8) + 1e-18:
            raise AssertionError('Returned selection exceeds the exact original-weight risk budget')
        if info['score_sum'] < selections[list(selections)[-1]][1]['score_sum'] - 1e-8:
            raise AssertionError('Larger degree cap lost the feasible preceding score')
        selections[f'ADF_RISK{cap}'] = (ids, info)

    prices = pd.read_parquet(ROOT / 'data/prices.parquet.gz')
    output_hashes = {}
    for model, (ids, info) in selections.items():
        selected = source.copy() if model == 'SOURCE_M' else edges.iloc[ids].copy()
        selected['MODEL'] = model
        selected['CANDIDATE_INDEX'] = [
            int(i) for i in ids
        ] if model != 'SOURCE_M' else [
            int(next(i for i in baseline if tuple(edges.loc[i, ['ASSET_1', 'ASSET_2']]) == pair))
            for pair in original_pairs
        ]
        daily, trading_meta = trade_interval(prices, selected, pd.Timestamp(key), pd.Timestamp(end))
        if any(trading_meta[k] for k in ('infinite_returns', 'infinite_pair_pnl_elements',
                                         'infinite_pair_capital_elements')):
            raise ValueError(f'Nonfinite source arithmetic for {key}/{model}')
        daily['formation_cutoff'] = pd.Timestamp(key)
        if model == 'SOURCE_M':
            ref = pd.read_parquet(ROOT / 'results/intervals' / f'{key}_MAX_WEIGHT.parquet')
            pd.testing.assert_index_equal(daily.index, ref.index)
            np.testing.assert_allclose(daily[['pnl', 'capital', 'returns', 'active_pairs']],
                                       ref[['pnl', 'capital', 'returns', 'active_pairs']],
                                       rtol=1e-12, atol=1e-10, equal_nan=True)
        selected_name = f'{key}_{model}_selection.parquet'
        daily_name = f'{key}_{model}.parquet'
        selected.to_parquet(folder / selected_name, index=False)
        daily.to_parquet(folder / daily_name)
        output_hashes[selected_name] = digest(folder / selected_name)
        output_hashes[daily_name] = digest(folder / daily_name)
        models[model] = {'selected_indices': ids.tolist(), 'selection_file': selected_name,
                         'daily_file': daily_name, 'solver': info, 'trade': trading_meta}

    ci = int(prices.index.get_loc(pd.Timestamp(key)))
    meta['history'] = oracle.metadata()
    result = {'identity': identity, 'cutoff': str(pd.Timestamp(key).date()),
              'end': str(pd.Timestamp(end).date()),
              'train_start': str(prices.index[ci-settings['train_days']+1].date()),
              'train_end': str(pd.Timestamp(key).date()), 'train_days': settings['train_days'],
              'K': len(baseline), 'baseline_variance': budget, 'features_metadata': meta,
              'models': models, 'output_sha256': output_hashes,
              'seconds': time.monotonic() - begun,
              'completed_at': datetime.now(timezone.utc).isoformat()}
    dump(receipt_path, result)
    return {'key': key, 'seconds': result['seconds'], 'reused': False,
            'scores': {m: models[m]['solver']['score_sum'] for m in MODELS}}


def analyze(output, keys, complete):
    output = Path(output)
    folder = output / 'analysis'
    folder.mkdir(parents=True, exist_ok=True)
    frames, metric_rows, annual_rows, selection_rows = [], [], [], []
    for key in keys:
        rec = json.loads((output / 'months' / f'{key}.json').read_text())
        for model in MODELS:
            solver = rec['models'][model]['solver']
            selection_rows.append({'cutoff': rec['cutoff'], 'model': model, 'K': rec['K'],
                'score_sum': solver['score_sum'], 'variance': solver['variance'],
                'risk_budget': rec['baseline_variance'],
                'certified_optimal': solver.get('certified_optimal', False),
                'max_degree': solver.get('max_degree'),
                'candidate_edges': rec['features_metadata']['candidates']['unordered_candidates']})
    for scope, (start, end) in SCOPES.items():
        scope_keys = [k for k in keys if scope == 'extended_2017' or int(k[:4]) >= 2020]
        if not scope_keys: continue
        for model in MODELS:
            parts = [pd.read_parquet(output / 'months' / f'{k}_{model}.parquet') for k in scope_keys]
            # Identical to src.run.summarize: retain every duplicated boundary row.
            daily = pd.concat(parts).sort_index().loc[start:end]
            if not len(daily): continue
            daily.index.name = 'date'
            daily['scope'], daily['model'] = scope, model
            frames.append(daily.reset_index())
            metric = summarize_original_style(daily.returns)
            wealth = np.r_[1., np.cumprod(1+daily.returns.fillna(0).to_numpy())]
            metric.update(scope=scope, model=model, start=str(daily.index.min().date()),
                end=str(daily.index.max().date()), unique_dates=daily.index.nunique(),
                duplicate_rows=int(daily.index.duplicated().sum()),
                annual_vol=float(daily.returns.std()*np.sqrt(252)),
                max_drawdown_supplement=float(np.min(wealth/np.maximum.accumulate(wealth)-1)),
                complete_requested_period=complete)
            metric_rows.append(metric)
            for year, part in daily.groupby(daily.index.year):
                ar = summarize_original_style(part.returns)
                ar.update(scope=scope, model=model, year=int(year))
                annual_rows.append(ar)
            if complete and model == 'SOURCE_M':
                ref = pd.read_csv(ROOT / 'results' / f'{scope}_MAX_WEIGHT.csv', parse_dates=['date'])
                np.testing.assert_array_equal(daily.index.to_numpy(), ref.date.to_numpy())
                np.testing.assert_allclose(daily.returns.to_numpy(), ref.returns.to_numpy(),
                                           rtol=0, atol=1e-12, equal_nan=True)
    pd.concat(frames, ignore_index=True).to_csv(folder / 'daily.csv', index=False)
    pd.DataFrame(metric_rows).to_csv(folder / 'metrics.csv', index=False)
    pd.DataFrame(annual_rows).to_csv(folder / 'yearly.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(folder / 'selection_metrics.csv', index=False)
    return metric_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'output')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--months', type=int, default=None, help='Explicit partial smoke run, never labeled full')
    parser.add_argument('--train-days', type=int, default=252)
    parser.add_argument('--time-limit', type=float, default=8.)
    args = parser.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    if args.workers < 1 or args.train_days < 2 or args.time_limit <= 0:
        parser.error('workers, train-days and time-limit must be positive (train-days >= 2)')
    if args.months is not None and args.months < 1: parser.error('months must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / 'run_identity.json'
    snapshot = immutable_snapshot()
    if identity_path.exists():
        old = json.loads(identity_path.read_text())
        if old['immutable_files'] != snapshot:
            raise ValueError('Original tracked files changed since this study started')
    else:
        dump(identity_path, {'created_at': datetime.now(timezone.utc).isoformat(),
                            'immutable_files': snapshot})
    all_keys = sorted(p.stem for p in (ROOT / 'data/selection').glob('*.parquet'))
    ends = dict(zip(all_keys, all_keys[1:] + ['20230531']))
    keys = all_keys if args.months is None else all_keys[:args.months]
    settings = {'train_days': args.train_days, 'time_limit': args.time_limit,
                'structural_milp': False, 'max_moves': 1000}
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(month, key, ends[key], str(args.output), settings): key for key in keys}
        for future in as_completed(pending):
            result = future.result()
            done.append(result)
            print(json.dumps(clean({'completed': len(done), 'total': len(keys), **result})), flush=True)
    complete = keys == all_keys
    metrics = analyze(args.output, keys, complete)
    if immutable_snapshot() != snapshot:
        raise AssertionError('Original tracked files were modified during this run')
    dump(args.output / 'execution.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'months': keys, 'complete': complete, 'settings': settings, 'code': code_identity(),
        'immutable_files_unchanged': True, 'metrics': metrics,
        'costs': 'Original replay has no active cost path; gross only'})
    print(json.dumps(clean({'complete': complete, 'metrics': metrics})), flush=True)


if __name__ == '__main__': main()
