"""Extend the frozen full-size study to degree 4, 5, and unlimited overlap.

The frozen b=3 solution starts the extension; all stages retain the original
SOURCE_M risk anchor and monthly cardinality. The original trading, feature,
and optimization kernels are imported unchanged. A supplied candidate cache
is read-only and accepted only after input/code/content hash checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'research/adf_risk_selection'
FROZEN = ORIGINAL / 'output'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ORIGINAL))
from features import LazyPairHistory, SCHEMA, original_select, prepare_month
from optimizer import solve, support_state
from selection import EXPECTED_SHA256
from trading import trade_interval, summarize_original_style

MODELS = ('ADF_RISK4', 'ADF_RISK5', 'ADF_RISK_UNLIMITED')
SETTINGS = {'train_days': 252, 'time_limit': 8., 'structural_milp': False,
            'max_moves': 1000, 'variance_rtol': 1e-8, 'variance_atol': 0.,
            'warm_start': 'frozen ADF_RISK3 -> ADF_RISK4 -> ADF_RISK5 -> ADF_RISK_UNLIMITED'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def cap_for_model(model, k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('K must be a positive integer')
    if model not in MODELS:
        raise ValueError('Unknown extension model')
    # Every vertex touches at most K of a portfolio's K distinct edges.
    # Hence cap=K is exactly the model with no stock-overlap restriction.
    return int(k) if model == 'ADF_RISK_UNLIMITED' else int(model[-1])


def writable_output(path):
    output = Path(path).resolve()
    work = (ROOT / 'work').resolve()
    if output == work or not output.is_relative_to(work):
        raise ValueError('Generated extension results must be in a subdirectory of work/.')
    return output


def verify_frozen():
    files = json.loads((ROOT / 'checks/frozen_inputs.json').read_text())['files']
    for name, expected in files.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    return len(files)


def code_identity():
    names = ['release/extend_overlap.py', 'config/overlap_extension.json',
             'research/adf_risk_selection/features.py', 'research/adf_risk_selection/optimizer.py',
             'src/selection.py', 'src/trading.py']
    return {name: digest(ROOT / name) for name in names}


def candidate_identity(key, original_receipt):
    return {'schema': SCHEMA, 'cutoff': key, 'price_sha256': EXPECTED_SHA256,
            'original_selection_sha256': original_receipt['selection_sha256'],
            'original_receipt_sha256': digest(ROOT / 'data/selection' / (key + '.json')),
            'selection_kernel_sha256': digest(ROOT / 'src/selection.py'),
            'feature_code_sha256': digest(ORIGINAL / 'features.py'),
            'direction_rule': 'retain_source_M_direction_else_lexicographic_among_available_directions'}


def read_candidates(key, cache_root, source_receipt, base):
    """Read both cache hashes, never call a writer in the external cache."""
    folder = Path(cache_root)
    meta_path = folder / (key + '_edges.json')
    metadata = json.loads(meta_path.read_text())
    if metadata['identity'] != candidate_identity(key, source_receipt):
        raise ValueError('Candidate cache input/code identity differs: ' + key)
    for filename, hash_key in [(key + '_edges.parquet', 'edges_sha256'),
                               (key + '_directed.parquet', 'directed_sha256')]:
        if digest(folder / filename) != metadata[hash_key]:
            raise ValueError('Candidate cache content changed: ' + filename)
    if any(metadata[name] for name in ('candidate_truncation', 'p_value_filter', 'extra_history_filter')):
        raise ValueError('Candidate cache is restricted')
    if not metadata['source_eligible_membership_exact']:
        raise ValueError('Candidate cache lacks original eligible-membership verification')
    edges = pd.read_parquet(folder / (key + '_edges.parquet'))
    if len(edges) != metadata['unordered_candidates'] or metadata['directed_candidates'] != source_receipt['candidate_n']:
        raise ValueError('Candidate count differs from original graph')
    if not np.array_equal(edges.EDGE_ID.to_numpy(), np.arange(len(edges))):
        raise ValueError('Candidate row IDs changed')
    original_pairs = list(base[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))
    baseline = edges.loc[edges.IS_SOURCE_M].sort_values('SOURCE_POSITION')
    if list(baseline[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None)) != original_pairs:
        raise ValueError('Candidate graph changed original orientation or membership')
    np.testing.assert_allclose(baseline.T_STAT, base.T_STAT, rtol=0, atol=0)
    return edges, {**metadata, 'reused_verified_candidates': True,
                   'read_only_external_cache': True, 'metadata_sha256': digest(meta_path)}


def selection_ids(edges, selected):
    """Map by oriented pair and verify the frozen row IDs and scores."""
    lookup = {pair: i for i, pair in enumerate(edges[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))}
    pairs = list(selected[['ASSET_1', 'ASSET_2']].itertuples(index=False, name=None))
    try:
        indices = np.array([lookup[pair] for pair in pairs], int)
    except KeyError as exc:
        raise ValueError('Frozen b=3 pair/orientation absent from full candidate graph') from exc
    np.testing.assert_array_equal(indices, selected.CANDIDATE_INDEX.to_numpy())
    np.testing.assert_allclose(edges.iloc[indices].T_STAT, selected.T_STAT, rtol=0, atol=0)
    return indices


def one_month(key, end, output, cache_root):
    begun = time.monotonic()
    output = Path(output)
    folder = output / 'months'
    frozen_receipt_path = FROZEN / 'months' / (key + '.json')
    frozen_receipt = json.loads(frozen_receipt_path.read_text())
    warm_item = frozen_receipt['models']['ADF_RISK3']
    warm_path = FROZEN / 'months' / warm_item['selection_file']
    if digest(warm_path) != frozen_receipt['output_sha256'][warm_path.name]:
        raise ValueError('Frozen b=3 selection receipt mismatch')
    identity = {'key': key, 'end': end, 'settings': SETTINGS, 'code': code_identity(),
                'prices': digest(ROOT / 'data/prices.parquet.gz'),
                'source_selection': digest(ROOT / 'data/selection' / (key + '.parquet')),
                'source_receipt': digest(ROOT / 'data/selection' / (key + '.json')),
                'frozen_b3_selection': digest(warm_path), 'frozen_month_receipt': digest(frozen_receipt_path)}
    if identity['prices'] != EXPECTED_SHA256:
        raise ValueError('Original price bytes differ')
    receipt_path = folder / (key + '.json')
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        hashes = prior.get('output_sha256', {})
        if prior.get('identity') == identity and len(hashes) == 6 and all(
                (folder / name).is_file() and digest(folder / name) == sha for name, sha in hashes.items()):
            return {'key': key, 'reused': True, 'seconds': time.monotonic() - begun,
                    'scores': {name: prior['models'][name]['solver']['score_sum'] for name in MODELS}}
    prices = pd.read_parquet(ROOT / 'data/prices.parquet.gz')
    _, base, source_receipt = original_select(key)
    if cache_root:
        edges, candidates = read_candidates(key, cache_root, source_receipt, base)
        baseline = edges.loc[edges.IS_SOURCE_M].sort_values('SOURCE_POSITION').index.to_numpy(int)
        oracle = LazyPairHistory(prices, edges, pd.Timestamp(key), train_days=252)
        features = {'candidates': candidates}
    else:
        edges, baseline, oracle, features = prepare_month(key, output, train_days=252)
    _, _, reference_returns, budget = support_state(oracle, baseline)
    reference, _ = trade_interval(prices, base, oracle.dates[0], oracle.dates[-1])
    np.testing.assert_allclose(reference_returns, reference.returns.fillna(0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(budget, frozen_receipt['baseline_variance'], rtol=1e-8, atol=1e-16)
    k = len(baseline)
    if k != frozen_receipt['K']:
        raise ValueError('Original cardinality changed')
    warm = pd.read_parquet(warm_path)
    chosen = selection_ids(edges, warm)
    if len(chosen) != k:
        raise ValueError('Frozen b=3 cardinality differs')
    _, _, _, warm_variance = support_state(oracle, chosen)
    if warm_variance > budget * (1 + 1e-8):
        raise ValueError('Frozen b=3 is not feasible for original risk budget')
    np.testing.assert_allclose(warm_variance, warm_item['solver']['variance'], rtol=1e-8, atol=1e-16)
    previous_score = float(-warm.T_STAT.sum())
    np.testing.assert_allclose(previous_score, warm_item['solver']['score_sum'], rtol=1e-12, atol=1e-8)
    models, hashes = {}, {}
    folder.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        cap = cap_for_model(model, k)
        ids, info = solve(edges, oracle, baseline, cap=cap, initial_indices=chosen,
                          time_limit=8., max_moves=1000, variance_rtol=1e-8,
                          variance_atol=0., structural_time_limit=0.)
        if (not info['certified_optimal'] and info['evaluated_supports'] == 0
                and info['stop_reason'] in ('time_limit', 'enumeration_time_limit')):
            raise RuntimeError('Search time exhausted before any support risk evaluation')
        chosen = np.asarray(ids, int)
        _, _, _, variance = support_state(oracle, chosen)
        score = float(-edges.iloc[chosen].T_STAT.sum())
        if variance > budget * (1 + 1e-8) or score < previous_score - 1e-8:
            raise AssertionError('Extension worsened score or exceeded original risk budget')
        previous_score = score
        info.update(score_sum=score, variance=variance, risk_budget=budget,
                    overlap_unrestricted=model == 'ADF_RISK_UNLIMITED',
                    effective_cap=cap, original_cardinality=k)
        selected = edges.iloc[chosen].copy()
        selected['MODEL'], selected['CANDIDATE_INDEX'] = model, chosen
        daily, trade = trade_interval(prices, selected, pd.Timestamp(key), pd.Timestamp(end))
        if any(trade[name] for name in ('infinite_returns', 'infinite_pair_pnl_elements', 'infinite_pair_capital_elements')):
            raise ValueError('Nonfinite original trading arithmetic')
        daily['formation_cutoff'] = pd.Timestamp(key)
        selection_name, daily_name = key + '_' + model + '_selection.parquet', key + '_' + model + '.parquet'
        selected.to_parquet(folder / selection_name, index=False)
        daily.to_parquet(folder / daily_name)
        hashes.update({selection_name: digest(folder / selection_name), daily_name: digest(folder / daily_name)})
        models[model] = {'selected_indices': chosen.tolist(), 'selection_file': selection_name,
                         'daily_file': daily_name, 'solver': info, 'trade': trade}
    features.update(history=oracle.metadata(), baseline_ids=np.asarray(baseline).tolist())
    record = {'identity': identity, 'cutoff': str(pd.Timestamp(key).date()), 'end': str(pd.Timestamp(end).date()),
              'train_start': str(oracle.dates[0].date()), 'train_end': str(oracle.dates[-1].date()),
              'train_days': 252, 'K': k, 'baseline_variance': budget,
              'initial_model': 'ADF_RISK3', 'initial_score': float(-warm.T_STAT.sum()),
              'features_metadata': features, 'models': models, 'output_sha256': hashes,
              'seconds': time.monotonic() - begun, 'completed_at': datetime.now(timezone.utc).isoformat()}
    dump(receipt_path, record)
    return {'key': key, 'reused': False, 'seconds': record['seconds'],
            'scores': {name: models[name]['solver']['score_sum'] for name in MODELS},
            'max_degrees': {name: models[name]['solver']['max_degree'] for name in MODELS}}


def analyze(output, keys, complete):
    output = Path(output)
    metrics, years, selection = [], [], []
    for model in MODELS:
        daily = pd.concat([pd.read_parquet(output / 'months' / (key + '_' + model + '.parquet')) for key in keys])
        daily = daily.sort_index().loc['2017-01-03':'2023-05-31']
        if not len(daily):
            continue
        metric = summarize_original_style(daily.returns)
        wealth = np.r_[1., np.cumprod(1. + daily.returns.fillna(0).to_numpy())]
        metric.update(model=model, scope='extended_2017', annual_vol=float(daily.returns.std() * np.sqrt(252)),
                      max_drawdown_supplement=float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
                      start=str(daily.index.min().date()), end=str(daily.index.max().date()),
                      unique_dates=daily.index.nunique(), duplicate_rows=int(daily.index.duplicated().sum()),
                      complete_requested_period=complete)
        metrics.append(metric)
        for year, part in daily.groupby(daily.index.year):
            years.append({'year': int(year), 'model': model, **summarize_original_style(part.returns)})
    for key in keys:
        item = json.loads((output / 'months' / (key + '.json')).read_text())
        for model in MODELS:
            solver = item['models'][model]['solver']
            selection.append({'cutoff': item['cutoff'], 'model': model, 'K': item['K'],
                              'score_sum': solver['score_sum'], 'variance': solver['variance'],
                              'risk_budget': item['baseline_variance'], 'max_degree': solver['max_degree'],
                              'certified_optimal': solver['certified_optimal']})
    folder = output / 'analysis'
    folder.mkdir(exist_ok=True)
    for name, rows in [('metrics', metrics), ('yearly', years), ('selection_metrics', selection)]:
        pd.DataFrame(rows).to_csv(folder / (name + '.csv'), index=False)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'work/overlap_extension')
    parser.add_argument('--candidate-cache-root', type=Path, help='Read-only original features folder; all identity and content hashes are checked')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--months', type=int, help='Explicit partial smoke run; does not claim full-period results')
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.months is not None and args.months < 1):
        parser.error('workers and months must be positive')
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, '-m', 'release.extend_overlap', *sys.argv[1:]])
    try:
        output = writable_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    count = verify_frozen()
    all_keys = sorted(p.stem for p in (FROZEN / 'months').glob('*.json'))
    if len(all_keys) != 77:
        raise ValueError('Require the complete 77-month original study')
    keys = all_keys if args.months is None else all_keys[:args.months]
    ends = dict(zip(all_keys, all_keys[1:] + ['20230531']))
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_month, key, ends[key], str(output),
                               str(args.candidate_cache_root) if args.candidate_cache_root else None) for key in keys]
        for done, future in enumerate(as_completed(futures), 1):
            print(json.dumps(clean({'completed': done, 'total': len(keys), **future.result()})), flush=True)
    complete = keys == all_keys
    metrics = analyze(output, keys, complete)
    verify_frozen()
    dump(output / 'execution.json', {'months': keys, 'complete': complete, 'models': MODELS,
         'settings': SETTINGS, 'code': code_identity(), 'metrics': metrics,
         'protected_frozen_files': count, 'frozen_inputs_unchanged': True,
         'completed_at': datetime.now(timezone.utc).isoformat(),
         'costs': 'Gross only here; release.costs applies the common original-share fee overlay.',
         'unlimited_definition': 'Effective cap=K; nonbinding because exactly K distinct pairs are selected.',
         'interpretation': 'Bounded-search extension from frozen b=3; risk/cardinality preserved, no global optimum guarantee.'})
    print(json.dumps({'complete': complete, 'months': len(keys), 'output': str(output.relative_to(ROOT))}), flush=True)


if __name__ == '__main__':
    main()
