"""Conditional replay of the public aaac6d9 pair-selection code.

Preserves betas.py directed pairs and full-sample universe, default ADF/AIC,
all p<=.05 baseline pairs, and matching.py's unweighted maximum cardinality.
The author original environment/order is unavailable; this replay fixes hash seed 0.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy
from scipy.special import ndtr
import sklearn
import statsmodels
from statsmodels.tsa import adfvalues
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICES = ROOT / 'data/prices.parquet.gz'
EXPECTED_SHA256 = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'
SOURCE_COMMIT = 'aaac6d999d6b077304a887caff2aab514d7ea6a2'


def adf_pvalues(t):
    t = np.asarray(t)
    small = np.polyval(adfvalues._tau_smallps['c'][0][::-1], t)
    large = np.polyval(adfvalues._tau_largeps['c'][0][::-1], t)
    out = ndtr(np.where(t <= adfvalues._tau_stars['c'][0], small, large))
    return np.where(t > adfvalues._tau_maxs['c'][0], 1.,
                    np.where(t < adfvalues._tau_mins['c'][0], 0., out))


def _design(e, lag):
    diff = np.diff(e, axis=1)
    y = diff[:, lag:].copy()
    cols = [e[:, lag:-1]]
    cols.extend(diff[:, lag-j:-j] for j in range(1, lag+1))
    x = np.stack(cols, axis=2)
    # Frisch-Waugh elimination of the intercept retains the exact regressions.
    y -= y.mean(axis=1, keepdims=True)
    x -= x.mean(axis=1, keepdims=True)
    return x, y


def adf_aic_batch(residuals):
    """Default adfuller equivalent: common-sample AIC then selected-lag refit.

    Cholesky of the full centered Gram matrix yields RSS for every nested
    model. Ill-conditioned/near-tied/threshold-adjacent cases use statsmodels.
    Returns t, p, usedlag, icbest, fallback count. No fixed lag substitution.
    """
    e = np.asarray(residuals, float)
    if e.ndim != 2 or not np.isfinite(e).all():
        raise ValueError('Expected complete residual rows')
    nseries, n = e.shape
    maxlag = min(int(np.ceil(12.0*(n/100.0)**.25)), n//2-2)
    x, y = _design(e, maxlag)
    gram = x.transpose(0, 2, 1) @ x
    xy = np.einsum('bni,bn->bi', x, y)
    yy = np.einsum('bn,bn->b', y, y)
    fallback = np.zeros(nseries, dtype=bool)
    # Batched Cholesky may fail if one series is singular. Isolate those rows.
    chol = np.empty_like(gram)
    try:
        chol[:] = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError:
        for i in range(nseries):
            try:
                chol[i] = np.linalg.cholesky(gram[i])
            except np.linalg.LinAlgError:
                fallback[i] = True
                chol[i] = np.eye(maxlag+1)
    whitened = np.linalg.solve(chol, xy[..., None])[..., 0]
    rss = yy[:, None] - np.cumsum(whitened**2, axis=1)
    ncommon = y.shape[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        aic = ncommon*(np.log(2*np.pi)+1+np.log(rss/ncommon)) + 2*np.arange(2, maxlag+3)
    fallback |= (~np.isfinite(aic)).any(axis=1)
    order = np.argsort(aic, axis=1, kind='stable')
    usedlag = order[:, 0].copy()
    icbest = aic[np.arange(nseries), usedlag]
    with np.errstate(invalid='ignore'):
        gap = aic[np.arange(nseries), order[:, 1]] - icbest
    fallback |= gap < 1e-8
    # Diagonal-based warning is conservative; exact library fallback is cheap.
    diagonals = np.diagonal(chol, axis1=1, axis2=2)
    fallback |= diagonals.min(axis=1) < 1e-8*diagonals.max(axis=1)
    t = np.full(nseries, np.nan)
    for lag in np.unique(usedlag[~fallback]):
        ii = np.flatnonzero((usedlag == lag) & ~fallback)
        xx, dep = _design(e[ii], int(lag))
        gg = xx.transpose(0, 2, 1) @ xx
        gdep = np.einsum('bni,bn->bi', xx, dep)
        rhs = np.zeros((len(ii), lag+1, 2))
        rhs[:, :, 0] = gdep
        rhs[:, 0, 1] = 1.
        try:
            solved = np.linalg.solve(gg, rhs)
        except np.linalg.LinAlgError:
            fallback[ii] = True
            continue
        beta = solved[:, :, 0]
        errors = dep - np.einsum('bni,bi->bn', xx, beta)
        ssr = np.einsum('bn,bn->b', errors, errors)
        variance = ssr/(dep.shape[1]-(lag+2))
        with np.errstate(divide='ignore', invalid='ignore'):
            t[ii] = beta[:, 0]/np.sqrt(variance*solved[:, 0, 1])
    p = adf_pvalues(t)
    fallback |= ~np.isfinite(t) | (np.abs(p-.05) < 1e-8)
    for i in np.flatnonzero(fallback):
        try:
            ans = adfuller(e[i])
            t[i], p[i], usedlag[i], icbest[i] = ans[0], ans[1], ans[2], ans[5]
        except (ValueError, np.linalg.LinAlgError):
            # Source cache_spread catches failed/constant pair regressions.
            t[i], p[i], usedlag[i], icbest[i] = np.nan, np.nan, -1, np.nan
    return t, p, usedlag, icbest, int(fallback.sum())


def calculate_edges(log_prices, directed=False, batch_size=256):
    a = log_prices.to_numpy(float)
    centered = a-a.mean(axis=0)
    cov = centered.T@centered
    if directed:
        i, j = np.where(~np.eye(a.shape[1], dtype=bool))
    else:
        i, j = np.triu_indices(a.shape[1], k=1)
    beta = np.divide(cov[i, j], cov[i, i], out=np.zeros(len(i)), where=cov[i, i] != 0)
    alpha = a.mean(axis=0)[j]-beta*a.mean(axis=0)[i]
    t, p, usedlag, aic = (np.empty(len(i)) for _ in range(4))
    fallback_count = 0
    for start in range(0, len(i), batch_size):
        stop = start+batch_size
        e = (a[:, j[start:stop]] - beta[start:stop]*a[:, i[start:stop]] - alpha[start:stop]).T
        tt, pp, ll, aa, ff = adf_aic_batch(e)
        t[start:stop], p[start:stop], usedlag[start:stop], aic[start:stop] = tt, pp, ll, aa
        fallback_count += ff
    names = log_prices.columns.to_numpy()
    frame = pd.DataFrame({'ASSET_1': names[i], 'ASSET_2': names[j],
                          'BETA': beta, 'ALPHA': alpha, 'T_STAT': t,
                          'P_VALUE': p, 'USEDLAG': usedlag.astype(int), 'AIC': aic})
    # Resolve any floating-point-near sort tie with the public sklearn+ADF path.
    # Graph insertion order can affect an unweighted matching, so ordering matters.
    ordered = np.argsort(t, kind='stable')
    close = np.flatnonzero(np.diff(t[ordered]) < 1e-9)
    ambiguous = np.unique(np.r_[ordered[close], ordered[close+1]])
    if len(ambiguous):
        from sklearn.linear_model import LinearRegression
        for row in ambiguous:
            x, y = a[:, i[row]], a[:, j[row]]
            fit = LinearRegression().fit(np.column_stack([x, np.ones(len(x))]), y)
            residual = y-fit.coef_[0]*x-fit.intercept_
            exact = adfuller(residual)
            frame.loc[row, ['BETA', 'ALPHA', 'T_STAT', 'P_VALUE', 'USEDLAG', 'AIC']] = [
                fit.coef_[0], fit.intercept_, exact[0], exact[1], exact[2], exact[5]]
    failed = ~np.isfinite(frame.T_STAT) | ~np.isfinite(frame.P_VALUE)
    skipped = int(failed.sum())
    frame = frame.loc[~failed].copy()
    frame.attrs['exact_sort_tie_refits'] = int(len(ambiguous))
    frame.attrs['adf_failed_pairs_skipped'] = skipped
    # Same sort as pairs.py / betas.py cache_spread, including default sort kind.
    return frame.sort_values('P_VALUE').reset_index(drop=True), fallback_count


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'statsmodels': statsmodels.__version__,
            'networkx': nx.__version__, 'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED')}


def source_universe(prices):
    """Original betas.py full-sample padded pct_change missing-rate filter."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Start a fresh interpreter with PYTHONHASHSEED=0')
    returns = prices.ffill().pct_change(fill_method=None).iloc[1:]
    missing_rate = returns.isnull().sum() / len(returns)
    return list(set(missing_rate[missing_rate < .05].index))


def selection_cutoffs(prices):
    """Requested 2017 extension and original 202* scope share these boundaries.

    The original business-month-end writer names files by the last actual
    observation. The final boundary is end-only, not a holding-month selection.
    """
    ends = pd.date_range(prices.index[0], prices.index[-1], freq='BME')[24:]
    return [prices.loc[:end].index[-1] for end in ends
            if pd.Timestamp('2016-12-01') <= end <= pd.Timestamp('2023-05-31')]


def literal_matching(edges):
    data = edges.sort_values('T_STAT').copy()
    data.T_STAT = data.T_STAT.round(2)
    data['TUPLE'] = list(zip(data.ASSET_1, data.ASSET_2))
    graph = nx.Graph()
    graph.add_nodes_from(data.ASSET_1.unique(), bipartite=0, label='ASSET_1')
    graph.add_nodes_from(data.ASSET_2.unique(), bipartite=1, label='ASSET_2')
    graph.add_edges_from((x, y, {'rating': float(t)})
                         for x, y, t in data[['ASSET_1', 'ASSET_2', 'T_STAT']].itertuples(index=False, name=None))
    # Source stores rating but omits weight='rating': preserve that exact behavior.
    maximum = nx.max_weight_matching(graph, maxcardinality=1)
    baseline = set(data.loc[data.P_VALUE <= .05, 'TUPLE'])
    return baseline, maximum


def cache_identity(cutoff, prices_path):
    price_hash = sha256(prices_path)
    if price_hash != EXPECTED_SHA256:
        raise ValueError('Unexpected source price bytes')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Start a fresh interpreter with PYTHONHASHSEED=0')
    return {'source_commit': SOURCE_COMMIT, 'price_sha256': price_hash,
            'generator_sha256': sha256(__file__), 'environment': environment(),
            'cutoff': str(pd.Timestamp(cutoff).date()),
            'contract': 'directed_global_universe_default_ADF_unweighted_cardinality_v1'}


def generate_month(cutoff, prices_path=DEFAULT_PRICES, output_dir=None, force=False):
    started = time.monotonic()
    cutoff = pd.Timestamp(cutoff)
    cache = Path(output_dir or ROOT / 'data/selection')
    cache.mkdir(parents=True, exist_ok=True)
    key = cutoff.strftime('%Y%m%d')
    output, meta_path = cache / (key + '.parquet'), cache / (key + '.json')
    identity = cache_identity(cutoff, prices_path)
    if output.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get('cache_identity') == identity and meta.get('selection_sha256') == sha256(output):
                return meta
        except (ValueError, OSError):
            pass
    prices = pd.read_parquet(prices_path)
    names = source_universe(prices)
    hist = prices[names].loc[:cutoff].iloc[-504:]
    if len(hist) != 504 or hist.index[-1] != cutoff:
        raise ValueError(f'Invalid 504-row cutoff {cutoff}')
    # Missing/invalid inputs cause source sklearn OLS to fail and the pair to
    # be skipped. A complete-column filter is the equivalent efficient route.
    complete = np.isfinite(hist.to_numpy()).all(axis=0) & (hist.to_numpy() > 0).all(axis=0)
    hist = hist.loc[:, complete]
    edges, fallback_count = calculate_edges(np.log(hist), directed=True)
    exact_sort_refits = edges.attrs.get('exact_sort_tie_refits', 0)
    failed_pairs = edges.attrs.get('adf_failed_pairs_skipped', 0)
    adf_seconds = time.monotonic() - started
    baseline, matching = literal_matching(edges)
    index = edges.set_index(['ASSET_1', 'ASSET_2'])
    records = []
    for style, pairs in [('BASELINE', list(baseline)), ('MAX_WEIGHT', list(matching))]:
        for position, (x, y) in enumerate(pairs):
            record = index.loc[(x, y)].to_dict()
            record.update(METHOD=style, POSITION=position, ASSET_1=x, ASSET_2=y,
                          CUTOFF=cutoff, WINDOW_START=hist.index[0])
            records.append(record)
    selected = pd.DataFrame(records)
    selected['USEDLAG'] = selected.USEDLAG.astype(int)
    selected['T_STAT_ROUNDED'] = selected.T_STAT.round(2)
    tmp = output.with_suffix('.tmp.parquet')
    selected.to_parquet(tmp, index=False)
    tmp.replace(output)
    info = {'schema': 2, 'source_commit': SOURCE_COMMIT,
            'source_sha256': identity['price_sha256'], 'cache_identity': identity,
            'selection_sha256': sha256(output), 'cutoff': str(cutoff.date()),
            'window_start': str(hist.index[0].date()), 'window_rows': len(hist),
            'source_month': cutoff.strftime('%Y-%m'), 'source_cache_name': key + '.csv',
            'global_universe_n': len(names), 'global_universe_order': names,
            'eligible_n': hist.shape[1], 'eligible_order': list(hist.columns),
            'candidate_n': len(edges), 'baseline_n': len(baseline), 'matching_n': len(matching),
            'lag_counts': {str(k): int(v) for k, v in edges.USEDLAG.value_counts().sort_index().items()},
            'fallback_count': fallback_count, 'exact_sort_tie_refits': exact_sort_refits,
            'adf_failed_pairs_skipped': failed_pairs,
            'incomplete_pairs_skipped': len(names)*(len(names)-1)-hist.shape[1]*(hist.shape[1]-1),
            'adf_seconds': adf_seconds, 'total_seconds': time.monotonic()-started,
            **environment(),
            'matching': 'rating=T_STAT.round(2); maxcardinality=1; default weight key; no p filter',
            'baseline': 'all directed tuples with P_VALUE <= 0.05; no top-N cap',
            'selection_file': output.name,
            'limitations': ['The historical author hash/set and library environment is unknown.',
                           'The entire-sample universe intentionally preserves source lookahead.',
                           'These selection files do not recover the missing daily pair-cache generator.']}
    meta_tmp = meta_path.with_suffix('.tmp.json')
    meta_tmp.write_text(json.dumps(info, indent=2))
    meta_tmp.replace(meta_path)
    return info


def generate_all(prices_path=DEFAULT_PRICES, output_dir=None, cutoffs=None, workers=4, force=False):
    if workers < 1:
        raise ValueError('workers must be positive')
    prices = pd.read_parquet(prices_path)
    boundaries = [pd.Timestamp(x) for x in (cutoffs if cutoffs is not None else selection_cutoffs(prices))]
    cuts = sorted(boundaries[:-1], key=lambda x: (x < pd.Timestamp('2020-01-01'), x))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_month, c, prices_path, output_dir, force) for c in cuts]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(json.dumps({k: r[k] for k in ['cutoff', 'candidate_n', 'baseline_n', 'matching_n', 'total_seconds']}), flush=True)
    manifest = {'cutoffs': [str(x.date()) for x in boundaries],
                'selection_files': len(results), 'source_default_start': '2020-01-31',
                'expanded_report_start': '2017-01-03', 'end': '2023-05-31',
                'intervals': 'inclusive boundaries; the runner must retain duplicate boundary dates',
                'results': sorted(results, key=lambda x: x['cutoff'])}
    folder = Path(output_dir).parent if output_dir is not None else ROOT / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def _original_create_spread(source_path):
    """Load only the unedited original function, avoiding its top-level I/O."""
    from sklearn import linear_model
    tree = ast.parse(Path(source_path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_spread')
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {'pd': pd, 'np': np, 'linear_model': linear_model}
    exec(compile(module, str(source_path), 'exec'), namespace)
    original = namespace['create_spread']
    # New sklearn rejects the original mixed string/int feature labels. Clear
    # Series names only; observations, values, index, and function are unchanged.
    return lambda x, y: original(x.rename(None), y.rename(None))


def validate(prices_path=DEFAULT_PRICES, samples=80, source_path=None):
    """400 reference pairs at default samples, plus complete small graph checks."""
    from itertools import permutations
    from sklearn.linear_model import LinearRegression
    source_path = Path(source_path or ROOT / 'source/betas.py')
    create_spread = _original_create_spread(source_path)
    prices = pd.read_parquet(prices_path)
    if sha256(prices_path) != EXPECTED_SHA256:
        raise ValueError('Unexpected source price bytes')
    names = source_universe(prices)
    rng = np.random.default_rng(8841)
    records, graphs = [], []
    cutoffs = ['2016-12-30', '2018-01-31', '2020-01-31', '2020-03-31', '2023-04-28']
    for cutoff in cutoffs:
        hist = prices[names].loc[:cutoff].tail(504)
        hist = np.log(hist.loc[:, hist.notna().all() & (hist > 0).all()])
        a = hist.to_numpy()
        ii, jj = np.where(~np.eye(a.shape[1], dtype=bool))
        chosen = rng.choice(len(ii), samples, replace=False)
        residuals, selected_names = [], []
        for k in chosen:
            residuals.append(create_spread(hist.iloc[:, ii[k]], hist.iloc[:, jj[k]]).to_numpy())
            selected_names.append([str(hist.columns[ii[k]]), str(hist.columns[jj[k]])])
        residuals = np.asarray(residuals)
        t, p, lag, aic, fallback = adf_aic_batch(residuals)
        exact = [adfuller(e) for e in residuals]
        result = {'cutoff': cutoff, 'samples': samples, 'directed_pairs': selected_names,
                  'max_t_error': float(max(abs(t[i]-r[0]) for i, r in enumerate(exact))),
                  'max_p_error': float(max(abs(p[i]-r[1]) for i, r in enumerate(exact))),
                  'max_aic_error': float(max(abs(aic[i]-r[5]) for i, r in enumerate(exact))),
                  'lag_mismatches': int(sum(lag[i] != r[2] for i, r in enumerate(exact))),
                  'baseline_threshold_mismatches': int(sum((p[i] <= .05) != (r[1] <= .05) for i, r in enumerate(exact))),
                  'fallback_count': fallback}
        assert result['lag_mismatches'] == 0 and result['baseline_threshold_mismatches'] == 0, result
        assert result['max_t_error'] < 1e-8 and result['max_aic_error'] < 1e-7, result
        records.append(result)
        if cutoff in ['2016-12-30', '2020-01-31', '2023-04-28']:
            small = hist.iloc[:, :12]
            reference = []
            for xname, yname in permutations(small.columns, 2):
                x, y = small[xname], small[yname]
                fit = LinearRegression().fit(np.column_stack([x, np.ones(len(x))]), y)
                e = create_spread(x, y)
                r = adfuller(e)
                reference.append({'ASSET_1': xname, 'ASSET_2': yname, 'BETA': fit.coef_[0],
                                  'ALPHA': fit.intercept_, 'T_STAT': r[0], 'P_VALUE': r[1]})
            source_edges = pd.DataFrame(reference).sort_values('P_VALUE').reset_index(drop=True)
            fast_edges, _ = calculate_edges(small, directed=True)
            b0, m0 = literal_matching(source_edges)
            b1, m1 = literal_matching(fast_edges)
            roundtrip = pd.read_csv(io.StringIO(fast_edges[['ASSET_1', 'ASSET_2', 'T_STAT', 'P_VALUE']].to_csv(index=False)))
            b2, m2 = literal_matching(roundtrip)
            left = source_edges.set_index(['ASSET_1', 'ASSET_2'])
            right = fast_edges.set_index(['ASSET_1', 'ASSET_2']).loc[left.index]
            check = {'cutoff': cutoff, 'full_directed_subgraph_edges': len(source_edges),
                     'baseline_set_equal': b0 == b1, 'matching_oriented_tuple_set_equal': m0 == m1,
                     'source_csv_roundtrip_baseline_equal': b1 == b2,
                     'source_csv_roundtrip_matching_equal': m1 == m2,
                     'max_beta_error': float((left.BETA-right.BETA).abs().max()),
                     'max_alpha_error': float((left.ALPHA-right.ALPHA).abs().max()),
                     'max_t_error': float((left.T_STAT-right.T_STAT).abs().max())}
            assert all(check[k] for k in ['baseline_set_equal', 'matching_oriented_tuple_set_equal',
                                          'source_csv_roundtrip_baseline_equal', 'source_csv_roundtrip_matching_equal']), check
            assert check['max_t_error'] < 1e-8, check
            graphs.append(check)
    t, p, lag, _, _ = adf_aic_batch(np.stack([rng.normal(size=504), np.zeros(504)]))
    assert np.isfinite(t[0]) and np.isnan(t[1]) and lag[1] == -1
    rejected = False
    try:
        create_spread(pd.Series([0., np.nan, 2.]), pd.Series([0., 1., 2.]))
    except ValueError:
        rejected = True
    assert rejected
    return {'passed': True, 'source_commit': SOURCE_COMMIT,
            'test': 'Unedited original betas.py create_spread + default statsmodels.adfuller versus accelerated calculation',
            'price_sha256': sha256(prices_path), 'generator_sha256': sha256(__file__),
            'source_betas_sha256': sha256(source_path), 'environment': environment(),
            'reference_compatibility_shim': 'Clear input Series names only: modern sklearn rejects the original mixed string/int DataFrame feature labels. Original function AST and numerical inputs are unchanged.',
            'source_universe_n': len(names), 'source_universe_order': names,
            'samples': sum(r['samples'] for r in records), 'records': records,
            'full_small_graph_checks': graphs, 'constant_residual_skip_matches_source': True,
            'nan_OLS_skip_matches_source': True, 'tolerance_t': 1e-8, 'tolerance_aic': 1e-7,
            'scope': 'Five cutoffs and three complete 132-edge subgraphs; not full historical author cache identity.',
            'limitations': ['Unknown author hash/set order and historical package environment.',
                           'Daily pair SPREAD/BETA generation is absent from public code and is not covered by this selection validation.',
                           'The original entire-sample universe and missing graph weight are intentionally retained.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=DEFAULT_PRICES)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/selection')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--month')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--validation-output', type=Path, default=ROOT / 'checks/selection_validation.json')
    args = parser.parse_args()
    if args.validate:
        result = validate(args.prices)
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.write_text(json.dumps(result, indent=2))
        print(json.dumps({'passed': result['passed'], 'samples': result['samples'],
                          'graphs': result['full_small_graph_checks']}), flush=True)
    if args.month:
        result = generate_month(args.month, args.prices, args.output_dir, args.force)
        print(json.dumps({k: result[k] for k in ['cutoff', 'candidate_n', 'baseline_n', 'matching_n']}), flush=True)
    if args.all:
        generate_all(args.prices, args.output_dir, workers=args.workers, force=args.force)


if __name__ == '__main__':
    main()

