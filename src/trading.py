"""Preserve the public Q trade arithmetic with an explicit missing-cache assumption.

The public repository does not contain the writer of its daily pair SPREAD/BETA
CSVs. We supply trailing-504-observation log-OLS endpoint residuals and betas.
This is a completion assumption, not recovered author code. The visible trade
operations remain unchanged: Q20, Q25-Q75 denominator, round, shift1, same-day
beta shares, current-price capital, exp(beta)->log, NaNs, inclusive endpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Tuple

import numpy as np
import pandas as pd

WINDOW = 504
BOUND_WINDOW = 20
COMPLETION = (
    "Missing per-pair CSV generator supplied as 504-observation rolling "
    "log-OLS endpoint SPREAD/BETA; this is a reconstruction assumption, "
    "not recovered author code."
)


def _pair_list(pairs):
    if isinstance(pairs, pd.DataFrame):
        columns = ['x', 'y'] if {'x', 'y'} <= set(pairs.columns) else ['ASSET_1', 'ASSET_2']
        pairs = pairs[columns].itertuples(index=False, name=None)
    normalized, seen = [], set()
    for pair in pairs:
        pair = tuple(pair.split('-')) if isinstance(pair, str) else tuple(pair)
        if len(pair) != 2:
            raise ValueError(f'Expected oriented pair: {pair}')
        # Original pnl/cap dictionaries overwrite a repeated identical key.
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    return normalized


def _bounds(prices, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    left = int(prices.index.searchsorted(start, side='left'))
    right = int(prices.index.searchsorted(end, side='right'))
    if left >= right:
        raise ValueError(f'No observations in inclusive interval {start}..{end}')
    warm = max(0, left - WINDOW - BOUND_WINDOW)
    return warm, left, right


def rolling_endpoint(x, y, window=WINDOW):
    """Complete-window intercept OLS; no imputation or future backfill."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    ok = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(ok, x, 0.), np.where(ok, y, 0.)

    def roll(a):
        cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        result = np.full_like(a, np.nan, dtype=float)
        if len(a) >= window:
            result[window - 1:] = cs[window:] - cs[:-window]
        return result

    count, sx, sy = roll(ok.astype(float)), roll(xx), roll(yy)
    variance = roll(xx * xx) - sx * sx / window
    covariance = roll(xx * yy) - sx * sy / window
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = covariance / variance
        alpha = (sy - beta * sx) / window
    beta[(count < window) | (variance < 1e-12)] = np.nan
    alpha[~np.isfinite(beta)] = np.nan
    spread = y - alpha - beta * x
    return beta, spread


def public_arithmetic(log_x, log_y, beta, spread):
    """Literal public performance.py operations; no algebraic simplification."""
    residual = pd.DataFrame(spread)
    rolling = residual.rolling(BOUND_WINDOW)
    qscore = (residual - rolling.median()) / (rolling.quantile(.25) - rolling.quantile(.75))
    signal = np.sign(qscore) * np.round(qscore.abs())
    position = signal.shift().to_numpy()
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        px, py = np.exp(log_x), np.exp(log_y)
        exp_beta = np.exp(beta)
        capital_inner = py + np.abs(beta * px)
        capital_inner = np.where(np.isnan(capital_inner), 0., capital_inner)
        capital = np.abs(position) * capital_inner
        dx = np.vstack([np.full((1, px.shape[1]), np.nan), np.diff(px, axis=0)])
        dy = np.vstack([np.full((1, py.shape[1]), np.nan), np.diff(py, axis=0)])
        inner = dy + (-np.log(exp_beta) * dx)
        inner = np.where(np.isnan(inner), 0., inner)
        pnl = position * inner
    return pnl, capital, position, qscore.to_numpy()


def trade_interval(prices, pairs, start, end, batch_size=128):
    """Return daily results and metadata for inclusive [start, end].

    Prices are the original raw price/NaN frame with sorted unique dates. Pairs
    are oriented (x, y) tuples or rows. Callers must retain duplicate month-end
    dates when concatenating intervals to preserve the public runner.
    """
    pairs = _pair_list(pairs)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('Price dates must be sorted and unique')
    warm, left, right = _bounds(prices, start, end)
    dates = prices.index[left:right]
    total_pnl = np.zeros(len(dates))
    total_capital = np.zeros(len(dates))
    observed = np.zeros(len(dates), dtype=np.int64)
    lookup = {name: i for i, name in enumerate(prices.columns)}
    data = prices.iloc[warm:right].to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        logdata = np.log(data)
    # The omitted prefix has no complete 504-row regression and hence no Q.
    trim = min(WINDOW - 1, len(data))
    output_start = left - warm - trim
    if output_start < 0:
        raise ValueError('Not enough 504-row history before interval')
    overflow_beta_count = nonfinite_pnl = nonfinite_capital = 0
    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset:offset + batch_size]
        xi = np.array([lookup[x] for x, y in batch])
        yi = np.array([lookup[y] for x, y in batch])
        x, y = logdata[:, xi], logdata[:, yi]
        beta, spread = rolling_endpoint(x, y)
        with np.errstate(over='ignore', invalid='ignore'):
            overflow_beta_count += int(np.count_nonzero(
                np.isfinite(beta[trim:]) & np.isinf(np.exp(beta[trim:]))
            ))
        pnl, capital, position, _ = public_arithmetic(
            x[trim:], y[trim:], beta[trim:], spread[trim:]
        )
        pnl, capital, position = pnl[output_start:], capital[output_start:], position[output_start:]
        nonfinite_pnl += int(np.count_nonzero(np.isinf(pnl)))
        nonfinite_capital += int(np.count_nonzero(np.isinf(capital)))
        # Original DataFrame.sum skips NaN, retaining infinities.
        with np.errstate(invalid='ignore'):
            total_pnl += np.nansum(pnl, axis=1)
            total_capital += np.nansum(capital, axis=1)
        observed += np.count_nonzero(np.isfinite(position) & (position != 0), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = total_pnl / total_capital
    daily = pd.DataFrame({
        'pnl': total_pnl, 'capital': total_capital,
        'returns': returns, 'active_pairs': observed,
    }, index=dates)
    daily.index.name = 'date'
    metadata = {
        'start': str(pd.Timestamp(start).date()), 'end': str(pd.Timestamp(end).date()),
        'pairs': len(pairs), 'rows': len(daily),
        'nan_returns': int(daily.returns.isna().sum()),
        'infinite_returns': int(np.isinf(daily.returns).sum()),
        'exp_beta_overflow_elements': overflow_beta_count,
        'infinite_pair_pnl_elements': nonfinite_pnl,
        'infinite_pair_capital_elements': nonfinite_capital,
        'inclusive_endpoints': True,
        'signal': 'Q20 only; denominator Q25-Q75; round; shift1',
        'allocation': 'same-day beta shares versus one share; same-day gross capital denominator',
        'missing_cache_completion': COMPLETION, 'batch_size': batch_size,
    }
    return daily, metadata


def make_pair_frame(prices, x, y, start, end):
    """Supply the missing pair CSV schema, with observations for Q warmup."""
    warm, _, right = _bounds(prices, start, end)
    data = prices.iloc[warm:right][[x, y]]
    with np.errstate(divide='ignore', invalid='ignore'):
        lx, ly = np.log(data[x].to_numpy()), np.log(data[y].to_numpy())
    beta, spread = rolling_endpoint(lx, ly)
    return pd.DataFrame({
        x: lx, y: ly, 'BETA': beta[:, 0], 'SPREAD': spread[:, 0],
    }, index=data.index)


@contextmanager
def original_performance(source_path, home, pair_names):
    """Import unchanged source through absent-I/O shims; never change HOME."""
    names = ['tools', 'utils', 'multiprocess']
    saved = {name: sys.modules.get(name) for name in names}
    tools = ModuleType('tools')
    tools.fetch_optimal_pairs = lambda date, style: list(pair_names)
    utils = ModuleType('utils')
    utils.pd, utils.np, utils.Tuple, utils.HOME = pd, np, Tuple, str(home)
    sys.modules['tools'], sys.modules['utils'] = tools, utils
    # run_trade itself does not use multiprocessing.
    if importlib.util.find_spec('multiprocess') is None:
        sys.modules['multiprocess'] = multiprocessing
    module = ModuleType('_original_performance_readonly_sample')
    module.__file__ = str(source_path)
    try:
        exec(compile(Path(source_path).read_text(), str(source_path), 'exec'), module.__dict__)
        yield module
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def verify_original_sample(prices, pairs, start, end, source_path, atol=1e-10):
    """Check source trade arithmetic with supplied completion CSVs.

    This function shares rolling_endpoint with production, so it does not
    independently validate the missing-cache completion. See trading_validation
    for separate sklearn daily regressions and an end-to-end source comparison.
    """
    import os
    pairs = _pair_list(pairs)
    before_home = os.environ.get('HOME')
    with TemporaryDirectory(prefix='literal-run-trade-') as tmp:
        home = Path(tmp)
        folder = home / 'spread/pairs'
        folder.mkdir(parents=True)
        for x, y in pairs:
            make_pair_frame(prices, x, y, start, end).to_csv(
                folder / f'{x}_{y}.csv', date_format='%Y-%m-%d'
            )
        names = [x + '-' + y for x, y in pairs]
        with original_performance(source_path, home, names) as original:
            reference = original.Performance.run_trade((
                str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()), 'MAX_WEIGHT'
            ))
            vector, metadata = trade_interval(prices, pairs, start, end)
            candidate = vector.returns
            np.testing.assert_array_equal(
                reference.index.to_numpy(dtype='datetime64[ns]'),
                candidate.index.to_numpy(dtype='datetime64[ns]'),
            )
            np.testing.assert_allclose(reference, candidate, rtol=0, atol=atol, equal_nan=True)
            max_error = float(np.nanmax(np.abs(reference.to_numpy() - candidate.to_numpy()))) if reference.notna().any() else 0.
            report = original.Performance.simple_report(reference).PERFORMANCE.to_dict()
        assert os.environ.get('HOME') == before_home
    return {
        'start': str(pd.Timestamp(start).date()), 'end': str(pd.Timestamp(end).date()),
        'pairs': pairs, 'rows': len(reference), 'passed': True,
        'absolute_tolerance': atol, 'max_return_difference': max_error,
        'nan_mask_equal': bool(np.array_equal(reference.isna(), candidate.isna())),
        'original_simple_report': {k: float(v) for k, v in report.items()},
        'vector_metadata': metadata, 'environment_HOME_unchanged': True,
        'source_file_unmodified': str(source_path),
        'shims': [
            'tools.fetch_optimal_pairs returns supplied oriented pairs',
            'utils exposes pd,np,Tuple and local HOME; original utils side effects are not executed',
            'multiprocess maps to stdlib only if absent',
        ],
        'temporary_pair_CSVs': 'automatically removed',
        'scope': 'Original arithmetic equivalence conditional on the shared completion input.',
    }


def summarize_original_style(returns):
    """Preserve source semantics: Sharpe excludes NaN; cumprod fills NaN=0."""
    r = pd.Series(returns, dtype=float)
    return {
        'rows_including_duplicate_dates': len(r),
        'non_nan_rows': int(r.notna().sum()),
        'sharpe': float(np.sqrt(252) * r.mean() / r.std()),
        'sortino': float(np.sqrt(252) * r.mean() / r[r < 0].std()),
        'cumulative_return': float(((1 + r.fillna(0)).cumprod() - 1).iloc[-1]),
        'simple_return': float(r.fillna(0).cumsum().iloc[-1]),
    }
