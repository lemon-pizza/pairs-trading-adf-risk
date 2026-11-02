"""Read-only reproduction inputs and lazy histories for ADF/risk selection.

The original matching's full candidate graph and monthly cardinality are kept.
Only selection is new: one deterministic orientation per unordered pair, with
all original matching orientations retained. No p-value, top-N, or additional
price-history filter is imposed. Original trading kernels are imported without
modification, including their same-day beta/capital and missing-value rules.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selection import calculate_edges, EXPECTED_SHA256
from trading import WINDOW, BOUND_WINDOW, rolling_endpoint, public_arithmetic, trade_interval

SCHEMA = "original_graph_original_arithmetic_adf_risk_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@lru_cache(maxsize=1)
def _prices():
    path = ROOT / "data/prices.parquet.gz"
    price_hash = digest(path)
    if price_hash != EXPECTED_SHA256:
        raise ValueError("Unexpected original price bytes")
    prices = pd.read_parquet(path)
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted")
    if prices.columns.has_duplicates:
        raise ValueError("Price columns must be unique")
    return prices, price_hash


def original_select(key):
    """Read the unchanged monthly selection and verify its existing receipt."""
    key = pd.Timestamp(str(key)).strftime("%Y%m%d")
    path = ROOT / "data/selection" / f"{key}.parquet"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if digest(path) != receipt["selection_sha256"]:
        raise ValueError(f"Original selection hash mismatch: {key}")
    frame = pd.read_parquet(path)
    base = frame[frame.METHOD.eq("MAX_WEIGHT")].sort_values("POSITION").copy()
    if len(base) != receipt["matching_n"]:
        raise ValueError("Original matching cardinality mismatch")
    endpoints = np.r_[base.ASSET_1.to_numpy(), base.ASSET_2.to_numpy()]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Original matching is not non-overlapping")
    return frame, base, receipt


def _canonicalize(directed, base):
    """Keep baseline directions; otherwise lexicographic, never best ADF."""
    all_rows = directed.copy()
    x, y = all_rows.ASSET_1.astype(str), all_rows.ASSET_2.astype(str)
    all_rows["_lo"] = np.where(x < y, x, y)
    all_rows["_hi"] = np.where(x < y, y, x)
    if all_rows.duplicated(["ASSET_1", "ASSET_2"]).any() or (x == y).any():
        raise ValueError("Invalid directed candidate graph")
    chosen = all_rows.sort_values(["ASSET_1", "ASSET_2"], kind="stable").drop_duplicates(["_lo", "_hi"])
    baseline_pairs = list(base[["ASSET_1", "ASSET_2"]].itertuples(index=False, name=None))
    original_keys = {tuple(sorted(edge)) for edge in baseline_pairs}
    chosen = chosen.loc[[edge not in original_keys for edge in chosen[["_lo", "_hi"]].itertuples(index=False, name=None)]]
    lookup = all_rows.set_index(["ASSET_1", "ASSET_2"], drop=False)
    originals = lookup.loc[baseline_pairs].copy().reset_index(drop=True)
    chosen = pd.concat([chosen, originals], ignore_index=True).sort_values(["_lo", "_hi"], kind="stable").reset_index(drop=True)
    chosen = chosen.drop(columns=["_lo", "_hi"])
    baseline_positions = {edge: i for i, edge in enumerate(baseline_pairs)}
    chosen["SOURCE_POSITION"] = [baseline_positions.get(edge, -1) for edge in chosen[["ASSET_1", "ASSET_2"]].itertuples(index=False, name=None)]
    chosen["IS_SOURCE_M"] = chosen.SOURCE_POSITION.ge(0)
    chosen["SCORE"] = -chosen.T_STAT
    chosen["EDGE_ID"] = np.arange(len(chosen), dtype=int)
    if int(chosen.IS_SOURCE_M.sum()) != len(base):
        raise ValueError("Canonicalization lost original baseline edges")
    if not np.isfinite(chosen.SCORE.to_numpy()).all():
        raise ValueError("Nonfinite ADF score")
    return chosen


def _candidate_graph(key, output_dir, prices, price_hash, source, base, receipt):
    folder = Path(output_dir) / "features"
    folder.mkdir(parents=True, exist_ok=True)
    directed_path = folder / f"{key}_directed.parquet"
    edges_path = folder / f"{key}_edges.parquet"
    meta_path = folder / f"{key}_edges.json"
    identity = {
        "schema": SCHEMA,
        "cutoff": key,
        "price_sha256": price_hash,
        "original_selection_sha256": receipt["selection_sha256"],
        "original_receipt_sha256": digest(ROOT / "data/selection" / f"{key}.json"),
        "selection_kernel_sha256": digest(ROOT / "src/selection.py"),
        "feature_code_sha256": digest(__file__),
        "direction_rule": "retain_source_M_direction_else_lexicographic_among_available_directions",
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if (old.get("identity") == identity and edges_path.exists() and directed_path.exists()
                and digest(edges_path) == old.get("edges_sha256")
                and digest(directed_path) == old.get("directed_sha256")):
            return pd.read_parquet(edges_path), {**old, "reused_verified_candidates": True}
    started = time.monotonic()
    cutoff = pd.Timestamp(key)
    hist = prices.loc[:cutoff, receipt["eligible_order"]].tail(504)
    if len(hist) != 504 or hist.index[-1] != cutoff:
        raise ValueError("Original 504-observation selection window unavailable")
    if not (np.isfinite(hist.to_numpy()).all() and (hist.to_numpy() > 0).all()):
        raise ValueError("Original eligible columns contain invalid formation prices")
    directed, fallbacks = calculate_edges(np.log(hist), directed=True)
    if len(directed) != receipt["candidate_n"]:
        raise ValueError("Rebuilt original full candidate count differs")
    # Every stored source score must agree with the rebuilt same orientation.
    source_unique = source.drop_duplicates(["ASSET_1", "ASSET_2"])
    comparisons = source_unique.merge(directed, on=["ASSET_1", "ASSET_2"], suffixes=("_old", "_new"), validate="one_to_one")
    if len(comparisons) != len(source_unique):
        raise ValueError("Rebuilt graph is missing stored source orientations")
    errors = {}
    for field in ["T_STAT", "P_VALUE", "BETA", "ALPHA"]:
        a = comparisons[field + "_old"].to_numpy(float)
        b = comparisons[field + "_new"].to_numpy(float)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
        errors[field] = float(np.max(np.abs(a - b)))
    old_eligible = set(source.loc[source.METHOD.eq("BASELINE"), ["ASSET_1", "ASSET_2"]].itertuples(index=False, name=None))
    rebuilt_eligible = set(directed.loc[directed.P_VALUE.le(.05), ["ASSET_1", "ASSET_2"]].itertuples(index=False, name=None))
    if old_eligible != rebuilt_eligible:
        raise ValueError("Rebuilt source p<=.05 membership differs")
    edges = _canonicalize(directed, base)
    directed.to_parquet(directed_path, index=False)
    edges.to_parquet(edges_path, index=False)
    metadata = {
        "identity": identity,
        "directed_sha256": digest(directed_path),
        "edges_sha256": digest(edges_path),
        "directed_candidates": len(directed),
        "unordered_candidates": len(edges),
        "source_matching_pairs": len(base),
        "eligible_assets": len(receipt["eligible_order"]),
        "adf_fallbacks": int(fallbacks),
        "source_score_checks": errors,
        "source_eligible_membership_exact": True,
        "formation_first_date": str(hist.index[0].date()),
        "formation_last_date": str(hist.index[-1].date()),
        "candidate_truncation": False,
        "p_value_filter": False,
        "extra_history_filter": False,
        "seconds": time.monotonic() - started,
        "reused_verified_candidates": False,
    }
    _dump(meta_path, metadata)
    return edges, metadata


class LazyPairHistory:
    """Per-edge training P&L/capital computed with unchanged public kernels.

    ``arrays(ids)`` ensures and returns two arrays of shape (days, len(ids)).
    NaNs are replaced by zero separately in P&L and capital, exactly matching
    source column aggregation by nansum. No fill of source prices is performed.
    """
    def __init__(self, prices, edges, cutoff, train_days=252, batch_size=128):
        if isinstance(train_days, bool) or int(train_days) != train_days or train_days < 2:
            raise ValueError("train_days must be an integer >=2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        cutoff = pd.Timestamp(cutoff)
        end = int(prices.index.get_loc(cutoff)) + 1
        left = end - int(train_days)
        warm = max(0, left - WINDOW - BOUND_WINDOW)
        trim = min(WINDOW - 1, end - warm)
        self.output_start = left - warm - trim
        if left < 0 or self.output_start < 0:
            raise ValueError("Insufficient original rolling history for training window")
        self.dates = prices.index[left:end].copy()
        self.train_days = len(self.dates)
        self.trim = trim
        self.batch_size = int(batch_size)
        self.edges = edges
        self.lookup = {name: i for i, name in enumerate(prices.columns)}
        self.pairs = list(edges[["ASSET_1", "ASSET_2"]].itertuples(index=False, name=None))
        # Future prices are absent from the oracle's stored numeric data.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.logdata = np.log(prices.iloc[warm:end].to_numpy(float))
        self.raw_first_date = str(prices.index[warm].date())
        self.last_input_date = str(prices.index[end - 1].date())
        self._cache = {}
        self.nan_pnl_elements = 0
        self.nan_capital_elements = 0
        self.compute_seconds = 0.

    def _ids(self, ids):
        raw = np.asarray(ids)
        if raw.ndim != 1:
            raise ValueError("Edge ids must be a one-dimensional sequence")
        if not len(raw):
            return np.array([], dtype=int)
        if (not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all()
                or (raw != np.floor(raw)).any()):
            raise ValueError("Edge ids must be finite integers")
        result = raw.astype(int)
        if result.min() < 0 or result.max() >= len(self.edges):
            raise ValueError("Edge id outside candidate graph")
        return result

    def ensure(self, ids):
        ids = self._ids(ids)
        missing = sorted(set(ids.tolist()) - self._cache.keys())
        started = time.monotonic()
        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset:offset + self.batch_size]
            xi = [self.lookup[self.pairs[i][0]] for i in batch]
            yi = [self.lookup[self.pairs[i][1]] for i in batch]
            x, y = self.logdata[:, xi], self.logdata[:, yi]
            beta, spread = rolling_endpoint(x, y)
            pnl, capital, _, _ = public_arithmetic(x[self.trim:], y[self.trim:], beta[self.trim:], spread[self.trim:])
            pnl, capital = pnl[self.output_start:], capital[self.output_start:]
            if pnl.shape != (self.train_days, len(batch)) or capital.shape != pnl.shape:
                raise ValueError("Unexpected training array shape")
            if np.isinf(pnl).any() or np.isinf(capital).any():
                raise ValueError("Infinite source arithmetic in candidate training history")
            self.nan_pnl_elements += int(np.isnan(pnl).sum())
            self.nan_capital_elements += int(np.isnan(capital).sum())
            pnl, capital = np.nan_to_num(pnl, nan=0.), np.nan_to_num(capital, nan=0.)
            if (capital < 0).any():
                raise ValueError("Negative source gross capital")
            for col, idx in enumerate(batch):
                self._cache[idx] = (pnl[:, col].copy(), capital[:, col].copy())
        self.compute_seconds += time.monotonic() - started

    def arrays(self, ids):
        ids = self._ids(ids)
        self.ensure(ids)
        if not len(ids):
            return np.zeros((self.train_days, 0)), np.zeros((self.train_days, 0))
        return (np.column_stack([self._cache[int(i)][0] for i in ids]),
                np.column_stack([self._cache[int(i)][1] for i in ids]))

    def portfolio(self, ids):
        pnl, capital = self.arrays(ids)
        total_pnl, total_capital = pnl.sum(axis=1), capital.sum(axis=1)
        returns = np.divide(total_pnl, total_capital, out=np.zeros(self.train_days), where=total_capital > 0)
        return returns, total_pnl, total_capital

    def metadata(self):
        return {
            "training_first_date": str(self.dates[0].date()),
            "training_last_date": str(self.dates[-1].date()),
            "training_observations": self.train_days,
            "raw_first_date": self.raw_first_date,
            "last_input_date": self.last_input_date,
            "computed_edges": len(self._cache),
            "nan_pnl_elements_treated_as_source_nansum": self.nan_pnl_elements,
            "nan_capital_elements_treated_as_source_nansum": self.nan_capital_elements,
            "compute_seconds": self.compute_seconds,
            "risk_zero_capital_policy": "zero_return_for_variance; original test ledger NaNs unchanged",
            "risk_ddof": 1,
            "trading_kernel_sha256": digest(ROOT / "src/trading.py"),
        }

    def export_cached(self, path):
        ids = np.array(sorted(self._cache), dtype=int)
        pnl, capital = self.arrays(ids)
        np.savez_compressed(path, ids=ids, pnl=pnl, capital=capital, dates=self.dates.to_numpy())
        return {"path": str(path), "sha256": digest(path), "cached_edges": len(ids)}


def prepare_month(key, output_dir, train_days=252):
    """Return (candidate frame, original-M ids, lazy oracle, metadata)."""
    key = pd.Timestamp(str(key)).strftime("%Y%m%d")
    prices, price_hash = _prices()
    source, base, receipt = original_select(key)
    edges, candidate_meta = _candidate_graph(key, output_dir, prices, price_hash, source, base, receipt)
    # Source order is kept for exact aggregation/reference checks.
    baseline_ids = edges.loc[edges.IS_SOURCE_M].sort_values("SOURCE_POSITION").index.to_numpy(int)
    oracle = LazyPairHistory(prices, edges, pd.Timestamp(key), train_days=train_days)
    returns, total_pnl, total_capital = oracle.portfolio(baseline_ids)
    reference, _ = trade_interval(prices, base, oracle.dates[0], oracle.dates[-1])
    reference_returns = reference.returns.fillna(0).to_numpy()
    np.testing.assert_allclose(total_pnl, reference.pnl.to_numpy(), atol=1e-8, rtol=1e-11)
    np.testing.assert_allclose(total_capital, reference.capital.to_numpy(), atol=1e-8, rtol=1e-11)
    np.testing.assert_allclose(returns, reference_returns, atol=1e-12, rtol=1e-10)
    metadata = {
        "cutoff": key,
        "train_start": str(oracle.dates[0].date()),
        "train_end": str(oracle.dates[-1].date()),
        "train_days": oracle.train_days,
        "original_selection_sha256": receipt["selection_sha256"],
        "original_selection_receipt_sha256": digest(ROOT / "data/selection" / f"{key}.json"),
        "feature_code_sha256": digest(__file__),
        "trading_kernel_sha256": digest(ROOT / "src/trading.py"),
        "candidates": candidate_meta,
        "history": oracle.metadata(),
        "baseline_ids": baseline_ids.tolist(),
        "K": len(baseline_ids),
        "baseline_variance": float(np.var(returns, ddof=1)),
        "baseline_score": float(edges.iloc[baseline_ids].SCORE.sum()),
        "baseline_training_reference_max_return_error": float(np.max(np.abs(returns - reference_returns))),
        "baseline_training_reference_passed": True,
        "price_sha256": price_hash,
    }
    _dump(Path(output_dir) / "features" / f"{key}_history.json", metadata)
    return edges, baseline_ids, oracle, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default="20161230")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--train-days", type=int, default=252)
    args = parser.parse_args()
    _, _, oracle, metadata = prepare_month(args.key, args.output, args.train_days)
    print(json.dumps({"cutoff": metadata["cutoff"], "K": metadata["K"],
                      "candidate_edges": metadata["candidates"]["unordered_candidates"],
                      "baseline_variance": metadata["baseline_variance"],
                      "reference_max_error": metadata["baseline_training_reference_max_return_error"],
                      "history": oracle.metadata()}, indent=2))
