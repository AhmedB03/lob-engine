"""Correctness-gated wall-clock volatility benchmarks, with no speed claims.

Every method receives identical prices and timestamps and includes its own
return calculation and window setup. Data generation, validation, and output
serialization are excluded from timings. No precomputed bounds are supplied.
"""
from __future__ import annotations

import os
import platform
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from lob_engine.features.volatility import realized_vol_wallclock


def loop_volatility(prices, timestamps, horizon_ns):
    """Direct window reductions; monotonically advance the left boundary."""
    returns = np.diff(np.log(prices))
    out = np.full(len(prices), np.nan)
    start = 0
    for end in range(1, len(prices)):
        while start < end and timestamps[start + 1] <= timestamps[end] - horizon_ns:
            start += 1
        if start < end:
            window = returns[start:end]
            out[end] = np.sqrt(np.sum(window * window))
    return out


def pandas_volatility(prices, timestamps, horizon_ns):
    returns = np.diff(np.log(prices))
    series = pd.Series(returns * returns, index=pd.to_datetime(timestamps[1:], unit="ns"))
    values = series.rolling(pd.Timedelta(horizon_ns, unit="ns"), closed="right", min_periods=1).sum()
    return np.concatenate(([np.nan], np.sqrt(values.to_numpy())))


def workload(n_events: int, seed: int, scenario: str):
    if n_events < 2:
        raise ValueError("n_events must be at least 2")
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n_events)))
    if scenario == "regular":
        gaps = np.full(n_events, 1_000_000, dtype=np.int64)
    elif scenario == "bursts":
        gaps = rng.choice([0, 0, 1_000_000, 2_000_000], n_events)
    elif scenario == "gaps":
        gaps = rng.choice([1_000_000, 2_000_000, 200_000_000], n_events)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return prices, np.cumsum(gaps, dtype=np.int64)


METHODS = {
    "python_loop": loop_volatility,
    "numpy_prefix": realized_vol_wallclock,
    "pandas_rolling": pandas_volatility,
}


def run_benchmarks(sizes=(1_000, 10_000, 100_000), *, repeats=5, seed=42,
                   scenarios=("regular", "bursts", "gaps"), horizon_ns=100_000_000):
    sizes, scenarios = tuple(sizes), tuple(scenarios)
    if repeats < 1 or horizon_ns < 1 or not sizes or not scenarios:
        raise ValueError("positive repeats/horizon and nonempty sizes/scenarios required")
    results = []
    for scenario in scenarios:
        for n_events in sizes:
            prices, timestamps = workload(n_events, seed, scenario)
            args = prices, timestamps, horizon_ns
            expected = loop_volatility(*args)
            # Validate the entire output, including NaN positions, before timing.
            # This call also warms each method. Tolerance covers cancellation
            # when subtracting long prefix sums; do not silently skip failures.
            for method in METHODS.values():
                np.testing.assert_allclose(method(*args), expected, rtol=1e-7, atol=1e-10, equal_nan=True)
            samples = {name: [] for name in METHODS}
            names = list(METHODS)
            for repetition in range(repeats):
                # Rotate order so the same implementation is not always first.
                offset = repetition % len(names)
                for name in names[offset:] + names[:offset]:
                    start = perf_counter()
                    METHODS[name](*args)
                    samples[name].append(perf_counter() - start)
            for name, timings in samples.items():
                seconds = median(timings)
                results.append({
                    "scenario": scenario, "n_events": n_events, "method": name,
                    "median_seconds": seconds, "min_seconds": min(timings),
                    "samples_seconds": timings,
                    "events_per_second": n_events / seconds if seconds > 0 else None,
                    "input_bytes": prices.nbytes + timestamps.nbytes,
                })
    return {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "thread_environment": {key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "config": {"seed": seed, "repeats": repeats, "horizon_ns": horizon_ns,
                   "sizes": list(sizes), "scenarios": list(scenarios)},
        "correctness": {"passed": True, "rtol": 1e-7, "atol": 1e-10, "equal_nan": True},
        "timing_scope": "prices + timestamps to volatility; includes returns and bounds; excludes generation, validation and serialization",
        "results": results,
    }
