"""Canonical LOB frame schema and lossless memory optimization.

Real tick datasets are ingested at float64/int64 by default, which wastes
memory bandwidth — the true bottleneck for vectorized workloads (a modern core
retires arithmetic far faster than DRAM can feed it, so halving bytes-per-row
is worth more than micro-optimizing FLOPs). This module downcasts columns to
the smallest dtype that provably loses nothing:

* **Sizes** — downcast to the smallest unsigned integer that holds the
  observed maximum (data-driven, via ``pd.to_numeric``).
* **Prices** — downcast float64 -> float32 only if the worst-case round-trip
  error across the column is under half a tick, i.e. every price still rounds
  back to the exact tick it came from. For a $100 name with a $0.01 tick,
  float32's ~1e-5 relative precision passes easily; for a $50,000 name it
  would fail the check and stay float64 — the guard makes the decision, not
  a hard-coded assumption.

The zero-loss alternative, storing prices as int32 *tick counts*, is what
production systems do; we keep floats at this layer for Pandas ergonomics and
revisit integer ticks in the benchmarking phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_PREFIXES: tuple[str, ...] = ("bid_px_", "ask_px_")
SIZE_PREFIXES: tuple[str, ...] = ("bid_sz_", "ask_sz_")


def frame_memory_mb(df: pd.DataFrame) -> float:
    """Total memory footprint of a DataFrame in MiB (deep, incl. index)."""
    return float(df.memory_usage(index=True, deep=True).sum()) / 2**20


def optimize_memory(df: pd.DataFrame, *, tick_size: float) -> pd.DataFrame:
    """Return a copy of ``df`` with columns downcast to lossless minimal dtypes.

    Parameters
    ----------
    df
        A frame in the canonical LOB schema (``ts_ns``, ``{bid,ask}_px_{k}``,
        ``{bid,ask}_sz_{k}``). Unrecognized columns pass through untouched.
    tick_size
        Minimum price increment; defines the price round-trip tolerance.

    Returns
    -------
    pd.DataFrame
        Shallow copy with downcast columns. The input frame is not modified.
    """
    if tick_size <= 0.0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")

    out = df.copy(deep=False)
    max_price_err = tick_size / 2.0

    for col in df.columns:
        if col.startswith(SIZE_PREFIXES):
            out[col] = pd.to_numeric(df[col], downcast="unsigned")
        elif col.startswith(PRICE_PREFIXES):
            values = df[col].to_numpy(dtype=np.float64)
            as_f32 = values.astype(np.float32)
            round_trip_err = np.abs(as_f32.astype(np.float64) - values).max()
            if round_trip_err < max_price_err:
                out[col] = as_f32
    return out
