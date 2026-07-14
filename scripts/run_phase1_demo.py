"""Phase 1 end-to-end demo: generate a feed, optimize memory, compute signals.

Usage (from the repo root)::

    python scripts/run_phase1_demo.py [--n-events 1000000] [--levels 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running straight from a source checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine import (  # noqa: E402
    LOBConfig,
    MicrostructureFeatures,
    SyntheticLOBGenerator,
    frame_memory_mb,
    optimize_memory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-events", type=int, default=1_000_000)
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()

    config = LOBConfig(n_events=args.n_events, n_levels=args.levels)
    print(f"config: {config.n_events:,} events x {config.n_levels} levels\n")

    # --- 1. Generation ------------------------------------------------
    t0 = time.perf_counter()
    book = SyntheticLOBGenerator(config).generate()
    gen_s = time.perf_counter() - t0
    print(f"[generate]  {gen_s:6.3f}s  ({config.n_events / gen_s:,.0f} events/s)")

    # --- 2. Memory optimization ---------------------------------------
    raw_mb = frame_memory_mb(book)
    t0 = time.perf_counter()
    book = optimize_memory(book, tick_size=config.tick_size)
    opt_s = time.perf_counter() - t0
    slim_mb = frame_memory_mb(book)
    print(
        f"[downcast]  {opt_s:6.3f}s  {raw_mb:,.1f} MiB -> {slim_mb:,.1f} MiB "
        f"({100 * (1 - slim_mb / raw_mb):.0f}% saved)"
    )

    # --- 3. Feature extraction ----------------------------------------
    t0 = time.perf_counter()
    feats = MicrostructureFeatures(book)
    signals = feats.compute_all()
    feat_s = time.perf_counter() - t0
    print(
        f"[features]  {feat_s:6.3f}s  ({config.n_events / feat_s:,.0f} events/s, "
        f"{signals.shape[1]} signals)\n"
    )

    print("--- sample (first 5 events) " + "-" * 24)
    with_time = signals.copy()
    with_time.insert(0, "time", book["ts_ns"].astype("datetime64[ns]"))
    print(with_time.head().to_string(float_format=lambda x: f"{x:.5f}"))

    print("\n--- signal summary " + "-" * 33)
    print(
        signals[["spread_bps", "obi_l1", "micro_dev"]]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .to_string(float_format=lambda x: f"{x:.6f}")
    )

    # Sanity: the synthetic mid walk is independent of the book state, so
    # micro_dev must NOT predict the next mid move. A near-zero correlation
    # here validates that any alpha we find later on structured data is
    # real, not an artifact (e.g. look-ahead leakage) of the pipeline.
    mid = signals["mid"].to_numpy()
    fwd_return = np.diff(mid) / mid[:-1]
    corr = float(np.corrcoef(signals["micro_dev"].to_numpy()[:-1], fwd_return)[0, 1])
    print(
        f"\nnull check: corr(micro_dev, next-tick mid return) = {corr:+.5f} "
        "(expected ~0: the synthetic feed has no tradeable signal)"
    )


if __name__ == "__main__":
    main()
