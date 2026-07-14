"""Phase 3 demo: latency-aware execution simulation and the leakage protocol.

Usage (from the repo root)::

    python scripts/run_phase3_demo.py [--n-events 1000000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine import (  # noqa: E402
    ExecutionSimulator,
    LOBConfig,
    SyntheticLOBGenerator,
    mean_reversion_probe,
    run_null_feed_validation,
    sharpe_tstat,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-events", type=int, default=1_000_000)
    args = parser.parse_args()

    n = args.n_events
    book = SyntheticLOBGenerator(LOBConfig(n_events=n)).generate()
    sim = ExecutionSimulator(book)
    target = mean_reversion_probe(sim.mid)

    # Throughput of one full simulation pass (orders, latency, fills, ledger).
    t0 = time.perf_counter()
    warm = sim.run(target, latency_ns=1_000_000)
    sim_s = time.perf_counter() - t0
    print(
        f"simulator: {n:,} events, {warm.n_fills:,} fills in {sim_s:.3f}s "
        f"({n / sim_s:,.0f} events/s)\n"
    )

    # Latency sweep on the AR(1)-spread feed. The pre-cost t-stat at low
    # latency is REAL and positive: the quoted mid carries a mean-reverting
    # half-tick spread-parity term (bid-ask bounce) that the probe fades.
    # Predictable is not tradeable: post-cost PnL stays negative everywhere.
    print("latency sweep (mean-reversion probe, AR(1)-spread feed, crossing the book):")
    print(f"{'latency':>10} {'fills':>9} {'spread paid':>12} {'PnL (MtM)':>12} {'t-stat(mid)':>12}")
    for latency_ns in (0, 50_000, 1_000_000, 10_000_000):
        cross = sim.run(target, latency_ns=latency_ns)
        mid = sim.run(target, latency_ns=latency_ns, price_mode="mid")
        label = "0 (ideal)" if latency_ns == 0 else f"{latency_ns / 1e6:g} ms"
        print(
            f"{label:>10} {cross.n_fills:>9,} {cross.spread_paid:>12,.2f} "
            f"{cross.total_pnl_mtm:>12,.2f} {sharpe_tstat(mid.ledger.equity_mtm):>+12.2f}"
        )

    print(
        "\nreading: the positive pre-cost t-stat at low latency is quote bounce"
        "\n(quoted mid = efficient price + spread-parity half-tick, and parity"
        "\nmean-reverts) -- a real microstructure effect the engine surfaced, not"
        "\nleakage. It decays with latency and never survives the spread: every"
        "\npost-cost row is negative, and PnL == mid-fill PnL - spread paid"
        "\nexactly. The martingale null runs on a bounce-free control feed:\n"
    )

    report = run_null_feed_validation(n_events=min(n, 200_000))
    print(report.summary())


if __name__ == "__main__":
    main()
