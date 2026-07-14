"""Phase 2 demo: AR(1) spreads and dual-clock realized volatility at line rate.

Usage (from the repo root)::

    python scripts/run_phase2_demo.py [--n-events 1000000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine import (  # noqa: E402
    LOBConfig,
    MicrostructureFeatures,
    SyntheticLOBGenerator,
    realized_vol_events,
    realized_vol_wallclock,
    wallclock_window_starts,
)

NS_PER_SECOND = 1_000_000_000


def best_of(fn, reps: int = 3) -> tuple[float, object]:
    """Best-of-N wall time; standard practice for microbenchmarks."""
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-events", type=int, default=1_000_000)
    args = parser.parse_args()

    config = LOBConfig(n_events=args.n_events)
    n = config.n_events
    print(f"config: {n:,} events, spread AR(1) phi={config.spread_phi}\n")

    # --- 1. Generation with AR(1) spreads ------------------------------
    t0 = time.perf_counter()
    book = SyntheticLOBGenerator(config).generate()
    gen_s = time.perf_counter() - t0
    spread_ticks = (
        (book["ask_px_0"] - book["bid_px_0"]).to_numpy() / config.tick_size
    )
    lag1 = np.corrcoef(spread_ticks[:-1], spread_ticks[1:])[0, 1]
    print(
        f"[generate]   {gen_s:6.3f}s  ({n / gen_s:,.0f} events/s)  "
        f"spread: mean={spread_ticks.mean():.2f} ticks, "
        f"lag-1 autocorr={lag1:.3f} (latent phi={config.spread_phi})"
    )

    # --- 2. Micro-price -------------------------------------------------
    micro = MicrostructureFeatures(book).micro_price()
    ts_ns = book["ts_ns"].to_numpy()

    # --- 3. Event-time realized vol -------------------------------------
    ev_s, vol_ev = best_of(lambda: realized_vol_events(micro, window=200))
    print(f"[vol/events] {ev_s:6.3f}s  ({n / ev_s:,.0f} events/s)  window=200 ticks")

    # --- 4. Wall-clock realized vol --------------------------------------
    wc_s, vol_wc = best_of(
        lambda: realized_vol_wallclock(micro, ts_ns, horizon_ns=NS_PER_SECOND)
    )
    print(f"[vol/clock]  {wc_s:6.3f}s  ({n / wc_s:,.0f} events/s)  horizon=1s")

    # --- 5. Honest benchmark vs pandas -----------------------------------
    # Same task, raw arrays in -> vol array out. Pandas' Cython indexer
    # finds window bounds with an O(n) two-pointer scan; our searchsorted
    # pays O(n log n), so pandas wins the single-statistic race. Our edge is
    # amortization: bounds are computed once and every further statistic
    # over the same horizon is a pure prefix-sum pass.
    def pandas_wallclock() -> np.ndarray:
        r2 = pd.Series(
            np.diff(np.log(micro)) ** 2,
            index=pd.DatetimeIndex(ts_ns[1:].astype("datetime64[ns]")),
        )
        return np.sqrt(r2.rolling(pd.Timedelta(seconds=1)).sum()).to_numpy()

    pd_s, vol_pd = best_of(pandas_wallclock)
    max_diff = float(np.nanmax(np.abs(vol_wc[1:] - vol_pd)))

    st_s, starts = best_of(lambda: wallclock_window_starts(ts_ns, NS_PER_SECOND))
    reuse_s, _ = best_of(
        lambda: realized_vol_wallclock(
            micro, ts_ns, NS_PER_SECOND, demean=True, starts=starts
        )
    )
    print(
        f"[vs pandas]  {pd_s:6.3f}s  .rolling('1s') end-to-end, identical "
        f"numbers (max |diff| = {max_diff:.1e}). Pandas wins the "
        f"single-statistic race {wc_s / pd_s:.1f}x: its Cython indexer "
        "finds bounds with an O(n) two-pointer scan, our searchsorted "
        "pays O(n log n)."
    )
    print(
        f"[amortized]  {st_s:6.3f}s  bounds once, then {reuse_s:.3f}s per "
        "further statistic (here: demeaned vol) over the same bounds -- "
        "per-statistic parity with pandas, on bounds we control: any "
        "prefix-summable statistic and burst-exact point-in-time semantics. "
        "Closing the O(n log n) -> O(n) bounds gap is a Phase 4 target."
    )

    # --- 6. The two clocks measure different things ----------------------
    both = np.isfinite(vol_ev) & np.isfinite(vol_wc)
    corr = np.corrcoef(vol_ev[both], vol_wc[both])[0, 1]
    print("\n--- dual-clock summary " + "-" * 29)
    frame = pd.DataFrame({"vol_event_200t": vol_ev, "vol_clock_1s": vol_wc})
    print(
        frame.describe()
        .loc[["mean", "std", "min", "max"]]
        .to_string(float_format=lambda x: f"{x:.6f}")
    )
    print(
        f"\ncorr(event-clock, wall-clock) = {corr:.3f} -- related but not "
        "interchangeable: the event clock stretches wall-clock windows when "
        "arrival intensity drops."
    )


if __name__ == "__main__":
    main()
