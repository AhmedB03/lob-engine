"""Zero-alpha null-feed validation: the leakage detector.

The mathematics of the detector
-------------------------------
Mark-to-market PnL of any strategy decomposes as
``sum_j X_j * (mid_{j+1} - mid_j)`` plus execution costs. If the marking
price is a **martingale** and the position process X is **causal** (X_j
determined by information up to event j), every term has zero expectation
regardless of the strategy, so:

* pre-cost PnL must be statistically indistinguishable from zero, and
* post-cost PnL must equal pre-cost minus the half-spreads paid — i.e.
  the strategy *must* bleed exactly the spread.

The martingale premise is subtler than it looks — the detector's first
catch was real. With ``bid = m - s//2`` and ``ask = bid + s`` on the tick
grid, the *quoted* midpoint is ``m + (s mod 2)/2`` ticks: the efficient
price plus a half-tick **spread-parity term**. Under the Phase-2 AR(1)
spread that parity term is mean-reverting, so the quoted mid predictably
bounces — a genuine microstructure effect (bid-ask bounce), not a code
bug, and the probe below happily "profits" from it pre-cost while still
losing post-cost. The null protocol therefore runs on a **bounce-free
control feed** (``spread_sigma_ticks = 0``: constant spread, constant
parity, quoted mid == efficient price == martingale). The AR(1)-spread
feed is exercised separately in the test suite to prove the bounce is
real *and* that the spread eats it — predictable is not tradeable.

Now suppose the engine has a look-ahead bug — latency alignment off by one,
a signal indexed against a future row, an execution before its signal.
Then X_j depends on ``mid_{j+1} - mid_j`` itself, and the right probe makes
that dependence explosive: the **last-move probe** ``x_t =
-sign(mid_t - mid_{t-1})`` (a one-tick mean-reversion strategy) has zero
true edge on a random walk, but executed even one event early it captures
``-|mid_t - mid_{t-1}|`` on *every* move — a t-statistic in the hundreds,
not a subtle drift. Misalignment cannot hide.

Three runs, three checks
------------------------
1. ``mid`` fills, zero latency — the frictionless twin. Sharpe t-stat must
   be ~0 (fake alpha shows up here first).
2. ``mid`` fills, real latency — latency alone cannot *create* alpha;
   t-stat must still be ~0.
3. ``cross`` fills, real latency — must lose money, and must satisfy the
   exact decomposition ``pnl_cross = pnl_mid_latency - spread_paid``
   (same fills, prices differ by the half-spread each: an identity, so the
   tolerance is float noise, not statistics).

The test suite also runs the contrapositive: it *injects* a one-event
look-ahead through the engine's ``exec_idx`` testing hook and asserts the
detector fires. A detector that has never caught a planted bug is
decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lob_engine.backtest.engine import ExecutionSimulator
from lob_engine.data.generator import LOBConfig, SyntheticLOBGenerator

_F64 = npt.NDArray[np.float64]


def mean_reversion_probe(mid: _F64) -> npt.NDArray[np.int8]:
    """Sample strategy: fade the last mid move (short after up-ticks, long
    after down-ticks, flat when the mid did not move).

    Zero true edge on a random-walk mid; maximal sensitivity to backward
    misalignment (see module docstring). Uses only information available at
    event t: the move *into* t.
    """
    move = np.sign(np.diff(mid, prepend=mid[:1]))
    return (-move).astype(np.int8)


def sharpe_tstat(equity: _F64) -> float:
    """t-statistic of the mean per-event equity increment.

    ``t = mean(d) / std(d) * sqrt(len(d))`` — this is the per-event Sharpe
    ratio scaled by sqrt(n), so "Sharpe indistinguishable from zero" and
    "|t| below threshold" are the same statement.
    """
    d = np.diff(np.asarray(equity, dtype=np.float64))
    if d.size < 2:
        return 0.0
    s = d.std(ddof=1)
    return 0.0 if s == 0.0 else float(d.mean() / s * np.sqrt(d.size))


@dataclass(frozen=True, slots=True)
class NullFeedReport:
    """Outcome of the zero-alpha validation. ``passed`` is the verdict;
    the remaining fields let a human audit *why*."""

    n_events: int
    latency_ns: int
    n_fills: int
    tstat_frictionless: float
    tstat_mid_latency: float
    pnl_frictionless: float
    pnl_mid_latency: float
    pnl_post_cost: float
    spread_paid: float
    decomposition_error: float
    tstat_threshold: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lines = [
            f"null-feed validation over {self.n_events:,} events, bounce-free "
            f"control feed (latency {self.latency_ns:,} ns, {self.n_fills:,} fills)",
            f"  pre-cost Sharpe t-stat: {self.tstat_frictionless:+.2f} (frictionless), "
            f"{self.tstat_mid_latency:+.2f} (with latency)  [threshold {self.tstat_threshold}]",
            f"  post-cost PnL: {self.pnl_post_cost:+,.2f}  "
            f"(= mid-fill PnL {self.pnl_mid_latency:+,.2f} - spread paid "
            f"{self.spread_paid:,.2f}; identity error {self.decomposition_error:.2e})",
            f"  verdict: {'PASSED - no leakage detected' if self.passed else 'FAILED'}",
        ]
        lines.extend(f"  FAILURE: {f}" for f in self.failures)
        return "\n".join(lines)


def run_null_feed_validation(
    *,
    n_events: int = 200_000,
    seed: int = 1729,
    latency_ns: int = 2_000_000,
    tstat_threshold: float = 4.0,
) -> NullFeedReport:
    """Run the full zero-alpha protocol on a fresh synthetic feed.

    Returns a report rather than raising: callers embedding this in CI (the
    test suite does) assert ``report.passed`` and get the field-level
    diagnosis for free on failure.
    """
    if latency_ns < 1:
        raise ValueError("validation requires a real latency (>= 1 ns)")

    # Bounce-free control feed: constant spread pins the parity term, so
    # the quoted mid is a true martingale (see module docstring).
    config = LOBConfig(
        n_events=n_events,
        seed=seed,
        mean_spread_ticks=2.0,
        spread_sigma_ticks=0.0,
    )
    book = SyntheticLOBGenerator(config).generate()
    sim = ExecutionSimulator(book)
    target = mean_reversion_probe(sim.mid)

    frictionless = sim.run(target, latency_ns=0, price_mode="mid")
    mid_latency = sim.run(target, latency_ns=latency_ns, price_mode="mid")
    post_cost = sim.run(target, latency_ns=latency_ns, price_mode="cross")

    t_fric = sharpe_tstat(frictionless.ledger.equity_mtm)
    t_lat = sharpe_tstat(mid_latency.ledger.equity_mtm)
    decomp_err = abs(
        post_cost.total_pnl_mtm - (mid_latency.total_pnl_mtm - post_cost.spread_paid)
    )

    failures: list[str] = []
    if abs(t_fric) >= tstat_threshold:
        failures.append(
            f"frictionless run shows fake alpha: |t| = {abs(t_fric):.2f} "
            f">= {tstat_threshold} on a feed with no signal"
        )
    if abs(t_lat) >= tstat_threshold:
        failures.append(
            f"latency-aligned run shows fake alpha: |t| = {abs(t_lat):.2f} "
            f">= {tstat_threshold} - latency alignment is leaking"
        )
    if post_cost.n_fills == 0:
        failures.append("probe generated no fills; validation is vacuous")
    if post_cost.spread_paid <= 0.0:
        failures.append("crossing the book paid no spread; execution pricing is wrong")
    if post_cost.total_pnl_mtm >= 0.0:
        failures.append(
            f"post-cost PnL is {post_cost.total_pnl_mtm:+.2f}: a zero-edge "
            "strategy must lose the spread"
        )
    if decomp_err > 1e-6:
        failures.append(
            f"cost decomposition broken: |pnl_cross - (pnl_mid - spread)| = {decomp_err:.2e}"
        )

    return NullFeedReport(
        n_events=n_events,
        latency_ns=latency_ns,
        n_fills=post_cost.n_fills,
        tstat_frictionless=t_fric,
        tstat_mid_latency=t_lat,
        pnl_frictionless=frictionless.total_pnl_mtm,
        pnl_mid_latency=mid_latency.total_pnl_mtm,
        pnl_post_cost=post_cost.total_pnl_mtm,
        spread_paid=post_cost.spread_paid,
        decomposition_error=decomp_err,
        tstat_threshold=tstat_threshold,
        failures=tuple(failures),
    )
