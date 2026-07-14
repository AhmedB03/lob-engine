"""Synthetic Level-2 limit order book feed generation.

Every stochastic process here (arrival times, mid-price walk, spread, depth)
is drawn as a flat NumPy array and combined via broadcasting, so generating
10 million book snapshots costs a handful of vectorized passes over memory
rather than 10 million interpreter iterations.

Model summary
-------------
* **Arrivals** — Poisson process: exponential inter-arrival gaps, prefix-summed
  into strictly increasing int64 nanosecond timestamps. Integer nanoseconds are
  the canonical clock (exact arithmetic; datetime64 is a presentation format).
* **Mid-price** — tick-constrained random walk: with probability
  ``move_probability`` an event moves the mid by ±1 tick. Prices live on an
  integer tick grid until the final float conversion, so the book can never be
  crossed by floating-point drift.
* **Spread** — stationary AR(1) in ticks, ``S_t = c + phi * S_{t-1} + eps_t``
  with ``|phi| < 1``. The recursion is evaluated loop-free as a first-order
  IIR filter (``scipy.signal.lfilter``; see :func:`ar1_process` for why the
  "obvious" closed-form cumsum is numerically unusable) and then quantized to
  the tick grid with a one-tick floor. Mean-reverting, autocorrelated spreads
  reproduce the tight/wide liquidity episodes that Phase-2 volatility and
  spread-dynamics features key on.
* **Depth** — lognormal size draws per level, scaled by a linear depth profile
  so resting liquidity grows away from the touch, as in real books.

One structural choice is deliberate: the mid-price walk is independent of the
spread, sizes, and arrival processes, so no feature computed from the book can
predict the next mid move. The feed is a known-zero-alpha null case for the
backtester (Phase 3) — the spread is autocorrelated (realistic volatility/
liquidity clustering for Phase-2 features to measure), but there is still
nothing to *trade*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import lfilter

_NS_PER_SECOND = 1_000_000_000


def ar1_process(
    n: int,
    *,
    mu: float,
    phi: float,
    sigma: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Sample a stationary AR(1) path with no Python-level looping.

    The process::

        x_t = mu + d_t,    d_t = phi * d_{t-1} + eps_t,    eps_t ~ N(0, sigma^2)

    Parameters
    ----------
    n
        Number of samples.
    mu
        Stationary mean (``mu = c / (1 - phi)`` in the ``c + phi*S`` form).
    phi
        Persistence; must satisfy ``|phi| < 1`` for stationarity.
    sigma
        Innovation standard deviation.
    rng
        Source of randomness. Draw order (``n`` innovations, then one
        initial deviation) is part of the reproducibility contract.

    Vectorization note
    ------------------
    The recursion carries a serial dependency, so it is not a ``cumsum``.
    The closed form ``d_t = phi**t * (d_0 + sum_i eps_i * phi**-i)`` *looks*
    vectorizable, but ``phi**-i`` overflows float64 after roughly
    ``log(1.8e308) / log(1/phi)`` steps — ~14k samples at ``phi = 0.95`` —
    a classic numerically-unusable "vectorization". Instead, treat the
    recursion as what it is: a first-order IIR filter with transfer function
    ``1 / (1 - phi * z^-1)``, and evaluate it with ``scipy.signal.lfilter`` —
    a compiled O(n) pass over contiguous memory, loop-free at the Python
    level and exact at the numerical level.

    The initial deviation ``d_{-1}`` is drawn from the stationary
    distribution ``N(0, sigma^2 / (1 - phi^2))``, so the path is stationary
    from the very first sample — no burn-in is generated or discarded.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not -1.0 < phi < 1.0:
        raise ValueError(f"AR(1) stationarity requires |phi| < 1, got {phi}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    eps = rng.normal(0.0, sigma, size=n)
    stationary_std = sigma / np.sqrt(1.0 - phi * phi)
    d_initial = rng.normal(0.0, stationary_std)
    # lfilter state for y[t] = eps[t] + phi*y[t-1] is zi = [phi * y[-1]].
    deviations, _ = lfilter([1.0], [1.0, -phi], eps, zi=np.array([phi * d_initial]))
    return mu + deviations


@dataclass(frozen=True, slots=True)
class LOBConfig:
    """Parameters for the synthetic feed.

    Attributes
    ----------
    n_events
        Number of book snapshots (ticks) to generate.
    n_levels
        Book depth per side (Level-2 price levels).
    initial_mid
        Starting mid price, in currency units.
    tick_size
        Minimum price increment. All prices are integer multiples of this.
    events_per_second
        Mean Poisson arrival rate of book updates.
    move_probability
        Per-event probability that the mid moves by one tick.
    mean_spread_ticks
        Stationary mean of the AR(1) spread process, in ticks (must be >= 1).
    spread_phi
        AR(1) persistence of the spread; ``|spread_phi| < 1``. Higher values
        produce longer tight/wide liquidity episodes.
    spread_sigma_ticks
        Innovation standard deviation of the spread process, in ticks.
    median_touch_size
        Median resting size at the best bid/ask (lognormal median).
    size_sigma
        Log-space standard deviation of size draws (depth heterogeneity).
    depth_growth
        Linear growth of median size per level away from the touch.
    start_time
        Timestamp of the first event.
    seed
        PRNG seed. Identical configs produce identical feeds.
    """

    n_events: int = 1_000_000
    n_levels: int = 5
    initial_mid: float = 100.00
    tick_size: float = 0.01
    events_per_second: float = 500.0
    move_probability: float = 0.15
    mean_spread_ticks: float = 1.5
    spread_phi: float = 0.95
    spread_sigma_ticks: float = 0.25
    median_touch_size: float = 400.0
    size_sigma: float = 0.8
    depth_growth: float = 0.35
    start_time: np.datetime64 = np.datetime64("2026-01-05T09:30:00", "ns")
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_events <= 0:
            raise ValueError(f"n_events must be positive, got {self.n_events}")
        if self.n_levels <= 0:
            raise ValueError(f"n_levels must be positive, got {self.n_levels}")
        if self.tick_size <= 0.0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if self.events_per_second <= 0.0:
            raise ValueError(
                f"events_per_second must be positive, got {self.events_per_second}"
            )
        if not 0.0 <= self.move_probability <= 1.0:
            raise ValueError(
                f"move_probability must lie in [0, 1], got {self.move_probability}"
            )
        if self.mean_spread_ticks < 1.0:
            raise ValueError(
                f"mean_spread_ticks must be >= 1 tick, got {self.mean_spread_ticks}"
            )
        if not -1.0 < self.spread_phi < 1.0:
            raise ValueError(
                f"spread_phi must lie in (-1, 1) for stationarity, "
                f"got {self.spread_phi}"
            )
        if self.spread_sigma_ticks < 0.0:
            raise ValueError(
                f"spread_sigma_ticks must be non-negative, "
                f"got {self.spread_sigma_ticks}"
            )
        if self.median_touch_size < 1.0:
            raise ValueError(
                f"median_touch_size must be >= 1, got {self.median_touch_size}"
            )
        if self.initial_mid < self.tick_size:
            raise ValueError("initial_mid must be at least one tick")


class SyntheticLOBGenerator:
    """Vectorized generator of synthetic L2 book snapshots.

    Produces a DataFrame with one row per book update:

    ==========  =======  ==================================================
    column      dtype    meaning
    ==========  =======  ==================================================
    ts_ns       int64    event timestamp, nanoseconds since epoch
    bid_px_{k}  float64  bid price at level k (0 = best)
    bid_sz_{k}  uint32   resting bid size at level k
    ask_px_{k}  float64  ask price at level k (0 = best)
    ask_sz_{k}  uint32   resting ask size at level k
    ==========  =======  ==================================================

    Invariants guaranteed by construction: timestamps strictly increase, the
    book is never crossed or locked (``ask_px_0 > bid_px_0``), price ladders
    are strictly monotone one tick apart, and all sizes are >= 1.
    """

    def __init__(self, config: LOBConfig | None = None) -> None:
        self._cfg = config if config is not None else LOBConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    @property
    def config(self) -> LOBConfig:
        return self._cfg

    def generate(self) -> pd.DataFrame:
        """Generate the full feed in O(n_events * n_levels) vectorized work."""
        cfg = self._cfg
        n, levels = cfg.n_events, cfg.n_levels

        ts_ns = self._arrival_times(n)
        mid_ticks = self._mid_price_walk(n)
        spread_ticks = self._spreads(n)

        # Split the spread around the mid on the integer tick grid. The exact
        # identity ask - bid == spread holds even for odd spreads.
        half = spread_ticks // 2
        bid_ticks = mid_ticks - half
        ask_ticks = bid_ticks + spread_ticks

        deepest_bid = int(bid_ticks.min()) - (levels - 1)
        if deepest_bid < 1:
            raise RuntimeError(
                "Mid-price walk drifted too close to zero for the configured "
                "initial_mid; raise initial_mid or lower move_probability."
            )

        # Dense ladder: level k sits k ticks behind the touch. Broadcasting an
        # (n,) column against an (L,) row yields the (n, L) price matrix in one
        # pass — no per-event work in Python.
        offsets = np.arange(levels, dtype=np.int64)
        bid_px = (bid_ticks[:, None] - offsets) * cfg.tick_size
        ask_px = (ask_ticks[:, None] + offsets) * cfg.tick_size

        bid_sz = self._depth_sizes(n, levels)
        ask_sz = self._depth_sizes(n, levels)

        return self._assemble(ts_ns, bid_px, bid_sz, ask_px, ask_sz)

    # ------------------------------------------------------------------
    # Component processes
    # ------------------------------------------------------------------

    def _arrival_times(self, n: int) -> npt.NDArray[np.int64]:
        """Poisson arrivals as strictly increasing int64 nanosecond stamps."""
        mean_gap_ns = _NS_PER_SECOND / self._cfg.events_per_second
        gaps = self._rng.exponential(mean_gap_ns, size=n)
        # Floor gaps at 1 ns so truncation to integers cannot create ties.
        gaps = np.maximum(gaps, 1.0)
        start_ns = self._cfg.start_time.astype("datetime64[ns]").astype(np.int64)
        return start_ns + np.cumsum(gaps).astype(np.int64)

    def _mid_price_walk(self, n: int) -> npt.NDArray[np.int64]:
        """Tick-grid random walk: cumulative sum of {-1, 0, +1} steps."""
        cfg = self._cfg
        initial_ticks = round(cfg.initial_mid / cfg.tick_size)
        moves = self._rng.random(n) < cfg.move_probability
        signs = self._rng.integers(0, 2, size=n, dtype=np.int64) * 2 - 1
        steps = np.where(moves, signs, 0)
        steps[0] = 0  # first snapshot sits exactly at initial_mid
        return initial_ticks + np.cumsum(steps)

    def _spreads(self, n: int) -> npt.NDArray[np.int64]:
        """Stationary AR(1) spread, quantized to the tick grid.

        The *latent* process is exactly AR(1) (see :func:`ar1_process`); the
        observed spread is its ``rint`` image with a one-tick floor, since
        quoted spreads live on the tick grid and a locked/crossed book is
        not a valid state. Quantization mildly attenuates the measured
        autocorrelation (~0.87 observed vs 0.95 latent at defaults) and the
        floor biases the observed mean slightly above ``mean_spread_ticks``
        when the process is tight against the one-tick boundary — both are
        documented, tested properties rather than surprises.
        """
        cfg = self._cfg
        latent = ar1_process(
            n,
            mu=cfg.mean_spread_ticks,
            phi=cfg.spread_phi,
            sigma=cfg.spread_sigma_ticks,
            rng=self._rng,
        )
        return np.maximum(np.rint(latent), 1.0).astype(np.int64)

    def _depth_sizes(self, n: int, levels: int) -> npt.NDArray[np.uint32]:
        """Lognormal sizes with liquidity growing away from the touch."""
        cfg = self._cfg
        profile = 1.0 + cfg.depth_growth * np.arange(levels, dtype=np.float64)
        raw = self._rng.lognormal(
            mean=np.log(cfg.median_touch_size), sigma=cfg.size_sigma, size=(n, levels)
        )
        return np.maximum(raw * profile, 1.0).astype(np.uint32)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble(
        ts_ns: npt.NDArray[np.int64],
        bid_px: npt.NDArray[np.float64],
        bid_sz: npt.NDArray[np.uint32],
        ask_px: npt.NDArray[np.float64],
        ask_sz: npt.NDArray[np.uint32],
    ) -> pd.DataFrame:
        # This loop runs n_levels times (a constant ~5), not n_events times —
        # the vectorization discipline applies to the event axis only.
        data: dict[str, np.ndarray] = {"ts_ns": ts_ns}
        for k in range(bid_px.shape[1]):
            data[f"bid_px_{k}"] = bid_px[:, k]
            data[f"bid_sz_{k}"] = bid_sz[:, k]
            data[f"ask_px_{k}"] = ask_px[:, k]
            data[f"ask_sz_{k}"] = ask_sz[:, k]
        return pd.DataFrame(data)
