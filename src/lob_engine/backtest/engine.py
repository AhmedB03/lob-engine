"""Vectorized execution simulation: latency alignment and spread-crossing fills.

The classic backtester is an event loop: walk the feed, maintain pending
orders, fill them when their time comes, mutate inventory. Correct, and
unusably slow in Python. This engine expresses the identical semantics as
four array passes, none of which iterates the event axis in the interpreter:

1. **Orders are sparse diffs.** A strategy emits a *target position*
   ``x_t in {-1, 0, +1}`` per event (computed point-in-time). Orders exist
   exactly where the target changes: ``delta = diff(target, prepend=0)``,
   nonzeros only. Everything downstream operates on this (much smaller)
   order set.
2. **Latency is one binary search.** An order born at event t (stamp
   ``T_t``) may first execute at the first event k with
   ``T_k >= T_t + latency_ns`` — which is ``np.searchsorted(ts, ts +
   latency, side="left")``, the same boundary idiom as the Phase-2
   volatility windows, evaluated for every event simultaneously. A result
   of n means the feed ended before the order could fill; such orders are
   dropped, never force-filled.
3. **Fills are gathers.** A buy crosses the book at ``ask_px_0[k]``, a sell
   at ``bid_px_0[k]`` — transaction costs are *implicit*: you pay the
   quantized AR(1) spread prevailing at execution time, not a bolted-on fee.
4. **State is prefix sums.** Position and cash paths are ``bincount`` +
   ``cumsum`` (see ``accounting``); no inventory variable is ever carried
   across a Python iteration.

Causality guarantees (load-bearing, tested)
-------------------------------------------
For ``latency_ns >= 1``, ``exec_idx[t] > t`` holds *even inside nanosecond
bursts*: every event sharing ``T_t`` fails ``T_k >= T_t + 1``, so an order
can never fill at, or before, the event whose information created it.
Because timestamps are sorted and latency is constant, ``exec_idx`` is
non-decreasing — fills land in signal order without any sorting step.
``latency_ns = 0`` is allowed as an explicitly *idealized* control (fill at
the signal event itself); it exists so experiments can isolate spread cost
from latency cost, and the null-feed validation exercises it.

The ``exec_idx`` override on :meth:`ExecutionSimulator.run` is the testing
hook that lets the leak-detection suite *inject* a look-ahead (executing
before the signal event) and prove the detector fires. Production callers
never pass it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from lob_engine.backtest.accounting import Ledger, build_ledger

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

_PRICE_MODES = ("cross", "mid")


def latency_align(ts_ns: npt.ArrayLike, latency_ns: int) -> _I64:
    """Map every event index t to its earliest execution index k.

    ``k = min { j : ts[j] >= ts[t] + latency_ns }``, or ``n`` if no such
    event exists (order outlives the feed). One vectorized binary search
    answers all n queries; monotonicity of both arrays makes the result
    non-decreasing, which downstream accounting relies on.
    """
    stamps = np.asarray(ts_ns, dtype=np.int64)
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError(f"need a 1-d non-empty timestamp array, got shape {stamps.shape}")
    if np.any(np.diff(stamps) < 0):
        raise ValueError("ts_ns must be non-decreasing")
    if latency_ns < 0:
        raise ValueError(f"latency_ns must be >= 0, got {latency_ns}")
    if latency_ns == 0:
        # Idealized control: execution at the signal event itself.
        return np.arange(stamps.size, dtype=np.int64)
    exec_idx = np.searchsorted(stamps, stamps + latency_ns, side="left")
    return exec_idx.astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class Fills:
    """Executed orders, in signal order (and therefore execution order).

    Attributes
    ----------
    signal_idx
        Event index whose information generated the order.
    exec_idx
        Event index at which the order filled (strictly < n_events).
    qty
        Signed size: +1 open long / close short leg, +-2 for a flip.
    price
        Fill price: ask for buys, bid for sells ("cross" mode), or the
        contemporaneous mid ("mid" mode, frictionless control).
    """

    signal_idx: _I64
    exec_idx: _I64
    qty: _I64
    price: _F64

    def __len__(self) -> int:
        return self.qty.size


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """A completed simulation: fills, full ledger, and cost attribution."""

    fills: Fills
    ledger: Ledger
    latency_ns: int
    price_mode: str
    spread_paid: float  # sum of qty * (fill price - mid at exec); >= 0 in cross mode

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    @property
    def total_pnl_mtm(self) -> float:
        """Final mark-to-market equity (starting cash = 0)."""
        return float(self.ledger.equity_mtm[-1])

    @property
    def total_pnl_liq(self) -> float:
        """Final mark-to-liquidation equity: PnL if the terminal position
        were closed instantly by crossing the book."""
        return float(self.ledger.equity_liq[-1])

    @property
    def final_position(self) -> int:
        return int(self.ledger.position[-1])


class ExecutionSimulator:
    """Latency-aware, spread-crossing execution over a canonical L2 frame.

    The constructor extracts the touch and timestamps once (the only pandas
    contact); :meth:`run` is then pure NumPy and can sweep many strategies
    and latencies over the same feed without re-extraction — the same
    "expensive part computed once, reused" pattern as the Phase-2 window
    bounds.
    """

    def __init__(self, book: pd.DataFrame) -> None:
        for col in ("ts_ns", "bid_px_0", "ask_px_0"):
            if col not in book.columns:
                raise ValueError(f"book frame is missing required column '{col}'")
        self._ts = book["ts_ns"].to_numpy(dtype=np.int64)
        if np.any(np.diff(self._ts) < 0):
            raise ValueError("book ts_ns must be non-decreasing")
        self._bid = book["bid_px_0"].to_numpy(dtype=np.float64)
        self._ask = book["ask_px_0"].to_numpy(dtype=np.float64)
        self._mid = 0.5 * (self._bid + self._ask)

    @property
    def n_events(self) -> int:
        return self._ts.size

    @property
    def mid(self) -> _F64:
        """Contemporaneous mid; strategies may build signals from it."""
        return self._mid

    def run(
        self,
        target_position: npt.ArrayLike,
        *,
        latency_ns: int,
        price_mode: str = "cross",
        exec_idx: npt.NDArray[np.int64] | None = None,
    ) -> BacktestResult:
        """Simulate a target-position strategy under latency and spread cost.

        Parameters
        ----------
        target_position
            Desired position per event, values in {-1, 0, +1}, computed
            from information available at that event.
        latency_ns
            Signal-to-execution delay. 0 is the idealized instant-fill
            control (see module docstring).
        price_mode
            "cross": buys lift the ask, sells hit the bid (costs implicit
            in the spread). "mid": fills at the contemporaneous mid — the
            frictionless twin used by null experiments; ``spread_paid`` is
            exactly 0.
        exec_idx
            TESTING HOOK ONLY: a precomputed per-event execution index that
            bypasses ``latency_align``. The leak-detection tests use it to
            inject deliberate look-ahead and prove the detector fires.

        Returns
        -------
        BacktestResult
        """
        n = self.n_events
        if price_mode not in _PRICE_MODES:
            raise ValueError(f"price_mode must be one of {_PRICE_MODES}, got {price_mode!r}")
        target = np.asarray(target_position)
        if target.shape != (n,):
            raise ValueError(
                f"target_position must have shape ({n},), got {target.shape}"
            )
        if not np.isin(target, (-1, 0, 1)).all():
            raise ValueError("target_position values must lie in {-1, 0, +1}")
        target = target.astype(np.int64, copy=False)

        if exec_idx is None:
            exec_map = latency_align(self._ts, latency_ns)
        else:
            exec_map = np.asarray(exec_idx, dtype=np.int64)
            if exec_map.shape != (n,):
                raise ValueError(
                    f"exec_idx override must have shape ({n},), got {exec_map.shape}"
                )

        # Orders: the sparse set of target changes (delta != 0).
        deltas = np.diff(target, prepend=0)
        signal_idx = np.flatnonzero(deltas)
        k = exec_map[signal_idx]
        filled = k < n  # orders the feed outlives are dropped, never forced
        signal_idx = signal_idx[filled]
        k = k[filled]
        qty = deltas[signal_idx]

        if price_mode == "cross":
            price = np.where(qty > 0, self._ask[k], self._bid[k])
        else:
            price = self._mid[k]
        spread_paid = float(np.sum(qty * (price - self._mid[k])))

        fills = Fills(signal_idx=signal_idx, exec_idx=k, qty=qty, price=price)
        ledger = build_ledger(
            n_events=n,
            exec_idx=k,
            qty=qty,
            price=price,
            bid=self._bid,
            ask=self._ask,
            mid=self._mid,
        )
        return BacktestResult(
            fills=fills,
            ledger=ledger,
            latency_ns=latency_ns,
            price_mode=price_mode,
            spread_paid=spread_paid,
        )
