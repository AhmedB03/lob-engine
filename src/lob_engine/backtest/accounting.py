"""Vectorized position ledger: portfolio state as cumulative array algebra.

An accounting loop carries three state variables event to event: position,
cash, and the entry price of the open position. All three vectorize:

* **Position & cash** — fills scatter onto the event axis with
  ``np.bincount`` (a compiled scatter-add that handles multiple fills per
  event natively), then ``np.cumsum`` turns per-event flow into state:
  ``X_j = sum of signed qty filled at events <= j``, ``C_j`` likewise from
  ``-qty * price``.
* **Entry price** — a last-observation-carried-forward (LOCF) problem, and
  LOCF has a classic loop-free form: write each fill event's index where a
  new entry was set (else -1), take ``np.maximum.accumulate``, and gather.
  The running maximum *is* the "most recent event that set state" — a
  vectorized state machine.
* **Everything else is derived**, not tracked: mark-to-market equity is
  ``cash + position * mid``; mark-to-liquidation replaces the mid with the
  price you would actually get crossing the book *now* (sell at bid if
  long, buy back at ask if short) — so ``equity_liq <= equity_mtm``
  whenever a position is open, with the gap equal to the exit half-spread.

Why positions are restricted to {-1, 0, +1}
--------------------------------------------
Unit inventory makes the entry price a pure LOCF: every fill that leaves a
nonzero position (0 -> +-1, or a +-2 flip) *is* a fresh entry at that fill's
price. General inventory needs average-cost tracking,
``avg_t = f(avg_{t-1}, ...)`` — a genuine serial recursion (data-dependent,
unlike the AR(1) filter) that does not vectorize exactly. Scoping Phase 3 to
unit inventory keeps every number exact and loop-free; the restriction is
enforced, not assumed.

Realized/unrealized split (exact, by construction)::

    unrealized_j = X_j * (mid_j - entry_j)      (0 when flat)
    realized_j   = equity_mtm_j - unrealized_j  ( = cash_j + X_j * entry_j)

When flat, realized equals cash — every round trip's PnL, including the
spread it paid, is locked in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Ledger:
    """Per-event portfolio state; every field is a length-n_events array.

    Attributes
    ----------
    position
        Net position after all fills at each event; values in {-1, 0, +1}.
    cash
        Cumulative cash: buys subtract ``qty * price``, sells add it.
    equity_mtm
        ``cash + position * mid`` — mark-to-market at every tick.
    equity_liq
        ``cash + position * exit_price`` — mark-to-liquidation, where the
        exit price crosses the book (bid if long, ask if short).
    entry_price
        Fill price that opened the current position; NaN while flat.
    realized_pnl
        PnL locked by completed (or flipped-out-of) positions.
    unrealized_pnl
        Open-position PnL at the mid; realized + unrealized == equity_mtm.
    """

    position: _I64
    cash: _F64
    equity_mtm: _F64
    equity_liq: _F64
    entry_price: _F64
    realized_pnl: _F64
    unrealized_pnl: _F64


def build_ledger(
    *,
    n_events: int,
    exec_idx: _I64,
    qty: _I64,
    price: _F64,
    bid: _F64,
    ask: _F64,
    mid: _F64,
) -> Ledger:
    """Construct the full ledger from executed fills, loop-free.

    Parameters
    ----------
    n_events
        Length of the event axis.
    exec_idx, qty, price
        Parallel fill arrays (see ``engine.Fills``); ``exec_idx`` must lie
        in ``[0, n_events)``. Fills at the same event are netted for
        position/cash; the *last* fill at an event defines the entry-price
        candidate, matching sequential-application semantics.
    bid, ask, mid
        Touch prices per event, for marking.

    Raises
    ------
    ValueError
        If fills are malformed or the implied position ever leaves
        {-1, 0, +1} — accounting refuses to produce numbers whose
        exactness it cannot guarantee.
    """
    if not (exec_idx.shape == qty.shape == price.shape):
        raise ValueError("exec_idx, qty and price must be parallel arrays")
    if exec_idx.size and (exec_idx.min() < 0 or exec_idx.max() >= n_events):
        raise ValueError("exec_idx out of range: unfilled orders must be dropped upstream")

    # Entry-price bookkeeping needs fills in execution order; the engine
    # already produces them sorted (monotone latency map), so this argsort
    # only reorders under injected/adversarial exec indices.
    if exec_idx.size and np.any(np.diff(exec_idx) < 0):
        order = np.argsort(exec_idx, kind="stable")
        exec_idx, qty, price = exec_idx[order], qty[order], price[order]

    # --- position & cash: scatter-add then prefix sum ------------------
    qty_f = qty.astype(np.float64)
    flow_qty = np.bincount(exec_idx, weights=qty_f, minlength=n_events)
    flow_cash = np.bincount(exec_idx, weights=-qty_f * price, minlength=n_events)
    position_f = np.cumsum(flow_qty)
    cash = np.cumsum(flow_cash)

    position = np.rint(position_f).astype(np.int64)
    if np.any(np.abs(position) > 1):
        raise ValueError(
            "implied position left {-1, 0, +1}; unit-inventory accounting "
            "cannot represent this fill stream exactly"
        )

    # --- entry price: LOCF via maximum.accumulate -----------------------
    # Last fill price per event: with duplicate indices, NumPy fancy
    # assignment keeps the final write — exactly "last fill wins".
    last_fill_px = np.zeros(n_events)
    has_fill = np.zeros(n_events, dtype=bool)
    last_fill_px[exec_idx] = price
    has_fill[exec_idx] = True

    # Any fill event with a nonzero post-position is a fresh entry (unit
    # inventory: 0 -> +-1 opens, +-2 flips re-enter at the same fill).
    event_idx = np.arange(n_events, dtype=np.int64)
    setter = np.where(has_fill, event_idx, -1)
    setter = np.maximum.accumulate(setter)  # most recent fill event, per event
    entry_raw = np.where(setter >= 0, last_fill_px[np.maximum(setter, 0)], 0.0)

    open_pos = position != 0
    unrealized_pnl = np.where(open_pos, position * (mid - entry_raw), 0.0)
    equity_mtm = cash + position * mid
    exit_px = np.where(position > 0, bid, ask)
    equity_liq = cash + position * exit_px
    realized_pnl = equity_mtm - unrealized_pnl
    entry_price = np.where(open_pos, entry_raw, np.nan)

    return Ledger(
        position=position,
        cash=cash,
        equity_mtm=equity_mtm,
        equity_liq=equity_liq,
        entry_price=entry_price,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
    )
