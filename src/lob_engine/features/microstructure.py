"""Vectorized market microstructure features over an L2 book frame.

Design: **Pandas at the edges, NumPy in the core.** The constructor extracts
the ``(n_events, n_levels)`` price/size matrices from the DataFrame exactly
once (one contiguous copy per matrix); every feature method is then a handful
of compiled array operations over those matrices, returning bare ndarrays.
``compute_all`` re-wraps results into a labeled DataFrame at the boundary.
This keeps the hot path free of per-call column lookups, index alignment, and
block-manager overhead — the usual hidden costs of "using Pandas" naively.

Signals
-------
Order Book Imbalance (depth-``d``, level-decay ``lambda``)::

    OBI = (B - A) / (B + A),   B = sum_k w_k * bid_sz_k,   w_k = exp(-lambda*k)

OBI in [-1, +1] measures net resting buy pressure. The weighted depth sums
are computed as a single matrix-vector product (BLAS-backed, cache-friendly).

Micro-price (Gatheral/Stoikov)::

    P_micro = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)

Note the *crossed* weighting: a heavy bid queue pushes the micro-price toward
the ask, because the thin side of the book is the one likely to be consumed
first — the micro-price is the imbalance-adjusted expectation of where the
mid is headed, and ``micro - mid`` is the tradeable signal we backtest later.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

_F64 = npt.NDArray[np.float64]


class MicrostructureFeatures:
    """Signal computation over a canonical L2 frame (see ``data.generator``).

    Parameters
    ----------
    book
        DataFrame with ``{bid,ask}_px_{k}`` and ``{bid,ask}_sz_{k}`` columns
        for ``k = 0 .. n_levels-1``. Depth is inferred from the columns.

    Raises
    ------
    ValueError
        If the frame is missing any of the four column groups at any level.
    """

    def __init__(self, book: pd.DataFrame) -> None:
        self._n_levels = self._infer_n_levels(book.columns)
        self._index = book.index
        cols = [
            (f"{side}_{field}_{k}")
            for side in ("bid", "ask")
            for field in ("px", "sz")
            for k in range(self._n_levels)
        ]
        missing = [c for c in cols if c not in book.columns]
        if missing:
            raise ValueError(f"book frame is missing columns: {missing}")

        def matrix(side: str, field: str) -> _F64:
            names = [f"{side}_{field}_{k}" for k in range(self._n_levels)]
            return book[names].to_numpy(dtype=np.float64)

        # One-time conversion to float64: exact for uint32 sizes, and it keeps
        # every downstream reduction/division in a single fast dtype.
        self._bid_px = matrix("bid", "px")
        self._ask_px = matrix("ask", "px")
        self._bid_sz = matrix("bid", "sz")
        self._ask_sz = matrix("ask", "sz")

    @staticmethod
    def _infer_n_levels(columns: pd.Index) -> int:
        levels = sum(1 for c in columns if c.startswith("bid_px_"))
        if levels == 0:
            raise ValueError(
                "no 'bid_px_{k}' columns found; not a canonical LOB frame"
            )
        return levels

    @property
    def n_levels(self) -> int:
        return self._n_levels

    # ------------------------------------------------------------------
    # Core quotes
    # ------------------------------------------------------------------

    def mid_price(self) -> _F64:
        """Arithmetic mid of the best bid and ask."""
        return 0.5 * (self._bid_px[:, 0] + self._ask_px[:, 0])

    def spread(self) -> _F64:
        """Quoted bid-ask spread in price units."""
        return self._ask_px[:, 0] - self._bid_px[:, 0]

    def spread_bps(self) -> _F64:
        """Quoted spread in basis points of the mid."""
        mid = self.mid_price()
        return np.divide(
            1e4 * self.spread(), mid, out=np.zeros_like(mid), where=mid > 0
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def order_book_imbalance(self, depth: int = 1, *, decay: float = 0.0) -> _F64:
        """Depth-weighted order book imbalance in [-1, +1].

        Parameters
        ----------
        depth
            Number of levels per side to aggregate (1 = touch only).
        decay
            Exponential level-decay rate ``lambda``; level ``k`` receives
            weight ``exp(-lambda * k)``. ``0.0`` weights all levels equally.

        Notes
        -----
        The weighted depth on each side is an ``(n, d) @ (d,)`` mat-vec — one
        BLAS call over contiguous memory instead of ``d`` separate scaled
        column reductions.
        """
        if not 1 <= depth <= self._n_levels:
            raise ValueError(
                f"depth must lie in [1, {self._n_levels}], got {depth}"
            )
        weights = np.exp(-decay * np.arange(depth, dtype=np.float64))
        bid_depth = self._bid_sz[:, :depth] @ weights
        ask_depth = self._ask_sz[:, :depth] @ weights
        total = bid_depth + ask_depth
        return np.divide(
            bid_depth - ask_depth, total, out=np.zeros_like(total), where=total > 0
        )

    def micro_price(self) -> _F64:
        """Size-weighted micro-price at the touch; falls back to mid where
        both touch queues are empty (cannot occur on generated data, but the
        engine must not emit NaN/inf on arbitrary real feeds)."""
        bid_px, ask_px = self._bid_px[:, 0], self._ask_px[:, 0]
        bid_sz, ask_sz = self._bid_sz[:, 0], self._ask_sz[:, 0]
        total = bid_sz + ask_sz
        return np.divide(
            bid_px * ask_sz + ask_px * bid_sz,
            total,
            out=self.mid_price(),
            where=total > 0,
        )

    # ------------------------------------------------------------------
    # Boundary: labeled output
    # ------------------------------------------------------------------

    def compute_all(
        self, *, obi_depth: int | None = None, obi_decay: float = 0.5
    ) -> pd.DataFrame:
        """Compute the full Phase-1 feature set as a labeled DataFrame.

        Parameters
        ----------
        obi_depth
            Depth for the weighted OBI column; defaults to the full book.
        obi_decay
            Level-decay rate for the weighted OBI column.

        Returns
        -------
        pd.DataFrame
            Indexed like the input book, with columns ``mid``, ``spread``,
            ``spread_bps``, ``obi_l1``, ``obi_l{d}_w``, ``micro_price`` and
            ``micro_dev`` (micro-price minus mid: the directional signal).
        """
        depth = self._n_levels if obi_depth is None else obi_depth
        mid = self.mid_price()
        micro = self.micro_price()
        return pd.DataFrame(
            {
                "mid": mid,
                "spread": self.spread(),
                "spread_bps": self.spread_bps(),
                "obi_l1": self.order_book_imbalance(1),
                f"obi_l{depth}_w": self.order_book_imbalance(depth, decay=obi_decay),
                "micro_price": micro,
                "micro_dev": micro - mid,
            },
            index=self._index,
        )
