"""Backtest layer: vectorized execution simulation, accounting, validation.

No event loop anywhere: orders are sparse diffs of a target-position array,
latency is one ``searchsorted``, fills scatter via ``bincount``, and all
portfolio state is prefix sums plus a vectorized LOCF. See ``engine`` for
the execution model, ``accounting`` for the ledger algebra, and
``validation`` for the zero-alpha leakage detector.
"""

from lob_engine.backtest.accounting import Ledger, build_ledger
from lob_engine.backtest.engine import (
    BacktestResult,
    ExecutionSimulator,
    Fills,
    latency_align,
)
from lob_engine.backtest.validation import (
    NullFeedReport,
    mean_reversion_probe,
    run_null_feed_validation,
    sharpe_tstat,
)

__all__ = [
    "ExecutionSimulator",
    "BacktestResult",
    "Fills",
    "latency_align",
    "Ledger",
    "build_ledger",
    "NullFeedReport",
    "mean_reversion_probe",
    "run_null_feed_validation",
    "sharpe_tstat",
]
