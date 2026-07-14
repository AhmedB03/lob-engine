"""lob_engine: high-performance, vectorized limit order book analytics.

A research-grade engine for simulating Level-2 market data feeds, extracting
market microstructure signals, and backtesting them — with the hard constraint
that no O(n_events) work is ever done in the Python interpreter loop. All
event-axis processing is delegated to NumPy's compiled kernels.
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
from lob_engine.data.generator import LOBConfig, SyntheticLOBGenerator, ar1_process
from lob_engine.data.schema import frame_memory_mb, optimize_memory
from lob_engine.features.microstructure import MicrostructureFeatures
from lob_engine.features.volatility import (
    log_returns,
    realized_vol_events,
    realized_vol_wallclock,
    wallclock_window_starts,
)

__version__ = "0.3.0"

__all__ = [
    "LOBConfig",
    "SyntheticLOBGenerator",
    "MicrostructureFeatures",
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
    "ar1_process",
    "log_returns",
    "realized_vol_events",
    "realized_vol_wallclock",
    "wallclock_window_starts",
    "optimize_memory",
    "frame_memory_mb",
    "__version__",
]
