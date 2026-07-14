"""Invariant tests for the synthetic LOB generator.

These assert structural properties that must hold for *every* generated feed
(no crossed books, monotone ladders, strictly increasing time), not just
statistical tendencies — the same invariants downstream code relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_engine import LOBConfig, SyntheticLOBGenerator, ar1_process, optimize_memory

N_EVENTS = 20_000
N_LEVELS = 5


@pytest.fixture(scope="module")
def book() -> pd.DataFrame:
    config = LOBConfig(n_events=N_EVENTS, n_levels=N_LEVELS, seed=7)
    return SyntheticLOBGenerator(config).generate()


def test_shape_and_dtypes(book: pd.DataFrame) -> None:
    assert len(book) == N_EVENTS
    assert book["ts_ns"].dtype == np.int64
    for k in range(N_LEVELS):
        assert book[f"bid_px_{k}"].dtype == np.float64
        assert book[f"ask_px_{k}"].dtype == np.float64
        assert book[f"bid_sz_{k}"].dtype == np.uint32
        assert book[f"ask_sz_{k}"].dtype == np.uint32


def test_timestamps_strictly_increasing(book: pd.DataFrame) -> None:
    assert np.all(np.diff(book["ts_ns"].to_numpy()) > 0)


def test_book_never_crossed_or_locked(book: pd.DataFrame) -> None:
    assert np.all(book["ask_px_0"].to_numpy() > book["bid_px_0"].to_numpy())


def test_price_ladders_strictly_monotone(book: pd.DataFrame) -> None:
    bid = book[[f"bid_px_{k}" for k in range(N_LEVELS)]].to_numpy()
    ask = book[[f"ask_px_{k}" for k in range(N_LEVELS)]].to_numpy()
    assert np.all(np.diff(bid, axis=1) < 0), "bid ladder must strictly decrease"
    assert np.all(np.diff(ask, axis=1) > 0), "ask ladder must strictly increase"


def test_all_prices_on_tick_grid(book: pd.DataFrame) -> None:
    tick = LOBConfig().tick_size
    px_cols = [c for c in book.columns if "_px_" in c]
    ticks = book[px_cols].to_numpy() / tick
    np.testing.assert_allclose(ticks, np.round(ticks), atol=1e-6)


def test_sizes_at_least_one(book: pd.DataFrame) -> None:
    sz_cols = [c for c in book.columns if "_sz_" in c]
    assert int(book[sz_cols].to_numpy().min()) >= 1


def test_reproducible_across_instances() -> None:
    config = LOBConfig(n_events=1_000, seed=123)
    frame_a = SyntheticLOBGenerator(config).generate()
    frame_b = SyntheticLOBGenerator(config).generate()
    pd.testing.assert_frame_equal(frame_a, frame_b)


def test_first_event_sits_at_initial_mid() -> None:
    # spread_sigma_ticks=0 pins the AR(1) spread at exactly its mean of 1 tick.
    config = LOBConfig(
        n_events=100, mean_spread_ticks=1.0, spread_sigma_ticks=0.0, seed=0
    )
    book = SyntheticLOBGenerator(config).generate()
    first_mid = 0.5 * (book.at[0, "bid_px_0"] + book.at[0, "ask_px_0"])
    # With a 1-tick spread, bid = mid and ask = mid + tick by construction.
    assert first_mid == pytest.approx(config.initial_mid + config.tick_size / 2)


@pytest.mark.parametrize(
    "bad_config",
    [
        {"n_events": 0},
        {"n_levels": -1},
        {"tick_size": 0.0},
        {"move_probability": 1.5},
        {"mean_spread_ticks": 0.5},
        {"events_per_second": -10.0},
        {"spread_phi": 1.0},
        {"spread_phi": -1.0},
        {"spread_sigma_ticks": -0.1},
    ],
)
def test_config_validation(bad_config: dict) -> None:
    with pytest.raises(ValueError):
        LOBConfig(**bad_config)


class TestAR1Process:
    """The vectorized (lfilter) AR(1) kernel, tested exactly and statistically."""

    def test_exact_recursion_matches_python_loop(self) -> None:
        n, mu, phi, sigma = 4_000, 2.0, 0.9, 0.5
        out = ar1_process(n, mu=mu, phi=phi, sigma=sigma, rng=np.random.default_rng(17))

        # Mirror the documented draw order (n innovations, then the initial
        # deviation) — it is part of the reproducibility contract, and this
        # test is what pins it.
        rng = np.random.default_rng(17)
        eps = rng.normal(0.0, sigma, size=n)
        dev = rng.normal(0.0, sigma / np.sqrt(1.0 - phi * phi))
        expected = np.empty(n)
        for t in range(n):  # the reference implementation is *supposed* to loop
            dev = phi * dev + eps[t]
            expected[t] = mu + dev
        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)

    def test_stationary_moments(self) -> None:
        n, mu, phi, sigma = 400_000, 5.0, 0.8, 1.0
        x = ar1_process(n, mu=mu, phi=phi, sigma=sigma, rng=np.random.default_rng(23))

        target_var = sigma**2 / (1.0 - phi**2)
        assert abs(x.mean() - mu) < 0.05
        assert abs(x.var() / target_var - 1.0) < 0.05
        lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        assert abs(lag1 - phi) < 0.02

    def test_stationary_from_first_sample(self) -> None:
        # If the initial state were 0 instead of a stationary draw, early
        # samples would be under-dispersed: Var(x_0) would be sigma^2, not
        # sigma^2/(1-phi^2) = 2.78 * sigma^2 at phi=0.8. Check the first
        # sample across many independent paths.
        phi, sigma = 0.8, 1.0
        first = np.array(
            [
                ar1_process(1, mu=0.0, phi=phi, sigma=sigma, rng=np.random.default_rng(s))[0]
                for s in range(4_000)
            ]
        )
        target_var = sigma**2 / (1.0 - phi**2)
        assert abs(first.var() / target_var - 1.0) < 0.15

    def test_sigma_zero_is_constant(self) -> None:
        x = ar1_process(100, mu=3.0, phi=0.9, sigma=0.0, rng=np.random.default_rng(1))
        np.testing.assert_allclose(x, 3.0)

    @pytest.mark.parametrize("phi", [1.0, -1.0, 1.5])
    def test_nonstationary_phi_rejected(self, phi: float) -> None:
        with pytest.raises(ValueError):
            ar1_process(10, mu=0.0, phi=phi, sigma=1.0, rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def spreads() -> np.ndarray:
    """Quantized spreads from a feed configured for clean AR(1) statistics.

    A mean of 3 ticks keeps the process well clear of the one-tick floor,
    so quantization is the only distortion of the latent AR(1).
    """
    config = LOBConfig(
        n_events=200_000,
        mean_spread_ticks=3.0,
        spread_phi=0.95,
        spread_sigma_ticks=0.3,
        seed=29,
    )
    book = SyntheticLOBGenerator(config).generate()
    raw = (book["ask_px_0"] - book["bid_px_0"]).to_numpy() / config.tick_size
    return np.rint(raw)


class TestAR1Spread:
    """Statistical properties of the quantized spread in generated books."""

    def test_integer_ticks_at_least_one(self, spreads: np.ndarray) -> None:
        assert spreads.min() >= 1.0

    def test_mean_reverts_to_target(self, spreads: np.ndarray) -> None:
        assert abs(spreads.mean() - 3.0) < 0.15

    def test_persistence_survives_quantization(self, spreads: np.ndarray) -> None:
        # Latent phi = 0.95; rounding noise attenuates the measured lag-1
        # autocorrelation to ~0.87. Anything above 0.6 confirms the spread
        # is genuinely persistent, not i.i.d.
        lag1 = np.corrcoef(spreads[:-1], spreads[1:])[0, 1]
        assert 0.6 < lag1 < 0.99

    def test_mean_reversion_direction(self, spreads: np.ndarray) -> None:
        # Above-mean spreads must predict a *narrowing*: corr(deviation,
        # next change) is negative for any mean-reverting process, and 0
        # for a random walk.
        deviation = spreads[:-1] - spreads.mean()
        change = np.diff(spreads)
        assert np.corrcoef(deviation, change)[0, 1] < -0.05


class TestOptimizeMemory:
    def test_prices_downcast_when_safe(self, book: pd.DataFrame) -> None:
        slim = optimize_memory(book, tick_size=LOBConfig().tick_size)
        assert slim["bid_px_0"].dtype == np.float32
        # Round trip must land within half a tick of the original everywhere.
        err = np.abs(
            slim["bid_px_0"].to_numpy(dtype=np.float64)
            - book["bid_px_0"].to_numpy()
        ).max()
        assert err < LOBConfig().tick_size / 2

    def test_prices_kept_f64_when_unsafe(self) -> None:
        # A ~$5M price with a fine tick cannot survive float32 (~24-bit mantissa).
        frame = pd.DataFrame({"bid_px_0": np.array([5_000_000.01, 5_000_000.02])})
        slim = optimize_memory(frame, tick_size=0.01)
        assert slim["bid_px_0"].dtype == np.float64

    def test_input_frame_untouched(self, book: pd.DataFrame) -> None:
        original_dtypes = book.dtypes.copy()
        optimize_memory(book, tick_size=LOBConfig().tick_size)
        pd.testing.assert_series_equal(book.dtypes, original_dtypes)

    def test_strictly_smaller(self, book: pd.DataFrame) -> None:
        from lob_engine import frame_memory_mb

        slim = optimize_memory(book, tick_size=LOBConfig().tick_size)
        assert frame_memory_mb(slim) < frame_memory_mb(book)
