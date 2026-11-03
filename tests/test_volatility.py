"""Volatility engine tests.

The core methodology: every vectorized kernel is checked for *exact*
mathematical equivalence against a deliberately naive O(n * window) Python
loop that encodes the definition directly. The loops here are the spec; the
vectorized code is the implementation. Edge cases target the two failure
modes irregular tick data actually produces: nanosecond bursts (duplicate
timestamps) and structural silences (gaps wider than the window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_engine import (
    LOBConfig,
    MicrostructureFeatures,
    SyntheticLOBGenerator,
    log_returns,
    realized_vol_events,
    realized_vol_wallclock,
    wallclock_window_starts,
)

# ----------------------------------------------------------------------
# Brute-force reference implementations (the definition, written naively)
# ----------------------------------------------------------------------


def _vol_of(window_returns: np.ndarray, demean: bool) -> float:
    r = window_returns
    if demean:
        r = r - r.mean()
    return float(np.sqrt(np.sum(r * r)))


def _brute_events(prices: np.ndarray, window: int, demean: bool) -> np.ndarray:
    r = np.diff(np.log(prices))
    out = np.full(prices.size, np.nan)
    for i in range(window, prices.size):
        out[i] = _vol_of(r[i - window : i], demean)
    return out


def _brute_wallclock(
    prices: np.ndarray,
    ts_ns: np.ndarray,
    horizon_ns: int,
    min_periods: int,
    demean: bool,
) -> np.ndarray:
    r = np.diff(np.log(prices))
    arrivals = ts_ns[1:]
    out = np.full(prices.size, np.nan)
    for i in range(prices.size):
        # Returns observable at event i: index j < i (event order), arrival
        # inside the half-open window (t_i - horizon, t_i].
        in_window = [
            r[j] for j in range(i) if arrivals[j] > ts_ns[i] - horizon_ns
        ]
        if len(in_window) >= min_periods:
            out[i] = _vol_of(np.asarray(in_window), demean)
    return out


# ----------------------------------------------------------------------
# Shared irregular test data: bursts, silences, and everything between
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def irregular_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(37)
    n = 400
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, size=n)))
    # Gap menu spanning three regimes: 0 ns (nanosecond bursts), a few ns
    # (normal flow), and 5000 ns (silences far wider than small horizons).
    gaps = rng.choice([0, 0, 0, 1, 2, 7, 5_000], size=n)
    gaps[0] = 0
    ts_ns = 1_000_000 + np.cumsum(gaps).astype(np.int64)
    return prices, ts_ns


class TestEventTimeVol:
    @pytest.mark.parametrize("window", [1, 2, 5, 50])
    @pytest.mark.parametrize("demean", [False, True])
    def test_matches_brute_force(self, irregular_data, window: int, demean: bool):
        prices, _ = irregular_data
        out = realized_vol_events(prices, window, demean=demean)
        expected = _brute_events(prices, window, demean)
        # atol covers exactly-zero demeaned windows, where prefix-sum
        # differencing leaves a ~1e-11 residue that the direct loop does not
        # (signal scale here is ~1e-4, so 1e-9 is still 5 decades of margin).
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)

    def test_nan_head_then_finite(self, irregular_data):
        prices, _ = irregular_data
        out = realized_vol_events(prices, window=10)
        assert np.all(np.isnan(out[:10]))
        assert np.all(np.isfinite(out[10:]))

    def test_constant_prices_zero_vol(self):
        out = realized_vol_events(np.full(50, 100.0), window=5)
        np.testing.assert_allclose(out[5:], 0.0)

    def test_window_wider_than_series_is_all_nan(self):
        out = realized_vol_events(np.array([100.0, 100.1, 100.2]), window=10)
        assert np.all(np.isnan(out))

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            realized_vol_events(np.array([100.0, 101.0]), window=0)
        with pytest.raises(ValueError):
            realized_vol_events(np.array([100.0, -1.0]), window=1)
        with pytest.raises(ValueError):
            realized_vol_events(np.array([100.0]), window=1)


class TestWallClockVol:
    @pytest.mark.parametrize("horizon_ns", [1, 7, 1_000, 10_000_000])
    @pytest.mark.parametrize("demean", [False, True])
    @pytest.mark.parametrize("min_periods", [1, 3])
    def test_matches_brute_force(
        self, irregular_data, horizon_ns: int, demean: bool, min_periods: int
    ):
        prices, ts_ns = irregular_data
        out = realized_vol_wallclock(
            prices, ts_ns, horizon_ns, min_periods=min_periods, demean=demean
        )
        expected = _brute_wallclock(prices, ts_ns, horizon_ns, min_periods, demean)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)

    def test_burst_is_point_in_time(self):
        # Five events, three sharing t=10ns. At event index 2 (the middle of
        # the burst) exactly two returns have arrived (at t=10, both inside
        # the window) — the return arriving at event 3, in the *same
        # nanosecond*, must be invisible. Timestamp math alone cannot
        # distinguish these; event ordering must.
        prices = np.array([100.0, 100.2, 100.1, 100.4, 100.3])
        ts_ns = np.array([0, 10, 10, 10, 20], dtype=np.int64)
        r = np.diff(np.log(prices))

        out = realized_vol_wallclock(prices, ts_ns, horizon_ns=15)
        expected_at_2 = np.sqrt(r[0] ** 2 + r[1] ** 2)
        np.testing.assert_allclose(out[2], expected_at_2)
        # Event 3 then sees all three burst returns.
        np.testing.assert_allclose(out[3], np.sqrt((r[:3] ** 2).sum()))

    def test_silence_gap_shrinks_window(self):
        # A gap wider than the horizon leaves exactly one observable return
        # (the one spanning the gap itself, arriving at the current event).
        prices = np.array([100.0, 100.1, 100.2, 100.5, 100.4])
        ts_ns = np.array([0, 1, 2, 10_000, 10_001], dtype=np.int64)
        r = np.diff(np.log(prices))

        out = realized_vol_wallclock(prices, ts_ns, horizon_ns=100)
        np.testing.assert_allclose(out[3], abs(r[2]))

        # With min_periods=2 the post-silence window is honestly "not enough
        # information" -> NaN, never a vol estimate from one point.
        strict = realized_vol_wallclock(prices, ts_ns, horizon_ns=100, min_periods=2)
        assert np.isnan(strict[3])
        assert np.isfinite(strict[4])  # next event has 2 returns in window

    def test_left_edge_excluded(self):
        # Window is (t - T, t]: a return arriving exactly at t - T is out.
        prices = np.array([100.0, 100.1, 100.2])
        ts_ns = np.array([0, 100, 200], dtype=np.int64)
        r = np.diff(np.log(prices))
        out = realized_vol_wallclock(prices, ts_ns, horizon_ns=100)
        np.testing.assert_allclose(out[2], abs(r[1]))  # arrival at 100 == t-T: excluded

    def test_event_zero_always_nan(self, irregular_data):
        prices, ts_ns = irregular_data
        out = realized_vol_wallclock(prices, ts_ns, horizon_ns=1_000)
        assert np.isnan(out[0])

    def test_matches_pandas_rolling(self, irregular_data):
        # Same numbers as the "obvious" pandas implementation — we replace
        # it for speed and reusable window bounds, not for different math.
        prices, ts_ns = irregular_data
        horizon_ns = 1_000
        out = realized_vol_wallclock(prices, ts_ns, horizon_ns)

        r2 = pd.Series(
            np.diff(np.log(prices)) ** 2,
            index=pd.DatetimeIndex(ts_ns[1:].astype("datetime64[ns]")),
        )
        expected = np.sqrt(r2.rolling(pd.Timedelta(horizon_ns, "ns")).sum())
        np.testing.assert_allclose(out[1:], expected.to_numpy(), rtol=1e-9)

    def test_precomputed_starts_bitwise_identical(self, irregular_data):
        # The amortized path (bounds computed once, reused per statistic)
        # must be indistinguishable from the direct path.
        prices, ts_ns = irregular_data
        starts = wallclock_window_starts(ts_ns, 1_000)
        direct = realized_vol_wallclock(
            prices, ts_ns, 1_000, min_periods=2, demean=True
        )
        reused = realized_vol_wallclock(
            prices, ts_ns, 1_000, min_periods=2, demean=True, starts=starts
        )
        np.testing.assert_array_equal(direct, reused)

    def test_validation(self, irregular_data):
        prices, ts_ns = irregular_data
        with pytest.raises(ValueError):
            realized_vol_wallclock(prices, ts_ns[:-1], 100)  # length mismatch
        with pytest.raises(ValueError):
            realized_vol_wallclock(prices, ts_ns[::-1].copy(), 100)  # decreasing
        with pytest.raises(ValueError):
            realized_vol_wallclock(prices, ts_ns, 0)  # zero horizon
        with pytest.raises(ValueError):
            realized_vol_wallclock(prices, ts_ns, 100, min_periods=0)
        with pytest.raises(ValueError):
            realized_vol_wallclock(
                prices, ts_ns, 100, starts=np.zeros(3, dtype=np.int64)
            )  # bounds shape mismatch
        with pytest.raises(ValueError):
            wallclock_window_starts(ts_ns[::-1].copy(), 100)
        with pytest.raises(ValueError):
            wallclock_window_starts(ts_ns, 0)


class TestNumericalStability:
    @pytest.mark.parametrize("bad_price", [np.inf, -np.inf, np.nan])
    @pytest.mark.parametrize("position", [0, 1, 3])
    def test_nonfinite_prices_rejected(self, bad_price, position):
        prices = np.array([100., 101., 102., 103.])
        prices[position] = bad_price
        timestamps = np.arange(4, dtype=np.int64)
        starts = wallclock_window_starts(timestamps, 2)
        # The direct helper and both clocks, including reused bounds, share
        # the same validation instead of poisoning later prefix sums.
        calls = [
            lambda: log_returns(prices),
            lambda: realized_vol_events(prices, 2),
            lambda: realized_vol_wallclock(prices, timestamps, 2),
            lambda: realized_vol_wallclock(prices, timestamps, 2, starts=starts),
        ]
        for call in calls:
            with pytest.raises(ValueError, match="finite"):
                call()

    def test_demeaned_never_negative_under_cancellation(self):
        # Identical returns make the demeaned sum of squares exactly 0 in
        # real arithmetic; prefix-sum differencing can land at -1e-20 in
        # floats. sqrt must never see it.
        prices = 100.0 * 1.0001 ** np.arange(500, dtype=np.float64)
        for out in (
            realized_vol_events(prices, window=20, demean=True),
            realized_vol_wallclock(
                prices, np.arange(500, dtype=np.int64) * 10, 200, demean=True
            ),
        ):
            finite = out[np.isfinite(out)]
            assert np.all(finite >= 0.0)
            np.testing.assert_allclose(finite, 0.0, atol=1e-10)


class TestEngineIntegration:
    def test_micro_price_vol_end_to_end(self):
        """Full pipeline smoke test: feed -> micro-price -> both vol clocks."""
        book = SyntheticLOBGenerator(LOBConfig(n_events=50_000, seed=41)).generate()
        micro = MicrostructureFeatures(book).micro_price()
        ts_ns = book["ts_ns"].to_numpy()

        ev = realized_vol_events(micro, window=100)
        wc = realized_vol_wallclock(micro, ts_ns, horizon_ns=1_000_000_000)

        for out in (ev, wc):
            assert out.shape == micro.shape
            finite = out[np.isfinite(out)]
            assert finite.size > 0.9 * out.size  # densely defined
            assert np.all(finite >= 0.0)
        # log_returns alignment sanity: first entry undefined on both clocks.
        assert np.isnan(ev[0]) and np.isnan(wc[0])
        assert log_returns(micro).size == micro.size - 1
