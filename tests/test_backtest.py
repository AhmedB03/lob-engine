"""Backtest engine tests.

Three layers, in escalating order of paranoia:

1. Hand-computed micro-scenarios: latency mapping, fill prices, and a full
   round trip whose every ledger number is derived on paper.
2. Brute-force equivalence: a deliberately naive event-loop backtester (the
   definition, written sequentially) must agree with the vectorized engine
   on every curve, over random strategies on bursty and generated feeds.
3. Leakage detection: the zero-alpha protocol passes on the honest engine
   AND fires when a one-event look-ahead is injected through the testing
   hook. A leak detector is only trustworthy once it has caught a planted
   bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_engine import (
    ExecutionSimulator,
    LOBConfig,
    SyntheticLOBGenerator,
    latency_align,
    mean_reversion_probe,
    run_null_feed_validation,
    sharpe_tstat,
)

# ----------------------------------------------------------------------
# Brute-force reference: the event loop the engine refuses to write
# ----------------------------------------------------------------------


def brute_force_backtest(
    ts: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    target: np.ndarray,
    latency_ns: int,
) -> dict[str, np.ndarray]:
    n = ts.size
    mid = 0.5 * (bid + ask)

    # Orders and their fills, sequentially.
    fills: list[tuple[int, int, float]] = []  # (exec_idx, qty, price)
    prev = 0
    for t in range(n):
        d = int(target[t]) - prev
        prev = int(target[t])
        if d == 0:
            continue
        if latency_ns == 0:
            k = t
        else:
            k = next((j for j in range(t, n) if ts[j] >= ts[t] + latency_ns), None)
            if k is None:
                continue  # feed ended first
        px = ask[k] if d > 0 else bid[k]
        fills.append((k, d, px))
    fills.sort(key=lambda f: f[0])  # stable: same-event fills keep signal order

    # The classic inventory loop.
    out = {
        name: np.zeros(n)
        for name in ("position", "cash", "equity_mtm", "equity_liq", "realized", "unrealized")
    }
    pos, cash, entry, fi = 0, 0.0, 0.0, 0
    for j in range(n):
        while fi < len(fills) and fills[fi][0] == j:
            _, d, px = fills[fi]
            fi += 1
            cash -= d * px
            pos += d
            if pos != 0:
                entry = px
        out["position"][j] = pos
        out["cash"][j] = cash
        out["equity_mtm"][j] = cash + pos * mid[j]
        out["equity_liq"][j] = cash + pos * (bid[j] if pos > 0 else ask[j])
        out["unrealized"][j] = pos * (mid[j] - entry) if pos != 0 else 0.0
        out["realized"][j] = out["equity_mtm"][j] - out["unrealized"][j]
    return out


def make_book(ts: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"ts_ns": ts.astype(np.int64), "bid_px_0": bid, "ask_px_0": ask}
    )


class TestLatencyAlign:
    def test_hand_computed_with_bursts(self):
        ts = np.array([0, 10, 10, 10, 25, 40], dtype=np.int64)
        # latency 15: signal at 0 -> first stamp >= 15 is index 4 (25);
        # signals in the t=10 burst -> first stamp >= 25 is index 4;
        # signal at 25 -> >= 40 is index 5; signal at 40 -> never (6 = n).
        np.testing.assert_array_equal(
            latency_align(ts, 15), [4, 4, 4, 4, 5, 6]
        )

    def test_zero_latency_is_identity(self):
        ts = np.array([0, 5, 9], dtype=np.int64)
        np.testing.assert_array_equal(latency_align(ts, 0), [0, 1, 2])

    def test_strictly_causal_even_in_bursts(self):
        # For latency >= 1ns, execution must land strictly after the signal
        # index, including for events sharing one nanosecond.
        rng = np.random.default_rng(5)
        ts = np.cumsum(rng.choice([0, 0, 1, 3], size=500)).astype(np.int64)
        for latency in (1, 2, 100):
            k = latency_align(ts, latency)
            idx = np.arange(ts.size)
            filled = k < ts.size
            assert np.all(k[filled] > idx[filled])
            assert np.all(np.diff(k) >= 0)  # monotone: fills stay in order

    def test_matches_brute_force(self):
        rng = np.random.default_rng(6)
        ts = np.cumsum(rng.choice([0, 1, 2, 50], size=300)).astype(np.int64)
        for latency in (1, 7, 55, 10_000):
            expected = [
                next((j for j in range(ts.size) if ts[j] >= ts[t] + latency), ts.size)
                for t in range(ts.size)
            ]
            np.testing.assert_array_equal(latency_align(ts, latency), expected)

    def test_validation(self):
        with pytest.raises(ValueError):
            latency_align(np.array([3, 2, 1]), 10)  # decreasing
        with pytest.raises(ValueError):
            latency_align(np.array([1, 2, 3]), -1)  # negative latency


class TestExecutionAndLedger:
    def test_round_trip_by_hand(self):
        # Book: constant 2-tick spread, mid drifts up one tick mid-way.
        ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
        bid = np.array([99.99, 99.99, 100.00, 100.00, 100.00])
        ask = np.array([100.01, 100.01, 100.02, 100.02, 100.02])
        sim = ExecutionSimulator(make_book(ts, bid, ask))

        # Long from event 0, flat from event 2; latency 100ns pushes the
        # buy to event 1 and the sell to event 3.
        target = np.array([1, 1, 0, 0, 0])
        res = sim.run(target, latency_ns=100)

        assert res.n_fills == 2
        np.testing.assert_array_equal(res.fills.signal_idx, [0, 2])
        np.testing.assert_array_equal(res.fills.exec_idx, [1, 3])
        np.testing.assert_array_equal(res.fills.qty, [1, -1])
        np.testing.assert_allclose(res.fills.price, [100.01, 100.00])  # ask, then bid

        led = res.ledger
        np.testing.assert_array_equal(led.position, [0, 1, 1, 0, 0])
        np.testing.assert_allclose(led.cash, [0.0, -100.01, -100.01, -0.01, -0.01])
        # While long: entry 100.01, unrealized = mid - entry.
        np.testing.assert_allclose(led.entry_price[1:3], [100.01, 100.01])
        assert np.isnan(led.entry_price[0]) and np.isnan(led.entry_price[4])
        np.testing.assert_allclose(
            led.unrealized_pnl, [0.0, -0.01, 0.0, 0.0, 0.0], atol=1e-12
        )
        # Round trip: bought 100.01, sold 100.00 -> realized -0.01 (the spread).
        np.testing.assert_allclose(led.realized_pnl[-1], -0.01)
        np.testing.assert_allclose(led.equity_mtm, led.realized_pnl + led.unrealized_pnl)
        # Liquidation mark is worse than (or equal to) MtM exactly when open.
        assert np.all(led.equity_liq <= led.equity_mtm + 1e-12)
        np.testing.assert_allclose(
            led.equity_mtm[led.position == 0], led.equity_liq[led.position == 0]
        )

    def test_flip_executes_double_and_resets_entry(self):
        ts = np.array([0, 10, 20], dtype=np.int64)
        bid = np.array([100.00, 100.10, 100.20])
        ask = np.array([100.02, 100.12, 100.22])
        sim = ExecutionSimulator(make_book(ts, bid, ask))

        target = np.array([1, -1, -1])  # long, then flip short
        res = sim.run(target, latency_ns=1)
        np.testing.assert_array_equal(res.fills.qty, [1, -2])
        np.testing.assert_array_equal(res.fills.exec_idx, [1, 2])
        # Flip sells 2 at the bid of event 2.
        np.testing.assert_allclose(res.fills.price, [100.12, 100.20])
        led = res.ledger
        np.testing.assert_array_equal(led.position, [0, 1, -1])
        # New entry is the flip's fill price.
        np.testing.assert_allclose(led.entry_price[2], 100.20)

    def test_unfilled_tail_order_is_dropped(self):
        ts = np.array([0, 10, 20], dtype=np.int64)
        px = np.full(3, 100.0)
        sim = ExecutionSimulator(make_book(ts, px - 0.01, px + 0.01))
        target = np.array([0, 0, 1])  # order born at the last event
        res = sim.run(target, latency_ns=100)
        assert res.n_fills == 0
        assert res.final_position == 0
        np.testing.assert_allclose(res.ledger.equity_mtm, 0.0)

    def test_input_validation(self):
        book = SyntheticLOBGenerator(LOBConfig(n_events=100, seed=3)).generate()
        sim = ExecutionSimulator(book)
        with pytest.raises(ValueError):
            sim.run(np.zeros(99), latency_ns=1)  # wrong length
        with pytest.raises(ValueError):
            sim.run(np.full(100, 2), latency_ns=1)  # position out of range
        with pytest.raises(ValueError):
            sim.run(np.zeros(100), latency_ns=1, price_mode="magic")
        with pytest.raises(ValueError):
            ExecutionSimulator(book.drop(columns=["ask_px_0"]))


class TestBruteForceEquivalence:
    @pytest.mark.parametrize("latency_ns", [0, 1, 5_000, 2_000_000, 10**12])
    def test_generated_feed_random_strategy(self, latency_ns: int):
        book = SyntheticLOBGenerator(LOBConfig(n_events=2_000, seed=13)).generate()
        sim = ExecutionSimulator(book)
        rng = np.random.default_rng(14)
        target = rng.integers(-1, 2, size=2_000)

        res = sim.run(target, latency_ns=latency_ns)
        ref = brute_force_backtest(
            book["ts_ns"].to_numpy(),
            book["bid_px_0"].to_numpy(),
            book["ask_px_0"].to_numpy(),
            target,
            latency_ns,
        )
        led = res.ledger
        np.testing.assert_array_equal(led.position, ref["position"])
        np.testing.assert_allclose(led.cash, ref["cash"], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(led.equity_mtm, ref["equity_mtm"], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(led.equity_liq, ref["equity_liq"], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(led.realized_pnl, ref["realized"], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(led.unrealized_pnl, ref["unrealized"], rtol=1e-12, atol=1e-9)

    def test_bursty_feed_with_ties(self):
        # Nanosecond bursts: multiple events share stamps, so latency must
        # skip whole bursts; the brute force encodes that by definition.
        rng = np.random.default_rng(15)
        n = 800
        ts = np.cumsum(rng.choice([0, 0, 1, 4, 200], size=n)).astype(np.int64)
        mid_ticks = 10_000 + np.cumsum(rng.choice([-1, 0, 1], size=n))
        spread = rng.integers(1, 4, size=n)
        bid = (mid_ticks - spread // 2) * 0.01
        ask = bid + spread * 0.01
        target = rng.integers(-1, 2, size=n)

        sim = ExecutionSimulator(make_book(ts, bid, ask))
        for latency in (1, 3, 500):
            res = sim.run(target, latency_ns=latency)
            ref = brute_force_backtest(ts, bid, ask, target, latency)
            np.testing.assert_array_equal(res.ledger.position, ref["position"])
            np.testing.assert_allclose(res.ledger.equity_mtm, ref["equity_mtm"], atol=1e-9)
            np.testing.assert_allclose(res.ledger.equity_liq, ref["equity_liq"], atol=1e-9)
            np.testing.assert_allclose(res.ledger.realized_pnl, ref["realized"], atol=1e-9)


@pytest.fixture(scope="module")
def null_setup() -> tuple[ExecutionSimulator, np.ndarray]:
    """Bounce-free control feed: constant spread means the quoted mid IS the
    efficient price, i.e. a true martingale — the setting in which 'any
    causal strategy has zero expected pre-cost PnL' actually holds."""
    config = LOBConfig(
        n_events=120_000, seed=97, mean_spread_ticks=2.0, spread_sigma_ticks=0.0
    )
    sim = ExecutionSimulator(SyntheticLOBGenerator(config).generate())
    return sim, mean_reversion_probe(sim.mid)


@pytest.fixture(scope="module")
def bounce_setup() -> tuple[ExecutionSimulator, np.ndarray]:
    """AR(1)-spread feed: the quoted mid = efficient price + half-tick
    spread-parity term, which mean-reverts — predictably bouncy quotes."""
    config = LOBConfig(n_events=120_000, seed=97)
    sim = ExecutionSimulator(SyntheticLOBGenerator(config).generate())
    return sim, mean_reversion_probe(sim.mid)


class TestZeroAlphaLeakage:
    def test_validation_protocol_passes(self):
        report = run_null_feed_validation(
            n_events=120_000, seed=97, latency_ns=2_000_000
        )
        assert report.passed, report.summary()
        assert abs(report.tstat_frictionless) < report.tstat_threshold
        assert abs(report.tstat_mid_latency) < report.tstat_threshold
        assert report.pnl_post_cost < 0
        assert report.spread_paid > 0
        assert report.decomposition_error < 1e-6

    def test_cost_decomposition_is_exact_identity(self, null_setup):
        # Same fills, prices differing by the half-spread each: cross-mode
        # PnL must equal mid-mode PnL minus spread_paid to float precision.
        sim, target = null_setup
        mid_run = sim.run(target, latency_ns=1_000_000, price_mode="mid")
        cross_run = sim.run(target, latency_ns=1_000_000, price_mode="cross")
        assert cross_run.n_fills == mid_run.n_fills
        assert cross_run.spread_paid > 0
        np.testing.assert_allclose(
            cross_run.total_pnl_mtm,
            mid_run.total_pnl_mtm - cross_run.spread_paid,
            rtol=1e-9,
        )

    def test_detector_fires_on_injected_lookahead(self, null_setup):
        # Contrapositive: execute each order one event BEFORE its signal —
        # the mean-reversion probe then fades moves it "hasn't seen yet",
        # capturing -|move| every time. The t-stat must explode.
        sim, target = null_setup
        n = sim.n_events
        leak_exec = np.maximum(np.arange(n, dtype=np.int64) - 1, 0)

        honest = sim.run(target, latency_ns=0, price_mode="mid")
        leaked = sim.run(target, latency_ns=0, price_mode="mid", exec_idx=leak_exec)

        t_honest = sharpe_tstat(honest.ledger.equity_mtm)
        t_leaked = sharpe_tstat(leaked.ledger.equity_mtm)
        assert abs(t_honest) < 4.0
        assert abs(t_leaked) > 10.0  # in practice it lands in the hundreds
        # And the sign is the physically meaningful one: front-running your
        # own fade signal *loses* |move| per move.
        assert t_leaked < 0

    def test_latency_cannot_create_alpha(self, null_setup):
        # Sweeping latency changes costs and timing, never expectation:
        # every mid-fill run on the null feed stays statistically flat.
        sim, target = null_setup
        for latency in (1, 100_000, 10_000_000):
            res = sim.run(target, latency_ns=latency, price_mode="mid")
            assert abs(sharpe_tstat(res.ledger.equity_mtm)) < 4.0

    def test_crossing_always_loses_on_null_feed(self, null_setup):
        sim, target = null_setup
        for latency in (1, 2_000_000):
            res = sim.run(target, latency_ns=latency, price_mode="cross")
            assert res.total_pnl_mtm < 0
            assert res.total_pnl_liq <= res.total_pnl_mtm + 1e-12


class TestQuoteBounceDiscovery:
    """The detector's first real catch, preserved as a positive test.

    On the AR(1)-spread feed the *quoted* midpoint is not a martingale:
    ``mid_quoted = efficient + (spread mod 2)/2`` ticks, and the parity
    term mean-reverts. Fading the last quoted-mid move therefore shows
    genuine pre-cost alpha — this is textbook bid-ask bounce, not leakage.
    The economics must still close: the bounce is worth ~half a tick while
    crossing the spread costs a full half-spread per fill, so post-cost the
    probe loses. If either half of this test breaks, the generator's
    microstructure changed and the validation protocol needs re-deriving.
    """

    def test_bounce_alpha_is_real_pre_cost(self, bounce_setup):
        sim, target = bounce_setup
        res = sim.run(target, latency_ns=0, price_mode="mid")
        t = sharpe_tstat(res.ledger.equity_mtm)
        # Strongly positive: the probe fades a genuinely mean-reverting
        # component of the quoted mid.
        assert t > 10.0

    def test_bounce_is_not_tradeable_post_cost(self, bounce_setup):
        sim, target = bounce_setup
        for latency in (0, 2_000_000):
            res = sim.run(target, latency_ns=latency, price_mode="cross")
            assert res.total_pnl_mtm < 0

    def test_bounce_alpha_decays_with_latency(self, bounce_setup):
        # The parity term reverts within ~an event or two, so even modest
        # latency destroys most of the pre-cost edge.
        sim, target = bounce_setup
        t_instant = sharpe_tstat(
            sim.run(target, latency_ns=0, price_mode="mid").ledger.equity_mtm
        )
        t_delayed = sharpe_tstat(
            sim.run(target, latency_ns=2_000_000, price_mode="mid").ledger.equity_mtm
        )
        assert abs(t_delayed) < abs(t_instant) / 2
