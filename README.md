# LOB Engine — Vectorized Limit Order Book Analytics & Signal Backtesting

A high-performance research engine for Level-2 market data, built on a single
hard constraint: **no O(n_events) work ever happens in the Python
interpreter.** Every stage, feed generation, feature extraction, execution
simulation, is expressed as vectorized NumPy/Pandas kernels that stream over
contiguous arrays, the way the hardware wants to be fed.

## Why vectorization is the whole point

Python's interpreter dispatches ~10⁷ simple ops/sec; NumPy kernels move
10⁸–10⁹ elements/sec because they run compiled loops over contiguous,
cache-line-friendly memory with no per-element boxing or refcounting. At
tick-data scale (10⁶–10⁹ events/day/instrument) that is the difference
between an interactive research loop and an overnight batch job. The
discipline this imposes, thinking in whole-array transformations, prefix
sums instead of running state, `searchsorted` instead of event loops,is the
same mental model that writes fast C++.

## Architecture

```
hrt/
├── pyproject.toml
├── README.md
├── src/lob_engine/
│   ├── data/                    # Phases 1–2 ✅
│   │   ├── generator.py         #   synthetic L2 feed: Poisson arrivals,
│   │   │                        #   tick-grid mid walk, AR(1) spreads
│   │   │                        #   (recursion vectorized as an IIR filter
│   │   │                        #   via scipy.signal.lfilter), lognormal depth
│   │   └── schema.py            #   canonical dtypes + lossless downcasting
│   ├── features/                # Phases 1–2 ✅
│   │   ├── microstructure.py    #   OBI, micro-price, spread dynamics
│   │   └── volatility.py        #   dual-clock realized vol: event-time and
│   │                            #   wall-clock windows via searchsorted +
│   │                            #   prefix sums (O(1) per window)
│   ├── backtest/                # Phase 3 ✅
│   │   ├── engine.py            #   latency via searchsorted, spread-crossing
│   │   │                        #   fills, orders as sparse target diffs
│   │   ├── accounting.py        #   ledger as cumsum algebra: position, cash,
│   │   │                        #   MtM + mark-to-liquidation, realized/
│   │   │                        #   unrealized via vectorized LOCF
│   │   └── validation.py        #   zero-alpha leakage detector (martingale
│   │                            #   null on a bounce-free control feed)
│   └── bench/                   # Phase 4 🔜
│       └── benchmarks.py        #   vectorized vs naive-loop harness
├── scripts/
│   ├── run_phase1_demo.py       # feed → memory optimization → signals
│   ├── run_phase2_demo.py       # AR(1) spreads + dual-clock vol + honest
│   │                            #   head-to-head vs pandas .rolling("1s")
│   └── run_phase3_demo.py       # latency sweep, quote-bounce economics,
│                                #   null-feed validation report
└── tests/                       # exact-value, invariant, statistical, and
                                 #   brute-force-equivalence suites
```

**Layering rule:** `data` → `features` → `backtest` is a one-way dependency
chain; `bench` may import everything. Pandas lives at the boundaries
(labeled I/O), NumPy in the cores (hot paths hold `(n_events, n_levels)`
matrices extracted once, never per-call column lookups).

## Quickstart

```bash
python scripts/run_phase1_demo.py --n-events 1000000   # feed + signals demo
python scripts/run_phase2_demo.py --n-events 1000000   # dual-clock vol demo
python scripts/run_phase3_demo.py --n-events 1000000   # backtest + leak check
python -m pytest                                       # test suite
```

Requires Python ≥ 3.10, NumPy ≥ 1.26, Pandas ≥ 2.1, SciPy ≥ 1.11.

## Phase 1 design decisions

- **Integer tick grid.** Prices are integer tick counts until the final
  float conversion, so books cannot be crossed by floating-point drift and
  the generator's invariants (monotone ladders, `ask − bid ≡ spread`) are
  exact. Timestamps are int64 nanoseconds, exact arithmetic for the latency
  modeling coming in Phase 3.
- **Memory before FLOPs.** Vectorized workloads are memory-bandwidth-bound,
  so `schema.optimize_memory` downcasts every column to the smallest dtype
  that is *provably* lossless (sizes by observed max; prices to float32 only
  if the worst-case round-trip error stays under half a tick).
- **Signals with a story.** OBI is the normalized resting-liquidity
  imbalance in [−1, 1]; the micro-price weights the touch quotes by
  *opposite* queue sizes, the thin side gets consumed first, so
  `micro − mid` is a directional expectation, and it is the signal Phase 3
  will trade.
- **Known-zero-alpha testbed.** The synthetic mid walk is independent of the
  book state: the demo verifies `corr(micro_dev, next return) ≈ 0`, so any
  alpha the backtester later reports on structured data cannot be a pipeline
  artifact (look-ahead leakage shows up immediately as fake alpha here).

## Phase 2 design decisions

- **Recursion ≠ loop.** An AR(1) spread `S_t = c + φS_{t−1} + ε_t` has a
  serial dependency, and the "obvious" closed-form vectorization
  (`φ^t · cumsum(ε·φ^−t)`) overflows float64 after ~14k steps at φ = 0.95.
  The recursion *is* a first-order IIR filter, so `scipy.signal.lfilter`
  evaluates it in one compiled O(n) pass, initialized from the stationary
  distribution so no burn-in is needed. Tests pin the exact recursion
  against a Python-loop reference and the stationary moments
  (mean, variance `σ²/(1−φ²)`, lag-1 autocorrelation ≈ φ).
- **Two clocks for volatility.** Event-time windows (last N ticks) adapt to
  activity; wall-clock windows (last T nanoseconds) are what risk speaks.
  Both run on one primitive: prefix sums make any window sum an O(1)
  difference, and `np.searchsorted` finds all n wall-clock window starts in
  one vectorized binary search. Window ends follow *event order*, so no
  event ever sees a return that arrived after it, even inside the same
  nanosecond burst.
- **Honest benchmarking.** Pandas `.rolling("1s")` produces identical
  numbers (asserted in tests, |diff| ~1e-15) and is ~2x faster for a single
  statistic, its Cython indexer finds bounds with an O(n) two-pointer scan,
  which NumPy has no primitive for. Our engine amortizes: bounds are
  computed once (`wallclock_window_starts`) and reused by any
  prefix-summable statistic. The gap and its cause are documented, measured,
  and queued as a Phase 4 target, a rigged benchmark would be worse than a
  lost one.
- **NaN over fabrication.** Windows with insufficient history (post-silence,
  warm-up) yield NaN, never 0: "no information" and "zero volatility" are
  different claims.

## Phase 3 design decisions

- **The event loop, factored into array algebra.** Orders are the sparse
  nonzeros of `diff(target_position)`; latency alignment is one
  `searchsorted` (`k = first event with T_k ≥ T_t + latency`, causal even
  inside nanosecond bursts); fills scatter onto the event axis with
  `bincount`; position and cash are `cumsum`; the entry price of the open
  position is a vectorized last-observation-carried-forward
  (`np.maximum.accumulate` over setter indices). A deliberately naive
  event-loop backtester lives in the test suite, and every ledger curve
  must match it exactly.
- **Costs are prices, not fees.** Buys lift the ask, sells hit the bid —
  the strategy pays the *quantized AR(1) spread prevailing at execution
  time*. The decomposition `pnl_cross = pnl_mid − spread_paid` is an
  algebraic identity, asserted to float precision.
- **Unit inventory, enforced.** Positions live in {−1, 0, +1}; that is
  what makes entry-price tracking a pure LOCF instead of an average-cost
  recursion (genuinely serial, unlike the AR(1) filter). The ledger
  *refuses* fill streams it cannot account for exactly.
- **The leak detector caught something real.** The quoted midpoint is
  `efficient price + (spread mod 2)/2` ticks, and under AR(1) spreads the
  parity term mean-reverts: quoted mids *predictably bounce*. A fade-the-
  last-move probe shows t-stat ≈ +81 pre-cost at zero latency, genuine
  bid-ask bounce, decaying to ~0 by 10ms of latency and always losing
  post-cost. The martingale null therefore runs on a bounce-free
  constant-spread control feed, the bounce economics are preserved as
  their own tests, and the suite *injects* a one-event look-ahead through
  a testing hook to prove the detector fires (|t| explodes). A leak
  detector that has never caught a planted bug is decoration.

## Roadmap

### Reproducible volatility benchmarks

Install the project with `python -m pip install -e ".[dev]"`, then run:

```bash
python scripts/run_benchmarks.py --sizes 1000 10000 100000 --repeats 5 --output benchmark-results/local.json
```

The harness compares a Python window-reduction loop, NumPy prefix sums, and
pandas rolling sums on identical seeded regular, burst, and gapped feeds.
Every full output (including NaN positions) must match the loop reference
before timing begins. Each method gets one warm-up/validation call; timed
execution order rotates between repetitions. Reports include all samples,
median/minimum latency, throughput, input bytes, dependency versions,
platform details, seed, and thread environment settings.

Timings include return calculation and window setup for each method. They
exclude data generation, correctness checks, and serialization. Input bytes
are not peak memory usage. Results depend on hardware and background load;
there are no speed thresholds in tests and no promise that NumPy beats pandas.
This is the first part of Phase 4: compiled O(n) bounds scans, other kernels,
and cache/bandwidth profiling remain future work.

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 1 | Data pipeline, synthetic L2 generator, OBI + micro-price | ✅ |
| 2 | AR(1) spread process, dual-clock realized volatility | ✅ |
| 3 | Vectorized backtester: latency, spread costs, ledger, leak detector | ✅ |
| 4 | Volatility benchmark harness shipped; O(n) bounds scan and cache/bandwidth analysis remain | In progress |
