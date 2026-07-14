"""Benchmark layer (Phase 4): vectorized vs. naive-loop comparisons.

Planned modules:
* ``benchmarks.py`` — timeit/cProfile harness pitting each vectorized kernel
  against a straightforward Python-loop reference implementation, with
  throughput (events/sec) and speedup tables, plus a discussion of memory
  bandwidth and cache behavior (float32 vs float64, integer ticks vs floats).
"""
