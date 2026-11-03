"""Run reproducible volatility comparisons; optionally save the full JSON report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_engine.bench.benchmarks import run_benchmarks


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-ns", type=positive_int, default=100_000_000)
    parser.add_argument("--scenarios", nargs="+", choices=["regular", "bursts", "gaps"], default=["regular", "bursts", "gaps"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sizes) < 2:
        parser.error("sizes must be at least 2")
    report = run_benchmarks(args.sizes, repeats=args.repeats, seed=args.seed,
                            scenarios=args.scenarios, horizon_ns=args.horizon_ns)
    print("Correctness: PASS (all outputs checked before timing)")
    print(f"{'scenario':<10} {'events':>9} {'method':<16} {'median ms':>12} {'events/s':>14}")
    for row in report["results"]:
        throughput = row['events_per_second']
        rate = f"{throughput:,.0f}" if throughput is not None else "n/a"
        print(f"{row['scenario']:<10} {row['n_events']:>9,} {row['method']:<16} {row['median_seconds'] * 1000:>12.3f} {rate:>14}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
