import json

import numpy as np
import pytest

from lob_engine.bench.benchmarks import METHODS, run_benchmarks, workload


@pytest.mark.parametrize("scenario", ["regular", "bursts", "gaps"])
def test_deterministic_workloads(scenario):
    first = workload(100, 42, scenario)
    second = workload(100, 42, scenario)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(first[1]) >= 0)


def test_burst_causality_and_left_boundary():
    prices = np.exp(np.array([0., 1., 3., 6., 10.]))
    stamps = np.array([0, 1, 1, 4, 100], dtype=np.int64)
    # At t=4, the returns at t=1 fall exactly on the excluded left edge.
    expected = [np.nan, 1., np.sqrt(5.), 3., 4.]
    for method in METHODS.values():
        np.testing.assert_allclose(method(prices, stamps, 3), expected, equal_nan=True)


def test_report_contains_samples_and_environment():
    report = run_benchmarks([20], repeats=2)
    assert len(report["results"]) == 9
    assert report["correctness"]["passed"]
    assert report["environment"]["numpy"]
    for row in report["results"]:
        assert len(row["samples_seconds"]) == 2
        assert row["input_bytes"] == 20 * 16
        assert row["median_seconds"] >= row["min_seconds"] >= 0
    json.dumps(report, allow_nan=False)


def test_wrong_implementation_fails_before_report(monkeypatch):
    monkeypatch.setitem(METHODS, "broken", lambda prices, ts, h: np.zeros(len(prices)))
    with pytest.raises(AssertionError):
        run_benchmarks([20], repeats=1)


@pytest.mark.parametrize("kwargs", [{"sizes": [1]}, {"sizes": []}, {"repeats": 0}, {"horizon_ns": 0}, {"scenarios": []}, {"scenarios": ["unknown"]}])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        run_benchmarks(**kwargs)
