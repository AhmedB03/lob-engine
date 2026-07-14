"""Feature correctness tests.

Two layers of assurance: exact-value checks against a tiny hand-computed
book (catches formula errors), and invariant checks over a large generated
feed (catches numerical/vectorization errors at scale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_engine import LOBConfig, MicrostructureFeatures, SyntheticLOBGenerator


@pytest.fixture()
def tiny_book() -> pd.DataFrame:
    """Two-level book, three events, all values chosen for mental arithmetic."""
    return pd.DataFrame(
        {
            "ts_ns": [1, 2, 3],
            "bid_px_0": [99.99, 99.98, 100.00],
            "bid_sz_0": [300, 100, 250],
            "ask_px_0": [100.01, 100.00, 100.02],
            "ask_sz_0": [100, 100, 250],
            "bid_px_1": [99.98, 99.97, 99.99],
            "bid_sz_1": [500, 400, 300],
            "ask_px_1": [100.02, 100.01, 100.03],
            "ask_sz_1": [500, 200, 300],
        }
    )


@pytest.fixture(scope="module")
def big_features() -> tuple[pd.DataFrame, MicrostructureFeatures]:
    book = SyntheticLOBGenerator(LOBConfig(n_events=50_000, seed=11)).generate()
    return book, MicrostructureFeatures(book)


class TestExactValues:
    def test_mid_and_spread(self, tiny_book: pd.DataFrame) -> None:
        feats = MicrostructureFeatures(tiny_book)
        np.testing.assert_allclose(feats.mid_price(), [100.00, 99.99, 100.01])
        np.testing.assert_allclose(feats.spread(), [0.02, 0.02, 0.02], atol=1e-12)

    def test_obi_touch(self, tiny_book: pd.DataFrame) -> None:
        # Row 0: (300-100)/(300+100) = 0.5; row 1 balanced; row 2 balanced.
        obi = MicrostructureFeatures(tiny_book).order_book_imbalance(depth=1)
        np.testing.assert_allclose(obi, [0.5, 0.0, 0.0])

    def test_obi_two_levels_unweighted(self, tiny_book: pd.DataFrame) -> None:
        # Row 1: B = 100+400 = 500, A = 100+200 = 300 -> 200/800 = 0.25.
        obi = MicrostructureFeatures(tiny_book).order_book_imbalance(depth=2)
        np.testing.assert_allclose(obi[1], 0.25)

    def test_obi_level_decay(self, tiny_book: pd.DataFrame) -> None:
        # Row 1 with lambda = ln(2): weights (1, 0.5).
        # B = 100 + 0.5*400 = 300, A = 100 + 0.5*200 = 200 -> 100/500 = 0.2.
        obi = MicrostructureFeatures(tiny_book).order_book_imbalance(
            depth=2, decay=np.log(2.0)
        )
        np.testing.assert_allclose(obi[1], 0.2)

    def test_micro_price(self, tiny_book: pd.DataFrame) -> None:
        # Row 0: (99.99*100 + 100.01*300) / 400 = 100.005 — heavy bid queue
        # pushes the micro-price toward the ask.
        micro = MicrostructureFeatures(tiny_book).micro_price()
        np.testing.assert_allclose(micro, [100.005, 99.99, 100.01])

    def test_micro_dev_sign_follows_imbalance(self, tiny_book: pd.DataFrame) -> None:
        feats = MicrostructureFeatures(tiny_book)
        dev = feats.micro_price() - feats.mid_price()
        assert dev[0] > 0  # bid-heavy book -> upward-adjusted expectation
        np.testing.assert_allclose(dev[1:], 0.0, atol=1e-12)


class TestInvariantsAtScale:
    def test_obi_bounded(self, big_features) -> None:
        _, feats = big_features
        for depth in (1, 3, 5):
            obi = feats.order_book_imbalance(depth=depth, decay=0.5)
            assert np.all(obi >= -1.0) and np.all(obi <= 1.0)
            assert np.all(np.isfinite(obi))

    def test_micro_price_inside_touch(self, big_features) -> None:
        book, feats = big_features
        micro = feats.micro_price()
        assert np.all(micro >= book["bid_px_0"].to_numpy())
        assert np.all(micro <= book["ask_px_0"].to_numpy())

    def test_compute_all_schema(self, big_features) -> None:
        book, feats = big_features
        frame = feats.compute_all()
        assert list(frame.columns) == [
            "mid",
            "spread",
            "spread_bps",
            "obi_l1",
            "obi_l5_w",
            "micro_price",
            "micro_dev",
        ]
        assert len(frame) == len(book)
        assert frame.index.equals(book.index)
        assert np.all(np.isfinite(frame.to_numpy()))


class TestValidation:
    def test_depth_out_of_range(self, tiny_book: pd.DataFrame) -> None:
        feats = MicrostructureFeatures(tiny_book)
        with pytest.raises(ValueError):
            feats.order_book_imbalance(depth=3)
        with pytest.raises(ValueError):
            feats.order_book_imbalance(depth=0)

    def test_missing_columns_rejected(self, tiny_book: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            MicrostructureFeatures(tiny_book.drop(columns=["ask_sz_1"]))

    def test_non_lob_frame_rejected(self) -> None:
        with pytest.raises(ValueError):
            MicrostructureFeatures(pd.DataFrame({"close": [1.0, 2.0]}))

    def test_empty_touch_falls_back_to_mid(self, tiny_book: pd.DataFrame) -> None:
        # Real feeds can have empty touch queues; the engine must emit the
        # mid there rather than NaN/inf.
        crippled = tiny_book.copy()
        crippled.loc[1, ["bid_sz_0", "ask_sz_0"]] = 0
        feats = MicrostructureFeatures(crippled)
        micro = feats.micro_price()
        np.testing.assert_allclose(micro[1], feats.mid_price()[1])
        assert np.all(np.isfinite(micro))
