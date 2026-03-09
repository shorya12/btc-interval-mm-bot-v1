"""Tests for gap detector."""

from datetime import datetime, timedelta
import pytest

from src.data_pipeline.gap_detector import detect_gaps, Gap
from src.persistence.models import CryptoPrice


def make_price(ts: datetime, symbol: str = "BTC/USDT") -> CryptoPrice:
    return CryptoPrice(
        id=None,
        symbol=symbol,
        price=50000.0,
        timestamp=ts,
        source="test",
    )


class TestDetectGaps:
    def test_no_gaps(self):
        base = datetime(2024, 1, 1, 0, 0, 0)
        candles = [make_price(base + timedelta(hours=i)) for i in range(10)]
        gaps = detect_gaps(candles, timeframe_seconds=3600)
        assert gaps == []

    def test_detects_single_gap(self):
        base = datetime(2024, 1, 1, 0, 0, 0)
        # Times: 0,1,2,3,4 then jump to 10 = gap from hour 4 to hour 10 = 6h gap
        times = [base + timedelta(hours=i) for i in range(5)]
        times.append(base + timedelta(hours=10))  # big jump
        times.extend([base + timedelta(hours=10 + i) for i in range(1, 5)])
        candles = [make_price(t) for t in times]

        gaps = detect_gaps(candles, timeframe_seconds=3600)
        assert len(gaps) == 1
        assert gaps[0].duration_seconds == pytest.approx(6 * 3600)

    def test_gap_threshold_is_2x_interval(self):
        base = datetime(2024, 1, 1)
        # Exactly 2x interval = NOT a gap (boundary)
        candles = [make_price(base), make_price(base + timedelta(hours=2))]
        gaps = detect_gaps(candles, timeframe_seconds=3600)
        assert gaps == []

    def test_gap_at_3x_interval(self):
        base = datetime(2024, 1, 1)
        candles = [make_price(base), make_price(base + timedelta(hours=3))]
        gaps = detect_gaps(candles, timeframe_seconds=3600)
        assert len(gaps) == 1

    def test_empty_input(self):
        gaps = detect_gaps([], timeframe_seconds=3600)
        assert gaps == []

    def test_single_candle(self):
        candles = [make_price(datetime(2024, 1, 1))]
        gaps = detect_gaps(candles, timeframe_seconds=3600)
        assert gaps == []

    def test_gap_dataclass(self):
        base = datetime(2024, 1, 1)
        end = base + timedelta(hours=5)
        candles = [make_price(base), make_price(end)]
        gaps = detect_gaps(candles, timeframe_seconds=3600)
        assert gaps[0].start == base
        assert gaps[0].end == end
        assert gaps[0].duration_seconds == 5 * 3600
