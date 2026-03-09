"""Gap detection for OHLCV candle sequences."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.common.logging import get_logger
from src.persistence.models import CryptoPrice, EventSeverity

logger = get_logger(__name__)


@dataclass
class Gap:
    """Represents a gap in candle data."""

    start: datetime
    end: datetime
    duration_seconds: float

    def __repr__(self) -> str:
        return f"Gap(start={self.start}, end={self.end}, duration={self.duration_seconds:.0f}s)"


def detect_gaps(candles: list[CryptoPrice], timeframe_seconds: int) -> list[Gap]:
    """
    Detect gaps in a candle sequence.

    A gap is any span > 2× the expected candle interval.

    Args:
        candles: List of CryptoPrice records, ordered by timestamp ascending
        timeframe_seconds: Expected interval between candles in seconds

    Returns:
        List of Gap objects
    """
    if len(candles) < 2:
        return []

    threshold = timeframe_seconds * 2
    gaps: list[Gap] = []

    for i in range(1, len(candles)):
        prev_ts = candles[i - 1].timestamp
        curr_ts = candles[i].timestamp

        # Handle both naive and aware datetimes
        if hasattr(prev_ts, "timestamp"):
            delta = (curr_ts - prev_ts).total_seconds()
        else:
            delta = float((curr_ts - prev_ts))

        if delta > threshold:
            gap = Gap(start=prev_ts, end=curr_ts, duration_seconds=delta)
            gaps.append(gap)
            logger.debug(
                "gap_found",
                start=str(prev_ts),
                end=str(curr_ts),
                duration_seconds=delta,
                expected_seconds=timeframe_seconds,
            )

    return gaps


async def flag_gaps(gaps: list[Gap], repository: Any) -> None:
    """
    Write each gap to EventLog with severity WARNING.

    Args:
        gaps: List of gaps to flag
        repository: Repository instance
    """
    for gap in gaps:
        await repository.log_event(
            event_type="data_gap_detected",
            message=f"Candle gap of {gap.duration_seconds:.0f}s detected",
            severity=EventSeverity.WARNING,
            data={
                "gap_start": str(gap.start),
                "gap_end": str(gap.end),
                "duration_seconds": gap.duration_seconds,
            },
        )
        logger.warning(
            "gap_flagged",
            start=str(gap.start),
            end=str(gap.end),
            duration_seconds=gap.duration_seconds,
        )


def filter_candles_with_gaps(
    candles: list[CryptoPrice],
    gaps: list[Gap],
    window_size: int,
    timeframe_seconds: int,
) -> list[list[CryptoPrice]]:
    """
    Split candles into contiguous segments excluding gap windows.

    Training windows that overlap a gap are excluded entirely.

    Args:
        candles: Ordered candle list
        gaps: Detected gaps
        window_size: Training window size in candles
        timeframe_seconds: Expected interval

    Returns:
        List of contiguous candle segments safe for training
    """
    if not gaps:
        return [candles]

    gap_intervals: set[int] = set()
    for gap in gaps:
        for i, c in enumerate(candles):
            ts = c.timestamp
            if gap.start <= ts <= gap.end:
                # Mark this index and surrounding window
                for j in range(max(0, i - window_size), min(len(candles), i + window_size)):
                    gap_intervals.add(j)

    segments: list[list[CryptoPrice]] = []
    current: list[CryptoPrice] = []

    for i, c in enumerate(candles):
        if i in gap_intervals:
            if len(current) >= window_size:
                segments.append(current)
            current = []
        else:
            current.append(c)

    if len(current) >= window_size:
        segments.append(current)

    return segments
