"""Binance OHLCV fetcher using CCXT."""

import asyncio
from datetime import datetime, timezone
from typing import Any

import ccxt.async_support as ccxt

from src.common.logging import get_logger
from src.persistence.models import CryptoPrice, EventSeverity

logger = get_logger(__name__)

# Timeframe to seconds mapping
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class BinanceFetcher:
    """Fetches OHLCV data from Binance via CCXT."""

    def __init__(self) -> None:
        self._exchange: ccxt.Exchange | None = None

    async def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            self._exchange = ccxt.binance({"enableRateLimit": True})
        return self._exchange

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Fetch OHLCV candles from Binance.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"
            timeframe: Candle interval, e.g. "1m", "1h"
            since: Start datetime (None = most recent)
            limit: Max candles to fetch

        Returns:
            List of candle dicts with keys: timestamp, open, high, low, close, volume
        """
        exchange = await self._get_exchange()
        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            raw = await exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception as exc:
            logger.error("fetch_ohlcv_failed", symbol=symbol, timeframe=timeframe, error=str(exc))
            raise

        candles = []
        for row in raw:
            ts_ms, open_, high, low, close, volume = row
            candles.append({
                "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            })

        logger.info("fetch_ohlcv_ok", symbol=symbol, timeframe=timeframe, count=len(candles))
        return candles

    def _candle_to_crypto_price(self, symbol: str, candle: dict[str, Any], source: str = "binance_ohlcv") -> CryptoPrice:
        """Convert a candle dict to CryptoPrice model."""
        return CryptoPrice(
            id=None,
            symbol=symbol,
            price=candle["close"],  # canonical price = close
            timestamp=candle["timestamp"].replace(tzinfo=None),  # store as naive UTC
            source=source,
            volume_24h=candle["volume"],
            metadata={
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
            },
        )


async def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since: datetime | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Module-level convenience wrapper for BinanceFetcher.fetch_ohlcv."""
    fetcher = BinanceFetcher()
    try:
        return await fetcher.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    finally:
        await fetcher.close()


async def backfill(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1m",
    repository: Any = None,
) -> int:
    """
    Backfill OHLCV data from start_date to end_date.

    Args:
        symbol: e.g. "BTC/USDT"
        start_date: UTC start datetime
        end_date: UTC end datetime
        timeframe: Candle interval
        repository: Optional Repository for persistence

    Returns:
        Total candles written
    """
    tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 60)
    fetcher = BinanceFetcher()
    total = 0

    try:
        current = start_date
        while current < end_date:
            candles = await fetcher.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
            if not candles:
                break

            if repository is not None:
                for candle in candles:
                    cp = fetcher._candle_to_crypto_price(symbol, candle)
                    await repository.create_crypto_price(cp)

            total += len(candles)
            last_ts: datetime = candles[-1]["timestamp"]
            # Advance past the last fetched candle
            current = last_ts.replace(tzinfo=None) if last_ts.tzinfo else last_ts
            # Add one interval to avoid re-fetching the last candle
            from datetime import timedelta
            current = current + timedelta(seconds=tf_seconds)

            if current >= end_date:
                break

            # Respect rate limits
            await asyncio.sleep(0.2)

    finally:
        await fetcher.close()

    logger.info("backfill_complete", symbol=symbol, timeframe=timeframe, total_candles=total)
    return total


async def fetch_since_last(
    symbol: str,
    timeframe: str,
    repository: Any,
) -> int:
    """
    Incremental fetch: reads last stored candle timestamp, fetches newer candles.

    Args:
        symbol: Trading pair
        timeframe: Candle interval
        repository: Repository instance

    Returns:
        Number of new candles written
    """
    rows = await repository.get_recent_crypto_prices(symbol, limit=1)
    if rows:
        last_ts = rows[0].timestamp
        since = last_ts
    else:
        # No data — start from 6 months ago
        from datetime import timedelta
        since = datetime.utcnow() - timedelta(days=180)

    candles = await fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    if not candles:
        return 0

    fetcher = BinanceFetcher()
    count = 0
    for candle in candles:
        cp = fetcher._candle_to_crypto_price(symbol, candle)
        # Skip if timestamp <= last stored
        if rows and cp.timestamp <= rows[0].timestamp:
            continue
        try:
            await repository.create_crypto_price(cp)
            count += 1
        except Exception as exc:
            logger.warning("skip_duplicate_candle", symbol=symbol, ts=str(cp.timestamp), error=str(exc))

    logger.info("incremental_fetch_complete", symbol=symbol, new_candles=count)
    return count
