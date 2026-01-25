"""Crypto price feed using CCXT."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Any

import ccxt.async_support as ccxt

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PriceSnapshot:
    """Single price observation."""

    symbol: str
    price: float
    timestamp: datetime
    volume_24h: float | None = None
    bid: float | None = None
    ask: float | None = None
    source: str = "ccxt"


@dataclass
class PriceHistory:
    """Price history for an asset."""

    symbol: str
    prices: deque[PriceSnapshot]
    max_size: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.prices, deque):
            self.prices = deque(self.prices, maxlen=self.max_size)

    @property
    def latest(self) -> PriceSnapshot | None:
        """Get most recent price."""
        return self.prices[-1] if self.prices else None

    @property
    def latest_price(self) -> float | None:
        """Get most recent price value."""
        return self.prices[-1].price if self.prices else None

    def get_prices(self, n: int | None = None) -> list[float]:
        """Get last N prices."""
        prices = [p.price for p in self.prices]
        if n is not None:
            prices = prices[-n:]
        return prices

    def get_returns(self, n: int | None = None) -> list[float]:
        """Get log returns."""
        import math
        prices = self.get_prices()
        if len(prices) < 2:
            return []
        returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
        ]
        if n is not None:
            returns = returns[-n:]
        return returns


class PriceFeed:
    """
    Async price feed for crypto assets via CCXT.

    Fetches prices from exchanges (default: Binance) and maintains
    price history for each tracked symbol.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbols: list[str] | None = None,
        max_history: int = 1000,
    ) -> None:
        """
        Initialize price feed.

        Args:
            exchange_id: CCXT exchange ID (e.g., "binance", "kraken")
            symbols: List of symbols to track (e.g., ["BTC/USDT", "ETH/USDT"])
            max_history: Maximum price history to keep per symbol
        """
        self.exchange_id = exchange_id
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        self.max_history = max_history

        self._exchange: ccxt.Exchange | None = None
        self._history: dict[str, PriceHistory] = {}
        self._running = False

        # Initialize history for each symbol
        for symbol in self.symbols:
            self._history[symbol] = PriceHistory(
                symbol=symbol,
                prices=deque(maxlen=max_history),
                max_size=max_history,
            )

    async def start(self) -> None:
        """Initialize exchange connection."""
        if self._exchange is not None:
            return

        exchange_class = getattr(ccxt, self.exchange_id)
        self._exchange = exchange_class({
            "enableRateLimit": True,
            "timeout": 10000,
        })

        logger.info(
            "price_feed_started",
            exchange=self.exchange_id,
            symbols=self.symbols,
        )

    async def stop(self) -> None:
        """Close exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
            logger.info("price_feed_stopped")

    async def fetch_price(self, symbol: str) -> PriceSnapshot | None:
        """
        Fetch current price for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            PriceSnapshot or None if failed
        """
        if self._exchange is None:
            await self.start()

        try:
            ticker = await self._exchange.fetch_ticker(symbol)

            snapshot = PriceSnapshot(
                symbol=symbol,
                price=ticker["last"],
                timestamp=datetime.utcnow(),
                volume_24h=ticker.get("quoteVolume"),
                bid=ticker.get("bid"),
                ask=ticker.get("ask"),
                source=self.exchange_id,
            )

            # Add to history
            if symbol in self._history:
                self._history[symbol].prices.append(snapshot)

            return snapshot

        except Exception as e:
            logger.error("fetch_price_failed", symbol=symbol, error=str(e))
            return None

    async def fetch_all_prices(self) -> dict[str, PriceSnapshot]:
        """
        Fetch prices for all tracked symbols.

        Returns:
            Dict of symbol -> PriceSnapshot
        """
        if self._exchange is None:
            await self.start()

        results: dict[str, PriceSnapshot] = {}

        # Fetch in parallel
        tasks = [self.fetch_price(symbol) for symbol in self.symbols]
        snapshots = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, snapshot in zip(self.symbols, snapshots):
            if isinstance(snapshot, PriceSnapshot):
                results[symbol] = snapshot
            elif isinstance(snapshot, Exception):
                logger.error("fetch_price_error", symbol=symbol, error=str(snapshot))

        return results

    def get_price(self, symbol: str) -> float | None:
        """
        Get latest cached price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Latest price or None
        """
        if symbol in self._history:
            return self._history[symbol].latest_price
        return None

    def get_history(self, symbol: str) -> PriceHistory | None:
        """
        Get price history for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            PriceHistory or None
        """
        return self._history.get(symbol)

    def get_prices(self, symbol: str, n: int | None = None) -> list[float]:
        """
        Get recent prices for a symbol.

        Args:
            symbol: Trading pair
            n: Number of prices (None for all)

        Returns:
            List of prices
        """
        if symbol not in self._history:
            return []
        return self._history[symbol].get_prices(n)

    def get_returns(self, symbol: str, n: int | None = None) -> list[float]:
        """
        Get log returns for a symbol.

        Args:
            symbol: Trading pair
            n: Number of returns (None for all)

        Returns:
            List of log returns
        """
        if symbol not in self._history:
            return []
        return self._history[symbol].get_returns(n)

    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self._history[symbol] = PriceHistory(
                symbol=symbol,
                prices=deque(maxlen=self.max_history),
                max_size=self.max_history,
            )

    async def __aenter__(self) -> "PriceFeed":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
