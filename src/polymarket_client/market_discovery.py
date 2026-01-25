"""Market discovery for finding active markets dynamically."""

import asyncio
import json as json_module
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import aiohttp

from src.common.logging import get_logger

logger = get_logger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# BTC Up/Down market intervals and their slug patterns
# Different intervals use different slug formats:
# - 15m: btc-updown-15m-{unix_timestamp}
# - 1h: bitcoin-up-or-down-{month}-{day}-{hour}{am/pm}-et (human-readable)
BTC_UPDOWN_INTERVALS = {
    "15m": {"seconds": 900, "slug_type": "timestamp", "slug_prefix": "btc-updown-15m"},
    "1h": {"seconds": 3600, "slug_type": "human_readable"},
    "4h": {"seconds": 14400, "slug_type": "timestamp", "slug_prefix": "btc-updown-4h"},
    "1d": {"seconds": 86400, "slug_type": "timestamp", "slug_prefix": "btc-updown-1d"},
    "1w": {"seconds": 604800, "slug_type": "timestamp", "slug_prefix": "btc-updown-1w"},
    "1M": {"seconds": 2592000, "slug_type": "timestamp", "slug_prefix": "btc-updown-1M"},  # ~30 days
}

BTCInterval = Literal["15m", "1h", "4h", "1d", "1w", "1M"]


def _get_et_timezone():
    """Get Eastern Time timezone."""
    try:
        import pytz
        return pytz.timezone("America/New_York")
    except ImportError:
        # Fallback: use fixed UTC-5 offset (doesn't handle DST)
        return timezone(timedelta(hours=-5))


def _generate_1h_slug(dt_et) -> str:
    """
    Generate human-readable slug for 1h BTC market.
    
    Format: bitcoin-up-or-down-{month}-{day}-{hour}{am/pm}-et
    Example: bitcoin-up-or-down-january-25-3pm-et
    """
    month = dt_et.strftime("%B").lower()
    day = dt_et.day
    hour = dt_et.hour
    
    # Format hour as 12-hour with am/pm
    if hour == 0:
        hour_str = "12am"
    elif hour < 12:
        hour_str = f"{hour}am"
    elif hour == 12:
        hour_str = "12pm"
    else:
        hour_str = f"{hour - 12}pm"
    
    return f"bitcoin-up-or-down-{month}-{day}-{hour_str}-et"


@dataclass
class DiscoveredMarket:
    """A discovered market from the API."""

    condition_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    description: str
    end_date: datetime | None
    slug: str
    active: bool
    market_id: str


class MarketDiscovery:
    """
    Discovers and tracks active markets on Polymarket.

    Useful for markets that rotate (e.g., hourly BTC up/down).
    """

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        """
        Initialize market discovery.

        Args:
            session: Optional aiohttp session (will create one if not provided)
        """
        self._session = session
        self._owns_session = session is None
        self._cache: dict[str, DiscoveredMarket] = {}
        self._cache_time: dict[str, datetime] = {}
        self._cache_ttl = 60  # Cache for 60 seconds

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the session if we own it."""
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    async def find_btc_updown_market(
        self,
        interval: BTCInterval = "1h",
        look_ahead_windows: int = 3,
    ) -> DiscoveredMarket | None:
        """
        Find a BTC Up/Down short-term market by constructing time-based slugs.

        Different intervals use different slug patterns:
        - 15m: btc-updown-15m-{unix_timestamp}
        - 1h: bitcoin-up-or-down-{month}-{day}-{hour}{am/pm}-et

        Args:
            interval: Market interval - "15m", "1h", "4h", "1d", "1w", or "1M"
            look_ahead_windows: Number of future windows to check

        Returns:
            DiscoveredMarket or None if not found
        """
        if interval not in BTC_UPDOWN_INTERVALS:
            logger.error("invalid_btc_interval", interval=interval)
            return None

        interval_config = BTC_UPDOWN_INTERVALS[interval]
        interval_seconds = interval_config["seconds"]
        slug_type = interval_config.get("slug_type", "timestamp")

        slugs_to_try = []

        if slug_type == "human_readable" and interval == "1h":
            # 1h markets use human-readable slugs based on ET time
            # Format: bitcoin-up-or-down-{month}-{day}-{hour}{am/pm}-et
            et_tz = _get_et_timezone()
            now_et = datetime.now(et_tz)
            
            # Generate slugs for current hour and upcoming hours
            for hour_offset in range(-1, look_ahead_windows + 1):
                try:
                    check_time = now_et + timedelta(hours=hour_offset)
                    slug = _generate_1h_slug(check_time)
                    if slug not in slugs_to_try:
                        slugs_to_try.append(slug)
                except Exception:
                    continue
        else:
            # Timestamp-based slugs (15m, 4h, etc.)
            slug_prefix = interval_config.get("slug_prefix", f"btc-updown-{interval}")
            current_ts = int(time.time())
            
            # Calculate window boundaries to check (current + upcoming)
            base_boundary = (current_ts // interval_seconds) * interval_seconds
            
            for i in range(-1, look_ahead_windows):
                window_ts = base_boundary + (i * interval_seconds)
                slug = f"{slug_prefix}-{window_ts}"
                slugs_to_try.append(slug)

        logger.debug(
            "btc_updown_slugs",
            interval=interval,
            slug_type=slug_type,
            slugs=slugs_to_try,
        )

        # Try each slug and return the first active, non-ended market
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        for slug in slugs_to_try:
            market = await self.get_market_by_slug(slug)
            if market and market.active and market.end_date and market.end_date > now:
                logger.info(
                    "btc_updown_market_found",
                    interval=interval,
                    slug=slug,
                    question=market.question[:50],
                    end_date=market.end_date.isoformat(),
                )
                return market

        logger.warning("no_btc_updown_market_found", interval=interval, slugs_tried=slugs_to_try)
        return None

    async def find_btc_market(self, interval: BTCInterval | None = "1h") -> DiscoveredMarket | None:
        """
        Find an active BTC/Bitcoin market.

        First tries to find BTC Up/Down short-term markets by slug pattern,
        then falls back to keyword search.

        Args:
            interval: Preferred interval for BTC Up/Down markets ("5m", "15m", "4h")
                     Set to None to skip short-term market search

        Returns:
            DiscoveredMarket or None if not found
        """
        # First, try to find BTC Up/Down short-term markets
        if interval:
            market = await self.find_btc_updown_market(interval=interval)
            if market:
                return market
            
            # Do NOT fallback to other intervals or keyword search
            # The user requested a specific interval type (e.g., "1h" up/down markets)
            # Falling back to other markets would give unexpected behavior
            logger.warning(
                "requested_interval_not_found",
                interval=interval,
                message="No market found for requested interval. Bot will not trade until this interval becomes available.",
            )
            return None

        # Only reach here if interval=None (no specific interval requested)
        # Fallback: search by keyword for any BTC market
        search_terms = ["bitcoin", "btc"]

        for term in search_terms:
            markets = await self.search_active_markets(term)
            if markets:
                # Filter for actually active, non-closed markets
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                active_markets = [
                    m for m in markets
                    if m.active and m.end_date and m.end_date > now
                ]

                if active_markets:
                    # Sort by end_date (soonest first for short-term trading)
                    active_markets.sort(key=lambda m: m.end_date or datetime.max)
                    selected = active_markets[0]

                    logger.info(
                        "btc_market_selected",
                        question=selected.question[:50],
                        end_date=selected.end_date.isoformat() if selected.end_date else None,
                        condition_id=selected.condition_id[:16] + "...",
                    )
                    return selected

        logger.warning("no_btc_markets_found")
        return None

    # Keep old method name as alias for backwards compatibility
    async def find_btc_hourly_market(self) -> DiscoveredMarket | None:
        """Alias for find_btc_market (uses 1h interval by default)."""
        return await self.find_btc_market(interval="1h")

    async def search_active_markets(self, query: str, limit: int = 100) -> list[DiscoveredMarket]:
        """
        Search for active markets using the Gamma markets API.

        Args:
            query: Search query (searches question field)
            limit: Maximum results

        Returns:
            List of discovered markets
        """
        session = await self._get_session()

        try:
            # Use /markets endpoint with active and closed filters
            url = f"{GAMMA_API_BASE}/markets"
            params = {
                "limit": limit,
                "active": "true",
                "closed": "false",
            }

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error("market_search_failed", status=resp.status)
                    return []

                all_markets = await resp.json()

            # Filter by query in question
            markets = []
            query_lower = query.lower()

            for market_data in all_markets:
                question = market_data.get("question", "").lower()
                if query_lower in question:
                    market = self._parse_market_from_api(market_data)
                    if market:
                        markets.append(market)

            logger.debug("active_markets_found", query=query, count=len(markets))
            return markets

        except Exception as e:
            logger.error("market_search_error", error=str(e))
            return []

    def _parse_market_from_api(self, market_data: dict) -> DiscoveredMarket | None:
        """Parse market from Gamma /markets API response."""
        import json as json_module
        
        try:
            condition_id = market_data.get("conditionId", "")
            tokens_raw = market_data.get("clobTokenIds", [])
            
            # clobTokenIds can be a JSON string or a list
            if isinstance(tokens_raw, str):
                tokens = json_module.loads(tokens_raw)
            else:
                tokens = tokens_raw

            if not condition_id or len(tokens) < 2:
                return None

            end_date = None
            if market_data.get("endDate"):
                try:
                    end_date = datetime.fromisoformat(
                        market_data["endDate"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except:
                    pass

            return DiscoveredMarket(
                condition_id=condition_id,
                token_id_yes=tokens[0],
                token_id_no=tokens[1],
                question=market_data.get("question", ""),
                description=market_data.get("description", ""),
                end_date=end_date,
                slug=market_data.get("slug", ""),
                active=market_data.get("active", True),
                market_id=str(market_data.get("id", "")),
            )

        except Exception as e:
            logger.debug("parse_market_error", error=str(e))
            return None

    async def search_markets(self, query: str, limit: int = 20) -> list[DiscoveredMarket]:
        """
        Search for markets by query string.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of discovered markets
        """
        session = await self._get_session()

        try:
            url = f"{GAMMA_API_BASE}/events"
            params = {"limit": limit, "active": "true"}

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error("market_search_failed", status=resp.status)
                    return []

                events = await resp.json()

            markets = []
            query_lower = query.lower()

            for event in events:
                title = event.get("title", "").lower()
                if query_lower in title or any(
                    query_lower in m.get("question", "").lower()
                    for m in event.get("markets", [])
                ):
                    # Parse markets from event
                    for market_data in event.get("markets", []):
                        market = self._parse_market(event, market_data)
                        if market:
                            markets.append(market)

            logger.debug("markets_found", query=query, count=len(markets))
            return markets

        except Exception as e:
            logger.error("market_search_error", error=str(e))
            return []

    async def get_market_by_slug(self, slug: str) -> DiscoveredMarket | None:
        """
        Get market by event slug.

        Args:
            slug: Event slug from URL

        Returns:
            DiscoveredMarket or None
        """
        # Check cache
        cache_key = f"slug:{slug}"
        if cache_key in self._cache:
            cache_age = (datetime.utcnow() - self._cache_time.get(cache_key, datetime.min)).total_seconds()
            if cache_age < self._cache_ttl:
                return self._cache[cache_key]

        session = await self._get_session()

        try:
            url = f"{GAMMA_API_BASE}/events"
            params = {"slug": slug}

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error("get_market_failed", status=resp.status, slug=slug)
                    return None

                events = await resp.json()

            if not events:
                return None

            event = events[0]
            market_data = event.get("markets", [{}])[0]
            market = self._parse_market(event, market_data)

            if market:
                self._cache[cache_key] = market
                self._cache_time[cache_key] = datetime.utcnow()

            return market

        except Exception as e:
            logger.error("get_market_error", error=str(e), slug=slug)
            return None

    async def get_market_by_condition(self, condition_id: str) -> DiscoveredMarket | None:
        """
        Get market by condition ID.

        Args:
            condition_id: Condition ID

        Returns:
            DiscoveredMarket or None
        """
        session = await self._get_session()

        try:
            url = f"{GAMMA_API_BASE}/markets"
            params = {"condition_id": condition_id}

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None

                markets = await resp.json()

            if not markets:
                return None

            market_data = markets[0]
            return self._parse_market_direct(market_data)

        except Exception as e:
            logger.error("get_market_error", error=str(e))
            return None

    def _parse_market(self, event: dict, market_data: dict) -> DiscoveredMarket | None:
        """Parse market from event and market data."""
        try:
            condition_id = market_data.get("conditionId", "")
            tokens_raw = market_data.get("clobTokenIds", [])
            
            # clobTokenIds can be a JSON string or a list
            if isinstance(tokens_raw, str):
                tokens = json_module.loads(tokens_raw)
            else:
                tokens = tokens_raw

            if not condition_id or len(tokens) < 2:
                return None

            end_date = None
            if market_data.get("endDate"):
                try:
                    end_date = datetime.fromisoformat(
                        market_data["endDate"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except:
                    pass

            return DiscoveredMarket(
                condition_id=condition_id,
                token_id_yes=tokens[0],
                token_id_no=tokens[1],
                question=market_data.get("question", event.get("title", "")),
                description=market_data.get("description", ""),
                end_date=end_date,
                slug=event.get("slug", ""),
                active=market_data.get("active", True),
                market_id=str(market_data.get("id", "")),
            )

        except Exception as e:
            logger.debug("parse_market_error", error=str(e))
            return None

    def _parse_market_direct(self, market_data: dict) -> DiscoveredMarket | None:
        """Parse market from direct market API response."""
        try:
            condition_id = market_data.get("condition_id", "")
            tokens = market_data.get("tokens", [])

            if not condition_id:
                return None

            # Extract token IDs
            token_yes = ""
            token_no = ""
            for token in tokens:
                if token.get("outcome") == "Yes":
                    token_yes = token.get("token_id", "")
                elif token.get("outcome") == "No":
                    token_no = token.get("token_id", "")

            if not token_yes or not token_no:
                # Try from clobTokenIds
                clob_tokens = market_data.get("clobTokenIds", [])
                if len(clob_tokens) >= 2:
                    token_yes = clob_tokens[0]
                    token_no = clob_tokens[1]

            end_date = None
            if market_data.get("end_date_iso"):
                try:
                    end_date = datetime.fromisoformat(
                        market_data["end_date_iso"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except:
                    pass

            return DiscoveredMarket(
                condition_id=condition_id,
                token_id_yes=token_yes,
                token_id_no=token_no,
                question=market_data.get("question", ""),
                description=market_data.get("description", ""),
                end_date=end_date,
                slug="",
                active=market_data.get("active", True),
                market_id=str(market_data.get("id", "")),
            )

        except Exception as e:
            logger.debug("parse_market_error", error=str(e))
            return None


async def get_current_btc_market() -> DiscoveredMarket | None:
    """
    Convenience function to get the current BTC hourly market.

    Returns:
        DiscoveredMarket or None
    """
    discovery = MarketDiscovery()
    try:
        return await discovery.find_btc_hourly_market()
    finally:
        await discovery.close()
