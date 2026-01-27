"""Polymarket CLOB client wrapper with retry logic."""

import asyncio
from datetime import datetime
from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OpenOrderParams,
    OrderArgs,
    OrderType as ClobOrderType,
    PartialCreateOrderOptions,
)
from py_clob_client.order_builder.constants import BUY, SELL

from src.common.errors import NetworkError, OrderError
from src.common.logging import get_logger
from .types import OrderBook, OrderBookLevel, Order, Fill, OrderSide, OrderType, OrderStatus

logger = get_logger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# Signature types
SIGNATURE_TYPE_EOA = 0  # Standard EOA (MetaMask, hardware wallets)
SIGNATURE_TYPE_POLY_PROXY = 1  # Email/Magic wallet signatures
SIGNATURE_TYPE_POLY_GNOSIS_SAFE = 2  # Browser wallet proxy


class PolymarketClient:
    """
    Async wrapper for Polymarket CLOB client.

    Provides:
    - Order book fetching
    - Order placement and cancellation
    - Fill tracking
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        host: str,
        chain_id: int,
        private_key: str,
        funder: str | None = None,
        signature_type: int = SIGNATURE_TYPE_EOA,
    ) -> None:
        """
        Initialize Polymarket client.

        Args:
            host: CLOB API host URL
            chain_id: Polygon chain ID (137 for mainnet)
            private_key: Wallet private key for signing
            funder: Funder address (for proxy wallets). If None, uses signer address.
            signature_type: Signature type (0=EOA, 1=Poly Proxy, 2=Gnosis Safe)
        """
        self.host = host
        self.chain_id = chain_id

        # Initialize py-clob-client with signature type and funder
        self._client = ClobClient(
            host=host,
            chain_id=chain_id,
            key=private_key,
            signature_type=signature_type,
            funder=funder,
        )

        # Derive or create API credentials for L2 authentication
        # This is required for placing orders
        try:
            api_creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(api_creds)
            logger.info("api_credentials_set", api_key=api_creds.api_key[:8] + "...")
        except Exception as e:
            logger.warning("api_credentials_failed", error=str(e))
            # Continue without L2 auth - read-only operations will still work

        self._order_cache: dict[str, Order] = {}
        self._tick_size_cache: dict[str, str] = {}
        self._neg_risk_cache: dict[str, bool] = {}

        logger.info(
            "polymarket_client_initialized",
            host=host,
            chain_id=chain_id,
            signature_type=signature_type,
        )

    async def _retry_async(
        self,
        operation: str,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with retry logic.

        Uses asyncio.to_thread for sync py-clob-client calls.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                # Run sync function in thread pool
                result = await asyncio.to_thread(func, *args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                delay = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                # Only log retries at debug level to reduce noise
                logger.debug(
                    "operation_retry",
                    operation=operation,
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    delay=delay,
                    error=str(e),
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)

        raise NetworkError(
            f"Operation {operation} failed after {MAX_RETRIES} retries",
            endpoint=self.host,
            details={"error": str(last_error)},
        )

    async def _get_tick_size(self, token_id: str) -> str:
        """Get tick size for a token (cached)."""
        if token_id not in self._tick_size_cache:
            tick_size = await self._retry_async(
                "get_tick_size",
                self._client.get_tick_size,
                token_id,
            )
            self._tick_size_cache[token_id] = tick_size
        return self._tick_size_cache[token_id]

    async def _get_neg_risk(self, token_id: str) -> bool:
        """Get neg_risk flag for a token (cached)."""
        if token_id not in self._neg_risk_cache:
            neg_risk = await self._retry_async(
                "get_neg_risk",
                self._client.get_neg_risk,
                token_id,
            )
            self._neg_risk_cache[token_id] = neg_risk
        return self._neg_risk_cache[token_id]

    def _round_to_tick(self, price: float, tick_size: str) -> float:
        """Round price to the nearest tick size."""
        tick = float(tick_size)
        return round(price / tick) * tick

    async def get_order_book(self, token_id: str) -> OrderBook:
        """
        Fetch L2 order book for a token.

        Args:
            token_id: Token ID to fetch book for

        Returns:
            OrderBook with bids and asks
        """
        try:
            raw_book = await self._retry_async(
                "get_order_book",
                self._client.get_order_book,
                token_id,
            )

            # Handle both dict and object response types
            if hasattr(raw_book, 'bids'):
                raw_bids = raw_book.bids or []
                raw_asks = raw_book.asks or []
            else:
                raw_bids = raw_book.get("bids", [])
                raw_asks = raw_book.get("asks", [])

            # Parse bids (sorted descending by price)
            bids = []
            for level in raw_bids:
                # Handle both dict and object level types
                if hasattr(level, 'price'):
                    price = float(level.price)
                    size = float(level.size)
                else:
                    price = float(level["price"])
                    size = float(level["size"])
                bids.append(OrderBookLevel(price=price, size=size))
            bids.sort(key=lambda x: x.price, reverse=True)

            # Parse asks (sorted ascending by price)
            asks = []
            for level in raw_asks:
                if hasattr(level, 'price'):
                    price = float(level.price)
                    size = float(level.size)
                else:
                    price = float(level["price"])
                    size = float(level["size"])
                asks.append(OrderBookLevel(price=price, size=size))
            asks.sort(key=lambda x: x.price)

            book = OrderBook(
                token_id=token_id,
                bids=bids,
                asks=asks,
                timestamp=datetime.utcnow(),
            )

            logger.debug(
                "order_book_fetched",
                token_id=token_id[:16] + "...",
                bid_levels=len(bids),
                ask_levels=len(asks),
                best_bid=book.best_bid_price,
                best_ask=book.best_ask_price,
            )

            return book

        except NetworkError:
            raise
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch order book: {e}",
                endpoint=f"{self.host}/book",
                details={"token_id": token_id},
            )

    async def place_limit_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.GTC,
        expiration: int | None = None,
    ) -> Order:
        """
        Place a limit order.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price (probability 0-1)
            size: Order size
            order_type: Order type (GTC, GTD, FOK)
            expiration: Expiration timestamp for GTD orders

        Returns:
            Created Order object
        """
        try:
            # Convert to py-clob-client types
            clob_side = BUY if side == OrderSide.BUY else SELL

            # Map order type
            if order_type == OrderType.FOK:
                clob_type = ClobOrderType.FOK
            elif order_type == OrderType.GTD:
                clob_type = ClobOrderType.GTD
            else:
                clob_type = ClobOrderType.GTC

            # Get market-specific tick size and neg_risk flag
            tick_size_str = await self._get_tick_size(token_id)
            neg_risk = await self._get_neg_risk(token_id)

            # Round price to tick size
            price = self._round_to_tick(price, tick_size_str)
            size = round(size, 2)

            logger.debug(
                "placing_order",
                token_id=token_id[:20] + "...",
                side=side.value,
                price=price,
                size=size,
                tick_size=tick_size_str,
                neg_risk=neg_risk,
            )

            # Build order args
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=clob_side,
                expiration=expiration if expiration else 0,
            )

            # Create options with tick_size and neg_risk flag
            create_options = PartialCreateOrderOptions(
                tick_size=tick_size_str,
                neg_risk=neg_risk,
            )

            # Create and sign order
            signed_order = await self._retry_async(
                "create_order",
                self._client.create_order,
                order_args,
                create_options,
            )

            # Post order to CLOB
            response = await self._retry_async(
                "post_order",
                self._client.post_order,
                signed_order,
                clob_type,
            )

            order_id = response.get("orderID", response.get("id", ""))

            order = Order(
                id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                original_size=size,
                order_type=order_type,
                status=OrderStatus.LIVE,
                created_at=datetime.utcnow(),
            )

            self._order_cache[order_id] = order

            logger.debug(
                "order_placed",
                order_id=order_id[:16] + "..." if order_id else "unknown",
                side=side.value,
                price=round(price, 4),
                size=size,
            )

            return order

        except NetworkError:
            raise
        except Exception as e:
            raise OrderError(
                f"Failed to place order: {e}",
                token_id=token_id,
                details={"side": side.value, "price": price, "size": size},
            )

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            await self._retry_async(
                "cancel_order",
                self._client.cancel,
                order_id,
            )

            # Update cache
            if order_id in self._order_cache:
                self._order_cache[order_id].status = OrderStatus.CANCELLED

            logger.debug("order_cancelled", order_id=order_id[:16] + "...")
            return True

        except Exception as e:
            logger.error("cancel_order_failed", order_id=order_id[:16] + "...", error=str(e))
            return False

    async def cancel_all_orders(self, token_id: str | None = None) -> int:
        """
        Cancel all open orders, optionally filtered by token.

        Args:
            token_id: Optional token to filter by

        Returns:
            Number of orders cancelled
        """
        try:
            if token_id:
                await self._retry_async(
                    "cancel_market_orders",
                    self._client.cancel_market_orders,
                    token_id,
                )
            else:
                await self._retry_async(
                    "cancel_all",
                    self._client.cancel_all,
                )

            # Update cache
            count = 0
            for order in self._order_cache.values():
                if order.status == OrderStatus.LIVE:
                    if token_id is None or order.token_id == token_id:
                        order.status = OrderStatus.CANCELLED
                        count += 1

            logger.info("orders_cancelled", count=count, token_id=token_id)
            return count

        except Exception as e:
            logger.error("cancel_all_failed", error=str(e))
            return 0

    async def get_open_orders(self, token_id: str | None = None) -> list[Order]:
        """
        Get all open orders.

        Args:
            token_id: Optional token to filter by

        Returns:
            List of open orders
        """
        try:
            # Build params for filtering
            params = OpenOrderParams(asset_id=token_id) if token_id else OpenOrderParams()

            raw_orders = await self._retry_async(
                "get_orders",
                self._client.get_orders,
                params,
            )

            orders = []
            for raw in raw_orders:
                side = OrderSide.BUY if raw.get("side") == "BUY" else OrderSide.SELL

                # Parse created_at - can be string or int timestamp
                created_at = datetime.utcnow()
                if "created_at" in raw:
                    ts = raw["created_at"]
                    if isinstance(ts, str):
                        created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    elif isinstance(ts, (int, float)):
                        created_at = datetime.utcfromtimestamp(ts / 1000 if ts > 1e12 else ts)

                order = Order(
                    id=raw["id"],
                    token_id=raw["asset_id"],
                    side=side,
                    price=float(raw["price"]),
                    size=float(raw.get("size_matched", 0)),
                    original_size=float(raw.get("original_size", raw.get("size", 0))),
                    status=OrderStatus.LIVE,
                    created_at=created_at,
                )
                orders.append(order)
                self._order_cache[order.id] = order

            return orders

        except Exception as e:
            logger.error("get_orders_failed", error=str(e))
            return []

    async def get_fills(
        self,
        token_id: str | None = None,
        since: datetime | None = None,
    ) -> list[Fill]:
        """
        Get recent fills/trades.

        Args:
            token_id: Optional token to filter by
            since: Optional timestamp to filter by

        Returns:
            List of fills
        """
        try:
            raw_fills = await self._retry_async(
                "get_trades",
                self._client.get_trades,
            )

            fills = []
            for raw in raw_fills:
                # Filter by token if specified
                if token_id and raw.get("asset_id") != token_id:
                    continue

                # Parse timestamp - can be ISO string or Unix timestamp
                created_at = datetime.utcnow()
                if "match_time" in raw:
                    match_time = raw["match_time"]
                    try:
                        if isinstance(match_time, (int, float)):
                            # Unix timestamp (seconds or milliseconds)
                            ts = match_time / 1000 if match_time > 1e12 else match_time
                            created_at = datetime.utcfromtimestamp(ts)
                        elif isinstance(match_time, str):
                            # Could be Unix timestamp as string or ISO format
                            if match_time.isdigit() or (match_time.replace('.', '', 1).isdigit()):
                                ts = float(match_time)
                                ts = ts / 1000 if ts > 1e12 else ts
                                created_at = datetime.utcfromtimestamp(ts)
                            else:
                                created_at = datetime.fromisoformat(
                                    match_time.replace("Z", "+00:00")
                                ).replace(tzinfo=None)
                    except (ValueError, OSError) as e:
                        logger.debug("match_time_parse_failed", match_time=match_time, error=str(e))
                        created_at = datetime.utcnow()

                # Filter by time if specified
                if since and created_at < since:
                    continue

                side = OrderSide.BUY if raw.get("side") == "BUY" else OrderSide.SELL
                fill = Fill(
                    id=raw.get("id", ""),
                    order_id=raw.get("maker_order_id", ""),
                    token_id=raw.get("asset_id", ""),
                    side=side,
                    price=float(raw.get("price", 0)),
                    size=float(raw.get("size", 0)),
                    fee=float(raw.get("fee", 0)),
                    timestamp=created_at,
                    tx_hash=raw.get("transaction_hash"),
                    taker_order_id=raw.get("taker_order_id"),
                )
                fills.append(fill)

            return fills

        except Exception as e:
            logger.error("get_fills_failed", error=str(e))
            return []

    async def get_balances(self) -> dict[str, float]:
        """
        Get token balances.

        Returns:
            Dict of token_id -> balance
        """
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        
        try:
            # Must pass params with signature_type for the API call to work
            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                signature_type=self._signature_type,
            )
            raw_balances = await self._retry_async(
                "get_balance_allowance",
                self._client.get_balance_allowance,
                params,
            )

            balances = {}
            if raw_balances:
                for item in raw_balances if isinstance(raw_balances, list) else [raw_balances]:
                    token_id = item.get("asset_id", item.get("token_id", ""))
                    balance = float(item.get("balance", 0))
                    if token_id:
                        balances[token_id] = balance

            return balances

        except Exception as e:
            logger.error("get_balances_failed", error=str(e))
            return {}

    async def get_usdc_balance(self) -> float:
        """
        Get USDC (collateral) balance available for trading.

        Tries multiple methods (in order of reliability):
        1. On-chain RPC query (most reliable, no auth needed)
        2. Polymarket Gamma API profile endpoint
        3. CLOB API balance/allowance

        Returns:
            USDC balance as float, or 0.0 if failed to fetch
        """
        import aiohttp

        # Method 1: Try on-chain query via RPC (most reliable, no retries needed)
        try:
            # USDC.e on Polygon: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
            usdc_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            rpc_url = "https://polygon-rpc.com"

            # ERC20 balanceOf call
            # balanceOf(address) = 0x70a08231 + padded address
            padded_address = self.address.lower().replace("0x", "").zfill(64)
            data = f"0x70a08231{padded_address}"

            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": usdc_address, "data": data}, "latest"],
                "id": 1,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        hex_balance = result.get("result", "0x0")
                        # USDC has 6 decimals
                        balance = int(hex_balance, 16) / 1e6
                        if balance >= 0:  # Allow 0 balance as valid
                            logger.debug("usdc_balance_fetched", source="on_chain", balance=round(balance, 2))
                            return balance
        except Exception as e:
            logger.debug("on_chain_balance_failed", error=str(e))

        # Method 2: Try Gamma API profile endpoint
        try:
            url = f"https://gamma-api.polymarket.com/profiles/{self.address.lower()}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Profile has collateralBalance field
                        balance = float(data.get("collateralBalance", 0))
                        if balance >= 0:
                            logger.debug("usdc_balance_fetched", source="gamma_api", balance=round(balance, 2))
                            return balance
        except Exception as e:
            logger.debug("gamma_api_balance_failed", error=str(e))

        # Method 3: Try CLOB API balance_allowance (requires proper params)
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            
            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=self._signature_type,
            )
            # Don't use retry here - if on-chain and gamma failed, CLOB likely will too
            result = self._client.get_balance_allowance(params)
            if result:
                # Result might be a dict or list
                if isinstance(result, dict):
                    balance = float(result.get("balance", 0))
                elif isinstance(result, list) and len(result) > 0:
                    # First item is usually USDC collateral
                    balance = float(result[0].get("balance", 0))
                else:
                    balance = 0.0

                if balance >= 0:
                    logger.debug("usdc_balance_fetched", source="clob_api", balance=round(balance, 2))
                    return balance
        except Exception as e:
            logger.debug("clob_api_balance_failed", error=str(e))

        logger.warning("usdc_balance_unavailable", message="Could not fetch balance from any source")
        return 0.0

    @property
    def address(self) -> str:
        """Get wallet address."""
        return self._client.get_address()

    async def get_positions_from_data_api(self) -> list[dict]:
        """
        Fetch positions from Polymarket Data API.

        This is the source of truth for actual positions, as it queries
        the blockchain state directly rather than relying on fill tracking.

        Returns:
            List of position dicts with token_id, size, avg_price, current_value, etc.
        """
        import aiohttp

        url = "https://data-api.polymarket.com/positions"
        params = {
            "user": self.address.lower(),
            "sizeThreshold": 0,  # Get all positions including small ones
            "limit": 500,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning("data_api_positions_failed", status=resp.status)
                        return []

                    data = await resp.json()

                    positions = []
                    for pos in data:
                        size = float(pos.get("size", 0))
                        if size > 0.01:  # Ignore dust
                            cur_price = float(pos.get("curPrice", 0.5))
                            avg_price = float(pos.get("avgPrice", cur_price))
                            current_value = float(pos.get("currentValue", size * cur_price))
                            initial_value = float(pos.get("initialValue", size * avg_price))

                            positions.append({
                                "token_id": pos.get("asset", ""),
                                "condition_id": pos.get("conditionId", ""),
                                "size": size,
                                "avg_price": avg_price,
                                "cur_price": cur_price,
                                "current_value": current_value,
                                "initial_value": initial_value,
                                "unrealized_pnl": current_value - initial_value,
                                "title": pos.get("title", "unknown"),
                                "outcome": pos.get("outcome", "unknown"),
                                "neg_risk": pos.get("negativeRisk", False),
                                "is_resolved_winning": cur_price >= 0.99,
                                "is_resolved_losing": cur_price <= 0.01,
                            })

                    logger.debug(
                        "data_api_positions_fetched",
                        count=len(positions),
                        total_value=round(sum(p["current_value"] for p in positions), 2),
                    )
                    return positions

        except Exception as e:
            logger.warning("data_api_positions_error", error=str(e))
            return []
