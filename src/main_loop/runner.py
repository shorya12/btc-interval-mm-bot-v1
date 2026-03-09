"""Main trading loop."""

import asyncio
import signal
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.common.config import Config, load_config, load_env_settings
from src.common.logging import setup_logging, get_logger
from src.persistence import Database, Repository
from src.persistence.models import EventSeverity
from src.belief_state import BeliefManager
from src.quoting import QuoteCalculator, QuoteContext
from src.lag_signal import PriceFeed, LagModel, SkewComputer, AssetConfig
from src.risk import RiskManager, PositionTracker
from src.polymarket_client import PolymarketClient, OrderBookManager, OrderManager, FillTracker, MarketDiscovery, DiscoveredMarket, close_all_positions
from src.polymarket_client.types import OrderSide, OrderBook, OrderBookLevel
from src.common.config import MarketConfig
from src.probability_model.model_adapter import ModelAdapter
from .dry_run import DryRunAdapter

import random

logger = get_logger(__name__)


@dataclass
class TickResult:
    """Result of a single tick."""

    tick_number: int
    timestamp: datetime
    mid_price: float | None
    spread_bps: float | None
    bid_placed: bool
    ask_placed: bool
    orders_cancelled: int
    vetoed: bool
    stop_triggered: bool
    error: str | None = None


class TradingLoop:
    """
    Main trading loop for market making.

    Orchestrates all components:
    - Order book fetching
    - Belief state estimation
    - Lag signal computation
    - Risk evaluation
    - Quote calculation
    - Order management
    """

    def __init__(
        self,
        config: Config,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize trading loop.

        Args:
            config: Bot configuration
            dry_run: Enable paper trading mode
        """
        self.config = config
        self.dry_run = dry_run

        # State
        self._running = False
        self._tick_count = 0
        self._start_time: datetime | None = None

        # Components (initialized in start())
        self.db: Database | None = None
        self.repo: Repository | None = None
        self.client: PolymarketClient | DryRunAdapter | None = None
        self.price_feed: PriceFeed | None = None
        self.lag_model: LagModel | None = None
        self.skew_computer: SkewComputer | None = None
        self.risk_manager: RiskManager | None = None
        self.quote_calculator: QuoteCalculator | None = None
        self.market_discovery: MarketDiscovery | None = None

        # Per-market components
        self.belief_managers: dict[str, BeliefManager] = {}
        self.orderbook_managers: dict[str, OrderBookManager] = {}
        self.order_managers: dict[str, OrderManager] = {}
        self.fill_trackers: dict[str, FillTracker] = {}

        # Position tracking for live mode
        self.position_tracker: PositionTracker | None = None

        # Active markets (may be updated by discovery)
        self.active_markets: list[MarketConfig] = []
        self._last_market_refresh: datetime | None = None
        self._market_refresh_interval = 300  # Refresh every 5 minutes
        self._last_status_log: datetime | None = None
        self._status_log_interval = 10  # Log status every 10 seconds
        
        # Market end dates - track when each market expires
        self._market_end_dates: dict[str, datetime] = {}
        self._expiry_buffer_seconds = self.config.risk.min_time_to_expiry_seconds  # Buffer before expiry to stop trading
        
        # Invalid orderbook tracking - triggers market refresh
        self._invalid_orderbook_counts: dict[str, int] = {}
        self._max_invalid_orderbook_count = 10  # Refresh market after 10 consecutive invalid orderbooks

        # Balance tracking for live mode
        self._last_balance_check: datetime | None = None
        self._balance_check_interval = 30  # Check balance every 30 seconds
        self._usdc_balance: float = 0.0
        self._min_balance_to_trade = self.config.risk.min_balance_to_trade

        # Data API position tracking (source of truth for actual positions)
        self._last_position_sync: datetime | None = None
        self._position_sync_interval = 5  # Sync positions every 5 seconds
        self._data_api_positions: list[dict] = []  # Positions from Data API

        # ML probability model adapter
        self.model_adapter: ModelAdapter | None = None
        self._last_retrain_time: datetime | None = None
        self._retrain_bss_history: list[float] = []  # Rolling BSS for regime detection
        self._retraining_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Initialize all components and start trading."""
        logger.info("trading_loop_starting", dry_run=self.dry_run)

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        # Initialize database
        self.db = Database(self.config.database.path)
        await self.db.connect()
        self.repo = Repository(self.db)

        # Log startup event
        await self.repo.log_event(
            event_type="STARTUP",
            message=f"Trading loop started (dry_run={self.dry_run})",
            severity=EventSeverity.INFO,
        )

        # Initialize client
        if self.dry_run:
            self.client = DryRunAdapter(
                initial_balance=self.config.dry_run.initial_balance,
                fill_rate=self.config.dry_run.simulated_fill_rate,
            )
        else:
            env = load_env_settings()
            self.client = PolymarketClient(
                host=self.config.network.clob_host,
                chain_id=self.config.network.chain_id,
                private_key=env.private_key,
                funder=env.funder_address if env.funder_address else None,
                signature_type=env.signature_type,
            )
            
            # Fetch actual USDC balance
            usdc_balance = await self.client.get_usdc_balance()
            self._usdc_balance = usdc_balance
            self._last_balance_check = datetime.utcnow()

            if usdc_balance < self._min_balance_to_trade:
                logger.error(
                    "insufficient_usdc_balance",
                    balance=usdc_balance,
                    min_required=self._min_balance_to_trade,
                    message="USDC balance too low to trade.",
                )
                raise RuntimeError(f"Insufficient USDC balance: ${usdc_balance:.2f} (need ${self._min_balance_to_trade})")

            logger.info("usdc_balance_loaded", balance=round(usdc_balance, 2))

            # Initialize position tracker for live mode with exposure limits
            # Use actual balance and config-based percentage limits
            max_open_order = usdc_balance * self.config.risk.max_open_order_pct
            max_position = usdc_balance * self.config.risk.max_position_pct

            logger.info(
                "exposure_limits_set",
                usdc_balance=round(usdc_balance, 2),
                max_total_exposure_pct=f"{self.config.risk.max_net_frac * 100:.0f}%",
                max_total_exposure_usd=round(usdc_balance * self.config.risk.max_net_frac, 2),
                max_open_order_per_market=round(max_open_order, 2),
                max_position_per_market=round(max_position, 2),
            )

            self.position_tracker = PositionTracker(
                max_exposure_pct=self.config.risk.max_net_frac,
                max_open_order_value=max_open_order,
                max_position_value=max_position,
            )
            self.position_tracker.set_bankroll(usdc_balance)

            # Sync allowances with CLOB API on startup
            # This ensures the CLOB API has the latest view of our on-chain approvals
            await self._sync_clob_allowances()

        # Initialize price feed and lag signal
        symbols = [asset.symbol for asset in self.config.lag_signal.assets]
        if symbols:
            self.price_feed = PriceFeed(
                exchange_id=self.config.lag_signal.exchange,
                symbols=symbols,
            )
            await self.price_feed.start()

            self.lag_model = LagModel(
                price_feed=self.price_feed,
                vol_window=self.config.belief.window_seconds,
            )

            self.skew_computer = SkewComputer(
                lag_model=self.lag_model,
                asset_configs=[
                    AssetConfig(
                        symbol=asset.symbol,
                        weight=asset.weight,
                        signal_type="vol_adjusted",
                    )
                    for asset in self.config.lag_signal.assets
                ],
                skew_multiplier=self.config.lag_signal.skew_multiplier,
            )

        # Initialize risk manager
        self.risk_manager = RiskManager(
            jump_z=self.config.belief.jump_z,
            momentum_z=self.config.belief.momentum_z,
            stop_prob_low=self.config.risk.stop_prob_low,
            stop_prob_high=self.config.risk.stop_prob_high,
            max_net_frac=self.config.risk.max_net_frac,
            min_time_to_expiry_seconds=self.config.risk.min_time_to_expiry_seconds,
            gamma_danger_threshold=self.config.risk.gamma_danger.threshold,
            gamma_danger_multiplier=self.config.risk.gamma_danger.gamma_multiplier,
        )

        # Initialize quote calculator
        self.quote_calculator = QuoteCalculator(
            gamma=self.config.avellaneda_stoikov.gamma,
            base_spread_x=self.config.avellaneda_stoikov.base_spread_x,
            kappa=self.config.avellaneda_stoikov.kappa,
            gamma_danger_threshold=self.config.risk.gamma_danger.threshold,
            gamma_danger_multiplier=self.config.risk.gamma_danger.gamma_multiplier,
        )

        # Initialize ML probability model adapter
        belief_cfg = self.config.belief
        self.model_adapter = ModelAdapter(
            live=not self.dry_run,
            model_path=belief_cfg.model_path,
            model_type=belief_cfg.model_type,
            fixed_prob=0.5,
            vol_regime_ratio_threshold=belief_cfg.vol_regime_ratio_threshold,
        )
        if self.model_adapter.load():
            logger.info("probability_model_loaded", model_type=belief_cfg.model_type, path=belief_cfg.model_path)
        else:
            logger.warning(
                "probability_model_not_loaded",
                path=belief_cfg.model_path,
                note="falling back to rolling-window belief estimator",
            )
        self._last_retrain_time = datetime.utcnow()

        # Initialize market discovery
        self.market_discovery = MarketDiscovery()

        # Discover/resolve markets
        await self._refresh_markets()

        self._running = True
        self._start_time = datetime.utcnow()

        # Log clean startup summary
        market_names = [m.description[:50] for m in self.active_markets]
        logger.info(
            "BOT_STARTED",
            mode="LIVE" if not self.dry_run else "DRY_RUN",
            balance=f"${self._usdc_balance:.2f}",
            markets=market_names,
            max_exposure=f"{self.config.risk.max_net_frac * 100:.0f}%",
        )

        # Start main loop
        await self._run_loop()

    async def _sync_clob_allowances(self) -> None:
        """
        Sync on-chain allowances with Polymarket CLOB API.
        
        The CLOB API maintains its own cache of balance/allowance state.
        This must be called to ensure the API knows about our on-chain approvals.
        Without this, orders will fail with "not enough balance / allowance".
        """
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
            
            if not isinstance(self.client, PolymarketClient):
                return
            
            logger.info("syncing_clob_allowances")
            
            # Sync USDC (collateral) allowance - required for buying positions
            try:
                usdc_params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                await asyncio.to_thread(
                    self.client._client.update_balance_allowance,
                    usdc_params,
                )
                logger.info("usdc_allowance_synced")
            except Exception as e:
                logger.warning("usdc_allowance_sync_failed", error=str(e))
            
            # CTF (conditional token) allowance sync often fails without a specific token_id
            # This is expected - the allowance is synced per-token when actually trading
            try:
                ctf_params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
                await asyncio.to_thread(
                    self.client._client.update_balance_allowance,
                    ctf_params,
                )
                logger.info("ctf_allowance_synced")
            except Exception as e:
                # Expected to fail without token_id - not a problem for buying
                logger.debug("ctf_allowance_sync_skipped", error=str(e))
            
            logger.info("clob_allowances_synced")
            
        except Exception as e:
            logger.error("clob_allowance_sync_error", error=str(e))
            # Don't fail startup, but warn that trading may fail
            logger.warning(
                "allowance_sync_warning",
                message="CLOB allowance sync failed - orders may fail with 'not enough balance/allowance'"
            )

    async def _refresh_markets(self) -> None:
        """Refresh market list, discovering auto-discover markets."""
        self.active_markets = []
        
        # Clear invalid orderbook counts on refresh
        self._invalid_orderbook_counts.clear()
        
        # Clear market end dates for stale markets
        self._market_end_dates.clear()
        
        # Clear tracked orders on market refresh (they'll be stale)
        if self.position_tracker:
            self.position_tracker.record_all_orders_cancelled()

        for market in self.config.markets:
            if market.auto_discover:
                # In dry run mode, try discovery but fall back to mock market
                discovered = None
                interval = market.interval  # Use interval from config
                
                if not self.dry_run:
                    # Live mode: require real market discovery
                    discovered = await self.market_discovery.find_btc_market(interval=interval)
                else:
                    # Dry run: try discovery, but don't fail if it doesn't work
                    try:
                        discovered = await self.market_discovery.find_btc_market(interval=interval)
                    except Exception as e:
                        logger.warning("market_discovery_error_dry_run", error=str(e), interval=interval)
                
                if discovered:
                    # Choose token based on outcome preference
                    if market.outcome.upper() == "YES":
                        token_id = discovered.token_id_yes
                    else:
                        token_id = discovered.token_id_no

                    resolved_market = MarketConfig(
                        token_id=token_id,
                        condition_id=discovered.condition_id,
                        description=discovered.question,
                        auto_discover=True,
                        outcome=market.outcome,
                    )
                    self.active_markets.append(resolved_market)
                    
                    # Store market end date for expiry tracking
                    if discovered.end_date:
                        self._market_end_dates[token_id] = discovered.end_date

                    # Calculate time until expiry for cleaner logging
                    expires_in = "N/A"
                    if discovered.end_date:
                        ttl = (discovered.end_date - datetime.utcnow()).total_seconds()
                        if ttl > 0:
                            expires_in = f"{int(ttl // 60)}m"
                    
                    logger.info(
                        "MARKET",
                        name=discovered.question,
                        expires_in=expires_in,
                        outcome=market.outcome,
                    )
                elif self.dry_run:
                    # Dry run fallback: create a mock market for paper trading
                    mock_token_id = f"dry_run_mock_{market.outcome.lower()}"
                    resolved_market = MarketConfig(
                        token_id=mock_token_id,
                        condition_id="mock_condition_dry_run",
                        description=f"[DRY RUN] Mock BTC Hourly ({market.outcome})",
                        auto_discover=True,
                        outcome=market.outcome,
                    )
                    self.active_markets.append(resolved_market)

                    logger.info(
                        "mock_market_created",
                        description=resolved_market.description,
                        token_id=mock_token_id,
                        reason="market_discovery_unavailable",
                    )
                else:
                    # Live mode: discovery failed, this is an error
                    logger.error("market_discovery_failed", market=market.description)
            else:
                # Use static configuration
                self.active_markets.append(market)

        # Initialize per-market components for new markets
        for market in self.active_markets:
            token_id = market.token_id

            if token_id not in self.belief_managers:
                self.belief_managers[token_id] = BeliefManager(
                    token_id=token_id,
                    window_seconds=self.config.belief.window_seconds,
                    sigma_b_floor=self.config.belief.sigma_b_floor,
                    robust_method=self.config.belief.robust_method,
                    jump_z=self.config.belief.jump_z,
                    momentum_z=self.config.belief.momentum_z,
                )

            if token_id not in self.orderbook_managers:
                self.orderbook_managers[token_id] = OrderBookManager(token_id=token_id)

            if not self.dry_run and isinstance(self.client, PolymarketClient):
                if token_id not in self.order_managers:
                    self.order_managers[token_id] = OrderManager(
                        client=self.client,
                        cancel_cooldown_seconds=self.config.execution.cancel_cooldown_seconds,
                        reprice_threshold_ticks=self.config.execution.reprice_threshold_ticks,
                        order_lifetime_seconds=self.config.execution.order_lifetime_seconds,
                    )

                if token_id not in self.fill_trackers:
                    self.fill_trackers[token_id] = FillTracker(
                        client=self.client,
                    )

        self._last_market_refresh = datetime.utcnow()

    async def stop(self) -> None:
        """Stop trading gracefully."""
        logger.info("trading_loop_stopping")
        self._running = False

        # Cancel all open orders
        if self.client:
            if self.dry_run and isinstance(self.client, DryRunAdapter):
                await self.client.cancel_all_orders()
            elif isinstance(self.client, PolymarketClient):
                await self.client.cancel_all_orders()
                # Clear tracked orders
                if self.position_tracker:
                    self.position_tracker.record_all_orders_cancelled()

        # Log final dry run summary
        if self.dry_run and isinstance(self.client, DryRunAdapter):
            await self._log_dry_run_summary()

        # Log shutdown
        if self.repo:
            await self.repo.log_event(
                event_type="SHUTDOWN",
                message=f"Trading loop stopped after {self._tick_count} ticks",
                severity=EventSeverity.INFO,
            )

        # Close connections
        if self.market_discovery:
            await self.market_discovery.close()

        if self.price_feed:
            await self.price_feed.stop()

        if self.db:
            await self.db.close()

        logger.info(
            "trading_loop_stopped",
            total_ticks=self._tick_count,
            runtime_seconds=(datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
        )

    async def _run_loop(self) -> None:
        """Main trading loop."""
        while self._running:
            tick_start = datetime.utcnow()

            try:
                # Check if markets need refresh (for auto-discover)
                if self._should_refresh_markets():
                    logger.info("refreshing_markets")
                    # Cancel all orders before switching markets
                    if self.client:
                        if isinstance(self.client, DryRunAdapter):
                            await self.client.cancel_all_orders()
                        elif isinstance(self.client, PolymarketClient):
                            await self.client.cancel_all_orders()
                            # Clear tracked orders
                            if self.position_tracker:
                                self.position_tracker.record_all_orders_cancelled()
                    await self._refresh_markets()

                # Check if model retraining is needed
                await self._check_retrain_schedule()

                # Process each market
                for market in self.active_markets:
                    await self._tick(market.token_id)

                # Fetch crypto prices for lag signal
                if self.price_feed:
                    await self.price_feed.fetch_all_prices()

            except Exception as e:
                logger.error("tick_error", error=str(e))
                if self.repo:
                    await self.repo.log_event(
                        event_type="TICK_ERROR",
                        message=str(e),
                        severity=EventSeverity.ERROR,
                    )

            self._tick_count += 1

            # Sync fills for live mode (every tick to update positions)
            if not self.dry_run:
                await self._sync_fills()
                # Sync positions from Data API (source of truth)
                await self._sync_positions_from_data_api()
                # Periodically check balance and stop if too low
                await self._maybe_refresh_balance()

            # Log periodic status for dry run
            if self.dry_run:
                await self._maybe_log_status()
            else:
                # Log periodic status for live mode (exposure tracking)
                await self._maybe_log_live_status()

            # Sleep for remainder of tick interval
            elapsed = (datetime.utcnow() - tick_start).total_seconds()
            sleep_time = max(0, 1.0 - elapsed)  # 1 second cadence
            await asyncio.sleep(sleep_time)

    async def _check_retrain_schedule(self) -> None:
        """
        Check whether the probability model needs retraining.

        Triggers on:
        1. Calendar schedule: every `retrain_interval_days` days
        2. Performance trigger: rolling BSS < threshold for 3 consecutive days
        """
        if self.model_adapter is None or self._retraining_task is not None:
            return

        now = datetime.utcnow()
        belief_cfg = self.config.belief
        retrain_interval_days = getattr(belief_cfg, "retrain_interval_days", 7)
        bss_threshold = getattr(belief_cfg, "bss_retrain_threshold", 0.0)
        bss_window_days = getattr(belief_cfg, "bss_window_days", 14)

        # Calendar trigger
        if self._last_retrain_time is not None:
            days_since = (now - self._last_retrain_time).total_seconds() / 86400
            if days_since >= retrain_interval_days:
                logger.info("retrain_calendar_trigger", days_since=round(days_since, 1))
                await self._trigger_background_retrain(reason="calendar")
                return

        # Performance trigger: check rolling BSS from stored predictions
        if self.repo and len(self._retrain_bss_history) >= 3:
            recent_bss = self._retrain_bss_history[-3:]
            if all(b < bss_threshold for b in recent_bss):
                if self.repo:
                    await self.repo.log_event(
                        event_type="REGIME_CHANGE_DETECTED",
                        message="BSS below threshold for 3 consecutive periods — triggering retrain",
                        severity=EventSeverity.WARNING,
                        data={"bss_history": recent_bss, "threshold": bss_threshold},
                    )
                logger.warning(
                    "retrain_performance_trigger",
                    bss_history=recent_bss,
                    threshold=bss_threshold,
                )
                await self._trigger_background_retrain(reason="performance")

    async def _trigger_background_retrain(self, reason: str = "calendar") -> None:
        """Launch a non-blocking background retrain task."""
        logger.info("retrain_starting", reason=reason)
        self._retraining_task = asyncio.create_task(self._background_retrain())

    async def _background_retrain(self) -> None:
        """
        Background retraining task.

        Runs in a separate asyncio task so it doesn't block the trading loop.
        """
        try:
            import pandas as pd
            from src.probability_model.xgboost_model import XGBoostModel
            from src.probability_model.trainer import WalkForwardTrainer

            if self.repo is None:
                return

            symbol = "BTC/USDT"
            prices = await self.repo.get_recent_crypto_prices(symbol, limit=200000)
            if len(prices) < 100:
                logger.warning("retrain_insufficient_data", n_candles=len(prices))
                return

            prices_sorted = sorted(prices, key=lambda p: p.timestamp)
            rows = []
            for p in prices_sorted:
                meta = p.metadata or {}
                rows.append({
                    "timestamp": p.timestamp,
                    "open": meta.get("open", p.price),
                    "high": meta.get("high", p.price),
                    "low": meta.get("low", p.price),
                    "close": p.price,
                    "volume": p.volume_24h or 0.0,
                })
            df = pd.DataFrame(rows).set_index("timestamp")

            trainer = WalkForwardTrainer(model_class=XGBoostModel, config=self.config.belief)
            fold_results = await asyncio.to_thread(trainer.run, df)

            if fold_results:
                belief_cfg = self.config.belief
                model_path = belief_cfg.model_path
                fold_results[-1].model.save(model_path)

                # Reload model into adapter
                if self.model_adapter is not None:
                    self.model_adapter.model_path = model_path
                    self.model_adapter.load()

                self._last_retrain_time = datetime.utcnow()
                logger.info("retrain_complete", n_folds=len(fold_results), model_path=model_path)

                if self.repo:
                    await self.repo.log_event(
                        event_type="MODEL_RETRAINED",
                        message=f"Model retrained successfully ({len(fold_results)} folds)",
                        severity=EventSeverity.INFO,
                        data={"n_folds": len(fold_results), "model_path": model_path},
                    )
        except Exception as exc:
            logger.error("retrain_failed", error=str(exc))
        finally:
            self._retraining_task = None

    def _should_refresh_markets(self) -> bool:
        """Check if markets should be refreshed (for auto-discover)."""
        # Only refresh if we have auto-discover markets
        has_auto_discover = any(m.auto_discover for m in self.config.markets)
        if not has_auto_discover:
            return False

        # Check refresh interval
        if self._last_market_refresh is None:
            return True

        elapsed = (datetime.utcnow() - self._last_market_refresh).total_seconds()
        if elapsed >= self._market_refresh_interval:
            return True
        
        # Check if any market is expired or about to expire
        now = datetime.utcnow()
        for token_id, end_date in self._market_end_dates.items():
            time_to_expiry = (end_date - now).total_seconds()
            if time_to_expiry <= 0:
                # Market has expired - need to refresh to get next market
                logger.info(
                    "market_expired_refresh",
                    token_id=token_id[:16] + "...",
                    end_date=end_date.isoformat(),
                )
                return True
        
        return False
    
    def _get_time_to_expiry(self, token_id: str) -> float | None:
        """Get time to expiry in seconds for a market."""
        end_date = self._market_end_dates.get(token_id)
        if end_date is None:
            return None
        return max(0, (end_date - datetime.utcnow()).total_seconds())

    async def _maybe_log_status(self) -> None:
        """Log periodic status update for dry run mode."""
        now = datetime.utcnow()
        
        if self._last_status_log is not None:
            elapsed = (now - self._last_status_log).total_seconds()
            if elapsed < self._status_log_interval:
                return
        
        self._last_status_log = now
        
        if not isinstance(self.client, DryRunAdapter):
            return
        
        stats = self.client.get_stats()
        
        # Calculate total PnL
        total_realized_pnl = sum(
            pos.realized_pnl 
            for pos in self.client.state.positions.values()
        )
        
        # Calculate unrealized PnL from current positions
        unrealized_pnl = 0.0
        for token_id, pos in self.client.state.positions.items():
            if pos.size != 0 and pos.avg_entry_price is not None:
                # Use last book price for unrealized PnL
                book = self.client._last_books.get(token_id)
                if book:
                    current_price = book.mid_price or 0.5
                    if pos.size > 0:
                        unrealized_pnl += pos.size * (current_price - pos.avg_entry_price)
                    else:
                        unrealized_pnl += abs(pos.size) * (pos.avg_entry_price - current_price)
        
        # Get position summary
        positions_str = ", ".join(
            f"{tid[:12]}={p['size']:.1f}" 
            for tid, p in stats.get("positions", {}).items()
        ) or "flat"
        
        logger.info(
            "dry_run_status",
            tick=self._tick_count,
            balance=round(stats["balance"], 2),
            realized_pnl=round(total_realized_pnl, 4),
            unrealized_pnl=round(unrealized_pnl, 4),
            total_pnl=round(total_realized_pnl + unrealized_pnl - stats["total_fees"], 4),
            total_fills=stats["total_fills"],
            total_fees=round(stats["total_fees"], 4),
            positions=positions_str,
        )

    async def _maybe_log_live_status(self) -> None:
        """Log periodic status update for live mode with clean formatting."""
        now = datetime.utcnow()

        if self._last_status_log is not None:
            elapsed = (now - self._last_status_log).total_seconds()
            if elapsed < self._status_log_interval:
                return

        self._last_status_log = now

        # Get market info
        market_name = "None"
        time_to_expiry_str = "N/A"
        current_price = 0.0
        for market in self.active_markets:
            market_name = market.description[:40] if market.description else "Unknown"
            ttl = self._get_time_to_expiry(market.token_id)
            if ttl is not None:
                minutes = int(ttl // 60)
                seconds = int(ttl % 60)
                time_to_expiry_str = f"{minutes}m {seconds}s"
            # Get current mid price
            ob_manager = self.orderbook_managers.get(market.token_id)
            if ob_manager and ob_manager.latest_snapshot:
                current_price = ob_manager.latest_snapshot.mid_price or 0.0
            break

        # Get position info from Data API (source of truth)
        # Only show positions for ACTIVE markets (not old resolved ones)
        position_size = 0.0
        position_value = 0.0
        unrealized_pnl = 0.0
        total_initial_value = 0.0
        avg_entry = 0.0

        # Get set of active market token IDs
        active_token_ids = {m.token_id for m in self.active_markets}

        # Use Data API positions for accurate tracking - filter to active markets only
        if self._data_api_positions:
            for pos in self._data_api_positions:
                token_id = pos.get("token_id", "")

                # Only include positions for active markets (skip old resolved positions)
                if token_id not in active_token_ids:
                    continue

                # Skip resolved positions (winning or losing)
                if pos.get("is_resolved_winning") or pos.get("is_resolved_losing"):
                    continue

                position_size += pos.get("size", 0)
                position_value += pos.get("current_value", 0)
                unrealized_pnl += pos.get("unrealized_pnl", 0)
                total_initial_value += pos.get("initial_value", 0)
                avg_entry = pos.get("avg_price", 0)
        
        # Get open orders count
        open_orders_count = 0
        open_notional = 0.0
        if self.position_tracker:
            status = self.position_tracker.get_status()
            open_notional = status.get("open_buy_notional", 0) + status.get("open_sell_notional", 0)
            for token_orders in status.get("open_orders", {}).values():
                open_orders_count += token_orders.get("count", 0)

        # Total PnL (unrealized from Data API - this is the actual PnL)
        total_pnl = unrealized_pnl
        
        # Format PnL with sign
        def fmt_pnl(val: float) -> str:
            if val >= 0:
                return f"+${val:.2f}"
            return f"-${abs(val):.2f}"

        # Clean single-line status log
        logger.info(
            "STATUS",
            balance=f"${self._usdc_balance:.2f}",
            market=market_name,
            expires=time_to_expiry_str,
            price=f"{current_price:.3f}" if current_price > 0 else "N/A",
            position=f"{position_size:.1f} @ {avg_entry:.3f}" if position_size > 0 else "None",
            pos_value=f"${position_value:.2f}" if position_value > 0 else "$0",
            pnl=fmt_pnl(total_pnl),
            open_orders=open_orders_count,
            open_notional=f"${open_notional:.2f}",
        )

    async def _close_position_at_market(
        self,
        token_id: str,
        position_size: float,
        snapshot: "OrderBookSnapshot",
    ) -> bool:
        """
        Close position by selling at market (best bid).
        
        Used when approaching market expiry to exit positions.
        
        Args:
            token_id: The token to sell
            position_size: Amount to sell
            snapshot: Current orderbook snapshot for pricing
            
        Returns:
            True if sell order was placed successfully
        """
        if not isinstance(self.client, PolymarketClient):
            return False
        
        if position_size <= 0:
            logger.info("no_position_to_close", token_id=token_id[:16] + "...")
            return False
        
        # Get best bid price from snapshot
        best_bid = snapshot.best_bid
        if best_bid is None or best_bid <= 0:
            logger.warning(
                "cannot_close_position_no_bid",
                token_id=token_id[:16] + "...",
                position_size=position_size,
            )
            return False
        
        # Calculate sell size - sell entire position
        sell_size = position_size
        order_value = best_bid * sell_size
        
        # Ensure minimum order value ($1)
        if order_value < 1.0:
            if best_bid > 0:
                sell_size = max(sell_size, 1.0 / best_bid)
            else:
                logger.warning(
                    "position_too_small_to_close",
                    token_id=token_id[:16] + "...",
                    position_size=position_size,
                    order_value=order_value,
                )
                return False
        
        logger.info(
            "closing_position_at_expiry",
            token_id=token_id[:16] + "...",
            position_size=position_size,
            sell_size=round(sell_size, 4),
            best_bid=best_bid,
            order_value=round(best_bid * sell_size, 2),
        )
        
        try:
            # Place sell order at best bid (market sell)
            order = await self.client.place_limit_order(
                token_id=token_id,
                side="SELL",
                price=best_bid,
                size=sell_size,
            )
            
            if order:
                order_value = best_bid * sell_size
                logger.info(
                    "CLOSING_POSITION",
                    side="SELL",
                    size=f"{sell_size:.2f}",
                    price=f"{best_bid:.3f}",
                    value=f"${order_value:.2f}",
                    reason="market_expiry",
                )
                return True
            else:
                logger.warning(
                    "position_close_order_failed",
                    token_id=token_id[:16] + "...",
                )
                return False
                
        except Exception as e:
            logger.error(
                "position_close_error",
                token_id=token_id[:16] + "...",
                error=str(e),
            )
            return False

    async def _sync_fills(self) -> None:
        """Sync fills from exchange and update position tracking."""
        for market in self.active_markets:
            token_id = market.token_id
            fill_tracker = self.fill_trackers.get(token_id)

            if fill_tracker:
                try:
                    # Sync fills from exchange
                    new_fills = await fill_tracker.sync_fills(token_id)

                    # Update position tracker with new fills
                    if self.position_tracker and new_fills:
                        for fill in new_fills:
                            self.position_tracker.record_fill(
                                order_id=fill.order_id or fill.id,
                                token_id=fill.token_id,
                                side=fill.side.value,
                                price=fill.price,
                                size=fill.size,
                            )
                            
                            # Log each fill as a clear TRADE notification
                            fill_value = fill.price * fill.size
                            logger.info(
                                "TRADE",
                                side=fill.side.value,
                                size=f"{fill.size:.2f}",
                                price=f"{fill.price:.3f}",
                                value=f"${fill_value:.2f}",
                            )
                except Exception as e:
                    logger.warning(
                        "fill_sync_error",
                        token_id=token_id[:16] + "...",
                        error=str(e),
                    )

    async def _sync_positions_from_data_api(self) -> None:
        """
        Sync positions from Polymarket Data API.

        This is the source of truth for actual positions, as the fill tracking
        via get_trades() has pagination issues and misses most trades.
        """
        if self.dry_run or not isinstance(self.client, PolymarketClient):
            return

        now = datetime.utcnow()

        # Check if enough time has passed since last sync
        if self._last_position_sync is not None:
            elapsed = (now - self._last_position_sync).total_seconds()
            if elapsed < self._position_sync_interval:
                return

        self._last_position_sync = now

        try:
            positions = await self.client.get_positions_from_data_api()
            self._data_api_positions = positions

            if positions:
                logger.debug(
                    "positions_synced_from_data_api",
                    count=len(positions),
                    total_value=round(sum(p["current_value"] for p in positions), 2),
                    total_unrealized_pnl=round(sum(p["unrealized_pnl"] for p in positions), 2),
                )
        except Exception as e:
            logger.warning("data_api_position_sync_error", error=str(e))

    async def _maybe_refresh_balance(self) -> None:
        """Periodically refresh USDC balance and update position tracker."""
        if self.dry_run or not isinstance(self.client, PolymarketClient):
            return

        now = datetime.utcnow()

        # Check if enough time has passed since last check
        if self._last_balance_check is not None:
            elapsed = (now - self._last_balance_check).total_seconds()
            if elapsed < self._balance_check_interval:
                return

        self._last_balance_check = now

        try:
            new_balance = await self.client.get_usdc_balance()
            old_balance = self._usdc_balance
            self._usdc_balance = new_balance

            # Update position tracker with new bankroll
            if self.position_tracker:
                self.position_tracker.set_bankroll(new_balance)

            # Log significant balance changes
            if old_balance > 0 and abs(new_balance - old_balance) > 1.0:
                logger.info(
                    "balance_updated",
                    old_balance=round(old_balance, 2),
                    new_balance=round(new_balance, 2),
                    change=round(new_balance - old_balance, 2),
                )

            # Check if balance is too low to continue trading
            if new_balance < self._min_balance_to_trade:
                logger.error(
                    "balance_too_low",
                    balance=round(new_balance, 2),
                    min_required=self._min_balance_to_trade,
                    message="Stopping trading - insufficient balance",
                )
                # Cancel all orders and stop
                if self.client:
                    await self.client.cancel_all_orders()
                    if self.position_tracker:
                        self.position_tracker.record_all_orders_cancelled()
                self._running = False

        except Exception as e:
            logger.warning("balance_refresh_error", error=str(e))

    async def _log_dry_run_summary(self) -> None:
        """Log final dry run performance summary."""
        if not isinstance(self.client, DryRunAdapter):
            return
        
        stats = self.client.get_stats()
        initial_balance = self.config.dry_run.initial_balance
        final_balance = stats["balance"]
        
        # Calculate total realized PnL
        total_realized_pnl = sum(
            pos.realized_pnl 
            for pos in self.client.state.positions.values()
        )
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for token_id, pos in self.client.state.positions.items():
            if pos.size != 0 and pos.avg_entry_price is not None:
                book = self.client._last_books.get(token_id)
                if book:
                    current_price = book.mid_price or 0.5
                    if pos.size > 0:
                        unrealized_pnl += pos.size * (current_price - pos.avg_entry_price)
                    else:
                        unrealized_pnl += abs(pos.size) * (pos.avg_entry_price - current_price)
        
        total_pnl = total_realized_pnl + unrealized_pnl - stats["total_fees"]
        pnl_percent = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0
        
        runtime = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
        
        # Count buy/sell fills
        buy_fills = sum(1 for f in self.client.state.fills if f.side == OrderSide.BUY)
        sell_fills = sum(1 for f in self.client.state.fills if f.side == OrderSide.SELL)
        
        logger.info("=" * 50)
        logger.info(
            "dry_run_final_summary",
            runtime_seconds=round(runtime, 1),
            total_ticks=self._tick_count,
            initial_balance=initial_balance,
            final_balance=round(final_balance, 2),
            realized_pnl=round(total_realized_pnl, 4),
            unrealized_pnl=round(unrealized_pnl, 4),
            total_fees=round(stats["total_fees"], 4),
            net_pnl=round(total_pnl, 4),
            pnl_percent=round(pnl_percent, 3),
            total_fills=stats["total_fills"],
            buy_fills=buy_fills,
            sell_fills=sell_fills,
        )
        
        # Log final positions
        for token_id, pos in self.client.state.positions.items():
            if pos.size != 0:
                logger.info(
                    "dry_run_final_position",
                    token_id=token_id[:20] + "...",
                    size=round(pos.size, 4),
                    avg_entry=round(pos.avg_entry_price, 4) if pos.avg_entry_price else None,
                    realized_pnl=round(pos.realized_pnl, 4),
                )
        logger.info("=" * 50)

    async def _tick(self, token_id: str) -> TickResult:
        """
        Execute single tick for a market.

        Args:
            token_id: Market token ID

        Returns:
            TickResult with tick outcome
        """
        result = TickResult(
            tick_number=self._tick_count,
            timestamp=datetime.utcnow(),
            mid_price=None,
            spread_bps=None,
            bid_placed=False,
            ask_placed=False,
            orders_cancelled=0,
            vetoed=False,
            stop_triggered=False,
        )

        try:
            # 1. Fetch order book
            if isinstance(self.client, PolymarketClient):
                book = await self.client.get_order_book(token_id)
            else:
                # Dry run: use cached book or create synthetic
                book = self.orderbook_managers[token_id].current
                if book is None:
                    # Generate synthetic order book for dry run
                    book = self._generate_synthetic_book(token_id)

            # Update order book manager
            snapshot = self.orderbook_managers[token_id].update(book)
            if snapshot is None:
                # Track consecutive invalid orderbooks
                self._invalid_orderbook_counts[token_id] = self._invalid_orderbook_counts.get(token_id, 0) + 1
                invalid_count = self._invalid_orderbook_counts[token_id]
                
                if invalid_count >= self._max_invalid_orderbook_count:
                    logger.warning(
                        "market_likely_expired",
                        token_id=token_id[:16] + "...",
                        invalid_count=invalid_count,
                        message="Too many consecutive invalid orderbooks - market may have expired",
                    )
                    # Force market refresh on next tick
                    self._last_market_refresh = None
                    
                return result
            
            # Reset invalid count on valid orderbook
            self._invalid_orderbook_counts[token_id] = 0

            result.mid_price = snapshot.mid_price
            result.spread_bps = snapshot.spread_bps

            # Update dry run adapter with latest book
            if self.dry_run and isinstance(self.client, DryRunAdapter):
                self.client.set_order_book(book)

            # 2. Update belief state
            belief = self.belief_managers[token_id].update(
                bid=book.best_bid_price,
                ask=book.best_ask_price,
            )

            # 3. Get current position
            position_size = 0.0
            position_pnl = 0.0
            bankroll = self.config.dry_run.initial_balance if self.dry_run else 10000.0

            if self.dry_run and isinstance(self.client, DryRunAdapter):
                pos = self.client.get_position(token_id)
                position_size = pos.size
                position_pnl = pos.realized_pnl
                bankroll = self.client.get_balance()
            elif isinstance(self.client, PolymarketClient):
                # In live mode, use Data API positions as source of truth
                # This fixes the issue where fill tracking misses most trades due to pagination
                for data_pos in self._data_api_positions:
                    if data_pos.get("token_id") == token_id:
                        # Skip resolved positions
                        if data_pos.get("is_resolved_winning") or data_pos.get("is_resolved_losing"):
                            continue
                        position_size = data_pos.get("size", 0)
                        position_pnl = data_pos.get("unrealized_pnl", 0)
                        break

                # Fallback to fill tracker if Data API hasn't synced yet
                if position_size == 0:
                    fill_tracker = self.fill_trackers.get(token_id)
                    if fill_tracker:
                        pos = fill_tracker.get_position(token_id)
                        position_size = pos.size
                        position_pnl = pos.realized_pnl

                # Use actual USDC balance as bankroll
                bankroll = self._usdc_balance if self._usdc_balance > 0 else 10000.0

            # 4. Compute lag signal skew
            signal_skew = 0.0
            if self.skew_computer:
                skew_signal = self.skew_computer.compute_weighted_skew()
                signal_skew = skew_signal.total_skew

            # 5. Evaluate risk (including time to expiry)
            time_to_expiry = self._get_time_to_expiry(token_id)
            
            risk_decision = self.risk_manager.evaluate(
                belief=belief,
                position_size=position_size,
                position_pnl=position_pnl,
                bankroll=bankroll,
                spread_bps=snapshot.spread_bps,
                bid_depth=snapshot.bid_depth_5,
                ask_depth=snapshot.ask_depth_5,
                time_to_expiry_seconds=time_to_expiry,
            )

            if risk_decision.veto.vetoed:
                result.vetoed = True
                return result

            if risk_decision.close_position:
                result.stop_triggered = True
                
                # Handle time-to-expiry stop: cancel orders, sell position, then refresh market
                if risk_decision.stop.stop_type and risk_decision.stop.stop_type.value == "time_to_expiry":
                    ttl_str = f"{int(time_to_expiry // 60)}m {int(time_to_expiry % 60)}s" if time_to_expiry else "0s"
                    logger.warning(
                        "EXPIRY_STOP",
                        message="Market expiring soon - closing position",
                        time_left=ttl_str,
                        position=f"{position_size:.2f}" if position_size > 0 else "None",
                    )
                    
                    # Use the proven close_all_positions logic (same as close_all_positions.py script)
                    if isinstance(self.client, PolymarketClient):
                        # close_all_positions handles: cancel orders, fetch positions from Data API, sell all
                        result = await asyncio.to_thread(
                            close_all_positions,
                            self.client._client,
                        )
                        logger.info(
                            "expiry_close_complete",
                            orders_placed=len(result.get("orders_placed", [])),
                            errors=len(result.get("errors", [])),
                            total_sold=result.get("total_sold_value", 0),
                        )
                    elif isinstance(self.client, DryRunAdapter):
                        await self.client.cancel_all_orders()
                    
                    if self.position_tracker:
                        self.position_tracker.record_all_orders_cancelled(token_id)
                    
                    # Force market refresh to move to next market
                    self._last_market_refresh = None
                
                return result

            # 6. Calculate quotes
            context = QuoteContext(
                belief=belief,
                inventory=position_size,
                time_remaining=1.0,  # Simplified: always 1.0
                signal_skew=signal_skew,
                gamma_multiplier=risk_decision.gamma_multiplier,
            )

            quote_decision = self.quote_calculator.calculate_two_sided(
                context=context,
                allow_buy=risk_decision.allow_buy,
                allow_sell=risk_decision.allow_sell,
            )

            if not quote_decision.should_quote or quote_decision.quote is None:
                return result

            quote = quote_decision.quote

            # 7. Cancel stale orders and place new quotes
            # Calculate order size: ensure minimum $1 value, then apply risk limits
            min_order_value = 1.0  # Polymarket minimum order value
            base_order_value = 5.0  # Target order value in USD
            
            # Calculate sizes based on price (ensuring minimum value)
            bid_size = self._calculate_order_size(
                price=quote.bid_price,
                min_value=min_order_value,
                target_value=base_order_value,
                side="BUY",
                position_size=position_size,
                bankroll=bankroll,
            )
            ask_size = self._calculate_order_size(
                price=quote.ask_price,
                min_value=min_order_value,
                target_value=base_order_value,
                side="SELL",
                position_size=position_size,
                bankroll=bankroll,
            )
            
            if self.dry_run and isinstance(self.client, DryRunAdapter):
                # Cancel existing orders
                result.orders_cancelled = await self.client.cancel_all_orders(token_id)

                # Place new orders
                if risk_decision.allow_buy and bid_size > 0:
                    await self.client.place_limit_order(
                        token_id=token_id,
                        side=OrderSide.BUY,
                        price=quote.bid_price,
                        size=bid_size,
                    )
                    result.bid_placed = True

                if risk_decision.allow_sell and ask_size > 0:
                    await self.client.place_limit_order(
                        token_id=token_id,
                        side=OrderSide.SELL,
                        price=quote.ask_price,
                        size=ask_size,
                    )
                    result.ask_placed = True

            elif isinstance(self.client, PolymarketClient):
                order_manager = self.order_managers.get(token_id)
                if order_manager:
                    # Cancel stale orders
                    stale = order_manager.get_stale_orders()
                    for order in stale:
                        await order_manager.cancel_order(order.id)
                        result.orders_cancelled += 1
                        # Track cancellation
                        if self.position_tracker:
                            self.position_tracker.record_order_cancelled(order.id, token_id)

                    # Apply position tracker limits for live mode
                    if self.position_tracker:
                        # Get allowed sizes based on exposure limits
                        bid_size = self.position_tracker.get_allowed_order_size(
                            token_id=token_id,
                            side="BUY",
                            price=quote.bid_price,
                            desired_size=bid_size,
                        )
                        ask_size = self.position_tracker.get_allowed_order_size(
                            token_id=token_id,
                            side="SELL",
                            price=quote.ask_price,
                            desired_size=ask_size,
                        )
                        
                        # Enforce Polymarket minimums after position tracker limits
                        # If exposure limits reduce size below minimums, skip the order
                        if bid_size > 0 and bid_size < self.MIN_ORDER_SHARES:
                            logger.debug(
                                "bid_size_below_minimum_shares",
                                bid_size=bid_size,
                                min_shares=self.MIN_ORDER_SHARES,
                            )
                            bid_size = 0
                        if bid_size > 0 and bid_size * quote.bid_price < self.MIN_ORDER_VALUE:
                            logger.debug(
                                "bid_value_below_minimum",
                                bid_size=bid_size,
                                bid_price=quote.bid_price,
                                value=bid_size * quote.bid_price,
                            )
                            bid_size = 0
                        
                        if ask_size > 0 and ask_size < self.MIN_ORDER_SHARES:
                            logger.debug(
                                "ask_size_below_minimum_shares",
                                ask_size=ask_size,
                                min_shares=self.MIN_ORDER_SHARES,
                            )
                            ask_size = 0
                        if ask_size > 0 and ask_size * quote.ask_price < self.MIN_ORDER_VALUE:
                            logger.debug(
                                "ask_value_below_minimum",
                                ask_size=ask_size,
                                ask_price=quote.ask_price,
                                value=ask_size * quote.ask_price,
                            )
                            ask_size = 0
                        
                        # Log exposure status periodically
                        exposure = self.position_tracker.get_exposure_summary()
                        if exposure.at_max_exposure:
                            logger.info(
                                "at_max_exposure",
                                token_id=token_id[:16] + "...",
                                total_exposure=round(exposure.total_exposure, 2),
                                exposure_pct=round(exposure.exposure_pct * 100, 1),
                            )

                    # Place new quotes
                    # Only place BUY orders (we can always buy with USDC)
                    if risk_decision.allow_buy and bid_size > 0:
                        order = await order_manager.place_order(
                            token_id=token_id,
                            side=OrderSide.BUY,
                            price=quote.bid_price,
                            size=bid_size,
                        )
                        result.bid_placed = True
                        # Track the order
                        if self.position_tracker and order:
                            self.position_tracker.record_order_placed(
                                order_id=order.id,
                                token_id=token_id,
                                side="BUY",
                                price=quote.bid_price,
                                size=bid_size,
                            )

                    # Only place SELL orders if we have tokens to sell
                    # In live mode, only sell if we have a positive position
                    can_sell = position_size > 0
                    if risk_decision.allow_sell and ask_size > 0 and can_sell:
                        # Limit sell size to what we actually have
                        sell_size = min(ask_size, position_size)
                        if sell_size > 0:
                            order = await order_manager.place_order(
                                token_id=token_id,
                                side=OrderSide.SELL,
                                price=quote.ask_price,
                                size=sell_size,
                            )
                            result.ask_placed = True
                            # Track the order
                            if self.position_tracker and order:
                                self.position_tracker.record_order_placed(
                                    order_id=order.id,
                                    token_id=token_id,
                                    side="SELL",
                                    price=quote.ask_price,
                                    size=sell_size,
                                )
                    elif risk_decision.allow_sell and not can_sell:
                        logger.debug(
                            "sell_skipped_no_position",
                            token_id=token_id[:16] + "...",
                            position_size=position_size,
                        )

            logger.debug(
                "tick_complete",
                tick=self._tick_count,
                mid=round(belief.mid_prob, 4) if belief else None,
                bid=round(quote.bid_price, 4),
                ask=round(quote.ask_price, 4),
                spread_bps=round(quote.spread_bps, 1),
            )

        except Exception as e:
            result.error = str(e)
            logger.error("tick_error", token_id=token_id[:16] + "...", error=str(e))

        return result

    # Polymarket minimum order requirements
    MIN_ORDER_SHARES = 5.0  # Minimum 5 shares per order
    MIN_ORDER_VALUE = 1.0   # Minimum $1 order value
    
    def _calculate_order_size(
        self,
        price: float,
        min_value: float,
        target_value: float,
        side: str,
        position_size: float,
        bankroll: float,
    ) -> float:
        """
        Calculate order size ensuring minimum value and respecting risk limits.
        
        Polymarket requirements:
        - Minimum 5 shares per order
        - Minimum $1 order value (price * size >= $1)
        
        Args:
            price: Order price (probability 0-1)
            min_value: Minimum order value in USD (Polymarket requires $1)
            target_value: Target order value in USD
            side: "BUY" or "SELL"
            position_size: Current position size
            bankroll: Total bankroll
            
        Returns:
            Order size in shares, or 0 if order cannot be placed
        """
        if price <= 0 or price >= 1:
            return 0.0
        
        # Calculate minimum size to meet minimum value requirement
        min_size_for_value = min_value / price
        
        # Polymarket requires minimum 5 shares
        min_size = max(min_size_for_value, self.MIN_ORDER_SHARES)
        
        # Calculate target size
        target_size = target_value / price
        
        # Use target size but ensure at least minimum
        desired_size = max(target_size, min_size)
        
        # Apply risk limits if risk manager is available
        if self.risk_manager:
            limited_size = self.risk_manager.get_order_size_limit(
                side=side,
                position_size=position_size,
                bankroll=bankroll,
                desired_size=desired_size,
            )
            # If risk limit reduces below minimum, don't place order
            if limited_size < min_size:
                logger.debug(
                    "order_size_below_minimum",
                    side=side,
                    desired_size=desired_size,
                    limited_size=limited_size,
                    min_size=min_size,
                    min_shares=self.MIN_ORDER_SHARES,
                )
                return 0.0
            desired_size = limited_size
        
        # Final check: ensure we meet minimums
        if desired_size < self.MIN_ORDER_SHARES:
            return 0.0
        if desired_size * price < self.MIN_ORDER_VALUE:
            return 0.0
        
        # Round to 2 decimal places (Polymarket precision)
        return round(desired_size, 2)

    def get_status(self) -> dict[str, Any]:
        """Get current trading status."""
        status = {
            "running": self._running,
            "tick_count": self._tick_count,
            "dry_run": self.dry_run,
            "markets": len(self.config.markets),
        }

        if self._start_time:
            status["runtime_seconds"] = (datetime.utcnow() - self._start_time).total_seconds()

        if self.dry_run and isinstance(self.client, DryRunAdapter):
            status["dry_run_stats"] = self.client.get_stats()
            
        # Add position tracker stats for live mode
        if self.position_tracker:
            status["position_tracker"] = self.position_tracker.get_status()

        return status

    def _generate_synthetic_book(self, token_id: str) -> OrderBook:
        """
        Generate a synthetic order book for dry run mode.

        Creates a realistic-looking book around a base price with
        some random walk behavior.
        """
        # Get last book to add random walk, or start at 0.5
        last_book = self.orderbook_managers[token_id].current
        if last_book and last_book.mid_price:
            base_price = last_book.mid_price
            # Small random walk
            base_price += random.gauss(0, 0.002)
            base_price = max(0.05, min(0.95, base_price))
        else:
            base_price = 0.5

        spread = 0.02  # 2% spread
        half_spread = spread / 2

        # Generate 5 levels on each side
        bids = []
        asks = []

        for i in range(5):
            bid_price = base_price - half_spread - (i * 0.005)
            ask_price = base_price + half_spread + (i * 0.005)

            # Random size between 50 and 200
            bid_size = random.uniform(50, 200)
            ask_size = random.uniform(50, 200)

            if bid_price > 0.01:
                bids.append(OrderBookLevel(price=round(bid_price, 4), size=round(bid_size, 2)))
            if ask_price < 0.99:
                asks.append(OrderBookLevel(price=round(ask_price, 4), size=round(ask_size, 2)))

        book = OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow(),
        )

        logger.debug(
            "synthetic_book_generated",
            token_id=token_id[:16] + "...",
            mid=round(base_price, 4),
            spread=round(spread, 4),
        )

        return book
