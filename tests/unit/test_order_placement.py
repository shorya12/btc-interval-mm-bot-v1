"""Tests for order placement and client configuration."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.polymarket_client.client import (
    PolymarketClient,
    SIGNATURE_TYPE_EOA,
    SIGNATURE_TYPE_POLY_PROXY,
    SIGNATURE_TYPE_POLY_GNOSIS_SAFE,
)
from src.polymarket_client.orders import OrderManager, OrderState
from src.polymarket_client.types import Order, OrderSide, OrderType, OrderStatus


class TestSignatureTypeConfiguration:
    """Tests for signature type configuration (Phantom wallet uses type 0)."""

    def test_signature_type_constants(self):
        """Test signature type constants are correct."""
        assert SIGNATURE_TYPE_EOA == 0  # Phantom wallet uses this
        assert SIGNATURE_TYPE_POLY_PROXY == 1
        assert SIGNATURE_TYPE_POLY_GNOSIS_SAFE == 2

    @patch('src.polymarket_client.client.ClobClient')
    def test_client_init_with_eoa_signature(self, mock_clob_client):
        """Test client initialization with EOA signature type (Phantom)."""
        mock_instance = MagicMock()
        mock_instance.create_or_derive_api_creds.return_value = MagicMock(
            api_key="test_key",
            api_secret="test_secret",
            api_passphrase="test_pass",
        )
        mock_clob_client.return_value = mock_instance

        client = PolymarketClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            private_key="0x" + "1" * 64,
            funder="0x86E35744AEAe5E693df6FbE02FF91AA566955DCc",  # User's funder address
            signature_type=SIGNATURE_TYPE_EOA,  # Phantom wallet
        )

        # Verify ClobClient was called with correct signature_type
        mock_clob_client.assert_called_once()
        call_kwargs = mock_clob_client.call_args.kwargs
        assert call_kwargs['signature_type'] == 0
        assert call_kwargs['funder'] == "0x86E35744AEAe5E693df6FbE02FF91AA566955DCc"

    @patch('src.polymarket_client.client.ClobClient')
    def test_client_init_with_proxy_signature(self, mock_clob_client):
        """Test client initialization with Poly Proxy signature type."""
        mock_instance = MagicMock()
        mock_instance.create_or_derive_api_creds.return_value = MagicMock(
            api_key="test_key",
            api_secret="test_secret",
            api_passphrase="test_pass",
        )
        mock_clob_client.return_value = mock_instance

        client = PolymarketClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            private_key="0x" + "1" * 64,
            signature_type=SIGNATURE_TYPE_POLY_PROXY,
        )

        call_kwargs = mock_clob_client.call_args.kwargs
        assert call_kwargs['signature_type'] == 1


class TestOrderValidation:
    """Tests for order validation before placement."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Polymarket client."""
        client = Mock(spec=PolymarketClient)
        client.place_limit_order = AsyncMock(return_value=Order(
            id="test_order_123",
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
            original_size=10.0,
            status=OrderStatus.LIVE,
            created_at=datetime.utcnow(),
        ))
        client.cancel_order = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def order_manager(self, mock_client):
        """Create order manager with mock client."""
        return OrderManager(
            client=mock_client,
            cancel_cooldown_seconds=2.0,
            reprice_threshold_ticks=2,
            order_lifetime_seconds=30.0,
        )

    @pytest.mark.asyncio
    async def test_price_bounds_validation(self, order_manager):
        """Test that prices outside 0.001-0.999 are rejected."""
        # Price too low
        result = await order_manager.place_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.0005,  # Below minimum
            size=10.0,
        )
        assert result is None

        # Price too high
        result = await order_manager.place_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.9995,  # Above maximum
            size=10.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_price_accepted(self, order_manager, mock_client):
        """Test that valid prices are accepted."""
        result = await order_manager.place_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
        )
        assert result is not None
        assert result.price == 0.5
        mock_client.place_limit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_price_rounding_to_tick(self, order_manager, mock_client):
        """Test that prices are rounded to tick size."""
        # Price with more precision than tick size (0.001)
        await order_manager.place_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.5556,  # Should round to 0.556
            size=10.0,
        )

        call_kwargs = mock_client.place_limit_order.call_args.kwargs
        # Price should be rounded to 0.556 (3 decimal places)
        assert call_kwargs['price'] == pytest.approx(0.556, rel=0.01)

    @pytest.mark.asyncio
    async def test_order_tracking(self, order_manager, mock_client):
        """Test that placed orders are tracked."""
        order = await order_manager.place_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
            metadata={"strategy": "test"},
        )

        assert order is not None
        assert order.id in order_manager._orders
        assert order_manager._orders[order.id].metadata["strategy"] == "test"


class TestOrderPlacementErrors:
    """Tests for order placement error handling."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client that fails."""
        client = Mock(spec=PolymarketClient)
        client.place_limit_order = AsyncMock(side_effect=Exception("API Error"))
        client.cancel_order = AsyncMock(return_value=False)
        return client

    @pytest.fixture
    def order_manager(self, mock_client):
        return OrderManager(client=mock_client)

    @pytest.mark.asyncio
    async def test_order_failure_returns_none(self, order_manager):
        """Test that order placement failure returns None."""
        result = await order_manager.place_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_failure_handled(self, order_manager, mock_client):
        """Test that cancel failure is handled gracefully."""
        # First make place_order work to get an order tracked
        mock_client.place_limit_order = AsyncMock(return_value=Order(
            id="test_order_123",
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
            original_size=10.0,
            status=OrderStatus.LIVE,
            created_at=datetime.utcnow(),
        ))
        
        await order_manager.place_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
        )
        
        # Now cancel fails
        mock_client.cancel_order = AsyncMock(return_value=False)
        result = await order_manager.cancel_order("test_order_123")
        assert result is False


class TestOrderLifecycle:
    """Tests for order lifecycle management."""

    @pytest.fixture
    def mock_client(self):
        client = Mock(spec=PolymarketClient)
        client.place_limit_order = AsyncMock()
        client.cancel_order = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def order_manager(self, mock_client):
        return OrderManager(
            client=mock_client,
            cancel_cooldown_seconds=2.0,
            reprice_threshold_ticks=2,
            order_lifetime_seconds=30.0,
        )

    def test_stale_order_detection(self, order_manager):
        """Test detection of stale orders."""
        # Create an old order state manually
        old_order = Order(
            id="old_order",
            token_id="test",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
            original_size=10.0,
            status=OrderStatus.LIVE,
            created_at=datetime.utcnow(),
        )
        
        state = OrderState(
            order=old_order,
            created_at=datetime(2020, 1, 1),  # Very old
            last_update=datetime(2020, 1, 1),
        )
        order_manager._orders["old_order"] = state

        stale = order_manager.get_stale_orders()
        assert len(stale) == 1
        assert stale[0].id == "old_order"

    def test_needs_reprice_large_price_change(self, order_manager):
        """Test reprice detection on large price change."""
        order = Order(
            id="test",
            token_id="test",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
            original_size=10.0,
            status=OrderStatus.LIVE,
            created_at=datetime.utcnow(),
        )

        # 3 ticks difference (threshold is 2)
        needs_reprice = order_manager.needs_reprice(order, 0.503)
        assert needs_reprice is True

    def test_no_reprice_small_price_change(self, order_manager):
        """Test no reprice on small price change."""
        order = Order(
            id="test",
            token_id="test",
            side=OrderSide.BUY,
            price=0.5,
            size=10.0,
            original_size=10.0,
            status=OrderStatus.LIVE,
            created_at=datetime.utcnow(),
        )

        # 1 tick difference (below threshold of 2)
        needs_reprice = order_manager.needs_reprice(order, 0.501)
        assert needs_reprice is False


class TestOrderSizeCalculation:
    """Tests for order size calculation based on risk parameters."""

    @pytest.fixture
    def risk_manager(self):
        from src.risk.risk_manager import RiskManager
        return RiskManager(max_net_frac=0.20)

    def test_order_size_within_limits(self, risk_manager):
        """Test order size calculation stays within limits."""
        size = risk_manager.get_order_size_limit(
            side="BUY",
            position_size=0,
            bankroll=1000,
            desired_size=500,
        )
        assert size == 200  # 20% of bankroll

    def test_order_size_with_existing_position(self, risk_manager):
        """Test order size accounts for existing position."""
        size = risk_manager.get_order_size_limit(
            side="BUY",
            position_size=150,  # Already 15% long
            bankroll=1000,
            desired_size=100,
        )
        assert size == 50  # Can only buy 5% more (200 - 150)

    def test_order_size_for_position_reduction(self, risk_manager):
        """Test full size allowed for position reduction."""
        size = risk_manager.get_order_size_limit(
            side="SELL",
            position_size=100,  # Long $100
            bankroll=1000,
            desired_size=50,  # Want to reduce
        )
        assert size == 50  # Full size allowed for reduction


class TestPhantomWalletIntegration:
    """Integration tests specifically for Phantom wallet setup."""

    def test_env_settings_phantom_config(self):
        """Test environment settings for Phantom wallet."""
        from src.common.config import EnvSettings
        
        # Test that default signature_type is 0 (EOA/Phantom)
        settings = EnvSettings(
            private_key="test_key",
            funder_address="0x86E35744AEAe5E693df6FbE02FF91AA566955DCc",
            signature_type=0,
        )
        
        assert settings.signature_type == 0
        assert settings.funder_address == "0x86E35744AEAe5E693df6FbE02FF91AA566955DCc"

    @patch('src.polymarket_client.client.ClobClient')
    def test_phantom_wallet_full_setup(self, mock_clob_client):
        """Test full Phantom wallet setup flow."""
        mock_instance = MagicMock()
        mock_instance.create_or_derive_api_creds.return_value = MagicMock(
            api_key="phantom_key",
            api_secret="phantom_secret",
            api_passphrase="phantom_pass",
        )
        mock_instance.get_address.return_value = "0xPhantomAddress"
        mock_clob_client.return_value = mock_instance

        client = PolymarketClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            private_key="0x" + "1" * 64,
            funder="0x86E35744AEAe5E693df6FbE02FF91AA566955DCc",
            signature_type=0,  # EOA for Phantom
        )

        # Verify setup
        call_kwargs = mock_clob_client.call_args.kwargs
        assert call_kwargs['signature_type'] == 0
        assert call_kwargs['funder'] == "0x86E35744AEAe5E693df6FbE02FF91AA566955DCc"
        
        # Verify API creds were set
        mock_instance.create_or_derive_api_creds.assert_called_once()
        mock_instance.set_api_creds.assert_called_once()
