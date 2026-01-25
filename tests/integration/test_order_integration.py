"""
Integration tests for order placement.

These tests require a valid .env file with:
- POLYBOT_PRIVATE_KEY: Your wallet private key
- POLYBOT_FUNDER_ADDRESS: Your funder address (0x86E35744AEAe5E693df6FbE02FF91AA566955DCc)
- POLYBOT_SIGNATURE_TYPE: 0 for Phantom wallet (EOA)

Run with: pytest tests/integration/test_order_integration.py -v -s
"""

import pytest
import asyncio
import os
from datetime import datetime
from pathlib import Path

# Skip all tests in this file if no .env credentials
pytestmark = pytest.mark.skipif(
    not os.getenv("POLYBOT_PRIVATE_KEY"),
    reason="No POLYBOT_PRIVATE_KEY in environment"
)


class TestOrderIntegration:
    """Integration tests for live order placement."""

    @pytest.fixture
    def env_settings(self):
        """Load environment settings."""
        from src.common.config import load_env_settings
        return load_env_settings()

    @pytest.fixture
    def config(self):
        """Load config or use defaults."""
        from src.common.config import Config, NetworkConfig
        return Config(
            network=NetworkConfig(
                chain_id=137,
                clob_host="https://clob.polymarket.com",
            )
        )

    @pytest.fixture
    async def client(self, env_settings, config):
        """Create live Polymarket client."""
        from src.polymarket_client.client import PolymarketClient
        
        print(f"\n--- Client Configuration ---")
        print(f"Host: {config.network.clob_host}")
        print(f"Chain ID: {config.network.chain_id}")
        print(f"Funder: {env_settings.funder_address}")
        print(f"Signature Type: {env_settings.signature_type}")
        
        client = PolymarketClient(
            host=config.network.clob_host,
            chain_id=config.network.chain_id,
            private_key=env_settings.private_key,
            funder=env_settings.funder_address if env_settings.funder_address else None,
            signature_type=env_settings.signature_type,
        )
        
        print(f"Wallet Address: {client.address}")
        return client

    @pytest.mark.asyncio
    async def test_client_connection(self, client):
        """Test basic client connectivity."""
        # Should be able to get address
        address = client.address
        assert address is not None
        assert address.startswith("0x")
        print(f"\n✓ Connected with address: {address}")

    @pytest.mark.asyncio
    async def test_get_balances(self, client):
        """Test fetching account balances."""
        try:
            balances = await client.get_balances()
            print(f"\n--- Account Balances ---")
            if balances:
                for token, balance in list(balances.items())[:5]:
                    print(f"  {token[:20]}...: {balance}")
            else:
                print("  No balances found")
            # Test passes regardless of balance amount
            assert isinstance(balances, dict)
        except Exception as e:
            print(f"\n✗ Failed to get balances: {e}")
            raise

    @pytest.mark.asyncio
    async def test_get_open_orders(self, client):
        """Test fetching open orders."""
        try:
            orders = await client.get_open_orders()
            print(f"\n--- Open Orders: {len(orders)} ---")
            for order in orders[:5]:
                print(f"  {order.id[:16]}... {order.side.value} {order.size}@{order.price}")
            assert isinstance(orders, list)
        except Exception as e:
            print(f"\n✗ Failed to get orders: {e}")
            raise

    @pytest.mark.asyncio
    async def test_get_order_book(self, client):
        """Test fetching order book for a known market."""
        # Use a known active BTC market token ID
        # This is a sample - you may need to update with a current market
        from src.polymarket_client.market_discovery import MarketDiscovery
        
        try:
            discovery = MarketDiscovery()
            market = await discovery.find_btc_market(interval="1h")
            await discovery.close()
            
            if market is None:
                pytest.skip("No active BTC market found")
            
            token_id = market.token_id_yes
            print(f"\n--- Testing Order Book for Token: {token_id[:20]}... ---")
            
            book = await client.get_order_book(token_id)
            
            print(f"  Best Bid: {book.best_bid_price}")
            print(f"  Best Ask: {book.best_ask_price}")
            print(f"  Spread: {book.spread} ({book.spread_bps:.1f} bps)")
            print(f"  Bid Levels: {len(book.bids)}")
            print(f"  Ask Levels: {len(book.asks)}")
            
            assert book.best_bid_price is not None or book.best_ask_price is not None
            
        except Exception as e:
            print(f"\n✗ Failed to get order book: {e}")
            raise


class TestOrderPlacementDiagnostics:
    """Diagnostic tests to identify order placement issues."""

    @pytest.fixture
    def env_settings(self):
        from src.common.config import load_env_settings
        return load_env_settings()

    @pytest.mark.asyncio
    async def test_api_credentials(self, env_settings):
        """Test API credential generation."""
        from py_clob_client.client import ClobClient
        
        print(f"\n--- API Credential Test ---")
        print(f"Signature Type: {env_settings.signature_type}")
        print(f"Funder Address: {env_settings.funder_address}")
        
        try:
            client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,
                key=env_settings.private_key,
                signature_type=env_settings.signature_type,
                funder=env_settings.funder_address if env_settings.funder_address else None,
            )
            
            print(f"Signer Address: {client.get_address()}")
            
            # Try to create/derive API creds
            api_creds = client.create_or_derive_api_creds()
            print(f"✓ API Key: {api_creds.api_key[:8]}...")
            
            client.set_api_creds(api_creds)
            print(f"✓ API credentials set successfully")
            
        except Exception as e:
            print(f"\n✗ API credential error: {e}")
            print(f"\nCommon causes:")
            print(f"  1. Wrong signature_type (use 0 for Phantom/EOA)")
            print(f"  2. Invalid private key format")
            print(f"  3. Funder address mismatch")
            raise

    @pytest.mark.asyncio
    async def test_allowance_sync(self, env_settings):
        """Test CLOB allowance synchronization."""
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        
        print(f"\n--- Allowance Sync Test ---")
        
        try:
            client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,
                key=env_settings.private_key,
                signature_type=env_settings.signature_type,
                funder=env_settings.funder_address if env_settings.funder_address else None,
            )
            
            # Set API creds
            api_creds = client.create_or_derive_api_creds()
            client.set_api_creds(api_creds)
            
            # Sync USDC allowance
            print("Syncing USDC (collateral) allowance...")
            try:
                usdc_params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                result = client.update_balance_allowance(usdc_params)
                print(f"✓ USDC allowance synced: {result}")
            except Exception as e:
                print(f"✗ USDC sync failed: {e}")
            
            # Get current balance/allowance
            print("\nFetching balance/allowance...")
            try:
                balance = client.get_balance_allowance()
                print(f"Balance info: {balance}")
            except Exception as e:
                print(f"Could not fetch balance: {e}")
                
        except Exception as e:
            print(f"\n✗ Allowance sync error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_order_signature_generation(self, env_settings):
        """Test order signature generation without posting."""
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY
        
        print(f"\n--- Order Signature Test ---")
        
        try:
            client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,
                key=env_settings.private_key,
                signature_type=env_settings.signature_type,
                funder=env_settings.funder_address if env_settings.funder_address else None,
            )
            
            api_creds = client.create_or_derive_api_creds()
            client.set_api_creds(api_creds)
            
            # Find a real market for testing
            from src.polymarket_client.market_discovery import MarketDiscovery
            discovery = MarketDiscovery()
            market = await discovery.find_btc_market(interval="1h")
            await discovery.close()
            
            if market is None:
                pytest.skip("No active BTC market found")
            
            token_id = market.token_id_yes
            
            # Get tick size and neg_risk
            tick_size = client.get_tick_size(token_id)
            neg_risk = client.get_neg_risk(token_id)
            
            print(f"Token ID: {token_id[:30]}...")
            print(f"Tick Size: {tick_size}")
            print(f"Neg Risk: {neg_risk}")
            
            # Create order args (use a low price that won't fill)
            order_args = OrderArgs(
                token_id=token_id,
                price=0.01,  # Very low price
                size=1.0,
                side=BUY,
                expiration=0,
            )
            
            # Create signed order
            from py_clob_client.clob_types import PartialCreateOrderOptions
            options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
            
            signed_order = client.create_order(order_args, options)
            print(f"✓ Order signed successfully")
            print(f"  Signature present: {bool(signed_order.get('signature') or signed_order.get('signatures'))}")
            
        except Exception as e:
            print(f"\n✗ Order signature error: {e}")
            print(f"\nCommon causes:")
            print(f"  1. Invalid token_id")
            print(f"  2. Wrong signature_type for your wallet")
            print(f"  3. API credentials not set")
            raise


class TestCommonOrderFailures:
    """Tests to diagnose common order failure scenarios."""

    @pytest.fixture
    def env_settings(self):
        from src.common.config import load_env_settings
        return load_env_settings()

    @pytest.mark.asyncio
    async def test_insufficient_balance_detection(self, env_settings):
        """Test detection of insufficient balance."""
        from src.polymarket_client.client import PolymarketClient
        
        print(f"\n--- Insufficient Balance Test ---")
        
        client = PolymarketClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            private_key=env_settings.private_key,
            funder=env_settings.funder_address if env_settings.funder_address else None,
            signature_type=env_settings.signature_type,
        )
        
        balances = await client.get_balances()
        
        # Check USDC balance (collateral)
        usdc_balance = 0.0
        for token, balance in balances.items():
            # USDC/collateral tokens often have specific identifiers
            if balance > usdc_balance:
                usdc_balance = balance
        
        print(f"Estimated max balance: ${usdc_balance:.2f}")
        
        if usdc_balance < 1.0:
            print(f"⚠ Warning: Very low balance. Orders may fail due to insufficient funds.")
        else:
            print(f"✓ Balance appears sufficient for testing")

    @pytest.mark.asyncio  
    async def test_market_active_check(self, env_settings):
        """Test that the target market is active and accepting orders."""
        from src.polymarket_client.market_discovery import MarketDiscovery
        
        print(f"\n--- Market Activity Test ---")
        
        discovery = MarketDiscovery()
        
        for interval in ["1h", "4h", "1d"]:
            try:
                market = await discovery.find_btc_market(interval=interval)
                if market:
                    print(f"✓ Found {interval} market: {market.question[:50]}...")
                    print(f"  End Date: {market.end_date}")
                    print(f"  Tokens: YES={market.token_id_yes[:16]}... NO={market.token_id_no[:16]}...")
                else:
                    print(f"✗ No {interval} market found")
            except Exception as e:
                print(f"✗ Error finding {interval} market: {e}")
        
        await discovery.close()

    @pytest.mark.asyncio
    async def test_price_validation(self, env_settings):
        """Test that order prices are valid."""
        from src.polymarket_client.orders import OrderManager
        from src.polymarket_client.types import OrderSide
        from unittest.mock import Mock, AsyncMock
        
        print(f"\n--- Price Validation Test ---")
        
        # Create mock client
        mock_client = Mock()
        mock_client.place_limit_order = AsyncMock()
        
        manager = OrderManager(client=mock_client)
        
        # Test boundary prices
        test_prices = [0.001, 0.01, 0.5, 0.99, 0.999, 0.0001, 1.0, 1.5]
        
        for price in test_prices:
            result = await manager.place_order(
                token_id="test",
                side=OrderSide.BUY,
                price=price,
                size=1.0,
            )
            status = "✓ Accepted" if mock_client.place_limit_order.called else "✗ Rejected"
            print(f"  Price {price}: {status}")
            mock_client.place_limit_order.reset_mock()


def run_diagnostics():
    """Run all diagnostic tests and print summary."""
    print("=" * 60)
    print("POLYMARKET ORDER PLACEMENT DIAGNOSTICS")
    print("=" * 60)
    print("\nRun with: pytest tests/integration/test_order_integration.py -v -s")
    print("\nMake sure your .env file contains:")
    print("  POLYBOT_PRIVATE_KEY=<your_private_key>")
    print("  POLYBOT_FUNDER_ADDRESS=0x86E35744AEAe5E693df6FbE02FF91AA566955DCc")
    print("  POLYBOT_SIGNATURE_TYPE=0  # For Phantom wallet")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
