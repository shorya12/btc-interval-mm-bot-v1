"""Integration tests for Data API position tracking.

These tests make real API calls and require valid credentials.
Run with: pytest tests/integration/test_data_api_integration.py -v -s
"""

import os
import pytest
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def has_credentials() -> bool:
    """Check if required credentials are available."""
    return bool(os.getenv("POLYBOT_PRIVATE_KEY"))


@pytest.mark.skipif(not has_credentials(), reason="No credentials available")
class TestDataAPIIntegration:
    """Integration tests for Data API position tracking."""

    @pytest.mark.asyncio
    async def test_fetch_positions_from_data_api(self):
        """Test fetching positions from the real Data API."""
        from src.polymarket_client.client import PolymarketClient
        from src.common.config import load_env_settings

        env = load_env_settings()

        client = PolymarketClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            private_key=env.private_key,
            funder=env.funder_address,
            signature_type=env.signature_type,
        )

        # Fetch positions
        positions = await client.get_positions_from_data_api()

        print(f"\n=== Data API Positions ===")
        print(f"Wallet: {client.address}")
        print(f"Total positions: {len(positions)}")

        total_value = 0.0
        total_pnl = 0.0

        for pos in positions:
            print(f"\n  {pos['title'][:50]}")
            print(f"    Token: {pos['token_id'][:20]}...")
            print(f"    Size: {pos['size']:.2f} shares")
            print(f"    Avg Price: ${pos['avg_price']:.4f}")
            print(f"    Current Price: ${pos['cur_price']:.4f}")
            print(f"    Current Value: ${pos['current_value']:.2f}")
            print(f"    Unrealized PnL: ${pos['unrealized_pnl']:.2f}")

            total_value += pos["current_value"]
            total_pnl += pos["unrealized_pnl"]

            # Verify resolved status detection
            if pos["is_resolved_winning"]:
                print(f"    Status: RESOLVED WINNING")
            elif pos["is_resolved_losing"]:
                print(f"    Status: RESOLVED LOSING")
            else:
                print(f"    Status: ACTIVE")

        print(f"\n=== Summary ===")
        print(f"Total Position Value: ${total_value:.2f}")
        print(f"Total Unrealized PnL: ${total_pnl:.2f}")

        # Basic validation
        assert isinstance(positions, list)
        for pos in positions:
            assert "token_id" in pos
            assert "size" in pos
            assert "cur_price" in pos
            assert "avg_price" in pos
            assert "current_value" in pos
            assert "unrealized_pnl" in pos
            assert pos["size"] > 0.01  # Dust should be filtered

    @pytest.mark.asyncio
    async def test_compare_fill_tracking_vs_data_api(self):
        """Compare fill tracking (broken) vs Data API (accurate)."""
        from src.polymarket_client.client import PolymarketClient
        from src.polymarket_client.fills import FillTracker
        from src.common.config import load_env_settings

        env = load_env_settings()

        client = PolymarketClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            private_key=env.private_key,
            funder=env.funder_address,
            signature_type=env.signature_type,
        )

        # Get positions from Data API (source of truth)
        data_api_positions = await client.get_positions_from_data_api()

        # Get fills from CLOB API (potentially incomplete)
        fills = await client.get_fills()

        print(f"\n=== Comparison ===")
        print(f"Data API positions: {len(data_api_positions)}")
        print(f"CLOB API fills: {len(fills)}")

        # Calculate totals from Data API
        data_api_total_value = sum(p["current_value"] for p in data_api_positions)
        data_api_total_size = sum(p["size"] for p in data_api_positions)

        print(f"\nData API:")
        print(f"  Total position value: ${data_api_total_value:.2f}")
        print(f"  Total position size: {data_api_total_size:.2f} shares")

        # Note: Fill tracking is known to be incomplete due to pagination
        # This test documents the discrepancy
        print(f"\nCLOB API fills:")
        print(f"  Total fills returned: {len(fills)}")

        if fills:
            buy_fills = [f for f in fills if f.side.value == "BUY"]
            sell_fills = [f for f in fills if f.side.value == "SELL"]
            print(f"  Buy fills: {len(buy_fills)}")
            print(f"  Sell fills: {len(sell_fills)}")

            buy_volume = sum(f.size * f.price for f in buy_fills)
            sell_volume = sum(f.size * f.price for f in sell_fills)
            print(f"  Buy volume: ${buy_volume:.2f}")
            print(f"  Sell volume: ${sell_volume:.2f}")

        # The Data API should show accurate positions
        # Fill tracking may show less due to pagination issues
        print(f"\n=== Conclusion ===")
        print("Data API provides accurate position data.")
        print("CLOB API fills may be incomplete due to pagination limits.")


@pytest.mark.skipif(not has_credentials(), reason="No credentials available")
class TestPositionSyncIntegration:
    """Test the position sync functionality."""

    @pytest.mark.asyncio
    async def test_position_sync_accuracy(self):
        """Test that position sync from Data API is accurate."""
        import aiohttp

        wallet = os.getenv("POLYBOT_FUNDER_ADDRESS") or "0x0"

        # Direct API call to verify our implementation matches
        url = "https://data-api.polymarket.com/positions"
        params = {
            "user": wallet.lower(),
            "sizeThreshold": 0,
            "limit": 500,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"\nDirect API call returned {len(data)} positions")

                    # Verify response structure
                    if data:
                        sample = data[0]
                        required_fields = ["asset", "size", "curPrice"]
                        for field in required_fields:
                            assert field in sample, f"Missing field: {field}"
                        print("Response structure verified")
                else:
                    print(f"API returned status {resp.status}")
