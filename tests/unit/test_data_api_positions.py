"""Tests for Data API position tracking."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime


class TestDataAPIPositions:
    """Tests for get_positions_from_data_api."""

    @pytest.fixture
    def sample_data_api_response(self):
        """Sample response from the Data API."""
        return [
            {
                "asset": "0xtoken1",
                "conditionId": "0xcondition1",
                "size": "100.5",
                "curPrice": "0.75",
                "avgPrice": "0.50",
                "currentValue": "75.375",
                "initialValue": "50.25",
                "title": "Will BTC be up?",
                "outcome": "Yes",
                "negativeRisk": False,
                "redeemable": False,
                "mergeable": False,
            },
            {
                "asset": "0xtoken2",
                "conditionId": "0xcondition2",
                "size": "50.0",
                "curPrice": "0.60",
                "avgPrice": "0.40",
                "currentValue": "30.0",
                "initialValue": "20.0",
                "title": "Will ETH be up?",
                "outcome": "Yes",
                "negativeRisk": False,
                "redeemable": False,
                "mergeable": False,
            },
            # Dust position (should be filtered)
            {
                "asset": "0xtoken3",
                "conditionId": "0xcondition3",
                "size": "0.005",  # Below 0.01 threshold
                "curPrice": "0.50",
                "avgPrice": "0.50",
                "currentValue": "0.0025",
                "initialValue": "0.0025",
                "title": "Dust position",
                "outcome": "Yes",
                "negativeRisk": False,
                "redeemable": False,
                "mergeable": False,
            },
        ]

    @pytest.mark.asyncio
    async def test_get_positions_parses_response(self, sample_data_api_response):
        """Test that positions are correctly parsed from API response."""
        from src.polymarket_client.client import PolymarketClient

        # Create a mock client
        with patch.object(PolymarketClient, '__init__', lambda self, *args, **kwargs: None):
            client = PolymarketClient.__new__(PolymarketClient)
            client._client = MagicMock()
            client._client.get_address.return_value = "0xwallet"

            # Mock the aiohttp session
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_data_api_response)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            with patch('aiohttp.ClientSession', return_value=mock_session_instance):
                positions = await client.get_positions_from_data_api()

        # Should have 2 positions (dust filtered out)
        assert len(positions) == 2

        # Check first position
        pos1 = positions[0]
        assert pos1["token_id"] == "0xtoken1"
        assert pos1["size"] == 100.5
        assert pos1["cur_price"] == 0.75
        assert pos1["avg_price"] == 0.50
        assert pos1["current_value"] == 75.375
        assert pos1["initial_value"] == 50.25
        assert pos1["unrealized_pnl"] == pytest.approx(25.125, rel=0.01)  # 75.375 - 50.25
        assert pos1["title"] == "Will BTC be up?"

        # Check second position
        pos2 = positions[1]
        assert pos2["token_id"] == "0xtoken2"
        assert pos2["size"] == 50.0
        assert pos2["unrealized_pnl"] == pytest.approx(10.0, rel=0.01)  # 30 - 20

    @pytest.mark.asyncio
    async def test_get_positions_handles_empty_response(self):
        """Test handling of empty positions."""
        from src.polymarket_client.client import PolymarketClient

        with patch.object(PolymarketClient, '__init__', lambda self, *args, **kwargs: None):
            client = PolymarketClient.__new__(PolymarketClient)
            client._client = MagicMock()
            client._client.get_address.return_value = "0xwallet"

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[])

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            with patch('aiohttp.ClientSession', return_value=mock_session_instance):
                positions = await client.get_positions_from_data_api()

        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_handles_api_error(self):
        """Test handling of API errors."""
        from src.polymarket_client.client import PolymarketClient

        with patch.object(PolymarketClient, '__init__', lambda self, *args, **kwargs: None):
            client = PolymarketClient.__new__(PolymarketClient)
            client._client = MagicMock()
            client._client.get_address.return_value = "0xwallet"

            mock_response = AsyncMock()
            mock_response.status = 500

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            with patch('aiohttp.ClientSession', return_value=mock_session_instance):
                positions = await client.get_positions_from_data_api()

        # Should return empty list on error, not raise
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_detects_resolved_positions(self):
        """Test that resolved positions are correctly identified."""
        from src.polymarket_client.client import PolymarketClient

        data_response = [
            {
                "asset": "0xwinning",
                "conditionId": "0xcondition",
                "size": "100",
                "curPrice": "0.99",  # Winning position
                "avgPrice": "0.50",
                "currentValue": "99",
                "initialValue": "50",
                "title": "Winning",
                "outcome": "Yes",
                "negativeRisk": False,
            },
            {
                "asset": "0xlosing",
                "conditionId": "0xcondition",
                "size": "100",
                "curPrice": "0.01",  # Losing position
                "avgPrice": "0.50",
                "currentValue": "1",
                "initialValue": "50",
                "title": "Losing",
                "outcome": "Yes",
                "negativeRisk": False,
            },
            {
                "asset": "0xactive",
                "conditionId": "0xcondition",
                "size": "100",
                "curPrice": "0.60",  # Active position
                "avgPrice": "0.50",
                "currentValue": "60",
                "initialValue": "50",
                "title": "Active",
                "outcome": "Yes",
                "negativeRisk": False,
            },
        ]

        with patch.object(PolymarketClient, '__init__', lambda self, *args, **kwargs: None):
            client = PolymarketClient.__new__(PolymarketClient)
            client._client = MagicMock()
            client._client.get_address.return_value = "0xwallet"

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=data_response)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            with patch('aiohttp.ClientSession', return_value=mock_session_instance):
                positions = await client.get_positions_from_data_api()

        assert len(positions) == 3

        winning = next(p for p in positions if p["token_id"] == "0xwinning")
        assert winning["is_resolved_winning"] is True
        assert winning["is_resolved_losing"] is False

        losing = next(p for p in positions if p["token_id"] == "0xlosing")
        assert losing["is_resolved_winning"] is False
        assert losing["is_resolved_losing"] is True

        active = next(p for p in positions if p["token_id"] == "0xactive")
        assert active["is_resolved_winning"] is False
        assert active["is_resolved_losing"] is False


class TestPnLCalculation:
    """Tests for PnL calculation accuracy."""

    def test_unrealized_pnl_calculation(self):
        """Test that unrealized PnL is calculated correctly."""
        # Position: bought 100 shares at $0.50, now worth $0.75
        size = 100.0
        avg_price = 0.50
        cur_price = 0.75

        initial_value = size * avg_price  # $50
        current_value = size * cur_price  # $75
        unrealized_pnl = current_value - initial_value  # $25

        assert initial_value == 50.0
        assert current_value == 75.0
        assert unrealized_pnl == 25.0

    def test_unrealized_pnl_loss(self):
        """Test unrealized PnL when position is losing."""
        size = 100.0
        avg_price = 0.60
        cur_price = 0.40

        initial_value = size * avg_price  # $60
        current_value = size * cur_price  # $40
        unrealized_pnl = current_value - initial_value  # -$20

        assert unrealized_pnl == -20.0

    def test_total_pnl_aggregation(self):
        """Test aggregating PnL across multiple positions."""
        positions = [
            {"size": 100, "current_value": 75, "initial_value": 50, "unrealized_pnl": 25},
            {"size": 50, "current_value": 30, "initial_value": 20, "unrealized_pnl": 10},
            {"size": 200, "current_value": 80, "initial_value": 100, "unrealized_pnl": -20},
        ]

        total_unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)
        total_current_value = sum(p["current_value"] for p in positions)
        total_initial_value = sum(p["initial_value"] for p in positions)

        assert total_unrealized_pnl == 15  # 25 + 10 - 20
        assert total_current_value == 185  # 75 + 30 + 80
        assert total_initial_value == 170  # 50 + 20 + 100
        assert total_current_value - total_initial_value == total_unrealized_pnl
