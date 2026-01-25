"""Pytest configuration and fixtures."""

import pytest
from datetime import datetime


@pytest.fixture
def sample_orderbook():
    """Sample orderbook data for testing."""
    return {
        "bids": [
            {"price": 0.45, "size": 100},
            {"price": 0.44, "size": 200},
            {"price": 0.43, "size": 150},
        ],
        "asks": [
            {"price": 0.55, "size": 100},
            {"price": 0.56, "size": 200},
            {"price": 0.57, "size": 150},
        ],
    }


@pytest.fixture
def sample_token_id():
    """Sample token ID for testing."""
    return "0x1234567890abcdef1234567890abcdef12345678"


@pytest.fixture
def sample_timestamp():
    """Sample timestamp for testing."""
    return datetime(2024, 1, 15, 12, 0, 0)
