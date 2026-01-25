"""Wallet and approval management for Polygon."""

from .constants import (
    POLYGON_CHAIN_ID,
    USDC_ADDRESS,
    CTF_ADDRESS,
    EXCHANGE_ADDRESS,
    NEG_RISK_ADAPTER_ADDRESS,
    NEG_RISK_EXCHANGE_ADDRESS,
    SPENDER_ADDRESSES,
)
from .approvals import ApprovalManager, ApprovalStatus
from .account import AccountManager

__all__ = [
    "POLYGON_CHAIN_ID",
    "USDC_ADDRESS",
    "CTF_ADDRESS",
    "EXCHANGE_ADDRESS",
    "NEG_RISK_ADAPTER_ADDRESS",
    "NEG_RISK_EXCHANGE_ADDRESS",
    "SPENDER_ADDRESSES",
    "ApprovalManager",
    "ApprovalStatus",
    "AccountManager",
]
