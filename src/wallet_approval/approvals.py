"""Token approval management for Polymarket trading."""

import asyncio
from dataclasses import dataclass
from typing import Any

from web3 import Web3
from web3.contract import Contract

from src.common.errors import ApprovalError
from src.common.logging import get_logger
from .constants import (
    USDC_ADDRESS,
    CTF_ADDRESS,
    EXCHANGE_ADDRESS,
    NEG_RISK_EXCHANGE_ADDRESS,
    NEG_RISK_ADAPTER_ADDRESS,
    SPENDER_ADDRESSES,
    ERC20_ABI,
    CTF_ABI,
    MAX_UINT256,
    USDC_DECIMALS,
)

logger = get_logger(__name__)


@dataclass
class ApprovalStatus:
    """Status of token approvals."""

    # USDC approvals for each spender
    usdc_exchange_approved: bool
    usdc_neg_risk_exchange_approved: bool
    usdc_neg_risk_adapter_approved: bool

    # CTF approvals for each spender
    ctf_exchange_approved: bool
    ctf_neg_risk_exchange_approved: bool
    ctf_neg_risk_adapter_approved: bool

    @property
    def all_approved(self) -> bool:
        """Check if all required approvals are in place."""
        return (
            self.usdc_exchange_approved
            and self.usdc_neg_risk_exchange_approved
            and self.usdc_neg_risk_adapter_approved
            and self.ctf_exchange_approved
            and self.ctf_neg_risk_exchange_approved
            and self.ctf_neg_risk_adapter_approved
        )


class ApprovalManager:
    """
    Manages token approvals for Polymarket trading.

    Required approvals (for all three spender contracts):
    1. USDC approval to Exchange, NegRiskExchange, NegRiskAdapter
    2. CTF setApprovalForAll to Exchange, NegRiskExchange, NegRiskAdapter
    """

    def __init__(
        self,
        web3: Web3,
        private_key: str,
    ) -> None:
        """
        Initialize approval manager.

        Args:
            web3: Web3 instance connected to Polygon
            private_key: Private key for signing transactions
        """
        self.web3 = web3
        self.account = web3.eth.account.from_key(private_key)
        self.address = self.account.address

        # Initialize USDC contract
        self.usdc: Contract = web3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=ERC20_ABI,
        )

        # Initialize CTF contract
        self.ctf: Contract = web3.eth.contract(
            address=Web3.to_checksum_address(CTF_ADDRESS),
            abi=CTF_ABI,
        )

        logger.info("approval_manager_initialized", address=self.address, usdc=USDC_ADDRESS)

    async def _check_usdc_allowance(self, spender: str) -> int:
        """Check USDC allowance for a spender."""
        return await asyncio.to_thread(
            self.usdc.functions.allowance(
                self.address,
                Web3.to_checksum_address(spender),
            ).call
        )

    async def _check_ctf_approval(self, spender: str) -> bool:
        """Check CTF approval for a spender."""
        return await asyncio.to_thread(
            self.ctf.functions.isApprovedForAll(
                self.address,
                Web3.to_checksum_address(spender),
            ).call
        )

    async def check_approvals(self) -> ApprovalStatus:
        """
        Check current approval status for all spenders.

        Returns:
            ApprovalStatus with current state
        """
        try:
            min_allowance = 1000 * (10 ** USDC_DECIMALS)

            # Check USDC allowances for all spenders
            usdc_exchange = await self._check_usdc_allowance(EXCHANGE_ADDRESS)
            usdc_neg_risk_exchange = await self._check_usdc_allowance(NEG_RISK_EXCHANGE_ADDRESS)
            usdc_neg_risk_adapter = await self._check_usdc_allowance(NEG_RISK_ADAPTER_ADDRESS)

            # Check CTF approvals for all spenders
            ctf_exchange = await self._check_ctf_approval(EXCHANGE_ADDRESS)
            ctf_neg_risk_exchange = await self._check_ctf_approval(NEG_RISK_EXCHANGE_ADDRESS)
            ctf_neg_risk_adapter = await self._check_ctf_approval(NEG_RISK_ADAPTER_ADDRESS)

            status = ApprovalStatus(
                usdc_exchange_approved=usdc_exchange >= min_allowance,
                usdc_neg_risk_exchange_approved=usdc_neg_risk_exchange >= min_allowance,
                usdc_neg_risk_adapter_approved=usdc_neg_risk_adapter >= min_allowance,
                ctf_exchange_approved=ctf_exchange,
                ctf_neg_risk_exchange_approved=ctf_neg_risk_exchange,
                ctf_neg_risk_adapter_approved=ctf_neg_risk_adapter,
            )

            logger.info(
                "approvals_checked",
                usdc_exchange=status.usdc_exchange_approved,
                usdc_neg_risk_exchange=status.usdc_neg_risk_exchange_approved,
                usdc_neg_risk_adapter=status.usdc_neg_risk_adapter_approved,
                ctf_exchange=status.ctf_exchange_approved,
                ctf_neg_risk_exchange=status.ctf_neg_risk_exchange_approved,
                ctf_neg_risk_adapter=status.ctf_neg_risk_adapter_approved,
                all_approved=status.all_approved,
            )

            return status

        except Exception as e:
            raise ApprovalError(f"Failed to check approvals: {e}")

    async def _approve_usdc_for_spender(self, spender: str, spender_name: str) -> str:
        """Approve USDC for a specific spender."""
        try:
            tx = self.usdc.functions.approve(
                Web3.to_checksum_address(spender),
                MAX_UINT256,
            ).build_transaction({
                "from": self.address,
                "nonce": await asyncio.to_thread(
                    self.web3.eth.get_transaction_count, self.address, 'pending'
                ),
                "gas": 100000,
                "gasPrice": self.web3.eth.gas_price,
            })

            signed = self.account.sign_transaction(tx)
            tx_hash = await asyncio.to_thread(
                self.web3.eth.send_raw_transaction, signed.raw_transaction
            )

            logger.info("usdc_approval_sent", spender=spender_name, tx_hash=tx_hash.hex())

            receipt = await asyncio.to_thread(
                self.web3.eth.wait_for_transaction_receipt, tx_hash, timeout=120
            )

            if receipt["status"] != 1:
                raise ApprovalError(f"USDC approval to {spender_name} failed")

            logger.info("usdc_approval_confirmed", spender=spender_name, tx_hash=tx_hash.hex())
            return tx_hash.hex()

        except ApprovalError:
            raise
        except Exception as e:
            raise ApprovalError(f"Failed to approve USDC for {spender_name}: {e}")

    async def _approve_ctf_for_spender(self, spender: str, spender_name: str) -> str:
        """Approve CTF for a specific spender."""
        try:
            tx = self.ctf.functions.setApprovalForAll(
                Web3.to_checksum_address(spender),
                True,
            ).build_transaction({
                "from": self.address,
                "nonce": await asyncio.to_thread(
                    self.web3.eth.get_transaction_count, self.address, 'pending'
                ),
                "gas": 100000,
                "gasPrice": self.web3.eth.gas_price,
            })

            signed = self.account.sign_transaction(tx)
            tx_hash = await asyncio.to_thread(
                self.web3.eth.send_raw_transaction, signed.raw_transaction
            )

            logger.info("ctf_approval_sent", spender=spender_name, tx_hash=tx_hash.hex())

            receipt = await asyncio.to_thread(
                self.web3.eth.wait_for_transaction_receipt, tx_hash, timeout=120
            )

            if receipt["status"] != 1:
                raise ApprovalError(f"CTF approval to {spender_name} failed")

            logger.info("ctf_approval_confirmed", spender=spender_name, tx_hash=tx_hash.hex())
            return tx_hash.hex()

        except ApprovalError:
            raise
        except Exception as e:
            raise ApprovalError(f"Failed to approve CTF for {spender_name}: {e}")

    async def ensure_approvals(self) -> dict[str, str]:
        """
        Ensure all required approvals are in place for all spenders.

        Returns:
            Dict of approval type -> tx hash (empty if already approved)
        """
        status = await self.check_approvals()
        tx_hashes: dict[str, str] = {}

        # USDC approvals
        if not status.usdc_exchange_approved:
            tx_hashes["usdc_exchange"] = await self._approve_usdc_for_spender(
                EXCHANGE_ADDRESS, "Exchange"
            )

        if not status.usdc_neg_risk_exchange_approved:
            tx_hashes["usdc_neg_risk_exchange"] = await self._approve_usdc_for_spender(
                NEG_RISK_EXCHANGE_ADDRESS, "NegRiskExchange"
            )

        if not status.usdc_neg_risk_adapter_approved:
            tx_hashes["usdc_neg_risk_adapter"] = await self._approve_usdc_for_spender(
                NEG_RISK_ADAPTER_ADDRESS, "NegRiskAdapter"
            )

        # CTF approvals
        if not status.ctf_exchange_approved:
            tx_hashes["ctf_exchange"] = await self._approve_ctf_for_spender(
                EXCHANGE_ADDRESS, "Exchange"
            )

        if not status.ctf_neg_risk_exchange_approved:
            tx_hashes["ctf_neg_risk_exchange"] = await self._approve_ctf_for_spender(
                NEG_RISK_EXCHANGE_ADDRESS, "NegRiskExchange"
            )

        if not status.ctf_neg_risk_adapter_approved:
            tx_hashes["ctf_neg_risk_adapter"] = await self._approve_ctf_for_spender(
                NEG_RISK_ADAPTER_ADDRESS, "NegRiskAdapter"
            )

        if tx_hashes:
            logger.info("approvals_set", transactions=list(tx_hashes.keys()))
        else:
            logger.info("all_approvals_already_set")

        return tx_hashes
