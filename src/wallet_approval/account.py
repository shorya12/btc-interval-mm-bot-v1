"""Account and balance management."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal

from web3 import Web3

from src.common.errors import InsufficientBalanceError
from src.common.logging import get_logger
from .constants import (
    USDC_ADDRESS,
    CTF_ADDRESS,
    ERC20_ABI,
    CTF_ABI,
    USDC_DECIMALS,
)

logger = get_logger(__name__)


@dataclass
class AccountBalance:
    """Account balance information."""

    address: str
    matic_balance: Decimal  # Native MATIC for gas
    usdc_balance: Decimal  # USDC.e balance
    matic_wei: int
    usdc_raw: int

    @property
    def has_gas(self) -> bool:
        """Check if account has enough MATIC for gas (~0.01 MATIC)."""
        return self.matic_balance >= Decimal("0.01")

    @property
    def usdc_balance_float(self) -> float:
        """Get USDC balance as float."""
        return float(self.usdc_balance)


@dataclass
class TokenBalance:
    """Balance of a specific outcome token."""

    token_id: str
    balance: int  # Raw balance (18 decimals typically)
    balance_decimal: Decimal

    @property
    def balance_float(self) -> float:
        """Get balance as float."""
        return float(self.balance_decimal)


class AccountManager:
    """
    Manages account balances and nonces.

    Tracks:
    - MATIC balance for gas
    - USDC.e balance for collateral
    - CTF token balances for positions
    """

    def __init__(
        self,
        web3: Web3,
        address: str,
    ) -> None:
        """
        Initialize account manager.

        Args:
            web3: Web3 instance
            address: Account address to manage
        """
        self.web3 = web3
        self.address = Web3.to_checksum_address(address)

        # Initialize contracts
        self.usdc = web3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=ERC20_ABI,
        )
        self.ctf = web3.eth.contract(
            address=Web3.to_checksum_address(CTF_ADDRESS),
            abi=CTF_ABI,
        )

        self._cached_nonce: int | None = None
        logger.info("account_manager_initialized", address=self.address)

    async def get_balances(self) -> AccountBalance:
        """
        Get current account balances.

        Returns:
            AccountBalance with MATIC and USDC balances
        """
        # Fetch balances in parallel
        matic_task = asyncio.to_thread(self.web3.eth.get_balance, self.address)
        usdc_task = asyncio.to_thread(
            self.usdc.functions.balanceOf(self.address).call
        )

        matic_wei, usdc_raw = await asyncio.gather(matic_task, usdc_task)

        matic_balance = Decimal(matic_wei) / Decimal(10 ** 18)
        usdc_balance = Decimal(usdc_raw) / Decimal(10 ** USDC_DECIMALS)

        balance = AccountBalance(
            address=self.address,
            matic_balance=matic_balance,
            usdc_balance=usdc_balance,
            matic_wei=matic_wei,
            usdc_raw=usdc_raw,
        )

        logger.debug(
            "balances_fetched",
            matic=float(matic_balance),
            usdc=float(usdc_balance),
        )

        return balance

    async def get_token_balance(self, token_id: str) -> TokenBalance:
        """
        Get balance of a specific CTF token.

        Args:
            token_id: Token ID (as hex string or int)

        Returns:
            TokenBalance
        """
        # Convert token_id to int if it's a hex string
        if isinstance(token_id, str):
            if token_id.startswith("0x"):
                token_id_int = int(token_id, 16)
            else:
                token_id_int = int(token_id)
        else:
            token_id_int = token_id

        balance_raw = await asyncio.to_thread(
            self.ctf.functions.balanceOf(self.address, token_id_int).call
        )

        # CTF tokens typically have 18 decimals
        balance_decimal = Decimal(balance_raw) / Decimal(10 ** 18)

        return TokenBalance(
            token_id=str(token_id),
            balance=balance_raw,
            balance_decimal=balance_decimal,
        )

    async def get_nonce(self, refresh: bool = False) -> int:
        """
        Get current nonce for transactions.

        Args:
            refresh: Force refresh from chain

        Returns:
            Current nonce
        """
        if self._cached_nonce is None or refresh:
            self._cached_nonce = await asyncio.to_thread(
                self.web3.eth.get_transaction_count, self.address
            )
        return self._cached_nonce

    def increment_nonce(self) -> int:
        """
        Increment cached nonce after sending transaction.

        Returns:
            New nonce value
        """
        if self._cached_nonce is not None:
            self._cached_nonce += 1
        return self._cached_nonce or 0

    async def check_usdc_balance(self, required: float) -> None:
        """
        Check if account has sufficient USDC balance.

        Args:
            required: Required USDC amount

        Raises:
            InsufficientBalanceError: If balance is insufficient
        """
        balance = await self.get_balances()

        if balance.usdc_balance_float < required:
            raise InsufficientBalanceError(
                "Insufficient USDC balance",
                required=required,
                available=balance.usdc_balance_float,
                asset="USDC",
            )

    async def check_gas_balance(self) -> None:
        """
        Check if account has sufficient MATIC for gas.

        Raises:
            InsufficientBalanceError: If balance is insufficient
        """
        balance = await self.get_balances()

        if not balance.has_gas:
            raise InsufficientBalanceError(
                "Insufficient MATIC for gas",
                required=0.01,
                available=float(balance.matic_balance),
                asset="MATIC",
            )

    async def estimate_gas_cost(self, gas_limit: int = 100000) -> Decimal:
        """
        Estimate gas cost in MATIC.

        Args:
            gas_limit: Gas limit for transaction

        Returns:
            Estimated cost in MATIC
        """
        gas_price = await asyncio.to_thread(
            lambda: self.web3.eth.gas_price
        )
        cost_wei = gas_price * gas_limit
        return Decimal(cost_wei) / Decimal(10 ** 18)

    async def get_position_value(
        self,
        token_id: str,
        current_price: float,
    ) -> float:
        """
        Get value of token position at current price.

        Args:
            token_id: Token ID
            current_price: Current market price

        Returns:
            Position value in USDC
        """
        token_balance = await self.get_token_balance(token_id)
        return token_balance.balance_float * current_price
