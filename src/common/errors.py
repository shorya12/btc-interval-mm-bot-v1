"""Custom exception classes for Polybot."""

from typing import Any


class PolybotError(Exception):
    """Base exception for all Polybot errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | {self.details}"
        return self.message


class ConfigError(PolybotError):
    """Configuration loading or validation error."""

    pass


class NetworkError(PolybotError):
    """Network communication error (API, RPC, WebSocket)."""

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if endpoint:
            details["endpoint"] = endpoint
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details)


class OrderError(PolybotError):
    """Order placement, cancellation, or tracking error."""

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        token_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if order_id:
            details["order_id"] = order_id
        if token_id:
            details["token_id"] = token_id
        super().__init__(message, details)


class RiskError(PolybotError):
    """Risk management violation or stop triggered."""

    def __init__(
        self,
        message: str,
        risk_type: str | None = None,
        threshold: float | None = None,
        current_value: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if risk_type:
            details["risk_type"] = risk_type
        if threshold is not None:
            details["threshold"] = threshold
        if current_value is not None:
            details["current_value"] = current_value
        super().__init__(message, details)


class InsufficientBalanceError(PolybotError):
    """Insufficient balance for operation."""

    def __init__(
        self,
        message: str,
        required: float,
        available: float,
        asset: str = "USDC",
    ) -> None:
        super().__init__(
            message,
            {"required": required, "available": available, "asset": asset},
        )


class ApprovalError(PolybotError):
    """Token approval error."""

    pass


class DatabaseError(PolybotError):
    """Database operation error."""

    pass
