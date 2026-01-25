"""Configuration loading and validation using Pydantic."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find project root (where .env should be located)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class NetworkConfig(BaseModel):
    """Network configuration."""

    chain_id: int = 137  # Polygon mainnet
    rpc_url: str = "https://polygon-mainnet.g.alchemy.com/v2/sPXfpsHtrLgfEO3eZL5KL"
    clob_host: str = "https://clob.polymarket.com"


class MarketConfig(BaseModel):
    """Single market configuration."""

    token_id: str = ""  # Can be empty if auto_discover is True
    condition_id: str = ""  # Can be empty if auto_discover is True
    description: str = ""
    auto_discover: bool = Field(default=False, description="Auto-discover BTC hourly market")
    outcome: str = Field(default="YES", description="Which outcome to trade (YES or NO)")
    interval: str = Field(default="1h", description="Market interval for auto-discover (15m, 1h, 4h, 1d, 1w, 1M)")
    
    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        allowed = {"15m", "1h", "4h", "1d", "1w", "1M"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {allowed}")
        return v


class AvellanedaStoikovConfig(BaseModel):
    """Avellaneda-Stoikov market making parameters."""

    gamma: float = Field(default=0.1, ge=0.0, le=1.0, description="Risk aversion parameter")
    base_spread_x: float = Field(default=0.01, ge=0.0, description="Base spread multiplier")
    kappa: float | None = Field(default=None, ge=0.0, description="Order arrival rate (optional)")


class BeliefConfig(BaseModel):
    """Belief state estimation parameters."""

    window_seconds: int = Field(default=300, ge=10, description="Rolling window for estimation")
    sigma_b_floor: float = Field(default=0.01, ge=0.001, description="Minimum belief volatility")
    robust_method: str = Field(default="median", description="Robust estimation method")
    jump_z: float = Field(default=3.0, ge=1.0, description="Z-score threshold for jump detection")
    momentum_z: float = Field(default=2.0, ge=1.0, description="Z-score threshold for momentum")

    @field_validator("robust_method")
    @classmethod
    def validate_robust_method(cls, v: str) -> str:
        allowed = {"median", "mean", "ewma", "huber"}
        if v not in allowed:
            raise ValueError(f"robust_method must be one of {allowed}")
        return v


class ExecutionConfig(BaseModel):
    """Order execution parameters."""

    cancel_cooldown_seconds: float = Field(default=2.0, ge=0.0, description="Min time between cancels")
    reprice_threshold_ticks: int = Field(default=2, ge=1, description="Ticks before repricing")
    order_lifetime_seconds: float = Field(default=30.0, ge=1.0, description="Max order age before cancel")


class GammaDangerConfig(BaseModel):
    """Gamma danger zone parameters for risk adjustment."""

    threshold: float = Field(default=0.1, ge=0.0, le=0.5, description="Distance from 0/1 to trigger")
    gamma_multiplier: float = Field(default=2.0, ge=1.0, description="Risk aversion multiplier in danger zone")


class RiskConfig(BaseModel):
    """Risk management parameters."""

    stop_prob_low: float = Field(default=0.02, ge=0.0, le=0.5, description="Stop trading if prob below this")
    stop_prob_high: float = Field(default=0.98, ge=0.5, le=1.0, description="Stop trading if prob above this")
    max_net_frac: float = Field(default=0.15, ge=0.0, le=1.0, description="Max TOTAL exposure as fraction of USDC balance")
    min_time_to_expiry_seconds: float = Field(default=300.0, ge=0.0, description="Stop trading before expiry")
    min_balance_to_trade: float = Field(default=5.0, ge=0.0, description="Stop trading if USDC balance below this")
    max_open_order_pct: float = Field(default=0.10, ge=0.0, le=1.0, description="Max open order value per market as % of balance")
    max_position_pct: float = Field(default=0.20, ge=0.0, le=1.0, description="Max position value per market as % of balance")
    gamma_danger: GammaDangerConfig = Field(default_factory=GammaDangerConfig)


class LagSignalAssetConfig(BaseModel):
    """Single asset configuration for lag signal."""

    symbol: str  # e.g., "BTC/USDT"
    weight: float = Field(default=1.0, ge=0.0, description="Weight in combined signal")
    vol_window: int = Field(default=60, ge=10, description="Realized vol window in seconds")


class LagSignalConfig(BaseModel):
    """Lag signal configuration."""

    assets: list[LagSignalAssetConfig] = Field(default_factory=list)
    skew_multiplier: float = Field(default=1.0, ge=0.0, description="Multiplier for weighted skew")
    exchange: str = Field(default="binance", description="Exchange for price feeds")


class DryRunConfig(BaseModel):
    """Dry run / paper trading configuration."""

    enabled: bool = False
    simulated_fill_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of simulated fill")
    initial_balance: float = Field(default=10000.0, ge=0.0, description="Starting paper balance")


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: str = "polybot.db"
    snapshot_interval_seconds: int = Field(default=60, ge=10, description="PNL snapshot interval")


class Config(BaseModel):
    """Root configuration model."""

    network: NetworkConfig = Field(default_factory=NetworkConfig)
    markets: list[MarketConfig] = Field(default_factory=list)
    avellaneda_stoikov: AvellanedaStoikovConfig = Field(default_factory=AvellanedaStoikovConfig)
    belief: BeliefConfig = Field(default_factory=BeliefConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    lag_signal: LagSignalConfig = Field(default_factory=LagSignalConfig)
    dry_run: DryRunConfig = Field(default_factory=DryRunConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


class EnvSettings(BaseSettings):
    """Environment variable settings."""

    model_config = SettingsConfigDict(
        env_prefix="POLYBOT_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars like API_KEY, API_SECRET, etc.
    )

    private_key: str = ""
    funder_address: str = ""  # Address that holds funds (for proxy wallets)
    signature_type: int = 0  # 0=EOA, 1=Poly Proxy, 2=Gnosis Safe


def load_config(config_path: str | Path) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Validated Config object

    Raises:
        ConfigError: If file not found or validation fails
    """
    from .errors import ConfigError

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return Config.model_validate(data)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ConfigError(f"Configuration validation error: {e}")


def load_env_settings() -> EnvSettings:
    """Load settings from environment variables."""
    return EnvSettings()
