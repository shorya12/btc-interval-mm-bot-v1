#!/usr/bin/env python3
"""Standalone script to set token approvals."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web3 import Web3
from dotenv import load_dotenv

from src.wallet_approval import ApprovalManager
from src.common.config import load_config


async def main() -> None:
    """Set token approvals for Polymarket trading."""
    load_dotenv()

    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("Error: config.yaml not found")
        sys.exit(1)

    config = load_config(config_path)

    # Get private key from environment
    private_key = os.getenv("POLYBOT_PRIVATE_KEY")
    if not private_key:
        print("Error: POLYBOT_PRIVATE_KEY not set")
        sys.exit(1)

    # Connect to Polygon
    web3 = Web3(Web3.HTTPProvider(config.network.rpc_url))

    if not web3.is_connected():
        print("Error: Failed to connect to RPC")
        sys.exit(1)

    print(f"Connected to chain {web3.eth.chain_id}")

    # Initialize approval manager
    manager = ApprovalManager(web3=web3, private_key=private_key)
    print(f"Wallet address: {manager.address}")

    # Check current status
    print("\nChecking current approvals...")
    status = await manager.check_approvals()

    print("  USDC approvals:")
    print(f"    Exchange: {status.usdc_exchange_approved}")
    print(f"    NegRisk Exchange: {status.usdc_neg_risk_exchange_approved}")
    print(f"    NegRisk Adapter: {status.usdc_neg_risk_adapter_approved}")
    print("  CTF approvals:")
    print(f"    Exchange: {status.ctf_exchange_approved}")
    print(f"    NegRisk Exchange: {status.ctf_neg_risk_exchange_approved}")
    print(f"    NegRisk Adapter: {status.ctf_neg_risk_adapter_approved}")

    if status.all_approved:
        print("\nAll required approvals are already set!")
        return

    # Set approvals
    print("\nSetting approvals...")
    tx_hashes = await manager.ensure_approvals(include_neg_risk=True)

    if tx_hashes:
        print("\nApprovals set successfully!")
        for name, tx_hash in tx_hashes.items():
            print(f"  {name}: {tx_hash}")
    else:
        print("\nNo new approvals needed.")


if __name__ == "__main__":
    asyncio.run(main())
