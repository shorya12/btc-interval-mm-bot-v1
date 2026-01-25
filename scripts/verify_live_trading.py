#!/usr/bin/env python3
"""Verify all prerequisites for live trading."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web3 import Web3
from dotenv import load_dotenv

from src.wallet_approval import ApprovalManager
from src.wallet_approval.constants import USDC_ADDRESS, USDC_DECIMALS, ERC20_ABI
from src.common.config import load_config


def check_mark(ok: bool) -> str:
    return "✅" if ok else "❌"


async def main() -> int:
    """Verify live trading prerequisites."""
    load_dotenv()
    
    print("=" * 60)
    print("POLYBOT LIVE TRADING VERIFICATION")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Check config file
    print("\n📋 CONFIGURATION")
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"  {check_mark(False)} config.yaml not found")
        return 1
    
    config = load_config(config_path)
    print(f"  {check_mark(True)} config.yaml loaded")
    print(f"      Chain ID: {config.network.chain_id}")
    print(f"      RPC URL: {config.network.rpc_url[:50]}...")
    print(f"      CLOB Host: {config.network.clob_host}")
    print(f"      Dry run: {config.dry_run.enabled}")
    
    if config.dry_run.enabled:
        print(f"  ⚠️  WARNING: dry_run.enabled is True in config - set to false for live trading")
    
    # 2. Check environment variables
    print("\n🔑 CREDENTIALS")
    private_key = os.getenv("POLYBOT_PRIVATE_KEY")
    api_key = os.getenv("POLYBOT_API_KEY")
    api_secret = os.getenv("POLYBOT_API_SECRET")
    api_passphrase = os.getenv("POLYBOT_API_PASSPHRASE")
    
    if not private_key:
        print(f"  {check_mark(False)} POLYBOT_PRIVATE_KEY not set")
        all_ok = False
        return 1
    else:
        # Don't print the key, just confirm it exists and check format
        has_0x = private_key.startswith("0x")
        key_len = len(private_key.replace("0x", ""))
        print(f"  {check_mark(key_len == 64)} POLYBOT_PRIVATE_KEY set (length: {key_len} hex chars, {'with' if has_0x else 'without'} 0x prefix)")
        if key_len != 64:
            print(f"      ⚠️  Expected 64 hex characters, got {key_len}")
            all_ok = False
    
    print(f"  {check_mark(bool(api_key))} POLYBOT_API_KEY {'set' if api_key else 'not set (optional)'}")
    print(f"  {check_mark(bool(api_secret))} POLYBOT_API_SECRET {'set' if api_secret else 'not set (optional)'}")
    print(f"  {check_mark(bool(api_passphrase))} POLYBOT_API_PASSPHRASE {'set' if api_passphrase else 'not set (optional)'}")
    
    # 3. Check RPC connection
    print("\n🌐 NETWORK CONNECTION")
    try:
        web3 = Web3(Web3.HTTPProvider(config.network.rpc_url, request_kwargs={'timeout': 10}))
        connected = web3.is_connected()
        print(f"  {check_mark(connected)} RPC connection")
        
        if connected:
            chain_id = web3.eth.chain_id
            block = web3.eth.block_number
            print(f"      Chain ID: {chain_id}")
            print(f"      Latest block: {block}")
            
            if chain_id != config.network.chain_id:
                print(f"  ⚠️  WARNING: RPC chain ID ({chain_id}) != config chain ID ({config.network.chain_id})")
                all_ok = False
        else:
            all_ok = False
    except Exception as e:
        print(f"  {check_mark(False)} RPC connection failed: {e}")
        all_ok = False
        return 1
    
    # 4. Initialize approval manager and get wallet address
    print("\n👛 WALLET")
    try:
        manager = ApprovalManager(web3=web3, private_key=private_key)
        address = manager.address
        print(f"  {check_mark(True)} Wallet address: {address}")
        
        # Check MATIC balance for gas
        matic_balance = web3.eth.get_balance(address)
        matic_balance_eth = matic_balance / 1e18
        has_gas = matic_balance_eth > 0.1  # Need at least 0.1 MATIC for gas
        print(f"  {check_mark(has_gas)} MATIC balance: {matic_balance_eth:.4f} MATIC")
        if not has_gas:
            print(f"      ⚠️  Need at least 0.1 MATIC for gas fees")
            all_ok = False
        
        # Check USDC balance
        usdc_contract = web3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=ERC20_ABI,
        )
        usdc_balance = usdc_contract.functions.balanceOf(address).call()
        usdc_balance_human = usdc_balance / (10 ** USDC_DECIMALS)
        has_usdc = usdc_balance_human >= 10  # Need at least $10 USDC
        print(f"  {check_mark(has_usdc)} USDC.e balance: {usdc_balance_human:.2f} USDC")
        if not has_usdc:
            print(f"      ⚠️  Need USDC to trade")
            all_ok = False
            
    except Exception as e:
        print(f"  {check_mark(False)} Wallet initialization failed: {e}")
        all_ok = False
        return 1
    
    # 5. Check token approvals
    print("\n🔐 TOKEN APPROVALS")
    try:
        status = await manager.check_approvals()
        
        print(f"  {check_mark(status.usdc_approved)} USDC.e → Exchange approved")
        print(f"      Allowance: {status.usdc_allowance / 1e6:,.2f} USDC")
        if not status.usdc_approved:
            print(f"      ⚠️  Run: python scripts/approve_allowances.py")
            all_ok = False
        
        print(f"  {check_mark(status.ctf_approved)} CTF → Exchange approved")
        if not status.ctf_approved:
            print(f"      ⚠️  Run: python scripts/approve_allowances.py")
            all_ok = False
        
        print(f"  {check_mark(status.neg_risk_ctf_approved)} Neg Risk CTF → Exchange approved")
        if not status.neg_risk_ctf_approved:
            print(f"      ℹ️  Optional: Run python scripts/approve_allowances.py")
            
    except Exception as e:
        print(f"  {check_mark(False)} Approval check failed: {e}")
        all_ok = False
    
    # 6. Check market discovery (optional test)
    print("\n🎯 MARKET DISCOVERY")
    try:
        from src.polymarket_client import MarketDiscovery
        discovery = MarketDiscovery()
        market = await discovery.find_btc_market(interval="1h")
        await discovery.close()
        
        if market:
            print(f"  {check_mark(True)} Found BTC 1h market")
            print(f"      Question: {market.question[:60]}...")
            print(f"      End date: {market.end_date}")
            print(f"      YES token: {market.token_id_yes[:30]}...")
            print(f"      NO token: {market.token_id_no[:30]}...")
        else:
            print(f"  {check_mark(False)} No active BTC 1h market found")
            all_ok = False
    except Exception as e:
        print(f"  {check_mark(False)} Market discovery failed: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Ready for live trading!")
        print("\nTo start live trading:")
        print("  polybot run")
        print("\nOr for dry run first:")
        print("  polybot run --dry-run")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        print("\nCommon fixes:")
        print("  1. Set approvals: python scripts/approve_allowances.py")
        print("  2. Add MATIC for gas: Send MATIC to your wallet")
        print("  3. Add USDC: Bridge or send USDC.e to your wallet")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
