#!/usr/bin/env python3
"""
Close all Polymarket positions.

This script:
1. Fetches all token balances from the wallet
2. Cancels all open orders
3. Sells all tokens at market price to close positions

Usage:
    python scripts/close_all_positions.py [--dry-run]

Arguments:
    --dry-run    Show what would be sold without executing
"""

import sys
import os
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import SELL


def get_positions_from_data_api(wallet_address: str) -> list[dict]:
    """
    Fetch positions from Polymarket Data API.
    
    Returns list of positions with token_id and size.
    """
    url = "https://data-api.polymarket.com/positions"
    params = {
        "user": wallet_address.lower(),
        "sizeThreshold": 0,  # Get all positions including small ones
        "limit": 500,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        positions = []
        for pos in data:
            size = float(pos.get("size", 0))
            if size > 0.01:  # Ignore dust
                cur_price = float(pos.get("curPrice", 0.5))
                positions.append({
                    "token_id": pos.get("asset", ""),
                    "condition_id": pos.get("conditionId", ""),
                    "balance": size,
                    "title": pos.get("title", "unknown"),
                    "outcome": pos.get("outcome", "unknown"),
                    "slug": pos.get("slug", "unknown"),
                    "current_value": float(pos.get("currentValue", 0)),
                    "cur_price": cur_price,
                    "neg_risk": pos.get("negativeRisk", False),
                    "redeemable": pos.get("redeemable", False),
                    "mergeable": pos.get("mergeable", False),
                    # A position is "resolved winning" if price is 1.0 (price takes priority)
                    "is_resolved_winning": cur_price >= 0.99,
                    # A position is "resolved losing" if price is 0.0 (even if API says redeemable)
                    "is_resolved_losing": cur_price <= 0.01 and not cur_price >= 0.99,
                })
        return positions
    except Exception as e:
        print(f"  Warning: Data API failed: {e}")
        return []


def main():
    """Main entry point."""
    load_dotenv()
    
    # Parse arguments
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    
    # Load credentials
    private_key = os.getenv("POLYBOT_PRIVATE_KEY")
    funder = os.getenv("POLYBOT_FUNDER_ADDRESS")
    signature_type = int(os.getenv("POLYBOT_SIGNATURE_TYPE", "0"))
    
    if not private_key:
        print("Error: POLYBOT_PRIVATE_KEY not set in environment")
        sys.exit(1)
    
    # Initialize client
    host = "https://clob.polymarket.com"
    chain_id = 137  # Polygon mainnet
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Connecting to Polymarket...")
    print(f"  Host: {host}")
    print(f"  Chain ID: {chain_id}")
    print(f"  Signature Type: {signature_type}")
    if funder:
        print(f"  Funder: {funder[:10]}...{funder[-6:]}")
    
    client = ClobClient(
        host,
        key=private_key,
        chain_id=chain_id,
        signature_type=signature_type,
        funder=funder if funder else None,
    )
    
    # Set API credentials
    try:
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        print(f"  API Key: {creds.api_key[:12]}...")
    except Exception as e:
        print(f"Error setting API credentials: {e}")
        sys.exit(1)
    
    wallet_address = client.get_address()
    print(f"\nWallet Address: {wallet_address}")
    
    # Step 1: Cancel all open orders first
    print("\n--- Step 1: Cancelling all open orders ---")
    try:
        result = client.cancel_all()
        print(f"  Cancelled orders: {result}")
    except Exception as e:
        print(f"  Warning: Failed to cancel orders: {e}")
    
    # Step 2: Get all token balances
    print("\n--- Step 2: Fetching token balances ---")
    
    # Use Data API to get positions
    print("  Fetching positions from Data API...")
    positions = get_positions_from_data_api(wallet_address)
    
    if not positions:
        print("\n  No positions to close!")
        return
    
    # Categorize positions
    active_positions = [p for p in positions if not p["is_resolved_winning"] and not p["is_resolved_losing"]]
    winning_positions = [p for p in positions if p["is_resolved_winning"]]
    losing_positions = [p for p in positions if p["is_resolved_losing"]]
    
    print(f"\n  Found {len(positions)} total positions:")
    print(f"    - Active (can sell): {len(active_positions)}")
    print(f"    - Resolved WINNING (need to redeem): {len(winning_positions)}")
    print(f"    - Resolved LOSING (worthless): {len(losing_positions)}")
    
    total_current_value = 0.0
    total_redeemable = 0.0
    
    for i, pos in enumerate(positions, 1):
        status = ""
        if pos["is_resolved_winning"]:
            status = " [REDEEMABLE - WON]"
            total_redeemable += pos["balance"]  # Each share worth $1
        elif pos["is_resolved_losing"]:
            status = " [RESOLVED - LOST]"
        
        print(f"\n    {i}. {pos.get('title', 'Unknown')[:50]}{status}")
        print(f"       Token: {pos['token_id'][:20]}...")
        print(f"       Outcome: {pos.get('outcome', 'unknown')}")
        print(f"       Balance: {pos['balance']:.4f} shares")
        print(f"       Current Price: ${pos.get('cur_price', 0.5):.4f}")
        print(f"       Current Value: ${pos.get('current_value', 0):.2f}")
        total_current_value += pos.get('current_value', 0)
    
    print(f"\n  Total Portfolio Value: ${total_current_value:.2f}")
    if winning_positions:
        print(f"  Total Redeemable (winning positions): ${total_redeemable:.2f}")
    
    # Step 3: Process positions
    print(f"\n--- Step 3: Processing positions ---")
    
    total_sold_value = 0.0
    errors = []
    orders_placed = []
    skipped_resolved = []
    
    # Handle winning positions first (need redemption, not selling)
    if winning_positions:
        print("\n  WINNING POSITIONS (need redemption on Polymarket website):")
        for pos in winning_positions:
            title = pos.get("title", "Unknown")[:40]
            print(f"    - {title}: {pos['balance']:.4f} shares = ${pos['balance']:.2f}")
        print("\n  To redeem winning positions:")
        print("    1. Go to https://polymarket.com/portfolio")
        print("    2. Click on each resolved winning position")
        print("    3. Click 'Redeem' to collect your winnings")
    
    # Handle losing positions
    if losing_positions:
        print("\n  LOSING POSITIONS (worthless, no action needed):")
        for pos in losing_positions:
            title = pos.get("title", "Unknown")[:40]
            print(f"    - {title}: {pos['balance']:.4f} shares (worth $0)")
    
    # Process active positions (can sell)
    if not active_positions:
        print("\n  No active positions to sell.")
    else:
        print(f"\n  {'[DRY RUN] Would sell' if dry_run else 'Selling'} {len(active_positions)} active positions:")
        
        for pos in active_positions:
            token_id = pos["token_id"]
            size = pos["balance"]
            title = pos.get("title", "Unknown")[:40]
            neg_risk = pos.get("neg_risk", False)
            
            # Get current order book to find best bid
            try:
                book = client.get_order_book(token_id)
                # Handle both dict and object response types
                if hasattr(book, 'bids'):
                    bids = book.bids or []
                else:
                    bids = book.get("bids", [])
                
                if not bids:
                    print(f"\n  [{title}]: No bids available, skipping")
                    errors.append(f"No bids for {title}")
                    continue

                # Get best bid price (handle both dict and object)
                first_bid = bids[0]
                if hasattr(first_bid, 'price'):
                    best_bid = float(first_bid.price)
                else:
                    best_bid = float(first_bid["price"])
                
                # Calculate order value
                order_value = best_bid * size
                
                print(f"\n  [{title}]")
                print(f"    Token: {token_id[:16]}...")
                print(f"    Size: {size:.4f} shares")
                print(f"    Best Bid: ${best_bid:.4f}")
                print(f"    Order Value: ${order_value:.2f}")
                
                if dry_run:
                    print(f"    [DRY RUN] Would place SELL order")
                    total_sold_value += order_value
                    continue
                
                # Check minimum order value ($1)
                if order_value < 1.0:
                    print(f"    Skipping: Order value ${order_value:.2f} below minimum ($1)")
                    errors.append(f"Order value too small for {title}")
                    continue
                
                # Get tick size for this token
                try:
                    tick_size = client.get_tick_size(token_id)
                except Exception:
                    tick_size = "0.01"  # Default
                
                # Round size to 2 decimals
                size = round(size, 2)
                
                # Create sell order at best bid (limit order)
                order_args = OrderArgs(
                    token_id=token_id,
                    price=best_bid,
                    size=size,
                    side=SELL,
                )
                
                # Create and post order
                create_options = PartialCreateOrderOptions(
                    tick_size=tick_size,
                    neg_risk=neg_risk,
                )
                
                signed_order = client.create_order(order_args, create_options)
                response = client.post_order(signed_order, OrderType.GTC)
                
                order_id = response.get("orderID", response.get("id", "unknown"))
                print(f"    Order placed: {order_id[:16]}...")
                total_sold_value += order_value
                orders_placed.append({
                    "order_id": order_id,
                    "title": title,
                    "size": size,
                    "price": best_bid,
                })
                
            except Exception as e:
                print(f"\n  Error selling [{title}]: {e}")
                errors.append(f"Error selling {title}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Total positions: {len(positions)}")
    print(f"    - Active positions: {len(active_positions)}")
    print(f"    - Winning (redeemable): {len(winning_positions)}")
    print(f"    - Losing (worthless): {len(losing_positions)}")
    
    if winning_positions:
        print(f"\n  Redeemable value: ${total_redeemable:.2f}")
        print("  --> Redeem these on Polymarket website!")
    
    if active_positions:
        print(f"\n  Active positions {'estimated' if dry_run else ''} sold: ${total_sold_value:.2f}")
    
    if orders_placed:
        print(f"\n  Orders placed: {len(orders_placed)}")
        for order in orders_placed:
            print(f"    - {order['title']}: {order['size']:.2f} @ ${order['price']:.4f}")
    
    if errors:
        print(f"\n  Errors/Warnings: {len(errors)}")
        for err in errors:
            print(f"    - {err}")
    
    if dry_run:
        print("\n[DRY RUN] No orders were placed. Run without --dry-run to execute.")
    else:
        if orders_placed:
            print("\nSell orders placed. They may take time to fill.")
            print("Check your open orders on Polymarket to confirm fills.")
        if winning_positions:
            print("\nDon't forget to redeem your winning positions on Polymarket!")


if __name__ == "__main__":
    main()
