"""Position management utilities for Polymarket."""

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import SELL

from src.common.logging import get_logger

logger = get_logger(__name__)


def get_positions_from_data_api(wallet_address: str) -> list[dict]:
    """
    Fetch positions from Polymarket Data API.
    
    Args:
        wallet_address: Wallet address to fetch positions for
        
    Returns:
        List of position dicts with token_id, balance, neg_risk, etc.
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
                    # A position is "resolved winning" if price is 1.0
                    "is_resolved_winning": cur_price >= 0.99,
                    # A position is "resolved losing" if price is 0.0
                    "is_resolved_losing": cur_price <= 0.01 and not cur_price >= 0.99,
                })
        return positions
    except Exception as e:
        logger.warning("data_api_failed", error=str(e))
        return []


def close_all_positions(client: ClobClient, dry_run: bool = False) -> dict:
    """
    Close all positions using the proven working logic.
    
    This function:
    1. Cancels all open orders
    2. Fetches actual positions from Data API
    3. Sells all active positions at best bid
    
    Args:
        client: Initialized ClobClient with API credentials set
        dry_run: If True, don't actually place orders
        
    Returns:
        Dict with results: orders_placed, errors, total_sold_value
    """
    wallet_address = client.get_address()
    
    # Step 1: Cancel all open orders first
    logger.info("closing_positions_cancel_orders")
    try:
        client.cancel_all()
    except Exception as e:
        logger.warning("cancel_orders_failed", error=str(e))
    
    # Step 2: Get all positions from Data API
    logger.info("closing_positions_fetch_positions")
    positions = get_positions_from_data_api(wallet_address)
    
    if not positions:
        logger.info("no_positions_to_close")
        return {"orders_placed": [], "errors": [], "total_sold_value": 0.0}
    
    # Filter to active positions only (not resolved)
    active_positions = [
        p for p in positions 
        if not p["is_resolved_winning"] and not p["is_resolved_losing"]
    ]
    
    if not active_positions:
        logger.info("no_active_positions_to_sell")
        return {"orders_placed": [], "errors": [], "total_sold_value": 0.0}
    
    logger.info(
        "closing_positions_found",
        total=len(positions),
        active=len(active_positions),
    )
    
    # Step 3: Sell all active positions
    total_sold_value = 0.0
    errors = []
    orders_placed = []
    
    for pos in active_positions:
        token_id = pos["token_id"]
        size = pos["balance"]
        title = pos.get("title", "Unknown")[:40]
        neg_risk = pos.get("neg_risk", False)
        
        try:
            # Get current order book to find best bid
            book = client.get_order_book(token_id)
            
            # Handle both dict and object response types
            if hasattr(book, 'bids'):
                bids = book.bids or []
            else:
                bids = book.get("bids", [])
            
            if not bids:
                logger.warning("no_bids_available", token_id=token_id[:16], title=title)
                errors.append(f"No bids for {title}")
                continue
            
            # Get best bid price
            first_bid = bids[0]
            if hasattr(first_bid, 'price'):
                best_bid = float(first_bid.price)
            else:
                best_bid = float(first_bid["price"])
            
            order_value = best_bid * size
            
            if dry_run:
                logger.info(
                    "dry_run_would_sell",
                    token_id=token_id[:16],
                    title=title,
                    size=size,
                    price=best_bid,
                    value=order_value,
                )
                total_sold_value += order_value
                continue
            
            # Check minimum order value ($1)
            if order_value < 1.0:
                logger.warning(
                    "order_value_too_small",
                    token_id=token_id[:16],
                    title=title,
                    value=order_value,
                )
                errors.append(f"Order value too small for {title}")
                continue
            
            # Get tick size for this token
            try:
                tick_size = client.get_tick_size(token_id)
            except Exception:
                tick_size = "0.01"
            
            # Round size to 2 decimals
            size = round(size, 2)
            
            # Create sell order at best bid
            order_args = OrderArgs(
                token_id=token_id,
                price=best_bid,
                size=size,
                side=SELL,
            )
            
            create_options = PartialCreateOrderOptions(
                tick_size=tick_size,
                neg_risk=neg_risk,
            )
            
            signed_order = client.create_order(order_args, create_options)
            response = client.post_order(signed_order, OrderType.GTC)
            
            order_id = response.get("orderID", response.get("id", "unknown"))
            
            logger.info(
                "sell_order_placed",
                token_id=token_id[:16],
                title=title,
                size=size,
                price=best_bid,
                order_id=order_id[:16] if order_id != "unknown" else "unknown",
            )
            
            total_sold_value += order_value
            orders_placed.append({
                "order_id": order_id,
                "token_id": token_id,
                "title": title,
                "size": size,
                "price": best_bid,
            })
            
        except Exception as e:
            logger.error("sell_order_failed", token_id=token_id[:16], title=title, error=str(e))
            errors.append(f"Error selling {title}: {e}")
    
    logger.info(
        "closing_positions_complete",
        orders_placed=len(orders_placed),
        errors=len(errors),
        total_sold_value=round(total_sold_value, 2),
    )
    
    return {
        "orders_placed": orders_placed,
        "errors": errors,
        "total_sold_value": total_sold_value,
    }
