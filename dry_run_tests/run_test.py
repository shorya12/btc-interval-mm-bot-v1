#!/usr/bin/env python3
"""
Dry Run Test Runner

Runs the polybot in dry-run mode for a specified duration and captures
performance metrics to measure expected PnL had the bot been running live.

Usage:
    python dry_run_tests/run_test.py --duration 300 --output results/test_001.json
    python dry_run_tests/run_test.py --duration 3600 --interval 1h --output results/hourly_test.json
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.main_loop.runner import TradingLoop
from src.main_loop.dry_run import DryRunAdapter

logger = get_logger(__name__)


class TestResults:
    """Collects and stores test results."""
    
    def __init__(self, config: dict, duration: int, interval: str):
        self.config = config
        self.duration_seconds = duration
        self.interval = interval
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.metrics: dict = {}
        self.snapshots: list[dict] = []
    
    def to_dict(self) -> dict:
        return {
            "test_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": self.duration_seconds,
                "interval": self.interval,
            },
            "config": self.config,
            "final_metrics": self.metrics,
            "snapshots": self.snapshots,
        }
    
    def save(self, output_path: str) -> None:
        """Save results to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


async def run_dry_run_test(
    config_path: str,
    duration: int,
    interval: str | None,
    snapshot_interval: int = 30,
) -> TestResults:
    """
    Run a dry-run test for the specified duration.
    
    Args:
        config_path: Path to config.yaml
        duration: Test duration in seconds
        interval: Market interval override (5m, 15m, 1h, 4h)
        snapshot_interval: How often to capture snapshots (seconds)
    
    Returns:
        TestResults with performance metrics
    """
    # Load config
    config = load_config(config_path)
    
    # Override interval if specified
    if interval:
        for market in config.markets:
            if market.auto_discover:
                market.interval = interval
    
    results = TestResults(
        config={
            "avellaneda_stoikov": config.avellaneda_stoikov.model_dump(),
            "risk": config.risk.model_dump(),
            "dry_run": config.dry_run.model_dump(),
        },
        duration=duration,
        interval=interval or config.markets[0].interval if config.markets else "15m",
    )
    
    results.start_time = datetime.utcnow()
    
    # Create trading loop in dry run mode
    loop = TradingLoop(config=config, dry_run=True)
    
    # Track snapshots
    snapshot_task: asyncio.Task | None = None
    
    async def capture_snapshots():
        """Capture periodic snapshots during test."""
        while True:
            await asyncio.sleep(snapshot_interval)
            if loop.client and isinstance(loop.client, DryRunAdapter):
                stats = loop.client.get_stats()
                snapshot = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "tick": loop._tick_count,
                    **stats,
                }
                results.snapshots.append(snapshot)
    
    # Handle graceful shutdown
    stop_event = asyncio.Event()
    
    async def shutdown():
        await asyncio.sleep(duration)
        stop_event.set()
        await loop.stop()
    
    try:
        # Start snapshot capture
        snapshot_task = asyncio.create_task(capture_snapshots())
        
        # Start shutdown timer
        shutdown_task = asyncio.create_task(shutdown())
        
        # Run trading loop
        await loop.start()
        
    except asyncio.CancelledError:
        pass
    finally:
        if snapshot_task:
            snapshot_task.cancel()
            try:
                await snapshot_task
            except asyncio.CancelledError:
                pass
    
    results.end_time = datetime.utcnow()
    
    # Capture final metrics
    if loop.client and isinstance(loop.client, DryRunAdapter):
        stats = loop.client.get_stats()
        
        # Calculate additional metrics
        initial_balance = config.dry_run.initial_balance
        final_balance = stats["balance"]
        
        total_realized_pnl = sum(
            pos.realized_pnl 
            for pos in loop.client.state.positions.values()
        )
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for token_id, pos in loop.client.state.positions.items():
            if pos.size != 0 and pos.avg_entry_price is not None:
                book = loop.client._last_books.get(token_id)
                if book:
                    current_price = book.mid_price or 0.5
                    if pos.size > 0:
                        unrealized_pnl += pos.size * (current_price - pos.avg_entry_price)
                    else:
                        unrealized_pnl += abs(pos.size) * (pos.avg_entry_price - current_price)
        
        total_pnl = total_realized_pnl + unrealized_pnl - stats["total_fees"]
        
        # Count fills by side
        buy_fills = sum(1 for f in loop.client.state.fills if f.side.value == "BUY")
        sell_fills = sum(1 for f in loop.client.state.fills if f.side.value == "SELL")
        
        results.metrics = {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_fees": stats["total_fees"],
            "net_pnl": total_pnl,
            "pnl_percent": (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0,
            "total_ticks": loop._tick_count,
            "total_fills": stats["total_fills"],
            "buy_fills": buy_fills,
            "sell_fills": sell_fills,
            "fill_stats": stats.get("fill_stats", {}),
            "positions": {
                tid: {
                    "size": pos.size,
                    "avg_entry": pos.avg_entry_price,
                    "realized_pnl": pos.realized_pnl,
                }
                for tid, pos in loop.client.state.positions.items()
            },
        }
    
    return results


def print_summary(results: TestResults) -> None:
    """Print a summary of the test results."""
    m = results.metrics
    
    print("\n" + "=" * 60)
    print("DRY RUN TEST SUMMARY")
    print("=" * 60)
    
    print(f"\nTest Duration: {results.duration_seconds}s")
    print(f"Market Interval: {results.interval}")
    print(f"Total Ticks: {m.get('total_ticks', 0)}")
    
    print("\n--- Performance ---")
    print(f"Initial Balance: ${m.get('initial_balance', 0):,.2f}")
    print(f"Final Balance:   ${m.get('final_balance', 0):,.2f}")
    print(f"Net PnL:         ${m.get('net_pnl', 0):,.4f} ({m.get('pnl_percent', 0):.3f}%)")
    print(f"  Realized:      ${m.get('realized_pnl', 0):,.4f}")
    print(f"  Unrealized:    ${m.get('unrealized_pnl', 0):,.4f}")
    print(f"  Fees:          ${m.get('total_fees', 0):,.4f}")
    
    print("\n--- Trading Activity ---")
    print(f"Total Fills: {m.get('total_fills', 0)} (Buy: {m.get('buy_fills', 0)}, Sell: {m.get('sell_fills', 0)})")
    
    fill_stats = m.get("fill_stats", {})
    if fill_stats:
        print(f"Fill Rate:   {fill_stats.get('actual_fill_rate', 0):.1%} actual, {fill_stats.get('avg_fill_probability', 0):.1%} avg prob")
        print(f"Rejected:    {fill_stats.get('rejected_probability', 0)} (prob), {fill_stats.get('rejected_price', 0)} (price)")
    
    print("\n--- Positions ---")
    positions = m.get("positions", {})
    if positions:
        for tid, pos in positions.items():
            if pos.get("size", 0) != 0:
                print(f"  {tid[:20]}...: {pos['size']:.2f} @ {pos.get('avg_entry', 0):.4f}")
    else:
        print("  No open positions")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run dry-run test to measure expected PnL"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--interval",
        choices=["15m", "1h", "4h", "1d", "1w", "1M"],
        help="Market interval to test (overrides config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=30,
        help="Seconds between snapshots (default: 30)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Generate default output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"dry_run_tests/results/test_{timestamp}.json"
    
    print(f"\n🚀 Starting dry-run test")
    print(f"   Duration: {args.duration}s")
    print(f"   Interval: {args.interval or 'from config'}")
    print(f"   Output: {args.output}\n")
    
    # Run test
    results = asyncio.run(
        run_dry_run_test(
            config_path=args.config,
            duration=args.duration,
            interval=args.interval,
            snapshot_interval=args.snapshot_interval,
        )
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    results.save(args.output)


if __name__ == "__main__":
    main()
