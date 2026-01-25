#!/usr/bin/env python3
"""
Dry Run Results Analyzer

Analyzes dry-run test results to calculate performance metrics
and compare across different test runs.

Usage:
    python dry_run_tests/analyze_results.py results/test_001.json
    python dry_run_tests/analyze_results.py results/*.json --compare
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def analyze_single(results: dict) -> dict:
    """Analyze a single test result."""
    test_info = results.get("test_info", {})
    metrics = results.get("final_metrics", {})
    snapshots = results.get("snapshots", [])
    
    # Calculate annualized returns
    duration_hours = test_info.get("duration_seconds", 0) / 3600
    if duration_hours > 0:
        pnl_percent = metrics.get("pnl_percent", 0)
        hours_per_year = 24 * 365
        annualized_return = pnl_percent * (hours_per_year / duration_hours)
    else:
        annualized_return = 0
    
    # Calculate Sharpe-like ratio from snapshots
    if len(snapshots) >= 2:
        pnls = []
        for i in range(1, len(snapshots)):
            prev_balance = snapshots[i-1].get("balance", 10000)
            curr_balance = snapshots[i].get("balance", 10000)
            pnls.append(curr_balance - prev_balance)
        
        if pnls:
            import statistics
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 0
            sharpe_proxy = mean_pnl / std_pnl if std_pnl > 0 else 0
        else:
            sharpe_proxy = 0
    else:
        sharpe_proxy = 0
    
    # Fill imbalance
    buy_fills = metrics.get("buy_fills", 0)
    sell_fills = metrics.get("sell_fills", 0)
    total_fills = buy_fills + sell_fills
    fill_imbalance = (buy_fills - sell_fills) / total_fills if total_fills > 0 else 0
    
    return {
        "test_file": "",  # Set by caller
        "start_time": test_info.get("start_time"),
        "duration_hours": round(duration_hours, 2),
        "interval": test_info.get("interval"),
        "net_pnl": metrics.get("net_pnl", 0),
        "pnl_percent": metrics.get("pnl_percent", 0),
        "annualized_return": round(annualized_return, 2),
        "sharpe_proxy": round(sharpe_proxy, 3),
        "total_fills": total_fills,
        "fill_imbalance": round(fill_imbalance, 3),
        "fill_rate": metrics.get("fill_stats", {}).get("actual_fill_rate", 0),
        "total_fees": metrics.get("total_fees", 0),
        "final_position_size": sum(
            pos.get("size", 0) 
            for pos in metrics.get("positions", {}).values()
        ),
    }


def print_analysis(analysis: dict, results: dict) -> None:
    """Print detailed analysis of a single test."""
    print("\n" + "=" * 70)
    print(f"ANALYSIS: {analysis['test_file']}")
    print("=" * 70)
    
    print(f"\n📅 Test Info")
    print(f"   Start:    {analysis['start_time']}")
    print(f"   Duration: {analysis['duration_hours']:.2f} hours")
    print(f"   Interval: {analysis['interval']}")
    
    print(f"\n💰 Performance")
    print(f"   Net PnL:           ${analysis['net_pnl']:.4f} ({analysis['pnl_percent']:.3f}%)")
    print(f"   Annualized Return: {analysis['annualized_return']:.1f}%")
    print(f"   Sharpe Proxy:      {analysis['sharpe_proxy']:.3f}")
    print(f"   Total Fees:        ${analysis['total_fees']:.4f}")
    
    print(f"\n📊 Trading Activity")
    print(f"   Total Fills:    {analysis['total_fills']}")
    print(f"   Fill Rate:      {analysis['fill_rate']:.1%}")
    print(f"   Fill Imbalance: {analysis['fill_imbalance']:+.3f} (buy-sell)")
    print(f"   Final Position: {analysis['final_position_size']:.2f}")
    
    # Analyze PnL over time from snapshots
    snapshots = results.get("snapshots", [])
    if len(snapshots) >= 2:
        print(f"\n📈 PnL Over Time ({len(snapshots)} snapshots)")
        
        # Show first, middle, and last snapshots
        indices = [0, len(snapshots) // 2, -1]
        for idx in indices:
            snap = snapshots[idx]
            print(f"   [{snap.get('tick', 0):4d}] Balance: ${snap.get('balance', 0):,.2f}")
    
    print()


def compare_results(analyses: list[dict]) -> None:
    """Compare multiple test results."""
    if not analyses:
        print("No results to compare")
        return
    
    # Create DataFrame for comparison
    df = pd.DataFrame(analyses)
    
    print("\n" + "=" * 90)
    print("COMPARISON OF DRY RUN TESTS")
    print("=" * 90)
    
    # Summary statistics
    print("\n📊 Summary Statistics")
    print(f"   Tests analyzed: {len(analyses)}")
    print(f"   Total duration: {df['duration_hours'].sum():.2f} hours")
    print(f"   Mean PnL:       ${df['net_pnl'].mean():.4f} ({df['pnl_percent'].mean():.3f}%)")
    print(f"   Std PnL:        ${df['net_pnl'].std():.4f}")
    print(f"   Mean Fill Rate: {df['fill_rate'].mean():.1%}")
    
    # Best and worst performers
    best_idx = df['net_pnl'].idxmax()
    worst_idx = df['net_pnl'].idxmin()
    
    print(f"\n🏆 Best Performer")
    print(f"   {analyses[best_idx]['test_file']}")
    print(f"   PnL: ${analyses[best_idx]['net_pnl']:.4f}")
    
    print(f"\n⚠️  Worst Performer")
    print(f"   {analyses[worst_idx]['test_file']}")
    print(f"   PnL: ${analyses[worst_idx]['net_pnl']:.4f}")
    
    # Group by interval if multiple intervals tested
    intervals = df['interval'].unique()
    if len(intervals) > 1:
        print(f"\n📊 Performance by Interval")
        for interval in intervals:
            interval_df = df[df['interval'] == interval]
            print(f"   {interval}: Mean PnL ${interval_df['net_pnl'].mean():.4f}, "
                  f"Fill Rate {interval_df['fill_rate'].mean():.1%}")
    
    # Print comparison table
    print("\n📋 Comparison Table")
    print("-" * 90)
    cols = ['test_file', 'duration_hours', 'interval', 'net_pnl', 'pnl_percent', 'fill_rate', 'total_fills']
    display_df = df[cols].copy()
    display_df['test_file'] = display_df['test_file'].apply(lambda x: Path(x).name)
    display_df['net_pnl'] = display_df['net_pnl'].apply(lambda x: f"${x:.4f}")
    display_df['pnl_percent'] = display_df['pnl_percent'].apply(lambda x: f"{x:.3f}%")
    display_df['fill_rate'] = display_df['fill_rate'].apply(lambda x: f"{x:.1%}")
    
    print(display_df.to_string(index=False))
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dry-run test results"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Result JSON files to analyze",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple test results",
    )
    parser.add_argument(
        "--csv",
        help="Export comparison to CSV file",
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    import glob
    all_files = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        all_files.extend(matches if matches else [pattern])
    
    if not all_files:
        print("No result files found")
        return
    
    # Load and analyze all results
    analyses = []
    all_results = []
    
    for file_path in all_files:
        try:
            results = load_results(file_path)
            analysis = analyze_single(results)
            analysis['test_file'] = file_path
            analyses.append(analysis)
            all_results.append(results)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not analyses:
        print("No valid results to analyze")
        return
    
    # Output based on mode
    if args.compare or len(analyses) > 1:
        compare_results(analyses)
    else:
        for analysis, results in zip(analyses, all_results):
            print_analysis(analysis, results)
    
    # Export to CSV if requested
    if args.csv:
        df = pd.DataFrame(analyses)
        df.to_csv(args.csv, index=False)
        print(f"\nExported to: {args.csv}")


if __name__ == "__main__":
    main()
