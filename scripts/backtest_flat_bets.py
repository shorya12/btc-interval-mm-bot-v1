"""
Flat-bet backtest on walk-forward validation predictions.

Simulates $100 bets on each predicted direction using predictions from all
walk-forward folds, deduplicating overlapping windows by keeping the latest
fold's prediction per timestamp.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def simulate_flat_bets(
    predictions: pd.DataFrame,
    bet_size: float = 100.0,
    starting_balance: float = 10_000.0,
    thresholds: list[float] | None = None,
) -> dict[float, dict]:
    """
    Run flat-bet simulation for each confidence threshold.

    Args:
        predictions: DataFrame with columns: timestamp, y_true, y_pred_calibrated, fold_index
        bet_size: Dollar amount per bet
        starting_balance: Starting account balance
        thresholds: List of edge thresholds — skip bet if |pred - 0.5| <= threshold

    Returns:
        Dict mapping threshold -> stats dict
    """
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.10, 0.15]

    results = {}
    for threshold in thresholds:
        balance = starting_balance
        balance_series = [balance]
        wins = 0
        losses = 0

        for _, row in predictions.iterrows():
            pred = float(row["y_pred_calibrated"])
            actual = int(row["y_true"])

            # Skip low-conviction bets
            if abs(pred - 0.5) <= threshold:
                continue

            # Determine bet direction
            bet_up = pred > 0.5
            win = (bet_up and actual == 1) or (not bet_up and actual == 0)

            if win:
                balance += bet_size
                wins += 1
            else:
                balance -= bet_size
                losses += 1

            balance_series.append(balance)

        n_bets = wins + losses
        win_rate = wins / n_bets if n_bets > 0 else 0.0
        final_balance = balance
        total_return_pct = (final_balance - starting_balance) / starting_balance * 100

        # Max drawdown
        peak = starting_balance
        max_dd = 0.0
        for b in balance_series:
            if b > peak:
                peak = b
            dd = (peak - b) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Annualized Sharpe (assume each bet is a 1h period)
        if len(balance_series) > 1:
            returns = np.diff(balance_series)
            mean_ret = returns.mean()
            std_ret = returns.std()
            sharpe = (mean_ret / (std_ret + 1e-10)) * math.sqrt(8760) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        results[threshold] = {
            "n_bets": n_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "final_balance": final_balance,
            "total_return_pct": total_return_pct,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
        }

    return results


def print_fold_summary(fold_results: list, predictions_all: pd.DataFrame) -> None:
    """Print per-fold P&L summary table."""
    table = Table(title="Per-Fold Backtest Summary (threshold=0.0)", show_lines=True)
    table.add_column("Fold", style="cyan")
    table.add_column("Val Period")
    table.add_column("N Bets", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("P&L ($)", justify="right")
    table.add_column("BSS Overall", justify="right")

    for fr in fold_results:
        fold_preds = predictions_all[predictions_all["fold_index"] == fr.fold_index]
        if fold_preds.empty:
            continue

        fold_stats = simulate_flat_bets(fold_preds, thresholds=[0.0])[0.0]
        val_period = f"{fr.val_start.strftime('%Y-%m-%d')} to {fr.val_end.strftime('%Y-%m-%d')}"
        pnl = fold_stats["final_balance"] - 10_000
        pnl_str = f"[green]+${pnl:.0f}[/green]" if pnl >= 0 else f"[red]-${abs(pnl):.0f}[/red]"
        bss = fr.eval_result.bss_overall

        table.add_row(
            str(fr.fold_index),
            val_period,
            str(fold_stats["n_bets"]),
            f"{fold_stats['win_rate']:.1%}",
            pnl_str,
            f"{bss:.4f}",
        )

    console.print(table)


def print_threshold_summary(
    threshold_results: dict[float, dict],
    starting_balance: float,
) -> None:
    """Print threshold sensitivity summary table."""
    table = Table(title="Threshold Sensitivity Summary", show_lines=True)
    table.add_column("Threshold", style="cyan")
    table.add_column("Total Bets", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Final Balance ($)", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Max Drawdown", justify="right")
    table.add_column("Sharpe", justify="right")

    for threshold, stats in threshold_results.items():
        final = stats["final_balance"]
        ret = stats["total_return_pct"]
        balance_str = f"[green]{final:.0f}[/green]" if final >= starting_balance else f"[red]{final:.0f}[/red]"
        ret_str = f"[green]+{ret:.1f}%[/green]" if ret >= 0 else f"[red]{ret:.1f}%[/red]"

        table.add_row(
            f"{threshold:.2f}",
            str(stats["n_bets"]),
            f"{stats['win_rate']:.1%}",
            balance_str,
            ret_str,
            f"{stats['max_drawdown']:.1%}",
            f"{stats['sharpe']:.3f}",
        )

    console.print(table)


async def run_backtest(
    config_path: str = "config.yaml",
    symbol: str = "BTC/USDT",
    bet_size: float = 100.0,
    starting_balance: float = 10_000.0,
    thresholds: list[float] | None = None,
) -> None:
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.10, 0.15]

    from src.common.config import load_config
    from src.persistence import Database, Repository
    from src.probability_model.xgboost_model import XGBoostModel
    from src.probability_model.trainer import WalkForwardTrainer

    cfg = load_config(config_path)
    db = Database(cfg.database.path)
    await db.connect()
    repo = Repository(db)

    try:
        prices = await repo.get_recent_crypto_prices(symbol, limit=500_000)
        if len(prices) < 100:
            console.print(f"[red]Not enough data: {len(prices)} candles[/red]")
            return

        prices_sorted = sorted(prices, key=lambda p: p.timestamp)
        rows = []
        for p in prices_sorted:
            meta = p.metadata or {}
            rows.append({
                "timestamp": p.timestamp,
                "open": meta.get("open", p.price),
                "high": meta.get("high", p.price),
                "low": meta.get("low", p.price),
                "close": p.price,
                "volume": p.volume_24h or 0.0,
            })
        df = pd.DataFrame(rows).set_index("timestamp")

        console.print(f"[yellow]Running walk-forward on {len(df)} candles...[/yellow]")
        trainer = WalkForwardTrainer(model_class=XGBoostModel, config=cfg.belief)
        fold_results = trainer.run(df)

        if not fold_results:
            console.print("[red]No folds produced — insufficient data[/red]")
            return

        # Collect and dedup predictions (keep last fold's prediction per timestamp)
        all_preds = pd.concat(
            [fr.val_predictions for fr in fold_results if not fr.val_predictions.empty],
            ignore_index=True,
        )
        all_preds = (
            all_preds
            .sort_values(["timestamp", "fold_index"])
            .groupby("timestamp")
            .last()
            .reset_index()
        )
        all_preds = all_preds.sort_values("timestamp").reset_index(drop=True)

        console.print(f"[green]{len(fold_results)} folds, {len(all_preds)} unique val predictions[/green]\n")

        # Per-fold table
        print_fold_summary(fold_results, all_preds)

        # Threshold sweep on deduplicated predictions
        console.print()
        threshold_results = simulate_flat_bets(
            all_preds, bet_size=bet_size, starting_balance=starting_balance, thresholds=thresholds
        )
        print_threshold_summary(threshold_results, starting_balance)

        # Save summary JSON
        profitable_folds = sum(
            1 for fr in fold_results
            if not fr.val_predictions.empty
            and (fr.val_predictions["y_true"] == (fr.val_predictions["y_pred_calibrated"] > 0.5).astype(int)).mean() > 0.5
        )
        summary = {
            "run_date": datetime.utcnow().isoformat(),
            "n_folds": len(fold_results),
            "val_start": str(all_preds["timestamp"].min()),
            "val_end": str(all_preds["timestamp"].max()),
            "n_predictions": len(all_preds),
            "profitable_folds": profitable_folds,
            "thresholds": {
                str(t): {
                    "n_bets": s["n_bets"],
                    "win_rate": round(s["win_rate"], 4),
                    "profit_factor": round(s["win_rate"] / max(1 - s["win_rate"], 1e-6), 4),
                    "final_balance": s["final_balance"],
                    "total_return_pct": round(s["total_return_pct"], 2),
                    "max_drawdown": round(s["max_drawdown"], 4),
                    "sharpe": round(s["sharpe"], 4),
                }
                for t, s in threshold_results.items()
            },
        }
        os.makedirs("models", exist_ok=True)
        with open("models/backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        console.print("[dim]Saved to models/backtest_summary.json[/dim]")

    finally:
        await db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flat-bet backtest on walk-forward predictions")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbol")
    parser.add_argument("--bet-size", type=float, default=100.0, help="Bet size in $")
    parser.add_argument("--starting-balance", type=float, default=10_000.0, help="Starting balance in $")
    parser.add_argument("--thresholds", default="0.0,0.05,0.10,0.15", help="Comma-separated edge thresholds")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    asyncio.run(run_backtest(
        config_path=args.config,
        symbol=args.symbol,
        bet_size=args.bet_size,
        starting_balance=args.starting_balance,
        thresholds=thresholds,
    ))
