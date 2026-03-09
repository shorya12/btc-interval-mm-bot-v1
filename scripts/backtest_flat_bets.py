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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    fee_rate: float = 0.25,
    fee_exponent: float = 2.0,
) -> dict[float, dict]:
    """
    Run flat-bet simulation for each confidence threshold.

    Args:
        predictions: DataFrame with columns: timestamp, y_true, y_pred_calibrated, fold_index
        bet_size: Dollar amount per bet
        starting_balance: Starting account balance
        thresholds: List of edge thresholds — skip bet if |pred - 0.5| <= threshold
        fee_rate: Polymarket taker fee rate (0.25 for crypto markets)
        fee_exponent: Fee formula exponent (2 for crypto markets)
            fee = bet_size * fee_rate * (p * (1-p))^fee_exponent

    Returns:
        Dict mapping threshold -> stats dict
    """
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.10, 0.15]

    results = {}
    for threshold in thresholds:
        balance = starting_balance
        balance_series = [balance]
        timestamps = []
        wins = 0
        losses = 0
        total_fees = 0.0

        for _, row in predictions.iterrows():
            pred = float(row["y_pred_calibrated"])
            actual = int(row["y_true"])

            # Skip low-conviction bets
            if abs(pred - 0.5) <= threshold:
                continue

            # Taker fee: fee = bet_size * fee_rate * (p * (1-p))^exponent
            fee = bet_size * fee_rate * (pred * (1.0 - pred)) ** fee_exponent
            total_fees += fee

            # Determine bet direction
            bet_up = pred > 0.5
            win = (bet_up and actual == 1) or (not bet_up and actual == 0)

            if win:
                balance += bet_size - fee
                wins += 1
            else:
                balance -= bet_size + fee
                losses += 1

            balance_series.append(balance)
            timestamps.append(row["timestamp"])

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
            "total_fees": total_fees,
            "balance_series": balance_series,
            "timestamps": timestamps,
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
    table.add_column("Fees Paid ($)", justify="right", style="dim")

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
            f"{stats.get('total_fees', 0.0):.2f}",
        )

    console.print(table)


def bootstrap_win_rate_ci(
    predictions: pd.DataFrame,
    threshold: float,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
) -> dict:
    """Compute bootstrap 95% CI for win rate at the given threshold."""
    mask = (predictions["y_pred_calibrated"] - 0.5).abs() > threshold
    subset = predictions[mask]
    if subset.empty:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n_bets": 0, "p_value_one_tailed": 1.0}

    pred = subset["y_pred_calibrated"].values
    actual = subset["y_true"].values
    wins = ((pred > 0.5) == (actual == 1)).astype(float)

    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(wins, size=len(wins), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    mean = float(wins.mean())
    p_value = float((boot_means <= 0.50).mean())

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "n_bets": len(wins),
        "p_value_one_tailed": p_value,
    }


def monte_carlo_p_value(
    win_rate: float,
    n_bets: int,
    n_simulations: int = 100_000,
) -> dict:
    """Monte Carlo p-value under H₀ = 50% coin flip."""
    if n_bets == 0:
        return {"p_value": 1.0, "z_score": 0.0, "significant_at_95": False, "significant_at_99": False}

    rng = np.random.default_rng(42)
    sim_wins = rng.binomial(n=n_bets, p=0.5, size=n_simulations)
    sim_rates = sim_wins / n_bets
    p_value = float((sim_rates >= win_rate).mean())
    z_score = (win_rate - 0.5) / math.sqrt(0.25 / n_bets)

    return {
        "p_value": p_value,
        "z_score": z_score,
        "significant_at_95": p_value < 0.05,
        "significant_at_99": p_value < 0.01,
    }


def plot_cumulative_pnl(
    threshold_results: dict[float, dict],
    starting_balance: float,
    output_path: str = "models/cumulative_pnl.png",
) -> None:
    """Save a cumulative PnL chart for each threshold."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for t, stats in threshold_results.items():
        series = stats.get("balance_series", [])
        if len(series) < 2:
            continue
        label = f"threshold={t:.2f} (WR={stats['win_rate']:.1%}, Sharpe={stats['sharpe']:.2f})"
        ax.plot(range(len(series)), series, label=label, linewidth=1.5)

    ax.axhline(y=starting_balance, color="gray", linestyle="--", linewidth=1.0, label="Starting balance")
    ax.set_title("Cumulative PnL — Walk-Forward Backtest")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    console.print(f"[dim]Chart saved to {output_path}[/dim]")


def run_significance_tests(
    predictions: pd.DataFrame,
    threshold_results: dict[float, dict],
    starting_balance: float,
    n_bootstrap: int = 10_000,
    n_simulations: int = 100_000,
) -> dict:
    """Run bootstrap CIs + Monte Carlo p-values for each threshold and print summary table."""
    table = Table(title="Significance Tests", show_lines=True)
    table.add_column("Threshold", style="cyan")
    table.add_column("Bets", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Significant", justify="center")

    combined: dict[str, dict] = {}
    for threshold, stats in threshold_results.items():
        n_bets = stats["n_bets"]
        win_rate = stats["win_rate"]

        boot = bootstrap_win_rate_ci(predictions, threshold, n_bootstrap=n_bootstrap)
        mc = monte_carlo_p_value(win_rate, n_bets, n_simulations=n_simulations)

        sig_label = (
            "[green]✓ (99%)[/green]" if mc["significant_at_99"]
            else "[yellow]✓ (95%)[/yellow]" if mc["significant_at_95"]
            else "[red]✗[/red]"
        )

        table.add_row(
            f"{threshold:.2f}",
            str(n_bets),
            f"{win_rate:.1%}",
            f"[{boot['lower']:.1%}, {boot['upper']:.1%}]",
            f"{mc['p_value']:.4f}",
            sig_label,
        )

        combined[str(threshold)] = {
            "n_bets": n_bets,
            "win_rate": round(win_rate, 4),
            "bootstrap_ci_lower": round(boot["lower"], 4),
            "bootstrap_ci_upper": round(boot["upper"], 4),
            "bootstrap_p_value_one_tailed": round(boot["p_value_one_tailed"], 4),
            "mc_p_value": round(mc["p_value"], 4),
            "z_score": round(mc["z_score"], 4),
            "significant_at_95": mc["significant_at_95"],
            "significant_at_99": mc["significant_at_99"],
        }

    console.print(table)
    return combined


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

        # Significance tests
        console.print()
        sig_results = run_significance_tests(all_preds, threshold_results, starting_balance)
        os.makedirs("models", exist_ok=True)
        with open("models/significance_tests.json", "w") as f:
            json.dump(sig_results, f, indent=2)
        console.print("[dim]Saved to models/significance_tests.json[/dim]")

        # Cumulative PnL chart
        plot_cumulative_pnl(threshold_results, starting_balance)

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
                    "total_fees": round(s.get("total_fees", 0.0), 2),
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
