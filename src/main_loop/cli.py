"""Command-line interface using Typer."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.common.config import load_config, load_env_settings
from src.common.logging import setup_logging, get_logger
from src.persistence import Database, Repository
from src.wallet_approval import ApprovalManager
from web3 import Web3

app = typer.Typer(
    name="polybot",
    help="Polymarket CLOB market-making bot",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


@app.command()
def run(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Enable paper trading mode",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Optional log file path",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs",
        help="Output logs in JSON format",
    ),
) -> None:
    """Start the trading bot."""
    # Setup logging
    setup_logging(level=log_level, json_output=json_logs, log_file=log_file)

    # Load config
    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    # Override dry_run from CLI
    if dry_run:
        cfg.dry_run.enabled = True

    console.print(Panel.fit(
        f"[green]Polybot Starting[/green]\n"
        f"Mode: {'[yellow]DRY RUN[/yellow]' if cfg.dry_run.enabled else '[red]LIVE[/red]'}\n"
        f"Markets: {len(cfg.markets)}\n"
        f"Config: {config}",
        title="Polybot",
    ))

    # Run trading loop
    from .runner import TradingLoop

    loop = TradingLoop(config=cfg, dry_run=cfg.dry_run.enabled)

    try:
        asyncio.run(loop.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")


@app.command()
def approve(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """Set token approvals for trading (USDC and CTF for all exchange contracts)."""
    setup_logging(level="INFO")

    # Load config and env
    try:
        cfg = load_config(config)
        env = load_env_settings()
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    if not env.private_key:
        console.print("[red]POLYBOT_PRIVATE_KEY not set in environment[/red]")
        raise typer.Exit(1)

    console.print("[yellow]Setting up token approvals...[/yellow]")

    async def do_approvals() -> None:
        web3 = Web3(Web3.HTTPProvider(cfg.network.rpc_url))

        if not web3.is_connected():
            console.print("[red]Failed to connect to RPC[/red]")
            raise typer.Exit(1)

        manager = ApprovalManager(web3=web3, private_key=env.private_key)

        # Check current status
        status = await manager.check_approvals()

        table = Table(title="Current Approval Status")
        table.add_column("Contract", style="cyan")
        table.add_column("Spender", style="cyan")
        table.add_column("Status", style="green")

        # USDC approvals
        table.add_row(
            "USDC.e", "Exchange",
            "[green]✓[/green]" if status.usdc_exchange_approved else "[red]✗[/red]",
        )
        table.add_row(
            "USDC.e", "NegRiskExchange",
            "[green]✓[/green]" if status.usdc_neg_risk_exchange_approved else "[red]✗[/red]",
        )
        table.add_row(
            "USDC.e", "NegRiskAdapter",
            "[green]✓[/green]" if status.usdc_neg_risk_adapter_approved else "[red]✗[/red]",
        )
        # CTF approvals
        table.add_row(
            "CTF", "Exchange",
            "[green]✓[/green]" if status.ctf_exchange_approved else "[red]✗[/red]",
        )
        table.add_row(
            "CTF", "NegRiskExchange",
            "[green]✓[/green]" if status.ctf_neg_risk_exchange_approved else "[red]✗[/red]",
        )
        table.add_row(
            "CTF", "NegRiskAdapter",
            "[green]✓[/green]" if status.ctf_neg_risk_adapter_approved else "[red]✗[/red]",
        )

        console.print(table)

        if not status.all_approved:
            # Set on-chain approvals
            console.print("\n[yellow]Setting on-chain approvals...[/yellow]")
            tx_hashes = await manager.ensure_approvals()

            if tx_hashes:
                console.print("\n[green]On-chain approvals set successfully![/green]")
                for name, tx_hash in tx_hashes.items():
                    console.print(f"  {name}: {tx_hash}")
        else:
            console.print("[green]All on-chain approvals already set![/green]")

        # CRITICAL: Sync allowances with Polymarket CLOB API
        # The CLOB API maintains its own cache of your balance/allowance state.
        # Even if on-chain approvals are set, you must call update_balance_allowance
        # to tell the CLOB API to refresh its view.
        console.print("\n[yellow]Syncing allowances with Polymarket CLOB API...[/yellow]")
        
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        
        # Use signature_type and funder from environment for proper wallet setup
        clob_client = ClobClient(
            host=cfg.network.clob_host,
            chain_id=cfg.network.chain_id,
            key=env.private_key,
            signature_type=env.signature_type,  # 0 for Phantom/EOA
            funder=env.funder_address if env.funder_address else None,
        )
        console.print(f"  Signature Type: {env.signature_type} (0=EOA/Phantom)")
        console.print(f"  Funder Address: {env.funder_address or 'None (using signer)'}")
        
        # Derive API credentials for L2 authentication
        try:
            api_creds = clob_client.create_or_derive_api_creds()
            clob_client.set_api_creds(api_creds)
        except Exception as e:
            console.print(f"[red]Failed to derive API credentials: {e}[/red]")
            raise typer.Exit(1)
        
        # Update USDC allowance (required for buying positions)
        usdc_synced = False
        try:
            usdc_params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            clob_client.update_balance_allowance(usdc_params)
            console.print("  [green]✓[/green] USDC allowance synced with CLOB API")
            usdc_synced = True
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to sync USDC allowance: {e}")
        
        # Update CTF allowance (for selling positions)
        # Note: CTF sync may fail without a specific token_id - this is OK for general approval
        # The allowance will be synced automatically when trading specific tokens
        try:
            ctf_params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
            clob_client.update_balance_allowance(ctf_params)
            console.print("  [green]✓[/green] CTF allowance synced with CLOB API")
        except Exception as e:
            # CTF sync often fails without a specific token - this is expected
            console.print(f"  [yellow]⚠[/yellow] CTF allowance sync skipped (synced per-token when trading)")
        
        if usdc_synced:
            console.print("\n[green]Approval setup complete! You can now place buy orders.[/green]")
        else:
            console.print("\n[red]USDC sync failed - orders may still fail.[/red]")
            console.print("[dim]Check that you have USDC.e balance on Polygon.[/dim]")

    asyncio.run(do_approvals())


@app.command()
def status(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """Show current positions and PNL."""
    setup_logging(level="WARNING")

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    async def show_status() -> None:
        db = Database(cfg.database.path)
        await db.connect()
        repo = Repository(db)

        # Get positions
        positions = await repo.get_nonzero_positions()

        if not positions:
            console.print("[yellow]No open positions[/yellow]")
        else:
            table = Table(title="Open Positions")
            table.add_column("Token", style="cyan")
            table.add_column("Size", justify="right")
            table.add_column("Avg Entry", justify="right")
            table.add_column("Realized PNL", justify="right")
            table.add_column("Unrealized PNL", justify="right")

            for pos in positions:
                table.add_row(
                    pos.token_id[:16] + "...",
                    f"{pos.size:.2f}",
                    f"{pos.avg_entry_price:.4f}" if pos.avg_entry_price else "-",
                    f"${pos.realized_pnl:.2f}",
                    f"${pos.unrealized_pnl:.2f}",
                )

            console.print(table)

        # Get recent fills
        fills = await repo.get_recent_fills(limit=10)

        if fills:
            console.print()
            table = Table(title="Recent Fills")
            table.add_column("Time", style="cyan")
            table.add_column("Side")
            table.add_column("Price", justify="right")
            table.add_column("Size", justify="right")
            table.add_column("Fee", justify="right")

            for fill in fills:
                side_color = "green" if fill.side.value == "BUY" else "red"
                table.add_row(
                    fill.created_at.strftime("%H:%M:%S"),
                    f"[{side_color}]{fill.side.value}[/{side_color}]",
                    f"{fill.price:.4f}",
                    f"{fill.size:.2f}",
                    f"${fill.fee:.4f}",
                )

            console.print(table)

        # Get recent snapshots
        snapshots = await repo.get_recent_snapshots(limit=1)

        if snapshots:
            snap = snapshots[0]
            console.print()
            console.print(Panel.fit(
                f"Total Equity: [green]${snap.total_equity:.2f}[/green]\n"
                f"Realized PNL: ${snap.total_realized_pnl:.2f}\n"
                f"Unrealized PNL: ${snap.total_unrealized_pnl:.2f}\n"
                f"Cash Balance: ${snap.cash_balance:.2f}\n"
                f"Last Update: {snap.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                title="Account Summary",
            ))

        await db.close()

    asyncio.run(show_status())


@app.command()
def events(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of events to show",
    ),
    event_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by event type",
    ),
) -> None:
    """Show recent event log."""
    setup_logging(level="WARNING")

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    async def show_events() -> None:
        db = Database(cfg.database.path)
        await db.connect()
        repo = Repository(db)

        events = await repo.get_recent_events(
            event_type=event_type,
            limit=limit,
        )

        if not events:
            console.print("[yellow]No events found[/yellow]")
        else:
            table = Table(title="Event Log")
            table.add_column("Time", style="cyan")
            table.add_column("Type")
            table.add_column("Severity")
            table.add_column("Message")

            for event in events:
                severity_color = {
                    "DEBUG": "dim",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red bold",
                }.get(event.severity.value, "white")

                table.add_row(
                    event.timestamp.strftime("%H:%M:%S"),
                    event.event_type,
                    f"[{severity_color}]{event.severity.value}[/{severity_color}]",
                    event.message[:50] + "..." if len(event.message) > 50 else event.message,
                )

            console.print(table)

        await db.close()

    asyncio.run(show_events())


@app.command()
def backfill(
    symbol: str = typer.Option(
        "BTC/USDT",
        "--symbol",
        "-s",
        help="Trading pair to backfill, e.g. BTC/USDT",
    ),
    start_date: str = typer.Option(
        ...,
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    ),
    end_date: str = typer.Option(
        ...,
        "--end-date",
        help="End date (YYYY-MM-DD)",
    ),
    timeframe: str = typer.Option(
        "1h",
        "--timeframe",
        "-t",
        help="Candle interval: 1m, 5m, 15m, 1h, 4h, 1d",
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """Backfill OHLCV candle data from Binance."""
    setup_logging(level="INFO")

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    from datetime import datetime

    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError as e:
        console.print(f"[red]Invalid date format: {e}[/red]")
        raise typer.Exit(1)

    async def do_backfill() -> None:
        from src.data_pipeline.binance_fetcher import backfill as _backfill
        from src.data_pipeline.gap_detector import detect_gaps, flag_gaps
        from src.data_pipeline.binance_fetcher import TIMEFRAME_SECONDS

        db = Database(cfg.database.path)
        await db.connect()
        repo = Repository(db)

        console.print(f"[yellow]Backfilling {symbol} {timeframe} from {start_date} to {end_date}...[/yellow]")

        try:
            total = await _backfill(symbol, start, end, timeframe=timeframe, repository=repo)
            console.print(f"[green]Backfill complete: {total} candles written[/green]")

            # Run gap detection
            tf_secs = TIMEFRAME_SECONDS.get(timeframe, 3600)
            prices = await repo.get_recent_crypto_prices(symbol, limit=total + 100)
            prices_sorted = sorted(prices, key=lambda p: p.timestamp)
            gaps = detect_gaps(prices_sorted, tf_secs)

            if gaps:
                console.print(f"[yellow]Detected {len(gaps)} gaps — flagging in EventLog[/yellow]")
                await flag_gaps(gaps, repo)
            else:
                console.print("[green]No gaps detected[/green]")

        except Exception as exc:
            console.print(f"[red]Backfill failed: {exc}[/red]")
            raise typer.Exit(1)
        finally:
            await db.close()

    asyncio.run(do_backfill())


@app.command()
def train(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    symbol: str = typer.Option(
        "BTC/USDT",
        "--symbol",
        "-s",
        help="Symbol to train on",
    ),
    output: str = typer.Option(
        "models/btc_prob_model.pkl",
        "--output",
        "-o",
        help="Path to save trained model",
    ),
) -> None:
    """Train the BTC probability model with walk-forward validation."""
    setup_logging(level="INFO")

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    async def do_train() -> None:
        import pandas as pd
        from src.probability_model.xgboost_model import XGBoostModel
        from src.probability_model.trainer import WalkForwardTrainer
        from src.probability_model.evaluator import print_eval_summary

        db = Database(cfg.database.path)
        await db.connect()
        repo = Repository(db)

        # Load all stored candles for the symbol
        prices = await repo.get_recent_crypto_prices(symbol, limit=500000)
        if len(prices) < 100:
            console.print(f"[red]Not enough data: {len(prices)} candles (need at least 100)[/red]")
            raise typer.Exit(1)

        prices_sorted = sorted(prices, key=lambda p: p.timestamp)

        # Build DataFrame from CryptoPrice records
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

        console.print(f"[yellow]Training on {len(df)} candles from {df.index[0]} to {df.index[-1]}[/yellow]")

        # Load options signals if available
        opts_df: pd.DataFrame | None = None
        try:
            from datetime import datetime as _dt
            data_start = df.index[0].to_pydatetime() if hasattr(df.index[0], "to_pydatetime") else df.index[0]
            data_end = df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else df.index[-1]
            opts_signals = await repo.get_options_signals("BTC", data_start, data_end)
            if opts_signals:
                opts_rows = [{"timestamp": s.timestamp, "dvol": s.dvol, "put_call_ratio": s.put_call_ratio} for s in opts_signals]
                opts_df = pd.DataFrame(opts_rows).set_index("timestamp")
                console.print(f"[yellow]Options signals loaded: {len(opts_df)} rows[/yellow]")
            else:
                console.print("[dim]No options signals in DB — training without options features[/dim]")
        except Exception as exc:
            console.print(f"[dim]Options signals unavailable ({exc}) — training without[/dim]")

        trainer = WalkForwardTrainer(model_class=XGBoostModel, config=cfg.belief)
        fold_results = trainer.run(df, options_df=opts_df)

        console.print(f"\n[green]Walk-forward complete: {len(fold_results)} folds[/green]")
        for i, result in enumerate(fold_results):
            console.print(f"\n[cyan]Fold {i + 1}[/cyan]")
            print_eval_summary(result.eval_result)

        # Save last fold's model as the production model
        if fold_results:
            import os
            os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
            fold_results[-1].model.save(output)
            console.print(f"\n[green]Model saved to {output}[/green]")

        await db.close()

    asyncio.run(do_train())


@app.command()
def backtest(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    symbol: str = typer.Option(
        "BTC/USDT",
        "--symbol",
        "-s",
        help="Symbol to backtest on",
    ),
    bet_size: float = typer.Option(
        100.0,
        "--bet-size",
        help="Dollar amount per bet",
    ),
    starting_balance: float = typer.Option(
        10_000.0,
        "--starting-balance",
        help="Starting account balance in $",
    ),
    thresholds: str = typer.Option(
        "0.0,0.05,0.10,0.15",
        "--thresholds",
        help="Comma-separated edge thresholds (skip bet if |pred-0.5| <= threshold)",
    ),
) -> None:
    """Run flat-bet backtest on walk-forward validation predictions."""
    setup_logging(level="INFO")

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    threshold_list = [float(t) for t in thresholds.split(",")]

    async def do_backtest() -> None:
        import pandas as pd
        from src.probability_model.xgboost_model import XGBoostModel
        from src.probability_model.trainer import WalkForwardTrainer
        from scripts.backtest_flat_bets import simulate_flat_bets, print_fold_summary, print_threshold_summary

        db = Database(cfg.database.path)
        await db.connect()
        repo = Repository(db)

        try:
            prices = await repo.get_recent_crypto_prices(symbol, limit=500_000)
            if len(prices) < 100:
                console.print(f"[red]Not enough data: {len(prices)} candles[/red]")
                raise typer.Exit(1)

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

            print_fold_summary(fold_results, all_preds)
            console.print()
            threshold_results = simulate_flat_bets(
                all_preds,
                bet_size=bet_size,
                starting_balance=starting_balance,
                thresholds=threshold_list,
            )
            print_threshold_summary(threshold_results, starting_balance)

            # Save backtest summary JSON for `polybot summary` command
            import json as _json
            import os as _os
            from datetime import datetime as _dt
            profitable_folds = sum(
                1 for fr in fold_results
                if not fr.val_predictions.empty
                and (fr.val_predictions["y_true"] == (fr.val_predictions["y_pred_calibrated"] > 0.5).astype(int)).mean() > 0.5
            )
            bt_summary = {
                "run_date": _dt.utcnow().isoformat(),
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
            _os.makedirs("models", exist_ok=True)
            with open("models/backtest_summary.json", "w") as _f:
                _json.dump(bt_summary, _f, indent=2)
            console.print("[dim]Saved to models/backtest_summary.json[/dim]")

        finally:
            await db.close()

    asyncio.run(do_backtest())


@app.command(name="backfill-options")
def backfill_options(
    symbol: str = typer.Option(
        "BTC",
        "--symbol",
        "-s",
        help="Currency to backfill DVOL for (BTC or ETH)",
    ),
    start_date: str = typer.Option(
        ...,
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    ),
    end_date: str = typer.Option(
        ...,
        "--end-date",
        help="End date (YYYY-MM-DD)",
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """Backfill Deribit DVOL (implied vol index) into options_signals table."""
    setup_logging(level="INFO")

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    from datetime import datetime as _datetime

    try:
        start = _datetime.fromisoformat(start_date)
        end = _datetime.fromisoformat(end_date)
    except ValueError as e:
        console.print(f"[red]Invalid date format: {e}[/red]")
        raise typer.Exit(1)

    async def do_backfill_options() -> None:
        from src.data_pipeline.deribit_options_fetcher import backfill_dvol

        db = Database(cfg.database.path)
        await db.connect()
        repo = Repository(db)

        console.print(f"[yellow]Backfilling DVOL for {symbol} from {start_date} to {end_date}...[/yellow]")

        try:
            total = await backfill_dvol(symbol, start, end, repository=repo, currency=symbol)
            console.print(f"[green]Options backfill complete: {total} rows written[/green]")
        except Exception as exc:
            console.print(f"[red]Options backfill failed: {exc}[/red]")
            raise typer.Exit(1)
        finally:
            await db.close()

    asyncio.run(do_backfill_options())


@app.command()
def summary(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """Show model status, backtest performance, live DVOL, and paper trading stats."""
    setup_logging(level="WARNING")

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)

    import json
    import os
    from datetime import datetime as _dt

    # --- Panel 1: Model Status ---
    model_path = cfg.belief.model_path
    model_exists = os.path.exists(model_path)
    if model_exists:
        mtime = _dt.fromtimestamp(os.path.getmtime(model_path))
        model_status = f"[green]Loaded[/green]  {model_path}\nLast modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}\nModel type: {cfg.belief.model_type}\nVol regime threshold: {cfg.belief.vol_regime_ratio_threshold}\nDirection threshold: {cfg.belief.direction_threshold}"
    else:
        model_status = f"[red]Not found[/red]  {model_path}\nRun: [cyan]polybot train[/cyan]"
    console.print(Panel.fit(model_status, title="Model Status"))

    # --- Panel 2: Backtest Performance ---
    summary_path = "models/backtest_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            bt = json.load(f)

        bt_table = Table(show_header=True, header_style="bold cyan")
        bt_table.add_column("Threshold")
        bt_table.add_column("Bets", justify="right")
        bt_table.add_column("Win Rate", justify="right")
        bt_table.add_column("Profit Factor", justify="right")
        bt_table.add_column("Sharpe", justify="right")
        bt_table.add_column("Max DD", justify="right")

        for t, s in bt.get("thresholds", {}).items():
            style = "bold yellow" if t in ("0.1", "0.10", "0.15") else ""
            bt_table.add_row(
                t, str(s["n_bets"]),
                f"{s['win_rate']:.1%}",
                f"{s['profit_factor']:.3f}",
                f"{s['sharpe']:.2f}",
                f"{s['max_drawdown']:.1%}",
                style=style,
            )

        bt_info = (
            f"Run: {bt.get('run_date', 'N/A')[:10]}\n"
            f"Folds: {bt.get('n_folds', '?')}  Profitable: {bt.get('profitable_folds', '?')}\n"
            f"Period: {bt.get('val_start', '')[:10]} → {bt.get('val_end', '')[:10]}\n"
            f"Predictions: {bt.get('n_predictions', '?')}"
        )
        console.print(Panel.fit(bt_info, title="Backtest Performance"))
        console.print(bt_table)
    else:
        console.print(Panel.fit(
            f"[yellow]No backtest summary found at {summary_path}[/yellow]\nRun: [cyan]polybot backtest[/cyan]",
            title="Backtest Performance",
        ))

    # --- Panel 3: Live Deribit DVOL ---
    async def fetch_dvol() -> dict | None:
        try:
            import aiohttp
            url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
            params = {"currency": "BTC", "resolution": "3600", "count": 1}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    data = await resp.json()
                    rows = data.get("result", {}).get("data", [])
                    if rows:
                        ts, open_, high, low, close = rows[-1]
                        dvol = float(close)
                        if dvol < 50:
                            regime, rec = "calm", "HIGH confidence"
                        elif dvol < 80:
                            regime, rec = "normal", "MEDIUM confidence"
                        elif dvol < 100:
                            regime, rec = "elevated", "LOW confidence"
                        else:
                            regime, rec = "extreme", "LOW confidence"
                        return {"dvol": dvol, "regime": regime, "recommendation": rec}
        except Exception as exc:
            return {"error": str(exc)}
        return None

    dvol_result = asyncio.run(fetch_dvol())
    if dvol_result and "dvol" in dvol_result:
        dvol_text = (
            f"DVOL: [bold]{dvol_result['dvol']:.1f}[/bold]\n"
            f"Regime: {dvol_result['regime']}\n"
            f"Model confidence: {dvol_result['recommendation']}"
        )
    elif dvol_result and "error" in dvol_result:
        dvol_text = f"[red]Failed to fetch DVOL: {dvol_result['error']}[/red]"
    else:
        dvol_text = "[yellow]No DVOL data returned[/yellow]"
    console.print(Panel.fit(dvol_text, title="Live Deribit DVOL"))

    # --- Panel 4: Paper Trading P&L ---
    async def fetch_paper_pnl() -> str:
        from datetime import date
        db = Database(cfg.database.path)
        await db.connect()
        repo = Repository(db)
        try:
            today_start = _dt.combine(date.today(), _dt.min.time())
            fills = await repo.get_recent_fills(limit=10000)
            today_fills = [f for f in fills if f.created_at >= today_start]

            gross_profit = sum(f.price * f.size for f in today_fills if f.side.value == "SELL")
            gross_loss = sum(f.price * f.size for f in today_fills if f.side.value == "BUY")
            realized_pnl = gross_profit - gross_loss
            net_pos = sum(f.size if f.side.value == "BUY" else -f.size for f in today_fills)

            return (
                f"Fills today: {len(today_fills)}\n"
                f"Gross profit (sells): ${gross_profit:.2f}\n"
                f"Gross cost (buys): ${gross_loss:.2f}\n"
                f"Realized P&L: {'[green]' if realized_pnl >= 0 else '[red]'}${realized_pnl:.2f}{'[/green]' if realized_pnl >= 0 else '[/red]'}\n"
                f"Net position: {net_pos:.2f} shares"
            )
        except Exception as exc:
            return f"[red]DB error: {exc}[/red]"
        finally:
            await db.close()

    pnl_text = asyncio.run(fetch_paper_pnl())
    console.print(Panel.fit(pnl_text, title="Paper Trading P&L (Today)"))


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
