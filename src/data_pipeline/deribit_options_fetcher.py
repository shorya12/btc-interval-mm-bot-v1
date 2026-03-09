"""
Deribit options data fetcher.

Fetches BTC DVOL (implied volatility index) historical data via the
Deribit public REST API. Also fetches a live put/call OI snapshot.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from src.common.logging import get_logger

logger = get_logger(__name__)

DERIBIT_API_BASE = "https://www.deribit.com/api/v2/public"


async def _get(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Make a single GET request to the Deribit public API."""
    url = f"{DERIBIT_API_BASE}/{endpoint}"
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    if "error" in payload:
        raise RuntimeError(f"Deribit API error: {payload['error']}")
    return payload.get("result", {})


async def _fetch_dvol_page(
    session: aiohttp.ClientSession,
    currency: str,
    start_ms: int,
    end_ms: int,
    continuation: str | None = None,
) -> tuple[list[tuple[int, float]], str | None]:
    """
    Fetch one page of DVOL data.

    Returns:
        (rows, continuation_token) where rows = [(timestamp_ms, dvol_close), ...]
    """
    params: dict[str, Any] = {
        "currency": currency,
        "resolution": 3600,  # 1h candles
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
    }
    if continuation:
        params["continuation"] = continuation

    result = await _get(session, "get_volatility_index_data", params)
    raw_data = result.get("data", [])
    # Each row: [ts_ms, open, high, low, close]
    rows = [(int(row[0]), float(row[4])) for row in raw_data]
    next_continuation = result.get("continuation")
    return rows, next_continuation


async def _fetch_put_call_ratio(
    session: aiohttp.ClientSession,
    currency: str = "BTC",
) -> float | None:
    """
    Fetch live put/call open interest ratio from Deribit.

    Returns put_call_ratio = put_oi / call_oi, or None on failure.
    """
    try:
        result = await _get(
            session,
            "get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
        )
        put_oi = 0.0
        call_oi = 0.0
        for instrument in result:
            name = instrument.get("instrument_name", "")
            oi = instrument.get("open_interest", 0.0) or 0.0
            if name.endswith("-P"):
                put_oi += oi
            elif name.endswith("-C"):
                call_oi += oi
        if call_oi > 0:
            return put_oi / call_oi
    except Exception as exc:
        logger.warning("put_call_ratio_fetch_failed", error=str(exc))
    return None


async def backfill_dvol(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    repository: Any,  # Repository instance
    currency: str = "BTC",
) -> int:
    """
    Backfill DVOL data from Deribit into the options_signals table.

    Args:
        symbol: Ignored (always BTC), kept for API consistency
        start_date: Inclusive start (UTC)
        end_date: Exclusive end (UTC)
        repository: Repository instance for DB writes
        currency: Deribit currency code ("BTC")

    Returns:
        Number of rows written
    """
    from src.persistence.models import OptionsSignal

    start_ms = int(start_date.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end_date.replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Fetch live put/call ratio (only available as a snapshot)
    total_written = 0

    async with aiohttp.ClientSession() as session:
        # Fetch put/call ratio once (live snapshot)
        put_call_ratio = await _fetch_put_call_ratio(session, currency)
        if put_call_ratio is not None:
            logger.info("put_call_ratio_fetched", put_call_ratio=put_call_ratio)

        # Paginate DVOL
        continuation: str | None = None
        all_rows: list[tuple[int, float]] = []

        while True:
            rows, continuation = await _fetch_dvol_page(
                session, currency, start_ms, end_ms, continuation
            )
            all_rows.extend(rows)
            logger.info("dvol_page_fetched", n_rows=len(rows), has_continuation=continuation is not None)
            if not continuation:
                break

        logger.info("dvol_total_rows_fetched", n=len(all_rows))

        for ts_ms, dvol_close in all_rows:
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
            signal = OptionsSignal(
                id=None,
                timestamp=ts,
                symbol=currency,
                dvol=dvol_close,
                put_call_ratio=None,  # Historical P/C not available
                source="deribit",
            )
            await repository.create_options_signal(signal)
            total_written += 1

    logger.info("dvol_backfill_complete", total_written=total_written)
    return total_written
