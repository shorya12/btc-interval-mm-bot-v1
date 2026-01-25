"""Async SQLite database connection and management."""

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite

from src.common.errors import DatabaseError
from src.common.logging import get_logger

logger = get_logger(__name__)

# Path to schema file
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Async SQLite database connection manager."""

    def __init__(self, db_path: str = "polybot.db") -> None:
        """
        Initialize database connection manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open database connection and run migrations."""
        async with self._lock:
            if self._connection is not None:
                return

            try:
                self._connection = await aiosqlite.connect(self.db_path)
                self._connection.row_factory = aiosqlite.Row

                # Enable foreign keys
                await self._connection.execute("PRAGMA foreign_keys = ON")

                # Run migrations
                await self._run_migrations()

                logger.info("database_connected", path=self.db_path)
            except Exception as e:
                raise DatabaseError(f"Failed to connect to database: {e}")

    async def close(self) -> None:
        """Close database connection."""
        async with self._lock:
            if self._connection is not None:
                await self._connection.close()
                self._connection = None
                logger.info("database_disconnected", path=self.db_path)

    async def _run_migrations(self) -> None:
        """Run database migrations from schema.sql."""
        if self._connection is None:
            raise DatabaseError("Database not connected")

        # Read schema file
        if not SCHEMA_PATH.exists():
            raise DatabaseError(f"Schema file not found: {SCHEMA_PATH}")

        schema_sql = SCHEMA_PATH.read_text()

        try:
            await self._connection.executescript(schema_sql)
            await self._connection.commit()
            logger.info("migrations_applied")
        except Exception as e:
            raise DatabaseError(f"Failed to apply migrations: {e}")

    @property
    def connection(self) -> aiosqlite.Connection:
        """Get the database connection, raising if not connected."""
        if self._connection is None:
            raise DatabaseError("Database not connected. Call connect() first.")
        return self._connection

    async def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> aiosqlite.Cursor:
        """
        Execute a SQL statement.

        Args:
            sql: SQL statement to execute
            parameters: Optional parameters for the statement

        Returns:
            Cursor with results
        """
        try:
            if parameters:
                return await self.connection.execute(sql, parameters)
            return await self.connection.execute(sql)
        except Exception as e:
            logger.error("sql_execute_error", sql=sql[:100], error=str(e))
            raise DatabaseError(f"SQL execution failed: {e}")

    async def execute_many(
        self,
        sql: str,
        parameters: list[tuple[Any, ...]] | list[dict[str, Any]],
    ) -> None:
        """
        Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement to execute
            parameters: List of parameter tuples/dicts
        """
        try:
            await self.connection.executemany(sql, parameters)
        except Exception as e:
            logger.error("sql_executemany_error", sql=sql[:100], error=str(e))
            raise DatabaseError(f"SQL executemany failed: {e}")

    async def fetch_one(
        self,
        sql: str,
        parameters: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch a single row.

        Args:
            sql: SQL query
            parameters: Optional parameters

        Returns:
            Row as dictionary or None if not found
        """
        cursor = await self.execute(sql, parameters)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def fetch_all(
        self,
        sql: str,
        parameters: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all rows.

        Args:
            sql: SQL query
            parameters: Optional parameters

        Returns:
            List of rows as dictionaries
        """
        cursor = await self.execute(sql, parameters)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.connection.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.connection.rollback()

    async def __aenter__(self) -> "Database":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if exc_type is not None:
            await self.rollback()
        await self.close()
