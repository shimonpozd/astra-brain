"""
Async SQLAlchemy helpers for the brain service.

This module centralises creation of the PostgreSQL engine and session factory so
startup logic can initialise them once and store them on the FastAPI app state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(database_url: str) -> AsyncEngine:
    """
    Configure an async SQLAlchemy engine for PostgreSQL (asyncpg driver).

    Args:
        database_url: e.g. ``postgresql+asyncpg://user:pass@host:5432/db``.
    """
    if not database_url.startswith("postgresql+asyncpg://"):
        raise ValueError(
            "DATABASE_URL must use the asyncpg driver "
            "(postgresql+asyncpg://...)"
        )
    return create_async_engine(
        database_url,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return an async session factory bound to the given engine."""
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """
    Provide a transactional scope for a series of operations.

    Example:
        async with session_scope(factory) as session:
            session.add(obj)
    """
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def shutdown_engine(engine: Optional[AsyncEngine]) -> None:
    """Dispose the engine on application shutdown."""
    if engine is not None:
        await engine.dispose()
