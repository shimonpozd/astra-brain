"""
Drop the legacy unique constraint on chat_threads.session_id and replace it with
the correct (user_id, session_id) uniqueness.
"""

from __future__ import annotations
import argparse
import asyncio

from sqlalchemy import text

from brain_service.core.database import create_engine
from brain_service.core.settings import Settings


async def fix_constraint(*, dry_run: bool = False) -> None:
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)

    drop_stmt = text("ALTER TABLE chat_threads DROP CONSTRAINT IF EXISTS uq_chat_threads_session_id")
    create_stmt = text(
        "ALTER TABLE chat_threads ADD CONSTRAINT uq_chat_threads_user_session UNIQUE (user_id, session_id)"
    )

    if dry_run:
        print("Would execute:")
        print(drop_stmt.text)
        print(create_stmt.text)
    else:
        async with engine.begin() as conn:
            await conn.execute(drop_stmt)
            await conn.execute(create_stmt)
        print("Constraint updated to UNIQUE (user_id, session_id).")

    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix chat thread unique constraints.")
    parser.add_argument("--dry-run", action="store_true", help="Show the SQL statements without applying them.")
    args = parser.parse_args()
    asyncio.run(fix_constraint(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
