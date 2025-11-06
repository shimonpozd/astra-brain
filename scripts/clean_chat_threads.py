"""
Utility to clean legacy chat thread records that were assigned to the wrong user.

Old deployments stored chat thread metadata keyed only by ``session_id`` which
allowed a session created by one user to be reassigned to another user. This
script analyses the ``chat_threads`` table and removes the stale rows so that
each session is owned by the correct user going forward.

Usage (inside the repo root):

    python -m brain_service.scripts.clean_chat_threads            # execute cleanup
    python -m brain_service.scripts.clean_chat_threads --dry-run  # report only
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence
import uuid

from sqlalchemy import delete, select

from brain_service.core.database import create_engine, create_session_factory
from brain_service.core.settings import Settings
from brain_service.models.db import ChatThread, User


@dataclass(frozen=True)
class ThreadRecord:
    """Lightweight representation of a chat thread joined with its owner."""

    thread_id: uuid.UUID
    session_id: str
    last_modified: datetime
    user_id: uuid.UUID
    username: str
    role: str


def _choose_rows_to_remove(records: Sequence[ThreadRecord]) -> list[ThreadRecord]:
    """
    Determine which records should be removed for a specific session_id.

    Strategy:
        * Prefer keeping the most recent non-admin row (real user ownership).
        * If no non-admin rows exist, keep the most recent row regardless.
        * Remove all older duplicates for the same session.
    """

    if len(records) <= 1:
        return []

    non_admin = [record for record in records if record.role != "admin"]
    keep_candidate = (non_admin or records)[0]

    to_remove: list[ThreadRecord] = []
    seen_users: set[uuid.UUID] = {keep_candidate.user_id}

    for record in records:
        if record.thread_id == keep_candidate.thread_id:
            continue
        if record.user_id in seen_users:
            to_remove.append(record)
            continue
        # If we reach here we have a second legitimate user for this session.
        # This should only happen due to legacy contamination, so drop it and
        # keep the primary candidate.
        to_remove.append(record)

    return to_remove


async def _collect_records() -> list[ThreadRecord]:
    """Fetch chat thread rows joined with user data."""
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            stmt = (
                select(
                    ChatThread.id,
                    ChatThread.session_id,
                    ChatThread.last_modified,
                    ChatThread.user_id,
                    User.username,
                    User.role,
                )
                .join(User, ChatThread.user_id == User.id)
                .order_by(ChatThread.session_id, ChatThread.last_modified.desc())
            )
            result = await session.execute(stmt)
            rows = [
                ThreadRecord(
                    thread_id=row[0],
                    session_id=row[1],
                    last_modified=row[2],
                    user_id=row[3],
                    username=row[4],
                    role=row[5],
                )
                for row in result.all()
            ]
        return rows
    finally:
        await engine.dispose()


async def _delete_threads(thread_ids: Iterable[uuid.UUID]) -> int:
    """Delete the supplied thread IDs, returning the count removed."""
    ids = list(thread_ids)
    if not ids:
        return 0

    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(delete(ChatThread).where(ChatThread.id.in_(ids)))
        return len(ids)
    finally:
        await engine.dispose()


async def clean_chat_threads(*, dry_run: bool = False) -> int:
    """
    Identify and remove legacy chat thread rows.

    Returns the number of rows that would be / were removed.
    """
    rows = await _collect_records()
    grouped: dict[str, list[ThreadRecord]] = defaultdict(list)
    for record in rows:
        grouped[record.session_id].append(record)

    # Ensure deterministic ordering inside each group (already ordered desc).
    for record_list in grouped.values():
        record_list.sort(key=lambda rec: rec.last_modified, reverse=True)

    to_remove: list[ThreadRecord] = []

    for session_id, record_list in grouped.items():
        candidates = _choose_rows_to_remove(record_list)
        if candidates:
            print(f"[LEGACY] session {session_id}: removing {len(candidates)} stale row(s)")
            for candidate in candidates:
                print(
                    f"    -> user={candidate.username} ({candidate.role}) "
                    f"thread_id={candidate.thread_id} last_modified={candidate.last_modified.isoformat()}"
                )
            to_remove.extend(candidates)

    if not to_remove:
        print("No legacy chat thread rows found.")
        return 0

    if dry_run:
        print(f"[DRY RUN] Would remove {len(to_remove)} chat thread row(s).")
        return len(to_remove)

    removed = await _delete_threads(record.thread_id for record in to_remove)
    print(f"Removed {removed} legacy chat thread row(s).")
    return removed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean legacy chat thread rows")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without deleting anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        removed = asyncio.run(clean_chat_threads(dry_run=args.dry_run))
    except KeyboardInterrupt:
        print("Aborted.")
        return
    if args.dry_run:
        print(f"Dry run complete. {removed} row(s) would be removed.")


if __name__ == "__main__":
    main()
