import json
import uuid
from datetime import datetime, timezone
from fnmatch import fnmatch

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from brain_service.core.database import create_engine, create_session_factory
from brain_service.core.settings import Settings
from brain_service.models.db import ChatThread, User
from brain_service.services.session_service import SessionService
from brain_service.services.user_service import UserService


class FakeRedis:
    """Minimal async Redis stub for exercising session listing logic."""

    def __init__(self, initial=None):
        self._store: dict[str, str] = dict(initial or {})

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str):
        self._store[key] = value

    async def delete(self, key: str):
        self._store.pop(key, None)

    async def scan_iter(self, pattern: str):
        for key in list(self._store.keys()):
            if fnmatch(key, pattern):
                yield key


def _session_payload(session_id: str, user_id: uuid.UUID, name: str) -> str:
    return json.dumps(
        {
            "persistent_session_id": session_id,
            "user_id": str(user_id),
            "name": name,
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }
    )


@pytest.mark.asyncio
async def test_upsert_thread_rejects_cross_user_reassignment():
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    session_factory = create_session_factory(engine)
    user_service = UserService(session_factory, encryption_secret="test-secret")

    user1 = user2 = None
    shared_session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    try:
        username1 = f"test_user_{uuid.uuid4().hex[:8]}"
        username2 = f"test_user_{uuid.uuid4().hex[:8]}"
        user1 = await user_service.create_user(username1, "password123!")
        user2 = await user_service.create_user(username2, "password123!")

        await user_service.upsert_thread(
            session_id=shared_session_id,
            user_id=user1.id,
            title="User1 Chat",
            last_modified=now,
            metadata={"source": "test"},
        )

        with pytest.raises(IntegrityError):
            await user_service.upsert_thread(
                session_id=shared_session_id,
                user_id=user2.id,
                title="User2 Chat",
                last_modified=now,
                metadata={"source": "test"},
            )

        async with session_factory() as session:
            result = await session.execute(
                select(ChatThread.user_id, ChatThread.title).where(
                    ChatThread.session_id == shared_session_id
                )
            )
            owner_id, title = result.one()

        assert owner_id == user1.id
        assert title == "User1 Chat"
    finally:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(ChatThread).where(ChatThread.session_id == shared_session_id)
                )
                if user1:
                    await session.execute(delete(User).where(User.id == user1.id))
                if user2:
                    await session.execute(delete(User).where(User.id == user2.id))
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_all_sessions_returns_only_current_user_threads():
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    session_factory = create_session_factory(engine)
    user_service = UserService(session_factory, encryption_secret="test-secret")

    user1 = user2 = None
    session1 = str(uuid.uuid4())
    session2 = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    try:
        username1 = f"test_user_{uuid.uuid4().hex[:8]}"
        username2 = f"test_user_{uuid.uuid4().hex[:8]}"
        user1 = await user_service.create_user(username1, "password123!")
        user2 = await user_service.create_user(username2, "password123!")

        await user_service.upsert_thread(
            session_id=session1,
            user_id=user1.id,
            title="User1 Primary Chat",
            last_modified=now,
        )
        await user_service.upsert_thread(
            session_id=session2,
            user_id=user2.id,
            title="User2 Primary Chat",
            last_modified=now,
        )

        fake_redis = FakeRedis(
            {
                f"session:{user1.id}:{session1}": _session_payload(session1, user1.id, "Redis Name 1"),
                f"session:{user2.id}:{session2}": _session_payload(session2, user2.id, "Redis Name 2"),
                "session:legacy_shared": _session_payload(session1, user1.id, "Should Be Ignored"),
            }
        )

        session_service = SessionService(fake_redis, user_service)

        user1_sessions = await session_service.get_all_sessions(str(user1.id))
        user2_sessions = await session_service.get_all_sessions(str(user2.id))

        assert [entry["session_id"] for entry in user1_sessions] == [session1]
        assert user1_sessions[0]["name"] == "Redis Name 1"

        assert [entry["session_id"] for entry in user2_sessions] == [session2]
        assert user2_sessions[0]["name"] == "Redis Name 2"
    finally:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(ChatThread).where(ChatThread.session_id.in_([session1, session2]))
                )
                if user1:
                    await session.execute(delete(User).where(User.id == user1.id))
                if user2:
                    await session.execute(delete(User).where(User.id == user2.id))
        await engine.dispose()
