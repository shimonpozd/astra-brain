from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from brain_service.core.security import hash_password, verify_password
from brain_service.models.db import (
    ChatThread,
    User,
    UserApiKey,
    UserSession,
    UserLoginEvent,
)
from brain_service.utils.crypto import decrypt_value, encrypt_value, EncryptionError


def _normalize_username(value: str) -> str:
    return value.strip().lower()


def _normalize_phone(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = "".join(ch for ch in value if ch.isdigit() or ch == "+")
    return normalized or None


class UserAlreadyExistsError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


class ApiKeyLimitExceeded(Exception):
    pass


class UserService:
    def __init__(self, session_factory, *, encryption_secret: str):
        self._session_factory = session_factory
        self._encryption_secret = encryption_secret

    async def create_user(
        self,
        username: str,
        password: str,
        role: str = "member",
        is_active: bool = True,
        created_manually: bool = False,
        *,
        phone_number: Optional[str] = None,
    ) -> User:
        normalized_username = _normalize_username(username)
        normalized_phone = _normalize_phone(phone_number)

        async with self._session_factory() as session:
            async with session.begin():
                if normalized_phone:
                    existing = await session.scalars(
                        select(User).where(User.phone_number == normalized_phone)
                    )
                    if existing.first():
                        raise UserAlreadyExistsError(normalized_phone)

                user = User(
                    username=normalized_username,
                    password_hash=hash_password(password),
                    role=role,
                    is_active=is_active,
                    created_manually=created_manually,
                    phone_number=normalized_phone,
                )
                session.add(user)
                try:
                    await session.flush()
                except IntegrityError as exc:
                    raise UserAlreadyExistsError(username) from exc
                await session.refresh(user)
                return user

    async def update_user(
        self,
        username: str,
        *,
        password: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
        phone_number: Optional[str] = None,
    ) -> User:
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.scalars(
                    select(User).where(User.username == username.lower())
                )
                user = result.first()
                if not user:
                    raise UserNotFoundError(username)

                if password is not None:
                    user.password_hash = hash_password(password)
                if role is not None:
                    user.role = role
                if is_active is not None:
                    user.is_active = is_active
                if phone_number is not None:
                    normalized_phone = _normalize_phone(phone_number)
                    if (
                        normalized_phone
                        and normalized_phone != user.phone_number
                    ):
                        existing = await session.scalars(
                            select(User).where(User.phone_number == normalized_phone)
                        )
                        if existing.first():
                            raise UserAlreadyExistsError(normalized_phone)
                    user.phone_number = normalized_phone
                await session.flush()
                await session.refresh(user)
                return user

    async def list_users(self) -> List[User]:
        async with self._session_factory() as session:
            result = await session.scalars(select(User).order_by(User.username))
            return list(result)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(User).where(User.username == username.lower())
            )
            return result.first()

    async def get_user_by_phone(self, phone_number: str) -> Optional[User]:
        normalized_phone = _normalize_phone(phone_number)
        if not normalized_phone:
            return None
        async with self._session_factory() as session:
            result = await session.scalars(
                select(User).where(User.phone_number == normalized_phone)
            )
            return result.first()

    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(User).where(User.id == user_id)
            )
            return result.first()

    async def authenticate(
        self,
        identifier: str,
        password: str,
        *,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[User]:
        normalized_identifier = identifier.strip()
        user = await self.get_user_by_username(normalized_identifier)
        if not user:
            user = await self.get_user_by_phone(normalized_identifier)

        reason = None
        if not user:
            reason = "user_not_found"
        elif not user.is_active:
            reason = "user_inactive"
        elif not verify_password(password, user.password_hash):
            reason = "invalid_password"

        if reason:
            await self.record_login_event(
                user_id=user.id if user else None,
                username=normalized_identifier,
                success=False,
                ip_address=ip_address,
                user_agent=user_agent,
                reason=reason,
            )
            return None

        await self._record_login(user.id)
        await self.record_login_event(
            user_id=user.id,
            username=normalized_identifier,
            success=True,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return user

    async def has_admin(self) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                select(User.id).where(User.role == "admin").limit(1)
            )
            return result.scalar_one_or_none() is not None

    async def deactivate_user(self, user_id: uuid.UUID) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                user = await session.get(User, user_id)
                if not user:
                    raise UserNotFoundError(str(user_id))
                user.is_active = False

    async def delete_user(self, user_id: uuid.UUID) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                user = await session.get(User, user_id)
                if not user:
                    raise UserNotFoundError(str(user_id))
                await session.delete(user)
                # Related API keys cascade delete

    async def upsert_thread(
        self,
        *,
        session_id: str,
        user_id: uuid.UUID,
        title: Optional[str],
        last_modified,
        metadata: Optional[dict] = None,
    ) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.scalars(
                    select(ChatThread).where(
                        ChatThread.session_id == session_id,
                        ChatThread.user_id == user_id,
                    )
                )
                thread = result.first()
                if thread:
                    thread.user_id = user_id
                    thread.title = title
                    thread.last_modified = last_modified
                    thread.metadata_json = metadata
                else:
                    session.add(
                        ChatThread(
                            session_id=session_id,
                            user_id=user_id,
                            title=title,
                            last_modified=last_modified,
                            metadata_json=metadata,
                        )
                    )

    async def list_threads_for_user(self, user_id: uuid.UUID) -> list[ChatThread]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(ChatThread)
                .where(ChatThread.user_id == user_id)
                .order_by(ChatThread.last_modified.desc())
            )
            return list(result)

    async def get_thread_for_user(
        self, user_id: uuid.UUID, session_id: str
    ) -> Optional[ChatThread]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(ChatThread).where(
                    ChatThread.user_id == user_id,
                    ChatThread.session_id == session_id,
                )
            )
            return result.first()

    async def delete_thread(self, user_id: uuid.UUID, session_id: str) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.scalars(
                    select(ChatThread).where(
                        ChatThread.user_id == user_id,
                        ChatThread.session_id == session_id,
                    )
                )
                thread = result.first()
                if not thread:
                    return False
                await session.delete(thread)
                return True

    async def list_api_keys(self, user_id: uuid.UUID) -> List[UserApiKey]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(UserApiKey)
                .where(UserApiKey.user_id == user_id)
                .order_by(UserApiKey.created_at.desc())
            )
            return list(result)

    async def create_api_key(
        self,
        user_id: uuid.UUID,
        provider: str,
        api_key: str,
        *,
        daily_limit: Optional[int] = None,
    ) -> UserApiKey:
        last_four = api_key[-4:] if api_key else ""
        try:
            encrypted = encrypt_value(api_key, self._encryption_secret)
        except EncryptionError as exc:
            raise ValueError("Failed to store API key") from exc

        async with self._session_factory() as session:
            async with session.begin():
                user = await session.get(User, user_id)
                if not user:
                    raise UserNotFoundError(str(user_id))

                model = UserApiKey(
                    user_id=user_id,
                    provider=provider,
                    api_key_encrypted=encrypted,
                    last_four=last_four,
                    daily_limit=daily_limit,
                    usage_today=0,
                    last_reset_at=datetime.now(timezone.utc),
                    is_active=True,
                )
                session.add(model)
                await session.flush()
                await session.refresh(model)
                return model

    async def update_api_key(
        self,
        key_id: uuid.UUID,
        *,
        daily_limit: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> UserApiKey:
        async with self._session_factory() as session:
            async with session.begin():
                key = await session.get(UserApiKey, key_id)
                if not key:
                    raise UserNotFoundError(str(key_id))
                if daily_limit is not None:
                    key.daily_limit = daily_limit
                if is_active is not None:
                    key.is_active = is_active
                await session.flush()
                await session.refresh(key)
                return key

    async def delete_api_key(self, key_id: uuid.UUID) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                key = await session.get(UserApiKey, key_id)
                if not key:
                    raise UserNotFoundError(str(key_id))
                await session.delete(key)

    async def get_active_api_key(
        self,
        user_id: uuid.UUID,
        provider: str = "openrouter",
    ) -> Optional[tuple[UserApiKey, str]]:
        async with self._session_factory() as session:
            await self._reset_usage_if_needed(session, user_id, provider)
            result = await session.scalars(
                select(UserApiKey)
                .where(
                    UserApiKey.user_id == user_id,
                    UserApiKey.provider == provider,
                    UserApiKey.is_active.is_(True),
                )
                .order_by(UserApiKey.created_at.desc())
            )
            key = result.first()
            if not key:
                return None
            if key.daily_limit is not None and key.usage_today >= key.daily_limit:
                raise ApiKeyLimitExceeded("Daily limit reached for API key")
            try:
                decrypted = decrypt_value(key.api_key_encrypted, self._encryption_secret)
            except EncryptionError:
                return None
            return key, decrypted

    async def increment_api_usage(self, key_id: uuid.UUID, *, amount: int = 1) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                key = await session.get(UserApiKey, key_id)
                if not key:
                    raise UserNotFoundError(str(key_id))
                await self._reset_usage_if_needed(session, key.user_id, key.provider)
                key.usage_today += amount

    async def reset_all_api_usage(self) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                now = datetime.now(timezone.utc)
                await session.execute(
                    UserApiKey.__table__.update().values(
                        usage_today=0,
                        last_reset_at=now,
                    )
                )

    async def _reset_usage_if_needed(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        provider: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        result = await session.scalars(
            select(UserApiKey)
            .where(
                UserApiKey.user_id == user_id,
                UserApiKey.provider == provider,
            )
            .with_for_update()
        )
        keys = list(result)
        for key in keys:
            if not key.last_reset_at or key.last_reset_at.date() != now.date():
                key.usage_today = 0
                key.last_reset_at = now

    async def create_session(
        self,
        user_id: uuid.UUID,
        *,
        refresh_token_hash: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> UserSession:
        async with self._session_factory() as session:
            async with session.begin():
                session_obj = UserSession(
                    user_id=user_id,
                    refresh_token_hash=refresh_token_hash,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    expires_at=expires_at,
                    last_used_at=datetime.now(timezone.utc),
                    is_active=True,
                )
                session.add(session_obj)
                await session.flush()
                await session.refresh(session_obj)
                return session_obj

    async def update_session(
        self,
        session_id: uuid.UUID,
        *,
        refresh_token_hash: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[UserSession]:
        async with self._session_factory() as session:
            async with session.begin():
                session_obj = await session.get(UserSession, session_id)
                if not session_obj:
                    return None
                session_obj.refresh_token_hash = refresh_token_hash
                session_obj.expires_at = expires_at
                session_obj.last_used_at = datetime.now(timezone.utc)
                session_obj.is_active = True
                if ip_address is not None:
                    session_obj.ip_address = ip_address
                if user_agent is not None:
                    session_obj.user_agent = user_agent
                await session.flush()
                await session.refresh(session_obj)
                return session_obj

    async def get_session_by_id(self, session_id: uuid.UUID) -> Optional[UserSession]:
        async with self._session_factory() as session:
            return await session.get(UserSession, session_id)

    async def list_sessions_for_user(self, user_id: uuid.UUID) -> list[UserSession]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(UserSession)
                .where(UserSession.user_id == user_id)
                .order_by(UserSession.created_at.desc())
            )
            return list(result)

    async def count_active_sessions(self, user_id: uuid.UUID) -> int:
        async with self._session_factory() as session:
            result = await session.scalar(
                select(func.count())
                .select_from(UserSession)
                .where(
                    UserSession.user_id == user_id,
                    UserSession.is_active.is_(True),
                )
            )
            return int(result or 0)

    async def revoke_session(self, session_id: uuid.UUID) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                session_obj = await session.get(UserSession, session_id)
                if session_obj:
                    session_obj.is_active = False

    async def list_login_events(
        self,
        user_id: uuid.UUID,
        limit: int = 20,
    ) -> list[UserLoginEvent]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(UserLoginEvent)
                .where(UserLoginEvent.user_id == user_id)
                .order_by(UserLoginEvent.created_at.desc())
                .limit(limit)
            )
            return list(result)

    async def _record_login(self, user_id: uuid.UUID) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                user = await session.get(User, user_id)
                if user:
                    user.last_login_at = datetime.now(timezone.utc)

    async def record_login_event(
        self,
        *,
        user_id: Optional[uuid.UUID],
        username: Optional[str],
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> UserLoginEvent:
        async with self._session_factory() as session:
            async with session.begin():
                event = UserLoginEvent(
                    user_id=user_id,
                    username=username,
                    success=success,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    reason=reason,
                )
                session.add(event)
                await session.flush()
                await session.refresh(event)
                return event
