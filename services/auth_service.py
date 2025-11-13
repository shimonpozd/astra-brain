from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from brain_service.core.security import (
    create_access_token,
    decode_access_token,
    hash_token,
    verify_token,
)
from brain_service.models.db import User
from brain_service.services.user_service import UserService


class AuthService:
    def __init__(
        self,
        user_service: UserService,
        *,
        jwt_secret: str,
        jwt_algorithm: str,
        jwt_expires_minutes: int,
        refresh_token_expires_days: int,
    ):
        self._user_service = user_service
        self._jwt_secret = jwt_secret
        self._jwt_algorithm = jwt_algorithm
        self._jwt_expires_minutes = jwt_expires_minutes
        self._refresh_token_ttl = timedelta(days=refresh_token_expires_days)

    async def authenticate_user(
        self,
        identifier: str,
        password: str,
        *,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[User]:
        return await self._user_service.authenticate(
            identifier,
            password,
            ip_address=ip_address,
            user_agent=user_agent,
        )

    def issue_token(self, user: User) -> str:
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "role": user.role,
        }
        return create_access_token(
            payload,
            secret=self._jwt_secret,
            algorithm=self._jwt_algorithm,
            expires_minutes=self._jwt_expires_minutes,
        )

    async def start_session(
        self,
        user: User,
        *,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[str, str]:
        token_secret = uuid.uuid4().hex
        refresh_hash = hash_token(token_secret)
        expires_at = datetime.now(timezone.utc) + self._refresh_token_ttl
        session = await self._user_service.create_session(
            user.id,
            refresh_token_hash=refresh_hash,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        access_token = self.issue_token(user)
        refresh_token = f"{session.id}:{token_secret}"
        return access_token, refresh_token

    async def refresh_session(
        self,
        refresh_token: str,
        *,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[str, str, User]:
        session_id, secret = self._parse_refresh_token(refresh_token)
        session = await self._user_service.get_session_by_id(session_id)
        if not session or not session.is_active or session.expires_at < datetime.now(timezone.utc):
            raise ValueError("Invalid refresh token")
        if not verify_token(secret, session.refresh_token_hash):
            await self._user_service.revoke_session(session.id)
            raise ValueError("Invalid refresh token")
        user = await self._user_service.get_user_by_id(session.user_id)
        if not user or not user.is_active:
            await self._user_service.revoke_session(session.id)
            raise ValueError("User no longer active")
        new_secret = uuid.uuid4().hex
        new_hash = hash_token(new_secret)
        expires_at = datetime.now(timezone.utc) + self._refresh_token_ttl
        await self._user_service.update_session(
            session.id,
            refresh_token_hash=new_hash,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        access_token = self.issue_token(user)
        return access_token, f"{session.id}:{new_secret}", user

    async def revoke_session(self, session_id: uuid.UUID) -> None:
        await self._user_service.revoke_session(session_id)

    async def revoke_refresh_token(self, refresh_token: str) -> None:
        session_id, _ = self._parse_refresh_token(refresh_token)
        await self._user_service.revoke_session(session_id)

    async def resolve_user(self, token: str) -> Optional[User]:
        data = decode_access_token(token, self._jwt_secret, self._jwt_algorithm)
        user_id = data.get("sub")
        if not user_id:
            return None
        try:
            uid = uuid.UUID(user_id)
        except ValueError:
            return None
        return await self._user_service.get_user_by_id(uid)

    def _parse_refresh_token(self, refresh_token: str) -> tuple[uuid.UUID, str]:
        if ":" not in refresh_token:
            raise ValueError("Invalid refresh token")
        session_part, secret = refresh_token.split(":", 1)
        return uuid.UUID(session_part), secret
