from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from brain_service.core.security import create_access_token, decode_access_token
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
    ):
        self._user_service = user_service
        self._jwt_secret = jwt_secret
        self._jwt_algorithm = jwt_algorithm
        self._jwt_expires_minutes = jwt_expires_minutes

    async def authenticate_user(
        self, username: str, password: str
    ) -> Optional[User]:
        return await self._user_service.authenticate(username, password)

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
