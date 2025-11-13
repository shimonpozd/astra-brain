from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt_sha256", "bcrypt"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password, scheme="bcrypt_sha256")


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def hash_token(value: str) -> str:
    return pwd_context.hash(value, scheme="bcrypt_sha256")


def verify_token(value: str, token_hash: str) -> bool:
    return pwd_context.verify(value, token_hash)


def create_access_token(
    data: Dict[str, Any],
    secret: str,
    algorithm: str,
    expires_minutes: int,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret, algorithm=algorithm)


def decode_access_token(token: str, secret: str, algorithm: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError as exc:
        raise ValueError("Invalid token") from exc
