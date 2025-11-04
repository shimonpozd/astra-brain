from __future__ import annotations

import base64
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""


def _build_fernet(secret: str) -> Fernet:
    """
    Create a Fernet helper from raw secret.
    Accepts either a base64 key or a raw string which will be padded to 32 bytes.
    """
    if not secret:
        raise EncryptionError("API key secret is not configured")

    try:
        # Accept base64-encoded secret
        secret_bytes = base64.urlsafe_b64decode(secret)
        if len(secret_bytes) != 32:
            raise ValueError
        return Fernet(base64.urlsafe_b64encode(secret_bytes))
    except Exception:
        # Fallback: derive from raw text by padding/truncating
        raw = secret.encode("utf-8")
        if len(raw) < 32:
            raw = raw.ljust(32, b"\0")
        elif len(raw) > 32:
            raw = raw[:32]
        return Fernet(base64.urlsafe_b64encode(raw))


def encrypt_value(value: str, secret: str) -> str:
    try:
        f = _build_fernet(secret)
        token = f.encrypt(value.encode("utf-8"))
        return token.decode("utf-8")
    except Exception as exc:
        raise EncryptionError("Failed to encrypt value") from exc


def decrypt_value(token: str, secret: str) -> str:
    try:
        f = _build_fernet(secret)
        decrypted = f.decrypt(token.encode("utf-8"))
        return decrypted.decode("utf-8")
    except InvalidToken as exc:
        raise EncryptionError("Invalid encryption token") from exc
    except Exception as exc:
        raise EncryptionError("Failed to decrypt value") from exc
