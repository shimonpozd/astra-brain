"""Redis repository for study domain state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(slots=True)
class RedisKeys:
    """Namespace helpers for study-related Redis keys."""

    window_prefix: str = "study:window"
    daily_session_prefix: str = "daily:sess"
    daily_top_prefix: str = "daily:top"
    bookshelf_prefix: str = "study:bookshelf"

    def daily_loading(self, session_id: str) -> str:
        return f"{self.daily_session_prefix}:{session_id}:loading"

    def window(self, ref: str) -> str:
        return f"{self.window_prefix}:{ref}"

    def daily_segments(self, session_id: str) -> str:
        return f"{self.daily_session_prefix}:{session_id}:segments"

    def daily_total(self, session_id: str) -> str:
        return f"{self.daily_session_prefix}:{session_id}:total"

    def daily_total_segments(self, session_id: str) -> str:
        return f"{self.daily_session_prefix}:{session_id}:total_segments"

    def daily_lock(self, session_id: str) -> str:
        return f"{self.daily_session_prefix}:{session_id}:lock"

    def daily_task(self, session_id: str, task_id: str) -> str:
        return f"{self.daily_session_prefix}:{session_id}:task:{task_id}"

    def daily_top(self, session_id: str) -> str:
        return f"{self.daily_top_prefix}:{session_id}"

    def bookshelf(self, cache_key: str) -> str:
        return f"{self.bookshelf_prefix}:{cache_key}"


class StudyRedisRepository:
    """Wrapper around Redis operations for study state."""

    def __init__(self, redis_client: Any, *, keys: Optional[RedisKeys] = None) -> None:
        self._redis = redis_client
        self._keys = keys or RedisKeys()

    @staticmethod
    def _ensure_positive_ttl(ttl_seconds: int | None) -> Optional[int]:
        if ttl_seconds is None:
            return None
        if ttl_seconds <= 0:
            return None
        return int(ttl_seconds)

    @staticmethod
    def _decode(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    async def clear_segments(self, session_id: str) -> None:
        """Remove stored segments for a session."""

        key = self._keys.daily_segments(session_id)
        await self._redis.delete(key)

    async def push_segment(self, session_id: str, segment_json: str, ttl_seconds: int) -> None:
        """Append a segment to the session list and refresh TTL."""

        key = self._keys.daily_segments(session_id)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        pipe = self._redis.pipeline()
        pipe.rpush(key, segment_json)
        if ttl:
            pipe.expire(key, ttl)
        await pipe.execute()

    async def set_total(self, session_id: str, total: int, ttl_seconds: int) -> None:
        """Persist the total segment count for a session."""

        key = self._keys.daily_total(session_id)
        key_alt = self._keys.daily_total_segments(session_id)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        try:
            incoming = int(total)
        except (TypeError, ValueError):
            incoming = 0

        existing = await self._redis.get(key)
        try:
            current = int(existing) if existing is not None else 0
        except (TypeError, ValueError):
            current = 0

        value = max(incoming, current)

        if ttl:
            await self._redis.set(key, value, ex=ttl)
            await self._redis.set(key_alt, value, ex=ttl)
        else:
            await self._redis.set(key, value)
            await self._redis.set(key_alt, value)

    async def set_loading(self, session_id: str, ttl_seconds: int) -> None:
        """Set the background-loading flag for a session."""

        key = self._keys.daily_loading(session_id)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        if ttl:
            await self._redis.set(key, "1", ex=ttl)
        else:
            await self._redis.set(key, "1")

    async def clear_loading(self, session_id: str) -> None:
        """Clear the background-loading flag."""

        key = self._keys.daily_loading(session_id)
        await self._redis.delete(key)

    async def is_loading(self, session_id: str) -> bool:
        """Return True when a background load is currently marked as running."""

        key = self._keys.daily_loading(session_id)
        return bool(await self._redis.exists(key))

    async def try_lock(self, session_id: str, ttl_seconds: int) -> bool:
        """Acquire a session-specific lock using SET NX."""

        key = self._keys.daily_lock(session_id)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        kwargs = {"nx": True}
        if ttl:
            kwargs["ex"] = ttl
        result = await self._redis.set(key, "1", **kwargs)
        return bool(result)

    async def release_lock(self, session_id: str) -> None:
        """Release the session lock if held."""

        key = self._keys.daily_lock(session_id)
        await self._redis.delete(key)

    async def mark_task(self, session_id: str, task_id: str, ttl_seconds: int) -> None:
        """Mark a background task idempotency key."""

        key = self._keys.daily_task(session_id, task_id)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        if ttl:
            await self._redis.set(key, "1", ex=ttl)
        else:
            await self._redis.set(key, "1")

    async def is_task_marked(self, session_id: str, task_id: str) -> bool:
        """Return True if the background task has already been scheduled/executed."""

        key = self._keys.daily_task(session_id, task_id)
        return bool(await self._redis.exists(key))

    async def fetch_segments(self, session_id: str, start: int, end: int) -> Iterable[str]:
        """Fetch a slice of segments for a session."""

        key = self._keys.daily_segments(session_id)
        items = await self._redis.lrange(key, start, end)
        return [self._decode(item) or "" for item in items]

    async def set_top_ref(self, session_id: str, payload_json: str, ttl_seconds: int) -> None:
        """Persist the top-level ref metadata for a session."""

        key = self._keys.daily_top(session_id)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        if ttl:
            await self._redis.set(key, payload_json, ex=ttl)
        else:
            await self._redis.set(key, payload_json)

    async def get_top_ref(self, session_id: str) -> Optional[str]:
        """Retrieve the stored top-level ref metadata if present."""

        key = self._keys.daily_top(session_id)
        value = await self._redis.get(key)
        return self._decode(value)

    async def cache_window(self, ref: str, payload_json: str, ttl_seconds: int) -> None:
        """Cache the study window payload for quick reuse."""

        key = self._keys.window(ref)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        if ttl:
            await self._redis.set(key, payload_json, ex=ttl)
        else:
            await self._redis.set(key, payload_json)

    async def fetch_window(self, ref: str) -> Optional[str]:
        """Fetch a previously cached window payload."""

        key = self._keys.window(ref)
        value = await self._redis.get(key)
        return self._decode(value)

    async def cache_bookshelf(self, cache_key: str, payload_json: str, ttl_seconds: int) -> None:
        """Cache a bookshelf payload keyed by hashed parameters."""

        key = self._keys.bookshelf(cache_key)
        ttl = self._ensure_positive_ttl(ttl_seconds)
        if ttl:
            await self._redis.set(key, payload_json, ex=ttl)
        else:
            await self._redis.set(key, payload_json)

    async def fetch_bookshelf(self, cache_key: str) -> Optional[str]:
        """Fetch a cached bookshelf payload if present."""

        key = self._keys.bookshelf(cache_key)
        value = await self._redis.get(key)
        return self._decode(value)
