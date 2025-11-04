import asyncio
from typing import Any

import pytest

import sys
import types

if "core" not in sys.modules:
    core_module = types.ModuleType("core")
    utils_module = types.ModuleType("core.utils")

    class _CompactText:
        def __init__(self, payload):
            self._payload = payload or {}

        def to_dict_min(self):
            return self._payload if isinstance(self._payload, dict) else {}

    async def _identity_async(value):
        return value

    async def _with_retries(callable_):
        result = callable_()
        if hasattr(result, "__await__"):
            return await result
        return result

    async def _get_from_sefaria(*_args, **_kwargs):
        return {}

    def _compact_links(_links, categories=None, limit=0):
        return []

    utils_module.CompactText = _CompactText
    utils_module.ok_and_has_text = lambda payload: bool(payload)
    utils_module.normalize_tref = _identity_async
    utils_module.with_retries = _with_retries
    utils_module.get_from_sefaria = _get_from_sefaria
    utils_module.compact_and_deduplicate_links = _compact_links

    core_module.utils = utils_module
    sys.modules["core"] = core_module
    sys.modules["core.utils"] = utils_module

from brain_service.services.study.redis_repo import RedisKeys, StudyRedisRepository


class FakeRedis:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._ttls: dict[str, int] = {}

    def pipeline(self):
        return _FakePipeline(self)

    async def rpush(self, key: str, value: str) -> int:
        values = self._data.setdefault(key, [])
        if not isinstance(values, list):
            raise TypeError("Key holds non-list value")
        values.append(value)
        return len(values)

    async def expire(self, key: str, ttl: int) -> bool:
        if key in self._data:
            self._ttls[key] = ttl
            return True
        return False

    async def set(self, key: str, value: Any, *, ex: int | None = None, nx: bool = False) -> bool:
        if nx and key in self._data:
            return False
        self._data[key] = value
        if ex is not None:
            self._ttls[key] = ex
        elif key in self._ttls:
            self._ttls.pop(key)
        return True

    async def delete(self, key: str) -> int:
        removed = int(key in self._data)
        self._data.pop(key, None)
        self._ttls.pop(key, None)
        return removed

    async def exists(self, key: str) -> int:
        return int(key in self._data)

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        values = self._data.get(key, [])
        if not isinstance(values, list):
            return []
        normalised_end = len(values) if end == -1 else end + 1
        sliced = values[start:normalised_end]
        return [self._encode(item) for item in sliced]

    async def get(self, key: str) -> bytes | None:
        if key not in self._data:
            return None
        value = self._data[key]
        if isinstance(value, list):
            raise TypeError("Cannot GET list value")
        return self._encode(value)

    def ttl_for(self, key: str) -> int | None:
        return self._ttls.get(key)

    @staticmethod
    def _encode(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")


class _FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def rpush(self, key: str, value: str):
        self._commands.append(("rpush", (key, value), {}))
        return self

    def expire(self, key: str, ttl: int):
        self._commands.append(("expire", (key, ttl), {}))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for name, args, kwargs in self._commands:
            method = getattr(self._redis, name)
            if asyncio.iscoroutinefunction(method):
                results.append(await method(*args, **kwargs))
            else:
                results.append(method(*args, **kwargs))
        self._commands.clear()
        return results


@pytest.mark.anyio("asyncio")
async def test_push_segment_appends_and_sets_ttl() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.push_segment("sess", "{\"foo\": 1}", ttl_seconds=120)

    key = RedisKeys().daily_segments("sess")
    assert fake._data[key] == ['{"foo": 1}']
    assert fake.ttl_for(key) == 120


@pytest.mark.anyio("asyncio")
async def test_set_total_persists_count_and_ttl() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.set_total("sess", total=42, ttl_seconds=86400)

    key = RedisKeys().daily_total("sess")
    assert fake._data[key] == 42
    assert fake.ttl_for(key) == 86400


@pytest.mark.anyio("asyncio")
async def test_try_lock_respects_nx_and_release() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    assert await repo.try_lock("sess", ttl_seconds=30) is True
    assert await repo.try_lock("sess", ttl_seconds=30) is False
    await repo.release_lock("sess")
    assert await repo.try_lock("sess", ttl_seconds=0) is True  # ttl optional


@pytest.mark.anyio("asyncio")
async def test_mark_task_and_check() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.mark_task("sess", "task", ttl_seconds=15)
    assert await repo.is_task_marked("sess", "task") is True
    assert fake.ttl_for(RedisKeys().daily_task("sess", "task")) == 15


@pytest.mark.anyio("asyncio")
async def test_fetch_segments_returns_utf8_strings() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    # prime list
    await repo.push_segment("sess", "one", ttl_seconds=60)
    await repo.push_segment("sess", "two", ttl_seconds=60)

    fetched = await repo.fetch_segments("sess", 0, -1)
    assert fetched == ["one", "two"]


@pytest.mark.anyio("asyncio")
async def test_clear_segments_removes_existing_data() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.push_segment("sess", "one", ttl_seconds=60)
    key = RedisKeys().daily_segments("sess")
    assert key in fake._data

    await repo.clear_segments("sess")
    assert key not in fake._data
    assert fake.ttl_for(key) is None

@pytest.mark.anyio("asyncio")
async def test_top_ref_round_trip() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.set_top_ref("sess", "{\"ref\": \"Genesis 1\"}", ttl_seconds=10)
    stored = await repo.get_top_ref("sess")
    assert stored == '{"ref": "Genesis 1"}'


@pytest.mark.anyio("asyncio")
async def test_window_cache_round_trip() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.cache_window("Genesis 1:1", "payload", ttl_seconds=-1)
    cached = await repo.fetch_window("Genesis 1:1")
    assert cached == "payload"


@pytest.mark.anyio("asyncio")
async def test_loading_flag_round_trip() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.set_loading("sess", ttl_seconds=5)
    assert await repo.is_loading("sess") is True
    assert fake.ttl_for(RedisKeys().daily_loading("sess")) == 5

    await repo.clear_loading("sess")
    assert await repo.is_loading("sess") is False


@pytest.mark.anyio("asyncio")
async def test_bookshelf_cache_round_trip() -> None:
    fake = FakeRedis()
    repo = StudyRedisRepository(fake)

    await repo.cache_bookshelf("hash", "{\"foo\": 1}", ttl_seconds=30)
    stored = await repo.fetch_bookshelf("hash")
    assert stored == '{"foo": 1}'
    assert fake.ttl_for(RedisKeys().bookshelf("hash")) == 30
