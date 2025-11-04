import sys
import types
import asyncio

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

import pytest

from brain_service.services.study.bookshelf import BookshelfService, _FALLBACK_CATEGORIES
from brain_service.services.study.config_schema import StudyConfig
from brain_service.services.study.errors import BookshelfUnavailable
from brain_service.services.study.redis_repo import StudyRedisRepository


class StubRedisClient:
    def __init__(self) -> None:
        self.storage: dict[str, str] = {}

    async def set(self, key: str, value: str, *, ex: int | None = None) -> bool:
        self.storage[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.storage.get(key)


class FakeSefariaService:
    def __init__(self) -> None:
        self.link_calls: list[dict[str, object]] = []
        self.text_calls: list[str] = []
        self.responses: list[dict[str, object]] = []
        self.text_payloads: dict[str, dict[str, object]] = {}
        self.raise_on_links = False

    async def get_related_links(self, ref: str, categories, limit: int):
        if self.raise_on_links:
            raise RuntimeError("boom")
        payload = self.responses.pop(0) if self.responses else {"ok": True, "data": []}
        self.link_calls.append({
            "ref": ref,
            "categories": tuple(categories),
            "limit": limit,
        })
        return payload

    async def get_text(self, ref: str):
        self.text_calls.append(ref)
        return self.text_payloads.get(
            ref,
            {"ok": True, "data": {"en_text": "", "he_text": ""}},
        )


def make_config(**overrides) -> StudyConfig:
    base = {
        "preview": {"max_len": 8},
        "bookshelf": {
            "limit_default": 2,
            "top_preview_fetch": 1,
            "default_categories": ["Commentary"],
            "cache_ttl_sec": 0,
        },
    }
    base_bookshelf = base["bookshelf"]
    base_bookshelf.update(overrides.pop("bookshelf", {}))
    base.update(overrides)
    return StudyConfig.model_validate(base)


def test_bookshelf_uses_cache_when_enabled():
    sefaria = FakeSefariaService()
    sefaria.responses = [
        {
            "ok": True,
            "data": [
                {
                    "ref": "Rashi on Genesis 1:1",
                    "commentator": "Rashi",
                    "category": "Commentary",
                    "indexTitle": "Genesis",
                    "title": "Rashi",
                },
            ],
        }
    ]
    sefaria.text_payloads["Rashi on Genesis 1:1"] = {
        "ok": True,
        "data": {"en_text": "Preview text", "he_text": ""},
    }

    repo = StudyRedisRepository(StubRedisClient())
    config = make_config(bookshelf={"cache_ttl_sec": 60})
    service = BookshelfService(sefaria, object(), repo, config)

    first = asyncio.run(service.get_for("Genesis 1:1"))
    assert first.items
    assert len(sefaria.link_calls) == 1

    sefaria.raise_on_links = True
    cached = asyncio.run(service.get_for("Genesis 1:1"))
    assert cached == first
    assert len(sefaria.link_calls) == 1


def test_bookshelf_falls_back_to_parent_ref():
    sefaria = FakeSefariaService()
    sefaria.responses = [
        {"ok": False, "data": []},
        {
            "ok": True,
            "data": [
                {
                    "ref": "Rashi on Genesis 1",
                    "commentator": "Rashi",
                    "category": "Commentary",
                    "indexTitle": "Genesis",
                    "title": "Rashi",
                }
            ],
        },
    ]
    sefaria.text_payloads["Rashi on Genesis 1"] = {
        "ok": True,
        "data": {"en_text": "", "he_text": ""},
    }

    service = BookshelfService(sefaria, object(), None, make_config())

    bookshelf = asyncio.run(service.get_for("Genesis 1:1"))

    assert len(bookshelf.items) == 1
    assert len(sefaria.link_calls) == 2
    assert sefaria.link_calls[0]["ref"] == "Genesis 1:1"
    assert sefaria.link_calls[1]["ref"] == "Genesis 1"


def test_bookshelf_unavailable_on_exception():
    sefaria = FakeSefariaService()
    sefaria.raise_on_links = True
    service = BookshelfService(sefaria, object(), None, make_config())

    with pytest.raises(BookshelfUnavailable):
        asyncio.run(service.get_for("Genesis 1:1"))


def test_bookshelf_default_categories_used_when_not_configured():
    sefaria = FakeSefariaService()
    sefaria.responses = [{"ok": True, "data": []}]
    config = make_config(bookshelf={"default_categories": []})
    service = BookshelfService(sefaria, object(), None, config)

    asyncio.run(service.get_for("Exodus 1:1"))

    assert sefaria.link_calls[0]["categories"] == tuple(_FALLBACK_CATEGORIES)
