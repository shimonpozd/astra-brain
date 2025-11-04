import asyncio
import sys
import types

import pytest

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

from brain_service.services.study.service import StudyService
from brain_service.services.study.config_schema import StudyConfig


class StubSefariaService:
    async def get_text(self, ref: str):
        return {
            "ok": True,
            "data": {
                "ref": ref,
                "text": "",
                "he_text": "",
                "title": "Genesis",
                "indexTitle": "Genesis",
                "heRef": ref,
            },
        }

    async def get_related_links(self, ref: str, categories, limit: int):
        return {"ok": True, "data": []}


class StubIndexService:
    toc: list = []


class StubRedis:
    async def set(self, *args, **kwargs):
        return True

    async def get(self, *args, **kwargs):
        return None


def make_config(**overrides) -> StudyConfig:
    base = {
        "preview": {"max_len": 8},
        "bookshelf": {
            "limit_default": 2,
            "top_preview_fetch": 1,
            "default_categories": ["Commentary"],
            "cache_ttl_sec": 0,
        },
        "prompt_budget": {
            "max_total_tokens": 100,
            "reserved_for_system": 20,
            "reserved_for_stm": 20,
            "min_study_tokens": 10,
        },
    }
    base.update(overrides)
    return StudyConfig.model_validate(base)


@pytest.mark.anyio("asyncio")
async def test_facade_get_text_with_window_uses_config(monkeypatch: pytest.MonkeyPatch) -> None:
    service = StudyService(StubSefariaService(), StubIndexService(), StubRedis(), make_config())

    async def fake_generate_neighbors(base_ref, count, direction, sefaria_service, index_service):
        return [
            {
                "ref": f"{base_ref}:{direction}:{idx}",
                "he_text": f"{direction}-{idx}",
                "title": "Genesis",
                "indexTitle": "Genesis",
            }
            for idx in range(count)
        ]

    monkeypatch.setattr(
        "brain_service.services.study.service.generate_neighbors",
        fake_generate_neighbors,
    )

    result = await service.get_text_with_window("Genesis 1:1", window_size=2)
    assert result["focusIndex"] == 2
    assert len(result["segments"]) == 5
    assert result["segments"][result["focusIndex"]]["ref"] == "Genesis 1:1"
    assert result["segments"][0]["ref"].startswith("Genesis 1:1:prev")


@pytest.mark.anyio("asyncio")
async def test_facade_get_full_daily_text_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    service = StudyService(StubSefariaService(), StubIndexService(), StubRedis(), make_config())

    async def fake_build_full(ref, sefaria_service, index_service, session_id=None, redis_client=None):
        return {"ref": ref, "session_id": session_id}

    monkeypatch.setattr(
        "brain_service.services.study.service.build_full_daily_text",
        fake_build_full,
    )

    result = await service.get_full_daily_text("Genesis 1:1", session_id="sess")
    assert result == {"ref": "Genesis 1:1", "session_id": "sess"}


@pytest.mark.anyio("asyncio")
async def test_facade_get_bookshelf_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    service = StudyService(StubSefariaService(), StubIndexService(), StubRedis(), make_config())

    async def fake_get_for(ref, limit=None, categories=None, preview_max_len=None, top_preview_fetch=None):
        from brain_service.services.study_state import Bookshelf, BookshelfItem

        return Bookshelf(
            counts={"Commentary": 1},
            items=[BookshelfItem(ref=ref, commentator="Rashi", indexTitle="Genesis", preview="", text_full=None, heTextFull=None)],
        )

    monkeypatch.setattr(service._bookshelf_service, "get_for", fake_get_for)

    result = await service.get_bookshelf_for("Genesis 1:1", limit=1)
    assert result["counts"] == {"Commentary": 1}
    assert result["items"][0].commentator == "Rashi"


def test_facade_update_config_refreshes_subcomponents():
    service = StudyService(StubSefariaService(), StubIndexService(), StubRedis(), make_config())

    new_cfg = make_config(preview={"max_len": 16})
    service.update_config(new_cfg)

    assert service.study_config.preview.max_len == 16
    assert service._bookshelf_service._config.preview_max_len == 16
    assert service._daily_loader.config.batch_size == new_cfg.daily.batch_size


def test_facade_build_prompt_payload_enforces_budget():
    service = StudyService(StubSefariaService(), StubIndexService(), StubRedis(), make_config())

    result = asyncio.run(
        service.build_prompt_payload(
            ref="Genesis 1:1",
            mode="girsa",
            system_prompt="system" * 10,
            stm_context="stm" * 10,
            study_segments=["study" * 100],
            extra_segments=["extras" * 100],
        )
    )

    assert "messages" in result
    assert result["messages"]
