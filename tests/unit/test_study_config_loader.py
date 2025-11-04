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

from brain_service.services.study.config_loader import (
    fetch_study_config,
    register_study_config_listener,
)
from brain_service.services.study.config_schema import StudyConfig
from brain_service.services.study.errors import StudyConfigInvalid


class FakeConfigService:
    def __init__(self, payload):
        self._payload = payload
        self.registered_section = None
        self.callback = None

    async def get_config_section(self, section: str, default=None):
        assert section == "study"
        return self._payload if self._payload is not None else default

    async def register_listener(self, section: str, callback):
        self.registered_section = section
        self.callback = callback


def test_fetch_study_config_defaults():
    service = FakeConfigService(payload=None)

    config = asyncio.run(fetch_study_config(service))

    assert isinstance(config, StudyConfig)
    assert config.window.size_default == 5
    assert config.daily.max_total_segments == 500
    assert config.daily.modular_loader_enabled is False
    assert config.bookshelf.cache_ttl_sec == 0
    assert config.chat_history.max_messages == 2000
    assert config.features.facade_enabled is False


def test_fetch_study_config_invalid_raises():
    service = FakeConfigService(payload={"window": {"size_min": 10, "size_default": 5, "size_max": 15}})

    with pytest.raises(StudyConfigInvalid) as exc_info:
        asyncio.run(fetch_study_config(service))

    assert exc_info.value.detail["type"] == "validation_error"
    assert exc_info.value.detail["errors"]


def test_register_study_config_listener_validates_payload():
    async def _scenario():
        service = FakeConfigService(payload={})
        received = {}

        async def on_update(config: StudyConfig):
            received["size_default"] = config.window.size_default

        await register_study_config_listener(service, on_update)

        assert service.registered_section == "study"

        await service.callback({"window": {"size_default": 7, "size_min": 1, "size_max": 10}})
        assert received["size_default"] == 7

        with pytest.raises(StudyConfigInvalid):
            await service.callback({"window": {"size_default": 5, "size_min": 9, "size_max": 4}})

    asyncio.run(_scenario())


def test_study_service_applies_config_updates():
    try:
        from brain_service.services.study_service import StudyService
    except ModuleNotFoundError:
        pytest.skip("StudyService dependencies unavailable for unit test")
    except ModuleNotFoundError:
        pytest.skip("StudyService dependencies unavailable for unit test")

    initial_config = load_study_config({"chat_history": {"max_messages": 10, "ttl_days": 1}})
    service = StudyService(
        redis_client=None,
        sefaria_service=None,
        sefaria_index_service=None,
        tool_registry=None,
        memory_service=None,
        study_config=initial_config,
    )

    assert service.max_chat_history_messages == 10
    assert service.chat_history_ttl_days == 1

    updated_config = load_study_config({"chat_history": {"max_messages": 25, "ttl_days": 3}})
    service.update_study_config(updated_config)

    assert service.max_chat_history_messages == 25
    assert service.chat_history_ttl_days == 3
    assert service.study_config is updated_config



