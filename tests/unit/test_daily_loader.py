import json
import sys
import types
from typing import Any

import pytest

# Stub out missing optional dependencies before importing the loader module.
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

from brain_service.services.study.daily_loader import DailyLoader
from brain_service.services.study.config_schema import StudyConfig


class StubRepo:
    def __init__(self) -> None:
        self.segments: list[str] = []
        self.segment_ttls: list[int] = []
        self.total_calls: list[tuple[str, int, int]] = []
        self.cleared_sessions: list[str] = []
        self.try_lock_result = True
        self.loading_sessions: list[tuple[str, int]] = []
        self.clear_loading_sessions: list[str] = []
        self.released_sessions: list[str] = []
        self.marked_tasks: set[tuple[str, str]] = set()
        self.loading_flag = False

    async def clear_segments(self, session_id: str) -> None:
        self.cleared_sessions.append(session_id)
        self.segments.clear()
        self.segment_ttls.clear()

    async def push_segment(self, session_id: str, payload: str, ttl_seconds: int) -> None:
        self.segments.append(payload)
        self.segment_ttls.append(ttl_seconds)

    async def set_total(self, session_id: str, total: int, ttl_seconds: int) -> None:
        self.total_calls.append((session_id, total, ttl_seconds))

    async def try_lock(self, session_id: str, ttl_seconds: int) -> bool:
        return self.try_lock_result

    async def release_lock(self, session_id: str) -> None:
        self.released_sessions.append(session_id)

    async def set_loading(self, session_id: str, ttl_seconds: int) -> None:
        self.loading_flag = True
        self.loading_sessions.append((session_id, ttl_seconds))

    async def clear_loading(self, session_id: str) -> None:
        self.loading_flag = False
        self.clear_loading_sessions.append(session_id)

    async def is_loading(self, session_id: str) -> bool:
        return self.loading_flag

    async def mark_task(self, session_id: str, task_id: str, ttl_seconds: int) -> None:
        self.marked_tasks.add((session_id, task_id))

    async def is_task_marked(self, session_id: str, task_id: str) -> bool:
        return (session_id, task_id) in self.marked_tasks


class StubSefaria:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_text(self, ref: str) -> dict[str, Any]:
        self.calls.append(ref)
        verse_no = ref.rsplit(":", 1)[-1]
        return {
            "ok": True,
            "data": {
                "text": f"en-{verse_no}",
                "he": [f"he-{verse_no}"],
                "title": "Genesis",
                "indexTitle": "Bereshit",
                "heRef": f"בראשית {verse_no}",
            },
        }


@pytest.mark.anyio
async def test_load_initial_persists_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = StubRepo()
    loader = DailyLoader(StubSefaria(), object(), repo, StudyConfig())

    sample_segments = [
        {
            "ref": f"Genesis 1:{idx + 1}",
            "text": f"En {idx + 1}",
            "heText": f"He {idx + 1}",
            "position": idx / 9,
            "metadata": {
                "title": "Genesis",
                "indexTitle": "Bereshit",
                "heRef": f"בראשית א:{idx + 1}",
            },
        }
        for idx in range(10)
    ]

    async def fake_build_full_daily_text(
        ref: str,
        sefaria: Any,
        index: Any,
        session_id: str,
        redis_client: Any,
    ) -> dict[str, Any]:
        return {
            "segments": sample_segments,
            "focusIndex": 2,
            "ref": ref,
        }

    monkeypatch.setattr(
        "brain_service.services.study.daily_loader.build_full_daily_text",
        fake_build_full_daily_text,
    )

    result = await loader.load_initial(ref="Genesis 1:1-10", session_id="session-1")

    assert repo.cleared_sessions == ["session-1"]
    assert len(repo.segments) == 10
    assert all(ttl == 604800 for ttl in repo.segment_ttls)
    assert repo.total_calls[-1] == ("session-1", 10, 604800)

    stored_payloads = [json.loads(item) for item in repo.segments]
    assert stored_payloads[0]["ref"] == "Genesis 1:1"
    assert stored_payloads[0]["he_text"] == "He 1"

    assert result["total_segments"] == 10
    assert result["loaded"] == 10
    assert result["remaining_plan"] == []
    assert len(result["segments"]) == 10


@pytest.mark.anyio
async def test_load_initial_produces_remaining_plan_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = StubRepo()
    loader = DailyLoader(StubSefaria(), object(), repo, StudyConfig())

    sample_segments = [
        {
            "ref": f"Genesis 1:{idx + 1}",
            "text": f"En {idx + 1}",
            "heText": f"He {idx + 1}",
            "position": idx / 39,
            "metadata": {
                "title": "Genesis",
                "indexTitle": "Bereshit",
                "heRef": f"בראשית א:{idx + 1}",
            },
        }
        for idx in range(30)
    ]

    async def fake_build_full_daily_text(
        ref: str,
        sefaria: Any,
        index: Any,
        session_id: str,
        redis_client: Any,
    ) -> dict[str, Any]:
        return {
            "segments": sample_segments,
            "focusIndex": 0,
            "ref": ref,
        }

    monkeypatch.setattr(
        "brain_service.services.study.daily_loader.build_full_daily_text",
        fake_build_full_daily_text,
    )

    result = await loader.load_initial(ref="Genesis 1:1-30", session_id="session-plan")

    remaining = result["remaining_plan"]
    assert remaining, "expected remaining plan entries"

    chunk = remaining[0]
    assert chunk["start"] == 20
    assert chunk["end"] == 30
    assert chunk["book_chapter"] == "Genesis 1"
    assert chunk["range_start_verse"] == 1
    assert chunk["chunk_start_verse"] == 21
    assert chunk["end_verse"] == 30


@pytest.mark.anyio
async def test_load_initial_handles_missing_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = StubRepo()
    loader = DailyLoader(StubSefaria(), object(), repo, StudyConfig())

    async def fake_build_full_daily_text(
        ref: str,
        sefaria: Any,
        index: Any,
        session_id: str,
        redis_client: Any,
    ) -> None:
        return None

    monkeypatch.setattr(
        "brain_service.services.study.daily_loader.build_full_daily_text",
        fake_build_full_daily_text,
    )

    result = await loader.load_initial(ref="Genesis 1:1", session_id="session-2")

    assert repo.cleared_sessions == ["session-2"]
    assert repo.segments == []
    assert repo.total_calls[-1] == ("session-2", 0, 604800)
    assert result["segments"] == []
    assert result["remaining_plan"] == []


@pytest.mark.anyio
async def test_load_background_respects_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = StubRepo()
    repo.marked_tasks.add(("session-3", "Genesis 1:1-20:1:20"))
    loader = DailyLoader(StubSefaria(), object(), repo, StudyConfig())

    await loader.load_background(
        ref="Genesis 1:1-20",
        session_id="session-3",
        start_verse=1,
        end_verse=20,
        book_chapter="Genesis 1",
        already_loaded=5,
        ttl_seconds=3600,
    )

    assert repo.segments == []
    assert repo.released_sessions == []


@pytest.mark.anyio
async def test_load_background_streams_remaining_segments() -> None:
    repo = StubRepo()
    loader = DailyLoader(StubSefaria(), object(), repo, StudyConfig())
    loader._sleep_between_requests = 0

    await loader.load_background(
        ref="Genesis 1:1-5",
        session_id="session-4",
        start_verse=1,
        end_verse=5,
        book_chapter="Genesis 1",
        already_loaded=2,
        ttl_seconds=3600,
    )

    assert len(repo.segments) == 3
    payloads = [json.loads(item) for item in repo.segments]
    assert [payload["ref"] for payload in payloads] == ["Genesis 1:3", "Genesis 1:4", "Genesis 1:5"]
    assert repo.total_calls[-1][1] == 5
    assert repo.released_sessions == ["session-4"]
    assert repo.clear_loading_sessions == ["session-4"]


@pytest.mark.anyio
async def test_load_background_respects_lock_conflict() -> None:
    repo = StubRepo()
    repo.try_lock_result = False
    loader = DailyLoader(StubSefaria(), object(), repo, StudyConfig())

    await loader.load_background(
        ref="Genesis 1:1-5",
        session_id="session-5",
        start_verse=1,
        end_verse=5,
        book_chapter="Genesis 1",
        already_loaded=0,
        ttl_seconds=3600,
    )

    assert repo.segments == []
    assert repo.loading_sessions
    assert repo.released_sessions == []
