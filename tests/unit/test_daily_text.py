import types
import sys

import pytest

from brain_service.services.study.daily_text import build_full_daily_text


class StubSefaria:
    async def get_text(self, ref: str):  # pragma: no cover - not used directly in tests
        raise AssertionError("unexpected direct get_text call: {ref}")


@pytest.mark.anyio
async def test_build_full_daily_text_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_try_load_range(_service, _ref):
        return {
            "ref": "Genesis 1",
            "text": ["En 1", "En 2"],
            "he": ["He 1", "He 2"],
            "title": "Genesis",
            "indexTitle": "Bereshit",
            "heRef": "בראשית א",
        }

    monkeypatch.setattr(
        "brain_service.services.study.daily_text.try_load_range",
        fake_try_load_range,
    )

    payload = await build_full_daily_text("Genesis 1", StubSefaria(), object())
    assert payload
    assert payload["focusIndex"] == 0
    assert len(payload["segments"]) == 2
    first = payload["segments"][0]
    assert first["ref"].endswith(":1")
    assert first["heText"] == "He 1"
    assert first["metadata"]["title"] == "Genesis"


@pytest.mark.anyio
async def test_build_full_daily_text_spanning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_try_load_range(_service, _ref):
        return {
            "ref": "Zevachim 24",
            "text": [["En a1", "En a2"], ["En b1"]],
            "he": [["He a1", "He a2"], ["He b1"]],
            "spanningRefs": ["Zevachim 24a", "Zevachim 24b"],
            "title": "Zevachim",
            "indexTitle": "Zevachim",
            "heRef": "זבחים כד",
            "isSpanning": True,
        }

    monkeypatch.setattr(
        "brain_service.services.study.daily_text.try_load_range",
        fake_try_load_range,
    )

    payload = await build_full_daily_text("Zevachim 24", StubSefaria(), object())
    assert payload
    segments = payload["segments"]
    assert len(segments) == 3
    assert segments[0]["ref"].endswith("24a.1")
    assert segments[-1]["ref"].endswith("24b.1")


@pytest.mark.anyio
async def test_build_full_daily_text_inter_chapter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_try_load_range(_service, _ref):
        return None

    async def fake_handle_inter(range_ref, *_args, **_kwargs):
        return {"ref": range_ref, "segments": ["ok"], "focusIndex": 0, "he_ref": None}

    monkeypatch.setattr(
        "brain_service.services.study.daily_text.try_load_range",
        fake_try_load_range,
    )
    monkeypatch.setattr(
        "brain_service.services.study.daily_text.handle_inter_chapter_range",
        fake_handle_inter,
    )

    payload = await build_full_daily_text("Genesis 1:1-2:3", StubSefaria(), object())
    assert payload
    assert payload["segments"] == ["ok"]


@pytest.mark.anyio
async def test_build_full_daily_text_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_try_load_range(_service, _ref):
        return None

    async def fake_handle_inter(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "brain_service.services.study.daily_text.try_load_range",
        fake_try_load_range,
    )
    monkeypatch.setattr(
        "brain_service.services.study.daily_text.handle_inter_chapter_range",
        fake_handle_inter,
    )
    monkeypatch.setattr(
        "brain_service.services.study.daily_text.handle_jerusalem_talmud_range",
        fake_handle_inter,
    )

    payload = await build_full_daily_text("Genesis 1", StubSefaria(), object())
    assert payload is None


class TalmudSefariaStub:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def get_text(self, ref: str) -> dict:
        self.seen.append(ref)
        if ref.endswith("53a:1"):
            return {
                "ok": True,
                "data": {"text": "EN A1", "he": "HE A1", "title": "Zevachim", "indexTitle": "Zevachim"},
            }
        if ref.endswith("53a:2"):
            return {
                "ok": True,
                "data": {"text": "EN A2", "he": "HE A2", "title": "Zevachim", "indexTitle": "Zevachim"},
            }
        if ref.endswith("53b:1"):
            return {
                "ok": True,
                "data": {"text": "EN B1", "he": "HE B1", "title": "Zevachim", "indexTitle": "Zevachim"},
            }
        if ref.endswith("53b:2"):
            return {
                "ok": True,
                "data": {"text": "EN B2", "he": "HE B2", "title": "Zevachim", "indexTitle": "Zevachim"},
            }
        return {"ok": False}


@pytest.mark.anyio
async def test_build_full_daily_text_talmud_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_try_load_range(_service, _ref):
        return {
            "ref": "Zevachim 53",
            "title": "Zevachim",
            "indexTitle": "Zevachim",
            "text_segments": ["Blob A", "Blob B"],
            "he_segments": ["", ""],
            "type": "Talmud",
        }

    monkeypatch.setattr(
        "brain_service.services.study.daily_text.try_load_range",
        fake_try_load_range,
    )

    sefaria = TalmudSefariaStub()
    payload = await build_full_daily_text("Zevachim 53", sefaria, object())
    assert payload
    segments = payload["segments"]
    assert len(segments) == 4
    assert segments[0]["ref"] == "Zevachim 53a.1"
    assert segments[-1]["ref"] == "Zevachim 53b.2"
    # Ensure fallback asked for both amud sides
    assert "Zevachim 53b:1" in sefaria.seen


class IndexStub:
    def __init__(self) -> None:
        self.aliases = {"mishneh torah, haflaah": "Mishneh Torah, Hafla'ah"}

    def resolve_book_name(self, name: str) -> str | None:
        return self.aliases.get(name.lower())


@pytest.mark.anyio
async def test_build_full_daily_text_normalizes_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    responses: dict[str, dict] = {
        "Mishneh Torah, Hafla'ah 5": {
            "ref": "Mishneh Torah, Hafla'ah 5",
            "text_segments": ["EN"],
            "he_segments": ["HE"],
            "title": "Mishneh Torah",
            "indexTitle": "Mishneh Torah",
            "heRef": "משנה תורה",
        }
    }

    async def fake_try_load_range(_service, ref: str):
        return responses.get(ref)

    monkeypatch.setattr(
        "brain_service.services.study.daily_text.try_load_range",
        fake_try_load_range,
    )

    sefaria = StubSefaria()
    index = IndexStub()
    payload = await build_full_daily_text("Mishneh Torah, Haflaah 5", sefaria, index)
    assert payload
    assert payload["ref"] == "Mishneh Torah, Hafla'ah 5"
    assert len(payload["segments"]) == 1


@pytest.mark.anyio
async def test_build_full_daily_text_flattens_nested_text_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_try_load_range(_service, _ref):
        return {
            "ref": "Genesis 12:1-12:4",
            "title": "Genesis",
            "indexTitle": "Genesis",
            "text_segments": [
                ["EN 12:1", "EN 12:2"],
                ["EN 12:3", "EN 12:4"],
            ],
            "he_segments": [
                ["HE 12:1", "HE 12:2"],
                ["HE 12:3", "HE 12:4"],
            ],
            "spanningRefs": ["Genesis 12:1", "Genesis 12:3"],
        }

    monkeypatch.setattr(
        "brain_service.services.study.daily_text.try_load_range",
        fake_try_load_range,
    )

    payload = await build_full_daily_text("Genesis 12:1-12:4", StubSefaria(), object())
    assert payload
    refs = [segment["ref"] for segment in payload["segments"]]
    assert refs == [
        "Genesis 12:1",
        "Genesis 12:2",
        "Genesis 12:3",
        "Genesis 12:4",
    ]
