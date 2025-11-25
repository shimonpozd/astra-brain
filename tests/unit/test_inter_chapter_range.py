import sys
from pathlib import Path

import pytest

# Ensure project root is on the path so importing brain_service and core works in isolation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain_service.services.study.range_handlers import handle_inter_chapter_range
from brain_service.services.study_utils import _handle_inter_chapter_range


def _make_data(ref: str) -> dict:
    return {
        "text": f"EN {ref}",
        "he": f"פסוק {ref}",
        "title": "Genesis",
        "indexTitle": "Bereshit",
        "heRef": "בראשית",
    }


class StubSefaria:
    def __init__(self, mapping: dict[str, dict]):
        self.mapping = mapping
        self.calls: list[str] = []

    async def get_text(self, ref: str) -> dict:
        self.calls.append(ref)
        data = self.mapping.get(ref)
        if data:
            return {"ok": True, "data": data}
        return {"ok": False, "data": None}


@pytest.mark.anyio
async def test_range_handlers_inter_chapter_mid_verse() -> None:
    refs = {
        "Genesis 1:21": _make_data("Genesis 1:21"),
        "Genesis 1:22": _make_data("Genesis 1:22"),
        "Genesis 2:1": _make_data("Genesis 2:1"),
        "Genesis 2:2": _make_data("Genesis 2:2"),
    }
    sefaria = StubSefaria(refs)

    payload = await handle_inter_chapter_range("Genesis 1:21-2:2", sefaria)

    assert payload is not None
    assert [seg["ref"] for seg in payload["segments"]] == [
        "Genesis 1:21",
        "Genesis 1:22",
        "Genesis 2:1",
        "Genesis 2:2",
    ]
    # Ensure we never try to fetch from the start of the chapter
    assert sefaria.calls[0] == "Genesis 1:21"


@pytest.mark.anyio
async def test_legacy_inter_chapter_mid_verse() -> None:
    refs = {
        "Genesis 1:21": _make_data("Genesis 1:21"),
        "Genesis 1:22": _make_data("Genesis 1:22"),
        "Genesis 2:1": _make_data("Genesis 2:1"),
        "Genesis 2:2": _make_data("Genesis 2:2"),
    }
    sefaria = StubSefaria(refs)

    payload = await _handle_inter_chapter_range("Genesis 1:21-2:2", sefaria)

    assert payload is not None
    assert [seg["ref"] for seg in payload["segments"]] == [
        "Genesis 1:21",
        "Genesis 1:22",
        "Genesis 2:1",
        "Genesis 2:2",
    ]
    assert sefaria.calls[0] == "Genesis 1:21"
