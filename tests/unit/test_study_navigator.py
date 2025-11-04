import types

import pytest

from brain_service.services.study.navigator import generate_neighbors


class StubSefariaService:
    def __init__(self) -> None:
        self._chapter_segments = ["v1", "v2", "v3"]
        self._verse_payloads = {
            "Genesis 1:1": "In the beginning",
            "Genesis 1:2": "Second verse",
            "Genesis 1:3": "Third verse",
        }

    async def get_text(self, ref: str):
        if ref == "Genesis 1":
            return {
                "ok": True,
                "data": {
                    "ref": ref,
                    "text": "\n".join(self._chapter_segments),
                    "text_segments": list(self._chapter_segments),
                    "title": "Genesis",
                    "indexTitle": "Genesis",
                },
            }
        verse_text = self._verse_payloads.get(ref)
        if verse_text:
            return {
                "ok": True,
                "data": {
                    "ref": ref,
                    "text": verse_text,
                    "he_text": verse_text,
                    "title": "Genesis",
                    "indexTitle": "Genesis",
                },
            }
        return {"ok": False, "data": None}


@pytest.mark.anyio
async def test_generate_neighbors_uses_segmented_chapter_lengths():
    sefaria = StubSefariaService()
    index_service = types.SimpleNamespace(toc=[{"title": "Genesis", "lengths": None}])

    neighbors = await generate_neighbors(
        "Genesis 1:1",
        2,
        direction="next",
        sefaria_service=sefaria,
        index_service=index_service,
    )

    assert [segment["ref"] for segment in neighbors] == ["Genesis 1:2", "Genesis 1:3"]
