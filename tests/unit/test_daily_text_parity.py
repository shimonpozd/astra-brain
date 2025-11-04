import json
from pathlib import Path

import pytest

from brain_service.services.study.daily_text import build_full_daily_text


FIXTURE_DIR = Path("docs/brain/fixtures")


def _segment_signature(segment):
    return {
        "ref": segment.get("ref"),
        "text": segment.get("text"),
        "heText": segment.get("heText"),
    }


class FixtureSefaria:
    def __init__(self, calls):
        self._responses = {}
        for entry in calls:
            key = (entry["ref"], entry.get("lang") or "default")
            self._responses[key] = entry["result"]

    async def get_text(self, ref: str, lang: str | None = None):
        key = (ref, lang or "default")
        if key not in self._responses:
            raise AssertionError(f"Fixture missing response for {key}")
        return self._responses[key]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture_name",
    [
        "Genesis_1.json",
        "Genesis_1-1-2-3.json",
        "Zevachim_24.json",
        "Jerusalem_Talmud_Sotah_5-4-3-6-3.json",
    ],
)
async def test_daily_text_parity_from_fixture(fixture_name: str) -> None:
    fixture_path = FIXTURE_DIR / fixture_name
    assert fixture_path.exists(), f"Fixture {fixture_path} not found"

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    stub_sefaria = FixtureSefaria(payload.get("sefaria_calls", []))

    result = await build_full_daily_text(payload["ref"], stub_sefaria, object())

    expected_legacy = payload.get("legacy")
    expected_modular = payload.get("modular")
    if not expected_legacy:
        assert result == expected_modular
        return

    assert result is not None, "Builder returned None"

    assert len(result["segments"]) == len(expected_legacy["segments"])
    for got, want in zip(result["segments"], expected_legacy["segments"]):
        assert _segment_signature(got) == _segment_signature(want)
