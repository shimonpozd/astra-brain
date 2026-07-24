import pytest
from brain_service.models.study_models import StudyResolveRequest
from brain_service.services.study_service import normalize_russian_ref, StudyService
from brain_service.services.study.formatter import clean_html, extract_hebrew_only


def test_normalize_russian_ref_chullin():
    assert normalize_russian_ref("Хулин 84а") == "Chullin 84a"
    assert normalize_russian_ref("Хулин 84б") == "Chullin 84b"
    assert normalize_russian_ref("Chullin 84а") == "Chullin 84a"
    assert normalize_russian_ref("Chullin 84a") == "Chullin 84a"


def test_normalize_russian_ref_other_tractates():
    assert normalize_russian_ref("Брахот 2а") == "Berakhot 2a"
    assert normalize_russian_ref("Бава Мециа 59б") == "Bava Metzia 59b"
    assert normalize_russian_ref("Бытие 1:1") == "Genesis 1:1"


@pytest.mark.asyncio
async def test_study_service_resolve_reference():
    service = StudyService(None, None, None, None)
    res = await service.resolve_reference(StudyResolveRequest(text="Хулин 84а"))
    assert res["ok"] is True
    assert res["ref"] == "Chullin 84a"


def test_clean_html_with_list_input():
    assert clean_html(["<p>Line 1</p>", "Line 2"]) == "Line 1 Line 2"
    assert extract_hebrew_only(["<span>Hebrew 1</span>", "Hebrew 2"]) == "Hebrew 1 Hebrew 2"
