import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from brain_service.main import app

client = TestClient(app)

MOCK_LLM_RESPONSE = {
    "spans": [
        {
            "sub_index": 0,
            "type": "Statement",
            "start_quote": "מאימתי קורין את שמע",
            "title_ru": "Разбор правила Мишны",
            "speaker": "Мудрецы"
        },
        {
            "sub_index": 1,
            "type": "Question",
            "start_quote": "עד שיכלה",
            "title_ru": "Вопрос Гмары",
            "speaker": None
        }
    ]
}


def test_sugya_calculate_map_validation():
    # Sending empty body should return 400
    response = client.post("/api/sugya/calculate-map", json={})
    assert response.status_code == 400


@patch("api.sugya.get_llm_for_task")
def test_sugya_calculate_map_success(mock_get_llm):
    # Mock LLM client response
    mock_client = AsyncMock()
    mock_choice = AsyncMock()
    mock_choice.message.content = f"```json\n{import_json_str()}\n```"
    mock_completion = AsyncMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    mock_get_llm.return_value = (mock_client, "gpt-4o", {}, ["json_mode"])

    payload = {
        "ref": "Chullin 89b:9",
        "segments": [
            {
                "ref": "Chullin 89b:9",
                "heText": "מאימתי קורין את שמע בערבין עד שיכלה",
                "enText": "From when do we read Shema in the evening"
            }
        ]
    }

    response = client.post("/api/sugya/calculate-map", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sugya_title"] == "Разбор правила Мишны"
    assert len(data["nodes"]) == 2
    assert data["nodes"][0]["type"] == "Statement"


def import_json_str():
    import json
    return json.dumps(MOCK_LLM_RESPONSE)
