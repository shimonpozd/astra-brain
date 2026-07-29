import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from brain_service.main import app

client = TestClient(app)

MOCK_LLM_RESPONSE = {
    "sugya_title": "Разбор правила Мишны",
    "markdown_tree": "# Statement\n## Question\n### Attack",
    "nodes": [
        {
            "id": "node_1",
            "level": 1,
            "type": "Statement",
            "title": "Тезис Мишны",
            "ref": "Chullin 89b:9",
            "start_anchor": "מאימתי",
            "end_anchor": "עד שיכלה"
        },
        {
            "id": "node_2",
            "level": 2,
            "type": "Question",
            "title": "Вопрос Гмары",
            "ref": "Chullin 89b:10",
            "start_anchor": "מנא הני מילי",
            "end_anchor": "דאמר קרא"
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
