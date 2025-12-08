import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config.prompts import get_prompt
from core.llm_config import LLMConfigError, get_llm_for_task
from brain_service.utils.text_processing import build_search_patterns

logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateConceptRequest(BaseModel):
    term_he: str
    synonyms: Optional[List[str]] = None


class GenerateConceptResponse(BaseModel):
    short_summary_html: str
    full_article_html: str
    search_patterns: List[str]


async def _run_completion(system_prompt: str, term_he: str, *, task: str) -> Dict[str, Any]:
    try:
        llm_client, model, reasoning_params, capabilities = get_llm_for_task(task)
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=f"LLM not configured: {exc}")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Термин: {term_he}\nСохрани огласовки в ключевых словах. Верни HTML.",
        },
    ]
    req: Dict[str, Any] = {
        **reasoning_params,
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if "json_mode" in capabilities:
        req["response_format"] = {"type": "json_object"}

    try:
        completion = await llm_client.chat.completions.create(**req)
    except Exception as exc:
        logger.error("LLM call failed for concept generation", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=502, detail="LLM generation failed")

    content = completion.choices[0].message.content if completion and completion.choices else ""
    if not content:
        raise HTTPException(status_code=502, detail="Empty LLM response")

    # Try to decode JSON payload first
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {"html": cleaned}


@router.post("/generate-concept", response_model=GenerateConceptResponse)
async def generate_concept_content(payload: GenerateConceptRequest):
    summary_prompt = get_prompt("actions.concept_summary_system")
    article_prompt = get_prompt("actions.concept_article_system")
    if not summary_prompt or not article_prompt:
        raise HTTPException(status_code=500, detail="Concept prompts not configured")

    summary_data = await _run_completion(summary_prompt, payload.term_he, task="TALMUDIC_CONCEPT_GEN")
    article_data = await _run_completion(article_prompt, payload.term_he, task="TALMUDIC_CONCEPT_GEN")

    summary_html = summary_data.get("summary_html") or summary_data.get("html") or ""
    synonyms = summary_data.get("synonyms") if isinstance(summary_data, dict) else None
    if not isinstance(synonyms, list):
        synonyms = payload.synonyms or []

    article_html = article_data.get("full_article_html") or article_data.get("html") or ""

    patterns = build_search_patterns(payload.term_he, *synonyms)

    return GenerateConceptResponse(
        short_summary_html=summary_html,
        full_article_html=article_html,
        search_patterns=patterns,
    )
