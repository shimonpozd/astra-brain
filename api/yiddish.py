from pathlib import Path
import json
import logging
from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy import select, or_

from brain_service.core.database import session_scope
from brain_service.core.dependencies import get_current_user, get_yiddish_service, get_wiktionary_yiddish_service
from brain_service.models.db import User
from brain_service.models.yiddish import YiddishWordCard
from brain_service.services.yiddish_service import YiddishService
from brain_service.services.wiktionary_yiddish import WiktionaryYiddishService
from config.prompts import get_prompt
from core.llm_config import get_llm_for_task, LLMConfigError

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple in-memory stores (fallback until DB/Redis wired)
_user_queue: Dict[str, List[Dict[str, Any]]] = {}
_user_attestations: Dict[str, List[Dict[str, Any]]] = {}
_user_vocab: Dict[str, Dict[str, Any]] = {}


def _get_session_factory(request: Request):
    session_factory = getattr(request.app.state, "db_session_factory", None)
    if session_factory is None:
        raise HTTPException(status_code=503, detail="Database session factory unavailable")
    return session_factory


def _load_sicha_file(sicha_id: str):
    """
    Load static Yiddish siha data from disk (demo fallback).
    """
    base = Path(__file__).resolve().parents[1] / "data" / "yiddish"
    file_path = base / f"{sicha_id}.json"
    if not file_path.exists():
        # allow generic filename
        file_path = base / "page_0001.json"
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load Yiddish siha file", extra={"file": str(file_path), "error": str(exc)})
        raise HTTPException(status_code=500, detail="Failed to load siha data")


@router.get("/yiddish/sichos")
async def list_sichos(
    current_user: User = Depends(get_current_user),
    yiddish_service: YiddishService = Depends(get_yiddish_service),
):
    """
    Demo list of available sichos (fallback until DB is wired).
    """
    return await yiddish_service.list_sichos(str(current_user.id))


@router.get("/yiddish/sicha/{sicha_id}")
async def get_sicha(
    sicha_id: str,
    current_user: User = Depends(get_current_user),
    yiddish_service: YiddishService = Depends(get_yiddish_service),
):
    """
    Return paragraphs/tokens/notes for a specific siha.
    """
    return await yiddish_service.get_sicha(sicha_id, str(current_user.id))


@router.post("/yiddish/attestation")
async def save_attestation(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    yiddish_service: YiddishService = Depends(get_yiddish_service),
):
    """
    Save attestation (lemma + sense in context).
    """
    return await yiddish_service.save_attestation(str(current_user.id), payload)


@router.post("/yiddish/queue")
async def update_queue(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    yiddish_service: YiddishService = Depends(get_yiddish_service),
):
    """
    Add/remove items from study queue.
    """
    action = payload.get("action")
    entry = {
        "lemma": payload.get("lemma"),
        "sense_id": payload.get("sense_id"),
        "source_pid": payload.get("source_pid"),
    }
    return await yiddish_service.update_queue(str(current_user.id), action, entry)


@router.post("/yiddish/wordcards/lookup")
async def lookup_wordcards(
    payload: Dict[str, Any],
    request: Request,
    current_user: User = Depends(get_current_user),
    ui_lang: str = "ru",
    version: int = 1,
):
    lemmas = payload.get("lemmas") or []
    surfaces = payload.get("surfaces") or []
    if not isinstance(lemmas, list) or not isinstance(surfaces, list):
        raise HTTPException(status_code=400, detail="Payload must include lemmas/surfaces lists")
    keys = list({*(str(l).strip() for l in lemmas if str(l).strip()), *(str(s).strip() for s in surfaces if str(s).strip())})
    if not keys:
        raise HTTPException(status_code=400, detail="Payload must include lemmas or surfaces list")

    session_factory = _get_session_factory(request)
    async with session_scope(session_factory) as session:
        result = await session.execute(
            select(YiddishWordCard).where(
                or_(YiddishWordCard.lemma.in_(keys), YiddishWordCard.word_surface.in_(keys)),
                YiddishWordCard.ui_lang == ui_lang,
                YiddishWordCard.source == "wiktionary",
                YiddishWordCard.version == version,
            )
        )
        items = []
        for row in result.scalars().all():
            if row.data:
                items.append(row.data)

    return {"ok": True, "items": items}


@router.post("/yiddish/exam/start")
async def start_exam(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    yiddish_service: YiddishService = Depends(get_yiddish_service),
):
    """
    Stub exam generator from queue entries.
    """
    items = payload.get("lemmas", [])
    return await yiddish_service.start_exam(str(current_user.id), items)


@router.post("/yiddish/exam/generate")
async def generate_exam(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    yiddish_service: YiddishService = Depends(get_yiddish_service),
):
    """
    Generate Mahjong-style tiles for Yiddish matching.
    """
    min_words = payload.get("min_words", 8)
    max_words = payload.get("max_words", 12)
    return await yiddish_service.generate_mahjong_exam(str(current_user.id), min_words=min_words, max_words=max_words)


@router.get("/yiddish/vocab/{lemma}")
async def get_vocab(
    lemma: str,
    current_user: User = Depends(get_current_user),
    yiddish_service: YiddishService = Depends(get_yiddish_service),
):
    """
    Return vocab entry (stub).
    """
    return await yiddish_service.get_vocab(str(current_user.id), lemma)


@router.post("/yiddish/tts")
async def yiddish_tts(payload: Dict[str, Any], current_user: User = Depends(get_current_user)):
    """
    Stub TTS endpoint; returns none engine.
    """
    return {"url": "", "engine_used": "none"}


@router.post("/yiddish/ask")
async def yiddish_ask(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """
    Ask/Explain preset questions in Yiddish mode using the YIDDISH agent.
    """
    system_prompt = get_prompt("actions.yiddish_system")
    user_template = get_prompt("actions.yiddish_user_template")
    if not system_prompt or not user_template:
        raise HTTPException(status_code=500, detail="Yiddish prompts not configured")

    try:
        llm_client, model, reasoning_params, _capabilities = get_llm_for_task("YIDDISH_ASK")
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=f"LLM not configured: {exc}")

    known_lemmas = payload.get("known_lemmas") or []
    known_lemmas_str = json.dumps(known_lemmas, ensure_ascii=False)
    meta = payload.get("meta") or {}
    task_label = payload.get("task") or "Explain meaning (RU)"

    user_msg = user_template.format(
        selected_text=payload.get("selected_text", ""),
        sentence_before=payload.get("sentence_before", ""),
        sentence_after=payload.get("sentence_after", ""),
        work=meta.get("work", ""),
        volume=meta.get("volume", ""),
        parsha=meta.get("parsha", ""),
        section=meta.get("section", ""),
        lang=meta.get("lang", "yi"),
        known_lemmas=known_lemmas_str,
        task=task_label,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    try:
        completion = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **reasoning_params,
        )
        content = completion.choices[0].message.content if completion and completion.choices else ""
        if not content:
            raise ValueError("Empty LLM response")
    except Exception as exc:
        logger.error("Yiddish ask failed", extra={"error": str(exc)}, exc_info=True)
        # Fallback answer so UI keeps working
        content = "Не удалось получить ответ от агента. Попробуйте позже."

    return {
        "answer": content,
        "task": task_label,
        "sicha_id": payload.get("sicha_id"),
        "anchor": payload.get("anchor"),
        "prompt_used": {
            "system": "actions.yiddish_system",
            "user": "actions.yiddish_user_template",
            "model": model,
        },
    }


@router.get("/yiddish/wordcard")
async def yiddish_wordcard(
    word: str,
    context: str | None = None,
    lemma_guess: str | None = None,
    pos_guess: str | None = None,
    ui_lang: str = "ru",
    include_evidence: bool = False,
    include_llm_output: bool = False,
    force_refresh: bool = False,
    allow_llm_fallback: bool = False,
    persist: bool = True,
    current_user: User = Depends(get_current_user),
    wiktionary_service: WiktionaryYiddishService = Depends(get_wiktionary_yiddish_service),
):
    """
    Build or fetch a Yiddish WordCard for Wiktionary-backed lookup.
    """
    return await wiktionary_service.get_wordcard(
        word_surface=word,
        lemma_guess=lemma_guess,
        pos_guess=pos_guess,
        context_sentence=context,
        ui_lang=ui_lang,
        include_evidence=include_evidence,
        include_llm_output=include_llm_output,
        force_refresh=force_refresh,
        allow_llm_fallback=allow_llm_fallback,
        persist=persist,
    )
