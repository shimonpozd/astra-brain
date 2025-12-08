import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.dependencies import get_profile_service, get_talmudic_concept_service
from brain_service.services.profile_service import ProfileService
from brain_service.services.talmudic_concept_service import TalmudicConceptService
from brain_service.utils.text_processing import generate_vowel_insensitive_regex, strip_niqqud

logger = logging.getLogger(__name__)
router = APIRouter()


ALLOWED_SAGE_PERIODS = {"zugot", "tannaim", "amoraim", "achronim"}


def _has_hebrew(text: str | None) -> bool:
    if not text:
        return False
    return any("\u0590" <= ch <= "\u05ff" for ch in text)


class SageHighlight(BaseModel):
    slug: str
    name_he: Optional[str] = None
    name_ru: Optional[str] = None
    period: Optional[str] = None
    generation: Optional[int] = None
    region: Optional[str] = None
    period_label_ru: Optional[str] = None
    lifespan: Optional[str] = None
    regex_pattern: str


class ConceptHighlight(BaseModel):
    slug: str
    term_he: Optional[str] = None
    search_patterns: List[str]
    short_summary_html: Optional[str] = None


@router.get("/highlight/sages", response_model=dict)
async def highlight_sages(profile_service: ProfileService = Depends(get_profile_service)):
    """
    Public endpoint: returns lightweight sage profiles with pre-built regex patterns
    for vowel-insensitive highlighting.
    """
    try:
        res = await profile_service.list_profiles(limit=5000)
        profiles = res.get("items") if isinstance(res, dict) else res
    except Exception as exc:
        logger.error("highlight_sages:list_profiles failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Failed to load sages data")

    highlights: list[SageHighlight] = []
    if not isinstance(profiles, list):
        profiles = []

    for item in profiles:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not slug:
            continue

        facts = item.get("facts") or {}
        author_facts = facts.get("author") if isinstance(facts, dict) else {}
        author_facts = author_facts if isinstance(author_facts, dict) else {}
        period_val_raw = item.get("period") or author_facts.get("period")
        period_val = period_val_raw.lower() if isinstance(period_val_raw, str) else None

        generation_val = None
        generation_raw = item.get("generation") or (author_facts.get("generation") if isinstance(author_facts, dict) else None)
        if isinstance(generation_raw, (int, float)):
            generation_val = int(generation_raw)
        elif isinstance(generation_raw, str) and generation_raw.strip().lstrip("+-").isdigit():
            generation_val = int(generation_raw.strip())

        display = author_facts.get("display") if isinstance(author_facts, dict) else {}
        display = display if isinstance(display, dict) else {}
        title_en = item.get("title_en")
        author_title_en = author_facts.get("title_en") if isinstance(author_facts, dict) else None

        name_he = (
            item.get("title_he")
            or author_facts.get("title_he")
            or display.get("name_he")
            or display.get("title_he")
            or (title_en if _has_hebrew(title_en) else None)
            or (author_title_en if _has_hebrew(author_title_en) else None)
            or slug
        )
        name_ru = display.get("name_ru") or display.get("title_ru")
        period_ru = display.get("period_ru") or author_facts.get("period_ru")
        region_val = author_facts.get("region")
        lifespan_val = author_facts.get("lifespan") or item.get("lifespan")
        normalized = strip_niqqud(name_he or slug)
        pattern = generate_vowel_insensitive_regex(normalized)
        if not pattern:
            continue

        highlights.append(
            SageHighlight(
                slug=slug,
                name_he=name_he,
                name_ru=name_ru,
                period=period_val_raw or "amoraim",
                generation=generation_val,
                region=region_val,
                period_label_ru=period_ru,
                lifespan=lifespan_val,
                regex_pattern=pattern,
            )
        )

    return {"items": [h.model_dump() for h in highlights]}


@router.get("/highlight/concepts", response_model=dict)
async def highlight_concepts(
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    """
    Public endpoint: returns published concepts with search patterns for highlighting.
    """
    try:
        concepts = await concept_service.list_published()
    except Exception as exc:
        logger.error("highlight_concepts:list_published failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Failed to load concepts data")

    items: list[ConceptHighlight] = []
    for c in concepts or []:
        search_patterns = c.get("search_patterns") if isinstance(c, dict) else None
        patterns = search_patterns if isinstance(search_patterns, list) else []
        items.append(
            ConceptHighlight(
                slug=c.get("slug"),
                term_he=c.get("term_he"),
                search_patterns=patterns,
                short_summary_html=c.get("short_summary_html"),
            )
        )

    return {"items": [i.model_dump() for i in items]}


# Backward-compatible singular alias
@router.get("/highlight/sage", response_model=dict)
async def highlight_sage_alias(profile_service: ProfileService = Depends(get_profile_service)):
    return await highlight_sages(profile_service)


@router.get("/highlight/concept", response_model=dict)
async def highlight_concept_alias(
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    return await highlight_concepts(concept_service)
