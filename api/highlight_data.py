import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.dependencies import get_profile_service, get_talmudic_concept_service, require_admin_user
from brain_service.models.db import User
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


class SageMappingPayload(BaseModel):
    sage_slug: str
    raw_text: str


class ConceptMappingPayload(BaseModel):
    concept_slug: str
    raw_text: str


class CustomConceptPayload(BaseModel):
    term_he: str
    pattern: str
    short_summary_html: Optional[str] = None
    slug: Optional[str] = None


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
        
        patterns_list: list[str] = []
        base_norm = strip_niqqud(name_he or slug)
        base_pat = generate_vowel_insensitive_regex(base_norm)
        if base_pat:
            patterns_list.append(base_pat)
            
        aliases = display.get("aliases") or []
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    alias_pat = generate_vowel_insensitive_regex(strip_niqqud(alias.strip()))
                    if alias_pat and alias_pat not in patterns_list:
                        patterns_list.append(alias_pat)
                        
        if not patterns_list:
            continue
            
        pattern = "|".join(patterns_list)

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


@router.post("/highlight/sages/mapping")
async def add_sage_mapping(
    payload: SageMappingPayload,
    profile_service: ProfileService = Depends(get_profile_service),
    admin: User = Depends(require_admin_user),
):
    """
    Save a raw text alias/pattern mapping for a sage profile.
    """
    profile_res = await profile_service.get_profile(payload.sage_slug.strip())
    if not profile_res or not profile_res.get("ok"):
        raise HTTPException(status_code=404, detail="Sage profile not found")

    profile_data = profile_res.get("profile") or {}
    facts = profile_data.get("facts") or {}
    if not isinstance(facts, dict):
        facts = {}
    author_facts = facts.get("author") if isinstance(facts, dict) else {}
    if not isinstance(author_facts, dict):
        author_facts = {}
    display = author_facts.get("display") if isinstance(author_facts, dict) else {}
    if not isinstance(display, dict):
        display = {}

    aliases = display.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []

    raw_clean = strip_niqqud(payload.raw_text.strip())
    import re
    unprefixed = re.sub(r'^[ובכלמדהשה]+(?=(?:רבי|רב|רן|מר|שמעון|אלעזר|יוחנן|יהודה|יוסי|חנינא|אשי|פפא|אבא|רבא|אביי|ריש)\b)', '', raw_clean)
    
    added_any = False
    for candidate in [raw_clean, unprefixed]:
        if candidate and candidate not in aliases:
            aliases.append(candidate)
            added_any = True

    if added_any:
        display["aliases"] = aliases
        author_facts["display"] = display
        facts["author"] = author_facts
        await profile_service.save_manual_profile(
            slug=payload.sage_slug.strip(),
            summary_html=profile_data.get("summary_html"),
            facts=facts,
            verified_by=admin.username,
        )
    return {"ok": True, "sage_slug": payload.sage_slug, "alias": raw_clean}


@router.post("/highlight/concepts/mapping")
async def add_concept_mapping(
    payload: ConceptMappingPayload,
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
    admin: User = Depends(require_admin_user),
):
    """
    Save a raw text pattern mapping for an existing concept.
    """
    concept = await concept_service.get(payload.concept_slug.strip())
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")

    patterns = concept.get("search_patterns") or []
    new_pat = payload.raw_text.strip()
    if new_pat and new_pat not in patterns:
        patterns.append(new_pat)
        res = await concept_service.upsert(
            slug=concept["slug"],
            term_he=concept["term_he"],
            search_patterns=patterns,
            short_summary_html=concept.get("short_summary_html"),
            status=concept.get("status") or "published",
        )
        return {"ok": True, "concept": res}
    return {"ok": True, "concept": concept}


@router.post("/highlight/concepts/custom")
async def add_custom_concept(
    payload: CustomConceptPayload,
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
    admin: User = Depends(require_admin_user),
):
    """
    Add or update a custom Talmudic concept pattern for dynamic highlighting with optional description.
    """
    import re
    if payload.slug:
        slug = payload.slug.strip()
        existing = await concept_service.get(slug)
        patterns = (existing.get("search_patterns") or []) if existing else []
        if payload.pattern.strip() and payload.pattern.strip() not in patterns:
            patterns.append(payload.pattern.strip())
        summary = payload.short_summary_html if payload.short_summary_html is not None else (existing.get("short_summary_html") if existing else None)
        res = await concept_service.upsert(
            slug=slug,
            term_he=payload.term_he.strip(),
            search_patterns=patterns,
            short_summary_html=summary,
            status="published",
        )
    else:
        clean_term = re.sub(r'[\s]+', '-', payload.term_he.strip())
        slug = f"custom-{clean_term}"
        existing = await concept_service.get(slug)
        patterns = (existing.get("search_patterns") or []) if existing else []
        if payload.pattern.strip() and payload.pattern.strip() not in patterns:
            patterns.append(payload.pattern.strip())
        res = await concept_service.upsert(
            slug=slug,
            term_he=payload.term_he.strip(),
            search_patterns=patterns,
            short_summary_html=payload.short_summary_html,
            status="published",
        )
    return {"ok": True, "concept": res}


# Backward-compatible singular alias
@router.get("/highlight/sage", response_model=dict)
async def highlight_sage_alias(profile_service: ProfileService = Depends(get_profile_service)):
    return await highlight_sages(profile_service)


@router.get("/highlight/concept", response_model=dict)
async def highlight_concept_alias(
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    return await highlight_concepts(concept_service)

