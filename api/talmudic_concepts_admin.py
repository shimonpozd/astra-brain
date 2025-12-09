import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from core.dependencies import get_talmudic_concept_service, require_admin_user
from brain_service.services.talmudic_concept_service import TalmudicConceptService
from brain_service.utils.text_processing import build_search_patterns, strip_niqqud

logger = logging.getLogger(__name__)
router = APIRouter()


class ConceptPayload(BaseModel):
    slug: str = Field(..., description="Unique identifier for the concept")
    term_he: str = Field(..., description="Primary Hebrew term")
    search_patterns: Optional[List[str]] = Field(default=None, description="Pre-generated regex patterns")
    short_summary_html: Optional[str] = None
    full_article_html: Optional[str] = None
    status: str = Field(default="draft", pattern="^(draft|published)$")


class ConceptResponse(ConceptPayload):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    generated_at: Optional[str] = None


class BatchConceptPayload(BaseModel):
    term_he: Optional[str] = Field(None, description="Primary Hebrew term")
    term: Optional[str] = Field(None, description="Alias for term_he")
    slug: Optional[str] = None
    short_summary_html: Optional[str] = None
    full_article_html: Optional[str] = None
    status: str = Field(default="published", pattern="^(draft|published)$")


@router.get("/admin/talmudic_concepts", dependencies=[Depends(require_admin_user)])
async def list_concepts(
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    items = await concept_service.list_all()
    return {"items": items}


@router.get(
    "/admin/talmudic_concepts/{slug}",
    dependencies=[Depends(require_admin_user)],
    response_model=ConceptResponse,
)
async def get_concept(
    slug: str = Path(..., description="Concept slug"),
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    obj = await concept_service.get(slug)
    if not obj:
        raise HTTPException(status_code=404, detail="Concept not found")
    return obj


@router.post(
    "/admin/talmudic_concepts",
    dependencies=[Depends(require_admin_user)],
    response_model=ConceptResponse,
)
async def create_concept(
    payload: ConceptPayload,
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    return await concept_service.upsert(
        slug=payload.slug,
        term_he=payload.term_he,
        search_patterns=payload.search_patterns,
        short_summary_html=payload.short_summary_html,
        full_article_html=payload.full_article_html,
        status=payload.status,
    )


@router.put(
    "/admin/talmudic_concepts/{slug}",
    dependencies=[Depends(require_admin_user)],
    response_model=ConceptResponse,
)
async def update_concept(
    payload: ConceptPayload,
    slug: str = Path(..., description="Concept slug"),
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    return await concept_service.upsert(
        slug=slug,
        term_he=payload.term_he,
        search_patterns=payload.search_patterns,
        short_summary_html=payload.short_summary_html,
        full_article_html=payload.full_article_html,
        status=payload.status,
    )


@router.delete("/admin/talmudic_concepts/{slug}", dependencies=[Depends(require_admin_user)])
async def delete_concept(
    slug: str,
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    await concept_service.delete(slug)
    return {"ok": True}


@router.post(
    "/admin/talmudic_concepts/batch",
    dependencies=[Depends(require_admin_user)],
    response_model=dict,
)
async def batch_import_concepts(
    payload: List[BatchConceptPayload],
    concept_service: TalmudicConceptService = Depends(get_talmudic_concept_service),
):
    """
    Batch import/update concepts. Slug is auto-generated from term_he if missing.
    """
    if not payload:
        return {"items": []}

    created_items: list[dict] = []
    for item in payload:
        term_he = (item.term_he or item.term or "").strip()
        if not term_he:
            continue
        slug = (item.slug or strip_niqqud(term_he).replace(" ", "-")).strip()
        if not slug:
            continue
        patterns = build_search_patterns(term_he)
        saved = await concept_service.upsert(
            slug=slug,
            term_he=term_he,
            search_patterns=patterns,
            short_summary_html=item.short_summary_html,
            full_article_html=item.full_article_html,
            status=item.status or "published",
        )
        created_items.append(saved)

    return {"items": created_items}
