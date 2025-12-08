import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.dependencies import get_profile_service, get_current_user, require_admin_user
from brain_service.models.db import User
from brain_service.services.profile_service import ProfileService

logger = logging.getLogger(__name__)
router = APIRouter()


class ProfileUpdatePayload(BaseModel):
    slug: str = Field(..., min_length=1)
    summary_html: str | None = None
    facts: dict | None = None
    title_en: str | None = None
    title_he: str | None = None

class AuthorOnlyPayload(BaseModel):
    name: str = Field(..., min_length=1)
    wiki_url: str | None = None
    raw_text: str | None = None
    period: str | None = None
    period_ru: str | None = None
    region: str | None = None
    generation: int | None = None
    sub_period: str | None = None
    force: bool = False


class ProfileListQuery(BaseModel):
    q: str | None = None
    unverified: bool = False
    limit: int = 100


@router.get("/profile")
async def profile_handler(
    slug: str,
    profile_service: ProfileService = Depends(get_profile_service),
    current_user: User = Depends(get_current_user),
):
    """
    Build or return a cached author/work profile (Sefaria + Wikipedia + LLM).
    """
    result = await profile_service.get_profile(slug)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Profile not found"))
    return result


@router.delete("/profile")
async def profile_delete_handler(
    slug: str,
    profile_service: ProfileService = Depends(get_profile_service),
    admin: User = Depends(require_admin_user),
):
    """
    Delete profile cache entry (does not remove works/authors).
    """
    await profile_service.delete_profile(slug.strip())
    return {"ok": True}


@router.patch("/profile")
async def profile_update_handler(
    payload: ProfileUpdatePayload,
    profile_service: ProfileService = Depends(get_profile_service),
    admin: User = Depends(require_admin_user),
):
    """
    Save manual edits to a profile and mark it verified.
    """
    result = await profile_service.save_manual_profile(
        payload.slug.strip(),
        payload.summary_html,
        payload.facts,
        verified_by=admin.username,
        title_en=payload.title_en,
        title_he=payload.title_he,
    )
    return result


@router.post("/profile/regenerate")
async def profile_regenerate_handler(
    slug: str,
    profile_service: ProfileService = Depends(get_profile_service),
    admin: User = Depends(require_admin_user),
):
    """
    Force regenerate a profile, clearing manual edits.
    """
    result = await profile_service.regenerate_profile(slug.strip())
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Profile not found"))
    return result


@router.post("/profile/author_only")
async def profile_author_only_handler(
    payload: AuthorOnlyPayload,
    profile_service: ProfileService = Depends(get_profile_service),
    admin: User = Depends(require_admin_user),
):
    """
    Generate/save author-only profile (без произведения) по имени на иврите/slug.
    """
    result = await profile_service.generate_author_profile(
        name=payload.name.strip(),
        wiki_url=payload.wiki_url,
        raw_text=payload.raw_text,
        period=payload.period,
        period_ru=payload.period_ru,
        region=payload.region,
        generation=payload.generation,
        sub_period=payload.sub_period,
        force=payload.force,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate author profile"))
    return result


@router.get("/profile/list")
async def profile_list_handler(
    q: str | None = None,
    unverified: bool = False,
    limit: int = 100,
    profile_service: ProfileService = Depends(get_profile_service),
    admin: User = Depends(require_admin_user),
):
    return await profile_service.list_profiles(query=q.strip() if q else None, only_unverified=unverified, limit=limit)
