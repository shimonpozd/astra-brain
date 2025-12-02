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


@router.get("/profile/list")
async def profile_list_handler(
    q: str | None = None,
    unverified: bool = False,
    limit: int = 100,
    profile_service: ProfileService = Depends(get_profile_service),
    admin: User = Depends(require_admin_user),
):
    return await profile_service.list_profiles(query=q.strip() if q else None, only_unverified=unverified, limit=limit)
