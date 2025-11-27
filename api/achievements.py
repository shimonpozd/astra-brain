from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from brain_service.core.dependencies import get_current_user, get_achievement_service
from brain_service.models.db import User
from brain_service.services.achievement_service import AchievementService

router = APIRouter()


class AchievementResponse(BaseModel):
    category: str
    level: str
    value: int
    to_next: int | None


@router.get("/achievements", response_model=list[AchievementResponse])
async def get_achievements(
    current_user: User = Depends(get_current_user),
    achievement_service: AchievementService = Depends(get_achievement_service),
):
    progress = await achievement_service.get_progress(str(current_user.id))
    levels = achievement_service.compute_levels(progress)
    result: list[AchievementResponse] = []
    for cat, info in levels.items():
        result.append(
            AchievementResponse(
                category=cat,
                level=info["level"],
                value=info["value"],
                to_next=info["to_next"],
            )
        )
    return result
