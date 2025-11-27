from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from brain_service.core.dependencies import get_current_user, get_xp_service
from brain_service.models.db import User
from brain_service.services.xp_service import XpService

router = APIRouter()


class XpEventPayload(BaseModel):
  source: Literal["chat", "focus", "workbench", "lexicon", "daily"]
  verb: Optional[str] = None
  session_id: Optional[str] = None
  ref: Optional[str] = None
  title: Optional[str] = None
  chars: Optional[int] = Field(default=None, description="Plain text length if applicable")
  duration_ms: Optional[int] = Field(default=None, description="Playback duration if applicable")
  amount: Optional[int] = Field(default=None, description="Override computed XP")
  event_id: Optional[str] = Field(default=None, description="Idempotency key; if absent computed server-side")
  ts: Optional[int] = Field(default=None, description="Client timestamp (ms)")


class XpProfileResponse(BaseModel):
  xp_total: int
  level: int
  xp_in_level: int
  xp_to_next: int
  last_level_up_at: Optional[int] = None

class XpEvent(BaseModel):
  source: str
  verb: Optional[str] = None
  amount: int
  ref: Optional[str] = None
  title: Optional[str] = None
  ts: Optional[int] = None


@router.post("/xp/event", response_model=XpProfileResponse)
async def record_xp_event(
    payload: XpEventPayload,
    current_user: User = Depends(get_current_user),
    xp_service: XpService = Depends(get_xp_service),
):
  xp_profile = await xp_service.record_event(
      user_id=str(current_user.id),
      payload={
          "source": payload.source,
          "verb": payload.verb,
          "session_id": payload.session_id,
          "ref": payload.ref,
          "title": payload.title,
          "chars": payload.chars,
          "duration_ms": payload.duration_ms,
          "amount": payload.amount,
          "event_id": payload.event_id,
          "bucket": int((payload.ts or int(datetime.utcnow().timestamp() * 1000)) / 5000),
      },
  )
  return XpProfileResponse(**xp_profile.to_payload())


@router.get("/xp/profile", response_model=XpProfileResponse)
async def get_xp_profile(
    current_user: User = Depends(get_current_user),
    xp_service: XpService = Depends(get_xp_service),
):
  xp_profile = await xp_service.get_profile(str(current_user.id))
  return XpProfileResponse(**xp_profile.to_payload())


@router.get("/xp/history", response_model=List[XpEvent])
async def get_xp_history(
    current_user: User = Depends(get_current_user),
    xp_service: XpService = Depends(get_xp_service),
    limit: int = 50,
):
  events = await xp_service.list_history(str(current_user.id), limit=limit)
  return [XpEvent(**event) for event in events]
