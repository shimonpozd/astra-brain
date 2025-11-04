import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.dependencies import (
    get_chat_service,
    get_session_service,
    get_redis_client,
    get_current_user,
)
from core.rate_limiting import rate_limit_dependency
from brain_service.services.chat_service import ChatService
from brain_service.services.session_service import SessionService
from brain_service.services.study.stream_router import select_today_unit
from brain_service.services.study.tz_utils import now_in_tz, resolve_timezone, seconds_until_next_midnight, next_midnight

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Models ---
class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None

# --- Endpoints ---
@router.post("/chat/stream")
async def chat_stream_handler(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    _: bool = Depends(rate_limit_dependency(limit=5)),  # Stricter limit for LLM endpoints
    current_user=Depends(get_current_user),
):
    """Stream chat response with LLM and tool integration."""
    expected_user_id = str(current_user.id)
    if request.user_id and request.user_id != expected_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User mismatch")
    return StreamingResponse(
        chat_service.process_chat_stream(
            text=request.text,
            user_id=expected_user_id,
            session_id=request.session_id,
            agent_id=request.agent_id
        ),
        media_type="application/x-ndjson"
    )

@router.post("/chat/stream-blocks")
async def chat_stream_blocks_handler(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    _: bool = Depends(rate_limit_dependency(limit=5)),  # Stricter limit for LLM endpoints
    current_user=Depends(get_current_user),
):
    """Stream chat response with block-by-block rendering."""
    expected_user_id = str(current_user.id)
    if request.user_id and request.user_id != expected_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User mismatch")
    return StreamingResponse(
        chat_service.process_chat_stream_with_blocks(
            text=request.text,
            user_id=expected_user_id,
            session_id=request.session_id,
            agent_id=request.agent_id
        ),
        media_type="application/x-ndjson"
    )

@router.get("/chats")
async def get_chats(
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Get chat sessions for the authenticated user."""
    return await chat_service.get_all_chats(str(current_user.id))

@router.get("/chats/{session_id}")
async def get_chat_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Get chat history for a specific session."""
    history = await chat_service.get_chat_history(session_id, str(current_user.id))
    return {"history": history}

@router.delete("/sessions/{session_id}/{session_type}", status_code=204)
async def delete_session(
    session_id: str, 
    session_type: str, 
    chat_service: ChatService = Depends(get_chat_service),
    current_user=Depends(get_current_user),
):
    """Delete a session by ID and type."""
    from brain_service.services.session_service import SessionService  # lazy import to avoid circular

    session_service = SessionService(
        chat_service.redis_client,
        chat_service.user_service,  # type: ignore[arg-type]
    )

    success = await session_service.delete_session(session_id, str(current_user.id))
    if not success:
        raise HTTPException(status_code=404, detail=f"{session_type.title()} session not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# --- Session Management with SessionService ---

@router.get("/sessions")
async def get_all_sessions_handler(
    session_service: SessionService = Depends(get_session_service),
    current_user=Depends(get_current_user),
):
    """Get all chat and study sessions using SessionService."""
    return await session_service.get_all_sessions(str(current_user.id))

@router.get("/sessions/{session_id}")
async def get_session_handler(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    current_user=Depends(get_current_user),
):
    """Get a specific session by ID using SessionService."""
    session = await session_service.get_session(str(current_user.id), session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# --- Daily Learning Sessions ---

@router.get("/daily/calendar")
async def get_daily_calendar(
    tz: Optional[str] = Query(None, description="IANA timezone, e.g. Europe/Amsterdam")
):
    """Get today's calendar items for virtual daily chat list."""
    import httpx

    try:
        tz_obj = resolve_timezone(tz, None)
        today_dt = now_in_tz(tz_obj)

        params = {
            "diaspora": "1",
            "custom": "ashkenazi",
            "year": today_dt.year,
            "month": today_dt.month,
            "day": today_dt.day,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get("https://www.sefaria.org/api/calendars", params=params, timeout=30.0)
            response.raise_for_status()
            calendar_data = response.json()

        today_iso = today_dt.strftime("%Y-%m-%d")
        virtual_chats: list[dict[str, object]] = []

        total_items = len(calendar_data.get("calendar_items", []))
        logger.info(f"?? CALENDAR DEBUG: Processing {total_items} calendar items")

        for idx, item in enumerate(calendar_data.get("calendar_items", [])):
            title_en = item.get("title", {}).get("en", "")
            ref = item.get("ref")

            logger.info(f"?? ITEM #{idx+1}: title={title_en!r}, ref={ref!r}, has_ref={bool(ref)}")

            if not ref:
                logger.warning(f"?? SKIPPING ITEM #{idx+1}: {title_en!r} - no ref")
                continue

            slug = title_en.lower().replace(" ", "-").replace("(", "").replace(")", "")
            session_id = f"daily-{today_iso}-{slug}"

            try:
                unit_ref, meta = select_today_unit(item, tz=tz_obj, override_today=today_dt)
            except ValueError as exc:
                logger.warning(f"?? SKIPPING ITEM #{idx+1}: {title_en!r} - {exc}")
                continue

            logger.info(
                f"?? ADDING ITEM #{idx+1}: {title_en!r} -> session_id={session_id!r}, unit_ref={unit_ref!r}"
            )

            virtual_chats.append({
                "session_id": session_id,
                "title": title_en,
                "he_title": item.get("title", {}).get("he", ""),
                "display_value": item.get("displayValue", {}).get("en", ""),
                "he_display_value": item.get("displayValue", {}).get("he", ""),
                "ref": unit_ref,
                "category": item.get("category", ""),
                "order": item.get("order", 0),
                "date": today_iso,
                "exists": False,
                "stream": {
                    "stream_id": meta.stream_id,
                    "title": meta.title,
                    "units_total": meta.units_total,
                    "unit_index_today": meta.unit_index_today,
                },
            })

        virtual_chats.sort(key=lambda x: x["order"])

        return {
            "date": today_iso,
            "virtual_chats": virtual_chats,
            "total": len(virtual_chats),
            "next_reset_at": next_midnight(tz_obj, today_dt).isoformat(),
            "seconds_until_reset": seconds_until_next_midnight(tz_obj, today_dt),
            "timezone": tz_obj.key,
        }

    except Exception as e:
        logger.error(f"Failed to get daily calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendar: {str(e)}")


@router.post("/daily/create/{session_id}")
async def create_daily_session_lazy(
    session_id: str,
    tz: Optional[str] = Query(None, description="IANA timezone, e.g. Europe/Amsterdam"),
    session_service: SessionService = Depends(get_session_service),
    current_user=Depends(get_current_user),
):
    """Lazy create daily session when first accessed."""
    from datetime import datetime
    import httpx
    import re
    
    # Check if already exists
    existing_session = await session_service.get_session(str(current_user.id), session_id)
    if existing_session:
        return {"session_id": session_id, "message": "Daily session already exists", "created": False}
    
    # Parse session_id: daily-2025-10-02-daf-yomi
    match = re.match(r'daily-(\d{4}-\d{2}-\d{2})-(.+)', session_id)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid daily session ID format")
    
    date_str, slug = match.groups()
    
    try:
        tz_obj = resolve_timezone(tz, None)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz_obj)

        params = {
            "diaspora": "1",
            "custom": "ashkenazi",
            "year": date_obj.year,
            "month": date_obj.month,
            "day": date_obj.day,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get("https://www.sefaria.org/api/calendars", params=params, timeout=30.0)
            response.raise_for_status()
            calendar_data = response.json()
        
        # Find matching calendar item
        target_item = None
        for item in calendar_data.get("calendar_items", []):
            title_en = item.get("title", {}).get("en", "")
            item_slug = title_en.lower().replace(" ", "-").replace("(", "").replace(")", "")
            if item_slug == slug:
                target_item = item
                break
        
        if not target_item:
            raise HTTPException(status_code=404, detail="Calendar item not found")
        
        unit_ref, meta = select_today_unit(target_item, tz=tz_obj, override_today=date_obj)

        # Create session data
        session_data = {
            "ref": unit_ref,
            "date": date_str,
            "title": target_item.get("title", {}).get("en", ""),
            "he_title": target_item.get("title", {}).get("he", ""),
            "display_value": target_item.get("displayValue", {}).get("en", ""),
            "category": target_item.get("category", ""),
            "order": target_item.get("order", 0),
            "completed": False,
            "created_at": datetime.now(tz_obj).isoformat(),
            "session_type": "daily",
            "stream_id": meta.stream_id,
            "units_total": meta.units_total,
            "unit_index_today": meta.unit_index_today,
            "timezone": tz_obj.key,
        }

        # Save session
        success = await session_service.save_session(str(current_user.id), session_id, session_data, "daily")
        
        if success:
            return {
                "session_id": session_id,
                "ref": unit_ref,
                "title": target_item.get("title", {}).get("en", ""),
                "message": "Daily session created successfully",
                "created": True
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create daily session")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format")
    except Exception as e:
        logger.error(f"Failed to create daily session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create daily session: {str(e)}")

@router.patch("/daily/{session_id}/complete")
async def mark_daily_complete(
    session_id: str,
    completed: bool,
    session_service: SessionService = Depends(get_session_service),
    current_user=Depends(get_current_user),
):
    """Mark daily session as completed or uncompleted."""
    
    # Get existing session
    session = await session_service.get_session(str(current_user.id), session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Daily session not found")
    
    # Update completed status
    session["completed"] = completed
    
    # Save updated session
    success = await session_service.save_session(str(current_user.id), session_id, session, "daily")
    
    if success:
        return {"session_id": session_id, "completed": completed, "message": "Status updated"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update session")

@router.get("/daily/{session_id}/segments")
async def get_daily_segments(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    redis_client = Depends(get_redis_client),
    current_user=Depends(get_current_user),
):
    """Get all loaded segments for a daily session."""
    
    try:
        # Get segments from Redis
        segments_key = f"daily:sess:{session_id}:segments"
        total_key = f"daily:sess:{session_id}:total_segments"
        
        # Get all segments
        segments_data = await redis_client.lrange(segments_key, 0, -1)
        total_segments = await redis_client.get(total_key)
        
        # Parse segments
        segments = []
        total_int = int(total_segments or 0)
        for segment_json in segments_data:
            import json
            segment = json.loads(segment_json)
            metadata = segment.get("metadata") or {}
            merged_metadata = {
                "title": segment.get("title"),
                "indexTitle": segment.get("indexTitle"),
                "heRef": segment.get("heRef"),
            }
            merged_metadata.update({k: v for k, v in metadata.items() if v is not None})
            merged_metadata = {k: v for k, v in merged_metadata.items() if v is not None}

            if total_int > 1:
                denominator = max(1, total_int - 1)
                position = len(segments) / denominator
            else:
                position = 0.0

            segments.append({
                "ref": segment.get("ref"),
                "text": segment.get("en_text", ""),
                "heText": segment.get("he_text", ""),
                "position": float(position),
                "metadata": merged_metadata
            })
        
        return {
            "session_id": session_id,
            "segments": segments,
            "total_segments": total_int,
            "loaded_segments": len(segments)
        }
        
    except Exception as e:
        logger.error(f"Failed to get daily segments for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get segments: {str(e)}")
