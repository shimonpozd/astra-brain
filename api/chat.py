import json
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, List, Dict, Any

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

CATALOG_PATH = Path(__file__).resolve().parents[2] / 'astra-web-client/public/sefaria-cache/catalog.json'


def _load_catalog_works() -> list[dict]:
    try:
        with CATALOG_PATH.open('r', encoding='utf-8') as catalog_file:
            payload = json.load(catalog_file)
        works = payload.get('works')
        if isinstance(works, list):
            return works
        return []
    except FileNotFoundError:
        logger.warning("Daily catalog missing at %s", CATALOG_PATH)
        return []
    except Exception as exc:
        logger.warning("Failed to read daily catalog %s: %s", CATALOG_PATH, exc)
        return []


class _CatalogReferenceTranslator:
    def __init__(self, works: list[dict]):
        entries: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for work in works:
            seed_short = work.get('seedShorts') or {}
            short_ru = seed_short.get('short_ru')
            if not short_ru:
                continue
            candidates = []
            short_en = seed_short.get('short_en')
            if short_en:
                candidates.append(short_en)
            title = work.get('title')
            if title:
                candidates.append(title)
            for candidate in candidates:
                normalized = (candidate or '').strip()
                if not normalized:
                    continue
                normalized_lower = normalized.lower()
                if normalized_lower in seen:
                    continue
                seen.add(normalized_lower)
                entries.append((normalized, normalized_lower, short_ru))
        self.entries = sorted(entries, key=lambda item: len(item[0]), reverse=True)

    def translate(self, text: str) -> str:
        candidate = (text or '').strip()
        if not candidate or not self.entries:
            return candidate
        lowered = candidate.lower()
        for english, english_lower, russian in self.entries:
            if lowered.startswith(english_lower):
                suffix = candidate[len(english):]
                return f"{russian}{suffix}"
        return candidate


CATALOG_TRANSLATOR = _CatalogReferenceTranslator(_load_catalog_works())

TITLE_TRANSLATIONS = {
    'Parashat Hashavua': 'Недельная глава',
    'Haftarah': 'Афтара',
    'Daf Yomi': 'Даф йоми',
    'Daily Mishnah': 'Мишна йомит',
    'Daily Rambam': 'Рамбам',
    'Daf a Week': 'Даф за неделю',
    'Halakhah Yomit': 'Галаха йомит',
    'Arukh HaShulchan Yomi': 'Арух а-Шульхан йоми',
    'Tanakh Yomi': 'Танах йоми',
    'Chok LeYisrael': 'Хок ле-Исраэль',
    'Tanya Yomi': 'Танья йоми',
    'Yerushalmi Yomi': 'Йерушалми йоми',
    '929': '929',
}
SUFFIX_TRANSLATIONS = {
    '(3 Chapters)': '(3 главы)',
}

def _translate_daily_title(title_en: str) -> str:
    if not title_en:
        return ''
    normalized = title_en.strip()
    for english_key, russian_value in TITLE_TRANSLATIONS.items():
        if normalized == english_key:
            return russian_value
        if normalized.startswith(english_key):
            suffix = normalized[len(english_key):].strip()
            if suffix:
                suffix = SUFFIX_TRANSLATIONS.get(suffix, suffix)
                return f"{russian_value} {suffix}"
            return russian_value
    return normalized


def _translate_catalog_reference(value: Optional[str]) -> str:
    if not value:
        return ''
    return CATALOG_TRANSLATOR.translate(value)


# --- Gamification helpers ---

def _progress_day_key(user_id: str, date_str: str) -> str:
    return f"daily:progress:{user_id}:{date_str}"


def _streak_best_key(user_id: str) -> str:
    return f"daily:streak:best:{user_id}"


WEEKLY_CATEGORIES = {"haftarah", "haftarah", "daf a week", "daf-a-week", "parashat hashavua", "parasha"}


def _is_weekly_category(category: Optional[str]) -> bool:
    if not category:
        return False
    return category.strip().lower() in WEEKLY_CATEGORIES


def _weekly_key(user_id: str, week_id: str) -> str:
    return f"daily:weekly:{user_id}:{week_id}"


async def _load_day(redis_client, user_id: str, date_str: str) -> List[Dict[str, Any]]:
    if not redis_client:
        return []
    raw = await redis_client.get(_progress_day_key(user_id, date_str))
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


async def _save_day(redis_client, user_id: str, date_str: str, entries: List[Dict[str, Any]]) -> None:
    if not redis_client:
        return
    if entries:
        await redis_client.set(
            _progress_day_key(user_id, date_str),
            json.dumps(entries, ensure_ascii=False),
            ex=3600 * 24 * 400,
        )
    else:
        await redis_client.delete(_progress_day_key(user_id, date_str))


async def _record_completion(
    redis_client,
    user_id: str,
    date_str: str,
    session_id: str,
    completed: bool,
    *,
    category: str | None = None,
    category_label: str | None = None,
    ref: str | None = None,
    title: str | None = None,
) -> None:
    entries = await _load_day(redis_client, user_id, date_str)
    entries = [entry for entry in entries if entry.get("session_id") != session_id]
    if completed:
        entries.append(
            {
                "session_id": session_id,
                "category": category,
                "category_label": category_label or category,
                "ref": ref,
                "title": title,
                "ts": datetime.utcnow().isoformat(),
            }
        )
    await _save_day(redis_client, user_id, date_str, entries)

    if _is_weekly_category(category):
        try:
            week_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            week_date = date.today()
        iso = week_date.isocalendar()
        week_id = f"{iso.year}-W{iso.week:02d}"
        week_key = _weekly_key(user_id, week_id)
        if not redis_client:
            return
        if completed:
            await redis_client.set(
                week_key,
                json.dumps(
                    {
                        "session_id": session_id,
                        "category": category,
                        "category_label": category_label or category,
                        "ref": ref,
                        "title": title,
                        "week": week_id,
                    },
                    ensure_ascii=False,
                ),
                ex=3600 * 24 * 400,
            )
        else:
            await redis_client.delete(week_key)


async def _compute_streak(redis_client, user_id: str, today_date: datetime.date) -> Dict[str, int]:
    current = 0
    best = 0

    try:
        if redis_client:
            raw_best = await redis_client.get(_streak_best_key(user_id))
            if raw_best:
                best = int(raw_best)
    except Exception:
        best = 0

    cursor = today_date
    for _ in range(400):
        day_entries = await _load_day(redis_client, user_id, cursor.strftime("%Y-%m-%d"))
        if not day_entries:
            break
        current += 1
        cursor = cursor - timedelta(days=1)

    if current > best and redis_client:
        try:
            await redis_client.set(_streak_best_key(user_id), current, ex=3600 * 24 * 400)
            best = current
        except Exception:
            pass

    return {"current": current, "best": best}


async def _load_progress_range(redis_client, user_id: str, today_date: datetime.date, days: int = 90) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for offset in range(days):
        day = today_date - timedelta(days=offset)
        date_str = day.strftime("%Y-%m-%d")
        entries = await _load_day(redis_client, user_id, date_str)
        items.append({"date": date_str, "entries": entries, "completed": bool(entries)})

    # Overlay weekly completions onto a week anchor (ISO week end, Sunday)
    if redis_client:
        try:
            max_weeks = max(1, days // 7 + 2)
            for w_offset in range(max_weeks):
                start_of_week = today_date - timedelta(days=today_date.isoweekday() - 1)
                week_end = start_of_week - timedelta(days=7 * w_offset) + timedelta(days=6)
                iso = week_end.isocalendar()
                week_id = f"{iso.year}-W{iso.week:02d}"
                raw = await redis_client.get(_weekly_key(user_id, week_id))
                if not raw:
                    continue
                try:
                    weekly_entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                anchor_str = week_end.strftime("%Y-%m-%d")
                existing = next((item for item in items if item["date"] == anchor_str), None)
                if existing:
                    existing_entries = existing.get("entries") or []
                    existing_entries.append({**weekly_entry, "weekly": True})
                    existing["entries"] = existing_entries
                    existing["completed"] = True
                else:
                    items.append({"date": anchor_str, "entries": [{**weekly_entry, "weekly": True}], "completed": True})
        except Exception:
            pass

    items.sort(key=lambda x: x["date"], reverse=True)
    return items

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
            ref = item.get("ref") or item.get("url") or ""
            display_values = item.get("displayValue") or {}
            display_value_en = display_values.get("en") or ref or ""
            display_value_ru = display_values.get("ru") or _translate_catalog_reference(display_value_en)

            logger.info(f"?? ITEM #{idx+1}: title={title_en!r}, ref={ref!r}, has_ref={bool(ref)}")

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
                "title_ru": _translate_daily_title(title_en),
                "he_title": item.get("title", {}).get("he", ""),
                "display_value": display_value_en,
                "he_display_value": display_values.get("he", ""),
                "display_value_ru": display_value_ru,
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
        title_en = target_item.get("title", {}).get("en", "")
        title_ru = _translate_daily_title(title_en)
        session_data = {
            "ref": unit_ref,
            "date": date_str,
            "title": title_en,
            "title_ru": title_ru,
            "he_title": target_item.get("title", {}).get("he", ""),
            "display_value": target_item.get("displayValue", {}).get("en", ""),
            "display_value_ru": _translate_catalog_reference(target_item.get("displayValue", {}).get("en", "")),
            "category": target_item.get("category", ""),
            "category_label": title_ru or title_en,
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
    redis_client = Depends(get_redis_client),
    current_user=Depends(get_current_user),
    tz: Optional[str] = Query(None, description="IANA timezone, e.g. Europe/Amsterdam"),
):
    """Mark daily session as completed or uncompleted."""
    
    # Get existing session
    session = await session_service.get_session(str(current_user.id), session_id)
    if not session:
        # Try to lazily create from today's calendar snapshot for robustness
        try:
            tz_obj = resolve_timezone(tz, None)
            today_dt = now_in_tz(tz_obj)
            calendar_resp = await create_daily_session_lazy(session_id, tz=tz, session_service=session_service, current_user=current_user)  # type: ignore[arg-type]
            session = await session_service.get_session(str(current_user.id), session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Daily session not found")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail="Daily session not found")
    
    # Update completed status
    session["completed"] = completed
    
    # Save updated session
    success = await session_service.save_session(str(current_user.id), session_id, session, "daily")
    
    if success:
        tz_obj = resolve_timezone(tz, None)
        today_dt = now_in_tz(tz_obj)
        date_str = today_dt.strftime("%Y-%m-%d")

        await _record_completion(
            redis_client,
            str(current_user.id),
            date_str,
            session_id,
            completed,
            category=session.get("category"),
            category_label=session.get("category_label") or session.get("title_ru") or session.get("title"),
            ref=session.get("ref"),
            title=session.get("title"),
        )
        streak = await _compute_streak(redis_client, str(current_user.id), today_dt.date())

        return {
            "session_id": session_id,
            "completed": completed,
            "date": date_str,
            "streak": streak,
            "message": "Status updated",
        }
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


@router.get("/daily/progress")
async def get_daily_progress(
    days: int = Query(90, ge=1, le=365),
    tz: Optional[str] = Query(None, description="IANA timezone, e.g. Europe/Amsterdam"),
    redis_client = Depends(get_redis_client),
    current_user=Depends(get_current_user),
):
    """Return recent completion history and streaks for the authenticated user."""
    tz_obj = resolve_timezone(tz, None)
    today_dt = now_in_tz(tz_obj)
    today_date = today_dt.date()

    history = await _load_progress_range(redis_client, str(current_user.id), today_date, days)
    streak = await _compute_streak(redis_client, str(current_user.id), today_date)

    return {
        "today": today_dt.strftime("%Y-%m-%d"),
        "streak": streak,
        "history": history,
    }
