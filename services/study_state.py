# brain_service/services/study_state.py
import json
import logging
import datetime
import asyncio
from typing import Dict, Any, Optional, List, Union

import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Constants ---
SESSION_TTL_DAYS = 30

# --- Redis Key Schemas ---
def _history_key(session_id: str) -> str:
    if session_id.startswith('daily-'):
        return f"daily:sess:{session_id}:history"
    return f"study:sess:{session_id}:history"

def _cursor_key(session_id: str) -> str:
    if session_id.startswith('daily-'):
        return f"daily:sess:{session_id}:cursor"
    return f"study:sess:{session_id}:cursor"

def _top_key(session_id: str) -> str:
    if session_id.startswith('daily-'):
        return f"daily:sess:{session_id}:top"
    return f"study:sess:{session_id}:top"

# --- Data Models ---
class TextSegmentMetadata(BaseModel):
    verse: Optional[int] = None
    chapter: Optional[int] = None
    page: Optional[str] = None
    line: Optional[int] = None
    title: Optional[str] = None
    indexTitle: Optional[str] = None

class TextSegment(BaseModel):
    ref: str
    text: str
    heText: str
    position: float
    metadata: TextSegmentMetadata

class TextDisplay(BaseModel):
    segments: List[TextSegment]
    focusIndex: int
    ref: str
    he_ref: Optional[str] = None

class BookshelfItem(BaseModel):
    ref: str
    heRef: Optional[str] = None
    commentator: str
    indexTitle: str
    category: Optional[str] = None
    heCategory: Optional[str] = None
    commentaryNum: Optional[Any] = None
    score: Optional[float] = None
    preview: str
    text_full: Optional[str] = None
    heTextFull: Optional[str] = None
    title: Optional[str] = None
    heTitle: Optional[str] = None

class Bookshelf(BaseModel):
    counts: Dict[str, int]
    items: List[BookshelfItem]

class ChatMessage(BaseModel):
    role: str
    content: Union[str, Dict[str, Any]]
    content_type: str = "text.v1"
    timestamp: Optional[str] = None

class StudySnapshot(BaseModel):
    segments: Optional[List[TextSegment]] = None
    focusIndex: Optional[int] = None
    ref: Optional[str] = None
    bookshelf: Optional[Bookshelf] = None
    chat_local: List[ChatMessage] = Field(default_factory=list)
    ts: int
    workbench: Dict[str, Optional[Union[TextDisplay, BookshelfItem, str]]] = Field(default_factory=lambda: {"left": None, "right": None})
    discussion_focus_ref: Optional[str] = None
    stream: Optional[Dict[str, Any]] = None
    timezone: Optional[str] = None

# --- Core State Functions (Refactored) ---

async def get_current_snapshot(session_id: str, redis_client: redis.Redis) -> Optional[StudySnapshot]:
    """Retrieves the current snapshot from the cache (:top key)."""
    logger.info(f"🔥 GET_CURRENT_SNAPSHOT START: session_id={session_id}")
    if not redis_client:
        logger.warning("Redis client not available.")
        return None
    
    top_key = _top_key(session_id)
    logger.info(f"🔥 GET_CURRENT_SNAPSHOT KEY: {top_key}")
    snapshot_json = await redis_client.get(top_key)
    logger.info(f"🔥 GET_CURRENT_SNAPSHOT RESULT: {bool(snapshot_json)}")
    
    if snapshot_json:
        logger.info(f"🔥 GET_CURRENT_SNAPSHOT SUCCESS: Found snapshot")
        return StudySnapshot(**json.loads(snapshot_json))
    logger.warning(f"STUDY_STATE: No snapshot found for key: {top_key}")
    return None

async def push_new_snapshot(session_id: str, snapshot: StudySnapshot, redis_client: redis.Redis) -> bool:
    """Pushes a new snapshot, trimming forward history, and updates top/cursor."""
    if not redis_client:
        return False

    history_key = _history_key(session_id)
    cursor_key = _cursor_key(session_id)
    top_key = _top_key(session_id)

    try:
        cursor_str = await redis_client.get(cursor_key)
        cursor = int(cursor_str) if cursor_str is not None else -1

        if cursor > -1:
            await redis_client.ltrim(history_key, 0, cursor)

        snapshot_json = json.dumps(snapshot.model_dump())
        await redis_client.rpush(history_key, snapshot_json)

        new_cursor = await redis_client.llen(history_key) - 1
        
        async with redis_client.pipeline() as pipe:
            pipe.set(cursor_key, new_cursor)
            pipe.set(top_key, snapshot_json)
            
            ttl_seconds = SESSION_TTL_DAYS * 24 * 60 * 60
            pipe.expire(history_key, ttl_seconds)
            pipe.expire(cursor_key, ttl_seconds)
            pipe.expire(top_key, ttl_seconds)
            
            await pipe.execute()

        logger.info(f"Pushed new snapshot for session '{session_id}' at index {new_cursor}.")
        return True

    except Exception as e:
        logger.error(f"Failed to push snapshot for session '{session_id}': {e}", exc_info=True)
        return False

async def move_cursor(session_id: str, direction: int, redis_client: redis.Redis) -> Optional[StudySnapshot]:
    """Moves the cursor back (-1) or forward (+1) and returns the new top snapshot."""
    if not redis_client or direction not in [-1, 1]:
        return None

    history_key = _history_key(session_id)
    cursor_key = _cursor_key(session_id)
    top_key = _top_key(session_id)

    try:
        cursor, history_len = await asyncio.gather(
            redis_client.get(cursor_key),
            redis_client.llen(history_key)
        )
        cursor = int(cursor) if cursor is not None else -1

        new_cursor = cursor + direction

        if not (0 <= new_cursor < history_len):
            logger.warning(f"Cannot move cursor for session '{session_id}'. Current: {cursor}, Attempted: {new_cursor}, History: {history_len}")
            return None

        new_snapshot_json = await redis_client.lindex(history_key, new_cursor)
        if not new_snapshot_json:
            logger.error(f"Mismatch between history length and lindex for session '{session_id}'.")
            return None

        async with redis_client.pipeline() as pipe:
            pipe.set(cursor_key, new_cursor)
            pipe.set(top_key, new_snapshot_json)
            await pipe.execute()
        
        logger.info(f"Moved cursor for session '{session_id}' to index {new_cursor}.")
        return StudySnapshot(**json.loads(new_snapshot_json))

    except Exception as e:
        logger.error(f"Failed to move cursor for session '{session_id}': {e}", exc_info=True)
        return None

async def restore_by_index(session_id: str, index: int, redis_client: redis.Redis) -> Optional[StudySnapshot]:
    """Sets the cursor to a specific index without trimming history."""
    if not redis_client:
        return None

    history_key = _history_key(session_id)
    cursor_key = _cursor_key(session_id)
    top_key = _top_key(session_id)

    try:
        history_len = await redis_client.llen(history_key)
        if not (0 <= index < history_len):
            logger.warning(f"Invalid index for restore: {index}. History length: {history_len}")
            return None

        snapshot_json = await redis_client.lindex(history_key, index)
        if not snapshot_json:
            return None

        await redis_client.set(cursor_key, index)
        await redis_client.set(top_key, snapshot_json)

        logger.info(f"Restored session '{session_id}' to index {index}.")
        return StudySnapshot(**json.loads(snapshot_json))

    except Exception as e:
        logger.error(f"Failed to restore by index for session '{session_id}': {e}", exc_info=True)
        return None

async def update_local_chat(session_id: str, new_messages: List[Dict[str, str]], redis_client: redis.Redis) -> bool:
    """Appends messages to the local_chat of the current snapshot."""
    if not redis_client:
        return False

    top_key = _top_key(session_id)
    history_key = _history_key(session_id)
    cursor_key = _cursor_key(session_id)

    try:
        snapshot_json, cursor_str = await asyncio.gather(
            redis_client.get(top_key),
            redis_client.get(cursor_key)
        )

        if not snapshot_json or cursor_str is None:
            logger.warning(f"Cannot update chat for session '{session_id}', no active snapshot.")
            return False

        snapshot = StudySnapshot(**json.loads(snapshot_json))
        cursor = int(cursor_str)

        if snapshot.chat_local is None:
            snapshot.chat_local = []
        for msg in new_messages:
            snapshot.chat_local.append(ChatMessage(**msg))

        updated_snapshot_json = json.dumps(snapshot.model_dump())

        async with redis_client.pipeline() as pipe:
            pipe.set(top_key, updated_snapshot_json)
            pipe.lset(history_key, cursor, updated_snapshot_json)
            await pipe.execute()
        return True

    except Exception as e:
        logger.error(f"Failed to update local chat for session '{session_id}': {e}", exc_info=True)
        return False

async def replace_top_snapshot(session_id: str, snapshot: StudySnapshot, redis_client: redis.Redis) -> bool:
    """Replaces the most recent snapshot in history with a new one."""
    if not redis_client:
        return False

    history_key = _history_key(session_id)
    cursor_key = _cursor_key(session_id)
    top_key = _top_key(session_id)

    try:
        cursor_str = await redis_client.get(cursor_key)
        if cursor_str is None:
            return await push_new_snapshot(session_id, snapshot, redis_client)
        
        cursor = int(cursor_str)
        snapshot.ts = int(datetime.datetime.now().timestamp())
        snapshot_json = json.dumps(snapshot.model_dump())

        async with redis_client.pipeline() as pipe:
            pipe.lset(history_key, cursor, snapshot_json)
            pipe.set(top_key, snapshot_json)
            await pipe.execute()

        logger.info(f"Replaced snapshot for session '{session_id}' at index {cursor}.")
        return True

    except Exception as e:
        logger.error(f"Failed to replace snapshot for session '{session_id}': {e}", exc_info=True)
        return False
