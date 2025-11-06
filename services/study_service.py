import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator, Mapping
from collections import defaultdict

import redis.asyncio as redis
from brain_service.models.study_models import (
    StudyBookshelfRequest, StudyResolveRequest, StudySetFocusRequest, 
    StudyStateResponse, StudyNavigateRequest, StudyWorkbenchSetRequest, 
    StudyChatSetFocusRequest, StudyChatRequest
)
from .sefaria_service import SefariaService
from .sefaria_index_service import SefariaIndexService
from domain.chat.tools import ToolRegistry
from core.llm_config import get_llm_for_task, LLMConfigError
from brain_service.models.doc_v1_models import DocV1
from config.prompts import get_prompt
from config import personalities as personality_service
from .study.config_schema import StudyConfig, load_study_config

from .study_state import (
    get_current_snapshot, replace_top_snapshot, push_new_snapshot, 
    move_cursor, update_local_chat, StudySnapshot, TextDisplay, Bookshelf, BookshelfItem
)
from .study_utils import get_text_with_window, get_bookshelf_for

logger = logging.getLogger(__name__)

class StudyService:
    """
    Service for handling study functionality including state management,
    navigation, workbench operations, and study chat.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        sefaria_service: SefariaService,
        sefaria_index_service: SefariaIndexService,
        tool_registry: ToolRegistry,
        memory_service=None,
        study_config: Optional[Any] = None,
    ):
        self.redis_client = redis_client
        self.sefaria_service = sefaria_service
        self.sefaria_index_service = sefaria_index_service
        self.tool_registry = tool_registry
        self.memory_service = memory_service

        # Configurable parameters for chat history
        self.study_config: Optional[StudyConfig] = None
        self.max_chat_history_messages = 2000
        self.chat_history_ttl_days = 30
        self.update_study_config(study_config)

    def _resolve_study_config(self, study_config: Optional[Any]) -> StudyConfig:
        if isinstance(study_config, StudyConfig):
            return study_config

        if isinstance(study_config, Mapping):
            try:
                return load_study_config(study_config)
            except Exception as exc:
                logger.warning(
                    "Failed to load study config from mapping payload; using defaults: %s",
                    exc,
                    exc_info=True,
                )
        elif study_config is not None:
            logger.warning(
                "Unsupported study_config type %s; using defaults",
                type(study_config),
            )

        try:
            from config import get_config

            raw_config = get_config().get("study", {})
        except Exception as exc:
            logger.warning(
                "Failed to load study config from global config; using defaults: %s",
                exc,
                exc_info=True,
            )
            raw_config = {}

        try:
            return load_study_config(raw_config)
        except Exception as exc:
            logger.warning(
                "Falling back to default study config due to validation error: %s",
                exc,
                exc_info=True,
            )
            return load_study_config({})

    def update_study_config(self, study_config: Optional[Any]) -> None:
        resolved_config = self._resolve_study_config(study_config)
        self.study_config = resolved_config

        chat_history = getattr(resolved_config, "chat_history", None)
        if chat_history:
            self.max_chat_history_messages = chat_history.max_messages
            self.chat_history_ttl_days = chat_history.ttl_days
        else:
            self.max_chat_history_messages = 2000
            self.chat_history_ttl_days = 30

        logger.debug(
            "Study config applied",
            extra={
                "max_chat_history_messages": self.max_chat_history_messages,
                "chat_history_ttl_days": self.chat_history_ttl_days,
            },
        )
    
    async def _ensure_session_owner(self, session_id: str, user_id: str) -> None:
        """
        Ensure that the study session is owned by the specified user.

        If the session has no recorded owner, assign it to the user. Otherwise,
        raise PermissionError when attempting to access a session owned by
        someone else.
        """
        if not self.redis_client:
            return

        owner = await self.redis_client.hget("study:owners", session_id)
        if owner is None:
            await self.redis_client.hset("study:owners", session_id, user_id)
            return

        owner_value = owner.decode() if isinstance(owner, bytes) else str(owner)
        if owner_value != user_id:
            raise PermissionError("Session belongs to another user")
    
    async def _migrate_legacy_history_if_needed(self, session_id: str):
        """Migrate legacy history format to new format if needed."""
        # Use appropriate key prefix for legacy migration
        prefix = "daily" if session_id.startswith('daily-') else "study"
        legacy_key = f"{prefix}:sess:{session_id}:history"
        new_key = f"{prefix}:sess:{session_id}:history_list"
        try:
            if not self.redis_client:
                return
            if await self.redis_client.exists(new_key):
                return
            # Check if legacy key exists and is a string
            key_type = await self.redis_client.type(legacy_key)
            key_type = key_type.decode() if isinstance(key_type, (bytes, bytearray)) else key_type
            if key_type != 'string':
                logger.debug(f"Legacy key {legacy_key} is not a string (type: {key_type}), skip migration")
                return
            
            raw = await self.redis_client.get(legacy_key)
            if not raw:
                return
            try:
                arr = json.loads(raw)
                pipe = self.redis_client.pipeline()
                for item in arr:
                    pipe.rpush(new_key, json.dumps(item, ensure_ascii=False))
                pipe.expire(new_key, 3600 * 24 * self.chat_history_ttl_days)
                await pipe.execute()
                # Optionally delete legacy key: await self.redis_client.delete(legacy_key)
                logger.info(f"Migrated legacy history for {session_id}: {len(arr)} items")
            except json.JSONDecodeError:
                logger.warning(f"Legacy history is corrupted for {session_id}, skip migration")
        except Exception as e:
            logger.error(f"Migrate legacy history failed for {session_id}: {e}")
    
    async def get_state(self, session_id: str, user_id: str) -> StudyStateResponse:
        """Get the entire study state for a session."""
        await self._ensure_session_owner(session_id, user_id)
        snapshot = await get_current_snapshot(session_id, self.redis_client)
        if not snapshot:
            return StudyStateResponse(ok=True, state=None)
        
        # Migrate legacy history if needed
        await self._migrate_legacy_history_if_needed(session_id)
        
        # Update chat_local with messages from the new Redis list format
        chat_history = await self._get_study_chat_history(session_id)
        if chat_history:
            # Convert to the format expected by StudySnapshot with normalization
            from .study_state import ChatMessage
            norm = []
            for m in chat_history:
                m = dict(m)
                if "ts" not in m and "timestamp" in m:
                    # Guarantee float for ts
                    m["ts"] = float(m["timestamp"])
                elif "ts" not in m:
                    # Fallback to current timestamp
                    m["ts"] = datetime.now().timestamp()
                # Remove source to avoid pydantic validation issues
                m.pop("timestamp", None)
                norm.append(ChatMessage(**m))
            snapshot.chat_local = norm
        
        return StudyStateResponse(ok=True, state=snapshot)
    
    async def set_focus(self, request: StudySetFocusRequest, user_id: str) -> StudyStateResponse:
        """Set focus on a specific reference and update study state."""
        await self._ensure_session_owner(request.session_id, user_id)
        logger.info(f"🔥 SET_FOCUS REQUEST: session_id='{request.session_id}', ref='{request.ref}', window_size={request.window_size}")
        try:
            # Check if this is a daily session - use explicit flag
            is_daily_session = request.is_daily if request.is_daily is not None else request.session_id.startswith('daily-')
            logger.info(f"🔥 SESSION TYPE CHECK: is_daily_session={is_daily_session} (from flag: {request.is_daily}, from session_id: {request.session_id.startswith('daily-')})")

            segments: List[Dict[str, Any]] = []
            focus_index = 0
            focus_ref = request.focus_ref or request.ref
            bookshelf_data: Optional[Dict[str, Any]] = None
            window_data: Optional[Dict[str, Any]] = None
            
            if is_daily_session:
                # DAILY MODE: Load full text and segment it
                logger.info(f"🔥 DAILY MODE: Loading full text for {request.ref}")
                from .study_utils import get_full_daily_text, _load_remaining_segments_background
                
                # Save daily reference on first set_focus
                redis_key = f"daily:sess:{request.session_id}:top"
                try:
                    exists = await self.redis_client.exists(redis_key)
                    if not exists:
                        import json
                        await self.redis_client.set(
                            redis_key,
                            json.dumps({"ref": request.ref}, ensure_ascii=False),
                            ex=3600*24*7  # 7 days TTL
                        )
                        logger.info(f"🔥 DAILY REF SAVED: {request.ref}")
                except Exception as e:
                    logger.warning(f"Failed to persist daily top for {request.session_id}: {e}")
                
                # For daily mode, always load the full range, not individual segments
                # Get the original daily reference from the session
                daily_ref = await self._get_daily_reference(request.session_id)
                if daily_ref:
                    logger.info(f"🔥 DAILY MODE: Using original daily ref: {daily_ref}")
                    window_data = await get_full_daily_text(
                        daily_ref, 
                        self.sefaria_service, 
                        self.sefaria_index_service,
                        request.session_id,
                        self.redis_client
                    )
                else:
                    logger.info(f"🔥 DAILY MODE: Using requested ref: {request.ref}")
                    window_data = await get_full_daily_text(
                        request.ref, 
                        self.sefaria_service, 
                        self.sefaria_index_service,
                        request.session_id,
                        self.redis_client
                    )
                
                # Save segments to Redis for polling if this is a daily session
                segments: List[Dict[str, Any]] = []
                focus_index = 0
                focus_ref = request.focus_ref or request.ref

                if window_data:
                    segments = window_data.get("segments") or []
                    focus_index = window_data.get("focusIndex", 0) or 0
                    focus_ref = focus_ref or window_data.get("ref")

                    if segments:
                        # Build list of candidate references to match (specific verse, range start, etc.)
                        candidates: List[str] = []
                        for candidate in [request.focus_ref, request.ref]:
                            if candidate:
                                candidates.append(candidate)
                                if "-" in candidate and ":" in candidate:
                                    candidates.append(candidate.split("-", 1)[0].strip())

                        matched = False
                        for candidate in candidates:
                            normalized_candidate = self._normalize_ref(candidate)
                            if not normalized_candidate:
                                continue
                            for idx, segment in enumerate(segments):
                                if self._normalize_ref(segment.get("ref")) == normalized_candidate:
                                    focus_index = idx
                                    focus_ref = segment.get("ref", focus_ref)
                                    matched = True
                                    break
                            if matched:
                                break

                        # Fallback to the first segment if nothing matched
                        if not matched:
                            focus_index = min(max(focus_index, 0), len(segments) - 1)
                            focus_ref = segments[focus_index].get("ref", focus_ref)

                        window_data["focusIndex"] = focus_index
                    else:
                        focus_ref = focus_ref or window_data.get("ref")
                        window_data["focusIndex"] = focus_index

                if window_data and segments:
                    try:
                        import json
                        segments_key = f"daily:sess:{request.session_id}:segments"
                        
                        # Clear existing segments
                        await self.redis_client.delete(segments_key)
                        
                        # Save each segment to Redis
                        for segment in window_data["segments"]:
                            segment_data = {
                                "ref": segment.get("ref"),
                                "en_text": segment.get("enText") or segment.get("metadata", {}).get("enText", "") or segment.get("text", ""),
                                "he_text": segment.get("heText", ""),
                                "title": segment.get("metadata", {}).get("title", ""),
                                "indexTitle": segment.get("metadata", {}).get("indexTitle", ""),
                                "heRef": segment.get("metadata", {}).get("heRef", "")
                            }
                            await self.redis_client.lpush(segments_key, json.dumps(segment_data, ensure_ascii=False))
                        
                        # Save total segments count
                        total_segments = len(segments)
                        count_key = f"daily:sess:{request.session_id}:total_segments"
                        await self.redis_client.set(count_key, total_segments, ex=3600*24*7)  # 7 days TTL
                        
                        logger.info(f"🔥 SAVED SEGMENTS TO REDIS: {total_segments} segments for session {request.session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save segments to Redis: {e}")
                
                # Start background loading for remaining segments (only if not already loading)
                if window_data and window_data.get("segments") and len(window_data["segments"]) > 1:
                    # Check if background loading is already in progress
                    loading_key = f"daily:sess:{request.session_id}:loading"
                    is_loading = await self.redis_client.get(loading_key)
                    
                    if not is_loading:
                        logger.info(f"🔥 DAILY MODE: Starting background loading for remaining segments")
                        # Set loading flag
                        await self.redis_client.set(loading_key, "true", ex=300)  # 5 minutes TTL
                        
                        # Parse the range to get start/end verses and book chapter
                        ref_parts = (daily_ref or request.ref).split(':')
                        if len(ref_parts) >= 2:
                            book_chapter = ref_parts[0]
                            verse_part = ref_parts[1]
                            # Support different range formats: -, –, —, ..
                            import re
                            range_match = re.search(r'(\d+)[\-\–\—\.]+(\d+)', verse_part)
                            if range_match:
                                start_verse = int(range_match.group(1))
                                end_verse = int(range_match.group(2))
                                import asyncio as async_lib
                                # Calculate how many segments were already loaded
                                total_segments = end_verse - start_verse + 1
                                if total_segments <= 10:
                                    already_loaded = total_segments
                                elif total_segments <= 50:
                                    already_loaded = 10
                                else:
                                    already_loaded = 20
                                
                                async_lib.create_task(_load_remaining_segments_background(
                                    daily_ref or request.ref, 
                                    self.sefaria_service, 
                                    request.session_id,
                                    start_verse,
                                    end_verse,
                                    book_chapter,
                                    self.redis_client,
                                    already_loaded
                                ))
                    else:
                        logger.info(f"🔥 DAILY MODE: Background loading already in progress, skipping")
                # No bookshelf for daily mode
                bookshelf_data = None
                logger.info(f"🔥 DAILY MODE COMPLETE: segments={len(window_data.get('segments', [])) if window_data else 0}")
            else:
                # STUDY MODE: Windowed loading
                logger.info(f"🔥 STUDY MODE: Loading windowed text for {request.ref}")
                window_task = get_text_with_window(
                    request.ref, 
                    self.sefaria_service, 
                    self.sefaria_index_service, 
                    request.window_size
                )
                bookshelf_task = get_bookshelf_for(
                    request.ref, 
                    self.sefaria_service, 
                    self.sefaria_index_service
                )
                window_data, bookshelf_data = await asyncio.gather(window_task, bookshelf_task)
                logger.info(f"🔥 STUDY MODE COMPLETE: window_data={bool(window_data)}, bookshelf_data={bool(bookshelf_data)}")
            
            if not window_data:
                logger.error(f"🔥 NO DATA: ref={request.ref}")
                raise ValueError(f"Reference not found: {request.ref}")

        except Exception as e:
            logger.error(f"🔥 EXCEPTION IN SET_FOCUS: session_id={request.session_id}, ref={request.ref}, error={str(e)}", exc_info=True)
            raise ValueError("Failed to fetch data for the requested reference.")

        # Create snapshot data

        safe_ref = window_data.get('ref') if window_data else request.ref
        safe_focus_index = window_data.get('focusIndex', focus_index) if window_data else focus_index
        safe_segments = segments if segments else (window_data.get('segments') if window_data else None)

        snapshot_data = {
            "segments": safe_segments,
            "focusIndex": safe_focus_index,
            "ref": safe_ref,
            "bookshelf": bookshelf_data,
            "ts": int(datetime.now().timestamp()),
            "discussion_focus_ref": focus_ref or safe_ref
        }
        logger.info(f"🔥 SNAPSHOT_DATA CREATED: segments={len(snapshot_data.get('segments', []))}, ref={snapshot_data.get('ref')}")

        if is_daily_session:
            # DAILY MODE: Load chat history like in study mode
            logger.info(f"🔥 DAILY MODE: Loading chat history")
            try:
                chat_history = await self._get_study_chat_history(request.session_id)
                if chat_history:
                    # Convert to ChatMessage format
                    from .study_state import ChatMessage
                    norm = []
                    for m in chat_history:
                        m = dict(m)
                        if "ts" not in m and "timestamp" in m:
                            # Guarantee float for ts
                            m["ts"] = float(m["timestamp"])
                        elif "ts" not in m:
                            # Fallback to current timestamp
                            m["ts"] = datetime.now().timestamp()
                        # Remove source to avoid pydantic validation issues
                        m.pop("timestamp", None)
                        norm.append(ChatMessage(**m))
                    snapshot_data["chat_local"] = norm
                else:
                    snapshot_data["chat_local"] = []
            except Exception as e:
                logger.error(f"🔥 DAILY CHAT HISTORY LOADING FAILED: {str(e)}")
                snapshot_data["chat_local"] = []
        else:
            # STUDY MODE: Load existing snapshot and chat history
            logger.info(f"🔥 STUDY MODE: Loading existing snapshot")
            current_snapshot = await get_current_snapshot(request.session_id, self.redis_client)
            if current_snapshot:
                try:
                    logger.info(f"🔥 LOADING CHAT HISTORY: session_id={request.session_id}")
                    # Load current chat history from the new format
                    chat_history = await self._get_study_chat_history(request.session_id)
                    logger.info(f"🔥 CHAT HISTORY LOADED: {len(chat_history) if chat_history else 0} messages")
                    
                    if chat_history:
                        # Convert to ChatMessage format
                        from .study_state import ChatMessage
                        norm = []
                        for m in chat_history:
                            m = dict(m)
                            if "ts" not in m and "timestamp" in m:
                                # Guarantee float for ts
                                m["ts"] = float(m["timestamp"])
                            elif "ts" not in m:
                                # Fallback to current timestamp
                                m["ts"] = datetime.now().timestamp()
                            # Remove source to avoid pydantic validation issues
                            m.pop("timestamp", None)
                            norm.append(ChatMessage(**m))
                        snapshot_data["chat_local"] = norm
                    else:
                        # Fallback to old format if new format is empty
                        snapshot_data["chat_local"] = current_snapshot.chat_local
                    
                    if current_snapshot.workbench:
                        snapshot_data["workbench"] = current_snapshot.workbench
                except Exception as e:
                    logger.error(f"🔥 CHAT HISTORY LOADING FAILED: session_id={request.session_id}, error={str(e)}", exc_info=True)
                    # Continue without chat history
                    snapshot_data["chat_local"] = []

        try:
            new_snapshot = StudySnapshot(**snapshot_data)
            logger.info(f"🔥 SNAPSHOT CREATED: session_id={request.session_id}, segments_count={len(snapshot_data.get('segments', []))}")
        except Exception as e:
            logger.error(f"🔥 SNAPSHOT CREATION FAILED: session_id={request.session_id}, error={str(e)}, data_keys={list(snapshot_data.keys())}")
            raise ValueError(f"Failed to create snapshot: {str(e)}")

        if request.navigation_type == 'advance':
            success = await replace_top_snapshot(request.session_id, new_snapshot, self.redis_client)
        else:  # Default to drill_down
            success = await push_new_snapshot(request.session_id, new_snapshot, self.redis_client)

        if not success:
            logger.error(f"🔥 REDIS SAVE FAILED: session_id={request.session_id}")
            raise ValueError("Failed to save state to Redis.")

        logger.info(f"🔥 SNAPSHOT SAVED: session_id={request.session_id}, success={success}")
        return StudyStateResponse(ok=True, state=new_snapshot)
    
    async def _get_daily_reference(self, session_id: str) -> Optional[str]:
        """Get the original daily reference for a daily session."""
        try:
            # Get the session data from Redis
            session_data = await self.redis_client.get(f"daily:sess:{session_id}:top")
            if session_data:
                if isinstance(session_data, (bytes, bytearray)):
                    session_data = session_data.decode('utf-8', 'replace')
                import json
                data = json.loads(session_data)
                return data.get("ref")
        except Exception as e:
            logger.error(f"🔥 ERROR GETTING DAILY REFERENCE: {str(e)}")
        return None
    
    async def navigate_back(self, request: StudyNavigateRequest, user_id: str) -> StudyStateResponse:
        """Move the history cursor back one step and return the state."""
        await self._ensure_session_owner(request.session_id, user_id)
        new_snapshot = await move_cursor(request.session_id, -1, self.redis_client)
        if not new_snapshot:
            return StudyStateResponse(ok=False, error="No previous state available")
        return StudyStateResponse(ok=True, state=new_snapshot)
    
    async def navigate_forward(self, request: StudyNavigateRequest, user_id: str) -> StudyStateResponse:
        """Move the history cursor forward one step and return the state."""
        await self._ensure_session_owner(request.session_id, user_id)
        new_snapshot = await move_cursor(request.session_id, 1, self.redis_client)
        if not new_snapshot:
            return StudyStateResponse(ok=False, error="No next state available")
        return StudyStateResponse(ok=True, state=new_snapshot)
    
    async def set_workbench(self, request: StudyWorkbenchSetRequest, user_id: str) -> StudyStateResponse:
        """Set workbench items for a study session."""
        await self._ensure_session_owner(request.session_id, user_id)
        current_snapshot = await get_current_snapshot(request.session_id, self.redis_client)
        if not current_snapshot:
            return StudyStateResponse(ok=False, error="No current study state found")

        if not request.ref:
            # Clear the specified slot if ref is not provided
            if current_snapshot.workbench is None:
                current_snapshot.workbench = {"left": None, "right": None}
            try:
                current_snapshot.workbench[request.slot] = None
            except Exception as exc:
                logger.warning(f"Failed to assign None to workbench slot {request.slot}: {exc}", exc_info=True)
            success = await replace_top_snapshot(request.session_id, current_snapshot, self.redis_client)
            if not success:
                return StudyStateResponse(ok=False, error="Failed to save workbench state")
            return StudyStateResponse(ok=True, state=current_snapshot)

        # Find the reference in bookshelf or load from Sefaria
        workbench_item = None
        bookshelf_item = None
        
        # First, try to find in current bookshelf
        if current_snapshot.bookshelf and current_snapshot.bookshelf.items:
            for item in current_snapshot.bookshelf.items:
                if item.ref == request.ref:
                    bookshelf_item = item
                    break
        
        # If found in bookshelf, use it as base but load full text if needed
        if bookshelf_item:
            # Check if we have full text, if not load from Sefaria
            if not bookshelf_item.text_full or not bookshelf_item.heTextFull:
                try:
                    text_result = await self.sefaria_service.get_text(request.ref)
                    if text_result.get("ok") and text_result.get("data"):
                        data = text_result["data"]
                        en_text = data.get("en_text", "")
                        he_text = data.get("he_text", "")
                        
                        # Update the bookshelf item with full text
                        bookshelf_item.text_full = en_text
                        bookshelf_item.heTextFull = he_text
                        logger.info(f"Loaded full text for {request.ref}: en_text={len(en_text) if en_text else 0}, he_text={len(he_text) if he_text else 0}")
                    else:
                        logger.warning(f"No text data available for {request.ref}")
                except Exception as e:
                    logger.warning(f"Failed to load text for {request.ref}: {e}")
            
            workbench_item = bookshelf_item
        else:
            # If not in bookshelf, try to load from Sefaria
            try:
                text_result = await self.sefaria_service.get_text(request.ref)
                if text_result.get("ok") and text_result.get("data"):
                    data = text_result["data"]
                    en_text = data.get("en_text", "")
                    he_text = data.get("he_text", "")
                    
                    # Create a new BookshelfItem
                    workbench_item = BookshelfItem(
                        ref=request.ref,
                        commentator="Unknown",
                        indexTitle="Unknown",
                        preview=he_text[:100] if he_text else "",
                        text_full=en_text,
                        heTextFull=he_text
                    )
                    logger.info(f"Created new workbench item for {request.ref}: en_text={len(en_text) if en_text else 0}, he_text={len(he_text) if he_text else 0}")
                else:
                    # Fallback to just storing the ref
                    workbench_item = request.ref
                    logger.warning(f"No text data available for {request.ref}")
            except Exception as e:
                logger.warning(f"Failed to load text for {request.ref}: {e}")
                workbench_item = request.ref

        # Update workbench
        if not current_snapshot.workbench:
            current_snapshot.workbench = {}

        current_snapshot.workbench[request.slot] = workbench_item

        # Save updated snapshot
        success = await replace_top_snapshot(request.session_id, current_snapshot, self.redis_client)
        if not success:
            return StudyStateResponse(ok=False, error="Failed to save workbench state")

        return StudyStateResponse(ok=True, state=current_snapshot)
    
    async def get_bookshelf(self, request: StudyBookshelfRequest, user_id: str) -> Dict[str, Any]:
        """Get bookshelf data for a reference."""
        if request.session_id:
            await self._ensure_session_owner(request.session_id, user_id)
        try:
            bookshelf_data = await get_bookshelf_for(
                request.ref, 
                self.sefaria_service, 
                self.sefaria_index_service,
                limit=800,
                categories=request.categories
            )
            
            # If session_id is provided, update the snapshot with the bookshelf
            if request.session_id:
                snapshot = await get_current_snapshot(request.session_id, self.redis_client)
                if snapshot:
                    try:
                        # bookshelf_data is already a Bookshelf object from get_bookshelf_for
                        snapshot.bookshelf = bookshelf_data
                        await replace_top_snapshot(request.session_id, snapshot, self.redis_client)
                    except Exception as exc:
                        logger.error(f"Failed to attach refreshed bookshelf to snapshot {request.session_id}: {exc}", exc_info=True)
            
            return {"ok": True, "bookshelf": bookshelf_data}
        except Exception as e:
            logger.error(f"Failed to get bookshelf for ref {request.ref}: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}
    
    async def resolve_reference(self, request: StudyResolveRequest) -> Dict[str, Any]:
        """Resolve a book name to a reference."""
        try:
            # Use the SefariaIndexService method instead
            resolved_ref = self.sefaria_index_service.resolve_book_name(request.book_name)
            return {"ok": True, "ref": resolved_ref}
        except Exception as e:
            logger.error(f"Failed to resolve book name {request.book_name}: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}
    
    async def set_chat_focus(self, request: StudyChatSetFocusRequest, user_id: str) -> StudyStateResponse:
        """Set focus for study chat."""
        await self._ensure_session_owner(request.session_id, user_id)
        current_snapshot = await get_current_snapshot(request.session_id, self.redis_client)
        if not current_snapshot:
            return StudyStateResponse(ok=False, error="No current study state found")

        # Update discussion focus
        current_snapshot.discussion_focus_ref = request.ref

        # Save updated snapshot
        success = await replace_top_snapshot(request.session_id, current_snapshot, self.redis_client)
        if not success:
            return StudyStateResponse(ok=False, error="Failed to save chat focus state")

        return StudyStateResponse(ok=True, state=current_snapshot)
    
    async def process_study_chat_stream(
        self, 
        request: StudyChatRequest,
        user_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Process a study chat stream request with context-aware agent selection.
        
        Args:
            request: Study chat request with session and message info
            
        Yields:
            JSON strings with streaming events
        """
        await self._ensure_session_owner(request.session_id, user_id)
        logger.info(f"--- New Study Chat Request ---")
        
        # Get current study state
        current_snapshot = await get_current_snapshot(request.session_id, self.redis_client)
        if not current_snapshot:
            yield json.dumps({"type": "error", "data": {"message": "No study state found"}}) + '\n'
            return

        # Determine agent mode based on selected panel
        if request.selected_panel_id:
            # "Iyun" mode - focused explanation of selected panel
            logger.info(f"Study mode: IYUN (selected panel: {request.selected_panel_id})")
            async for chunk in self._run_iyun_mode(request, current_snapshot):
                yield chunk
        else:
            # "Girsa" mode - general study with tools
            logger.info("Study mode: GIRSA (no panel selected)")
            async for chunk in self._run_girsa_mode(request, current_snapshot):
                yield chunk

    async def _run_iyun_mode(
        self, 
        request: StudyChatRequest, 
        snapshot: StudySnapshot
    ) -> AsyncGenerator[str, None]:
        """Run Iyun mode - focused explanation of selected panel text."""
        try:
            # Get the selected panel text
            panel_text = await self._get_selected_panel_text(request.selected_panel_id, snapshot)
            if not panel_text:
                yield json.dumps({"type": "error", "data": {"message": f"No text found for panel: {request.selected_panel_id}"}}) + '\n'
                return

            # Get system prompt for panel explainer
            system_prompt = get_prompt("study.panel_explainer_system")
            if not system_prompt:
                system_prompt = "You are a helpful study assistant focused on explaining specific texts."

            # Replace placeholders in system prompt with safe string conversion
            system_content = system_prompt.replace("{discussion_ref}", str(panel_text.get('ref', 'Unknown')))
            system_content = system_content.replace("{hebrew_text}", str(panel_text.get('hebrew') or 'Текст не найден'))
            system_content = system_content.replace("{english_text}", str(panel_text.get('english') or 'Translation not available'))

            # Get STM context and inject into system prompt
            stm_context = ""
            if self.memory_service:
                stm = await self.memory_service.get_stm(request.session_id)
                if stm:
                    stm_context = self.memory_service.format_stm_for_prompt(stm)
            
            # Add STM context if available
            if stm_context:
                system_content = f"{system_content}\n\n[STM Context]\n{stm_context}"
            
            # Simple user message with the question
            user_message = f"Пользователь спрашивает: {request.text}"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message}
            ]

            # Stream LLM response
            full_response = ""
            final_doc_v1 = None
            
            async for chunk in self._stream_llm_response(messages, request.session_id):
                yield chunk
                try:
                    event = json.loads(chunk)
                    if event.get("type") == "llm_chunk":
                        chunk_data = event.get("data", {})
                        if isinstance(chunk_data, dict):
                            full_response += chunk_data.get("content", "")
                        else:
                            full_response += str(chunk_data)
                    elif event.get("type") == "doc_v1":
                        # Store the final doc.v1 for saving
                        final_doc_v1 = event.get("data", {})
                except json.JSONDecodeError:
                    pass
            
            # Save messages to chat history
            logger.info(f"Full response length: {len(full_response)}, has doc_v1: {final_doc_v1 is not None}")
            
            # Check if full_response is JSON (doc.v1 format)
            parsed_json = None
            if full_response.strip() and not final_doc_v1:
                try:
                    parsed_json = json.loads(full_response.strip())
                    if isinstance(parsed_json, dict):
                        # Check for doc.v1 format (version + content) or study format (title + summary + paragraphs)
                        if (("version" in parsed_json and "content" in parsed_json) or 
                            ("title" in parsed_json and "summary" in parsed_json and "paragraphs" in parsed_json)):
                            logger.info("Detected structured JSON format in full_response")
                            final_doc_v1 = parsed_json
                except json.JSONDecodeError:
                    pass
            
            # Determine what to save
            if final_doc_v1:
                # Save structured JSON response
                assistant_content = json.dumps(final_doc_v1, ensure_ascii=False)
                await self._save_study_chat_messages(request.session_id, request.text, assistant_content, "doc.v1")
            elif full_response.strip():
                # Save text response
                await self._save_study_chat_messages(request.session_id, request.text, full_response.strip(), "text.v1")
            else:
                logger.warning("No response content to save")
            
            # Update STM after response (write-after-final)
            if self.memory_service and (final_doc_v1 or full_response.strip()):
                # Build recent history for STM update (reuse stored chat history)
                recent_history = await self._get_study_chat_history(request.session_id)
                session_messages: List[Dict[str, Any]] = []
                if recent_history:
                    for entry in recent_history[-10:]:
                        role = entry.get("role") or "assistant"
                        content = entry.get("content")
                        # Ensure content is a simple string for token estimation
                        if isinstance(content, (dict, list)):
                            content = json.dumps(content, ensure_ascii=False)
                        session_messages.append(
                            {
                                "role": role,
                                "content": content or "",
                            }
                        )
                else:
                    assistant_content = json.dumps(final_doc_v1, ensure_ascii=False) if final_doc_v1 else full_response.strip()
                    session_messages = [
                        {"role": "user", "content": request.text},
                        {"role": "assistant", "content": assistant_content}
                    ]

                # Use consider_update_stm which handles all the logic
                updated = await self.memory_service.consider_update_stm(
                    request.session_id, session_messages
                )

                if updated:
                    logger.info("STM updated after panel explainer", extra={
                        "session_id": request.session_id,
                        "message_count": len(session_messages),
                        "token_count": sum(len(str(msg.get("content", ""))) for msg in session_messages) // 4
                    })

        except Exception as e:
            logger.error(f"Error in panel explainer agent: {e}", exc_info=True)
            yield json.dumps({"type": "error", "data": {"message": str(e)}}) + '\n'

    async def _run_girsa_mode(
        self, 
        request: StudyChatRequest, 
        snapshot: StudySnapshot
    ) -> AsyncGenerator[str, None]:
        """Run Girsa mode - general study with tools and broader context."""
        try:
            # Get personality configuration
            agent_id = request.agent_id or "chevruta_talmud"  # Default fallback
            
            try:
                from config.personalities import get_personality
                personality_config = get_personality(agent_id)
                system_prompt = personality_config.get("system_prompt", "") if personality_config else ""
            except Exception as e:
                logger.warning(f"Failed to load personality {agent_id}: {e}")
                personality_config = {}
                system_prompt = ""
            
            if not system_prompt:
                system_prompt = "You are a helpful study partner with access to research tools."

            # Build detailed context from current study state for Girsa mode
            context_parts = []
            
            # Focus reader panel
            if snapshot.ref:
                context_parts.append(f"Focus Reader: {snapshot.ref}")
            
            # Workbench panels - safe access since workbench might be a dict
            if snapshot.workbench:
                # Handle both dict and object access patterns
                if hasattr(snapshot.workbench, 'left'):
                    # Object-style access
                    if snapshot.workbench.left:
                        context_parts.append(f"Left Workbench: {snapshot.workbench.left}")
                    if snapshot.workbench.right:
                        context_parts.append(f"Right Workbench: {snapshot.workbench.right}")
                elif isinstance(snapshot.workbench, dict):
                    # Dict-style access
                    if snapshot.workbench.get('left'):
                        context_parts.append(f"Left Workbench: {snapshot.workbench['left']}")
                    if snapshot.workbench.get('right'):
                        context_parts.append(f"Right Workbench: {snapshot.workbench['right']}")
                else:
                    logger.warning(f"Unexpected workbench type: {type(snapshot.workbench)}")
            
            # Build comprehensive context message
            if context_parts:
                context = f"""Current Study Session Context:
{chr(10).join(context_parts)}

You can see what texts are currently open in the study interface. You have access to tools to research and explore these texts further. Use your tools to provide comprehensive responses."""
            else:
                context = "No texts currently loaded in the study interface. You have access to research tools to help with any study questions."

            # Get STM context and inject into system prompt
            stm_context = ""
            if self.memory_service:
                stm = await self.memory_service.get_stm(request.session_id)
                if stm:
                    stm_context = self.memory_service.format_stm_for_prompt(stm)
            
            # Replace placeholders in system prompt for Girsa mode
            discussion_ref = snapshot.ref or "No text loaded"
            
            # Build truncated context for prompt
            truncated_context_parts = []
            if snapshot.ref:
                truncated_context_parts.append(f"Focus: {snapshot.ref}")
            if snapshot.workbench:
                # Safe access to workbench
                if hasattr(snapshot.workbench, 'left'):
                    if snapshot.workbench.left:
                        truncated_context_parts.append(f"Left: {snapshot.workbench.left}")
                    if snapshot.workbench.right:
                        truncated_context_parts.append(f"Right: {snapshot.workbench.right}")
                elif isinstance(snapshot.workbench, dict):
                    if snapshot.workbench.get('left'):
                        truncated_context_parts.append(f"Left: {snapshot.workbench['left']}")
                    if snapshot.workbench.get('right'):
                        truncated_context_parts.append(f"Right: {snapshot.workbench['right']}")
            
            truncated_context = "\n".join(truncated_context_parts) if truncated_context_parts else "No texts loaded"
            
            # Replace placeholders in system prompt
            system_content = system_prompt.replace("{discussion_ref}", discussion_ref)
            system_content = system_content.replace("{truncated_context}", truncated_context)
            
            # Add STM context if available
            if stm_context:
                system_content = f"{system_content}\n\n[STM Context]\n{stm_context}"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Study context:\n{context}\n\nUser question: {request.text}"}
            ]

            # Stream LLM response with tools
            full_response = ""
            final_doc_v1 = None
            
            async for chunk in self._stream_llm_response(messages, request.session_id):
                yield chunk
                try:
                    event = json.loads(chunk)
                    if event.get("type") == "llm_chunk":
                        chunk_data = event.get("data", {})
                        if isinstance(chunk_data, dict):
                            full_response += chunk_data.get("content", "")
                        else:
                            full_response += str(chunk_data)
                    elif event.get("type") == "doc_v1":
                        # Store the final doc.v1 for saving
                        final_doc_v1 = event.get("data", {})
                except json.JSONDecodeError:
                    pass
            
            # Save messages to chat history
            logger.info(f"Full response length: {len(full_response)}, has doc_v1: {final_doc_v1 is not None}")
            
            # Check if full_response is JSON (doc.v1 format)
            parsed_json = None
            if full_response.strip() and not final_doc_v1:
                try:
                    parsed_json = json.loads(full_response.strip())
                    if isinstance(parsed_json, dict):
                        # Check for doc.v1 format (version + content) or study format (title + summary + paragraphs)
                        if (("version" in parsed_json and "content" in parsed_json) or 
                            ("title" in parsed_json and "summary" in parsed_json and "paragraphs" in parsed_json)):
                            logger.info("Detected structured JSON format in full_response")
                            final_doc_v1 = parsed_json
                except json.JSONDecodeError:
                    pass
            
            # Determine what to save
            if final_doc_v1:
                # Save structured JSON response
                assistant_content = json.dumps(final_doc_v1, ensure_ascii=False)
                await self._save_study_chat_messages(request.session_id, request.text, assistant_content, "doc.v1")
            elif full_response.strip():
                # Save text response
                await self._save_study_chat_messages(request.session_id, request.text, full_response.strip(), "text.v1")
            else:
                logger.warning("No response content to save")
            
            # Update STM after response (write-after-final)
            if self.memory_service and (final_doc_v1 or full_response.strip()):
                # Build recent history for STM update (reuse stored chat history)
                recent_history = await self._get_study_chat_history(request.session_id)
                session_messages: List[Dict[str, Any]] = []
                if recent_history:
                    for entry in recent_history[-10:]:
                        role = entry.get("role") or "assistant"
                        content = entry.get("content")
                        if isinstance(content, (dict, list)):
                            content = json.dumps(content, ensure_ascii=False)
                        session_messages.append(
                            {
                                "role": role,
                                "content": content or "",
                            }
                        )
                else:
                    assistant_content = json.dumps(final_doc_v1, ensure_ascii=False) if final_doc_v1 else full_response.strip()
                    session_messages = [
                        {"role": "user", "content": request.text},
                        {"role": "assistant", "content": assistant_content}
                    ]

                # Use consider_update_stm which handles all the logic
                updated = await self.memory_service.consider_update_stm(
                    request.session_id, session_messages
                )

                if updated:
                    logger.info("STM updated after general chavruta", extra={
                        "session_id": request.session_id,
                        "message_count": len(session_messages),
                        "token_count": sum(len(str(msg.get("content", ""))) for msg in session_messages) // 4
                    })

        except Exception as e:
            logger.error(f"Error in general chavruta agent: {e}", exc_info=True)
            yield json.dumps({"type": "error", "data": {"message": str(e)}}) + '\n'

    async def _get_focused_text(self, snapshot: StudySnapshot) -> Optional[str]:
        """Get the focused text from the current snapshot."""
        logger.info(f"Getting focused text. discussion_focus_ref: {snapshot.discussion_focus_ref}, segments count: {len(snapshot.segments) if snapshot.segments else 0}")
        
        if not snapshot.discussion_focus_ref or not snapshot.segments:
            logger.warning(f"No discussion_focus_ref or segments. discussion_focus_ref: {snapshot.discussion_focus_ref}, segments: {snapshot.segments}")
            return None

        # Normalize the focus ref for comparison
        focus_ref_normalized = self._normalize_ref(snapshot.discussion_focus_ref)
        logger.info(f"Normalized focus ref: '{focus_ref_normalized}'")

        # Find the focused segment
        for segment in snapshot.segments:
            segment_ref_normalized = self._normalize_ref(segment.ref)
            logger.info(f"Checking segment ref: {segment.ref} (normalized: {segment_ref_normalized}) against focus: {snapshot.discussion_focus_ref} (normalized: {focus_ref_normalized})")
            
            if segment_ref_normalized == focus_ref_normalized:
                logger.info(f"Found matching segment with text: {segment.text[:100] if segment.text else 'None'}...")
                return segment.text
        
        logger.warning(f"No matching segment found for discussion_focus_ref: {snapshot.discussion_focus_ref}")
        return None

    def _normalize_ref(self, ref: str) -> str:
        """Normalize a reference for comparison by converting to lowercase and standardizing separators."""
        if not ref:
            return ""
        
        # Convert to lowercase
        normalized = ref.lower().strip()
        
        # Replace dots with colons for consistency
        normalized = normalized.replace('.', ':')
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized

    async def _stream_llm_response(
        self, 
        messages: List[Dict[str, Any]], 
        session_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with tool support."""
        logger.info(f"Starting LLM response stream for session {session_id}")
        logger.info(f"Messages to LLM: {len(messages)} messages")
        
        try:
            client, model, reasoning_params, caps = get_llm_for_task("STUDY")
            logger.info(f"LLM configured: {model}")
        except LLMConfigError as e:
            logger.error(f"LLM not configured: {e}")
            yield json.dumps({"type": "error", "data": {"message": f"LLM not configured: {e}"}}) + '\n'
            return

        tools = self.tool_registry.get_tool_schemas()
        api_params = {**reasoning_params, "model": model, "messages": messages, "stream": True}
        if tools:
            api_params.update({"tools": tools, "tool_choice": "auto"})

        iter_count = 0
        while iter_count < 5:  # Max tool-use iterations
            iter_count += 1
            logger.info(f"LLM iteration {iter_count}, calling {model}")
            stream = await client.chat.completions.create(**api_params)
            
            tool_call_builders = defaultdict(lambda: {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            full_reply_content = ""
            
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_reply_content += delta.content
                    #logger.info(f"LLM chunk {chunk_count}: {delta.content[:50]}...")
                    yield json.dumps({"type": "llm_chunk", "data": delta.content}) + '\n'
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        builder = tool_call_builders[tc.index]
                        if tc.id: 
                            builder["id"] = tc.id
                        if tc.function:
                            if tc.function.name: 
                                builder["function"]["name"] = tc.function.name
                            if tc.function.arguments: 
                                builder["function"]["arguments"] += tc.function.arguments

            if not tool_call_builders:
                # If we already sent chunks, don't send doc_v1 - let the accumulated text be the final result
                if chunk_count > 0:
                    # We already streamed the content as chunks, no need to send doc_v1
                    yield json.dumps({"type": "end", "data": "Stream finished"}) + '\n'
                    return
                
                # Check if the response is a JSON document (doc.v1 format)
                try:
                    parsed_content = json.loads(full_reply_content)
                    if isinstance(parsed_content, dict):
                        # Check for direct doc.v1 format with blocks
                        if ((parsed_content.get("type") == "doc.v1" and "blocks" in parsed_content) or
                            ("blocks" in parsed_content and isinstance(parsed_content["blocks"], list))):
                            yield json.dumps({"type": "doc_v1", "data": parsed_content}) + '\n'
                        # Check for direct doc.v1 format with content (LLM streaming format)
                        elif (parsed_content.get("version") == "doc.v1" and 
                              "content" in parsed_content and isinstance(parsed_content["content"], list)):
                            # Transform to expected format
                            doc_v1_data = {
                                "type": "doc.v1",
                                "blocks": parsed_content["content"]
                            }
                            if "version" in parsed_content:
                                doc_v1_data["version"] = parsed_content["version"]
                            yield json.dumps({"type": "doc_v1", "data": doc_v1_data}) + '\n'
                        else:
                            yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                    else:
                        yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, send as regular text response
                    yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                return

            full_tool_calls = list(tool_call_builders.values())
            messages.append({"role": "assistant", "tool_calls": full_tool_calls, "content": full_reply_content or None})
            
            for tool_call in full_tool_calls:
                function_name = tool_call["function"]["name"]
                raw_args = tool_call["function"].get("arguments") or "{}"

                try:
                    function_args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    # LLM occasionally returns multiple JSON objects concatenated together.
                    # Try to recover by decoding the first valid JSON object.
                    try:
                        decoder = json.JSONDecoder()
                        function_args, _ = decoder.raw_decode(raw_args)
                        logger.warning(
                            "Recovered malformed tool arguments",
                            extra={
                                "tool_name": function_name,
                                "raw_arguments": raw_args,
                                "error": str(exc),
                            },
                        )
                    except json.JSONDecodeError:
                        error_message = (
                            f"Invalid tool arguments for {function_name}: {raw_args} ({exc})"
                        )
                        logger.error(error_message, exc_info=True)
                        yield json.dumps({"type": "error", "data": {"message": error_message}}) + '\n'
                        messages.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({"error": error_message})
                        })
                        continue
                except Exception as exc:
                    error_message = f"Error parsing arguments for tool {function_name}: {exc}"
                    logger.error(error_message, exc_info=True)
                    yield json.dumps({"type": "error", "data": {"message": error_message}}) + '\n'
                    messages.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": error_message})
                    })
                    continue

                try:
                    result = await self.tool_registry.call(function_name, session_id=session_id, **function_args)
                    yield json.dumps({"type": "tool_result", "data": result}) + '\n'
                    messages.append({
                        "tool_call_id": tool_call["id"], 
                        "role": "tool", 
                        "name": function_name, 
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    error_message = f"Error calling tool {function_name}: {e}"
                    logger.error(error_message, exc_info=True)
                    yield json.dumps({"type": "error", "data": {"message": error_message}}) + '\n'
                    messages.append({
                        "tool_call_id": tool_call["id"], 
                        "role": "tool", 
                        "name": function_name, 
                        "content": json.dumps({"error": error_message})
                    })
            
            api_params["messages"] = messages

        # End stream
        yield json.dumps({"type": "end", "data": "Stream finished"}) + '\n'
    
    async def _save_study_chat_messages(
        self, session_id: str, user_message: str,
        assistant_response: str, assistant_content_type: str = "text.v1"
    ):
        """Save chat messages to Redis using the new list-based format."""
        if not self.redis_client:
            logger.warning("No Redis client available, skipping chat history save")
            return

        try:
            # Use appropriate key prefix based on session type
            prefix = "daily" if session_id.startswith('daily-') else "study"
            key = f"{prefix}:sess:{session_id}:history_list"
            timestamp = datetime.now().timestamp()

            # User message
            user_msg = {
                "role": "user",
                "content": user_message,
                "content_type": "text.v1",
                "timestamp": timestamp
            }

            # Assistant message
            assistant_msg = {
                "role": "assistant",
                "content": assistant_response,
                "content_type": assistant_content_type,
                "timestamp": timestamp + 0.001  # Slightly later timestamp
            }
            
            # Add messages to Redis list
            pipe = self.redis_client.pipeline()
            pipe.rpush(key, json.dumps(user_msg, ensure_ascii=False))
            pipe.rpush(key, json.dumps(assistant_msg, ensure_ascii=False))
            
            # Set expiration
            pipe.expire(key, 3600 * 24 * self.chat_history_ttl_days)
            
            # Trim list to max size (keep most recent messages)
            pipe.ltrim(key, -self.max_chat_history_messages, -1)
            
            await pipe.execute()
            
            logger.info(f"Saved chat messages to {key}")

        except Exception as e:
            logger.error(f"Failed to save chat messages for session {session_id}: {e}", exc_info=True)

    async def _get_study_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history from Redis using the new list-based format."""
        if not self.redis_client:
            return []

        try:
            # Use appropriate key prefix based on session type
            prefix = "daily" if session_id.startswith('daily-') else "study"
            key = f"{prefix}:sess:{session_id}:history_list"
            raw_messages = await self.redis_client.lrange(key, 0, -1)
            
            messages = []
            for raw_msg in raw_messages:
                try:
                    msg = json.loads(raw_msg)
                    messages.append(msg)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse message in history: {e}")
                    continue
            
            logger.info(f"Retrieved {len(messages)} messages from {key}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get chat history for session {session_id}: {e}", exc_info=True)
            return []
                
    async def _get_selected_panel_text(self, panel_id: str, snapshot: StudySnapshot) -> Optional[Dict[str, Any]]:
        """
        Get text content for the selected panel.
        
        Args:
            panel_id: "focus", "left_workbench", "right_workbench"
            snapshot: Current study snapshot
            
        Returns:
            Dictionary with ref, hebrew, and english text
        """
        try:
            ref = None
            
            # Determine which reference to use based on panel
            if panel_id == "focus":
                ref = snapshot.ref
            elif panel_id == "left_workbench":
                # Safe access to workbench
                if hasattr(snapshot.workbench, 'left'):
                    ref = snapshot.workbench.left
                elif isinstance(snapshot.workbench, dict):
                    ref = snapshot.workbench.get('left')
            elif panel_id == "right_workbench":
                # Safe access to workbench
                if hasattr(snapshot.workbench, 'right'):
                    ref = snapshot.workbench.right
                elif isinstance(snapshot.workbench, dict):
                    ref = snapshot.workbench.get('right')
            else:
                logger.warning(f"Unknown panel_id: {panel_id}")
                return None
            
            if not ref:
                logger.warning(f"No reference found for panel: {panel_id}")
                return None
            
            # Extract ref string if it's a BookshelfItem object
            ref_string = ref
            if hasattr(ref, 'ref'):
                ref_string = ref.ref
            elif isinstance(ref, dict) and 'ref' in ref:
                ref_string = ref['ref']
            elif not isinstance(ref, str):
                logger.error(f"Invalid ref type for panel {panel_id}: {type(ref)}")
                return None
            
            # Get text from Sefaria
            text_result = await self.sefaria_service.get_text(ref_string)
            if not text_result.get("ok") or not text_result.get("data"):
                logger.error(f"Failed to get text for ref: {ref}")
                return None
            
            data = text_result["data"]
            return {
                "ref": ref_string,
                "hebrew": data.get("he_text") or "",
                "english": data.get("en_text") or ""
            }
            
        except Exception as e:
            logger.error(f"Error getting panel text for {panel_id}: {e}")
            return None
