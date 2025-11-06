import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from brain_service.services.user_service import UserService

logger = logging.getLogger(__name__)

class SessionService:
    """
    Service for managing chat and study sessions.
    
    Handles session storage, retrieval, and listing functionality
    for both chat sessions and study sessions.
    """
    
    def __init__(self, redis_client: redis.Redis, user_service: UserService):
        self.redis_client = redis_client
        self.user_service = user_service
        self._study_owner_hash = "study:owners"

    @staticmethod
    def _normalize_user_id(user_id: str) -> tuple[str, uuid.UUID]:
        try:
            user_uuid = uuid.UUID(str(user_id))
        except ValueError as exc:
            raise ValueError(f"Invalid user_id: {user_id}") from exc
        return str(user_uuid), user_uuid

    @staticmethod
    def _chat_key(user_id: str, session_id: str) -> str:
        return f"session:{user_id}:{session_id}"
    
    async def _get_study_owner(self, session_id: str) -> Optional[str]:
        if not self.redis_client:
            return None
        owner = await self.redis_client.hget(self._study_owner_hash, session_id)
        if owner is None:
            return None
        return owner.decode() if isinstance(owner, bytes) else str(owner)

    async def _set_study_owner(self, session_id: str, user_id: str) -> None:
        if not self.redis_client:
            return
        await self.redis_client.hset(self._study_owner_hash, session_id, user_id)

    async def _ensure_study_owner(self, session_id: str, user_id: str) -> bool:
        owner = await self._get_study_owner(session_id)
        if owner is None:
            await self._set_study_owner(session_id, user_id)
            return True
        return owner == user_id
    
    async def get_all_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get chat/study/daily sessions visible to a user.
        """
        if not self.redis_client:
            logger.warning("Redis client not available")
            return []

        user_id_str, user_uuid = self._normalize_user_id(user_id)
        sessions: List[Dict[str, Any]] = []
        seen: set[str] = set()

        # Chat sessions from relational metadata
        threads = await self.user_service.list_threads_for_user(user_uuid)
        for thread in threads:
            entry = {
                "session_id": thread.session_id,
                "name": thread.title or "Chat",
                "last_modified": thread.last_modified.isoformat() if thread.last_modified else None,
                "type": "chat",
            }
            redis_key = self._chat_key(user_id_str, thread.session_id)
            try:
                session_blob = await self.redis_client.get(redis_key)
                if not session_blob:
                    # Fallback: previous format without user prefix.
                    session_blob = await self.redis_client.get(f"session:{thread.session_id}")
                if session_blob:
                    session = json.loads(session_blob)
                    entry["name"] = session.get("name") or entry["name"]
                    entry["last_modified"] = session.get("last_modified", entry["last_modified"])
            except json.JSONDecodeError:
                logger.warning("Failed to decode chat session JSON", extra={"session_id": thread.session_id})

            sessions.append(entry)
            seen.add(thread.session_id)

        # Include any Redis chat sessions that might exist without DB metadata
        try:
            async for key in self.redis_client.scan_iter(f"session:{user_id_str}:*"):
                session_blob = await self.redis_client.get(key)
                if not session_blob:
                    continue
                try:
                    session = json.loads(session_blob)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode session JSON", extra={"key": key})
                    continue
                session_id = session.get("persistent_session_id")
                if not session_id or session_id in seen:
                    continue
                sessions.append({
                    "session_id": session_id,
                    "name": session.get("name", "Chat"),
                    "last_modified": session.get("last_modified"),
                    "type": "chat",
                })
                seen.add(session_id)
        except Exception as exc:
            logger.error("Error occurred while scanning chat sessions for user", extra={"error": str(exc)})

        # Legacy fallback: grab old-format chat sessions if user has no metadata
        if not sessions:
            try:
                async for key in self.redis_client.scan_iter("session:*"):
                    if key.startswith(f"session:{user_id_str}:"):
                        continue
                    session_blob = await self.redis_client.get(key)
                    if not session_blob:
                        continue
                    try:
                        session = json.loads(session_blob)
                    except json.JSONDecodeError:
                        continue
                    stored_user = session.get("user_id")
                    if stored_user != user_id_str:
                        continue
                    session_id = session.get("persistent_session_id")
                    if not session_id or session_id in seen:
                        continue
                    sessions.append({
                        "session_id": session_id,
                        "name": session.get("name", "Chat"),
                        "last_modified": session.get("last_modified"),
                        "type": "chat",
                    })
                    seen.add(session_id)
            except Exception as exc:
                logger.error("Legacy chat scan failed", extra={"error": str(exc)})

        # Study sessions (global)
        try:
            async for key in self.redis_client.scan_iter("study:sess:*:top"):
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) < 4:
                    logger.warning(
                        "Failed to process study session key",
                        extra={"key": key_str},
                    )
                    continue
                session_id = ":".join(parts[2:-1]) or parts[2]

                owner = await self._get_study_owner(session_id)
                if owner is None or owner != user_id_str:
                    continue

                name = "Study Session"
                last_modified_iso: Optional[str] = None

                try:
                    session_blob = await self.redis_client.get(key_str)
                    if session_blob:
                        data = json.loads(session_blob)
                        if isinstance(data, dict):
                            ref = data.get("he_ref") or data.get("ref")
                            if isinstance(ref, str) and ref.strip():
                                name = ref.strip()

                            raw_last_modified = data.get("last_modified") or data.get("ts")
                            if isinstance(raw_last_modified, (int, float)):
                                last_modified_iso = datetime.fromtimestamp(raw_last_modified).isoformat()
                            elif isinstance(raw_last_modified, str) and raw_last_modified.strip():
                                last_modified_iso = raw_last_modified.strip()
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to decode study session snapshot",
                        extra={"key": key},
                    )
                except Exception as exc:
                    logger.warning(
                        "Unexpected error while loading study snapshot",
                        extra={"key": key, "error": str(exc)},
                    )

                sessions.append(
                    {
                        "session_id": session_id,
                        "name": name,
                        "ref": name if name != "Study Session" else None,
                        "last_modified": last_modified_iso or datetime.now().isoformat(),
                        "type": "study",
                    }
                )
        except Exception as exc:
            logger.error("Error occurred while scanning for study sessions", extra={"error": str(exc)})

        # Daily sessions (global for now)
        try:
            daily_keys = [key async for key in self.redis_client.scan_iter("daily:sess:*:top")]
            for key in daily_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) < 4:
                    logger.warning("Failed to process daily session key", extra={"key": key_str})
                    continue
                session_id = ":".join(parts[2:-1]) or parts[2]

                owner = await self._get_study_owner(session_id)
                if owner is None or owner != user_id_str:
                    continue

                session_blob = await self.redis_client.get(key_str)
                if session_blob:
                    try:
                        session = json.loads(session_blob)
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode daily session JSON", extra={"key": key_str})
                        continue
                    sessions.append({
                        "session_id": session_id,
                        "name": session.get("title", "Daily Study"),
                        "last_modified": session.get("last_modified", datetime.now().isoformat()),
                        "type": "daily",
                        "completed": session.get("completed", False),
                    })
                else:
                    sessions.append({
                        "session_id": session_id,
                        "name": "Daily Study",
                        "last_modified": datetime.now().isoformat(),
                        "type": "daily",
                        "completed": False,
                    })
        except Exception as exc:
            logger.error("Error occurred while scanning for daily sessions", extra={"error": str(exc)})

        # Sort by last_modified where available
        sessions_with_dates = [s for s in sessions if s.get("last_modified")]
        sessions_without_dates = [s for s in sessions if not s.get("last_modified")]
        sessions_with_dates.sort(key=lambda item: item["last_modified"], reverse=True)

        return sessions_with_dates + sessions_without_dates
    
    async def get_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get session data for a user. Falls back to legacy study/daily sessions.
        """
        if not self.redis_client:
            return None

        user_id_str, _ = self._normalize_user_id(user_id)

        try:
            # Chat session (new key)
            session_data = await self.redis_client.get(self._chat_key(user_id_str, session_id))
            if not session_data:
                # Legacy key without user component
                session_data = await self.redis_client.get(f"session:{session_id}")

            if session_data:
                session = json.loads(session_data)
                stored_user = session.get("user_id")
                if stored_user in (None, user_id_str):
                    logger.info("Retrieved chat session", extra={"session_id": session_id})
                    return session

            # Study session (global)
            owner = await self._get_study_owner(session_id)
            if owner != user_id_str:
                return None

            study_data = await self.redis_client.get(f"study:sess:{session_id}:top")
            if study_data:
                logger.info("Retrieved study session", extra={"session_id": session_id})
                return json.loads(study_data)

            # Daily session (global)
            daily_data = await self.redis_client.get(f"daily:sess:{session_id}:top")
            if daily_data:
                owner = await self._get_study_owner(session_id)
                if owner != user_id_str:
                    return None
                logger.info("Retrieved daily session", extra={"session_id": session_id})
                return json.loads(daily_data)

            logger.info("Session not found", extra={"session_id": session_id})
            return None

        except json.JSONDecodeError:
            logger.error("Failed to decode session JSON", extra={"session_id": session_id})
            return None
        except Exception as exc:
            logger.error(
                "Error retrieving session",
                extra={"session_id": session_id, "error": str(exc)},
            )
            return None
    
    async def save_session(
        self,
        user_id: Optional[str],
        session_id: str,
        session_data: Dict[str, Any],
        session_type: str = "chat",
    ) -> bool:
        """
        Save session data to Redis.

        Args:
            user_id: Owning user (required for chat sessions).
            session_id: Session identifier.
            session_data: Payload to persist.
            session_type: "chat", "study", or "daily".
        """
        if not self.redis_client:
            return False

        try:
            session_data["last_modified"] = datetime.now().isoformat()

            if session_type == "chat":
                if not user_id:
                    raise ValueError("user_id is required for chat sessions")
                user_id_str, _ = self._normalize_user_id(user_id)
                key = self._chat_key(user_id_str, session_id)
                session_data.setdefault("user_id", user_id_str)
            elif session_type == "study":
                if not user_id:
                    raise ValueError("user_id is required for study sessions")
                user_id_str, _ = self._normalize_user_id(user_id)
                key = f"study:sess:{session_id}:top"
                await self._set_study_owner(session_id, user_id_str)
            elif session_type == "daily":
                if not user_id:
                    raise ValueError("user_id is required for daily sessions")
                user_id_str, _ = self._normalize_user_id(user_id)
                key = f"daily:sess:{session_id}:top"
                await self._set_study_owner(session_id, user_id_str)
            else:
                logger.error("Invalid session type", extra={"session_type": session_type})
                return False

            await self.redis_client.set(key, json.dumps(session_data, ensure_ascii=False))

            logger.info(
                "Session saved",
                extra={"session_id": session_id, "session_type": session_type},
            )
            return True

        except Exception as exc:
            logger.error(
                "Failed to save session",
                extra={
                    "session_id": session_id,
                    "session_type": session_type,
                    "error": str(exc),
                },
            )
            return False
    
    async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a session from Redis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully
        """
        if not self.redis_client:
            return False
        
        try:
            # Try to delete both chat and study session keys
            if user_id:
                user_id_str, _ = self._normalize_user_id(user_id)
                chat_key = self._chat_key(user_id_str, session_id)
            else:
                chat_key = f"session:{session_id}"
            study_key = f"study:sess:{session_id}:top"
            daily_key = f"daily:sess:{session_id}:top"
            owner_cleared = False
            
            deleted_count = 0
            if await self.redis_client.exists(chat_key):
                await self.redis_client.delete(chat_key)
                deleted_count += 1
            
            if await self.redis_client.exists(study_key):
                await self.redis_client.delete(study_key)
                deleted_count += 1
                owner_cleared = True
            
            if await self.redis_client.exists(daily_key):
                await self.redis_client.delete(daily_key)
                deleted_count += 1
                owner_cleared = True

            if owner_cleared:
                await self.redis_client.hdel(self._study_owner_hash, session_id)
            
            if deleted_count > 0:
                logger.info("Session deleted", extra={
                    "session_id": session_id,
                    "keys_deleted": deleted_count
                })
                return True
            else:
                logger.info("Session not found for deletion", extra={"session_id": session_id})
                return False
                
        except Exception as e:
            logger.error("Failed to delete session", extra={
                "session_id": session_id,
                "error": str(e)
            })
            return False
    
    async def get_session_count(self) -> Dict[str, int]:
        """
        Get count of sessions by type.
        
        Returns:
            Dictionary with session counts
        """
        if not self.redis_client:
            return {"chat": 0, "study": 0, "total": 0}
        
        try:
            chat_count = 0
            study_count = 0
            
            # Count chat sessions
            async for _ in self.redis_client.scan_iter("session:*"):
                chat_count += 1
            
            # Count study sessions
            async for _ in self.redis_client.scan_iter("study:sess:*:top"):
                study_count += 1
            
            total_count = chat_count + study_count
            
            return {
                "chat": chat_count,
                "study": study_count,
                "total": total_count
            }
            
        except Exception as e:
            logger.error("Failed to count sessions", extra={"error": str(e)})
            return {"chat": 0, "study": 0, "total": 0}
    
    async def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Clean up old sessions based on last_modified timestamp.
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        if not self.redis_client:
            return 0
        
        try:
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            cleaned_count = 0
            
            # Check chat sessions
            async for key in self.redis_client.scan_iter("session:*"):
                try:
                    session_data = await self.redis_client.get(key)
                    if session_data:
                        session = json.loads(session_data)
                        last_modified = session.get("last_modified")
                        
                        if last_modified:
                            # Parse ISO format timestamp
                            session_time = datetime.fromisoformat(last_modified.replace('Z', '+00:00')).timestamp()
                            if session_time < cutoff_date:
                                await self.redis_client.delete(key)
                                cleaned_count += 1
                                
                except (json.JSONDecodeError, ValueError, TypeError):
                    # If we can't parse the session, it might be corrupted - delete it
                    await self.redis_client.delete(key)
                    cleaned_count += 1
            
            logger.info("Session cleanup completed", extra={
                "cleaned_count": cleaned_count,
                "max_age_days": max_age_days
            })
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup old sessions", extra={
                "error": str(e),
                "max_age_days": max_age_days
            })
            return 0
