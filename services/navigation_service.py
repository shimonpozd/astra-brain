import json
import logging
from typing import Dict, Any, Optional, List
import redis.asyncio as redis
from brain_service.services.study_state import get_current_snapshot, push_new_snapshot, StudySnapshot
from brain_service.services.sefaria_service import SefariaService

logger = logging.getLogger(__name__)

class NavigationService:
    """Service for LLM-driven navigation and workspace management."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        sefaria_service: SefariaService
    ):
        self.redis_client = redis_client
        self.sefaria_service = sefaria_service
    
    async def navigate_to_ref(self, session_id: str, tref: str, reason: str = "") -> Dict[str, Any]:
        """
        Navigate focus reader to a specific text reference.
        
        Args:
            session_id: Study session ID
            tref: Text reference to navigate to (e.g., "Shabbat 21b:1")
            reason: Optional explanation for the navigation
            
        Returns:
            Success/error status and navigation details
        """
        try:
            # Validate the text reference
            text_data = await self.sefaria_service.get_text(tref)
            if not text_data.get("ok"):
                return {
                    "ok": False, 
                    "error": f"Invalid text reference: {tref}",
                    "action": "navigate_to_ref"
                }
            
            # Get current snapshot
            current_snapshot = await get_current_snapshot(session_id, self.redis_client)
            if not current_snapshot:
                return {
                    "ok": False,
                    "error": "No active study session found",
                    "action": "navigate_to_ref"
                }
            
            # Create new snapshot with updated focus
            new_snapshot = current_snapshot.model_copy(deep=True)
            new_snapshot.ref = tref

            segments = new_snapshot.segments or []
            focus_index = new_snapshot.focusIndex or 0
            normalized_target = self._normalize_ref(tref)

            if segments and normalized_target:
                found = False
                for idx, segment in enumerate(segments):
                    if self._normalize_ref(segment.ref) == normalized_target:
                        focus_index = idx
                        found = True
                        break

                if not found and "-" in tref and ":" in tref:
                    start_candidate = tref.split("-", 1)[0].strip()
                    normalized_start = self._normalize_ref(start_candidate)
                    for idx, segment in enumerate(segments):
                        if self._normalize_ref(segment.ref) == normalized_start:
                            focus_index = idx
                            found = True
                            break

                focus_index = min(max(focus_index, 0), len(segments) - 1)
                new_snapshot.focusIndex = focus_index
                new_snapshot.discussion_focus_ref = segments[focus_index].ref
            else:
                new_snapshot.focusIndex = focus_index
                new_snapshot.discussion_focus_ref = tref
            
            # Save the new snapshot
            success = await push_new_snapshot(session_id, new_snapshot, self.redis_client)
            
            if success:
                logger.info(f"LLM navigated to {tref}", extra={
                    "session_id": session_id,
                    "tref": tref,
                    "reason": reason,
                    "action": "navigate_to_ref"
                })
                
                return {
                    "ok": True,
                    "action": "navigate_to_ref",
                    "tref": tref,
                    "reason": reason,
                    "message": f"Навигация к {tref}" + (f": {reason}" if reason else "")
                }
            else:
                return {
                    "ok": False,
                    "error": "Failed to update study state",
                    "action": "navigate_to_ref"
                }
                
        except Exception as e:
            logger.error(f"Navigation failed for {tref}: {e}", exc_info=True, extra={
                "session_id": session_id,
                "tref": tref
            })
            return {
                "ok": False,
                "error": f"Navigation failed: {str(e)}",
                "action": "navigate_to_ref"
            }
    
    @staticmethod
    def _normalize_ref(ref: Optional[str]) -> str:
        if not ref:
            return ""
        normalized = ref.lower().replace('.', ':')
        normalized = ' '.join(normalized.split())
        return normalized
    
    async def load_commentary_to_workbench(
        self, 
        session_id: str, 
        commentary_ref: str, 
        panel: str, 
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Load a commentary into the specified workbench panel.
        
        Args:
            session_id: Study session ID
            commentary_ref: Commentary reference (e.g., "Rashi on Shabbat 21b:1:1")
            panel: "left" or "right" workbench panel
            reason: Optional explanation for loading the commentary
            
        Returns:
            Success/error status and loading details
        """
        if panel not in ["left", "right"]:
            return {
                "ok": False,
                "error": "Panel must be 'left' or 'right'",
                "action": "load_commentary_to_workbench"
            }
        
        try:
            # Validate the commentary reference
            commentary_data = await self.sefaria_service.get_text(commentary_ref)
            if not commentary_data.get("ok"):
                return {
                    "ok": False,
                    "error": f"Invalid commentary reference: {commentary_ref}",
                    "action": "load_commentary_to_workbench"
                }
            
            # Get current snapshot
            current_snapshot = await get_current_snapshot(session_id, self.redis_client)
            if not current_snapshot:
                return {
                    "ok": False,
                    "error": "No active study session found",
                    "action": "load_commentary_to_workbench"
                }
            
            # Create new snapshot with updated workbench
            new_snapshot = current_snapshot.model_copy(deep=True)
            
            # Ensure workbench exists
            if not new_snapshot.workbench:
                new_snapshot.workbench = {}
            
            # Load commentary into specified panel
            new_snapshot.workbench[panel] = commentary_ref
            
            # Save the new snapshot
            success = await push_new_snapshot(session_id, new_snapshot, self.redis_client)
            
            if success:
                logger.info(f"LLM loaded commentary to {panel} panel", extra={
                    "session_id": session_id,
                    "commentary_ref": commentary_ref,
                    "panel": panel,
                    "reason": reason,
                    "action": "load_commentary_to_workbench"
                })
                
                return {
                    "ok": True,
                    "action": "load_commentary_to_workbench",
                    "commentary_ref": commentary_ref,
                    "panel": panel,
                    "reason": reason,
                    "message": f"Загружен {commentary_ref} в {panel} панель" + (f": {reason}" if reason else "")
                }
            else:
                return {
                    "ok": False,
                    "error": "Failed to update workbench",
                    "action": "load_commentary_to_workbench"
                }
                
        except Exception as e:
            logger.error(f"Failed to load commentary {commentary_ref}: {e}", exc_info=True, extra={
                "session_id": session_id,
                "commentary_ref": commentary_ref,
                "panel": panel
            })
            return {
                "ok": False,
                "error": f"Failed to load commentary: {str(e)}",
                "action": "load_commentary_to_workbench"
            }
    
    async def clear_workbench_panel(self, session_id: str, panel: str) -> Dict[str, Any]:
        """
        Clear the specified workbench panel.
        
        Args:
            session_id: Study session ID
            panel: "left" or "right" workbench panel to clear
            
        Returns:
            Success/error status
        """
        if panel not in ["left", "right"]:
            return {
                "ok": False,
                "error": "Panel must be 'left' or 'right'",
                "action": "clear_workbench_panel"
            }
        
        try:
            # Get current snapshot
            current_snapshot = await get_current_snapshot(session_id, self.redis_client)
            if not current_snapshot:
                return {
                    "ok": False,
                    "error": "No active study session found",
                    "action": "clear_workbench_panel"
                }
            
            # Create new snapshot with cleared panel
            new_snapshot = current_snapshot.model_copy(deep=True)
            
            if new_snapshot.workbench and panel in new_snapshot.workbench:
                del new_snapshot.workbench[panel]
            
            # Save the new snapshot
            success = await push_new_snapshot(session_id, new_snapshot, self.redis_client)
            
            if success:
                logger.info(f"LLM cleared {panel} workbench panel", extra={
                    "session_id": session_id,
                    "panel": panel,
                    "action": "clear_workbench_panel"
                })
                
                return {
                    "ok": True,
                    "action": "clear_workbench_panel",
                    "panel": panel,
                    "message": f"Очищена {panel} панель"
                }
            else:
                return {
                    "ok": False,
                    "error": "Failed to clear workbench panel",
                    "action": "clear_workbench_panel"
                }
                
        except Exception as e:
            logger.error(f"Failed to clear {panel} panel: {e}", exc_info=True, extra={
                "session_id": session_id,
                "panel": panel
            })
            return {
                "ok": False,
                "error": f"Failed to clear panel: {str(e)}",
                "action": "clear_workbench_panel"
            }
