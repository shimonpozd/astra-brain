from typing import Awaitable, Callable, Dict, Any, List
import logging
import time
import json
import inspect

logger = logging.getLogger(__name__)

class ToolRegistry:
    """A registry for dynamically calling tools by name and providing their schemas."""

    def __init__(self):
        self._map: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, name: str, handler: Callable[..., Awaitable[Any]], schema: Dict[str, Any]):
        """Register a tool handler function and its schema by name."""
        logger.info(f"Registering tool: '{name}'", extra={
            "tool_name": name,
            "tool_description": schema.get("function", {}).get("description", "No description")
        })
        self._map[name] = handler
        self._schemas.append(schema)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns the list of schemas for all registered tools."""
        return self._schemas

    async def call(self, name: str, session_id: str = None, **kwargs: Any) -> Dict[str, Any]:
        """Call a registered tool by name with keyword arguments."""
        fn = self._map.get(name)
        if not fn:
            logger.error(f"Attempted to call unknown tool: {name}", extra={
                "tool_name": name,
                "session_id": session_id,
                "available_tools": list(self._map.keys())
            })
            return {"ok": False, "error": f"unknown tool: {name}"}
        
        start_time = time.time()
        try:
            # Log tool call start with structured data
            logger.info(f"Tool call started: {name}", extra={
                "tool_name": name,
                "session_id": session_id,
                "tool_args": kwargs,
                "tool_args_count": len(kwargs)
            })
            
            # Add session_id to kwargs only if the function accepts it
            if session_id is not None:
                sig = inspect.signature(fn)
                if 'session_id' in sig.parameters:
                    kwargs['session_id'] = session_id
            result = await fn(**kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log successful tool call completion
            logger.info(f"Tool call completed: {name}", extra={
                "tool_name": name,
                "session_id": session_id,
                "execution_time_ms": round(execution_time * 1000, 2),
                "result_ok": result.get("ok", True) if isinstance(result, dict) else True,
                "result_size": len(str(result)) if result else 0
            })
            
            # Log result preview for debugging (truncated)
            result_preview = str(result)[:200] if result else "None"
            logger.debug(f"Tool result preview: {result_preview}...", extra={
                "tool_name": name,
                "session_id": session_id
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool call failed: {name}", extra={
                "tool_name": name,
                "session_id": session_id,
                "execution_time_ms": round(execution_time * 1000, 2),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }, exc_info=True)
            return {"ok": False, "error": str(e)}