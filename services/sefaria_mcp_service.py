import asyncio
import inspect
import json
import logging
from typing import Any, Dict, List, Optional

_FASTMCP_IMPORT_ERROR: Optional[BaseException] = None

from typing import TYPE_CHECKING

try:
    from fastmcp.client import Client
    from fastmcp.client.transports import infer_transport
    from fastmcp.exceptions import ToolError
    import mcp.types as mcp_types
    _FASTMCP_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - handled at runtime
    Client = CallToolResult = infer_transport = ToolError = None  # type: ignore
    mcp_types = None  # type: ignore
    _FASTMCP_AVAILABLE = False
    _FASTMCP_IMPORT_ERROR = exc
except Exception as exc:  # pragma: no cover - defensive
    Client = CallToolResult = infer_transport = ToolError = None  # type: ignore
    mcp_types = None  # type: ignore
    _FASTMCP_AVAILABLE = False
    _FASTMCP_IMPORT_ERROR = exc
else:
    try:  # pragma: no cover - version compatibility
        from fastmcp.client.client import CallToolResult as _FastMCPCallToolResult
    except (ImportError, AttributeError):
        try:
            from mcp.types import CallToolResult as _FastMCPCallToolResult  # type: ignore
        except (ImportError, AttributeError):
            _FastMCPCallToolResult = Any  # type: ignore
    CallToolResult = _FastMCPCallToolResult  # type: ignore[misc]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import SimpleNamespace as CallToolResult  # type: ignore[misc]

logger = logging.getLogger(__name__)


class SefariaMCPService:
    """Thin async client around the remote Sefaria MCP server."""

    def __init__(self, endpoint: str, timeout: float = 30.0):
        self.endpoint = endpoint
        self.timeout = timeout

        if not _FASTMCP_AVAILABLE:
            detail = ""
            if _FASTMCP_IMPORT_ERROR:
                detail = f" (import error: {_FASTMCP_IMPORT_ERROR})"
            raise RuntimeError(
                "fastmcp and mcp packages are required for SefariaMCPService. "
                "Install them or disable the Sefaria MCP integration."
                f"{detail}"
            )
        self._transport = infer_transport(endpoint)
        self._timeout_supported = True
        try:
            self._client = Client(self._transport, timeout=timeout)
        except TypeError as exc:
            if "unexpected keyword argument 'timeout'" not in str(exc):
                raise
            logger.debug(
                "fastmcp Client timeout kwarg unsupported; falling back",
                extra={"endpoint": endpoint},
            )
            self._client = Client(self._transport)
            self._timeout_supported = False
        self._lock = asyncio.Lock()
        logger.info("Initialized SefariaMCPService", extra={"endpoint": endpoint})

    async def close(self) -> None:
        """Close underlying transport resources."""
        try:
            closer = getattr(self._client, "close", None)
            if callable(closer):
                result = closer()
                if inspect.isawaitable(result):
                    await result
                return

            async_exit = getattr(self._client, "__aexit__", None)
            if callable(async_exit):
                result = async_exit(None, None, None)
                if inspect.isawaitable(result):
                    await result
                return

            transport_close = getattr(self._transport, "close", None)
            if callable(transport_close):
                result = transport_close()
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to close Sefaria MCP client: %s", exc, exc_info=True)

    async def text_search(
        self,
        query: str,
        filters: Optional[List[str]] = None,
        size: int = 10,
    ) -> Dict[str, Any]:
        payload = await self._call_tool(
            "text_search",
            {
                "query": query,
                "filters": filters,
                "size": size,
            },
        )
        return {
            "ok": True,
            "query": query,
            "filters": filters or [],
            "size": size,
            "data": payload["data"],
            "raw": payload["raw"],
            "source": "sefaria_mcp",
            "tool": "text_search",
        }

    async def english_semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = await self._call_tool(
            "english_semantic_search",
            {
                "query": query,
                "filters": filters,
            },
        )
        return {
            "ok": True,
            "query": query,
            "filters": filters or {},
            "data": payload["data"],
            "raw": payload["raw"],
            "source": "sefaria_mcp",
            "tool": "english_semantic_search",
        }

    async def get_topic_details(
        self,
        topic_slug: str,
        with_links: Optional[bool] = None,
        with_refs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = await self._call_tool(
            "get_topic_details",
            {
                "topic_slug": topic_slug,
                "with_links": with_links,
                "with_refs": with_refs,
            },
        )
        return {
            "ok": True,
            "topic_slug": topic_slug,
            "with_links": with_links,
            "with_refs": with_refs,
            "data": payload["data"],
            "raw": payload["raw"],
            "source": "sefaria_mcp",
            "tool": "get_topic_details",
        }

    async def _call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call MCP tool and normalise the response."""
        cleaned_args = {k: v for k, v in arguments.items() if v is not None}
        logger.debug(
            "Calling Sefaria MCP tool",
            extra={"tool": name, "arguments": cleaned_args},
        )
        try:
            async with self._lock:
                async with self._client as client:
                    result = await client.call_tool(
                        name=name,
                        arguments=cleaned_args,
                        timeout=timeout,
                        raise_on_error=True,
                    )
        except ToolError as exc:
            logger.error(
                "Sefaria MCP tool error",
                extra={"tool": name, "arguments": cleaned_args},
                exc_info=True,
            )
            raise RuntimeError(f"Sefaria MCP tool '{name}' failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - network failures
            # Unwrap ExceptionGroup/BaseExceptionGroup (Python 3.11+) to surface root cause
            inner_messages: List[str] = []
            inner_types: List[str] = []
            try:
                exceptions_attr = getattr(exc, "exceptions", None)
                if exceptions_attr and isinstance(exceptions_attr, list) and exceptions_attr:
                    for sub in exceptions_attr:
                        inner_types.append(type(sub).__name__)
                        inner_messages.append(str(sub))
                elif hasattr(exc, "__cause__") and exc.__cause__ is not None:
                    inner_types.append(type(exc.__cause__).__name__)
                    inner_messages.append(str(exc.__cause__))
                elif hasattr(exc, "__context__") and exc.__context__ is not None:
                    inner_types.append(type(exc.__context__).__name__)
                    inner_messages.append(str(exc.__context__))
            except Exception:
                # Best-effort extraction only
                pass

            extra_details: Dict[str, Any] = {
                "tool": name,
                "arguments": cleaned_args,
                "endpoint": self.endpoint,
                "timeout_sec": self.timeout,
            }
            if inner_types or inner_messages:
                extra_details["inner_exception_types"] = inner_types
                extra_details["inner_exception_messages"] = inner_messages

            logger.error(
                "Unexpected MCP error",
                extra=extra_details,
                exc_info=True,
            )
            detail_suffix = ""
            if inner_types or inner_messages:
                detail_suffix = f" | inner={list(zip(inner_types, inner_messages))}"
            raise RuntimeError(
                f"Sefaria MCP tool '{name}' failed: {exc}{detail_suffix}"
            ) from exc

        return self._normalise_result(result)

    def _normalise_result(self, result: "CallToolResult") -> Dict[str, Any]:
        """Extract structured data or fallback to raw text."""
        if result.data is not None:
            return {"data": result.data, "raw": None}

        text_fragments: List[str] = []
        for block in result.content:
            if mcp_types and isinstance(block, mcp_types.TextContent):
                text_fragments.append(block.text)
            elif hasattr(block, "text"):
                text_fragments.append(getattr(block, "text"))

        raw_text = "\n".join(fragment for fragment in text_fragments if fragment).strip()
        parsed: Any = None
        if raw_text:
            try:
                parsed = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                parsed = None

        return {"data": parsed, "raw": raw_text or None}
