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
        # Detect whether call_tool supports 'timeout' kwarg (version compatibility)
        self._call_timeout_supported = True
        try:
            call_sig = inspect.signature(self._client.call_tool)  # type: ignore[attr-defined]
            if "timeout" not in call_sig.parameters:
                self._call_timeout_supported = False
                logger.debug(
                    "fastmcp Client.call_tool timeout kwarg unsupported; will omit",
                    extra={"endpoint": endpoint},
                )
        except Exception:
            # Best-effort; if inspection fails, keep default True and handle at runtime
            self._call_timeout_supported = True
        # Detect whether call_tool supports 'raise_on_error' kwarg (version compatibility)
        self._call_raise_on_error_supported = True
        try:
            call_sig = inspect.signature(self._client.call_tool)  # type: ignore[attr-defined]
            if "raise_on_error" not in call_sig.parameters:
                self._call_raise_on_error_supported = False
                logger.debug(
                    "fastmcp Client.call_tool raise_on_error kwarg unsupported; will omit",
                    extra={"endpoint": endpoint},
                )
        except Exception:
            self._call_raise_on_error_supported = True
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
                    call_kwargs: Dict[str, Any] = {
                        "name": name,
                        "arguments": cleaned_args,
                    }
                    if self._call_raise_on_error_supported:
                        call_kwargs["raise_on_error"] = True
                    if self._call_timeout_supported and timeout is not None:
                        call_kwargs["timeout"] = timeout
                    try:
                        result = await client.call_tool(**call_kwargs)
                    except TypeError as type_exc:
                        # Retry without timeout kwarg if this fastmcp version doesn't support it
                        if "unexpected keyword argument 'timeout'" in str(type_exc):
                            if "timeout" in call_kwargs:
                                call_kwargs.pop("timeout", None)
                                self._call_timeout_supported = False
                                logger.debug(
                                    "Retrying call_tool without timeout kwarg due to TypeError",
                                    extra={"tool": name, "endpoint": self.endpoint},
                                )
                                result = await client.call_tool(**call_kwargs)
                            else:
                                raise
                        elif "unexpected keyword argument 'raise_on_error'" in str(type_exc):
                            if "raise_on_error" in call_kwargs:
                                call_kwargs.pop("raise_on_error", None)
                                self._call_raise_on_error_supported = False
                                logger.debug(
                                    "Retrying call_tool without raise_on_error kwarg due to TypeError",
                                    extra={"tool": name, "endpoint": self.endpoint},
                                )
                                result = await client.call_tool(**call_kwargs)
                            else:
                                raise
                        else:
                            raise
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
        """Extract structured data or fallback to raw text.

        Be compatible with multiple fastmcp/mcp versions:
        - Result objects with .data/.content
        - Plain dicts with 'data' or 'content'
        - Lists (either content blocks or already-structured data)
        """
        # Case 1: modern object with .data
        if hasattr(result, "data"):
            data = getattr(result, "data")
            if data is not None:
                return {"data": data, "raw": None}

        # Helper to extract text from content-like iterables
        def extract_text_blocks(blocks: List[Any]) -> List[str]:
            texts: List[str] = []
            for block in blocks:
                if mcp_types and isinstance(block, mcp_types.TextContent):
                    texts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    # Some versions might return dict blocks
                    text_val = block.get("text")
                    if isinstance(text_val, str):
                        texts.append(text_val)
                elif hasattr(block, "text"):
                    text_attr = getattr(block, "text", None)
                    if isinstance(text_attr, str):
                        texts.append(text_attr)
                elif isinstance(block, str):
                    texts.append(block)
            return texts

        # Case 2: modern object with .content
        if hasattr(result, "content"):
            blocks = getattr(result, "content", None)
            if isinstance(blocks, list):
                text_fragments = extract_text_blocks(blocks)
                raw_text = "\n".join(fragment for fragment in text_fragments if fragment).strip()
                parsed: Any = None
                if raw_text:
                    try:
                        parsed = json.loads(raw_text)
                    except (json.JSONDecodeError, TypeError):
                        parsed = None
                return {"data": parsed, "raw": raw_text or None}

        # Case 3: dict result
        if isinstance(result, dict):
            if "data" in result and result["data"] is not None:
                return {"data": result["data"], "raw": None}
            if "content" in result and isinstance(result["content"], list):
                text_fragments = extract_text_blocks(result["content"])
                raw_text = "\n".join(fragment for fragment in text_fragments if fragment).strip()
                parsed: Any = None
                if raw_text:
                    try:
                        parsed = json.loads(raw_text)
                    except (json.JSONDecodeError, TypeError):
                        parsed = None
                return {"data": parsed, "raw": raw_text or None}
            # Assume already-structured data payload
            return {"data": result, "raw": None}

        # Case 4: list result
        if isinstance(result, list):
            # If it looks like content blocks, try to extract text; otherwise, treat as data
            text_fragments = extract_text_blocks(result)
            if text_fragments:
                raw_text = "\n".join(fragment for fragment in text_fragments if fragment).strip()
                parsed: Any = None
                if raw_text:
                    try:
                        parsed = json.loads(raw_text)
                    except (json.JSONDecodeError, TypeError):
                        parsed = None
                return {"data": parsed, "raw": raw_text or None}
            # Not text-like → assume it's already structured data
            return {"data": result, "raw": None}

        # Fallback: unknown type; provide stringified raw
        try:
            return {"data": None, "raw": str(result)}
        except Exception:
            return {"data": None, "raw": None}
