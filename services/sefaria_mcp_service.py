import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from fastmcp.client import Client
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.transports import infer_transport
    from fastmcp.exceptions import ToolError
    import mcp.types as mcp_types
    _FASTMCP_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    Client = CallToolResult = infer_transport = ToolError = None  # type: ignore
    mcp_types = None  # type: ignore
    _FASTMCP_AVAILABLE = False

logger = logging.getLogger(__name__)


class SefariaMCPService:
    """Thin async client around the remote Sefaria MCP server."""

    def __init__(self, endpoint: str, timeout: float = 30.0):
        self.endpoint = endpoint
        self.timeout = timeout

        if not _FASTMCP_AVAILABLE:
            raise RuntimeError(
                "fastmcp and mcp packages are required for SefariaMCPService. "
                "Install them or disable the Sefaria MCP integration."
            )
        self._transport = infer_transport(endpoint)
        self._client = Client(self._transport, timeout=timeout)
        self._lock = asyncio.Lock()
        logger.info("Initialized SefariaMCPService", extra={"endpoint": endpoint})

    async def close(self) -> None:
        """Close underlying transport resources."""
        try:
            await self._client.close()
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
            logger.error(
                "Unexpected MCP error",
                extra={"tool": name, "arguments": cleaned_args},
                exc_info=True,
            )
            raise RuntimeError(f"Sefaria MCP tool '{name}' failed: {exc}") from exc

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
