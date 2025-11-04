"""Bookshelf assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class BookshelfItem:
    """Represents a single bookshelf entry."""

    ref: str
    title: str
    preview: Optional[str] = None
    category: Optional[str] = None


class BookshelfService:
    """Placeholder bookshelf service awaiting extraction."""

    def __init__(self, sefaria_service: Any) -> None:
        self._sefaria_service = sefaria_service

    async def get_for(
        self,
        ref: str,
        limit: int = 800,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return a stub payload until the full bookshelf extraction ships."""

        _ = categories  # placeholder until filtering logic is ported
        return {
            "ref": ref,
            "items": [],
            "limit": limit,
        }
