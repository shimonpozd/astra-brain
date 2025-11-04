import logging
from typing import Dict, Any, Optional, List
import httpx

from core.utils import get_from_sefaria

logger = logging.getLogger(__name__)

class SefariaIndexService:
    def __init__(self, http_client: httpx.AsyncClient, sefaria_api_url: str, sefaria_api_key: str | None):
        self.http_client = http_client
        self.api_url = sefaria_api_url
        self.api_key = sefaria_api_key
        self.toc: List[Dict[str, Any]] = []
        self.aliases: Dict[str, str] = {}

    def _normalize_title(self, title: str) -> str:
        return title.lower().strip()

    def _build_aliases_recursive(self, contents_list: list, aliases_dict: dict):
        for item in contents_list:
            if "title" in item:
                canonical_title = item["title"]
                aliases_dict[self._normalize_title(canonical_title)] = canonical_title
                if "heTitle" in item:
                    aliases_dict[self._normalize_title(item["heTitle"])] = canonical_title
                for title_obj in item.get("titles", []):
                    if "text" in title_obj:
                        aliases_dict[self._normalize_title(title_obj["text"])] = canonical_title
            if "contents" in item:
                self._build_aliases_recursive(item["contents"], aliases_dict)

    async def load(self) -> None:
        """Loads the Sefaria table of contents and builds the alias map."""
        logger.info("SefariaIndexService: Loading Sefaria table of contents...")
        toc_data = await get_from_sefaria(
            client=self.http_client,
            endpoint="index",
            api_url=self.api_url,
            api_key=self.api_key
        )
        if toc_data and isinstance(toc_data, list):
            self.toc = toc_data
            logger.info("SefariaIndexService: TOC loaded. Building aliases...")
            aliases = {}
            self._build_aliases_recursive(self.toc, aliases)
            self.aliases = aliases
            logger.info(f"SefariaIndexService: Built {len(self.aliases)} aliases.")
        else:
            logger.error("SefariaIndexService: Failed to load Sefaria table of contents.")

    def resolve_book_name(self, user_name: str) -> Optional[str]:
        normalized_name = self._normalize_title(user_name)
        return self.aliases.get(normalized_name)

    def _find_book_recursive(self, contents_list: list, canonical_title: str) -> Optional[Dict[str, Any]]:
        for item in contents_list:
            if item.get("title") == canonical_title:
                return item
            if "contents" in item:
                found = self._find_book_recursive(item["contents"], canonical_title)
                if found:
                    return found
        return None

    def get_book_structure(self, canonical_title: str) -> Optional[Dict[str, Any]]:
        if not self.toc:
            logger.error("Cannot get book structure, TOC not loaded.")
            return None
        book_node = self._find_book_recursive(self.toc, canonical_title)
        if not book_node:
            return None
        return {
            "schema": book_node.get("schema"),
            "lengths": book_node.get("lengths"),
            "length": book_node.get("length"),
            "categories": book_node.get("categories", []),
            "title": book_node.get("title")
        }

    def get_bookshelf_categories(self) -> list[dict[str, str]]:
        """Returns the curated list of categories for the bookshelf UI."""
        # This list is curated and ordered specifically for the UI.
        return [
            {"name": "Commentary"},
            {"name": "Talmud"},
            {"name": "Halakhah"},
            {"name": "Responsa"},
            {"name": "Mishnah"},
            {"name": "Midrash"},
            {"name": "Jewish Thought"},
            {"name": "Chasidut"},
            {"name": "Kabbalah"},
            {"name": "Modern Works"},
            {"name": "Bible"},
        ]
