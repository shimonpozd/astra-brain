# brain_service/services/sefaria_index.py
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def normalize_title(title: str) -> str:
    """Normalizes a title for alias matching."""
    return title.lower().strip()

def _build_aliases_recursive(contents_list: list, aliases_dict: dict):
    """Recursively traverses the 'contents' list to find all books and their titles."""
    for item in contents_list:
        # It's a book if it has a 'title' key.
        if "title" in item:
            canonical_title = item["title"]
            
            # Add the canonical title itself (e.g., "Pesachim")
            aliases_dict[normalize_title(canonical_title)] = canonical_title
            
            # Add the Hebrew title
            if "heTitle" in item:
                aliases_dict[normalize_title(item["heTitle"])] = canonical_title
            
            # Add all other alternative titles listed in the 'titles' array
            for title_obj in item.get("titles", []):
                if "text" in title_obj:
                    aliases_dict[normalize_title(title_obj["text"])] = canonical_title

        # If it's a category, recurse into its contents.
        if "contents" in item:
            _build_aliases_recursive(item["contents"], aliases_dict)

def _find_book_recursive(contents_list: list, canonical_title: str) -> Optional[Dict[str, Any]]:
    """Recursively searches the TOC for a book matching the canonical title."""
    for item in contents_list:
        if item.get("title") == canonical_title:
            return item
        if "contents" in item:
            found = _find_book_recursive(item["contents"], canonical_title)
            if found:
                return found
    return None

def get_book_structure(canonical_title: str, toc_data: Optional[list] = None) -> Optional[Dict[str, Any]]:
    """Finds a book in the TOC and returns its schema and structure."""
    if not toc_data:
        logger.error("Cannot get book structure, TOC data not provided.")
        return None

    book_node = _find_book_recursive(toc_data, canonical_title)
    if not book_node:
        logger.warning(f"Could not find book structure for title: {canonical_title}")
        return None

    # The 'lengths' array holds the number of sections in the highest-order node (e.g., chapters in a book).
    # The 'length' property at the book level holds the total number of chapters.
    # The 'chapters' property (for some texts) holds the detailed breakdown.
    return {
        "schema": book_node.get("schema"),
        "lengths": book_node.get("lengths"),
        "length": book_node.get("length"),
        "categories": book_node.get("categories", []),
        "title": book_node.get("title")
    }

async def resolve_book_name(user_name: str, aliases_data: Optional[dict] = None) -> Optional[str]:
    """Resolves a user-provided book name to a canonical Sefaria title."""
    if not aliases_data:
        logger.error("Cannot resolve book name, aliases data not provided.")
        return None
        
    normalized_name = normalize_title(user_name)

    if normalized_name in aliases_data:
        return aliases_data[normalized_name]

    logger.warning(f"Could not resolve book name: {user_name}. The alias map may be incomplete.")
    return None

BOOKSHELF_CATEGORIES = [
    {"name": "Commentary", "color": "#FF5733"},
    {"name": "Quoting Commentary", "color": "#FFC300"},
    {"name": "Midrash", "color": "#DAF7A6"},
    {"name": "Mishnah", "color": "#C70039"},
    {"name": "Targum", "color": "#900C3F"},
    {"name": "Halakhah", "color": "#581845"},
    {"name": "Responsa", "color": "#2E86C1"},
    {"name": "Chasidut", "color": "#1ABC9C"},
    {"name": "Kabbalah", "color": "#F1C40F"},
    {"name": "Jewish Thought", "color": "#9B59B6"},
    {"name": "Liturgy", "color": "#34495E"},
    {"name": "Bible", "color": "#E67E22"},
    {"name": "Apocrypha", "color": "#7F8C8D"},
    {"name": "Modern Works", "color": "#27AE60"}
]

def get_bookshelf_categories() -> list[dict[str, str]]:
    return BOOKSHELF_CATEGORIES






































