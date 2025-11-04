import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import httpx
import json
from dataclasses import dataclass
import unicodedata
import asyncio
import html

logger = logging.getLogger(__name__)

# --- Sefaria API Communication ---

async def get_from_sefaria(
    client: httpx.AsyncClient, 
    endpoint: str, 
    api_url: str,
    api_key: str | None,
    params: dict | None = None
) -> dict | list | None:
    """
    Performs a GET request to a Sefaria API endpoint.
    """
    url = f"{api_url.rstrip('/')}/{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = await client.get(url, params=params, headers=headers, timeout=20.0)
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.info(f"Sefaria API HTTP error for {url}: {e.response.status_code} {e.response.text}")
        else:
            logger.error(f"Sefaria API HTTP error for {url}: {e.response.status_code} {e.response.text}")
        return {"error": f"HTTP error: {e.response.status_code}", "details": e.response.text}
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.error(f"Sefaria API request error for {url}: {e}")
        return {"error": "Request error", "details": str(e)}

async def with_retries(coro_factory, attempts=3, base_delay=0.5):
    delay = base_delay
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:
            if i == attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2

# --- Text Processing and Validation ---

def clamp_lines(s: str, max_lines: int = 8) -> str:
    return "\n".join(s.splitlines()[:max_lines]).strip()

def _clean_html(text: str) -> str:
    """Clean HTML tags and entities from text."""
    if not text:
        return text
    
    # Decode HTML entities like &nbsp; &amp; &lt; &thinsp; etc.
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up multiple spaces, tabs, and other whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def ok_and_has_text(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("error"):
        return False
    return bool(raw.get("text") or raw.get("he") or raw.get("versions"))

_SANITIZE_TRANSLATION = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "ʻ": "'", "´": "'",
    "–": "-", "—": "-",
})

async def normalize_tref(tref: str) -> str:
    if not isinstance(tref, str):
        return tref

    s = unicodedata.normalize('NFKC', tref)
    s = s.replace("\u200f", "").replace("\u200e", "")
    s = s.translate(_SANITIZE_TRANSLATION)
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s+", ", ", s)
    s = re.sub(r"Shulchan Arukh", "Shulchan Aruch", s, flags=re.IGNORECASE)
    s = re.sub(r"De’ah|De´ah|De`ah", "Deah", s, flags=re.IGNORECASE)
    return s

# --- Data Models and Structuring ---

@dataclass
class CompactText:
    ref: str = ""
    heRef: str = ""
    title: str = ""
    indexTitle: str = ""
    type: str = ""
    lang: str = ""
    text: str = ""
    en_text: Optional[str] = None
    he_text: Optional[str] = None

    def __init__(self, raw: dict, preferred_langs: Tuple[str, ...] = ('en', 'he')):
        if not isinstance(raw, dict):
            logger.warning("CompactText received non-dict raw data, initializing empty.")
            return
        self.ref = raw.get("ref", "")
        self.heRef = raw.get("heRef", "")
        self.title = raw.get("title", "")
        self.indexTitle = raw.get("indexTitle", "")
        self.type = raw.get("type", "")
        
        # Initialize text fields
        self.en_text = None
        self.he_text = None

        versions = raw.get("versions", [])
        
        for v in versions:
            if v.get("language") == "en" and v.get("text"):
                raw_text_en = v.get("text", "")
                processed_text_en = "\n".join(map(str, raw_text_en)) if isinstance(raw_text_en, list) else str(raw_text_en)
                self.en_text = clamp_lines(_clean_html(processed_text_en).strip(), max_lines=8)
                break

        for v in versions:
            if v.get("language") == "he" and v.get("text"):
                raw_text_he = v.get("text", "")
                processed_text_he = "\n".join(map(str, raw_text_he)) if isinstance(raw_text_he, list) else str(raw_text_he)
                self.he_text = clamp_lines(_clean_html(processed_text_he).strip(), max_lines=8)
                break

        en_text_raw = raw.get("text")
        if en_text_raw and not self.en_text:
            processed_en = "\n".join(map(str, en_text_raw)) if isinstance(en_text_raw, list) else str(en_text_raw)
            self.en_text = clamp_lines(_clean_html(processed_en).strip(), max_lines=8)

        he_text_raw = raw.get("he")
        if he_text_raw and not self.he_text:
            processed_he = "\n".join(map(str, he_text_raw)) if isinstance(he_text_raw, list) else str(he_text_raw)
            self.he_text = clamp_lines(_clean_html(processed_he).strip(), max_lines=8)

        if self.en_text:
            self.text = self.en_text
            self.lang = 'en'
        elif self.he_text:
            self.text = self.he_text
            self.lang = 'he'
        else:
            self.text = ""
            self.lang = preferred_langs[0] if preferred_langs else ""

    def to_dict_min(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "heRef": self.heRef,
            "title": self.title,
            "indexTitle": self.indexTitle,
            "type": self.type,
            "lang": self.lang,
            "text": self.text,
            "en_text": self.en_text,
            "he_text": self.he_text,
        }

def compact_and_deduplicate_links(raw_links: list, categories: Optional[List[str]], limit: int = 150) -> List[Dict[str, Any]]:
    if not isinstance(raw_links, list):
        return []

    filtered = []
    seen_dedup_keys = set()

    for link in raw_links:
        link_category = link.get("category")
        if categories and link_category not in categories:
            continue

        ref = link.get("ref") or link.get("sourceRef") or link.get("anchorRef")
        if not ref:
            continue

        commentator = link.get("commentator") or \
                      (link.get("collectiveTitle", {}).get("en")) or \
                      link.get("indexTitle")
        
        if not commentator:
            ref_parts = ref.split(" on ")
            if len(ref_parts) > 1:
                commentator = ref_parts[0]
            else:
                continue

        dedup_key = (commentator, ref)
        if dedup_key in seen_dedup_keys:
            continue
        seen_dedup_keys.add(dedup_key)

        filtered.append({
            "ref": ref,
            "heRef": link.get("heRef"),
            "commentator": commentator,
            "indexTitle": link.get("indexTitle", commentator),
            "category": link_category,
            "heCategory": link.get("heCategory"),
            "commentaryNum": link.get("commentaryNum")
        })

    def sort_key(link):
        category_order = {"Commentary": 0, "Midrash": 1, "Halakhah": 2, "Targum": 3}
        cat = link.get("category", "Unknown")
        return (category_order.get(cat, 99), link.get("indexTitle", ""))

    try:
        filtered.sort(key=sort_key)
    except Exception as e:
        logger.error(f"Failed to sort links: {e}")

    return filtered[:limit]