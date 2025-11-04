import logging
import json
import re
from typing import Dict, Any
from urllib.parse import quote

import httpx
import redis.asyncio as redis

from core.utils import (
    CompactText, ok_and_has_text, normalize_tref, with_retries, 
    get_from_sefaria, compact_and_deduplicate_links
)

logger = logging.getLogger(__name__)

class SefariaService:
    def __init__(self, http_client: httpx.AsyncClient, redis_client: redis.Redis, sefaria_api_url: str, sefaria_api_key: str | None, cache_ttl_sec: int = 60):
        self.http_client = http_client
        self.redis_client = redis_client
        self.api_url = sefaria_api_url
        self.api_key = sefaria_api_key
        self.cache_ttl = cache_ttl_sec

    def _cache_key(self, ref: str, params: Dict[str, Any]) -> str:
        param_str = "&".join(sorted(f"{k}={v}" for k, v in params.items()))
        return f"sefaria_cache:v1:{ref}:{param_str}"

    async def get_text(self, tref: str, lang: str | None = None) -> Dict[str, Any]:
        # Request both Hebrew (source) and English (translation) versions
        params = {"version": ["source", "translation"]}
        if lang:
            params["lang"] = lang

        # Coerce accidental Talmud-like Bible refs, e.g. "Genesis 19b.18" -> "Genesis 19:18"
        try:
            lowered = (tref or "").lower()
            bible_books = ['genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy', 'joshua', 'judges', 'samuel', 'kings', 'isaiah', 'jeremiah', 'ezekiel', 'psalms', 'proverbs', 'job', 'song', 'ruth', 'lamentations', 'ecclesiastes', 'esther', 'daniel', 'ezra', 'nehemiah', 'chronicles']
            if any(book in lowered for book in bible_books):
                m = re.match(r"([\w\s'.]+) (\d+)[ab][\.:](\d+)$", tref, re.IGNORECASE)
                if m:
                    coerced = f"{m.group(1).strip()} {int(m.group(2))}:{int(m.group(3))}"
                    logger.debug({"coerced_bible_ref": {"from": tref, "to": coerced}})
                    tref = coerced
        except Exception:
            pass

        # Normalize the reference but don't change the format
        final_ref = await normalize_tref(tref)
        cache_key = self._cache_key(final_ref, params)

        if self.redis_client:
            try:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"Sefaria cache HIT for key: {cache_key}")
                    return json.loads(cached_result)
            except Exception as e:
                logger.error(f"Redis cache read failed for key {cache_key}: {e}")

        logger.info(f"SEFARIA_SERVICE: Attempting fetch for ref: '{final_ref}' with params: {params}")
        
        api_call = lambda: get_from_sefaria(
            self.http_client, f"v3/texts/{quote(final_ref)}", 
            api_url=self.api_url, api_key=self.api_key, params=params
        )
        raw_result = await with_retries(api_call)

        # 3. Process the final result
        if isinstance(raw_result, list) and len(raw_result) > 0:
            # Handle case where Sefaria returns a list of comments/texts
            logger.info(f"SEFARIA_SERVICE: Fetch SUCCEEDED for ref: '{final_ref}' (list of {len(raw_result)} items)")
            # For now, return the list as-is and let the calling code handle it
            result = {"ok": True, "data": raw_result}
        elif ok_and_has_text(raw_result):
            logger.info(f"SEFARIA_SERVICE: Fetch SUCCEEDED for ref: '{final_ref}'")
            compacted_text = CompactText(raw_result).to_dict_min()
            en_text = compacted_text.get('en_text', '')
            he_text = compacted_text.get('he_text', '')
            logger.info(f"SEFARIA_SERVICE: CompactText result - en_text: {bool(en_text)} (len={len(en_text) if en_text else 0}), he_text: {bool(he_text)} (len={len(he_text) if he_text else 0})")
            if he_text:
                logger.info(f"SEFARIA_SERVICE: HE_TEXT PREVIEW: [Hebrew text - {len(he_text)} chars]")
            # Attempt to fetch segmented verses using v2 texts endpoint
            try:
                segment_payload = await get_from_sefaria(
                    self.http_client,
                    f"texts/{quote(final_ref)}",
                    api_url=self.api_url,
                    api_key=self.api_key,
                    params={"commentary": 0, "context": 0, "pad": 0},
                )
            except Exception as seg_exc:  # pragma: no cover - best effort
                logger.warning(
                    "SEFARIA_SERVICE: Segment fetch failed",
                    extra={"ref": final_ref, "error": str(seg_exc)},
                )
                segment_payload = None

            raw_text = None
            raw_he = None
            if isinstance(segment_payload, dict):
                raw_text = segment_payload.get("text")
                raw_he = segment_payload.get("he")
            else:
                raw_text = raw_result.get("text")
                raw_he = raw_result.get("he")

            if isinstance(raw_text, list):
                compacted_text["text_segments"] = raw_text
            if isinstance(raw_he, list):
                compacted_text["he_segments"] = raw_he
            result = {"ok": True, "data": compacted_text}
        else:
            logger.warning(f"SEFARIA_SERVICE: Fetch FAILED for {final_ref} after all fallbacks.")
            result = {"ok": False, "error": f"Text not found for '{final_ref}'"}

        # 4. Store in cache if successful
        if result["ok"] and self.redis_client:
            # Use the original cache key (without 'he') to store the successful result
            try:
                await self.redis_client.set(cache_key, json.dumps(result), ex=self.cache_ttl)
                logger.info(f"Sefaria cache WRITE for key: {cache_key}")
            except Exception as e:
                logger.error(f"Redis cache write failed for key {cache_key}: {e}")

        return result

    async def get_related_links(self, ref: str, categories: list[str] | None = None, limit: int = 120) -> Dict[str, Any]:
        norm_ref = await normalize_tref(ref)
        links = []
        try:
            logger.info(f"Fetching related links for '{norm_ref}' via /api/links/")
            api_call = lambda: get_from_sefaria(
                self.http_client, f"links/{quote(norm_ref)}", 
                api_url=self.api_url, api_key=self.api_key, params={"with_text": 0, "with_sheet_links": 0}
            )
            l = await with_retries(api_call)
            links = l if isinstance(l, list) else l.get("links", [])
        except Exception as e:
            logger.error(f"/api/links call failed for {norm_ref}: {e}", exc_info=True)

        if not links:
            logger.info(f"/api/links returned no data, falling back to /api/related for '{norm_ref}'")
            try:
                api_call = lambda: get_from_sefaria(
                    self.http_client, f"related/{quote(norm_ref)}", 
                    api_url=self.api_url, api_key=self.api_key
                )
                r = await with_retries(api_call)
                links = (r or {}).get("links") or []
            except Exception as e:
                logger.error(f"/api/related call failed for {norm_ref}: {e}", exc_info=True)

        if not categories:
            cats = ['Commentary', 'Midrash', 'Halakhah', 'Targum', 'Philosophy', 'Liturgy', 'Kabbalah', 'Tanaitic', 'Modern Commentary']
        else:
            cats = categories

        compacted = compact_and_deduplicate_links(links, categories=cats, limit=limit)
        return {"ok": True, "data": compacted}
