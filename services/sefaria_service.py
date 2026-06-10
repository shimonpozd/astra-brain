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

    async def get_index_raw(self, index_title: str) -> Dict[str, Any]:
        """
        Fetch the raw index payload for a book/commentary from Sefaria.
        """
        cache_key = f"sefaria_index_cache:v1:{index_title}"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as exc:  # pragma: no cover - best-effort
                logger.warning("Sefaria index cache read failed", extra={"slug": index_title, "error": str(exc)})

        try:
            result = await with_retries(
                lambda: get_from_sefaria(
                    self.http_client,
                    f"v2/raw/index/{quote(index_title)}",
                    api_url=self.api_url,
                    api_key=self.api_key,
                )
            )
            if isinstance(result, dict) and result.get("error"):
                payload = {"ok": False, "error": result.get("error"), "data": result}
            else:
                payload = {"ok": isinstance(result, dict), "data": result if isinstance(result, dict) else None}
        except Exception as exc:
            logger.error("Sefaria index fetch failed", extra={"slug": index_title, "error": str(exc)}, exc_info=True)
            payload = {"ok": False, "error": str(exc)}

        if payload.get("ok") and self.redis_client:
            try:
                await self.redis_client.set(cache_key, json.dumps(payload), ex=self.cache_ttl)
            except Exception as exc:  # pragma: no cover - best-effort
                logger.warning("Sefaria index cache write failed", extra={"slug": index_title, "error": str(exc)})

        return payload

    async def get_text(self, tref: str, lang: str | None = None) -> Dict[str, Any]:
        # Legacy parameters for the reliable bilingual endpoint
        legacy_params = {"commentary": 0, "context": 0, "pad": 0}
        if lang:
            legacy_params["lang"] = lang

        # Coerce accidental Talmud-like Bible refs
        try:
            lowered = (tref or "").lower()
            bible_books = ['genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy', 'joshua', 'judges', 'samuel', 'kings', 'isaiah', 'jeremiah', 'ezekiel', 'psalms', 'proverbs', 'job', 'song', 'ruth', 'lamentations', 'ecclesiastes', 'esther', 'daniel', 'ezra', 'nehemiah', 'chronicles']
            if any(book in lowered for book in bible_books):
                m = re.match(r"([\w\s'.]+) (\d+)[ab][\.:](\d+)$", tref, re.IGNORECASE)
                if m:
                    coerced = f"{m.group(1).strip()} {int(m.group(2))}:{int(m.group(3))}"
                    tref = coerced
        except Exception:
            pass

        final_ref = await normalize_tref(tref)
        
        # Cache check
        cache_key = self._cache_key(final_ref, legacy_params)
        if self.redis_client:
            try:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            except Exception as e:
                logger.error(f"Redis cache read failed for key {cache_key}: {e}")

        # PREFER LEGACY API: It is much more reliable for getting both Hebrew and English at once.
        logger.info(f"SEFARIA_SERVICE: Fetching '{final_ref}' using legacy API")
        
        try:
            api_call = lambda: get_from_sefaria(
                self.http_client,
                f"texts/{quote(final_ref)}",
                api_url=self.api_url,
                api_key=self.api_key,
                params=legacy_params,
            )
            raw_result = await with_retries(api_call)
        except Exception as e:
            logger.error(f"Legacy API fetch failed for {final_ref}: {e}")
            raw_result = {"error": str(e)}

        # Fallback to V3 if legacy failed or returned error
        if not raw_result or (isinstance(raw_result, dict) and raw_result.get("error")):
            logger.info(f"SEFARIA_SERVICE: Legacy API failed for {final_ref}, trying V3")
            v3_params = {"context": 0, "pad": 0}
            v3_call = lambda: get_from_sefaria(
                self.http_client, f"v3/texts/{quote(final_ref)}", 
                api_url=self.api_url, api_key=self.api_key, params=v3_params
            )
            raw_result = await with_retries(v3_call)

        # Process the final result
        if isinstance(raw_result, list) and len(raw_result) > 0:
            result = {"ok": True, "data": raw_result}
        elif ok_and_has_text(raw_result):
            # Try to extract the data manually first to avoid CompactText stripping complex structures like Talmud spanning arrays
            raw_text = raw_result.get("text", [])
            raw_he = raw_result.get("he", [])
            
            # Use CompactText as a baseline but manually override the text fields
            try:
                compacted_text = CompactText(raw_result).to_dict_min()
            except Exception as e:
                logger.warning(f"SEFARIA_SERVICE: CompactText parsing failed for {final_ref}: {e}")
                compacted_text = dict(raw_result)
            
            # For backward compatibility and specialized Talmud processing
            if isinstance(raw_text, list):
                compacted_text["text_segments"] = raw_text
                compacted_text["text"] = raw_text
                compacted_text["en_text"] = raw_text # Ensure en_text is populated
            elif isinstance(raw_text, str):
                compacted_text["text"] = raw_text
                compacted_text["en_text"] = raw_text

            if isinstance(raw_he, list):
                compacted_text["he_segments"] = raw_he
                compacted_text["he"] = raw_he
                compacted_text["he_text"] = raw_he # Ensure he_text is populated
            elif isinstance(raw_he, str):
                compacted_text["he"] = raw_he
                compacted_text["he_text"] = raw_he
                
            result = {"ok": True, "data": compacted_text}
        else:
            logger.warning(f"SEFARIA_SERVICE: Fetch FAILED for {final_ref} after fallback.")
            result = {"ok": False, "error": f"Text not found for '{final_ref}'"}

        # Store in cache
        if result["ok"] and self.redis_client:
            try:
                await self.redis_client.set(cache_key, json.dumps(result), ex=self.cache_ttl)
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

    async def get_links_with_text(self, ref: str) -> Dict[str, Any]:
        """
        Fetch links for a reference along with their text content.
        Uses /api/links/{ref}?with_text=1
        """
        norm_ref = await normalize_tref(ref)
        cache_key = f"sefaria_links_text_cache:v1:{norm_ref}"
        
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Sefaria links+text cache read failed for {norm_ref}: {e}")

        try:
            logger.info(f"Fetching links with text for '{norm_ref}'")
            api_call = lambda: get_from_sefaria(
                self.http_client, 
                f"links/{quote(norm_ref)}", 
                api_url=self.api_url, 
                api_key=self.api_key, 
                params={"with_text": 1, "with_sheet_links": 0}
            )
            links = await with_retries(api_call)
            payload = {"ok": True, "data": links if isinstance(links, list) else []}
        except Exception as e:
            logger.error(f"/api/links?with_text=1 call failed for {norm_ref}: {e}", exc_info=True)
            payload = {"ok": False, "error": str(e)}

        if payload["ok"] and self.redis_client:
            try:
                await self.redis_client.set(cache_key, json.dumps(payload), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"Sefaria links+text cache write failed for {norm_ref}: {e}")

        return payload
