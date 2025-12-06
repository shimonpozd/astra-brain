import json
import logging
import re
from typing import Dict, Any, List, Optional
from urllib.parse import quote, urljoin
import asyncio

import httpx
import redis.asyncio as redis
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ALLOWED_ARTICLE_TAGS = {"p", "h2", "h3", "ul", "li", "blockquote", "img", "small", "a"}
STRIP_SELECTORS = [
    "table.vertical-navbox",
    "table.navbox",
    "table.metadata",
    "div.hatnote",
    "div.reflist",
    "ol.references",
    "div.mw-references-wrap",
    "div.toc",
    "div.thumb",
    "div.gallery",
    "div.noprint",
    "nav",
    "header",
    "footer",
    "form",
    "button",
    "span.mw-editsection",
    "span.mw-cite-backlink",
    "sup.reference",
    "script",
    "style",
    "noscript",
]


class WikiService:
    """
    Service for searching Wikipedia and Chabadpedia for biographical and reference information.
    
    Provides search capabilities with language routing, content cleaning, and caching.
    """
    
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl_sec: int = 21600,  # 6 hours
        max_chars: int = 4000,
        top_k: int = 3
    ):
        self.http_client = http_client
        self.redis_client = redis_client
        self.cache_ttl_sec = cache_ttl_sec
        self.max_chars = max_chars
        self.top_k = top_k
        
        # Language priorities
        self.wiki_langs_priority = ["he", "en", "ru"]
        
        # Better User agent with contact info
        self.user_agent = "AstraStudyAssistant/1.0 (Educational use; contact: support@astra.ai)"
    
    def _cache_key(self, service: str, query: str, lang: str = "") -> str:
        """Generate cache key for wiki searches."""
        return f"wiki:{service}:{lang}:{query.lower()}"

    async def fetch_wikipedia_page_via_mcp(self, mcp_base_url: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Try to fetch cleaned article content through an MCP Wikipedia server.
        Expects JSON with 'html' or 'text' fields.
        """
        if not mcp_base_url or not url:
            return None
        full = mcp_base_url.rstrip("/") + "/resources"
        try:
            resp = await self.http_client.get(
                full, params={"uri": url}, headers={"User-Agent": self.user_agent}, timeout=15.0
            )
            resp.raise_for_status()
            data = resp.json()
            # Try to normalize various possible shapes
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if isinstance(data, list) and data:
                data = data[0]
            html = None
            text = None
            if isinstance(data, dict):
                html = data.get("html") or data.get("content") or data.get("body")
                text = data.get("text") or data.get("plain") or data.get("content")
            if not html and not text:
                return None
            return {"html": html or "", "content": text or "", "url": url}
        except Exception as exc:
            logger.warning("MCP wikipedia fetch failed", extra={"url": url, "error": str(exc)})
            return None

    async def fetch_wikipedia_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch full page HTML by URL (no search). Returns dict with html/text/title if possible.
        """
        if not url:
            return None
        cache_key = f"wiki:page:{url}"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    logger.info("Wikipedia page cache HIT", extra={"url": url})
                    return json.loads(cached)
            except Exception as e:
                logger.error("Redis cache read failed for Wikipedia page", extra={"url": url, "error": str(e)})

        headers = {"User-Agent": self.user_agent}
        try:
            resp = await self.http_client.get(url, headers=headers, timeout=20.0)
            resp.raise_for_status()
            html = resp.text
            cleaned_html, text = self._extract_main_content(html)
            data = {"html": cleaned_html or html, "raw_html": html, "content": text, "url": url}
            if self.redis_client:
                try:
                    await self.redis_client.setex(cache_key, self.cache_ttl_sec, json.dumps(data))
                    logger.info("Wikipedia page cache WRITE", extra={"url": url})
                except Exception as e:
                    logger.error("Redis cache write failed for Wikipedia page", extra={"url": url, "error": str(e)})
            return data
        except Exception as e:
            logger.warning("Failed to fetch Wikipedia page", extra={"url": url, "error": str(e)})
            return None

    def _extract_main_content(self, html: str) -> tuple[str, str]:
        """
        Keep only the article body to reduce LLM tokens.
        Returns (clean_html, plain_text).
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            main = soup.find("div", class_="mw-parser-output") or soup.find("div", id="mw-content-text") or soup

            for selector in STRIP_SELECTORS:
                for el in main.select(selector):
                    el.decompose()

            for el in main.find_all(True):
                tag = (el.name or "").lower()
                if tag not in ALLOWED_ARTICLE_TAGS:
                    el.unwrap()
                    continue
                attrs = dict(el.attrs)
                for attr in list(attrs.keys()):
                    if tag == "a" and attr in {"href", "title"}:
                        continue
                    if tag == "img" and attr in {"src", "alt"}:
                        continue
                    el.attrs.pop(attr, None)

            clean_html = str(main)
            text = main.get_text(" ", strip=True)
            # cap to avoid oversized payloads (не слишком жёстко, но чтобы не бить лимиты)
            max_len = 50000
            if len(text) > max_len:
                text = text[:max_len]
            return clean_html, text
        except Exception as exc:
            logger.warning("Failed to extract main content", extra={"error": str(exc)})
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            return "", text
    
    async def search_wikipedia(self, query: str, lang_priority: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search Wikipedia with language fallback.
        
        Args:
            query: Search query
            lang_priority: Language priority list (defaults to he,en,ru)
            
        Returns:
            Dictionary with search results or error information
        """
        if not query or not query.strip():
            return {"ok": False, "error": "Query parameter is required"}
        
        query = query.strip()
        langs = lang_priority or self.wiki_langs_priority
        
        for lang in langs:
            cache_key = self._cache_key("wikipedia", query, lang)
            
            # Try cache first
            if self.redis_client:
                try:
                    cached_result = await self.redis_client.get(cache_key)
                    if cached_result:
                        logger.info("Wikipedia cache HIT", extra={"query": query, "lang": lang})
                        return json.loads(cached_result)
                except Exception as e:
                    logger.error("Redis cache read failed for Wikipedia", extra={
                        "query": query, "lang": lang, "error": str(e)
                    })
            
            # Try to fetch from Wikipedia
            try:
                result = await self._fetch_wikipedia(query, lang)
                if result.get("ok") and result.get("data", {}).get("results"):
                    # Cache successful result
                    if self.redis_client:
                        try:
                            await self.redis_client.setex(
                                cache_key, self.cache_ttl_sec, json.dumps(result)
                            )
                            logger.info("Wikipedia cache WRITE", extra={"query": query, "lang": lang})
                        except Exception as e:
                            logger.error("Redis cache write failed for Wikipedia", extra={
                                "query": query, "lang": lang, "error": str(e)
                            })
                    
                    return result
                    
            except Exception as e:
                logger.warning(f"Wikipedia search failed for lang {lang}", extra={
                    "query": query, "lang": lang, "error": str(e)
                })
                continue
        
        return {
            "ok": False,
            "error": f"No Wikipedia results found for '{query}' in any language"
        }
    
    async def search_chabadpedia(self, query: str) -> Dict[str, Any]:
        """
        Search Chabadpedia for Chabad-related information.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results or error information
        """
        if not query or not query.strip():
            return {"ok": False, "error": "Query parameter is required"}
        
        query = query.strip()
        cache_key = self._cache_key("chabadpedia", query)
        
        # Try cache first
        if self.redis_client:
            try:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    logger.info("Chabadpedia cache HIT", extra={"query": query})
                    return json.loads(cached_result)
            except Exception as e:
                logger.error("Redis cache read failed for Chabadpedia", extra={
                    "query": query, "error": str(e)
                })
        
        try:
            result = await self._fetch_chabadpedia(query)
            
            # Cache result
            if self.redis_client:
                try:
                    await self.redis_client.setex(
                        cache_key, self.cache_ttl_sec, json.dumps(result)
                    )
                    logger.info("Chabadpedia cache WRITE", extra={"query": query})
                except Exception as e:
                    logger.error("Redis cache write failed for Chabadpedia", extra={
                        "query": query, "error": str(e)
                    })
            
            return result
            
        except Exception as e:
            logger.error("Chabadpedia search failed", extra={
                "query": query, "error": str(e)
            })
            return {
                "ok": False,
                "error": f"Chabadpedia search failed: {str(e)}"
            }
    
    async def _fetch_wikipedia(self, query: str, lang: str) -> Dict[str, Any]:
        """Fetch search results from Wikipedia using MediaWiki API."""
        # Use MediaWiki API for search (more reliable than REST API)
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        
        headers = {"User-Agent": self.user_agent}
        
        # Search parameters for MediaWiki API
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": self.top_k,
            "srprop": "snippet|titlesnippet|size",
            "utf8": 1
        }
        
        try:
            # First, search for articles
            logger.info(f"Making Wikipedia search request", extra={
                "url": search_url,
                "params": search_params,
                "headers": headers,
                "query": query,
                "lang": lang
            })
            
            response = await self.http_client.get(search_url, headers=headers, params=search_params, timeout=20.0)
            response.raise_for_status()
            
            search_data = response.json()
            logger.info(f"Wikipedia search response", extra={
                "status_code": response.status_code,
                "response_keys": list(search_data.keys()),
                "query": query,
                "lang": lang
            })
            
            search_results = search_data.get("query", {}).get("search", [])
            logger.info(f"Wikipedia search results count: {len(search_results)}", extra={
                "query": query,
                "lang": lang,
                "results_count": len(search_results),
                "first_result_title": search_results[0].get("title") if search_results else None
            })
            
            if not search_results:
                logger.warning(f"No Wikipedia search results found for query: {query} (lang: {lang})")
                return {"ok": True, "data": {"results": [], "source": f"Wikipedia ({lang})"}}
            
            # Get content for top results
            results = []
            for result in search_results[:self.top_k]:
                page_title = result.get("title", "")
                if not page_title:
                    continue
                
                # Get page extract using MediaWiki API
                extract_params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "titles": page_title,
                    "exintro": True,
                    "explaintext": True,
                    "exsectionformat": "plain",
                    "exchars": self.max_chars,
                    "utf8": 1
                }
                
                try:
                    logger.info(f"Fetching Wikipedia page content for: {page_title}", extra={
                        "page_title": page_title,
                        "extract_params": extract_params,
                        "query": query,
                        "lang": lang
                    })
                    
                    content_response = await self.http_client.get(search_url, headers=headers, params=extract_params, timeout=15.0)
                    content_response.raise_for_status()
                    
                    content_data = content_response.json()
                    pages = content_data.get("query", {}).get("pages", {})
                    
                    logger.info(f"Wikipedia page content response", extra={
                        "page_title": page_title,
                        "pages_count": len(pages),
                        "page_ids": list(pages.keys()),
                        "query": query,
                        "lang": lang
                    })
                    
                    extract = ""
                    for page_id, page_data in pages.items():
                        if page_id != "-1":  # -1 means page not found
                            extract = page_data.get("extract", "")
                            logger.info(f"Found extract for page {page_title}", extra={
                                "page_id": page_id,
                                "extract_length": len(extract),
                                "has_extract": bool(extract),
                                "query": query,
                                "lang": lang
                            })
                            break
                        else:
                            logger.warning(f"Page not found (page_id=-1) for: {page_title}", extra={
                                "page_title": page_title,
                                "query": query,
                                "lang": lang
                            })
                    
                    if extract:
                        # Clean and truncate content
                        clean_content = self._clean_html(extract)
                        truncated_content = self._truncate_text(clean_content, self.max_chars)
                        
                        results.append({
                            "title": page_title,
                            "content": truncated_content,
                            "url": f"https://{lang}.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}",
                            "source": f"Wikipedia ({lang})",
                            "snippet": self._clean_html(result.get("snippet", ""))
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to get content for page '{page_title}': {e}")
                    # Still add result with snippet if available
                    snippet = self._clean_html(result.get("snippet", ""))
                    if snippet:
                        results.append({
                            "title": page_title,
                            "content": snippet,
                            "url": f"https://{lang}.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}",
                            "source": f"Wikipedia ({lang})",
                            "snippet": snippet
                        })
                    continue
            
            return {
                "ok": True,
                "data": {
                    "results": results,
                    "source": f"Wikipedia ({lang})",
                    "query": query
                }
            }
            
        except httpx.HTTPStatusError as e:
            logger.warning("Wikipedia API error", extra={
                "query": query, "lang": lang, "status_code": e.response.status_code, "url": search_url
            })
            return {"ok": False, "error": f"Wikipedia API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}", extra={"query": query, "lang": lang})
            return {"ok": False, "error": str(e)}
    
    async def _fetch_chabadpedia(self, query: str) -> Dict[str, Any]:
        """Fetch search results from Chabadpedia API."""
        # First, search for pages
        search_params = {
            "action": "opensearch",
            "search": query,
            "limit": self.top_k,
            "namespace": 0,
            "format": "json"
        }
        
        headers = {"User-Agent": self.user_agent}
        
        try:
            logger.info(f"Making Chabadpedia search request", extra={
                "url": "https://chabadpedia.co.il/api.php",
                "params": search_params,
                "headers": headers,
                "query": query
            })
            
            response = await self.http_client.get(
                "https://chabadpedia.co.il/api.php", 
                params=search_params, 
                headers=headers, 
                timeout=20.0
            )
            response.raise_for_status()
            
            search_data = response.json()
            logger.info(f"Chabadpedia search response", extra={
                "status_code": response.status_code,
                "response_length": len(search_data) if isinstance(search_data, list) else 0,
                "response_type": type(search_data).__name__,
                "query": query
            })
            
            if len(search_data) < 2 or not search_data[1]:
                logger.warning(f"No Chabadpedia search results found for query: {query}")
                return {"ok": True, "data": {"results": [], "source": "Chabadpedia"}}
            
            titles = search_data[1]  # List of page titles
            urls = search_data[3] if len(search_data) > 3 else []  # List of URLs
            
            logger.info(f"Chabadpedia search results", extra={
                "query": query,
                "titles_count": len(titles),
                "urls_count": len(urls),
                "first_title": titles[0] if titles else None
            })
            
            # Get content for each page
            results = []
            for i, title in enumerate(titles[:self.top_k]):
                if not title:
                    continue
                
                # Get page content
                parse_params = {
                    "action": "parse",
                    "page": title,
                    "format": "json",
                    "prop": "text",
                    "section": 0  # Get only the intro section
                }
                
                try:
                    logger.info(f"Fetching Chabadpedia page content for: {title}", extra={
                        "page_title": title,
                        "parse_params": parse_params,
                        "query": query
                    })
                    
                    content_response = await self.http_client.get(
                        "https://chabadpedia.co.il/api.php",
                        params=parse_params,
                        headers=headers,
                        timeout=15.0
                    )
                    content_response.raise_for_status()
                    
                    content_data = content_response.json()
                    logger.info(f"Chabadpedia page content response", extra={
                        "page_title": title,
                        "response_keys": list(content_data.keys()) if isinstance(content_data, dict) else [],
                        "has_parse": "parse" in content_data if isinstance(content_data, dict) else False,
                        "query": query
                    })
                    
                    # Extract and clean HTML content
                    html_content = content_data.get("parse", {}).get("text", {}).get("*", "")
                    if html_content:
                        # Clean HTML and extract text
                        clean_text = self._clean_html(html_content)
                        
                        # Limit content size
                        truncated_text = self._truncate_text(clean_text, self.max_chars)
                        
                        if truncated_text.strip():
                            results.append({
                                "title": title,
                                "content": truncated_text,
                                "url": urls[i] if i < len(urls) else f"https://chabadpedia.co.il/index.php/{quote(title)}",
                                "source": "Chabadpedia"
                            })
                
                except Exception as e:
                    logger.warning(f"Failed to fetch Chabadpedia page content for {title}", extra={
                        "title": title, "error": str(e)
                    })
                    continue
            
            return {
                "ok": True,
                "data": {
                    "results": results,
                    "source": "Chabadpedia",
                    "query": query
                }
            }
            
        except Exception as e:
            logger.error("Chabadpedia fetch error", extra={
                "query": query, "error": str(e)
            })
            return {"ok": False, "error": f"Chabadpedia fetch failed: {str(e)}"}
    
    def _clean_html(self, html: str) -> str:
        """Clean HTML content and extract readable text."""
        if not html:
            return ""
            
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "sup", "table", "div.navbox"]):
                element.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n\s*\n", "\n", text)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Failed to clean HTML content: {e}")
            return html  # Return original if cleaning fails

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(' ', 1)[0] + "..."
