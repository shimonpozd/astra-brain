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
        
        # API endpoints
        self.wikipedia_api = "https://{lang}.wikipedia.org/api/rest_v1"
        self.chabadpedia_api = "https://chabadpedia.co.il/api.php"
        
        # Language priorities
        self.wiki_langs_priority = ["he", "en", "ru"]
        
        # User agent for requests
        self.user_agent = "AstraStudyAssistant/1.0 (Educational use)"
    
    def _cache_key(self, service: str, query: str, lang: str = "") -> str:
        """Generate cache key for wiki searches."""
        return f"wiki:{service}:{lang}:{query.lower()}"
    
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
        """Fetch search results from Wikipedia API."""
        # First, search for articles
        search_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/search/{quote(query)}"
        
        headers = {"User-Agent": self.user_agent}
        
        try:
            response = await self.http_client.get(search_url, headers=headers, timeout=20.0)
            response.raise_for_status()
            
            search_data = response.json()
            pages = search_data.get("pages", [])
            
            if not pages:
                return {"ok": True, "data": {"results": [], "source": f"Wikipedia ({lang})"}}
            
            # Get content for top results
            results = []
            for page in pages[:self.top_k]:
                page_title = page.get("title", "")
                if not page_title:
                    continue
                
                # Get page content
                content_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(page_title)}"
                
                try:
                    content_response = await self.http_client.get(content_url, headers=headers, timeout=15.0)
                    content_response.raise_for_status()
                    
                    content_data = content_response.json()
                    
                    # Extract and clean content
                    extract = content_data.get("extract", "")
                    if extract:
                        # Limit content size
                        if len(extract) > self.max_chars:
                            extract = extract[:self.max_chars] + "..."
                        
                        results.append({
                            "title": page_title,
                            "content": extract,
                            "url": content_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            "lang": lang
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to fetch Wikipedia page content for {page_title}", extra={
                        "title": page_title, "lang": lang, "error": str(e)
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
                "query": query, "lang": lang, "status_code": e.response.status_code
            })
            return {"ok": False, "error": f"Wikipedia API error: {e.response.status_code}"}
        
        except Exception as e:
            logger.error("Wikipedia fetch error", extra={
                "query": query, "lang": lang, "error": str(e)
            })
            return {"ok": False, "error": f"Wikipedia fetch failed: {str(e)}"}
    
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
            response = await self.http_client.get(
                self.chabadpedia_api, 
                params=search_params, 
                headers=headers, 
                timeout=20.0
            )
            response.raise_for_status()
            
            search_data = response.json()
            
            if len(search_data) < 2 or not search_data[1]:
                return {"ok": True, "data": {"results": [], "source": "Chabadpedia"}}
            
            titles = search_data[1]  # List of page titles
            urls = search_data[3] if len(search_data) > 3 else []  # List of URLs
            
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
                    content_response = await self.http_client.get(
                        self.chabadpedia_api,
                        params=parse_params,
                        headers=headers,
                        timeout=15.0
                    )
                    content_response.raise_for_status()
                    
                    content_data = content_response.json()
                    
                    # Extract and clean HTML content
                    html_content = content_data.get("parse", {}).get("text", {}).get("*", "")
                    if html_content:
                        # Clean HTML and extract text
                        clean_text = self._clean_html_content(html_content)
                        
                        # Limit content size
                        if len(clean_text) > self.max_chars:
                            clean_text = clean_text[:self.max_chars] + "..."
                        
                        if clean_text.strip():
                            results.append({
                                "title": title,
                                "content": clean_text,
                                "url": urls[i] if i < len(urls) else f"https://chabadpedia.co.il/index.php/{quote(title)}",
                                "lang": "he"
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
    
    def _clean_html_content(self, html: str) -> str:
        """Clean HTML content and extract readable text."""
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

