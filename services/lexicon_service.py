import json
import logging
from typing import Dict, Any, Optional
from urllib.parse import quote

import httpx
import redis.asyncio as redis

from core.utils import with_retries

logger = logging.getLogger(__name__)

class LexiconService:
    """
    Service for interacting with Sefaria's lexicon API.
    
    Provides word definitions, etymologies, and contextual explanations
    with caching and LLM-enhanced explanations.
    """
    
    def __init__(
        self, 
        http_client: httpx.AsyncClient, 
        redis_client: Optional[redis.Redis] = None,
        sefaria_api_url: str = "https://www.sefaria.org/api",
        cache_ttl_sec: int = 3600  # 1 hour cache for lexicon entries
    ):
        self.http_client = http_client
        self.redis_client = redis_client
        self.api_url = sefaria_api_url.rstrip('/')
        self.cache_ttl = cache_ttl_sec
    
    def _cache_key(self, word: str) -> str:
        """Generate cache key for lexicon entry."""
        return f"lexicon:v1:{word.lower()}"
    
    async def get_word_definition(self, word: str) -> Dict[str, Any]:
        """
        Get word definition from Sefaria lexicon API.
        
        Args:
            word: Hebrew/Aramaic word to look up
            
        Returns:
            Dictionary with lexicon data or error information
        """
        if not word or not word.strip():
            return {"ok": False, "error": "Word parameter is required"}
        
        word = word.strip()
        cache_key = self._cache_key(word)
        
        # Try cache first
        if self.redis_client:
            try:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    logger.info("Lexicon cache HIT", extra={"word": word})
                    return json.loads(cached_result)
            except Exception as e:
                logger.error("Redis cache read failed for lexicon", extra={
                    "word": word,
                    "error": str(e)
                })
        
        # Fetch from Sefaria API
        logger.info("Fetching lexicon entry from Sefaria", extra={"word": word})
        
        try:
            api_call = lambda: self._fetch_from_sefaria(word)
            result = await with_retries(api_call)
            
            # Cache successful results
            if result["ok"] and self.redis_client:
                try:
                    await self.redis_client.set(
                        cache_key, 
                        json.dumps(result, ensure_ascii=False), 
                        ex=self.cache_ttl
                    )
                    logger.info("Lexicon cache WRITE", extra={"word": word})
                except Exception as e:
                    logger.error("Redis cache write failed for lexicon", extra={
                        "word": word,
                        "error": str(e)
                    })
            
            return result
            
        except Exception as e:
            logger.error("Failed to fetch lexicon entry", extra={
                "word": word,
                "error": str(e)
            })
            return {
                "ok": False, 
                "error": f"Failed to fetch definition for '{word}': {str(e)}"
            }
    
    async def _fetch_from_sefaria(self, word: str) -> Dict[str, Any]:
        """
        Fetch word definition directly from Sefaria API.
        
        Args:
            word: Word to look up
            
        Returns:
            Dictionary with API response data
        """
        url = f"{self.api_url}/words/{quote(word)}"
        
        try:
            response = await self.http_client.get(url, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            
            # Process and clean the response
            processed_data = self._process_lexicon_response(data, word)
            
            logger.info("Successfully fetched lexicon entry", extra={
                "word": word,
                "entries_count": len(processed_data.get("entries", []))
            })
            
            return {"ok": True, "data": processed_data}
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info("Word not found in Sefaria lexicon", extra={"word": word})
                return {
                    "ok": False, 
                    "error": f"Word '{word}' not found in lexicon",
                    "status_code": 404
                }
            else:
                logger.warning("Sefaria API error", extra={
                    "word": word,
                    "status_code": e.response.status_code,
                    "response": e.response.text[:200] if hasattr(e.response, 'text') else None
                })
                return {
                    "ok": False,
                    "error": f"Sefaria API returned status {e.response.status_code}",
                    "status_code": e.response.status_code
                }
                
        except httpx.RequestError as e:
            logger.warning("Network error fetching lexicon entry", extra={
                "word": word,
                "error": str(e)
            })
            return {
                "ok": False,
                "error": f"Could not connect to Sefaria API: {str(e)}"
            }
    
    def _process_lexicon_response(self, data: Any, word: str) -> Dict[str, Any]:
        """
        Process and clean lexicon response from Sefaria.
        
        Args:
            data: Raw API response
            word: Original word queried
            
        Returns:
            Processed lexicon data
        """
        if not isinstance(data, dict):
            return {"word": word, "entries": [], "raw_response": data}
        
        processed = {
            "word": word,
            "entries": [],
            "metadata": {}
        }
        
        # Extract entries if they exist
        if "entries" in data and isinstance(data["entries"], list):
            for entry in data["entries"]:
                if isinstance(entry, dict):
                    processed_entry = self._process_lexicon_entry(entry)
                    if processed_entry:
                        processed["entries"].append(processed_entry)
        
        # Extract metadata
        for key in ["word", "language", "lexicon", "parent_lexicon"]:
            if key in data:
                processed["metadata"][key] = data[key]
        
        # If no structured entries, try to extract any definition-like content
        if not processed["entries"] and data:
            # Look for common definition fields
            definition_fields = ["definition", "meaning", "sense", "gloss", "translation"]
            for field in definition_fields:
                if field in data and data[field]:
                    processed["entries"].append({
                        "definition": str(data[field]),
                        "type": "basic",
                        "source": field
                    })
                    break
        
        return processed
    
    def _process_lexicon_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single lexicon entry.
        
        Args:
            entry: Raw lexicon entry
            
        Returns:
            Processed entry or None if invalid
        """
        if not isinstance(entry, dict):
            return None
        
        processed = {
            "definition": "",
            "type": "entry",
            "metadata": {}
        }
        
        # Extract definition/meaning
        for field in ["definition", "meaning", "sense", "gloss"]:
            if field in entry and entry[field]:
                processed["definition"] = str(entry[field])
                break
        
        # Extract metadata
        for key in ["part_of_speech", "grammar", "root", "etymology", "language"]:
            if key in entry and entry[key]:
                processed["metadata"][key] = entry[key]
        
        # Only return if we have a definition
        return processed if processed["definition"] else None
    
    async def get_multiple_definitions(self, words: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get definitions for multiple words concurrently.
        
        Args:
            words: List of words to look up
            
        Returns:
            Dictionary mapping words to their definitions
        """
        if not words:
            return {}
        
        import asyncio
        
        # Create tasks for all words
        tasks = {}
        for word in words:
            if word and word.strip():
                tasks[word] = asyncio.create_task(self.get_word_definition(word))
        
        # Wait for all tasks to complete
        results = {}
        for word, task in tasks.items():
            try:
                results[word] = await task
            except Exception as e:
                logger.error("Failed to get definition in batch", extra={
                    "word": word,
                    "error": str(e)
                })
                results[word] = {
                    "ok": False,
                    "error": f"Failed to fetch definition: {str(e)}"
                }
        
        return results
    
    async def clear_cache(self, word: Optional[str] = None) -> bool:
        """
        Clear lexicon cache.
        
        Args:
            word: Specific word to clear, or None to clear all
            
        Returns:
            True if cache was cleared successfully
        """
        if not self.redis_client:
            return False
        
        try:
            if word:
                # Clear specific word
                cache_key = self._cache_key(word)
                await self.redis_client.delete(cache_key)
                logger.info("Cleared lexicon cache for word", extra={"word": word})
            else:
                # Clear all lexicon cache
                pattern = "lexicon:v1:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    logger.info("Cleared all lexicon cache", extra={"keys_count": len(keys)})
            
            return True
            
        except Exception as e:
            logger.error("Failed to clear lexicon cache", extra={
                "word": word,
                "error": str(e)
            })
            return False


    async def get_word_definition_for_tool(self, word: str) -> Dict[str, Any]:
        """
        Get word definition from Sefaria lexicon API with LIMITED data for tool usage.
        This method returns only essential information to prevent token overuse.
        
        Args:
            word: Hebrew/Aramaic word to look up
            
        Returns:
            Dictionary with LIMITED lexicon data or error information
        """
        # Get full definition first
        full_result = await self.get_word_definition(word)
        
        if not full_result.get("ok", False):
            return full_result
        
        # Extract and limit the data
        data = full_result.get("data", {})
        
        # Create limited response
        limited_result = {
            "ok": True,
            "data": {
                "word": data.get("word", word),
                "entries": [],
                "metadata": data.get("metadata", {})
            }
        }
        
        # Take only first 2 entries and limit their content
        entries = data.get("entries", [])
        for i, entry in enumerate(entries[:2]):  # Max 2 entries
            if isinstance(entry, dict):
                limited_entry = {
                    "definition": self._truncate_text(entry.get("definition", ""), 150),
                    "type": entry.get("type", "entry"),
                    "metadata": {}
                }
                
                # Add only essential metadata
                if "part_of_speech" in entry.get("metadata", {}):
                    limited_entry["metadata"]["part_of_speech"] = entry["metadata"]["part_of_speech"]
                
                limited_result["data"]["entries"].append(limited_entry)
        
        return limited_result
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to maximum length, preserving word boundaries.
        
        Args:
            text: Text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        # Find last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        
        if last_space > max_length * 0.8:  # If space is reasonably close to end
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."

