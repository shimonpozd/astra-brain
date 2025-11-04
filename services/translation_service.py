import json
import re
import logging
import html
from typing import Dict, Any, AsyncGenerator, Optional, List

from brain_service.services.sefaria_service import SefariaService
from brain_service.services.llm_service import LLMService
from config.prompts import get_prompt
from core.llm_config import get_llm_for_task, LLMConfigError

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Service for translating Hebrew/Aramaic texts.
    
    Provides translation functionality using LLM with context from Sefaria texts.
    """
    
    def __init__(
        self, 
        sefaria_service: SefariaService,
        llm_service: Optional[LLMService] = None
    ):
        self.sefaria_service = sefaria_service
        self.llm_service = llm_service
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing HTML entities, extra spaces, and other artifacts.
        
        Args:
            text: Raw text from Sefaria
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # HTML unescape
        text = html.unescape(text)
        
        # Remove HTML entities
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#39;', "'", text)
        
        # Remove non-breaking spaces and other Unicode whitespace
        text = text.replace("\u00A0", " ")  # Non-breaking space
        text = text.replace("\u200B", "")   # Zero-width space
        text = text.replace("\u200C", "")   # Zero-width non-joiner
        text = text.replace("\u200D", "")   # Zero-width joiner
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    async def translate_text_reference(self, tref: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Translate a text reference using Sefaria data and LLM.
        
        Args:
            tref: Text reference to translate
            
        Yields:
            Translation events in NDJSON format
        """
        if not tref or not tref.strip():
            yield {"type": "error", "data": {"message": "No text reference provided for translation."}}
            return
        
        # Load text from Sefaria
        try:
            text_result = await self.sefaria_service.get_text(tref)
            
            # Debug: log the type and content of text_result
            logger.info("Translation debug", extra={
                "tref": tref,
                "text_result_type": str(type(text_result)),
                "text_result_keys": list(text_result.keys()) if isinstance(text_result, dict) else "not a dict",
                "text_result_preview": str(text_result)[:200] if text_result else "None"
            })
            
            if not isinstance(text_result, dict) or not text_result.get("ok") or not text_result.get("data"):
                yield {
                    "type": "error", 
                    "data": {
                        "message": f"Could not load text for reference: {tref}",
                        "stage": "fetch",
                        "tref": tref,
                        "text_result_type": str(type(text_result)),
                        "text_result_keys": list(text_result.keys()) if isinstance(text_result, dict) else "not a dict"
                    }
                }
                return
            
            data = text_result["data"]
            
            # Handle case where data might be a list of comments/segments
            if isinstance(data, list):
                if not data:
                    yield {"type": "error", "data": {"message": f"No data found for reference: {tref}"}}
                    return
                
                # Combine all segments with separators
                segs_he = [self._normalize_text(x.get("he_text", "")) for x in data if isinstance(x, dict)]
                segs_en = [self._normalize_text(x.get("en_text", "")) for x in data if isinstance(x, dict)]
                
                hebrew_text = "\n\n### SEGMENT BREAK ###\n\n".join([s for s in segs_he if s])
                english_text = "\n\n### SEGMENT BREAK ###\n\n".join([s for s in segs_en if s]) or "(Not provided)"
            else:
                # Ensure data is a dictionary
                if not isinstance(data, dict):
                    yield {"type": "error", "data": {"message": f"Unexpected data format for reference: {tref}"}}
                    return
                    
                hebrew_text = self._normalize_text(data.get("he_text", ""))
                english_text = self._normalize_text(data.get("en_text", ""))
            
            if not hebrew_text:
                yield {"type": "error", "data": {"message": f"No Hebrew text found for reference: {tref}"}}
                return
                
        except Exception as e:
            logger.error("Failed to load text for translation", extra={
                "tref": tref,
                "error": str(e),
                "error_type": str(type(e)),
                "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else "No traceback"
            })
            yield {"type": "error", "data": {"message": f"Failed to load text: {str(e)}"}}
            return
        
        # Get prompts
        system_prompt = get_prompt('actions.translator_system')
        user_prompt_template = get_prompt('actions.translator_user_template')
        
        if not system_prompt or not user_prompt_template:
            yield {"type": "error", "data": {"message": "Translation prompts not found."}}
            return
        
        # Format user prompt
        user_prompt = user_prompt_template.format(
            hebrew_text=hebrew_text,
            english_text=english_text or "(Not provided)"
        )
        
        # Log translation request details
        logger.info("Translation request", extra={
            "tref": tref,
            "hebrew_text_length": len(hebrew_text),
            "english_text_length": len(english_text) if english_text else 0,
            "hebrew_preview": hebrew_text[:100] if hebrew_text else "(Not provided)",
            "english_preview": english_text[:100] if english_text else "(Not provided)"
        })
        
        # Get LLM client
        try:
            client, model, reasoning_params, capabilities = get_llm_for_task('TRANSLATOR')
        except LLMConfigError:
            try:
                logger.warning("LLM config for 'TRANSLATOR' task not found, falling back to 'CHAT'.")
                client, model, reasoning_params, capabilities = get_llm_for_task('CHAT')
            except LLMConfigError as e:
                yield {"type": "error", "data": {"message": f"LLM config failed: {e}"}}
                return
        
        # Perform translation
        try:
            completion_args = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                **reasoning_params
            }
            
            # Temporarily disable json_mode to test
            # if "json_mode" in capabilities:
            #     completion_args["response_format"] = {"type": "json_object"}
            
            response = await client.chat.completions.create(**completion_args)
            llm_response_text = response.choices[0].message.content
            
            # Debug: log the LLM response
            logger.info(f"LLM response debug: tref={tref}, type={type(llm_response_text)}, preview={str(llm_response_text)[:500] if llm_response_text else 'None'}, length={len(llm_response_text) if llm_response_text else 0}")
            
            if not llm_response_text:
                raise ValueError("LLM returned an empty response.")
            
            # Parse translation response - simplified
            translation_text = self._extract_translation(llm_response_text)
            
            # Debug: log the extracted translation
            logger.info(f"Translation extraction debug: tref={tref}, type={type(translation_text)}, preview={str(translation_text)[:200] if translation_text else 'None'}, length={len(translation_text) if translation_text else 0}")
            
            # If we got some text, return it; otherwise return the raw response
            if translation_text:
                yield {"type": "llm_chunk", "data": translation_text}
            else:
                # Fallback: return raw response if extraction failed
                yield {"type": "llm_chunk", "data": llm_response_text.strip()}
            
        except Exception as e:
            logger.exception("Error during translation LLM call", extra={
                "tref": tref,
                "error": str(e),
                "error_type": str(type(e)),
                "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else "No traceback"
            })
            yield {
                "type": "error", 
                "data": {
                    "message": str(e),
                    "stage": "llm",
                    "tref": tref,
                    "model": model if 'model' in locals() else "unknown",
                    "capabilities": capabilities if 'capabilities' in locals() else "unknown",
                    "hebrew_text_length": len(hebrew_text) if 'hebrew_text' in locals() else 0,
                    "english_text_length": len(english_text) if 'english_text' in locals() else 0
                }
            }
    
    async def translate_custom_text(
        self, 
        hebrew_text: str, 
        english_text: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Translate custom Hebrew/Aramaic text.
        
        Args:
            hebrew_text: Hebrew text to translate
            english_text: Optional existing English translation for context
            
        Yields:
            Translation events in NDJSON format
        """
        if not hebrew_text or not hebrew_text.strip():
            yield {"type": "error", "data": {"message": "No Hebrew text provided for translation."}}
            return
        
        # Get prompts
        system_prompt = get_prompt('actions.translator_system')
        user_prompt_template = get_prompt('actions.translator_user_template')
        
        if not system_prompt or not user_prompt_template:
            yield {"type": "error", "data": {"message": "Translation prompts not found."}}
            return
        
        # Format user prompt
        user_prompt = user_prompt_template.format(
            hebrew_text=hebrew_text.strip(),
            english_text=english_text or "(Not provided)"
        )
        
        logger.info("Custom text translation request", extra={
            "hebrew_text_length": len(hebrew_text),
            "english_text_length": len(english_text) if english_text else 0
        })
        
        # Get LLM client
        try:
            client, model, reasoning_params, capabilities = get_llm_for_task('TRANSLATOR')
        except LLMConfigError:
            try:
                logger.warning("LLM config for 'TRANSLATOR' task not found, falling back to 'CHAT'.")
                client, model, reasoning_params, capabilities = get_llm_for_task('CHAT')
            except LLMConfigError as e:
                yield {"type": "error", "data": {"message": f"LLM config failed: {e}"}}
                return
        
        # Perform translation
        try:
            completion_args = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                **reasoning_params
            }
            
            # Temporarily disable json_mode to test
            # if "json_mode" in capabilities:
            #     completion_args["response_format"] = {"type": "json_object"}
            
            response = await client.chat.completions.create(**completion_args)
            llm_response_text = response.choices[0].message.content
            
            if not llm_response_text:
                raise ValueError("LLM returned an empty response.")
            
            # Parse translation response
            translation_text = self._extract_translation(llm_response_text)
            
            yield {"type": "llm_chunk", "data": translation_text}
            
        except Exception as e:
            logger.exception("Error during custom text translation", extra={
                "error": str(e)
            })
            yield {"type": "error", "data": {"message": str(e)}}
    
    def _extract_translation(self, llm_response: str) -> str:
        """
        Extract translation text from LLM response - simplified version.
        Just return the text as-is, with basic cleanup.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Cleaned translation text
        """
        if not llm_response or not llm_response.strip():
            return ""
            
        txt = llm_response.strip()
        
        # Remove markdown code fences if present
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL)
            txt = txt.strip()
        
        # Remove quotes if the entire response is wrapped in quotes
        if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
            txt = txt[1:-1]
        
        # Basic cleanup
        txt = txt.strip()
        
        return txt
    
    async def get_translation_quality_score(self, hebrew_text: str, translation: str) -> float:
        """
        Get a quality score for a translation (placeholder for future implementation).
        
        Args:
            hebrew_text: Original Hebrew text
            translation: Translation to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # This is a placeholder - could be implemented with:
        # - Back-translation comparison
        # - Semantic similarity scoring
        # - Grammar/fluency checking
        # - Human feedback integration
        
        logger.debug("Translation quality scoring not yet implemented")
        return 0.8  # Default score
    
    async def get_translation_alternatives(
        self, 
        hebrew_text: str, 
        count: int = 3
    ) -> List[str]:
        """
        Get alternative translations for Hebrew text (placeholder for future implementation).
        
        Args:
            hebrew_text: Hebrew text to translate
            count: Number of alternatives to generate
            
        Returns:
            List of alternative translations
        """
        # This could be implemented by:
        # - Multiple LLM calls with different prompts
        # - Different temperature settings
        # - Different models
        # - Integration with translation databases
        
        logger.debug("Translation alternatives not yet implemented")
        return []  # Placeholder
