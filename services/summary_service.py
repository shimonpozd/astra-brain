import json
import time
import logging
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)

# Regex for Sefaria references
TREF_RE = re.compile(r"[A-Z][a-zA-Z]+(?:\s[0-9]+[ab])?[:\s]\d+(?::\d+)?")

class SummaryService:
    """
    Service for LLM-based conversation summarization.
    
    Compresses recent messages into compact bullet points for STM injection
    with configurable quality/cost trade-offs.
    """
    
    def __init__(self, llm_service, config: Optional[Dict[str, Any]] = None, redis_client=None):
        self.llm_service = llm_service
        self.config = config or {}
        self.redis = redis_client
        
        # Load STM summary configuration
        stm_summary_config = self.config.get("stm", {}).get("summary", {})
        self.enabled = stm_summary_config.get("enabled", True)
        self.input_tokens_budget = stm_summary_config.get("input_tokens_budget", 1200)
        self.output_bullets_min = stm_summary_config.get("output_bullets_min", 3)
        self.output_bullets_max = stm_summary_config.get("output_bullets_max", 8)
        self.bullet_max_chars = stm_summary_config.get("bullet_max_chars", 140)
        self.allow_refs = stm_summary_config.get("allow_refs", True)
        self.max_refs = stm_summary_config.get("max_refs", 5)
        self.cooldown_sec = stm_summary_config.get("cooldown_sec", 30)
        self.trigger_msgs_high = stm_summary_config.get("trigger_msgs_high", 10)
        self.trigger_msgs_low = stm_summary_config.get("trigger_msgs_low", 6)
        self.trigger_tokens_high = stm_summary_config.get("trigger_tokens_high", 2500)
        self.trigger_tokens_low = stm_summary_config.get("trigger_tokens_low", 1500)
        self.log_verbose = stm_summary_config.get("log_verbose", False)
        self.partial_min_tokens = stm_summary_config.get("partial_min_tokens", 50)
        
        # Load LLM task configuration
        llm_summary_config = self.config.get("llm", {}).get("tasks", {}).get("summary", {})
        self.model = llm_summary_config.get("model", "gpt-4o-mini")
        self.temperature = llm_summary_config.get("temperature", 0.2)
        self.top_p = llm_summary_config.get("top_p", 1.0)
        self.max_tokens_out = llm_summary_config.get("max_tokens_out", 512)
        self.timeout_s = llm_summary_config.get("timeout_s", 25)
        self.retries = llm_summary_config.get("retries", 2)
        self.backoff_ms = llm_summary_config.get("backoff_ms", 400)
        self.response_format_json = llm_summary_config.get("response_format_json", True)
        
        # Load system prompt from prompts system
        try:
            import sys
            import os
            # Add project root to path if not already there
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from config.prompts import get_prompt
            prompt_data = get_prompt("actions.summary_system")
            # get_prompt returns a string, not a dict
            self.system_prompt = prompt_data if isinstance(prompt_data, str) and prompt_data.strip() else self._get_default_prompt()
        except Exception as e:
            logger.warning("Failed to load summary prompt from prompts system", extra={"error": str(e)})
            self.system_prompt = self._get_default_prompt()
    
    async def _get_meta(self, session_id: str) -> Dict[str, Any]:
        """Get summary meta data from Redis."""
        try:
            if self.redis:
                raw = await self.redis.get(f"stm:summary:meta:{session_id}")
                return json.loads(raw) if raw else {}
            return {}
        except Exception as e:
            logger.warning("Failed to get summary meta", extra={"session_id": session_id, "error": str(e)})
            return {}
    
    async def _set_meta(self, session_id: str, meta: Dict[str, Any]) -> None:
        """Set summary meta data in Redis."""
        try:
            if self.redis:
                await self.redis.set(
                    f"stm:summary:meta:{session_id}", 
                    json.dumps(meta), 
                    ex=7*24*3600  # 7 days TTL
                )
        except Exception as e:
            logger.warning("Failed to set summary meta", extra={"session_id": session_id, "error": str(e)})
    
    def _validate_refs(self, refs: List[str]) -> List[str]:
        """Validate Sefaria references using regex pattern."""
        if not self.allow_refs:
            return []
        
        valid_refs = []
        for ref in refs:
            ref = ref.strip()
            if TREF_RE.search(ref):
                valid_refs.append(ref)
        
        return valid_refs[:self.max_refs]
    
    def _get_default_prompt(self) -> str:
        """Get default system prompt for summarization."""
        return """Ваша задача — сжать последние сообщения диалога в 3–8 пунктов.
Правила:
• Пишите кратко, по делу, без воды и общих слов.
• Каждый пункт ≤ 140 символов.
• Используйте нейтральный стиль, без эмоций и оценок.
• Сохраняйте конкретику (имена, tref, номера, параметры).
• Если есть явные ссылки на источники (Sefaria tref) — выделите их в поле "refs".
• Отвечайте строго JSON-объектом со схемой:
  {"version":"1.0","bullets":[...], "refs":[...]}
НЕ добавляйте комментарии, пояснения, Markdown — только JSON."""
    
    async def summarize(self, session_id: str, last_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize recent messages using LLM.
        
        Args:
            session_id: Session identifier
            last_messages: Recent messages to summarize
            
        Returns:
            Dict with bullets, refs, and metadata
        """
        if not self.enabled:
            return self._generate_fallback_summary(last_messages)
        
        start_time = time.time()
        
        try:
            # Prepare compressed input messages
            compressed_messages = self._compress_messages(last_messages)
            
            if self.log_verbose:
                logger.info("Summary input prepared", extra={
                    "session_id": session_id,
                    "original_count": len(last_messages),
                    "compressed_count": len(compressed_messages),
                    "input_tokens_est": sum(len(str(msg.get("content", ""))) for msg in compressed_messages) // 4
                })
            
            # Call LLM for summarization
            result = await self.llm_service.summarize(
                messages=compressed_messages,
                prompt=self.system_prompt,
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens_out,
                timeout_s=self.timeout_s,
                retries=self.retries,
                backoff_ms=self.backoff_ms,
                response_format_json=self.response_format_json
            )
            
            # Validate and process result
            try:
                processed_result = self._validate_and_process_result(result)
            except ValueError as ve:
                # Summary too weak, don't update STM
                logger.warning("Summary too weak, skipping STM update", extra={
                    "session_id": session_id,
                    "error": str(ve)
                })
                return {
                    "version": "1.0",
                    "bullets": [],
                    "refs": [],
                    "meta": {
                        "tokens_in": sum(len(str(msg.get("content", ""))) for msg in compressed_messages) // 4,
                        "tokens_out": len(str(result)) // 4,
                        "model": self.model,
                        "latency_ms": (time.time() - start_time) * 1000,
                        "method": "llm_weak"
                    }
                }
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Add metadata
            processed_result["meta"] = {
                "tokens_in": sum(len(str(msg.get("content", ""))) for msg in compressed_messages) // 4,
                "tokens_out": len(str(result)) // 4,
                "model": self.model,
                "latency_ms": latency_ms,
                "method": "llm"
            }
            
            if self.log_verbose:
                logger.info("Summary completed", extra={
                    "session_id": session_id,
                    "bullets_count": len(processed_result.get("bullets", [])),
                    "refs_count": len(processed_result.get("refs", [])),
                    "latency_ms": latency_ms
                })
            
            # Save meta data after successful summary
            await self._set_meta(session_id, {"last_update_ts": time.time()})
            
            return processed_result
            
        except Exception as e:
            logger.error("Summary failed, using fallback", extra={
                "session_id": session_id,
                "error": str(e)
            })
            return self._generate_fallback_summary(last_messages)
    
    def _compress_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress messages to fit within token budget. Clean first, then count, then decide."""
        compressed = []
        current_tokens = 0
        
        # Process messages in reverse order (most recent first)
        for msg in reversed(messages):
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            
            # Clean content first
            cleaned = self._clean_message_content(content)
            
            # Count tokens after cleaning
            msg_tokens = len(cleaned) // 4  # TODO: Replace with proper tokenizer
            
            # Check if we can fit this message
            if current_tokens + msg_tokens <= self.input_tokens_budget:
                compressed.insert(0, {
                    "role": msg.get("role"),
                    "content": cleaned
                })
                current_tokens += msg_tokens
            else:
                # Partially include if we have meaningful budget left
                remaining = self.input_tokens_budget - current_tokens
                if remaining > self.partial_min_tokens:  # Configurable minimum tokens
                    part = cleaned[:remaining * 4].rstrip() + "..."
                    compressed.insert(0, {
                        "role": msg.get("role"),
                        "content": part
                    })
                break
        
        return compressed
    
    def _clean_message_content(self, content: str) -> str:
        """Clean message content by removing noise and compressing."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove common noise patterns
        noise_patterns = [
            r'^\s*[•\-*]\s*',  # Leading bullet points
            r'\s*[•\-*]\s*$',  # Trailing bullet points
            r'^\s*\d+\.\s*',   # Leading numbers
            r'\s*\d+\.\s*$',   # Trailing numbers
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content)
        
        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."
        
        return content.strip()
    
    def _validate_and_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM result and ensure it meets requirements."""
        try:
            # Ensure we have the expected structure
            if not isinstance(result, dict):
                raise ValueError("Result is not a dictionary")
            
            bullets = result.get("bullets", [])
            refs = result.get("refs", [])
            
            # Validate bullets
            if not isinstance(bullets, list):
                bullets = []
            
            # 1) Normalize bullets without padding
            valid_bullets = []
            if isinstance(bullets, list):
                for bullet in bullets:
                    if isinstance(bullet, str) and bullet.strip():
                        # Truncate to max length
                        truncated = bullet.strip()[:self.bullet_max_chars]
                        valid_bullets.append(truncated)
            
            # Validate refs
            if not isinstance(refs, list):
                refs = []
            
            # Use Sefaria reference validation
            valid_refs = self._validate_refs([ref for ref in refs if isinstance(ref, str) and ref.strip()])
            
            # 2) Check usefulness BEFORE padding
            if len(valid_bullets) == 0 and not valid_refs:
                raise ValueError("Summary too weak to update STM")
            
            # 3) Now bring to acceptable bounds
            if len(valid_bullets) > self.output_bullets_max:
                valid_bullets = valid_bullets[:self.output_bullets_max]
            elif len(valid_bullets) < self.output_bullets_min:
                # Soft padding, but only after usefulness check
                while len(valid_bullets) < self.output_bullets_min:
                    valid_bullets.append("Conversation continued...")
            
            return {
                "version": "1.0",
                "bullets": valid_bullets,
                "refs": valid_refs
            }
            
        except Exception as e:
            logger.warning("Failed to validate summary result", extra={
                "error": str(e),
                "result": str(result)[:200]
            })
            # Return minimal valid structure
            return {
                "version": "1.0",
                "bullets": ["Summary generation failed"],
                "refs": []
            }
    
    def _generate_fallback_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback summary using local heuristics."""
        bullets = []
        
        # Extract recent user questions
        user_messages = [m for m in messages if m.get("role") == "user"]
        for msg in user_messages[-3:]:  # Last 3 user messages
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Truncate and clean
                bullet = content.strip()[:self.bullet_max_chars]
                if bullet.endswith("?"):
                    bullets.append(f"Q: {bullet}")
                else:
                    bullets.append(f"User: {bullet}")
        
        # Extract recent assistant responses
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        for msg in assistant_messages[-2:]:  # Last 2 assistant messages
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Extract first sentence or truncate
                first_sentence = content.split('.')[0].strip()
                bullet = first_sentence[:self.bullet_max_chars]
                bullets.append(f"A: {bullet}")
        
        # Ensure minimum bullets
        while len(bullets) < self.output_bullets_min:
            bullets.append("Conversation continued...")
        
        # Limit to max bullets
        bullets = bullets[:self.output_bullets_max]
        
        return {
            "version": "1.0",
            "bullets": bullets,
            "refs": [],
            "meta": {
                "tokens_in": sum(len(str(msg.get("content", ""))) for msg in messages) // 4,
                "tokens_out": len(str(bullets)) // 4,
                "model": "fallback",
                "latency_ms": 0,
                "method": "fallback"
            }
        }
    
    async def should_update_summary(self, session_id: str, message_count: int, token_count: int) -> bool:
        """Check if summary should be updated based on triggers and cooldown."""
        if not self.enabled:
            return False
        
        # Hysteresis logic
        trigger = (
            message_count >= self.trigger_msgs_high or 
            token_count >= self.trigger_tokens_high or
            (message_count >= self.trigger_msgs_low and token_count >= self.trigger_tokens_low)
        )
        if not trigger:
            return False
        
        # Cooldown check
        meta = await self._get_meta(session_id)
        last_ts = float(meta.get("last_update_ts", 0) or 0)
        if time.time() - last_ts < self.cooldown_sec:
            return False
        
        return True
