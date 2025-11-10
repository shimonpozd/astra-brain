import json
import time
import logging
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Enhanced Short-Term Memory (STM) service with structured slots, hysteresis triggers,
    and semantic deduplication using SimHash.
    
    STM provides working memory between messages without overflowing context,
    with TTL and deterministic integration into prompts.
    """
    
    # Default configuration
    DEFAULT_COOLDOWN_SEC = 30
    DEFAULT_TTL_SEC = 86400
    
    # Trigger thresholds with hysteresis
    TRIGGER_MSGS_HIGH = 10
    TRIGGER_MSGS_LOW = 6
    TRIGGER_TOKENS_HIGH = 2500
    TRIGGER_TOKENS_LOW = 1500
    
    # Limits for each slot
    MAX_FACTS = 50
    MAX_OPEN_LOOPS = 10
    MAX_REFS = 10
    MAX_GLOSSARY = 20
    MAX_ACTION_ITEMS = 12
    MAX_SUMMARY_ITEMS = 8
    MAX_SUMMARY_LEN = 140
    
    # SimHash threshold for deduplication
    HAMMING_THRESHOLD = 6
    
    # Regex for Sefaria references
    TREF_RE = re.compile(r"[A-Z][a-zA-Z]+(?:\s[0-9]+[ab])?[:\s]\d+(?::\d+)?")
    
    ACTION_KEYWORDS = [
        "open", "review", "study", "explore", "try", "compare", "focus", "analyze",
        "practice", "read", "check", "explain", "consider", "look at", "summarize",
        "outline", "plan", "draft", "implement", "debug", "investigate", "analyze",
        "apply", "follow", "examine", "walk through", "remember", "use",
        "открой", "посмотри", "прочитай", "изучи", "перейди", "попробуй",
        "проанализируй", "сравни", "ответь", "объясни", "подготовь", "расскажи",
        "разбери", "упомяни", "напомни"
    ]
    
    def __init__(self, redis_client: redis.Redis, ttl_sec: int = DEFAULT_TTL_SEC, config: Optional[Dict[str, Any]] = None, summary_service=None):
        self.redis_client = redis_client
        self.ttl = ttl_sec
        self.config = config or {}
        self.summary_service = summary_service
        
        # Load configuration with defaults
        self.enabled = self.config.get("stm", {}).get("enabled", True)
        self.ttl = self.config.get("stm", {}).get("ttl_sec", ttl_sec)
        
        # Trigger settings
        trigger_config = self.config.get("stm", {}).get("trigger", {})
        self.trigger_msgs_high = trigger_config.get("msgs_high", self.TRIGGER_MSGS_HIGH)
        self.trigger_msgs_low = trigger_config.get("msgs_low", self.TRIGGER_MSGS_LOW)
        self.trigger_tokens_high = trigger_config.get("tokens_high", self.TRIGGER_TOKENS_HIGH)
        self.trigger_tokens_low = trigger_config.get("tokens_low", self.TRIGGER_TOKENS_LOW)
        self.cooldown_sec = trigger_config.get("cooldown_sec", self.DEFAULT_COOLDOWN_SEC)
        
        # Slot settings
        slots_config = self.config.get("stm", {}).get("slots", {})
        self.max_facts = slots_config.get("facts_max_items", self.MAX_FACTS)
        self.max_open_loops = slots_config.get("open_loops_max_items", self.MAX_OPEN_LOOPS)
        self.max_refs = slots_config.get("refs_max_items", self.MAX_REFS)
        self.max_glossary = slots_config.get("glossary_max_items", self.MAX_GLOSSARY)
        self.max_action_items = slots_config.get("actions_max_items", self.MAX_ACTION_ITEMS)
        self.max_summary_items = slots_config.get("summary_max_items", self.MAX_SUMMARY_ITEMS)
        self.hamming_threshold = slots_config.get("facts_hamm_thresh", self.HAMMING_THRESHOLD)
        
        # Decay settings
        decay_config = self.config.get("stm", {}).get("decay", {})
        self.half_life_min = decay_config.get("half_life_min", 240)
        self.min_score_keep = decay_config.get("min_score_keep", 0.1)
        
        # Injection settings
        inject_config = self.config.get("stm", {}).get("inject", {})
        self.inject_top_facts = inject_config.get("top_facts", 3)
        self.inject_top_loops = inject_config.get("top_open_loops", 2)
        self.inject_top_refs = inject_config.get("top_refs", 3)
        self.inject_top_actions = inject_config.get("top_actions", 3)
        self.include_when_empty = inject_config.get("include_when_empty", False)
        self.max_chars_budget = inject_config.get("max_chars_budget", 1200)
    
    async def get_stm(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve STM data for a session.
        
        Returns:
            Dict with structured slots: summary_v1, salient_facts, open_loops, 
            glossary, refs, persona_ctx, ts_updated or None
        """
        if not self.redis_client:
            return None
            
        try:
            stm_data = {}
            raw = await self.redis_client.get(f"stm:{session_id}")
            if raw:
                stm_data = json.loads(raw)
            last_raw = await self.redis_client.get(f"stm:last:{session_id}")
            if last_raw:
                try:
                    stm_data["last_assistant_msg"] = json.loads(last_raw)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode last assistant cache", extra={"session_id": session_id})
            if stm_data:
                logger.debug("STM retrieved", extra={
                    "session_id": session_id,
                    "facts_count": len(stm_data.get("salient_facts", [])),
                    "loops_count": len(stm_data.get("open_loops", [])),
                    "refs_count": len(stm_data.get("refs", []))
                })
                return stm_data
        except Exception as e:
            logger.error("Failed to retrieve STM", extra={
                "session_id": session_id,
                "error": str(e)
            })
        
        return None
    
    async def consider_update_stm(self, session_id: str, last_messages: List[Dict[str, Any]]) -> bool:
        """
        Consider updating STM based on triggers and cooldown.
        
        Args:
            session_id: Session identifier
            last_messages: Recent messages to process
            
        Returns:
            True if STM was updated, False otherwise
        """
        if not self.enabled:
            return False
        
        # Count messages and estimate tokens
        message_count = len(last_messages)
        token_count = sum(len(str(msg.get("content", ""))) for msg in last_messages) // 4
        has_new_assistant = any(msg.get("role") == "assistant" for msg in last_messages)
        
        # Check if we should update
        should_update = await self.should_update_stm(session_id, message_count, token_count)
        
        if should_update:
            await self.update_stm(session_id, last_messages)
            return True
        
        # Ensure most recent assistant reply is still cached for prompt injection
        if has_new_assistant:
            await self._cache_last_assistant_message(session_id, last_messages)
        
        return False
    
    async def should_update_stm(
        self,
        session_id: str,
        message_count: int,
        token_count: int,
        trigger_msgs_high: Optional[int] = None,
        trigger_msgs_low: Optional[int] = None,
        trigger_tokens_high: Optional[int] = None,
        trigger_tokens_low: Optional[int] = None,
        cooldown_sec: Optional[int] = None,
    ) -> bool:
        """
        Determine if STM should be updated with hysteresis and cooldown.
        
        Args:
            session_id: Session identifier
            message_count: Number of messages since last update
            token_count: Approximate token count since last update
            trigger_msgs_high/low: Message count thresholds (uses config if None)
            trigger_tokens_high/low: Token count thresholds (uses config if None)
            cooldown_sec: Minimum seconds between updates (uses config if None)
            
        Returns:
            True if STM should be updated
        """
        if not self.redis_client or not self.enabled:
            return False
        
        # Use config values if not provided (safe substitution for falsy values)
        if trigger_msgs_high is None:
            trigger_msgs_high = self.trigger_msgs_high
        if trigger_msgs_low is None:
            trigger_msgs_low = self.trigger_msgs_low
        if trigger_tokens_high is None:
            trigger_tokens_high = self.trigger_tokens_high
        if trigger_tokens_low is None:
            trigger_tokens_low = self.trigger_tokens_low
        if cooldown_sec is None:
            cooldown_sec = self.cooldown_sec
            
        try:
            meta_key = f"stm:meta:{session_id}"
            now = time.time()
            meta = await self.redis_client.hgetall(meta_key)
            
            last_ts = float(meta.get(b"last_update_ts", 0) or 0)
            since = now - last_ts
            is_bootstrap = last_ts == 0
            if is_bootstrap and message_count > 0:
                logger.debug("STM bootstrap update", extra={
                    "session_id": session_id,
                    "msgs": message_count,
                    "tokens": token_count
                })
                return True
            
            # Check cooldown
            if since < cooldown_sec:
                logger.debug("STM update skipped due to cooldown", extra={
                    "session_id": session_id,
                    "since_last_sec": since,
                    "cooldown_sec": cooldown_sec
                })
                return False
            
            # Hysteresis logic
            soft_msgs = max(2, trigger_msgs_low // 2 or 1)
            soft_tokens = max(200, trigger_tokens_low // 2 or 50)
            should_update = (
                message_count >= trigger_msgs_high or 
                token_count >= trigger_tokens_high or
                (message_count >= trigger_msgs_low and token_count >= trigger_tokens_low) or
                (since >= cooldown_sec * 2 and (message_count >= soft_msgs or token_count >= soft_tokens))
            )
            
            logger.debug("STM update decision", extra={
                "session_id": session_id,
                "since_last_sec": since,
                "msgs": message_count,
                "tokens": token_count,
                "decision": should_update
            })
            
            return should_update
            
        except Exception as e:
            logger.error("Failed to check STM update conditions", extra={
                "session_id": session_id,
                "error": str(e)
            })
            return False
    
    async def mark_updated(self, session_id: str) -> None:
        """Mark STM as updated with current timestamp."""
        if not self.redis_client:
            return
            
        try:
            meta_key = f"stm:meta:{session_id}"
            await self.redis_client.hset(meta_key, mapping={
                "last_update_ts": str(time.time())
            })
            # Set TTL for meta key
            await self.redis_client.expire(meta_key, self.ttl)
        except Exception as e:
            logger.error("Failed to mark STM as updated", extra={
                "session_id": session_id,
                "error": str(e)
            })
    
    def _simhash64(self, text: str) -> int:
        """
        Generate 64-bit SimHash for text deduplication.
        
        Uses bigrams and bit counting for semantic similarity detection.
        """
        text = text.lower()
        grams = [text[i:i+2] for i in range(len(text)-1)]
        bit_counts = [0] * 64
        
        for g in grams:
            h = int(hashlib.blake2b(g.encode('utf-8'), digest_size=8).hexdigest(), 16)
            for b in range(64):
                bit_counts[b] += (1 if (h >> b) & 1 else -1)
        
        sig = 0
        for b, c in enumerate(bit_counts):
            if c >= 0:
                sig |= (1 << b)
        return sig
    
    def _hamming_distance(self, a: int, b: int) -> int:
        """Calculate Hamming distance between two SimHash signatures."""
        return (a ^ b).bit_count()
    
    async def update_stm(self, session_id: str, last_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update STM with structured slots and semantic deduplication.
        
        Args:
            session_id: Session identifier
            last_messages: Recent messages to compress
            
        Returns:
            Updated STM data with structured slots
        """
        if not self.redis_client:
            logger.warning("Redis client not available, skipping STM update")
            return {}
        
        start_time = time.time()
        
        try:
            # Get existing STM
            existing_stm = await self.get_stm(session_id) or {}
            
            # Extract basic refs first
            refs = self._extract_refs(last_messages)
            
            # Generate LLM-based summary if SummaryService is available
            summary_v2 = existing_stm.get("summary_v2", [])  # Keep existing if new one is empty
            if self.summary_service:
                try:
                    summary_result = await self.summary_service.summarize(session_id, last_messages)
                    new_summary_v2 = summary_result.get("bullets", [])
                    # Only update if we got meaningful content
                    if new_summary_v2:
                        summary_v2 = new_summary_v2
                        # Merge refs from summary with extracted refs
                        summary_refs = summary_result.get("refs", [])
                        if summary_refs:
                            refs = self._merge_refs(refs, summary_refs)
                except Exception as e:
                    logger.error("Summary generation failed", extra={
                        "session_id": session_id,
                        "error": str(e)
                    })
                    # Don't fallback to local summary if we have existing summary_v2
                    if not summary_v2:
                        summary_v2 = self._generate_running_summary(last_messages)
            else:
                # Fallback to local summary only if no existing summary_v2
                if not summary_v2:
                    summary_v2 = self._generate_running_summary(last_messages)
            
            # Keep legacy summary_v1 for backward compatibility (avoid double calculation)
            summary_v1 = summary_v2 or self._generate_running_summary(last_messages)
            
            # Extract other structured data
            salient_facts = self._extract_salient_facts_structured(last_messages)
            open_loops = self._extract_open_loops_structured(last_messages)
            glossary = self._extract_glossary(last_messages)
            action_items = self._extract_action_items(last_messages)
            new_assistant_msg = self._extract_last_assistant_message(last_messages)
            last_assistant_msg = new_assistant_msg or existing_stm.get("last_assistant_msg")
            
            # Merge with existing data using SimHash deduplication
            merged_facts = self._merge_facts_structured(
                existing_stm.get("salient_facts", []),
                salient_facts
            )
            merged_loops = self._merge_loops_structured(
                existing_stm.get("open_loops", []),
                open_loops
            )
            merged_glossary = self._merge_glossary_structured(
                existing_stm.get("glossary", []),
                glossary
            )
            merged_refs = self._merge_refs(
                existing_stm.get("refs", []),
                refs
            )
            merged_actions = self._merge_structured_items(
                existing_stm.get("action_items", []),
                action_items,
                self.hamming_threshold
            )
            
            # Apply decay to existing items
            merged_facts = self._apply_decay(merged_facts)
            merged_loops = self._apply_decay(merged_loops)
            merged_glossary = self._apply_decay(merged_glossary)
            merged_actions = self._apply_decay(merged_actions)
            
            # Create new STM with structured slots
            stm = {
                "summary_v1": summary_v1,  # Legacy summary for backward compatibility
                "summary_v2": summary_v2,  # LLM-generated summary (primary)
                "salient_facts": merged_facts[:self.max_facts],
                "open_loops": merged_loops[:self.max_open_loops],
                "glossary": merged_glossary[:self.max_glossary],
                "refs": merged_refs[:self.max_refs],
                "action_items": merged_actions[:self.max_action_items],
                "persona_ctx": existing_stm.get("persona_ctx", {}),
                "ts_updated": time.time()
            }
            if last_assistant_msg:
                stm["last_assistant_msg"] = last_assistant_msg
            
            # Save to Redis
            await self.redis_client.set(
                f"stm:{session_id}",
                json.dumps(stm, ensure_ascii=False),
                ex=self.ttl
            )
            if new_assistant_msg:
                await self._cache_last_assistant_message(session_id, message=new_assistant_msg)
            
            # Mark as updated
            await self.mark_updated(session_id)
            
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info("STM updated", extra={
                "session_id": session_id,
                "facts_in": len(salient_facts),
                "facts_out": len(merged_facts),
                "loops_in": len(open_loops),
                "loops_out": len(merged_loops),
                "refs_in": len(refs),
                "refs_out": len(merged_refs),
                "actions_in": len(action_items),
                "actions_out": len(merged_actions),
                "summary_len": len(summary_v1),
                "latency_ms": latency_ms
            })
            
            return stm
            
        except Exception as e:
            logger.error("Failed to update STM", extra={
                "session_id": session_id,
                "error": str(e)
            })
            return {}
    
    def _generate_running_summary(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Generate running summary as structured bullet points.
        
        Returns:
            List of summary bullets (max 8, each max 140 chars)
        """
        if not messages:
            return []
        
        bullets = []
        
        # 1) Recent user questions
        for msg in reversed(messages):
            if len(bullets) >= self.MAX_SUMMARY_ITEMS:
                break
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    condensed = self._condense_text(content, self.MAX_SUMMARY_LEN)
                    if condensed and condensed.endswith("?"):
                        bullets.append(f"Q: {condensed}")
        
        # 2) Key assistant responses
        for msg in reversed(messages):
            if len(bullets) >= self.MAX_SUMMARY_ITEMS:
                break
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    condensed = self._condense_text(content, self.MAX_SUMMARY_LEN)
                    if condensed:
                        bullets.append(f"A: {condensed}")
        
        return bullets[:self.max_summary_items]
    
    def _condense_text(self, text: str, max_len: int) -> str:
        """Condense text to max length while preserving meaning."""
        if len(text) <= max_len:
            return text.strip()
        
        # Try to break at sentence boundary
        sentences = text.split('. ')
        result = ""
        for sentence in sentences:
            if len(result + sentence + '. ') <= max_len:
                result += sentence + '. '
            else:
                break
        
        if result:
            return result.strip()
        
        # Fallback: truncate at word boundary
        words = text.split()
        result = ""
        for word in words:
            if len(result + word + ' ') <= max_len:
                result += word + ' '
            else:
                break
        
        return result.strip()
    
    def _extract_last_assistant_message(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Pick the most recent assistant reply and condense it for prompt injection."""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            condensed = self._condense_text(content, self.MAX_SUMMARY_LEN)
            if condensed:
                return {
                    "text": condensed,
                    "ts": time.time()
                }
        return None
    
    def _extract_salient_facts_structured(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract salient facts as structured objects with scores and signatures."""
        facts = []
        
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 20:
                continue
                
            # Look for factual statements
            if any(indicator in content.lower() for indicator in 
                   ["is", "are", "was", "were", "has", "have", "means", "refers to", "defines", "indicates"]):
                
                # Extract concise fact
                fact_text = self._extract_concise_fact(content)
                if not fact_text:
                    continue
                
                fact = {
                    "text": fact_text[:200],  # Limit length
                    "score": 1.0 + (0.5 if "?" not in fact_text else 0.0),  # Boost non-questions
                    "ts": time.time(),
                    "sig": self._simhash64(fact_text)
                }
                facts.append(fact)
        
        # Sort by score and recency
        facts.sort(key=lambda x: (-x["score"], -x["ts"]))
        return facts[:10]
    
    def _extract_concise_fact(self, content: str) -> str:
        """Extract concise factual statement from content."""
        # Look for sentences with factual indicators
        sentences = content.split('. ')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in 
                   ["is", "are", "was", "were", "has", "have", "means", "refers to"]):
                # Clean up the sentence
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 200:
                    return sentence
        
        # Fallback: return first sentence if it's reasonable
        first_sentence = content.split('.')[0].strip()
        if 20 <= len(first_sentence) <= 200:
            return first_sentence
        
        return ""
    
    def _extract_open_loops_structured(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract open questions and unresolved topics as structured objects."""
        open_loops = []
        
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
                
            # Look for questions
            if "?" in content:
                question = self._extract_question(content)
                if question:
                    loop = {
                        "text": question[:150],
                        "score": 1.0,
                        "ts": time.time(),
                        "sig": self._simhash64(question)
                    }
                    open_loops.append(loop)
            
            # Look for incomplete statements
            elif any(content.lower().endswith(marker) for marker in 
                     ["but", "however", "although", "though", "need to", "should", "must"]):
                loop = {
                    "text": content[:150],
                    "score": 0.8,
                    "ts": time.time(),
                    "sig": self._simhash64(content)
                }
                open_loops.append(loop)
        
        # Sort by score and recency
        open_loops.sort(key=lambda x: (-x["score"], -x["ts"]))
        return open_loops[:5]
    
    def _extract_question(self, content: str) -> str:
        """Extract the main question from content."""
        # Find sentences with question marks
        sentences = content.split('. ')
        for sentence in sentences:
            if '?' in sentence:
                # Clean up the question
                question = sentence.strip()
                if len(question) > 10 and len(question) <= 150:
                    return question
        
        # Fallback: return content if it's a reasonable length
        if 10 <= len(content) <= 150:
            return content
        
        return ""
    
    def _extract_glossary(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract glossary terms and definitions."""
        glossary = []
        
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
                
            # Look for definition patterns
            if any(marker in content.lower() for marker in 
                   ["means", "refers to", "is defined as", "— это", "означает"]):
                
                term_def = self._extract_term_definition(content)
                if term_def:
                    entry = {
                        "term": term_def["term"],
                        "definition": term_def["definition"],
                        "score": 1.0,
                        "ts": time.time(),
                        "sig": self._simhash64(f"{term_def['term']}: {term_def['definition']}")
                    }
                    glossary.append(entry)
        
        return glossary[:5]
    
    def _extract_action_items(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Capture actionable assistant suggestions/next steps."""
        actions = []
        
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            
            segments = self._split_action_segments(content)
            for segment in segments:
                normalized = segment.strip(" -*•")
                if not normalized or len(normalized) < 8 or len(normalized) > 200:
                    continue
                lowered = normalized.lower()
                if not self._looks_like_action(lowered):
                    continue
                action = {
                    "text": normalized,
                    "score": 1.0 + (0.2 if normalized.endswith(("!", ".")) else 0.0),
                    "ts": time.time(),
                    "sig": self._simhash64(normalized)
                }
                actions.append(action)
        
        # Prioritize the latest actionable suggestions
        actions.sort(key=lambda x: (-x["score"], -x["ts"]))
        return actions[:self.max_action_items]
    
    def _split_action_segments(self, text: str) -> List[str]:
        """Split assistant text into candidate actionable segments."""
        segments: List[str] = []
        for line in re.split(r"[\r\n]+", text):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\-\•\*\d\.\)]+", "", line).strip()
            if not line:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", line)
            for sentence in sentences:
                sentence = sentence.strip(" -*•")
                if sentence:
                    segments.append(sentence)
        return segments
    
    def _looks_like_action(self, lowered_text: str) -> bool:
        """Heuristic check for actionable commands/instructions."""
        prefixes = [
            "please ", "let's ", "let us ", "давай ", "нужно ", "надо ", "попробуй ",
            "постарайся ", "советую ", "рекомендую "
        ]
        for prefix in prefixes:
            if lowered_text.startswith(prefix):
                lowered_text = lowered_text[len(prefix):].lstrip()
                break
        for keyword in self.ACTION_KEYWORDS:
            if lowered_text.startswith(keyword):
                return True
            if f" {keyword}" in lowered_text:
                return True
        return False
    
    def _extract_term_definition(self, content: str) -> Optional[Dict[str, str]]:
        """Extract term and definition from content."""
        # Look for patterns like "X means Y" or "X — это Y"
        patterns = [
            r"([A-Z][a-zA-Z\s]+)\s+(?:means|refers to|is defined as)\s+(.+)",
            r"([A-Z][a-zA-Z\s]+)\s*—\s*это\s+(.+)",
            r"([A-Z][a-zA-Z\s]+)\s*означает\s+(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                # Clean up
                if len(term) > 3 and len(definition) > 5 and len(definition) < 100:
                    return {"term": term, "definition": definition}
        
        return None
    
    def _extract_refs(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract Sefaria references from messages."""
        refs = []
        
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
                
            # Find Sefaria references
            for match in self.TREF_RE.findall(content):
                ref = match.strip()
                if ref not in refs:
                    refs.append(ref)
        
        return refs[:10]
    
    def _merge_facts_structured(self, existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge facts using SimHash deduplication."""
        return self._merge_structured_items(existing, new, self.hamming_threshold)
    
    def _merge_loops_structured(self, existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge open loops using SimHash deduplication."""
        return self._merge_structured_items(existing, new, self.hamming_threshold)
    
    def _merge_glossary_structured(self, existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge glossary entries using SimHash deduplication."""
        return self._merge_structured_items(existing, new, self.hamming_threshold)
    
    def _merge_refs(self, existing: List[str], new: List[str]) -> List[str]:
        """Merge references, removing exact duplicates."""
        all_refs = existing + new
        unique_refs = []
        seen = set()
        for ref in all_refs:
            if ref not in seen:
                unique_refs.append(ref)
                seen.add(ref)
        return unique_refs
    
    def _merge_structured_items(self, existing: List[Dict[str, Any]], new: List[Dict[str, Any]], threshold: int) -> List[Dict[str, Any]]:
        """Merge structured items using SimHash deduplication."""
        result = []
        
        # Add existing items
        for item in existing:
            if "sig" not in item:
                item["sig"] = self._simhash64(item.get("text", ""))
            result.append(item)
        
        # Add new items, checking for duplicates
        for new_item in new:
            if "sig" not in new_item:
                new_item["sig"] = self._simhash64(new_item.get("text", ""))
            
            # Check if similar item already exists
            is_duplicate = False
            for existing_item in result:
                if self._hamming_distance(new_item["sig"], existing_item["sig"]) <= threshold:
                    # Update existing item with higher score and newer timestamp
                    if new_item.get("score", 0) > existing_item.get("score", 0):
                        existing_item.update(new_item)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result.append(new_item)
        
        # Sort by score and recency
        result.sort(key=lambda x: (-x.get("score", 0), -x.get("ts", 0)))
        return result
    
    async def _cache_last_assistant_message(
        self,
        session_id: str,
        last_messages: Optional[List[Dict[str, Any]]] = None,
        message: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store the most recent assistant reply separately for lightweight recall."""
        if not self.redis_client:
            return
        
        if message is None and last_messages:
            message = self._extract_last_assistant_message(last_messages)
        if not message:
            return
        
        try:
            await self.redis_client.set(
                f"stm:last:{session_id}",
                json.dumps(message, ensure_ascii=False),
                ex=self.ttl
            )
        except Exception as e:
            logger.error("Failed to cache last assistant reply", extra={
                "session_id": session_id,
                "error": str(e)
            })
    
    def _apply_decay(self, items: List[Dict[str, Any]], decay_rate: Optional[float] = None) -> List[Dict[str, Any]]:
        """Apply exponential decay to item scores based on age."""
        import math
        
        now = time.time()
        decayed_items = []
        
        # Calculate decay rate from half-life
        if decay_rate is None:
            # Convert half-life from minutes to hours and calculate decay rate
            half_life_hours = self.half_life_min / 60.0
            decay_rate = math.log(2) / max(half_life_hours, 0.01)  # True exponential decay
        
        for item in items:
            age_hours = (now - item.get("ts", now)) / 3600
            decayed_score = item.get("score", 0) * math.exp(-decay_rate * age_hours)
            
            if decayed_score > self.min_score_keep:  # Keep items with meaningful score
                item["score"] = decayed_score
                decayed_items.append(item)
        
        return decayed_items
    
    async def clear_stm(self, session_id: str) -> bool:
        """Clear STM and metadata for a session."""
        if not self.redis_client:
            return False
        
        try:
            # Clear both STM data and metadata
            await self.redis_client.delete(f"stm:{session_id}")
            await self.redis_client.delete(f"stm:meta:{session_id}")
            await self.redis_client.delete(f"stm:last:{session_id}")
            
            logger.info("STM cleared", extra={"session_id": session_id})
            return True
        except Exception as e:
            logger.error("Failed to clear STM", extra={
                "session_id": session_id,
                "error": str(e)
            })
            return False
    
    def format_stm_for_prompt(self, stm: Dict[str, Any], max_facts: Optional[int] = None, max_loops: Optional[int] = None, max_refs: Optional[int] = None, max_actions: Optional[int] = None, max_chars_budget: Optional[int] = None) -> str:
        """
        Format STM data for inclusion in LLM prompts.
        
        Args:
            stm: STM data dictionary
            max_facts: Maximum number of facts to include (uses config if None)
            max_loops: Maximum number of open loops to include (uses config if None)
            max_refs: Maximum number of references to include (uses config if None)
            max_actions: Maximum number of action items to include (uses config if None)
            
        Returns:
            Formatted string for prompt inclusion
        """
        if not stm and not self.include_when_empty:
            return ""
        
        # Use config values if not provided
        max_facts = max_facts or self.inject_top_facts
        max_loops = max_loops or self.inject_top_loops
        max_refs = max_refs or self.inject_top_refs
        max_actions = max_actions or self.inject_top_actions
        if max_chars_budget is None:
            max_chars_budget = self.max_chars_budget
        
        parts = []
        budget = 0
        
        def add_line(s: str) -> bool:
            nonlocal budget
            if budget + len(s) + 1 > max_chars_budget:
                return False
            parts.append(s)
            budget += len(s) + 1
            return True
        
        # Always include the most recent assistant reply when available
        recent_reply = (stm.get("last_assistant_msg") or {}) if isinstance(stm, dict) else {}
        recent_text = recent_reply.get("text") if isinstance(recent_reply, dict) else ""
        if recent_text:
            if not add_line("[Recent Assistant Reply]"):
                return "\n".join(parts)
            if not add_line(f"- {recent_text}"):
                return "\n".join(parts)
            add_line("")
        
        # Add summary bullets (prefer summary_v2, fallback to summary_v1) - highest priority
        summary = stm.get("summary_v2", []) or stm.get("summary_v1", [])
        if summary:
            if not add_line("[STM Summary]"):
                return "\n".join(parts)
            for bullet in summary[:5]:  # Max 5 bullets
                if not add_line(f"• {bullet}"):
                    return "\n".join(parts)
            add_line("")  # Empty line
        
        # Add top facts - second priority
        facts = stm.get("salient_facts", [])
        if facts:
            if not add_line("[Key Facts]"):
                return "\n".join(parts)
            for fact in facts[:max_facts]:
                if not add_line(f"• {fact.get('text', '')}"):
                    return "\n".join(parts)
            add_line("")
        
        # Add open loops - third priority
        loops = stm.get("open_loops", [])
        if loops:
            if not add_line("[Open Questions]"):
                return "\n".join(parts)
            for loop in loops[:max_loops]:
                if not add_line(f"• {loop.get('text', '')}"):
                    return "\n".join(parts)
            add_line("")
        
        # Add assistant action suggestions - fourth priority
        actions = stm.get("action_items", [])
        if actions:
            if not add_line("[Action Suggestions]"):
                return "\n".join(parts)
            for action in actions[:max_actions]:
                if not add_line(f"• {action.get('text', '')}"):
                    return "\n".join(parts)
            add_line("")
        
        # Add references - lowest priority
        refs = stm.get("refs", [])
        if refs:
            if not add_line("[References]"):
                return "\n".join(parts)
            for ref in refs[:max_refs]:
                if not add_line(f"• {ref}"):
                    return "\n".join(parts)
            add_line("")
        
        return "\n".join(parts).strip()
    
    async def get_stm_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about STM for a session."""
        stm = await self.get_stm(session_id)
        if not stm:
            return {}
        
        return {
            "summary_items": len(stm.get("summary_v1", [])),
            "facts_count": len(stm.get("salient_facts", [])),
            "loops_count": len(stm.get("open_loops", [])),
            "glossary_count": len(stm.get("glossary", [])),
            "action_count": len(stm.get("action_items", [])),
            "refs_count": len(stm.get("refs", [])),
            "last_updated": stm.get("ts_updated", 0),
            "age_hours": (time.time() - stm.get("ts_updated", 0)) / 3600
        }
