import logging
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
from collections import defaultdict

import redis.asyncio as redis
from models.chat_models import Session
from domain.chat.tools import ToolRegistry
from brain_service.models.db import UserApiKey
from brain_service.services.user_service import UserService, ApiKeyLimitExceeded
from core.dependencies import get_memory_service
from .block_stream_service import BlockStreamService
from core.llm_config import get_llm_for_task, LLMConfigError, get_tooling_config
from config import personalities as personality_service

logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for handling chat functionality including session management,
    LLM streaming, and tool integration.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        tool_registry: ToolRegistry,
        memory_service,
        user_service: UserService,
    ):
        self.redis_client = redis_client
        self.tool_registry = tool_registry
        self.memory_service = memory_service
        self.user_service = user_service
        self.block_stream_service = BlockStreamService()
        self._session_ttl_seconds = 3600 * 24 * 7

    @staticmethod
    def _normalize_user_id(user_id: str) -> tuple[str, uuid.UUID]:
        try:
            user_uuid = uuid.UUID(str(user_id))
        except ValueError as exc:
            raise ValueError(f"Invalid user_id: {user_id}") from exc
        return str(user_uuid), user_uuid

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        return f"session:{user_id}:{session_id}"
    
    async def get_session_from_redis(self, session_id: str, user_id: str, agent_id: str) -> Session:
        """Retrieve session from Redis or create new one."""
        user_id_str, _ = self._normalize_user_id(user_id)
        if self.redis_client:
            redis_key = self._session_key(user_id_str, session_id)
            session_data = await self.redis_client.get(redis_key)
            if session_data:
                try:
                    session = Session.from_dict(json.loads(session_data))
                    if agent_id and session.agent_id != agent_id:
                        session.agent_id = agent_id
                    return session
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Failed to decode session {session_id}: {e}")
        return Session(user_id=user_id_str, agent_id=agent_id, persistent_session_id=session_id)
    
    async def save_session_to_redis(self, session: Session):
        """Save session to Redis."""
        if not self.redis_client:
            return
        user_id_str, _ = self._normalize_user_id(session.user_id)
        redis_key = self._session_key(user_id_str, session.persistent_session_id)
        session.last_modified = datetime.now().isoformat()
        await self.redis_client.set(
            redis_key,
            json.dumps(session.to_dict()),
            ex=self._session_ttl_seconds,
        )
    
    async def get_llm_response_stream(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        *,
        user_uuid: Optional[uuid.UUID] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate LLM response stream with tool support and STM integration.
        
        Args:
            messages: List of message dictionaries
            session_id: Session ID for STM integration
            user_uuid: Optional user identifier for API key lookup
            
        Yields:
            JSON strings with streaming events
        """
        key_record = None
        stream_emitted = False
        try:
            client, model, reasoning_params, caps, key_record = await self._prepare_llm_client("CHAT", user_uuid)
        except ApiKeyLimitExceeded:
            yield json.dumps({"type": "error", "data": {"message": "Daily usage limit reached for your API key. Please contact an administrator."}}) + "\n"
            return
        except LLMConfigError as e:
            yield json.dumps({"type": "error", "data": {"message": f"LLM not configured: {e}"}}) + '\n'
            return
        except Exception as exc:
            logger.error("Failed to prepare LLM client", extra={"session_id": session_id, "error": str(exc)})
            yield json.dumps({"type": "error", "data": {"message": "Unable to prepare language model client."}}) + '\n'
            return

        # Integrate STM if available
        if self.memory_service:
            stm_data = await self.memory_service.get_stm(session_id)
            if stm_data:
                stm_context = self.memory_service.format_stm_for_prompt(stm_data)
                if stm_context:
                    stm_message = {"role": "system", "content": f"[STM Context]\n{stm_context}"}
                    messages.insert(0, stm_message)

        tooling_cfg = get_tooling_config()
        parallel_tool_calls = bool(tooling_cfg.get("parallel_tool_calls", False))
        retry_on_empty_stream = bool(tooling_cfg.get("retry_on_empty_stream", True))

        tools = self.tool_registry.get_tool_schemas()
        api_params = {**reasoning_params, "model": model, "messages": messages, "stream": True}
        if tools:
            api_params.update({"tools": tools, "tool_choice": "auto", "parallel_tool_calls": parallel_tool_calls})

        iter_count = 0
        empty_reply_retry = False

        def _is_tool_directive(text: str) -> bool:
            stripped = (text or "").strip()
            return stripped.startswith("[TOOL_CALLS") or stripped.startswith("[CALL_TOOL")

        try:
            while iter_count < 5:
                iter_count += 1

                if len(messages) > 20:
                    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
                    recent_messages = messages[-19:]
                    messages = ([system_msg] + recent_messages) if system_msg else recent_messages

                stream = await client.chat.completions.create(**api_params)

                tool_call_builders = defaultdict(lambda: {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                full_reply_content = ""
                chunk_count = 0

                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        chunk_count += 1
                        stream_emitted = True
                        if _is_tool_directive(delta.content):
                            logger.debug("Filtered tool directive chunk", extra={"session_id": session_id, "content": delta.content})
                            continue
                        full_reply_content += delta.content
                        yield json.dumps({"type": "llm_chunk", "data": delta.content}) + '\n'
                    if delta and delta.tool_calls:
                        for tc in delta.tool_calls:
                            builder = tool_call_builders[tc.index]
                            builder["index"] = tc.index
                            if tc.id:
                                builder["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    builder["function"]["name"] = tc.function.name
                                if tc.function.arguments:
                                    builder["function"]["arguments"] += tc.function.arguments

                if not tool_call_builders:
                    if chunk_count > 0:
                        stripped_reply = full_reply_content.strip()
                        if stripped_reply:
                            return
                        if retry_on_empty_stream and not empty_reply_retry and iter_count < 5:
                            empty_reply_retry = True
                            logger.warning(
                                "LLM returned only whitespace after tool use; retrying with explicit answer instruction.",
                                extra={"session_id": session_id, "model": model},
                            )
                            messages.append({"role": "system", "content": "You must now answer the user's last question in natural language. Summarize the tool findings and provide a helpful chat response."})
                            api_params.pop("tools", None)
                            api_params["tool_choice"] = "none"
                            api_params["parallel_tool_calls"] = False
                            api_params["messages"] = messages
                            continue
                        fallback_message = ("I was unable to generate a helpful answer after consulting available tools. Please rephrase the question or try again.")
                        stream_emitted = True
                        yield json.dumps({"type": "llm_chunk", "data": fallback_message}) + '\n'
                        yield json.dumps({"type": "end", "data": "Stream finished"}) + '\n'
                        return

                    try:
                        parsed_content, _ = self._find_valid_json_prefix(full_reply_content)
                        if parsed_content is None:
                            stream_emitted = True
                            yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                            return
                        if isinstance(parsed_content, dict):
                            if ((parsed_content.get("type") == "doc.v1" and "blocks" in parsed_content) or
                                ("blocks" in parsed_content and isinstance(parsed_content["blocks"], list))):
                                stream_emitted = True
                                yield json.dumps({"type": "doc_v1", "data": parsed_content}) + '\n'
                                return
                            elif (parsed_content.get("version") == "doc.v1" and
                                  "content" in parsed_content and isinstance(parsed_content["content"], list)):
                                try:
                                    doc_v1_data = {"type": "doc.v1", "blocks": parsed_content["content"]}
                                    if self._validate_doc_v1_structure(doc_v1_data):
                                        stream_emitted = True
                                        yield json.dumps({"type": "doc_v1", "data": doc_v1_data}) + '\n'
                                    else:
                                        stream_emitted = True
                                        yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                                    return
                                except Exception as e:
                                    logger.error(f"Error processing doc.v1 content: {e}")
                                    stream_emitted = True
                                    yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                                    return
                            elif ("doc" in parsed_content and isinstance(parsed_content["doc"], dict) and
                                  "content" in parsed_content["doc"] and isinstance(parsed_content["doc"]["content"], list)):
                                doc_data = parsed_content["doc"]
                                doc_v1_data = {"type": "doc.v1", "blocks": doc_data.get("content", [])}
                                if self._validate_doc_v1_structure(doc_v1_data):
                                    stream_emitted = True
                                    yield json.dumps({"type": "doc_v1", "data": doc_v1_data}) + '\n'
                                else:
                                    stream_emitted = True
                                    yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                                return
                            else:
                                stream_emitted = True
                                yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                                return
                        else:
                            stream_emitted = True
                            yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                            return
                    except Exception as e:
                        logger.error(f"Error parsing LLM response: {e}", exc_info=True)
                        stream_emitted = True
                        yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                        return

                tool_sorted = sorted(tool_call_builders.values(), key=lambda x: x.get("index", 0))
                tool_events, tool_results = await self._handle_tool_calls(tool_sorted, messages, session_id)

                if not tool_events:
                    stream_emitted = True
                    yield json.dumps({"type": "full_response", "data": full_reply_content}) + '\n'
                    return

                for event in tool_events:
                    stream_emitted = True
                    yield json.dumps(event) + '\n'

                messages.extend(tool_results)
                api_params["messages"] = messages

            yield json.dumps({"type": "end", "data": "Stream finished"}) + '\n'
        except Exception as e:
            logger.error("LLM streaming error", extra={"session_id": session_id, "error": str(e)})
            yield json.dumps({"type": "error", "data": {"message": str(e)}}) + '\n'
        finally:
            if key_record and stream_emitted:
                await self.user_service.increment_api_usage(key_record.id)

    async def _prepare_llm_client(
        self, task: str, user_uuid: Optional[uuid.UUID]
    ):
        api_key_override: Optional[str] = None
        key_record: Optional[UserApiKey] = None
        if user_uuid is not None:
            try:
                result = await self.user_service.get_active_api_key(user_uuid, provider="openrouter")
                if result:
                    key_record, api_key_override = result
            except ApiKeyLimitExceeded:
                raise
        provider_override = key_record.provider if key_record else None
        client, model, reasoning_params, caps = get_llm_for_task(
            task,
            api_key_override=api_key_override,
            provider_override=provider_override,
        )
        return client, model, reasoning_params, caps, key_record

    async def get_llm_response_stream_with_blocks(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        *,
        user_uuid: Optional[uuid.UUID] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate LLM response stream with block-by-block streaming.
        
        Args:
            messages: List of message dictionaries
            session_id: Session ID for STM integration
            user_uuid: Optional user identifier for API key lookup
            
        Yields:
            JSON strings with streaming events (including block events)
        """
        key_record = None
        try:
            client, model, reasoning_params, caps, key_record = await self._prepare_llm_client("CHAT", user_uuid)
        except ApiKeyLimitExceeded:
            yield json.dumps({"type": "error", "data": {"message": "Daily usage limit reached for your API key. Please contact an administrator."}}) + "\n"
            return
        except LLMConfigError as e:
            yield json.dumps({"type": "error", "data": {"message": f"LLM not configured: {e}"}}) + '\n'
            return

        if self.memory_service:
            stm_data = await self.memory_service.get_stm(session_id)
            if stm_data:
                stm_context = self.memory_service.format_stm_for_prompt(stm_data)
                if stm_context:
                    stm_message = {"role": "system", "content": f"[STM Context]\n{stm_context}"}
                    messages.insert(0, stm_message)

        tools = self.tool_registry.get_tool_schemas()
        api_params = {**reasoning_params, "model": model, "messages": messages, "stream": True}
        if tools:
            api_params.update({"tools": tools, "tool_choice": "auto"})

        try:
            stream = await client.chat.completions.create(**api_params)

            async def text_stream():
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            async for block_event in self.block_stream_service.stream_blocks_from_text(text_stream(), session_id):
                yield json.dumps(block_event) + '\n'

            yield json.dumps({"type": "end", "data": "Stream finished"}) + '\n'
        except Exception as e:
            logger.error(f"Error in LLM stream with blocks: {e}", exc_info=True)
            yield json.dumps({"type": "error", "data": {"message": str(e)}}) + '\n'
        finally:
            if key_record:
                await self.user_service.increment_api_usage(key_record.id)

    def _find_valid_json_prefix(self, buffer: str) -> tuple[Optional[Dict[str, Any]], int]:
        """
        Find the last valid JSON prefix in buffer.
        
        Returns:
            Tuple of (parsed_object, end_position) or (None, 0) if no valid prefix
        """
        if not buffer.strip():
            return None, 0
        
        # Try to find valid JSON prefix by testing progressively longer substrings
        max_pos = len(buffer)
        
        # Start from the end and work backwards to find the longest valid prefix
        for end_pos in range(max_pos, 0, -1):
            test_str = buffer[:end_pos].strip()
            if not test_str:
                continue
                
            try:
                # Try to parse as JSON
                obj = json.loads(test_str)
                if isinstance(obj, dict):
                    return obj, end_pos
            except json.JSONDecodeError:
                # Try to find a complete object within the string
                try:
                    # Look for complete objects by counting braces
                    brace_count = 0
                    start_pos = 0
                    in_string = False
                    escape_next = False
                    
                    for i, char in enumerate(test_str):
                        if escape_next:
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            escape_next = True
                            continue
                            
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                            
                        if not in_string:
                            if char == '{':
                                if brace_count == 0:
                                    start_pos = i
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found complete object
                                    obj_str = test_str[start_pos:i+1]
                                    obj = json.loads(obj_str)
                                    if isinstance(obj, dict):
                                        return obj, start_pos + len(obj_str)
                    
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return None, 0
    
    def _validate_doc_v1_structure(self, doc_data: Dict[str, Any]) -> bool:
        """
        Validate doc.v1 structure to ensure it's complete and valid.
        
        Args:
            doc_data: The doc.v1 data to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check required fields
            if not isinstance(doc_data, dict):
                return False
            
            if "type" not in doc_data or doc_data["type"] != "doc.v1":
                return False
            
            if "blocks" not in doc_data or not isinstance(doc_data["blocks"], list):
                return False
            
            # Validate each block
            for block in doc_data["blocks"]:
                if not isinstance(block, dict):
                    return False
                
                if "type" not in block:
                    return False
                
                # Check for required content field
                if "content" not in block:
                    return False
                
                # Validate content based on type
                block_type = block["type"]
                content = block["content"]
                
                if block_type == "paragraph" and not isinstance(content, str):
                    return False
                elif block_type == "heading" and not isinstance(content, str):
                    return False
                elif block_type == "list" and not isinstance(content, list):
                    return False
                # Add more validations as needed
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating doc.v1 structure: {e}")
            return False
    
    async def process_chat_stream(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat stream request.
        
        Args:
            text: User's message text
            user_id: User identifier
            session_id: Optional session ID
            agent_id: Optional agent ID
            
        Yields:
            JSON strings with streaming events
        """
        logger.info("--- New General Chat Request ---")

        user_id_str, user_uuid = self._normalize_user_id(user_id)
        active_session_id = session_id or str(uuid.uuid4())
        session = await self.get_session_from_redis(
            active_session_id,
            user_id_str,
            agent_id or "default"
        )

        session.add_message(role="user", content=text)

        personality_config = personality_service.get_personality(session.agent_id) or {}
        system_prompt = personality_config.get("system_prompt", "You are a helpful assistant.")

        prompt_messages = [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in session.short_term_memory]

        accumulated_chunks: List[str] = []
        doc_response: Optional[Dict[str, Any]] = None
        final_text: Optional[str] = None
        error_message: Optional[str] = None

        async for chunk in self.get_llm_response_stream(
            prompt_messages, session.persistent_session_id, user_uuid=user_uuid
        ):
            yield chunk
            try:
                event = json.loads(chunk)
                event_type = event.get("type")
                if event_type == "llm_chunk":
                    accumulated_chunks.append(event.get("data", ""))
                elif event_type == "doc_v1":
                    doc_response = event.get("data")
                elif event_type == "full_response":
                    final_text = event.get("data")
                elif event_type == "error":
                    error_message = event.get("data", {}).get("message")
            except json.JSONDecodeError:
                continue

        if doc_response:
            session.add_message(
                role="assistant",
                content=json.dumps(doc_response),
                content_type="doc.v1",
            )
        else:
            reply = final_text or "".join(accumulated_chunks).strip()
            if reply:
                session.add_message(role="assistant", content=reply, content_type="text.v1")
            elif error_message:
                session.add_message(role="assistant", content=error_message, content_type="text.v1")

        await self.save_session_to_redis(session)

        await self.user_service.upsert_thread(
            session_id=session.persistent_session_id,
            user_id=uuid.UUID(user_id_str),
            title=session.name or "Chat",
            last_modified=datetime.now(),
            metadata={"agent_id": session.agent_id},
        )

        if self.memory_service and session.short_term_memory:
            recent_messages = [m.model_dump() for m in session.short_term_memory[-10:]]
            await self.memory_service.consider_update_stm(session.persistent_session_id, recent_messages)

    async def process_chat_stream_with_blocks(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat stream request with block-by-block streaming.
        
        Args:
            text: User's message text
            user_id: User identifier
            session_id: Optional session ID
            agent_id: Optional agent ID
            
        Yields:
            JSON strings with streaming events (including block events)
        """
        logger.info(f"--- New Block Streaming Chat Request ---")
        
        # Get or create session
        user_id_str, user_uuid = self._normalize_user_id(user_id)
        active_session_id = session_id or str(uuid.uuid4())
        session = await self.get_session_from_redis(
            active_session_id,
            user_id_str,
            agent_id or "default"
        )
        
        # Add user message to session
        session.add_message(role="user", content=text)

        # Get personality configuration
        personality_config = personality_service.get_personality(session.agent_id) or {}
        system_prompt = personality_config.get("system_prompt", "You are a helpful assistant.")
        
        # Build prompt messages
        prompt_messages = [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in session.short_term_memory]
        
        # Stream LLM response with block streaming
        full_response = ""
        final_message = None  # Fix: Track what to save in history
        block_doc = {"version": "1.0", "blocks": []}  # Fix: Aggregate blocks into doc
        block_ids = {}  # Fix: Track block_ids for stable keys
        
        async for chunk in self.get_llm_response_stream_with_blocks(
            prompt_messages, session.persistent_session_id, user_uuid=user_uuid
        ):
            yield chunk
            try:
                event = json.loads(chunk)
                if event.get("type") == "llm_chunk":
                    full_response += event.get("data", "")
                elif event.get("type") == "block_start":
                    # Fix: Track block start with block_id
                    block_data = event.get("data", {})
                    block_index = block_data.get("block_index", 0)
                    block_type = block_data.get("block_type", "paragraph")
                    block_id = block_data.get("block_id", f"block_{block_index}")
                    block_ids[block_index] = block_id
                    # Ensure we have enough blocks in the array
                    while len(block_doc["blocks"]) <= block_index:
                        block_doc["blocks"].append({"type": block_type, "text": "", "block_id": block_id})
                elif event.get("type") == "block_delta":
                    # Fix: Update block content with block_id
                    block_data = event.get("data", {})
                    block_index = block_data.get("block_index", 0)
                    block = block_data.get("block", {})
                    if block_index < len(block_doc["blocks"]):
                        # Preserve block_id from block_start
                        block["block_id"] = block_ids.get(block_index, f"block_{block_index}")
                        block_doc["blocks"][block_index] = block
                elif event.get("type") == "block_end":
                    # Fix: Finalize block with block_id
                    block_data = event.get("data", {})
                    block_index = block_data.get("block_index", 0)
                    if block_index < len(block_doc["blocks"]):
                        block_doc["blocks"][block_index]["finalized"] = True
                        # Ensure block_id is preserved
                        if "block_id" not in block_doc["blocks"][block_index]:
                            block_doc["blocks"][block_index]["block_id"] = block_ids.get(block_index, f"block_{block_index}")
            except json.JSONDecodeError:
                pass

        # Add assistant response to session
        if block_doc["blocks"]:
            # Fix: Guarantee finalization of all blocks before saving
            for i, block in enumerate(block_doc["blocks"]):
                if "finalized" not in block:
                    block["finalized"] = True
                if "block_id" not in block:
                    block["block_id"] = block_ids.get(i, f"block_{i}")
            
            # Fix: Save aggregated doc.v1 with unified content_type
            final_message = {
                "content": json.dumps(block_doc),
                "content_type": "doc.v1"
            }
            session.add_message(
                role="assistant", 
                content=final_message["content"],
                content_type=final_message["content_type"]
            )
        elif full_response.strip():
            # Fallback to text if no blocks
            session.add_message(role="assistant", content=full_response.strip())

        # Save session first
        await self.save_session_to_redis(session)

        # Update STM after stream completion (write-after-final)
        if self.memory_service and (block_doc["blocks"] or full_response.strip()):
            # Prepare recent messages for STM update
            recent_messages = [m.model_dump() for m in session.short_term_memory[-10:]]  # Last 10 messages
            
            # Use consider_update_stm which handles all the logic
            updated = await self.memory_service.consider_update_stm(
                session.persistent_session_id, recent_messages
            )
            if updated:
                logger.info("STM updated after block streaming chat completion")


    def _text_to_async_generator(self, text: str) -> AsyncGenerator[str, None]:
        """Convert text to async generator for block streaming"""
        async def _gen():
            yield text
        return _gen()
    
    def _json_to_async_generator(self, json_obj: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Convert JSON object to async generator for block streaming"""
        async def _gen():
            yield json_obj
        return _gen()
