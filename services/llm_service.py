import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
import httpx
from core.settings import Settings
from .memory_service import MemoryService

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for managing LLM interactions with STM integration.
    
    Handles streaming responses, tool calls, and automatic STM updates.
    """
    
    def __init__(self, http_client: httpx.AsyncClient, memory_service: MemoryService, settings: Settings):
        self.http_client = http_client
        self.memory_service = memory_service
        self.settings = settings
    
    async def stream_chat(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response with STM integration.
        
        Args:
            session_id: Session identifier for STM
            messages: List of messages for the conversation
            tools: Available tools for the LLM
            model: LLM model to use
            temperature: Sampling temperature
            
        Yields:
            Streaming events: llm_chunk, tool_call, tool_result, end
        """
        try:
            # Get STM and inject into system message
            stm_context = await self._get_stm_context(session_id)
            if stm_context:
                messages = self._inject_stm_context(messages, stm_context)
            
            # Stream the response
            async for event in self._stream_llm_response(messages, tools, model, temperature):
                yield event
            
            # Update STM after response completion
            await self._update_stm_if_needed(session_id, messages)
            
        except Exception as e:
            logger.error("LLM streaming error", extra={
                "session_id": session_id,
                "error": str(e)
            })
            yield json.dumps({"type": "error", "data": {"message": str(e)}}) + '\n'
        finally:
            yield json.dumps({"type": "end"}) + '\n'
    
    async def _get_stm_context(self, session_id: str) -> Optional[str]:
        """Get formatted STM context for prompt injection."""
        try:
            stm = await self.memory_service.get_stm(session_id)
            if stm:
                return self.memory_service.format_stm_for_prompt(stm)
        except Exception as e:
            logger.error("Failed to get STM context", extra={
                "session_id": session_id,
                "error": str(e)
            })
        return None
    
    def _inject_stm_context(self, messages: List[Dict[str, Any]], stm_context: str) -> List[Dict[str, Any]]:
        """Inject STM context into the first system message or create one."""
        if not stm_context:
            return messages
        
        # Look for existing system message
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                # Append STM to existing system message
                existing_content = msg.get("content", "")
                msg["content"] = f"{existing_content}\n\n{stm_context}"
                return messages
        
        # No system message found, create one
        system_message = {
            "role": "system",
            "content": f"[STM Context]\n{stm_context}"
        }
        
        return [system_message] + messages
    
    async def _stream_llm_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Dict[str, Any]],
        model: Optional[str],
        temperature: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from LLM provider."""
        # This is a placeholder implementation
        # In a real implementation, this would call the actual LLM provider
        
        # For now, simulate a streaming response
        response_text = "This is a simulated response from the enhanced STM-enabled LLM service."
        
        # Stream the response in chunks
        chunk_size = 10
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield json.dumps({"type": "llm_chunk", "data": chunk}) + '\n'
            # Simulate streaming delay
            await asyncio.sleep(0.05)
    
    async def _update_stm_if_needed(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Update STM if conditions are met."""
        try:
            # Count messages and estimate tokens
            message_count = len(messages)
            token_count = sum(len(str(msg.get("content", ""))) for msg in messages) // 4  # Rough estimate
            
            # Check if STM should be updated
            should_update = await self.memory_service.should_update_stm(
                session_id, message_count, token_count
            )
            
            if should_update:
                # Get recent messages for STM update
                recent_messages = messages[-10:]  # Last 10 messages
                await self.memory_service.update_stm(session_id, recent_messages)
                
                logger.info("STM updated after LLM response", extra={
                    "session_id": session_id,
                    "message_count": message_count,
                    "token_count": token_count
                })
                
        except Exception as e:
            logger.error("Failed to update STM", extra={
                "session_id": session_id,
                "error": str(e)
            })
    
    async def get_available_models(self) -> List[str]:
        """Get list of available LLM models."""
        # This would typically call the LLM provider's API
        return ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "claude-3-haiku"]
    
    async def summarize(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 512,
        timeout_s: int = 25,
        retries: int = 2,
        backoff_ms: int = 400,
        response_format_json: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize messages using LLM with JSON response format.
        
        Args:
            messages: Messages to summarize
            prompt: System prompt for summarization
            model: LLM model to use
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            timeout_s: Request timeout
            retries: Number of retries
            backoff_ms: Backoff between retries
            response_format_json: Force JSON response format
            
        Returns:
            Parsed JSON response from LLM
        """
        try:
            # Build messages for LLM
            llm_messages = [
                {"role": "system", "content": prompt},
                {"role": "assistant", "content": "You output JSON only."}
            ] + messages
            
            # Get LLM client and configuration
            from core.llm_config import get_llm_for_task
            try:
                client, default_model, reasoning_params, caps = get_llm_for_task("CHAT")
            except Exception as e:
                logger.error(f"Failed to get LLM client: {e}")
                raise
            
            # Use provided model or fallback to default
            model_to_use = model or default_model
            
            # Build API parameters
            api_params = {
                "model": model_to_use,
                "messages": llm_messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add JSON response format if supported
            if response_format_json and caps.get("json_mode"):
                api_params["response_format"] = {"type": "json_object"}
            
            # Make LLM request with retries
            for attempt in range(retries + 1):
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(**api_params),
                        timeout=timeout_s
                    )
                    
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response from LLM")
                    
                    # Parse JSON response
                    try:
                        result = json.loads(content)
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {e}")
                        # Try to extract JSON from response
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group())
                            return result
                        else:
                            raise ValueError(f"Invalid JSON response: {content[:200]}")
                    
                except asyncio.TimeoutError:
                    if attempt < retries:
                        logger.warning(f"LLM request timeout, retrying... (attempt {attempt + 1})")
                        await asyncio.sleep(backoff_ms / 1000)
                        continue
                    else:
                        raise
                except Exception as e:
                    if attempt < retries:
                        logger.warning(f"LLM request failed, retrying... (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(backoff_ms / 1000)
                        continue
                    else:
                        raise
            
            # This should never be reached
            raise RuntimeError("All retry attempts failed")
            
        except Exception as e:
            logger.error("LLM summarization failed", extra={
                "error": str(e),
                "model": model,
                "message_count": len(messages)
            })
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM service."""
        try:
            # Simple test request
            test_messages = [{"role": "user", "content": "Hello"}]
            
            start_time = time.time()
            # This would be a real LLM call in production
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "models_available": await self.get_available_models()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
