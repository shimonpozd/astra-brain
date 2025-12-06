# Enhanced logging for _stream_llm_response method
# This shows the improved logging at the start of the method

    async def _stream_llm_response(
        self, 
        messages: List[Dict[str, Any]], 
        session_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with tool support."""
        
        # Enhanced session and context logging
        logger.info("Starting LLM response stream", extra={
            "session_id": session_id,
            "message_count": len(messages),
            "system_message_present": any(msg.get("role") == "system" for msg in messages),
            "user_message_present": any(msg.get("role") == "user" for msg in messages),
            "tool_message_count": sum(1 for msg in messages if msg.get("role") == "tool")
        })
        
        # Log message types for debugging
        message_roles = [msg.get("role", "unknown") for msg in messages]
        logger.debug("Message roles in conversation", extra={
            "session_id": session_id,
            "message_roles": message_roles,
            "last_user_message_preview": next(
                (msg.get("content", "")[:100] for msg in reversed(messages) if msg.get("role") == "user"), 
                "No user message"
            )
        })
        
        try:
            client, model, reasoning_params, caps = get_llm_for_task("STUDY")
            logger.info("LLM configuration loaded", extra={
                "session_id": session_id,
                "model": model,
                "capabilities": caps,
                "reasoning_params_keys": list(reasoning_params.keys())
            })
        except LLMConfigError as e:
            logger.error("LLM configuration failed", extra={
                "session_id": session_id,
                "error": str(e)
            })
            yield json.dumps({"type": "error", "data": {"message": f"LLM not configured: {e}"}}) + '\n'
            return

        tools = self.tool_registry.get_tool_schemas()
        logger.info("Tool registry status", extra={
            "session_id": session_id,
            "available_tools_count": len(tools),
            "available_tools": [tool.get("function", {}).get("name", "unknown") for tool in tools]
        })
        
        api_params = {**reasoning_params, "model": model, "messages": messages, "stream": True}
        if tools:
            api_params.update({"tools": tools, "tool_choice": "auto"})
            logger.info("Tools enabled for LLM call", extra={
                "session_id": session_id,
                "tool_choice": "auto"
            })
        else:
            logger.warning("No tools available for LLM call", extra={
                "session_id": session_id
            })

        iter_count = 0
        while iter_count < 5:  # Max tool-use iterations
            iter_count += 1
            logger.info("Starting LLM iteration", extra={
                "session_id": session_id,
                "iteration": iter_count,
                "max_iterations": 5,
                "model": model
            })
            
            # ... rest of the method continues ...



































