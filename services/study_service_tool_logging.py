# Patch for study_service.py to add tool call logging
# This shows the changes needed around line 768-790

            # Log tool calls summary before execution
            logger.info(f"Executing {len(full_tool_calls)} tool calls", extra={
                "session_id": session_id,
                "iteration": iter_count,
                "tool_calls": [{"name": tc["function"]["name"], "id": tc["id"]} for tc in full_tool_calls]
            })
            
            for tool_call in full_tool_calls:
                function_name = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                
                try:
                    function_args = json.loads(tool_call["function"].get("arguments") or "{}")
                    
                    # Enhanced logging before tool call
                    logger.info(f"Executing tool call: {function_name}", extra={
                        "session_id": session_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": function_name,
                        "tool_args": function_args,
                        "iteration": iter_count
                    })
                    
                    # Call tool with session_id for logging context
                    result = await self.tool_registry.call(function_name, session_id=session_id, **function_args)
                    
                    # Log successful tool result
                    logger.info(f"Tool call successful: {function_name}", extra={
                        "session_id": session_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": function_name,
                        "result_ok": result.get("ok", True) if isinstance(result, dict) else True,
                        "result_has_data": "data" in result if isinstance(result, dict) else bool(result)
                    })
                    
                    yield json.dumps({"type": "tool_result", "data": result}) + '\n'
                    messages.append({
                        "tool_call_id": tool_call_id, 
                        "role": "tool", 
                        "name": function_name, 
                        "content": json.dumps(result)
                    })
                    
                except json.JSONDecodeError as e:
                    error_message = f"Invalid JSON arguments for tool {function_name}: {e}"
                    logger.error(error_message, extra={
                        "session_id": session_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": function_name,
                        "raw_arguments": tool_call["function"].get("arguments", ""),
                        "error_type": "json_decode_error"
                    })
                    yield json.dumps({"type": "error", "data": {"message": error_message}}) + '\n'
                    messages.append({
                        "tool_call_id": tool_call_id, 
                        "role": "tool", 
                        "name": function_name, 
                        "content": json.dumps({"error": error_message})
                    })
                    
                except Exception as e:
                    error_message = f"Error calling tool {function_name}: {e}"
                    logger.error(error_message, extra={
                        "session_id": session_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": function_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }, exc_info=True)
                    yield json.dumps({"type": "error", "data": {"message": error_message}}) + '\n'
                    messages.append({
                        "tool_call_id": tool_call_id, 
                        "role": "tool", 
                        "name": function_name, 
                        "content": json.dumps({"error": error_message})
                    })






































