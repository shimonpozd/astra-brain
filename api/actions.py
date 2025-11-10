import logging
import json
import re
from collections import defaultdict
import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from core.dependencies import get_http_client, get_lexicon_service, get_translation_service
from brain_service.models.actions_models import TranslateRequest, ExplainTermRequest

# Assuming these functions will be moved to a service layer later
from config.prompts import get_prompt
from core.llm_config import get_llm_for_task, LLMConfigError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/translate")
async def translate_handler(
    request: TranslateRequest, 
    translation_service = Depends(get_translation_service)
):
    """Translate a text reference using TranslationService."""
    
    async def serialize_to_ndjson():
        async for chunk in translation_service.translate_text_reference(request.tref):
            yield json.dumps(chunk, ensure_ascii=False) + '\n'
    
    return StreamingResponse(
        serialize_to_ndjson(),
        media_type="application/x-ndjson"
    )



@router.post("/explain-term")
async def explain_term_handler(
    request: ExplainTermRequest,
    lexicon_service = Depends(get_lexicon_service)
):
    logger.info("Received term explanation request", extra={"term": request.term})

    lexicon_result = await lexicon_service.get_word_definition(request.term)
    sefaria_data = lexicon_result["data"] if lexicon_result.get("ok") else {"error": lexicon_result.get("error", "Unknown error")}

    try:
        llm_client, model, reasoning_params, _ = get_llm_for_task("LEXICON")
    except LLMConfigError as e:
        raise HTTPException(status_code=500, detail=f"LLM not configured: {e}")

    system_prompt = get_prompt("actions.lexicon_system")
    user_prompt_template = get_prompt("actions.lexicon_user_template")

    if not system_prompt or not user_prompt_template:
        raise HTTPException(status_code=500, detail="Lexicon prompts not configured.")

    context_text = (request.context_text or "").strip()
    if not context_text:
        logger.warning("LEXICON: Received request for term '%s' without context_text", request.term)
        context_text = "Context not provided."

    user_prompt = user_prompt_template.replace("{term}", request.term)
    user_prompt = user_prompt.replace("{context_text}", context_text)
    user_prompt = user_prompt.replace("{sefaria_data}", json.dumps(sefaria_data, indent=2, ensure_ascii=False))
    user_prompt += "\n\nYou may invoke the available lexicon tools to look up alternate spellings or root forms when needed."

    tool_handlers = {
        "sefaria_get_lexicon": lexicon_service.get_word_definition_for_tool,
        "sefaria_guess_root": lexicon_service.get_root_suggestions,
    }

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "sefaria_get_lexicon",
                "description": "Fetch the lexicon entry for a specific Hebrew/Aramaic word.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "The exact spelling to look up in the Sefaria lexicon."
                        }
                    },
                    "required": ["word"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "sefaria_guess_root",
                "description": "Suggest related root forms or alternative spellings and return any known lexicon data for them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "The word whose root or related spellings should be explored."
                        },
                        "max_candidates": {
                            "type": "integer",
                            "description": "Optional number of suggestions to return (default 5).",
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["word"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    async def stream_llm_response():
        enable_tools = True
        max_rounds = 5

        while max_rounds > 0:
            max_rounds -= 1
            tokens: list[str] = []
            tool_call_builders = defaultdict(lambda: {
                "id": "",
                "index": 0,
                "function": {"name": "", "arguments": ""}
            })

            stream_kwargs = {
                **reasoning_params,
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if enable_tools:
                stream_kwargs["tools"] = tool_schemas
                stream_kwargs["tool_choice"] = "auto"

            try:
                stream = await llm_client.chat.completions.create(**stream_kwargs)
            except Exception as exc:
                if enable_tools:
                    enable_tools = False
                    logger.warning("LEXICON tools disabled due to error; retrying without tools", extra={"error": str(exc)})
                    continue
                logger.error("LEXICON_STREAM: LLM error", extra={"error": str(exc)}, exc_info=True)
                yield "Error: language model failed to generate an explanation."
                return

            async for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta and delta.content:
                    tokens.append(delta.content)

                if delta and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        builder = tool_call_builders[tool_call.index]
                        builder["index"] = tool_call.index
                        if tool_call.id:
                            builder["id"] = tool_call.id
                        if tool_call.function:
                            if tool_call.function.name:
                                builder["function"]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                builder["function"]["arguments"] += tool_call.function.arguments

            if tool_call_builders:
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": []
                }
                for builder in tool_call_builders.values():
                    if not builder["id"]:
                        builder["id"] = f"lexicon-tool-{builder['index']}"
                    assistant_tool_msg["tool_calls"].append({
                        "id": builder["id"],
                        "type": "function",
                        "function": builder["function"]
                    })
                messages.append(assistant_tool_msg)

                for builder in tool_call_builders.values():
                    tool_name = builder["function"].get("name")
                    handler = tool_handlers.get(tool_name)
                    if handler is None:
                        tool_output = {"ok": False, "error": f"Unknown tool '{tool_name}'"}
                    else:
                        raw_args = builder["function"].get("arguments") or "{}"
                        try:
                            parsed_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            parsed_args = {}
                        try:
                            tool_output = await handler(**parsed_args)
                        except Exception as tool_exc:
                            logger.error("LEXICON tool handler failed", extra={
                                "tool": tool_name,
                                "error": str(tool_exc)
                            }, exc_info=True)
                            tool_output = {"ok": False, "error": str(tool_exc)}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": builder["id"],
                        "content": json.dumps(tool_output, ensure_ascii=False)
                    })
                continue

            if tokens:
                for token in tokens:
                    yield token
                return

            yield "I was unable to generate an explanation for this term."
            return

        yield "Stopped after multiple tool attempts without a final answer."

    return StreamingResponse(stream_llm_response(), media_type="text/event-stream")


# ===== Speechify (Text -> concise spoken text for TTS) =====

class SpeechifyRequest(BaseModel):
    text: str | None = None
    hebrew_text: str | None = None
    english_text: str | None = None


@router.post("/speechify")
async def speechify_handler(request: SpeechifyRequest):
    """
    Rewrites provided text (Hebrew + English aware) into a short, natural spoken
    English paragraph suitable for TTS playback.

    Returns JSON: { "speech_text": string }
    """
    # Build user content from provided fields using configured preference
    try:
        llm_client, model, reasoning_params, _ = get_llm_for_task("SPEECHIFY")
    except LLMConfigError as e:
        raise HTTPException(status_code=500, detail=f"LLM not configured: {e}")

    system_prompt = get_prompt("actions.speechify_system")
    if not system_prompt:
        raise HTTPException(status_code=500, detail="Speechify prompt not configured.")

    hebrew_text = (request.hebrew_text or "").strip()
    english_text = (request.english_text or request.text or "").strip()

    # Prepare a simple user payload with placeholders identical to overrides contract
    user_prompt = (
        f"Hebrew (may be empty):\n{hebrew_text}\n\n"
        f"English (may be empty):\n{english_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Non-streaming one-shot completion; expect compact JSON or plain text
        completion = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **reasoning_params,
        )
        content = completion.choices[0].message.content or ""

        # Try parse JSON {"speech_text": "..."}; otherwise fallback to raw text
        speech_text = None
        try:
            # Clean potential code fences
            cleaned = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and isinstance(parsed.get("speech_text"), str):
                speech_text = parsed["speech_text"].strip()
        except Exception:
            pass

        if not speech_text:
            # Heuristics: remove <think> blocks if present
            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            speech_text = cleaned

        if not speech_text:
            raise HTTPException(status_code=502, detail="Empty response from LLM")

        return JSONResponse({"speech_text": speech_text})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SPEECHIFY: error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating speech text")
