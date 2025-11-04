import logging
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Response, status

# Imports from the new model location
from models.admin_models import PersonalityFull, PersonalityPublic, PromptUpdateRequest
from core.dependencies import require_admin_user

# Imports from the existing config modules (assuming they are in PYTHONPATH)
from config import get_config, update_config
from config import personalities as personality_service
from config.prompts import get_prompt, list_prompts, update_prompt

logger = logging.getLogger(__name__)
router = APIRouter()

# --- ADMIN ENDPOINTS ---
@router.get("/config")
async def get_config_handler(_: str = Depends(require_admin_user)):
    return get_config()

@router.get("/config/public")
async def get_public_config_handler():
    """Get public configuration (no auth required)."""
    config = get_config()
    # Return only public parts of config
    return {
        "personalities": config.get("personalities", {}),
        "llm": config.get("llm", {}),
        "voice": config.get("voice", {}),
        "memory": config.get("memory", {}),
        "services": config.get("services", {})
    }

@router.patch("/config")
async def update_config_handler(settings: Dict[str, Any], _: str = Depends(require_admin_user)):
    try:
        updated_config = update_config(settings)
        logger.info("AUDIT: Configuration updated", extra={"settings_changed": settings})
        return updated_config
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompts")
async def list_prompts_handler(_: str = Depends(require_admin_user)):
    return list_prompts()

@router.get("/prompts/{prompt_id:path}")
async def get_prompt_handler(prompt_id: str, _: str = Depends(require_admin_user)):
    prompt_text = get_prompt(prompt_id)
    if prompt_text is None:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found.")
    return {"id": prompt_id, "text": prompt_text}

@router.put("/prompts/{prompt_id:path}")
async def update_prompt_handler(prompt_id: str, request: PromptUpdateRequest, _: str = Depends(require_admin_user)):
    success = update_prompt(prompt_id, request.text)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to update prompt '{prompt_id}'.")
    logger.info(f"AUDIT: Prompt '{prompt_id}' updated.", extra={"prompt_id": prompt_id, "new_text": request.text})
    return {"status": "ok"}

@router.get("/personalities", response_model=List[PersonalityPublic])
async def list_personalities_handler(_: str = Depends(require_admin_user)):
    return personality_service.list_personalities()

@router.get("/personalities/public", response_model=List[PersonalityPublic])
async def list_personalities_public_handler():
    """Get personalities list (no auth required)."""
    return personality_service.list_personalities()

@router.get("/personalities/{personality_id}", response_model=PersonalityFull)
async def get_personality_handler(personality_id: str, _: str = Depends(require_admin_user)):
    personality = personality_service.get_personality(personality_id)
    if not personality:
        raise HTTPException(status_code=404, detail=f"Personality '{personality_id}' not found.")
    return personality

@router.post("/personalities", response_model=PersonalityFull, status_code=status.HTTP_201_CREATED)
async def create_personality_handler(personality_data: PersonalityFull, _: str = Depends(require_admin_user)):
    created = personality_service.create_personality(personality_data.model_dump())
    if not created:
        raise HTTPException(status_code=409, detail=f"Personality with ID '{personality_data.id}' already exists or is invalid.")
    return created

@router.put("/personalities/{personality_id}", response_model=PersonalityFull)
async def update_personality_handler(personality_id: str, personality_data: PersonalityFull, _: str = Depends(require_admin_user)):
    updated = personality_service.update_personality(personality_id, personality_data.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail=f"Personality with ID '{personality_id}' not found.")
    return updated

@router.delete("/personalities/{personality_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_personality_handler(personality_id: str, _: str = Depends(require_admin_user)):
    success = personality_service.delete_personality(personality_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Personality with ID '{personality_id}' not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
