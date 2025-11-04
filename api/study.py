import logging
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from models.study_models import (
    StudyBookshelfRequest, StudyResolveRequest, StudySetFocusRequest, 
    StudyStateResponse, StudyNavigateRequest, StudyWorkbenchSetRequest, 
    StudyChatSetFocusRequest, StudyChatRequest
)
from core.dependencies import get_study_service, get_lexicon_service
from brain_service.services.study_service import StudyService
from brain_service.services.lexicon_service import LexiconService

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Study State & Navigation Endpoints ---

@router.get("/state", response_model=StudyStateResponse)
async def study_get_state_handler(
    session_id: str, 
    study_service: StudyService = Depends(get_study_service)
):
    """Get the entire study state for a session."""
    return await study_service.get_state(session_id)

@router.post("/set_focus", response_model=StudyStateResponse)
async def study_set_focus_handler(
    request: StudySetFocusRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Set focus on a specific reference and update study state."""
    try:
        return await study_service.set_focus(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in set_focus: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/back", response_model=StudyStateResponse)
async def study_back_handler(
    request: StudyNavigateRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Move the history cursor back one step and return the state."""
    return await study_service.navigate_back(request)

@router.post("/forward", response_model=StudyStateResponse)
async def study_forward_handler(
    request: StudyNavigateRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Move the history cursor forward one step and return the state."""
    return await study_service.navigate_forward(request)

@router.post("/workbench/set", response_model=StudyStateResponse)
async def study_workbench_set_handler(
    request: StudyWorkbenchSetRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Set workbench items for a study session."""
    return await study_service.set_workbench(request)

# --- Study Data Endpoints ---

@router.get("/categories")
async def study_categories_handler():
    """Get bookshelf categories."""
    from brain_service.services.sefaria_index import get_bookshelf_categories
    return get_bookshelf_categories()

@router.post("/bookshelf")
async def study_bookshelf_handler(
    request: StudyBookshelfRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Get bookshelf data for a reference."""
    logger.info(f"Bookshelf request: ref='{request.ref}', session_id='{request.session_id}', categories={request.categories}")
    
    if not request.ref or not request.ref.strip():
        raise HTTPException(status_code=400, detail="Reference (ref) is required and cannot be empty")
    
    return await study_service.get_bookshelf(request)

@router.post("/resolve")
async def study_resolve_handler(
    request: StudyResolveRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Resolve a book name to a reference."""
    return await study_service.resolve_reference(request)

# --- Study Chat Endpoints ---

@router.post("/chat")
async def study_chat_handler(
    request: StudyChatRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Process a study chat request with streaming response."""
    from fastapi.responses import StreamingResponse
    
    async def generate():
        async for chunk in study_service.process_study_chat_stream(request):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@router.post("/chat/set_focus", response_model=StudyStateResponse)
async def study_chat_set_focus_handler(
    request: StudyChatSetFocusRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Set focus for study chat."""
    return await study_service.set_chat_focus(request)

@router.post("/chat/stream")
async def study_chat_stream_handler(
    request: StudyChatRequest, 
    study_service: StudyService = Depends(get_study_service)
):
    """Stream study chat response with context-aware agent selection."""
    async def generate():
        async for chunk in study_service.process_study_chat_stream(request):
            yield chunk
    
    return StreamingResponse(
        generate(), 
        media_type="application/x-ndjson"
    )

# --- Lexicon Endpoint ---

@router.get("/lexicon")
async def study_lexicon_handler(
    word: str,
    lexicon_service: LexiconService = Depends(get_lexicon_service)
):
    """
    Get lexicon definition for a Hebrew/Aramaic word.
    
    Args:
        word: The word to look up
        
    Returns:
        Lexicon data from Sefaria API
    """
    if not word or not word.strip():
        raise HTTPException(status_code=400, detail="Word parameter is required")
    
    result = await lexicon_service.get_word_definition(word.strip())
    
    if result.get("ok"):
        return result["data"]
    else:
        # Handle different error types
        status_code = result.get("status_code", 500)
        if status_code == 404:
            raise HTTPException(status_code=404, detail=result.get("error", "Word not found"))
        else:
            raise HTTPException(status_code=status_code, detail=result.get("error", "Internal server error"))