"""
Audio API endpoints for TTS message handling
"""
import os
import uuid
import shutil
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

# from brain_service.services.chat_service import ChatService
# from core.dependencies import get_chat_service

router = APIRouter(prefix="/audio", tags=["audio"])

# Audio storage directory
AUDIO_STORAGE_DIR = Path("audio_storage")
AUDIO_STORAGE_DIR.mkdir(exist_ok=True)

class BaseResponse(BaseModel):
    success: bool
    message: str

class AudioMessageRequest(BaseModel):
    text: str
    chat_id: str
    voice_id: Optional[str] = None
    language: Optional[str] = "en"
    speed: Optional[float] = 1.0
    provider: Optional[str] = "yandex"

class AudioMessageResponse(BaseModel):
    id: str
    chat_id: str
    audio_url: str
    duration: Optional[float] = None
    size: Optional[int] = None
    provider: str
    voice_id: Optional[str] = None

@router.post("/synthesize", response_model=AudioMessageResponse)
async def synthesize_audio_message(
    request: AudioMessageRequest
):
    """
    Synthesize audio from text and save as message in chat
    """
    try:
        # Generate unique audio file ID
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.wav"
        audio_path = AUDIO_STORAGE_DIR / audio_filename
        
        # Create chat-specific directory
        chat_dir = AUDIO_STORAGE_DIR / request.chat_id
        chat_dir.mkdir(exist_ok=True)
        
        # Move audio file to chat directory
        final_audio_path = chat_dir / audio_filename
        
        # TODO: Call TTS service to generate audio
        # For now, create a placeholder file
        with open(final_audio_path, 'wb') as f:
            f.write(b'placeholder audio data')
        
        # Calculate file size
        file_size = final_audio_path.stat().st_size
        
        # Create audio message in chat
        audio_message = {
            "id": audio_id,
            "role": "assistant",
            "content_type": "audio.v1",
            "content": {
                "text": request.text,
                "audioUrl": f"/api/audio/{request.chat_id}/{audio_filename}",
                "provider": request.provider,
                "voiceId": request.voice_id,
                "format": "wav",
                "size": file_size,
            },
            "timestamp": int(time.time() * 1000),
        }
        
        # TODO: Save message to chat
        # For now, just log the message
        print(f"Audio message created: {audio_message}")
        
        return AudioMessageResponse(
            id=audio_id,
            chat_id=request.chat_id,
            audio_url=f"/api/audio/{request.chat_id}/{audio_filename}",
            size=file_size,
            provider=request.provider,
            voice_id=request.voice_id,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to synthesize audio: {str(e)}")

@router.get("/{chat_id}/{filename}")
async def get_audio_file(chat_id: str, filename: str):
    """
    Serve audio files
    """
    try:
        audio_path = AUDIO_STORAGE_DIR / chat_id / filename
        
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Determine media type based on file extension
        media_type = "audio/wav"
        if filename.endswith('.mp3'):
            media_type = "audio/mpeg"
        elif filename.endswith('.ogg'):
            media_type = "audio/ogg"
        
        return FileResponse(
            path=str(audio_path),
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve audio file: {str(e)}")

@router.delete("/{chat_id}/{filename}")
async def delete_audio_file(chat_id: str, filename: str):
    """
    Delete audio file
    """
    try:
        audio_path = AUDIO_STORAGE_DIR / chat_id / filename
        
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        audio_path.unlink()
        
        return BaseResponse(success=True, message="Audio file deleted")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete audio file: {str(e)}")

@router.get("/{chat_id}/list")
async def list_chat_audio_files(chat_id: str):
    """
    List all audio files for a chat
    """
    try:
        chat_dir = AUDIO_STORAGE_DIR / chat_id
        
        if not chat_dir.exists():
            return {"files": []}
        
        files = []
        for file_path in chat_dir.iterdir():
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "created": file_path.stat().st_ctime,
                })
        
        return {"files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list audio files: {str(e)}")

@router.post("/upload")
async def upload_audio_file(
    chat_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload audio file for a chat
    """
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'wav'
        audio_filename = f"{file_id}.{file_extension}"
        
        # Create chat directory
        chat_dir = AUDIO_STORAGE_DIR / chat_id
        chat_dir.mkdir(exist_ok=True)
        
        # Save file
        audio_path = chat_dir / audio_filename
        with open(audio_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        file_size = audio_path.stat().st_size
        
        return {
            "id": file_id,
            "filename": audio_filename,
            "size": file_size,
            "url": f"/api/audio/{chat_id}/{audio_filename}",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload audio file: {str(e)}")
