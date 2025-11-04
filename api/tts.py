import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from config import get_config

router = APIRouter()
logger = logging.getLogger(__name__)


def _tts_base_url() -> str:
    cfg = get_config() or {}
    services = cfg.get("services", {})
    return services.get("tts_service_url", "http://localhost:7040").rstrip("/")


@router.post("/synthesize")
async def synthesize(payload: dict):
    base_url = _tts_base_url()
    if not base_url:
        raise HTTPException(status_code=500, detail="TTS service URL not configured")
    url = f"{base_url}/tts/synthesize"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.error("TTS synthesize error %s: %s", resp.status_code, resp.text)
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            # Assume binary audio
            content_type = resp.headers.get("content-type", "audio/mpeg")
            return StreamingResponse(
                resp.aiter_bytes(), 
                media_type=content_type,
                headers={
                    "Content-Type": content_type,
                    "Content-Disposition": "inline"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Proxy synthesize failed: %s", e)
        raise HTTPException(status_code=500, detail="TTS proxy error")


@router.get("/voices")
async def list_voices():
    base_url = _tts_base_url()
    if not base_url:
        raise HTTPException(status_code=500, detail="TTS service URL not configured")
    url = f"{base_url}/tts/voices"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                logger.error("TTS voices error %s: %s", resp.status_code, resp.text)
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return JSONResponse(resp.json())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Proxy voices failed: %s", e)
        raise HTTPException(status_code=500, detail="TTS proxy error")


