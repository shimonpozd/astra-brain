from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import httpx

from core.settings import Settings

router = APIRouter()
settings = Settings()


class LocationPayload(BaseModel):
    name: Optional[str] = None
    lat: float
    lon: float
    elevation_m: Optional[float] = Field(default=None, alias="elevation_m")


class ZmanimRequest(BaseModel):
    date: str
    timezone: str
    location: LocationPayload
    methods: List[str] = []
    use_elevation: Optional[bool] = None
    ateret_torah_sunset_offset: Optional[float] = None


@router.get("/zmanim/methods")
async def list_methods() -> Dict[str, Any]:
    url = f"{settings.ZMANIM_SERVICE_URL}/methods"
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Zmanim service error: {exc}") from exc


@router.post("/zmanim/calculate")
async def calculate_zmanim(payload: ZmanimRequest) -> Dict[str, Any]:
    url = f"{settings.ZMANIM_SERVICE_URL}/calculate"
    body = {
        "date": payload.date,
        "timezone": payload.timezone,
        "location": {
            "name": payload.location.name,
            "lat": payload.location.lat,
            "lon": payload.location.lon,
            "elevationM": payload.location.elevation_m,
        },
        "methods": payload.methods,
        "useElevation": payload.use_elevation,
        "ateretTorahSunsetOffset": payload.ateret_torah_sunset_offset,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Zmanim service error: {exc}") from exc
