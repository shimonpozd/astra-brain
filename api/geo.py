from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

from core.settings import Settings

router = APIRouter()
settings = Settings()


class ElevationRequest(BaseModel):
    lat: float
    lon: float


@router.post("/geo/elevation")
async def get_elevation(payload: ElevationRequest):
    url = settings.OPENTOPO_URL_TEMPLATE.format(lat=payload.lat, lon=payload.lon)
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Elevation service error: {exc}") from exc

    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        raise HTTPException(status_code=502, detail="Elevation service returned no results")
    elevation = results[0].get("elevation")
    return {"elevation_m": elevation, "source": "opentopodata"}
