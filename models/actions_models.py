from pydantic import BaseModel
from typing import Optional

class TranslateRequest(BaseModel):
    tref: str  # Text reference (e.g., "Genesis 1:1" or "Rashi on Genesis 1:1:1")

class ExplainTermRequest(BaseModel):
    term: str
    context_text: str
