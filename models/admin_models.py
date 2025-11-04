from pydantic import BaseModel
from typing import Optional

class PromptUpdateRequest(BaseModel):
    text: str

class PersonalityPublic(BaseModel):
    id: str
    name: str
    description: str
    flow: str

class PersonalityFull(PersonalityPublic):
    system_prompt: Optional[str] = None
    use_sefaria_tools: Optional[bool] = False
    use_research_memory: Optional[bool] = False
