from typing import Optional, List

from pydantic import BaseModel
from brain_service.services.study_state import StudySnapshot

# Request Models
class StudyBookshelfRequest(BaseModel):
    ref: str
    session_id: Optional[str] = None
    categories: Optional[List[str]] = None

class StudyResolveRequest(BaseModel):
    text: str

class StudySetFocusRequest(BaseModel):
    session_id: str
    ref: str
    focus_ref: Optional[str] = None
    window_size: Optional[int] = 5
    navigation_type: str = "drill_down"
    is_daily: Optional[bool] = None  # Explicit flag for daily mode
    timezone: Optional[str] = None
    stream_id: Optional[str] = None
    units_total: Optional[int] = None
    unit_index_today: Optional[int] = None

class StudyNavigateRequest(BaseModel):
    session_id: str

class StudyWorkbenchSetRequest(BaseModel):
    session_id: str
    slot: str
    ref: Optional[str] = None
    focus_ref: Optional[str] = None

class StudyChatSetFocusRequest(BaseModel):
    session_id: str
    ref: str
    update_main_focus: bool = False

class StudyChatRequest(BaseModel):
    session_id: str
    text: str
    agent_id: Optional[str] = None
    selected_panel_id: Optional[str] = None  # "focus", "left_workbench", "right_workbench", or None

# Response Models
class StudyStateResponse(BaseModel):
    ok: bool
    state: Optional[StudySnapshot] = None
