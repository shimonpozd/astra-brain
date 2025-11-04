from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: Any
    content_type: str = "text.v1"
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        # Ensure backward compatibility for callers expecting "ts"
        data.setdefault("ts", data.get("timestamp"))
        return data


class Session(BaseModel):
    user_id: str
    agent_id: str
    persistent_session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    short_term_memory: List[Message] = Field(default_factory=list)
    last_modified: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_research_plan: Optional[Dict[str, Any]] = None
    last_research_collection: Optional[str] = None
    seen_refs: set = Field(default_factory=set)

    def add_message(
        self,
        role: str,
        content: Any,
        *,
        content_type: str = "text.v1",
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Append a message to the short-term memory buffer.
        """
        message = Message(
            role=role,
            content=content,
            content_type=content_type,
            timestamp=timestamp or datetime.now().timestamp(),
        )
        self.short_term_memory.append(message)

    def to_dict(self) -> Dict[str, Any]:
        # seen_refs is a set (not JSON serialisable) – exclude it
        return self.model_dump(exclude={"seen_refs"})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(**data)
