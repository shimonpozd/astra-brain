from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, TimestampMixin


class TalmudicConcept(Base, TimestampMixin):
    """
    Lightweight model to store talmudic concepts for highlighting and content pages.
    """

    __tablename__ = "talmudic_concepts"

    slug: Mapped[str] = mapped_column(String(255), primary_key=True)
    term_he: Mapped[str] = mapped_column(String(512), nullable=False)
    search_patterns: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    short_summary_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_article_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    generated_at: Mapped[datetime | None] = mapped_column(nullable=True)
