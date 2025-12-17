from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, Float, UniqueConstraint, ForeignKey, Integer, Index, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, TimestampMixin


class YiddishSichaProgress(Base, TimestampMixin):
    __tablename__ = "yiddish_sicha_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "sicha_id", name="uq_yiddish_progress_user_sicha"),
        Index("ix_yiddish_progress_user", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    sicha_id: Mapped[str] = mapped_column(String(128), nullable=False)
    read_pids: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    last_opened_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    progress_read_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    progress_vocab_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class YiddishVocabEntry(Base, TimestampMixin):
    __tablename__ = "yiddish_vocab"
    __table_args__ = (
        UniqueConstraint("user_id", "lemma", "sense_id", name="uq_yiddish_vocab_user_lemma_sense"),
        Index("ix_yiddish_vocab_user", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    sense_id: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="learning")
    srs_stage: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    srs_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class YiddishAttestation(Base, TimestampMixin):
    __tablename__ = "yiddish_attestations"
    __table_args__ = (
        Index("ix_yiddish_attest_user", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    sense_id: Mapped[str] = mapped_column(String(256), nullable=False)
    pid: Mapped[str] = mapped_column(String(64), nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=True)
    "end"  # noqa: E701, just to retain attribute name in type hints
    end: Mapped[int] = mapped_column(Integer, nullable=True)
    surface: Mapped[str] = mapped_column(Text, nullable=False)
    context_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    sicha_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class YiddishQueueItem(Base, TimestampMixin):
    __tablename__ = "yiddish_queue_items"
    __table_args__ = (
        UniqueConstraint("user_id", "lemma", "sense_id", name="uq_yiddish_queue_user_lemma_sense"),
        Index("ix_yiddish_queue_user", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    sense_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_pid: Mapped[str | None] = mapped_column(String(64), nullable=True)


class YiddishWordCard(Base, TimestampMixin):
    """
    Cached WordCard for Yiddish lemmas sourced from Wiktionary.
    Stores the canonical JSON plus raw evidence used for LLM translation.
    """

    __tablename__ = "yiddish_wordcards"
    __table_args__ = (
        UniqueConstraint("lemma", "lang", "source", "version", name="uq_yid_wordcard_lemma_lang_src_ver"),
        Index("ix_yid_wordcard_lemma", "lemma"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    lang: Mapped[str] = mapped_column(String(16), nullable=False, default="yi")
    ui_lang: Mapped[str] = mapped_column(String(16), nullable=False, default="ru")
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="wiktionary")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    word_surface: Mapped[str | None] = mapped_column(String(256), nullable=True)
    pos_default: Mapped[str | None] = mapped_column(String(32), nullable=True)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
