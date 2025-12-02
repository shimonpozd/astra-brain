from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for the brain service models."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    phone_number: Mapped[str | None] = mapped_column(String(32), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        Enum("admin", "member", name="user_role"),
        nullable=False,
        default="member",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_manually: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    threads: Mapped[list["ChatThread"]] = relationship(
        "ChatThread",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    api_keys: Mapped[list["UserApiKey"]] = relationship(
        "UserApiKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["UserSession"]] = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    login_events: Mapped[list["UserLoginEvent"]] = relationship(
        "UserLoginEvent",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class UserSession(Base, TimestampMixin):
    __tablename__ = "user_sessions"
    __table_args__ = (
        Index("ix_user_sessions_user_id", "user_id"),
        Index("ix_user_sessions_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    refresh_token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="sessions")


class UserLoginEvent(Base, TimestampMixin):
    __tablename__ = "user_login_events"
    __table_args__ = (
        Index("ix_user_login_events_user_id", "user_id"),
        Index("ix_user_login_events_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="login_events")


class ChatThread(Base, TimestampMixin):
    __tablename__ = "chat_threads"
    __table_args__ = (
        UniqueConstraint("user_id", "session_id", name="uq_chat_threads_user_session"),
        Index("ix_chat_threads_user_id", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    last_modified: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    metadata_json: Mapped[dict[str, object] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    user: Mapped["User"] = relationship("User", back_populates="threads")


class UserApiKey(Base, TimestampMixin):
    __tablename__ = "user_api_keys"
    __table_args__ = (
        Index("ix_user_api_keys_user_id", "user_id"),
        Index("ix_user_api_keys_provider", "provider"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="openrouter")
    api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    last_four: Mapped[str] = mapped_column(String(8), nullable=False)
    daily_limit: Mapped[int | None] = mapped_column(nullable=True)
    usage_today: Mapped[int] = mapped_column(default=0, nullable=False)
    last_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="api_keys")


class Profile(Base, TimestampMixin):
    """
    Cached author/work profile assembled from Sefaria + Wikipedia + LLM action.
    """

    __tablename__ = "profiles"

    slug: Mapped[str] = mapped_column(String(255), primary_key=True)
    title_en: Mapped[str | None] = mapped_column(String(512), nullable=True)
    title_he: Mapped[str | None] = mapped_column(String(512), nullable=True)

    json_raw: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    summary_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    facts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    authors: Mapped[list | dict | None] = mapped_column(JSONB, nullable=True)

    lifespan: Mapped[str | None] = mapped_column(String(128), nullable=True)
    period: Mapped[str | None] = mapped_column(String(128), nullable=True)
    comp_place: Mapped[str | None] = mapped_column(String(256), nullable=True)
    pub_place: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Manual curation / verification
    manual_summary_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    manual_facts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false", default=False)
    verified_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Work(Base, TimestampMixin):
    __tablename__ = "works"

    index_title: Mapped[str] = mapped_column(String(512), primary_key=True)
    title_en: Mapped[str | None] = mapped_column(String(512), nullable=True)
    title_he: Mapped[str | None] = mapped_column(String(512), nullable=True)
    en_desc: Mapped[str | None] = mapped_column(Text, nullable=True)
    comp_date: Mapped[dict | list | str | None] = mapped_column(JSONB, nullable=True)
    pub_date: Mapped[dict | list | str | None] = mapped_column(JSONB, nullable=True)
    comp_place: Mapped[str | None] = mapped_column(String(256), nullable=True)
    pub_place: Mapped[str | None] = mapped_column(String(256), nullable=True)
    categories: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    links: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    summary_html: Mapped[str | None] = mapped_column(Text, nullable=True)


class Author(Base, TimestampMixin):
    __tablename__ = "authors"

    slug: Mapped[str] = mapped_column(String(256), primary_key=True)
    name_en: Mapped[str | None] = mapped_column(String(512), nullable=True)
    name_he: Mapped[str | None] = mapped_column(String(512), nullable=True)
    summary_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    lifespan: Mapped[str | None] = mapped_column(String(128), nullable=True)
    period: Mapped[str | None] = mapped_column(String(128), nullable=True)
    links: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class WorkAuthor(Base, TimestampMixin):
    __tablename__ = "work_authors"
    __table_args__ = (
        Index("ix_work_authors_work", "work_id"),
        Index("ix_work_authors_author", "author_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    work_id: Mapped[str] = mapped_column(String(512), ForeignKey("works.index_title", ondelete="CASCADE"))
    author_id: Mapped[str] = mapped_column(String(256), ForeignKey("authors.slug", ondelete="CASCADE"))
