"""Typed configuration for the study domain."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


class WindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_default: PositiveInt = Field(default=5)
    size_min: int = Field(default=1, ge=0)
    size_max: PositiveInt = Field(default=15)

    @model_validator(mode="after")
    def validate_bounds(self) -> "WindowConfig":
        if not self.size_min <= self.size_default <= self.size_max:
            raise ValueError("Window size bounds are inconsistent")
        return self


class PreviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_len: PositiveInt = Field(default=600)


class DailyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_small: PositiveInt = Field(default=10)
    initial_medium: PositiveInt = Field(default=20)
    initial_large: PositiveInt = Field(default=30)
    large_threshold: PositiveInt = Field(default=50)
    background_delay_ms: PositiveInt = Field(default=100)
    lock_ttl_sec: PositiveInt = Field(default=900)
    redis_ttl_days: PositiveInt = Field(default=7)
    max_total_segments: PositiveInt = Field(default=500)
    retry_backoff_ms: list[int] = Field(default_factory=lambda: [100, 500, 1000])
    max_retries: int = Field(default=3, ge=0)
    batch_size: PositiveInt = Field(default=20)
    modular_loader_enabled: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "DailyConfig":
        if self.initial_small > self.initial_medium:
            raise ValueError("initial_small must be <= initial_medium")
        if self.initial_medium > self.initial_large:
            raise ValueError("initial_medium must be <= initial_large")
        if self.initial_large > self.max_total_segments:
            raise ValueError("initial_large must be <= max_total_segments")
        return self

    @field_validator("retry_backoff_ms")
    @classmethod
    def validate_backoff(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("retry_backoff_ms cannot be empty")
        if any(item <= 0 for item in value):
            raise ValueError("retry_backoff_ms values must be positive")
        if any(value[index] >= value[index + 1] for index in range(len(value) - 1)):
            raise ValueError("retry_backoff_ms must be strictly increasing")
        return value


class BookshelfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_preview_fetch: PositiveInt = Field(default=20)
    limit_default: PositiveInt = Field(default=40)
    default_categories: list[str] = Field(default_factory=list)
    cache_ttl_sec: int = Field(default=0, ge=0)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_runtime: str = Field(default="INFO")
    sample_debug_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class PromptBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_total_tokens: PositiveInt = Field(default=6000)
    reserved_for_system: PositiveInt = Field(default=1000)
    reserved_for_stm: PositiveInt = Field(default=1500)
    min_study_tokens: PositiveInt = Field(default=2000)

    @model_validator(mode="after")
    def validate_budget(self) -> "PromptBudgetConfig":
        allocated = self.reserved_for_system + self.reserved_for_stm + self.min_study_tokens
        if allocated > self.max_total_tokens:
            raise ValueError("Prompt budget allocations exceed max_total_tokens")
        return self


class ChatHistoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_messages: PositiveInt = Field(default=2000)
    ttl_days: PositiveInt = Field(default=30)


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facade_enabled: bool = Field(default=False)


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: WindowConfig = Field(default_factory=WindowConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    daily: DailyConfig = Field(default_factory=DailyConfig)
    bookshelf: BookshelfConfig = Field(default_factory=BookshelfConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prompt_budget: PromptBudgetConfig = Field(default_factory=PromptBudgetConfig)
    chat_history: ChatHistoryConfig = Field(default_factory=ChatHistoryConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)


def load_study_config(raw: Mapping[str, Any] | None) -> StudyConfig:
    """Load ``StudyConfig`` from a raw mapping safely."""

    data = raw or {}
    return StudyConfig.model_validate(data)


__all__ = [
    "StudyConfig",
    "WindowConfig",
    "PreviewConfig",
    "DailyConfig",
    "BookshelfConfig",
    "LoggingConfig",
    "PromptBudgetConfig",
    "ChatHistoryConfig",
    "FeaturesConfig",
    "load_study_config",
]
