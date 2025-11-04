"""Prompt budgeting utilities for study prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config_schema import PromptBudgetConfig


@dataclass(slots=True)
class PromptBudget:
    """Represents the token allocations for prompt construction."""

    total_tokens: int
    reserved_for_system: int
    reserved_for_stm: int
    min_study_tokens: int

    @property
    def available_for_study(self) -> int:
        return max(self.total_tokens - self.reserved_for_system - self.reserved_for_stm, 0)


def build_budget(config: PromptBudgetConfig) -> PromptBudget:
    """Construct a ``PromptBudget`` from configuration."""

    return PromptBudget(
        total_tokens=config.max_total_tokens,
        reserved_for_system=config.reserved_for_system,
        reserved_for_stm=config.reserved_for_stm,
        min_study_tokens=config.min_study_tokens,
    )


def summarize_trim(removed_tokens: int, remaining_tokens: int) -> Dict[str, int]:
    """Return a structured summary for logging/metrics."""

    return {
        "removed_tokens": removed_tokens,
        "remaining_tokens": remaining_tokens,
    }
