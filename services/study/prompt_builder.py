"""Utilities for constructing study prompts with budget enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .prompt_budget import PromptBudget, summarize_trim
from .logging import log_prompt_trimmed

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptParts:
    system: str
    stm: str
    study: Iterable[str]
    extras: Iterable[str]


def assemble_prompt(
    parts: PromptParts,
    budget: PromptBudget,
    *,
    token_counter: "TokenCounter",
) -> Dict[str, Any]:
    """Assemble a prompt structure while enforcing the provided budget."""

    messages: List[Dict[str, str]] = []

    system_tokens = token_counter.count(parts.system)
    stm_tokens = token_counter.count(parts.stm)

    if system_tokens > budget.reserved_for_system:
        trimmed_system = token_counter.trim(parts.system, budget.reserved_for_system)
        removed = system_tokens - budget.reserved_for_system
        log_prompt_trimmed("system", removed_tokens=removed, remaining_tokens=budget.reserved_for_system)
        system_content = trimmed_system
    else:
        system_content = parts.system

    if parts.system:
        messages.append({"role": "system", "content": system_content})

    stm_available = budget.reserved_for_stm
    stm_content = parts.stm
    if stm_tokens > stm_available and stm_available > 0:
        trimmed_stm = token_counter.trim(parts.stm, stm_available)
        removed = stm_tokens - stm_available
        log_prompt_trimmed("stm", removed_tokens=removed, remaining_tokens=stm_available)
        stm_content = trimmed_stm

    if stm_content:
        messages.append({"role": "system", "content": stm_content})

    tokens_remaining = budget.available_for_study

    study_segments = []
    for segment in parts.study:
        cost = token_counter.count(segment)
        if tokens_remaining - cost < budget.min_study_tokens and tokens_remaining < budget.min_study_tokens:
            break
        if cost <= tokens_remaining:
            study_segments.append(segment)
            tokens_remaining -= cost
        else:
            trimmed = token_counter.trim(segment, max(tokens_remaining, 0))
            removed = cost - len(trimmed)
            log_prompt_trimmed("study", removed_tokens=removed, remaining_tokens=tokens_remaining)
            if trimmed:
                study_segments.append(trimmed)
            tokens_remaining = 0
            break

    if study_segments:
        messages.append({"role": "assistant", "content": "\n".join(study_segments)})

    extra_segments = []
    for extra in parts.extras:
        if tokens_remaining <= 0:
            break
        cost = token_counter.count(extra)
        if cost <= tokens_remaining:
            extra_segments.append(extra)
            tokens_remaining -= cost
        else:
            trimmed = token_counter.trim(extra, max(tokens_remaining, 0))
            removed = cost - len(trimmed)
            log_prompt_trimmed("extras", removed_tokens=removed, remaining_tokens=tokens_remaining)
            if trimmed:
                extra_segments.append(trimmed)
            tokens_remaining = 0
            break

    if extra_segments:
        messages.append({"role": "system", "content": "\n".join(extra_segments)})

    return {"messages": messages, "budget_summary": summarize_trim(0, tokens_remaining)}


class TokenCounter:
    """Simple character-based token counter placeholder."""

    def count(self, text: str) -> int:
        return len(text or "")

    def trim(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        return (text or "")[:limit]


try:  # pragma: no cover - optional dependency
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    tiktoken = None


class TiktokenCounter(TokenCounter):
    def __init__(self, model_hint: str = "gpt-4o") -> None:
        self._encoding = None
        if tiktoken:
            try:
                self._encoding = tiktoken.encoding_for_model(model_hint)
            except KeyError:
                try:
                    self._encoding = tiktoken.get_encoding("cl100k_base")
                except Exception:  # pragma: no cover
                    self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding:
            return len(self._encoding.encode(text or ""))
        return super().count(text)

    def trim(self, text: str, limit: int) -> str:
        if self._encoding and limit >= 0:
            tokens = self._encoding.encode(text or "")
            return self._encoding.decode(tokens[:limit])
        return super().trim(text, limit)


class RatioTokenCounter(TokenCounter):
    def __init__(self, characters_per_token: float = 4.0) -> None:
        self._ratio = max(characters_per_token, 1.0)

    def count(self, text: str) -> int:
        length = len(text or "")
        return max(int(round(length / self._ratio)), 0)

    def trim(self, text: str, limit: int) -> str:
        char_limit = int(round(limit * self._ratio))
        return super().trim(text, char_limit)


class TokenCounterFactory:
    """Factory for obtaining token counters per provider/model."""

    def __init__(self, default: Optional[TokenCounter] = None) -> None:
        self._default = default or TokenCounter()
        self._registry: List[Tuple[str, TokenCounter]] = []

    def register(self, prefix: str, counter: TokenCounter) -> None:
        self._registry.append((prefix.lower(), counter))

    def get(self, key: Optional[str]) -> TokenCounter:
        if key:
            lowered = key.lower()
            for prefix, counter in self._registry:
                if lowered.startswith(prefix):
                    return counter
        return self._default
