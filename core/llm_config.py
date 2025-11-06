from __future__ import annotations

from openai import AsyncOpenAI

import os
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple
import logging

def _resolve_env_var(value: str) -> str:
    """
    Resolves a value that might be an environment variable placeholder.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return os.getenv(var_name, "")
    return value

logger = logging.getLogger(__name__)


class LLMConfigError(Exception):
    pass


USE_ASTRA_CONFIG = os.getenv("ASTRA_CONFIG_ENABLED", "false").lower() in {"1", "true", "yes"}
LLM_CONFIG: Dict[str, Any] = {}

def reload_llm_config():
    """
    Forces a reload of the LLM configuration from the central config.
    """
    global LLM_CONFIG, USE_ASTRA_CONFIG
    
    USE_ASTRA_CONFIG = os.getenv("ASTRA_CONFIG_ENABLED", "false").lower() in {"1", "true", "yes"}
    if not USE_ASTRA_CONFIG:
        LLM_CONFIG = {}
        logger.info("[CONFIG] LLM hot-reload skipped: ASTRA_CONFIG_ENABLED is false.")
        return

    try:
        from config import get_config_section

        raw_config = get_config_section("llm", {})
        if isinstance(raw_config, Mapping):
            LLM_CONFIG = dict(raw_config)
            logger.info("[CONFIG] Hot-reloaded LLM configuration.")
        else:
            LLM_CONFIG = {}
            logger.warning("[CONFIG] llm section missing or malformed during hot-reload.")
            
    except Exception as exc:
        LLM_CONFIG = {}
        logger.error(f"[CONFIG] Failed to hot-reload central config (llm): {exc}", exc_info=True)


# Initial load
reload_llm_config()


TASK_ENV_MAPPING: Dict[str, List[str]] = {
    "CHAT": ["OPENAI_MODEL", "ASTRA_MODEL_CHAT"],
    "DRAFTER": ["DRAFTER_MODEL", "ASTRA_MODEL_DRAFTER"],
    "CRITIC": ["CRITIC_MODEL", "ASTRA_MODEL_CRITIC"],
    "META_REASONER": ["META_REASONER_MODEL", "ASTRA_MODEL_META_REASONER"],
    "CURATOR": ["CURATOR_MODEL", "ASTRA_MODEL_CURATOR"],
    "SUMMARIZER": ["SUMMARIZER_MODEL", "ASTRA_MODEL_SUMMARIZER"],
    "TRANSLATOR": ["TRANSLATOR_MODEL", "ASTRA_MODEL_TRANSLATOR"],
    "PLANNER": ["PLANNER_MODEL", "ASTRA_MODEL_PLANNER"],
    "STUDY": ["STUDY_MODEL", "ASTRA_MODEL_STUDY"],
    "LEXICON": ["LEXICON_MODEL", "ASTRA_MODEL_LEXICON"],
    "SPEECHIFY": ["SPEECHIFY_MODEL", "ASTRA_MODEL_SPEECHIFY"],
}

TASK_OVERRIDE_MAPPING: Dict[str, str] = {
    "CHAT": "chat",
    "DRAFTER": "drafter",
    "CRITIC": "critic",
    "META_REASONER": "meta_reasoner",
    "CURATOR": "curator",
    "SUMMARIZER": "summarizer",
    "TRANSLATOR": "translator",
    "PLANNER": "planner",
    "STUDY": "study",
    "LEXICON": "lexicon",
    "SPEECHIFY": "speechify",
}


def _get_api_section(name: str) -> Dict[str, Any]:
    api_section = LLM_CONFIG.get("api") if isinstance(LLM_CONFIG.get("api"), Mapping) else {}
    section = api_section.get(name) if isinstance(api_section, Mapping) else {}
    return dict(section) if isinstance(section, Mapping) else {}


def _get_model_from_config(task: str) -> str | None:
    overrides = LLM_CONFIG.get("overrides") if isinstance(LLM_CONFIG.get("overrides"), Mapping) else {}
    override_key = TASK_OVERRIDE_MAPPING.get(task)
    if override_key and isinstance(overrides, Mapping):
        model = overrides.get(override_key)
        if isinstance(model, str) and model.strip():
            return model.strip()

    model = LLM_CONFIG.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()

    return None


def _get_model_from_env(task: str) -> str | None:
    for env_var in TASK_ENV_MAPPING.get(task, []):
        value = os.getenv(env_var)
        if value:
            return value
    return os.getenv("OPENAI_MODEL")


def _get_reasoning_params_from_config() -> Dict[str, Any]:
    params = LLM_CONFIG.get("parameters") if isinstance(LLM_CONFIG.get("parameters"), Mapping) else {}
    temperature = params.get("temperature") if isinstance(params, Mapping) else None
    top_p = params.get("top_p") if isinstance(params, Mapping) else None

    result: Dict[str, Any] = {}
    if isinstance(temperature, (int, float)):
        result["temperature"] = float(temperature)
    if isinstance(top_p, (int, float)):
        result["top_p"] = float(top_p)
    return result


def get_llm_for_task(
    task: str,
    *,
    provider_override: str | None = None,
    api_key_override: str | None = None,
) -> Tuple[AsyncOpenAI, str, Dict[str, Any], List[str]]:
    """Return OpenAI-compatible client, resolved model id, reasoning params, and capabilities."""
    if task not in TASK_ENV_MAPPING:
        raise LLMConfigError(f"Unknown task: {task}")

    model: str | None = None
    if USE_ASTRA_CONFIG:
        model = _get_model_from_config(task)

    if not model:
        model = _get_model_from_env(task)

    if not model:
        model = "ollama/qwen3:8b"

    model = model.strip()

    reasoning_params: Dict[str, Any] = {"temperature": 0.3, "top_p": 0.9}
    if USE_ASTRA_CONFIG:
        reasoning_params.update(_get_reasoning_params_from_config())

    provider = provider_override
    if not provider:
        if model.startswith("ollama/"):
            provider = "ollama"
        elif model.startswith("openrouter/"):
            provider = "openrouter"
        else:
            provider = "openai"

    if provider == "ollama":
        api_cfg = _get_api_section("ollama") if USE_ASTRA_CONFIG else {}
        ollama_base_url = _resolve_env_var(api_cfg.get("base_url")) or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        client = AsyncOpenAI(base_url=f"{ollama_base_url}/v1", api_key="ollama")
        clean_model = model.replace("ollama/", "")
        capabilities: List[str] = []

    elif provider == "openrouter":
        api_cfg = _get_api_section("openrouter") if USE_ASTRA_CONFIG else {}

        openrouter_base_url = _resolve_env_var(api_cfg.get("base_url")) or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        api_key = api_key_override or _resolve_env_var(api_cfg.get("api_key")) or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMConfigError("OPENROUTER_API_KEY not set for OpenRouter models")

        default_headers: Dict[str, Any] = {}
        referrer = _resolve_env_var(api_cfg.get("referrer")) or os.getenv("OPENROUTER_REFERRER")
        title = _resolve_env_var(api_cfg.get("title")) or os.getenv("OPENROUTER_TITLE")
        if referrer:
            default_headers["HTTP-Referer"] = referrer
        if title:
            default_headers["X-Title"] = title

        client_kwargs: Dict[str, Any] = {"base_url": openrouter_base_url, "api_key": api_key}
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        client = AsyncOpenAI(**client_kwargs)
        clean_model = model.replace("openrouter/", "") if model.startswith("openrouter/") else model
        capabilities = ["json_mode"]

    else:
        api_cfg = _get_api_section("openai") if USE_ASTRA_CONFIG else {}

        api_key = api_key_override or _resolve_env_var(api_cfg.get("api_key")) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigError("OPENAI_API_KEY not set for OpenAI models")

        organization = _resolve_env_var(api_cfg.get("organization")) or os.getenv("OPENAI_ORG_ID")
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if organization:
            client_kwargs["organization"] = organization
        client = AsyncOpenAI(**client_kwargs)
        clean_model = model
        capabilities = ["json_mode"]

    return client, clean_model, reasoning_params, capabilities


def get_reasoning_params() -> Dict[str, Any]:
    if USE_ASTRA_CONFIG:
        params = {"temperature": 0.3, "top_p": 0.9}
        params.update(_get_reasoning_params_from_config())
        return params
    return {"temperature": 0.3, "top_p": 0.9}


def get_tooling_config() -> Dict[str, Any]:
    """Return tooling configuration with safe defaults."""

    defaults = {
        "parallel_tool_calls": False,
        "retry_on_empty_stream": True,
    }

    if not USE_ASTRA_CONFIG:
        return defaults

    raw_tooling = LLM_CONFIG.get("tooling")
    if isinstance(raw_tooling, Mapping):
        merged = defaults.copy()
        for key, value in raw_tooling.items():
            merged[key] = value
        return merged
    return defaults



























