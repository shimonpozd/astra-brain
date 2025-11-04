"""Utilities for loading study configuration via ConfigService."""

from __future__ import annotations

from typing import Awaitable, Callable

from pydantic import ValidationError

from ..config_service import ConfigService
from .config_schema import StudyConfig, load_study_config
from .errors import StudyConfigInvalid

StudyConfigCallback = Callable[[StudyConfig], Awaitable[None]]


def _raise_invalid(error: ValidationError) -> StudyConfigInvalid:
    return StudyConfigInvalid(
        "Invalid study configuration",
        detail={"errors": error.errors(), "type": "validation_error"},
    )


async def fetch_study_config(config_service: ConfigService) -> StudyConfig:
    """Fetch and validate the study configuration from ``ConfigService``."""

    raw_config = await config_service.get_config_section("study", default=None)
    try:
        return load_study_config(raw_config)
    except ValidationError as exc:
        raise _raise_invalid(exc) from exc


async def register_study_config_listener(
    config_service: ConfigService, callback: StudyConfigCallback
) -> None:
    """Register a listener that receives validated ``StudyConfig`` instances."""

    async def _wrapped(section_data):
        try:
            config = load_study_config(section_data)
        except ValidationError as exc:
            raise _raise_invalid(exc) from exc
        await callback(config)

    await config_service.register_listener("study", _wrapped)


__all__ = ["fetch_study_config", "register_study_config_listener", "StudyConfigCallback"]
