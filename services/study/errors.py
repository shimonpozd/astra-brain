""" Typed errors for the modular study service. """

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Dict, Optional, Tuple


@dataclass(slots=True)
class StudyError(Exception):
    """Base class for all study-domain errors."""

    message: str
    ref: Optional[str] = None
    session_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    code: str = "study_error"
    status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:  # pragma: no cover - simple passthrough
        return self.message


class RangeNotFound(StudyError):
    code = "study.range_not_found"
    status = HTTPStatus.NOT_FOUND


class RangeValidationError(StudyError):
    code = "study.range_invalid"
    status = HTTPStatus.BAD_REQUEST


class NavigationBoundsExceeded(StudyError):
    code = "study.navigation_bounds"
    status = HTTPStatus.BAD_REQUEST


class DailyLockBusy(StudyError):
    code = "study.daily_lock_busy"
    status = HTTPStatus.ACCEPTED


class DailyIdempotencyCollision(StudyError):
    code = "study.daily_idempotency"
    status = HTTPStatus.CONFLICT


class BookshelfUnavailable(StudyError):
    code = "study.bookshelf_unavailable"
    status = HTTPStatus.SERVICE_UNAVAILABLE


class StudyConfigInvalid(StudyError):
    code = "study.config_invalid"
    status = HTTPStatus.INTERNAL_SERVER_ERROR


def to_http_payload(error: StudyError) -> Tuple[int, Dict[str, Any]]:
    """Convert a study error into an HTTP payload tuple."""

    status_code = int(error.status)
    body: Dict[str, Any] = {
        "error": {
            "code": error.code,
            "message": error.message,
            "ref": error.ref,
            "session_id": error.session_id,
            "detail": error.detail or None,
        }
    }
    return status_code, body
