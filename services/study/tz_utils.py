from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

FALLBACK_TZ = "America/New_York"


def resolve_timezone(user_tz: str | None, calendar_tz: str | None) -> ZoneInfo:
    """Pick the timezone to use for daily calculations."""
    candidates = [user_tz, calendar_tz, FALLBACK_TZ]
    for tz_name in candidates:
        if not tz_name:
            continue
        try:
            return ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            continue
    # ultimate fallback
    return ZoneInfo(FALLBACK_TZ)


def now_in_tz(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


def next_midnight(tz: ZoneInfo, from_dt: datetime | None = None) -> datetime:
    base = from_dt.astimezone(tz) if from_dt else now_in_tz(tz)
    midnight = base.replace(hour=0, minute=0, second=0, microsecond=0)
    if base == midnight:
        return midnight
    return (midnight + timedelta(days=1)) if base > midnight else midnight


def seconds_until_next_midnight(tz: ZoneInfo, from_dt: datetime | None = None) -> int:
    base = from_dt.astimezone(tz) if from_dt else now_in_tz(tz)
    target = next_midnight(tz, base)
    delta = target - base
    return max(int(delta.total_seconds()), 60)
