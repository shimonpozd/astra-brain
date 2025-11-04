from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

from .tz_utils import now_in_tz


@dataclass(frozen=True)
class StreamMeta:
    stream_id: str
    title: Dict[str, str]
    units_total: int
    unit_index_today: int


ALIAH_AMOUNT = 7


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "stream"


def jewish_weekday(dt: datetime) -> int:
    # Diaspora mapping: Sunday=0 ... Saturday=6
    return (dt.weekday() + 1) % 7


def _stream_id_for_parasha(title_en: str, today: datetime) -> str:
    iso_year, iso_week, _ = today.isocalendar()
    slug = slugify(title_en)
    return f"parasha-{slug}-{iso_year}w{iso_week:02d}"


def _stream_id_for_daily(title_en: str, today: datetime) -> str:
    slug = slugify(title_en)
    date_str = today.strftime("%Y-%m-%d")
    return f"{slug}-{date_str}"


def select_today_unit(
    item: Dict,
    *,
    tz: ZoneInfo,
    override_today: datetime | None = None,
) -> Tuple[str, StreamMeta]:
    today = override_today.astimezone(tz) if override_today else now_in_tz(tz)

    title = item.get("title") or {}
    title_en = title.get("en") or item.get("displayValue", {}).get("en", "") or "Daily Stream"
    title_he = title.get("he") or item.get("displayValue", {}).get("he", "") or title_en
    ref = item.get("ref") or item.get("url")

    if not ref:
        raise ValueError("Calendar item does not provide a reference")

    extra_details = item.get("extraDetails") or {}
    aliyot_raw = extra_details.get("aliyot") or []
    aliyot = [r for r in aliyot_raw if r][:ALIAH_AMOUNT]

    if "parashat hashavua" in title_en.lower() and len(aliyot) >= ALIAH_AMOUNT:
        dow = min(jewish_weekday(today), ALIAH_AMOUNT - 1)
        unit_ref = aliyot[dow]
        meta = StreamMeta(
            stream_id=_stream_id_for_parasha(title_en, today),
            title={"en": title_en, "he": title_he},
            units_total=ALIAH_AMOUNT,
            unit_index_today=dow,
        )
        return unit_ref, meta

    # fallback for parsha without full aliyot or for other streams
    meta = StreamMeta(
        stream_id=_stream_id_for_daily(title_en, today),
        title={"en": title_en, "he": title_he},
        units_total=1,
        unit_index_today=0,
    )
    return ref, meta
