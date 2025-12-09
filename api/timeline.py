import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from core.dependencies import get_profile_service, get_current_user
from brain_service.services.profile_service import ProfileService
from brain_service.models.db import Author

logger = logging.getLogger(__name__)
router = APIRouter()


class TimelinePerson(BaseModel):
  slug: str
  name_en: str
  name_he: str | None = None
  name_ru: str | None = None
  birthYear: int | None = None
  deathYear: int | None = None
  flouritYear: int | None = None
  lifespan: str | None = None
  lifespan_range: dict | None = None
  period: str
  subPeriod: str | None = None
  generation: int | None = None
  region: str | None = None
  summary_html: str | None = None
  categories: list[str] | None = None
  images: list[str] | None = None
  is_verified: bool | None = None


class TimelineResponse(BaseModel):
  people: List[TimelinePerson]
  periods: list[dict]


@router.get("/timeline/people", response_model=TimelineResponse)
async def get_timeline_people(
  q: Optional[str] = None,
  periods: Optional[str] = None,
  regions: Optional[str] = None,
  generations: Optional[str] = None,
  start: Optional[int] = None,
  end: Optional[int] = None,
  profile_service: ProfileService = Depends(get_profile_service),
):
  """
  Public read endpoint for timeline people derived from profiles.
  Public (no auth); returns periods + lightweight people list.
  """
  period_filter = set(p.strip() for p in (periods.split(",") if periods else []) if p.strip())
  region_filter = set(r.strip() for r in (regions.split(",") if regions else []) if r.strip())
  generation_filter = set(int(g) for g in (generations.split(",") if generations else []) if g.strip().isdigit())

  try:
    res = await profile_service.list_profiles(query=q.strip() if q else None, only_unverified=False, limit=5000)
    profiles = res.get("items") if isinstance(res, dict) else res
  except Exception as exc:
    logger.error("timeline:list_profiles failed", extra={"error": str(exc)})
    raise HTTPException(status_code=500, detail="Failed to load timeline data")

  people: List[TimelinePerson] = []
  if not isinstance(profiles, list):
    profiles = []

  def _normalize_period(period_val: Optional[str], sub_period_val: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Map legacy/child period ids to canonical top-level periods."""
    if not period_val:
      return None, sub_period_val
    lower = period_val.lower()
    base = lower.split(":")[0]
    # Танах: любые torah_* -> torah
    if base.startswith("torah_") or base in {"patriarchs", "twelve_tribes", "postflood_nations", "postflood_root", "torah"}:
      return "torah", sub_period_val or lower
    # Шофтим: shoftim_generations -> shoftim
    if base.startswith("shoftim"):
      return "shoftim", sub_period_val or lower
    # Цари разделённые: malakhim_divided_israel/judah -> malakhim_divided
    if base.startswith("malakhim_divided"):
      return "malakhim_divided", sub_period_val or base.replace("malakhim_divided_", "")
    # Таннаим/Амораим сокращения
    if base in {"tanna_second"}:
      return "tannaim_temple", sub_period_val
    if base in {"tanna_post"}:
      return "tannaim_post_temple", sub_period_val
    if base in {"amora_eretz"}:
      return "amoraim_israel", sub_period_val
    if base in {"amora_bavel"}:
      return "amoraim_babylonia", sub_period_val
    # Савораим/Гаоним подварианты
    if base.startswith("savora_") or base.startswith("savoraim_"):
      return "savoraim", sub_period_val or lower
    if base.startswith("gaon_") or base.startswith("gaonim_"):
      return "geonim", sub_period_val or lower
    return base, sub_period_val

  def _infer_period_from_sub(sub_period_val: Optional[str]) -> Optional[str]:
    """Best-effort derive period from subPeriod if period is missing."""
    if not sub_period_val:
      return None
    sub = sub_period_val.lower().split(":")[0]
    if sub.startswith("preflood") or sub.startswith("flood_") or sub.startswith("postflood") or sub.startswith("patriarchs") or sub.startswith("tribe_"):
      return "torah"
    if sub.startswith("shoftim"):
      return "shoftim"
    if sub.startswith("tanna_temple") or sub.startswith("tanna_second"):
      return "tannaim_temple"
    if sub.startswith("tanna_post"):
      return "tannaim_post_temple"
    if sub.startswith("amora_israel"):
      return "amoraim_israel"
    if sub.startswith("amora_bav"):
      return "amoraim_babylonia"
    if sub.startswith("savora"):
      return "savoraim"
    if sub.startswith("gaon_sura") or sub.startswith("gaon_pumbedita") or sub.startswith("gaon_israel"):
      return "geonim"
    return None

  for p in profiles:
    slug = p.get("slug") if isinstance(p, dict) else None
    if not slug:
      continue
    facts = p.get("facts") or {}
    if not isinstance(facts, dict):
      facts = {}
    author_facts = facts.get("author") if isinstance(facts, dict) else {}
    period_val = p.get("period") or (author_facts.get("period") if isinstance(author_facts, dict) else None)
    lifespan_val = p.get("lifespan") or (author_facts.get("lifespan") if isinstance(author_facts, dict) else None)
    def _to_int(val):
      if val is None:
        return None
      if isinstance(val, (int, float)):
        return int(val)
      if isinstance(val, str) and val.strip().lstrip("+-").isdigit():
        return int(val.strip())
      return None

    generation_val_raw = p.get("generation") or (author_facts.get("generation") if isinstance(author_facts, dict) else None)
    generation_val = _to_int(generation_val_raw)
    region_val = p.get("region") or p.get("region_id") or p.get("period_region") or (author_facts.get("region") if isinstance(author_facts, dict) else None)
    sub_period_val = p.get("subPeriod") or p.get("period_sub") or (author_facts.get("subPeriod") if isinstance(author_facts, dict) else None)
    lifespan_range = author_facts.get("lifespan_range") if isinstance(author_facts, dict) else None
    categories = author_facts.get("categories") if isinstance(author_facts, dict) else None
    images = author_facts.get("images") if isinstance(author_facts, dict) else None
    display = author_facts.get("display") if isinstance(author_facts, dict) else {}
    if not isinstance(display, dict):
      display = {}

    # Fallback to authors table if period/lifespan absent
    if not period_val or not lifespan_val:
      try:
        async with profile_service._session_factory() as session:  # type: ignore[attr-defined]
          author_row = await session.scalar(select(Author).where(Author.slug == slug))
          if author_row:
            period_val = period_val or author_row.period
            lifespan_val = lifespan_val or author_row.lifespan
            if not p.get("title_en") and author_row.name_en:
              p["title_en"] = author_row.name_en
            if not p.get("title_he") and author_row.name_he:
              p["title_he"] = author_row.name_he
            if not region_val and isinstance(author_row.links, dict):
              region_val = author_row.links.get("region")
            if not generation_val and isinstance(author_row.links, dict):
              generation_val = author_row.links.get("generation")
      except Exception as exc:
        logger.warning("timeline:author_fallback failed", extra={"slug": slug, "error": str(exc)})

    # Если период не задан, попробуем вывести его из subPeriod
    if not period_val:
      period_val = _infer_period_from_sub(sub_period_val)

    period_val, sub_period_val = _normalize_period(period_val, sub_period_val)

    # Если поколение не задано, но номер зашит в sub_period (genN) — извлечём.
    if generation_val is None and isinstance(sub_period_val, str):
      import re
      m = re.search(r"gen(\d+)", sub_period_val)
      if m:
        generation_val = int(m.group(1))

    # As last resort, hit full profile to extract period/lifespan
    if not period_val or period_val == "achronim" or not p.get("title_ru"):
      try:
        full_profile = await profile_service.get_profile(slug)
        if isinstance(full_profile, dict):
          period_val = full_profile.get("period") or period_val
          lifespan_val = full_profile.get("lifespan") or lifespan_val
          region_val = region_val or full_profile.get("region")
          if not p.get("title_ru"):
            facts_full = full_profile.get("facts") or {}
            display_full = (facts_full.get("author") or {}).get("display") if isinstance(facts_full, dict) else {}
            if isinstance(display_full, dict):
              p["title_ru"] = display_full.get("name_ru")
      except Exception as exc:
        logger.debug("timeline:get_profile fallback failed", extra={"slug": slug, "error": str(exc)})

    images_list = []
    if isinstance(images, list):
      for img in images:
        if isinstance(img, str):
          images_list.append(img)
        elif isinstance(img, dict) and img.get("url"):
          images_list.append(str(img.get("url")))

    categories_list = categories if isinstance(categories, list) else None

    person = TimelinePerson(
      slug=slug,
      name_en=p.get("title_en") or display.get("name_en") or slug,
      name_he=p.get("title_he"),
      name_ru=p.get("title_ru") or display.get("name_ru"),
      lifespan=lifespan_val,
      lifespan_range=lifespan_range if isinstance(lifespan_range, dict) else None,
      period=period_val or "achronim",
      subPeriod=sub_period_val,
      generation=generation_val,
      region=region_val,
      summary_html=None,  # тяжелый HTML не нужен для таймлайна
      categories=categories_list,
      images=images_list or None,
      is_verified=p.get("is_verified"),
    )

    # Populate crude birth/death years when possible
    if isinstance(person.lifespan_range, dict):
      person.birthYear = person.lifespan_range.get("start")
      person.deathYear = person.lifespan_range.get("end")
    elif isinstance(person.lifespan, str):
      parts = [s for s in person.lifespan.replace("c.", "").replace("ca.", "").replace("~", " ").split("-") if s.strip()]
      if len(parts) >= 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
        person.birthYear = int(parts[0].strip())
        person.deathYear = int(parts[1].strip())

    people.append(person)

  def _matches_filters(person: TimelinePerson) -> bool:
    if period_filter and person.period not in period_filter:
      return False
    if region_filter and person.region and person.region not in region_filter:
      return False
    if generation_filter and person.generation is not None and person.generation not in generation_filter:
      return False
    if start is not None or end is not None:
      def _to_int(val):
        if val is None:
          return None
        if isinstance(val, (int, float)):
          return int(val)
        if isinstance(val, str) and val.strip().lstrip("+-").isdigit():
          return int(val.strip())
        return None
      from_year = _to_int(person.birthYear) or _to_int(person.lifespan_range.get("start") if person.lifespan_range else None)
      to_year = _to_int(person.deathYear) or _to_int(person.lifespan_range.get("end") if person.lifespan_range else None)
      if from_year is not None and start is not None and to_year is not None and end is not None:
        if to_year < start or from_year > end:
          return False
    return True

  people = [p for p in people if _matches_filters(p)]

  periods_config = getattr(profile_service, "periods_config", lambda: [])()
  return TimelineResponse(
    people=people,
    periods=[p.model_dump() if hasattr(p, "model_dump") else p for p in periods_config] if periods_config else [],
  )
