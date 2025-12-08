import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from html.parser import HTMLParser
import re
from urllib.parse import unquote, urlparse
import os

import redis.asyncio as redis
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import async_sessionmaker

from config.prompts import get_prompt
from core.llm_config import LLMConfigError, get_llm_for_task
from brain_service.models.db import Profile, Work, Author, WorkAuthor
from brain_service.services.sefaria_service import SefariaService
from brain_service.services.wiki_service import WikiService

logger = logging.getLogger(__name__)

ALLOWED_TAGS = {"p", "h2", "h3", "ul", "li", "blockquote", "img", "small", "a"}
ALLOWED_ATTRS = {"a": {"href", "title"}, "img": {"src", "alt"}}


class _AllowlistSanitizer(HTMLParser):
    def __init__(self):
        super().__init__()
        self.output: list[str] = []
        self.tag_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in ALLOWED_TAGS:
            return
        allowed = ALLOWED_ATTRS.get(tag, set())
        clean_attrs = []
        for k, v in attrs:
            if k.lower() in allowed:
                clean_attrs.append((k, v))
        attr_text = "".join(f' {k}="{html_escape(str(v))}"' for k, v in clean_attrs if v)
        self.output.append(f"<{tag}{attr_text}>")
        self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self.tag_stack and tag == self.tag_stack[-1]:
            self.output.append(f"</{tag}>")
            self.tag_stack.pop()

    def handle_data(self, data):
        self.output.append(html_escape(data))


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def sanitize_html(html: str) -> str:
    parser = _AllowlistSanitizer()
    parser.feed(html or "")
    parser.close()
    return "".join(parser.output)


def _parse_year(text: Any) -> Optional[int]:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text)
    if isinstance(text, str):
        digits = re.findall(r"\d{3,4}", text)
        if digits:
            try:
                return int(digits[0])
            except Exception:
                return None
    return None


def parse_year_range(value: Any) -> Optional[Dict[str, Optional[int]]]:
    if value is None:
        return None
    if isinstance(value, list):
        start = _parse_year(value[0]) if len(value) > 0 else None
        end = _parse_year(value[1]) if len(value) > 1 else None
        if start or end:
            return {"start": start, "end": end}
    if isinstance(value, (int, float)):
        v = int(value)
        return {"start": v, "end": v}
    if isinstance(value, str):
        parts = re.split(r"[–-]", value)
        if len(parts) == 2:
            start = _parse_year(parts[0])
            end = _parse_year(parts[1])
            if start or end:
                return {"start": start, "end": end}
        year = _parse_year(value)
        if year:
            return {"start": year, "end": year}
    return None


class ProfileService:
    """
    Сервис для сборки профилей произведений и авторов (Sefaria + Wikipedia + LLM), кеш в Redis/DB.
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        session_factory: async_sessionmaker,
        sefaria_service: SefariaService,
        wiki_service: WikiService,
        *,
        cache_ttl_sec: int = 3600,
    ):
        self._redis = redis_client
        self._session_factory = session_factory
        self._sefaria = sefaria_service
        self._wiki = wiki_service
        self._cache_ttl = cache_ttl_sec
        self._mcp_wiki_url = os.getenv("MCP_WIKI_URL")

    # ---------- cache helpers ----------
    def _cache_key(self, slug: str) -> str:
        return f"profile:v1:{slug}"

    async def _get_cached(self, slug: str) -> Optional[Dict[str, Any]]:
        if not self._redis:
            return None
        try:
            cached = await self._redis.get(self._cache_key(slug))
            if cached:
                return json.loads(cached)
        except Exception as exc:
            logger.warning("PROFILE cache read failed", extra={"slug": slug, "error": str(exc)})
        return None

    async def _set_cached(self, slug: str, payload: Dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            await self._redis.set(self._cache_key(slug), json.dumps(payload, ensure_ascii=False), ex=self._cache_ttl)
        except Exception as exc:
            logger.warning("PROFILE cache write failed", extra={"slug": slug, "error": str(exc)})

    # ---------- helpers ----------
    @staticmethod
    def _normalize_authors(raw: Any) -> Optional[list[str]]:
        if raw is None:
            return None
        authors: list[str] = []
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str) and entry.strip():
                    authors.append(entry.strip())
                elif isinstance(entry, dict):
                    name = entry.get("en") or entry.get("name") or entry.get("he")
                    if isinstance(name, str) and name.strip():
                        authors.append(name.strip())
        elif isinstance(raw, str) and raw.strip():
            authors.append(raw.strip())
        return authors or None

    def _normalize_index_slug(self, slug: str) -> str:
        base = (slug or "").strip()
        base = base.split(",")[0].strip()
        base = re.sub(r"\s+\d+(?::\d+)*$", "", base)
        return base

    def _fallback_index_candidates(self, slug: str) -> list[str]:
        base = self._normalize_index_slug(slug)
        if " on " in base:
            commentator, work = base.split(" on ", 1)
            commentator = commentator.strip()
            work = work.strip()
        else:
            commentator, work = base, ""
        candidates = [base]
        if commentator and "Torah" not in base and "Genesis" not in base:
            candidates.extend([f"{commentator} on Torah", f"{commentator} on Genesis"])
        if work:
            candidates.append(commentator)
        uniq = []
        seen = set()
        for c in candidates:
            if c and c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _detect_lang_priority(self, query: str) -> list[str]:
        has_he = any("\u0590" <= ch <= "\u05ff" for ch in query)
        return ["he", "en", "ru"] if has_he else ["en", "he", "ru"]

    def _pick_best_wiki(self, query: str, results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not results:
            return None
        def norm(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()
        qn = norm(query)
        for res in results:
            title = res.get("title") or ""
            if qn and qn in norm(title):
                return res
        return results[0]

    async def _fetch_author_wiki(self, author_slug: str) -> Optional[Dict[str, Any]]:
        q = author_slug.replace("-", " ").replace("_", " ").strip()
        try:
            wiki_resp = await self._wiki.search_wikipedia(q, lang_priority=self._detect_lang_priority(q))
            if wiki_resp.get("ok"):
                return self._pick_best_wiki(q, (wiki_resp.get("data") or {}).get("results") or [])
        except Exception as exc:
            logger.warning("Failed to fetch author wiki", extra={"author": q, "error": str(exc)})
        return None

    async def _fetch_wiki_by_url(self, wiki_url: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = urlparse(wiki_url)
            title_raw = parsed.path.rsplit("/", 1)[-1]
            title_decoded = unquote(title_raw.replace("_", " "))
            lang = parsed.hostname.split(".")[0] if parsed.hostname else "he"
            # Попробуем сначала скачать страницу напрямую (HTML)
            try:
                resp = await self._wiki.fetch_wikipedia_page(wiki_url)
                if resp:
                    resp.setdefault("url", wiki_url)
                    resp.setdefault("title", title_decoded)
                    return resp
            except Exception as exc:
                logger.warning("Direct fetch wiki page failed", extra={"url": wiki_url, "error": str(exc)})
            # намеренно НЕ делаем поиск — если страница не скачалась, вернём None
        except Exception as exc:
            logger.warning("Failed to fetch wiki by URL", extra={"url": wiki_url, "error": str(exc)})
        return None

    def _fallback_summary(self, bundle: Dict[str, Any]) -> Dict[str, str]:
        idx = bundle.get("sefaria_index") or {}
        wiki = bundle.get("wiki") or {}
        author_wiki = bundle.get("author_wiki") or {}
        work_parts: list[str] = []
        if isinstance(idx, dict):
            desc = idx.get("enDesc") or idx.get("enShortDesc") or idx.get("desc")
            if desc:
                work_parts.append(f"<p>{desc}</p>")
            comp = idx.get("compDate") or idx.get("compPlace") or idx.get("pubDate") or idx.get("pubPlace")
            if comp:
                dates = idx.get("compDate") or idx.get("pubDate")
                places = [idx.get("compPlace"), idx.get("pubPlace")]
                meta_bits = []
                if dates:
                    if isinstance(dates, list) and len(dates) == 2:
                        meta_bits.append(f"Даты: {dates[0]}–{dates[1]}")
                    elif isinstance(dates, list) and dates:
                        meta_bits.append(f"Дата: {dates[0]}")
                    elif isinstance(dates, (int, str)):
                        meta_bits.append(f"Дата: {dates}")
                place_text = ", ".join([p for p in places if p])
                if place_text:
                    meta_bits.append(f"Место: {place_text}")
                if meta_bits:
                    work_parts.append(f"<p>{'; '.join(meta_bits)}</p>")
        author_summary = ""
        if isinstance(author_wiki, dict):
            snippet = author_wiki.get("content") or author_wiki.get("snippet")
            if snippet:
                author_summary = f"<p>{snippet}</p>"
        elif isinstance(wiki, dict):
            snippet = wiki.get("content") or wiki.get("snippet")
            if snippet:
                author_summary = f"<p>{snippet}</p>"
        return {"work": "".join(work_parts), "author": author_summary}

    async def _run_llm(self, bundle: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Calls the LLM to build profile facts. If the bundle is author-only and includes wiki_html/wiki_url,
        use the lightweight timeline_bio agent to save tokens and skip search.
        """
        is_author_only = bundle.get("author_only") or (not bundle.get("sefaria_index"))
        # Choose prompt set
        prompt_id_system = "actions.profile_inspector_system"
        prompt_id_user = "actions.profile_inspector_user_template"
        task_name = "PROFILE"

        if is_author_only and (bundle.get("author_extra") or {}).get("wiki_html"):
            prompt_id_system = "actions.timeline_bio_system"
            prompt_id_user = "actions.timeline_bio_user_template"
            task_name = "TIMELINE_BIO"

        system_prompt = get_prompt(prompt_id_system)
        user_template = get_prompt(prompt_id_user)
        if not system_prompt or not user_template:
            logger.error("Profile inspector prompts are not configured")
            return None
        payload = {
            "work_index": bundle.get("sefaria_index"),
            "wiki": bundle.get("wiki"),
            "author_wiki": bundle.get("author_wiki"),
            "authors": bundle.get("authors"),
            "previous_profile": previous,
            "author_extra": bundle.get("author_extra"),
            "period_override": bundle.get("period_override"),
            "period_ru_override": bundle.get("period_ru_override"),
            "wiki_url": (bundle.get("author_extra") or {}).get("wiki_url") or (bundle.get("author_wiki") or {}).get("url"),
            "wiki_html": (bundle.get("author_extra") or {}).get("wiki_html"),
            "author_only": is_author_only,
        }
        user_prompt = user_template.replace("{payload_json}", json.dumps(payload, ensure_ascii=False, indent=2))
        try:
            llm_client, model, reasoning_params, capabilities = get_llm_for_task(task_name)
        except LLMConfigError as exc:
            logger.error("LLM config error for %s", task_name, extra={"error": str(exc)})
            return None
        req: Dict[str, Any] = {
            **reasoning_params,
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if "json_mode" in capabilities:
            req["response_format"] = {"type": "json_object"}
        try:
            resp = await llm_client.chat.completions.create(**req)
        except Exception as exc:
            logger.error("LLM call for PROFILE failed", extra={"error": str(exc)}, exc_info=True)
            return None
        content = (resp.choices[0].message.content or "").strip() if resp and resp.choices else ""
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to decode PROFILE LLM JSON, returning raw string")
            return {"summary_html": content}

    def _ensure_fact_defaults(self, facts_work: Dict[str, Any], facts_author: Dict[str, Any]) -> None:
        for key, default in [
            ("title_en", None),
            ("title_he", None),
            ("compDate", None),
            ("pubDate", None),
            ("compPlace", None),
            ("pubPlace", None),
            ("categories", []),
            ("authors", []),
            ("links", {}),
            ("images", []),
            ("comp_range", None),
            ("pub_range", None),
            ("display", {}),
        ]:
            facts_work.setdefault(key, default)
        for key, default in [
            ("title_en", None),
            ("title_he", None),
            ("lifespan", None),
            ("period", None),
            ("links", {}),
            ("images", []),
            ("lifespan_range", None),
            ("display", {}),
        ]:
            facts_author.setdefault(key, default)

    def _build_payload(self, slug: str, bundle: Dict[str, Any], generated: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        idx = bundle.get("sefaria_index") or {}
        wiki = bundle.get("wiki") or {}
        author_wiki = bundle.get("author_wiki") or {}

        fallback = self._fallback_summary(bundle)
        summary_work_html = (generated or {}).get("summary_work_html") or fallback.get("work") or ""
        summary_author_html = (generated or {}).get("summary_author_html") or fallback.get("author") or ""

        facts_work_raw = (generated or {}).get("facts_work") or {}
        facts_author_raw = (generated or {}).get("facts_author") or {}
        facts_work: Dict[str, Any] = facts_work_raw if isinstance(facts_work_raw, dict) else {}
        facts_author: Dict[str, Any] = facts_author_raw if isinstance(facts_author_raw, dict) else {}

        self._ensure_fact_defaults(facts_work, facts_author)

        now_iso = datetime.utcnow().isoformat() + "Z"
        facts_work.setdefault("generated_at", now_iso)
        facts_author.setdefault("generated_at", now_iso)

        authors = (
            self._normalize_authors(facts_work.get("authors"))
            or self._normalize_authors(idx.get("authors"))
            or self._normalize_authors(bundle.get("authors"))
        )

        title_en = idx.get("title") or wiki.get("title") or slug
        title_he = idx.get("heTitle") or idx.get("heTitleFull")

        links_work = {}
        if isinstance(facts_work.get("links"), dict):
            links_work.update(facts_work["links"])
        if idx.get("refs"):
            links_work.setdefault("sefaria", idx.get("refs", {}).get("sefaria") if isinstance(idx.get("refs"), dict) else None)
        if wiki.get("url"):
            links_work.setdefault("wikipedia", wiki.get("url"))
        links_work = {k: v for k, v in links_work.items() if v}

        categories = facts_work.get("categories") or idx.get("categories")
        facts_work["title_en"] = facts_work.get("title_en") or title_en
        facts_work["title_he"] = facts_work.get("title_he") or title_he
        if authors:
            facts_work["authors"] = authors
        if categories:
            facts_work["categories"] = categories
        if links_work:
            facts_work["links"] = links_work
        if not facts_work.get("comp_range"):
            facts_work["comp_range"] = parse_year_range(facts_work.get("compDate"))
        if not facts_work.get("pub_range"):
            facts_work["pub_range"] = parse_year_range(facts_work.get("pubDate"))

        if facts_author is not None:
            if "title_en" not in facts_author and isinstance(author_wiki, dict) and author_wiki.get("title"):
                facts_author["title_en"] = author_wiki.get("title")
            if "links" not in facts_author and isinstance(author_wiki, dict) and author_wiki.get("url"):
                facts_author["links"] = {"wikipedia": author_wiki.get("url")}
            if not facts_author.get("lifespan_range"):
                facts_author["lifespan_range"] = parse_year_range(facts_author.get("lifespan"))

        combined_summary = "".join(filter(None, [summary_work_html, summary_author_html]))
        facts_combined = {
            "work": facts_work,
            "author": facts_author,
            "summary_work_html": summary_work_html,
            "summary_author_html": summary_author_html,
        }

        return {
            "ok": True,
            "slug": slug,
            "title_en": title_en,
            "title_he": title_he,
            "summary_html": combined_summary,
            "summary_work_html": summary_work_html,
            "summary_author_html": summary_author_html,
            "facts": facts_combined,
            "json_raw": bundle,
            "is_verified": False,
            "verified_by": None,
            "verified_at": None,
            "source": "generated",
        }

    def _enrich_profile_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pull commonly used timeline metadata (period/lifespan/region/generation/subPeriod)
        from facts into top-level fields so they are stored in the DB and surfaced by list_profiles.
        """
        facts = payload.get("facts") or {}
        author_facts = facts.get("author") if isinstance(facts, dict) else {}
        author_facts = author_facts if isinstance(author_facts, dict) else {}

        payload.setdefault("period", author_facts.get("period"))
        payload.setdefault("lifespan", author_facts.get("lifespan"))
        payload.setdefault("region", author_facts.get("region"))
        payload.setdefault("generation", author_facts.get("generation"))
        payload.setdefault("sub_period", author_facts.get("subPeriod") or author_facts.get("sub_period"))

        display = author_facts.get("display") if isinstance(author_facts, dict) else {}
        display = display if isinstance(display, dict) else {}
        if display.get("period_ru"):
            payload.setdefault("display", {})
            if isinstance(payload["display"], dict):
                payload["display"].setdefault("period_ru", display.get("period_ru"))
        return payload

    # ---------- DB helpers ----------
    async def _profile_to_payload(self, profile: Profile) -> Dict[str, Any]:
        summary = profile.manual_summary_html or profile.summary_html
        facts = profile.manual_facts or profile.facts or {}
        source = "manual" if profile.manual_summary_html or profile.manual_facts else "generated"
        author_facts = facts.get("author") if isinstance(facts, dict) else {}
        author_facts = author_facts if isinstance(author_facts, dict) else {}
        return {
            "ok": True,
            "slug": profile.slug,
            "title_en": profile.title_en,
            "title_he": profile.title_he,
            "summary_html": summary,
            "summary_work_html": facts.get("summary_work_html") if isinstance(facts, dict) else None,
            "summary_author_html": facts.get("summary_author_html") if isinstance(facts, dict) else None,
            "facts": facts,
            "json_raw": profile.json_raw,
            "is_verified": profile.is_verified,
            "verified_by": profile.verified_by,
            "verified_at": profile.verified_at.isoformat() if profile.verified_at else None,
            "source": source,
            "period": profile.period or author_facts.get("period"),
            "lifespan": profile.lifespan or author_facts.get("lifespan"),
            "region": author_facts.get("region"),
            "generation": author_facts.get("generation"),
            "subPeriod": author_facts.get("subPeriod") or author_facts.get("sub_period"),
        }

    async def _get_db_profile(self, slug: str) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(select(Profile).where(Profile.slug == slug))
            obj = result.first()
            if obj:
                return await self._profile_to_payload(obj)
        return None

    async def _save_db_profile(self, slug: str, payload: Dict[str, Any], raw_bundle: Dict[str, Any], *, clear_manual: bool = False) -> None:
        facts = payload.get("facts") or {}
        author_facts = facts.get("author") if isinstance(facts, dict) else {}
        author_facts = author_facts if isinstance(author_facts, dict) else {}
        async with self._session_factory() as session:
            async with session.begin():
                existing = await session.get(Profile, slug)
                if not existing:
                    existing = Profile(slug=slug)
                    session.add(existing)
                existing.title_en = payload.get("title_en")
                existing.title_he = payload.get("title_he")
                existing.summary_html = payload.get("summary_html")
                existing.facts = payload.get("facts") or {}
                existing.json_raw = raw_bundle
                existing.period = payload.get("period") or author_facts.get("period") or existing.period
                existing.lifespan = payload.get("lifespan") or author_facts.get("lifespan") or existing.lifespan
                if clear_manual:
                    existing.manual_summary_html = None
                    existing.manual_facts = None
                    existing.is_verified = False
                    existing.verified_by = None
                    existing.verified_at = None
                await session.flush()

    async def _persist_work_author(
        self,
        slug: str,
        bundle: Dict[str, Any],
        summary_work_html: str,
        summary_author_html: str,
        facts_work: Dict[str, Any],
        facts_author: Dict[str, Any],
        authors_list: list[str],
    ) -> None:
        idx = bundle.get("sefaria_index") or {}
        async with self._session_factory() as session:
            async with session.begin():
                work = await session.get(Work, slug)
                if not work:
                    work = Work(index_title=slug)
                    session.add(work)
                work.title_en = idx.get("title") or facts_work.get("title_en") or slug
                work.title_he = idx.get("heTitle") or facts_work.get("title_he")
                work.en_desc = idx.get("enDesc") or idx.get("enShortDesc")
                work.comp_date = idx.get("compDate") or facts_work.get("compDate")
                work.pub_date = idx.get("pubDate") or facts_work.get("pubDate")
                work.comp_place = idx.get("compPlace") or facts_work.get("compPlace")
                work.pub_place = idx.get("pubPlace") or facts_work.get("pubPlace")
                work.categories = idx.get("categories") or facts_work.get("categories")
                work.links = facts_work.get("links")
                work.summary_html = summary_work_html

                for author_slug in authors_list or []:
                    a = await session.get(Author, author_slug)
                    if not a:
                        a = Author(slug=author_slug)
                        session.add(a)
                    a.name_en = facts_author.get("title_en") or author_slug.replace("-", " ")
                    a.name_he = facts_author.get("title_he") or a.name_he
                    a.summary_html = summary_author_html or a.summary_html
                    a.lifespan = facts_author.get("lifespan") or a.lifespan
                    a.period = facts_author.get("period") or a.period
                    if facts_author.get("links"):
                        a.links = facts_author.get("links")
                    link_exists = await session.scalars(
                        select(WorkAuthor).where(WorkAuthor.work_id == slug, WorkAuthor.author_id == author_slug)
                    )
                    if not link_exists.first():
                        session.add(WorkAuthor(work_id=slug, author_id=author_slug))
                await session.flush()

    async def _persist_author_only(self, slug: str, facts_author: Dict[str, Any], summary_author_html: str) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                a = await session.get(Author, slug)
                if not a:
                    a = Author(slug=slug)
                    session.add(a)
                a.name_en = facts_author.get("title_en") or slug
                a.name_he = facts_author.get("title_he")
                a.summary_html = summary_author_html or a.summary_html
                a.lifespan = facts_author.get("lifespan") or a.lifespan
                a.period = facts_author.get("period") or a.period
                if facts_author.get("links"):
                    a.links = facts_author.get("links")
                await session.flush()

    # ---------- Public API ----------
    async def get_profile(self, slug: str) -> Dict[str, Any]:
        slug = (slug or "").strip()
        if not slug:
            return {"ok": False, "error": "slug is required"}
        cached = await self._get_cached(slug)
        if cached:
            return cached
        stored = await self._get_db_profile(slug)
        if stored:
            await self._set_cached(slug, stored)
            return stored
        bundle = await self._fetch_sources(slug)
        if not bundle.get("sefaria_index") and not bundle.get("wiki") and not bundle.get("author_wiki"):
            return {"ok": False, "error": "Profile sources not found"}
        generated = await self._run_llm(bundle, stored)
        payload = self._enrich_profile_metadata(self._build_payload(slug, bundle, generated))
        facts = payload.get("facts") or {}
        await self._persist_work_author(
            slug,
            bundle,
            payload.get("summary_work_html") or "",
            payload.get("summary_author_html") or "",
            (facts or {}).get("work") or {},
            (facts or {}).get("author") or {},
            bundle.get("authors") or [],
        )
        await self._save_db_profile(slug, payload, bundle)
        await self._set_cached(slug, payload)
        return payload

    async def regenerate_profile(self, slug: str) -> Dict[str, Any]:
        if self._redis:
            try:
                await self._redis.delete(self._cache_key(slug))
            except Exception:
                pass
        bundle = await self._fetch_sources(slug)
        if not bundle.get("sefaria_index") and not bundle.get("wiki") and not bundle.get("author_wiki"):
            return {"ok": False, "error": "Profile sources not found"}
        generated = await self._run_llm(bundle, None)
        payload = self._enrich_profile_metadata(self._build_payload(slug, bundle, generated))
        facts = payload.get("facts") or {}
        await self._persist_work_author(
            slug,
            bundle,
            payload.get("summary_work_html") or "",
            payload.get("summary_author_html") or "",
            (facts or {}).get("work") or {},
            (facts or {}).get("author") or {},
            bundle.get("authors") or [],
        )
        await self._save_db_profile(slug, payload, bundle, clear_manual=True)
        await self._set_cached(slug, payload)
        return payload

    async def save_manual_profile(
        self,
        slug: str,
        summary_html: str | None,
        facts: dict | None,
        verified_by: str | None,
        title_en: str | None = None,
        title_he: str | None = None,
    ) -> Dict[str, Any]:
        sanitized_html = sanitize_html(summary_html or "") if summary_html else None
        async with self._session_factory() as session:
            async with session.begin():
                profile = await session.get(Profile, slug)
                if not profile:
                    profile = Profile(slug=slug, title_en=slug)
                    session.add(profile)
                if sanitized_html is not None:
                    profile.manual_summary_html = sanitized_html
                if facts is not None:
                    profile.manual_facts = facts
                if title_en is not None:
                    profile.title_en = title_en.strip()
                if title_he is not None:
                    profile.title_he = title_he.strip()
                profile.is_verified = True
                profile.verified_by = verified_by
                profile.verified_at = datetime.utcnow()
                await session.flush()
                await session.refresh(profile)
                payload = await self._profile_to_payload(profile)
                await self._set_cached(slug, payload)
                return payload

    async def list_profiles(self, *, query: str | None = None, only_unverified: bool = False, limit: int = 100) -> Dict[str, Any]:
        stmt = select(Profile).order_by(Profile.updated_at.desc()).limit(max(1, min(limit, 500)))
        if query:
            like = f"%{query.lower()}%"
            stmt = stmt.where(or_(Profile.slug.ilike(like), Profile.title_en.ilike(like), Profile.title_he.ilike(like)))
        if only_unverified:
            stmt = stmt.where(Profile.is_verified.is_(False))
        async with self._session_factory() as session:
            result = await session.scalars(stmt)
            items = []
            for profile in result:
                facts = profile.manual_facts or profile.facts or {}
                author_facts = facts.get("author") if isinstance(facts, dict) else {}
                author_facts = author_facts if isinstance(author_facts, dict) else {}
                period_val = profile.period or author_facts.get("period")
                lifespan_val = profile.lifespan or author_facts.get("lifespan")
                items.append({
                    "slug": profile.slug,
                    "title_en": profile.title_en,
                    "title_he": profile.title_he,
                    "is_verified": profile.is_verified,
                    "verified_by": profile.verified_by,
                    "verified_at": profile.verified_at.isoformat() if profile.verified_at else None,
                    "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
                    "source": "manual" if profile.manual_summary_html or profile.manual_facts else "generated",
                    "period": period_val,
                    "lifespan": lifespan_val,
                    "region": author_facts.get("region"),
                    "generation": author_facts.get("generation"),
                    "subPeriod": author_facts.get("subPeriod") or author_facts.get("sub_period"),
                    "summary_html": profile.summary_html or profile.manual_summary_html,
                    "facts": facts,
                })
            return {"ok": True, "items": items}

    async def delete_profile(self, slug: str) -> None:
        if self._redis:
            try:
                await self._redis.delete(self._cache_key(slug))
            except Exception:
                pass
        async with self._session_factory() as session:
            async with session.begin():
                obj = await session.get(Profile, slug)
                if obj:
                    await session.delete(obj)

    async def generate_author_profile(
        self,
        name: str,
        wiki_url: str | None = None,
        raw_text: str | None = None,
        period: str | None = None,
        period_ru: str | None = None,
        region: str | None = None,
        generation: int | None = None,
        sub_period: str | None = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        slug = name.strip()
        if not force:
            existing = await self._get_db_profile(slug)
            if existing:
                existing["ok"] = True
                existing["skipped"] = True
                existing["reason"] = "already_exists"
                return existing
        author_wiki = None
        author_extra = {}
        # 1) если есть wiki_url — пробуем сразу скачать страницу и НЕ делаем поиск
        if wiki_url:
            author_extra["wiki_url"] = wiki_url
            page = None
            # сначала пытаемся взять очищенный контент с MCP, если настроен
            if self._mcp_wiki_url:
                page = await self._wiki.fetch_wikipedia_page_via_mcp(self._mcp_wiki_url, wiki_url)
            if not page:
                page = await self._wiki.fetch_wikipedia_page(wiki_url)
            if page:
                author_wiki = {"url": wiki_url, "title": page.get("title") or slug, "content": page.get("content"), "html": page.get("html")}
                if page.get("html"):
                    author_extra["wiki_html"] = page.get("html")
                else:
                    author_extra["wiki_html"] = page.get("content") or " "
            else:
                # даже если не скачали HTML, всё равно передаём URL и пропускаем поиск
                author_wiki = {"url": wiki_url, "title": slug}
                author_extra.setdefault("wiki_html", raw_text or " ")
        # 2) если нет wiki_url и нет raw_html — fallback на поиск по имени
        if not wiki_url and author_wiki is None and "wiki_html" not in author_extra:
            author_wiki = await self._fetch_author_wiki(slug)
        # 3) если raw_text передан — используем как wiki_html для агента
        if raw_text:
            author_extra["wiki_html"] = raw_text
        bundle = {
            "sefaria_index": None,
            "wiki": None,
            "author_wiki": author_wiki,
            "authors": [slug],
            "author_extra": author_extra,
            "period_override": period,
            "period_ru_override": period_ru,
            "region_override": region,
            "generation_override": generation,
            "sub_period_override": sub_period,
        }
        generated = await self._run_llm(bundle, None)
        payload = self._build_payload(slug, bundle, generated)
        facts = payload.get("facts") or {}
        if period and isinstance(facts.get("author"), dict):
            facts["author"]["period"] = period
        if period_ru and isinstance(facts.get("author"), dict):
            facts["author"].setdefault("display", {})
            facts["author"]["display"]["period_ru"] = period_ru
        if region and isinstance(facts.get("author"), dict):
            facts["author"]["region"] = region
        if generation and isinstance(facts.get("author"), dict):
            facts["author"]["generation"] = generation
        if sub_period and isinstance(facts.get("author"), dict):
            facts["author"]["subPeriod"] = sub_period
        facts["work"] = {}
        payload["summary_work_html"] = ""
        payload["summary_html"] = payload.get("summary_author_html") or payload.get("summary_html")
        payload["facts"] = facts
        payload = self._enrich_profile_metadata(payload)
        await self._persist_author_only(slug, facts.get("author") or {}, payload.get("summary_author_html") or "")
        await self._save_db_profile(slug, payload, bundle, clear_manual=True)
        await self._set_cached(slug, payload)
        return payload

    async def _fetch_sources(self, slug: str) -> Dict[str, Any]:
        sefaria_index = None
        wiki = None
        author_wiki = None

        for candidate in self._fallback_index_candidates(slug):
            try:
                resp = await self._sefaria.get_index_raw(candidate)
                if resp.get("ok") and resp.get("data"):
                    sefaria_index = resp.get("data")
                    break
            except Exception as exc:
                logger.warning("Failed to fetch Sefaria index", extra={"slug": candidate, "error": str(exc)})

        wiki_query = slug
        if isinstance(sefaria_index, dict):
            wiki_query = sefaria_index.get("title") or sefaria_index.get("heTitle") or slug

        try:
            wiki_resp = await self._wiki.search_wikipedia(wiki_query, lang_priority=self._detect_lang_priority(wiki_query))
            if wiki_resp.get("ok"):
                wiki = self._pick_best_wiki(wiki_query, (wiki_resp.get("data") or {}).get("results") or [])
        except Exception as exc:
            logger.warning("Failed to fetch Wikipedia", extra={"slug": slug, "error": str(exc), "query": wiki_query})

        authors_list = []
        if isinstance(sefaria_index, dict):
            raw_authors = sefaria_index.get("authors") or []
            if isinstance(raw_authors, list):
                authors_list = [a for a in raw_authors if isinstance(a, str) and a.strip()]
        if not authors_list and wiki and isinstance(wiki, dict):
            t = wiki.get("title")
            if isinstance(t, str) and t.strip():
                authors_list = [t.strip()]

        if authors_list:
            author_wiki = await self._fetch_author_wiki(authors_list[0])

        return {"sefaria_index": sefaria_index, "wiki": wiki, "author_wiki": author_wiki, "authors": authors_list}
