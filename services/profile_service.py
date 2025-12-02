import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from html.parser import HTMLParser
import re
from sqlalchemy import select, or_

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from config.prompts import get_prompt
from core.llm_config import LLMConfigError, get_llm_for_task
from brain_service.models.db import Profile, Work, Author, WorkAuthor
from brain_service.services.sefaria_service import SefariaService
from brain_service.services.wiki_service import WikiService

logger = logging.getLogger(__name__)


ALLOWED_TAGS = {"p", "h2", "h3", "ul", "li", "blockquote", "img", "small", "a"}
ALLOWED_ATTRS = {
    "a": {"href", "title"},
    "img": {"src", "alt"},
}


class _AllowlistSanitizer(HTMLParser):
    def __init__(self):
        super().__init__()
        self.output: list[str] = []
        self.tag_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in ALLOWED_TAGS:
            return
        clean_attrs = []
        allowed = ALLOWED_ATTRS.get(tag, set())
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


class ProfileService:
    """
    Fetches author/work profiles using Sefaria index data + Wikipedia,
    enriches them via an LLM action, and persists them in Postgres with a Redis cache.
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        session_factory: async_sessionmaker,
        sefaria_service: SefariaService,
        wiki_service: WikiService,
        *,
        cache_ttl_sec: int = 3600,
    ) -> None:
        self._redis = redis_client
        self._session_factory = session_factory
        self._sefaria = sefaria_service
        self._wiki = wiki_service
        self._cache_ttl = cache_ttl_sec

    def _cache_key(self, slug: str) -> str:
        return f"profile:v1:{slug}"

    async def _get_cached(self, slug: str) -> Optional[Dict[str, Any]]:
        if not self._redis:
            return None
        try:
            cached = await self._redis.get(self._cache_key(slug))
            if cached:
                return json.loads(cached)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("PROFILE cache read failed", extra={"slug": slug, "error": str(exc)})
        return None

    async def _set_cached(self, slug: str, payload: Dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            await self._redis.set(self._cache_key(slug), json.dumps(payload, ensure_ascii=False), ex=self._cache_ttl)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("PROFILE cache write failed", extra={"slug": slug, "error": str(exc)})

    @staticmethod
    def _profile_to_payload(profile: Profile) -> Dict[str, Any]:
        summary = profile.manual_summary_html or profile.summary_html
        facts = profile.manual_facts or profile.facts or {}
        source = "manual" if profile.manual_summary_html or profile.manual_facts else "generated"

        return {
            "ok": True,
            "slug": profile.slug,
            "title_en": profile.title_en,
            "title_he": profile.title_he,
            "summary_html": summary,
            "summary_work_html": facts.get("summary_work_html") if isinstance(facts, dict) else None,
            "summary_author_html": facts.get("summary_author_html") if isinstance(facts, dict) else None,
            "facts": facts,
            "authors": profile.authors,
            "lifespan": profile.lifespan,
            "period": profile.period,
            "comp_place": profile.comp_place,
            "pub_place": profile.pub_place,
            "json_raw": profile.json_raw,
            "is_verified": profile.is_verified,
            "verified_by": profile.verified_by,
            "verified_at": profile.verified_at.isoformat() if profile.verified_at else None,
            "source": source,
        }

    async def _get_db_profile(self, slug: str) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(select(Profile).where(Profile.slug == slug))
            obj = result.first()
            if obj:
                return self._profile_to_payload(obj)
        return None

    async def _save_db_profile(
        self,
        slug: str,
        payload: Dict[str, Any],
        raw_bundle: Dict[str, Any],
        *,
        clear_manual: bool = False,
    ) -> None:
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
                existing.authors = payload.get("authors")
                existing.lifespan = payload.get("lifespan")
                existing.period = payload.get("period")
                existing.comp_place = payload.get("comp_place")
                existing.pub_place = payload.get("pub_place")
                existing.json_raw = raw_bundle
                if clear_manual:
                    existing.manual_summary_html = None
                    existing.manual_facts = None
                    existing.is_verified = False
                    existing.verified_by = None
                    existing.verified_at = None

                await session.flush()

    def _detect_lang_priority(self, query: str) -> list[str]:
        has_hebrew = any("\u0590" <= ch <= "\u05ff" for ch in query)
        return ["he", "en", "ru"] if has_hebrew else ["en", "he", "ru"]

    def _normalize_index_slug(self, slug: str) -> str:
        """Strip verse/section numbers and trailing qualifiers from a ref-like slug to get an index title."""
        base = (slug or "").strip()
        # drop everything after a comma (e.g. ", Genesis")
        base = base.split(",")[0].strip()
        # remove trailing :num or space+num chains, e.g. "Rashi on Genesis 31:30:1" -> "Rashi on Genesis"
        base = re.sub(r"\s+\d+(?::\d+)*$", "", base)
        return base

    def _fallback_index_candidates(self, slug: str) -> list[str]:
        """Generate alternative index titles when the direct lookup fails."""
        base = self._normalize_index_slug(slug)
        if " on " in base:
            commentator = base.split(" on ", 1)[0].strip()
            work = base.split(" on ", 1)[1].strip()
        else:
            commentator = base
            work = ""

        candidates = []
        # Original base
        candidates.append(base)
        # Common Torah base
        if commentator and "Torah" not in base and "Genesis" not in base:
            candidates.append(f"{commentator} on Torah")
            candidates.append(f"{commentator} on Genesis")
        # If work present but looks like Tanakh book, try stripping numbers
        if work:
            candidates.append(commentator)
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    async def _fetch_author_wiki(self, author_slug: str) -> Optional[Dict[str, Any]]:
        q = author_slug.replace("-", " ").replace("_", " ").strip()
        try:
            wiki_resp = await self._wiki.search_wikipedia(q, lang_priority=self._detect_lang_priority(q))
            if wiki_resp.get("ok"):
                results = (wiki_resp.get("data") or {}).get("results") or []
                return self._pick_best_wiki(q, results)
        except Exception as exc:
            logger.warning("Failed to fetch author wiki", extra={"author": q, "error": str(exc)})
        return None

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

        # Build wiki query: prefer Sefaria title/author if available, otherwise slug
        wiki_query = slug
        if isinstance(sefaria_index, dict):
            wiki_query = sefaria_index.get("title") or sefaria_index.get("heTitle") or slug

        try:
            wiki_resp = await self._wiki.search_wikipedia(wiki_query, lang_priority=self._detect_lang_priority(wiki_query))
            if wiki_resp.get("ok"):
                results = (wiki_resp.get("data") or {}).get("results") or []
                wiki = self._pick_best_wiki(wiki_query, results)
        except Exception as exc:
            logger.warning("Failed to fetch Wikipedia", extra={"slug": slug, "error": str(exc), "query": wiki_query})

        # Author wiki (from Sefaria authors slug if present)
        authors_list = []
        if isinstance(sefaria_index, dict):
            raw_authors = sefaria_index.get("authors") or []
            if isinstance(raw_authors, list):
                authors_list = [a for a in raw_authors if isinstance(a, str) and a.strip()]
        if not authors_list and wiki and isinstance(wiki, dict):
            name = wiki.get("title")
            if isinstance(name, str) and name.strip():
                authors_list = [name.strip()]

        if authors_list:
            author_wiki = await self._fetch_author_wiki(authors_list[0])

        return {"sefaria_index": sefaria_index, "wiki": wiki, "author_wiki": author_wiki, "authors": authors_list}

    def _pick_best_wiki(self, query: str, results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not results:
            return None

        def norm(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

        qn = norm(query)
        # Exact/contains match first
        for res in results:
            title = res.get("title") or ""
            if qn and qn in norm(title):
                return res
        # fallback to first
        return results[0]

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

        return {
            "work": "".join(work_parts),
            "author": author_summary,
        }

    async def _run_llm(self, bundle: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        system_prompt = get_prompt("actions.profile_inspector_system")
        user_template = get_prompt("actions.profile_inspector_user_template")
        if not system_prompt or not user_template:
            logger.error("Profile inspector prompts are not configured")
            return None

        payload = {
            "work_index": bundle.get("sefaria_index"),
            "wiki": bundle.get("wiki"),
            "author_wiki": bundle.get("author_wiki"),
            "authors": bundle.get("authors"),
            "previous_profile": previous,
        }
        user_prompt = user_template.replace("{payload_json}", json.dumps(payload, ensure_ascii=False, indent=2))

        try:
            llm_client, model, reasoning_params, capabilities = get_llm_for_task("PROFILE")
        except LLMConfigError as exc:
            logger.error("LLM config error for PROFILE", extra={"error": str(exc)})
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

    def _build_payload(
        self,
        slug: str,
        bundle: Dict[str, Any],
        generated: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        idx = bundle.get("sefaria_index") or {}
        wiki = bundle.get("wiki") or {}

        fallback = self._fallback_summary(bundle)
        summary_work_html = (generated or {}).get("summary_work_html") or fallback.get("work") or ""
        summary_author_html = (generated or {}).get("summary_author_html") or fallback.get("author") or ""

        facts_work_raw = (generated or {}).get("facts_work") or {}
        facts_author_raw = (generated or {}).get("facts_author") or {}
        facts_work: Dict[str, Any] = facts_work_raw if isinstance(facts_work_raw, dict) else {}
        facts_author: Dict[str, Any] = facts_author_raw if isinstance(facts_author_raw, dict) else {}

        # Ensure required keys exist
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
        ]:
            facts_work.setdefault(key, default)
        for key, default in [
            ("title_en", None),
            ("title_he", None),
            ("lifespan", None),
            ("period", None),
            ("links", {}),
            ("images", []),
        ]:
            facts_author.setdefault(key, default)

        now_iso = datetime.utcnow().isoformat() + "Z"
        if "generated_at" not in facts_work:
            facts_work["generated_at"] = now_iso
        if "generated_at" not in facts_author:
            facts_author["generated_at"] = now_iso

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

        categories = facts_work.get("categories")
        if not categories and isinstance(idx, dict):
            categories = idx.get("categories")

        facts_work["title_en"] = facts_work.get("title_en") or title_en
        facts_work["title_he"] = facts_work.get("title_he") or title_he
        if authors:
            facts_work["authors"] = authors
        if categories:
            facts_work["categories"] = categories
        if links_work:
            facts_work["links"] = links_work

        if facts_author is not None:
            if "title_en" not in facts_author and isinstance(bundle.get("author_wiki"), dict):
                title_candidate = bundle.get("author_wiki", {}).get("title")
                if title_candidate:
                    facts_author["title_en"] = title_candidate
            if "links" not in facts_author and isinstance(bundle.get("author_wiki"), dict):
                url_candidate = bundle.get("author_wiki", {}).get("url")
                if url_candidate:
                    facts_author["links"] = {"wikipedia": url_candidate}

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
        }

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
                # Upsert work
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

                # Upsert authors and link
                for author_slug in authors_list or []:
                    a = await session.get(Author, author_slug)
                    if not a:
                        a = Author(slug=author_slug)
                        session.add(a)
                    a.name_en = facts_author.get("title_en") or author_slug.replace("-", " ")
                    a.summary_html = summary_author_html or a.summary_html
                    if facts_author:
                        a.lifespan = facts_author.get("lifespan") or a.lifespan
                        a.period = facts_author.get("period") or a.period
                        if facts_author.get("links"):
                            a.links = facts_author.get("links")

                    # Link table
                    link_exists = await session.scalars(
                        select(WorkAuthor).where(
                            WorkAuthor.work_id == slug,
                            WorkAuthor.author_id == author_slug
                        )
                    )
                    if not link_exists.first():
                        session.add(WorkAuthor(work_id=slug, author_id=author_slug))

                await session.flush()

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
        if not bundle.get("sefaria_index") and not bundle.get("wiki"):
            return {"ok": False, "error": "Profile sources not found"}

        generated = await self._run_llm(bundle, stored)
        payload = self._build_payload(slug, bundle, generated)

        # Persist structured work/author for reuse
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
        """
        Force-refresh profile from sources, clearing manual overrides.
        """
        # bypass cache by deleting it
        if self._redis:
            try:
                await self._redis.delete(self._cache_key(slug))
            except Exception:
                pass
        bundle = await self._fetch_sources(slug)
        if not bundle.get("sefaria_index") and not bundle.get("wiki"):
            return {"ok": False, "error": "Profile sources not found"}
        generated = await self._run_llm(bundle, None)
        payload = self._build_payload(slug, bundle, generated)
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

    async def save_manual_profile(self, slug: str, summary_html: str | None, facts: dict | None, verified_by: str | None) -> Dict[str, Any]:
        """
        Save manual edits and mark as verified.
        """
        sanitized_html = sanitize_html(summary_html or "") if summary_html else None
        async with self._session_factory() as session:
            async with session.begin():
                profile = await session.get(Profile, slug)
                if not profile:
                    profile = Profile(slug=slug, title_en=slug)
                    session.add(profile)

                if sanitized_html:
                    profile.manual_summary_html = sanitized_html
                elif sanitized_html == "":
                    profile.manual_summary_html = ""
                if facts is not None:
                    profile.manual_facts = facts

                profile.is_verified = True
                profile.verified_by = verified_by
                profile.verified_at = datetime.utcnow()

                await session.flush()
                await session.refresh(profile)

                payload = self._profile_to_payload(profile)
                await self._set_cached(slug, payload)
                return payload

    async def list_profiles(self, *, query: str | None = None, only_unverified: bool = False, limit: int = 100) -> Dict[str, Any]:
        stmt = select(Profile).order_by(Profile.updated_at.desc()).limit(max(1, min(limit, 500)))

        if query:
            like = f"%{query.lower()}%"
            stmt = stmt.where(
                or_(
                    Profile.slug.ilike(like),
                    Profile.title_en.ilike(like),
                    Profile.title_he.ilike(like),
                )
            )
        if only_unverified:
            stmt = stmt.where(Profile.is_verified.is_(False))

        async with self._session_factory() as session:
            result = await session.scalars(stmt)
            items = []
            for profile in result:
                items.append({
                    "slug": profile.slug,
                    "title_en": profile.title_en,
                    "title_he": profile.title_he,
                    "is_verified": profile.is_verified,
                    "verified_by": profile.verified_by,
                    "verified_at": profile.verified_at.isoformat() if profile.verified_at else None,
                    "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
                    "source": "manual" if profile.manual_summary_html or profile.manual_facts else "generated",
                })
            return {"ok": True, "items": items}
