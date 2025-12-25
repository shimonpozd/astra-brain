from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import quote

import httpx
import redis.asyncio as redis
from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from brain_service.core.database import session_scope
from brain_service.models.yiddish import YiddishWordCard
from config.prompts import get_prompt
from core.llm_config import get_llm_for_task, LLMConfigError

if TYPE_CHECKING:
    from domain.chat.tools import ToolRegistry

logger = logging.getLogger(__name__)

try:
    import yiddish as yiddish_lib
except Exception as exc:  # pragma: no cover - optional runtime guard
    yiddish_lib = None
    logger.warning("Yiddish library unavailable; falling back to basic normalization", extra={"error": str(exc)})


class WiktionaryYiddishService:
    """
    Fetches and caches Yiddish WordCards from EN Wiktionary.
    Pipeline:
      - normalize input
      - fetch wikitext (cached)
      - parse Yiddish section (POS, glosses, examples, forms, etymology)
      - call LLM to translate/compress into RU WordCard JSON
      - persist to Postgres (JSONB) for fast popup use
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        http_client: httpx.AsyncClient,
        redis_client: Optional[redis.Redis] = None,
        *,
        wiktionary_cache_days: int = 7,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        self.session_factory = session_factory
        self.http_client = http_client
        self.redis_client = redis_client
        self.wiktionary_cache_ttl = int(timedelta(days=wiktionary_cache_days).total_seconds())
        self.user_agent = "AstraYiddishWordcard/0.1 (+https://astra.local; contact: support@astra)"
        self.wordcard_version = 1
        self.tool_registry = tool_registry

    # -----------------------------
    # Public API
    # -----------------------------
    async def get_wordcard(
        self,
        word_surface: str,
        lemma_guess: Optional[str] = None,
        pos_guess: Optional[str] = None,
        context_sentence: Optional[str] = None,
        ui_lang: str = "ru",
        include_evidence: bool = False,
        include_llm_output: bool = False,
        force_refresh: bool = False,
        allow_llm_fallback: bool = False,
        redirect_depth: int = 0,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entry: return WordCard JSON, loading from DB cache or generating from Wiktionary.
        """
        normalized_surface = self._normalize_surface(word_surface)
        base_lemma = (lemma_guess or normalized_surface or word_surface or "").strip()
        title_candidates = self._candidate_titles(word_surface, lemma_guess, normalized_surface)
        if not force_refresh:
            cached_bundle = await self._load_cached_card_any(candidates=title_candidates, ui_lang=ui_lang)
            if cached_bundle:
                cached = cached_bundle.get("data")
                evidence = cached_bundle.get("evidence") or {}
                if cached and not cached.get("morphology"):
                    cached = self._finalize_agent_wordcard(
                        cached,
                        word_surface=word_surface,
                        lemma_fallback=base_lemma,
                        ui_lang=ui_lang,
                        evidence=evidence,
                    )
                    if persist:
                        await self._persist_wordcard(
                            lemma_key=cached.get("lemma") or base_lemma,
                            ui_lang=ui_lang,
                            pos_default=cached.get("pos_default"),
                            evidence=evidence,
                            wordcard=cached,
                        )
                if cached:
                    return cached
        if self.tool_registry:
            wordcard, last_evidence = await self._run_wordcard_agent(
                word_surface=word_surface,
                lemma_guess=lemma_guess,
                pos_guess=pos_guess,
                context_sentence=context_sentence,
                ui_lang=ui_lang,
            )
            last_evidence = last_evidence or {}
            if allow_llm_fallback and not last_evidence.get("pos_entries"):
                wordcard.setdefault("flags", {})
                wordcard["flags"]["needs_review"] = True
                wordcard["flags"]["evidence_missing"] = True
                wordcard["flags"]["auto_generated"] = True
            wordcard = self._finalize_agent_wordcard(
                wordcard,
                word_surface=word_surface,
                lemma_fallback=base_lemma,
                ui_lang=ui_lang,
                evidence=last_evidence,
            )
            if persist:
                await self._persist_wordcard(
                    lemma_key=base_lemma,
                    ui_lang=ui_lang,
                    pos_default=wordcard.get("pos_default"),
                    evidence=last_evidence or {},
                    wordcard=wordcard,
                )
            if include_evidence or include_llm_output:
                debug_payload: Dict[str, Any] = {}
                if include_evidence:
                    debug_payload["evidence"] = last_evidence or {}
                if include_llm_output:
                    debug_payload["llm_output"] = wordcard
                wordcard = {**wordcard, "_debug": debug_payload}
            return wordcard
        raise HTTPException(status_code=500, detail="Tool registry not configured")

    async def get_wordcard_for_tool(
        self,
        word: str,
        *,
        lemma_guess: Optional[str] = None,
        pos_guess: Optional[str] = None,
        ui_lang: str = "ru",
        allow_llm_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Lightweight tool output for LLM calls (smaller payload, no debug).
        """
        try:
            card = await self.get_wordcard(
                word_surface=word,
                lemma_guess=lemma_guess,
                pos_guess=pos_guess,
                ui_lang=ui_lang,
                include_evidence=False,
                include_llm_output=False,
                force_refresh=False,
                allow_llm_fallback=allow_llm_fallback,
            )
        except HTTPException as exc:
            return {"ok": False, "error": exc.detail, "status_code": exc.status_code}

        # Trim senses to reduce tokens
        senses = card.get("senses") or []
        slim_senses = []
        for sense in senses[:3]:
            slim_senses.append(
                {
                    "sense_id": sense.get("sense_id"),
                    "gloss_ru_short": sense.get("gloss_ru_short"),
                    "source_gloss_en": sense.get("source_gloss_en"),
                }
            )

        return {
            "ok": True,
            "data": {
                "schema": card.get("schema"),
                "lang": card.get("lang"),
                "ui_lang": card.get("ui_lang"),
                "word_surface": card.get("word_surface"),
                "lemma": card.get("lemma"),
                "translit_ru": card.get("translit_ru"),
                "pos_default": card.get("pos_default"),
                "pos_ru_short": card.get("pos_ru_short"),
                "pos_ru_full": card.get("pos_ru_full"),
                "popup": card.get("popup"),
                "senses": slim_senses,
                "flags": card.get("flags"),
                "sources": card.get("sources"),
                "version": card.get("version"),
            },
        }

    async def get_evidence_for_tool(
        self,
        word: str,
        *,
        lemma_guess: Optional[str] = None,
        pos_guess: Optional[str] = None,
        context_sentence: Optional[str] = None,
        ui_lang: str = "ru",
    ) -> Dict[str, Any]:
        """
        Fetch and return compact Wiktionary evidence (no LLM) for tool usage.
        """
        normalized_surface = self._normalize_surface(word)
        title_candidates = self._candidate_titles(word, lemma_guess, normalized_surface)

        wikitext = None
        title_found = None
        search_title = None
        for title in title_candidates:
            wikitext = await self._fetch_wikitext(title)
            if wikitext:
                title_found = title
                break
        if not wikitext:
            search_title = await self._search_wiktionary_title(title_candidates)
            if search_title:
                wikitext = await self._fetch_wikitext(search_title)
                if wikitext:
                    title_found = search_title

        lemma = (title_found or lemma_guess or normalized_surface or word or "").strip()
        if wikitext:
            parsed = self._parse_yiddish_section(wikitext)
        else:
            parsed = {"pos_entries": [], "etymologies": [], "forms": {}, "has_yiddish": False, "form_of_target": None}

        evidence = self._build_evidence_payload(
            word_surface=word,
            lemma=lemma,
            pos_guess=pos_guess,
            context_sentence=context_sentence,
            parsed=parsed,
            allow_guess=False,
        )
        evidence["title_found"] = title_found
        evidence["search_title"] = search_title
        evidence["suggested_variants"] = title_candidates[:10]
        evidence["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        evidence["ui_lang"] = ui_lang

        return {
            "ok": True,
            "found": bool(wikitext),
            "data": evidence,
        }

    # -----------------------------
    # Fetch & cache
    # -----------------------------
    def _normalize_surface(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = cleaned.strip(".,;:!?’“”\"'()[]{}").replace("״", '"').replace("׳", "'")
        return cleaned

    def _candidate_titles(self, word_surface: str, lemma_guess: Optional[str], normalized_surface: str) -> List[str]:
        candidates: List[str] = []
        for value in (normalized_surface, lemma_guess, word_surface):
            if not value:
                continue
            candidates.append(value)
            candidates.append(unicodedata.normalize("NFC", value))
            candidates.append(unicodedata.normalize("NFKC", value))
            if yiddish_lib:
                try:
                    candidates.append(yiddish_lib.replace_with_precombined(value))
                except Exception:
                    pass
                try:
                    candidates.append(yiddish_lib.strip_diacritics(value))
                except Exception:
                    pass
            stripped = self._strip_diacritics(value)
            if stripped:
                candidates.append(stripped)
            no_apost = self._strip_apostrophes(value)
            if no_apost and no_apost != value:
                candidates.append(no_apost)
            clitic = self._split_clitic(value)
            if clitic:
                candidates.append(clitic)
                candidates.append(self._strip_diacritics(clitic))
            for stripped in self._strip_common_prefixes(value):
                candidates.append(stripped)
                candidates.append(self._strip_diacritics(stripped))
        # If word ends with ayin/he, try stripping it and applying final-form normalization.
        for value in list(candidates):
            if value and value.endswith(("\u05e2", "\u05d4")) and len(value) > 1:
                base = value[:-1]
                candidates.append(self._apply_final_form(base))
        # Deduplicate while preserving order
        seen = set()
        unique: List[str] = []
        for item in candidates:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _strip_apostrophes(self, text: str) -> str:
        if not text:
            return text
        return re.sub(r"['’ʼʹ′＇\u05f3]", "", text)

    def _strip_common_prefixes(self, text: str) -> List[str]:
        if not text:
            return []
        prefixes = [
            "צו",
            "אויס",
            "אויף",
            "אופ",
            "איבער",
            "אונטער",
            "אריין",
            "ארויס",
            "צוריק",
            "ציריק",
            "דורכ",
            "פאר",
            "באַ",
            "בא",
            "גע",
            "אפ",
            "אָפּ",
        ]
        results: List[str] = []
        for prefix in prefixes:
            if text.startswith(prefix) and len(text) > len(prefix) + 1:
                results.append(text[len(prefix):])
        return results

    def _analyze_morphology(self, word: str) -> Optional[Dict[str, Any]]:
        if not word:
            return None
        base = self._strip_diacritics(self._strip_apostrophes(word))
        if len(base) < 3:
            return None

        prefix_rules = [
            {"prefix": "צו", "meaning_ru": "направление, присоединение"},
            {"prefix": "אויס", "meaning_ru": "вне, наружу"},
            {"prefix": "אופ", "meaning_ru": "вверх, на"},
            {"prefix": "אויף", "meaning_ru": "вверх, на"},
            {"prefix": "איבער", "meaning_ru": "через, над"},
            {"prefix": "אונטער", "meaning_ru": "под, вниз"},
            {"prefix": "אריין", "meaning_ru": "внутрь"},
            {"prefix": "ארויס", "meaning_ru": "наружу"},
            {"prefix": "צוריק", "meaning_ru": "назад"},
            {"prefix": "ציריק", "meaning_ru": "назад"},
            {"prefix": "דורכ", "meaning_ru": "через, насквозь"},
            {"prefix": "פאר", "meaning_ru": "вперед, для"},
            {"prefix": "באַ", "meaning_ru": "усиление/интенсивность"},
            {"prefix": "בא", "meaning_ru": "усиление/интенсивность"},
            {"prefix": "גע", "meaning_ru": "причастие/перфект"},
            {"prefix": "אפ", "meaning_ru": "снятие/вниз"},
            {"prefix": "אָפּ", "meaning_ru": "снятие/вниз"},
            {"prefix": "צוזאמען", "meaning_ru": "вместе, соединение"},
        ]

        suffix_rules = [
            {"suffix": "ן", "meaning_ru": "инфинитив/глагольная форма"},
            {"suffix": "ט", "meaning_ru": "форма прош.вр./причастие"},
            {"suffix": "לעך", "meaning_ru": "уменьшительный суффикс"},
            {"suffix": "יק", "meaning_ru": "прилагательное"},
            {"suffix": "ער", "meaning_ru": "лицо/существительное"},
            {"suffix": "קייט", "meaning_ru": "абстрактное качество"},
            {"suffix": "ונג", "meaning_ru": "действие/процесс"},
        ]

        remaining = base
        prefixes: List[Dict[str, str]] = []
        for _ in range(2):
            match = None
            for rule in sorted(prefix_rules, key=lambda r: len(r["prefix"]), reverse=True):
                if remaining.startswith(rule["prefix"]) and len(remaining) > len(rule["prefix"]) + 1:
                    match = rule
                    break
            if not match:
                break
            prefixes.append({"form": match["prefix"], "meaning_ru": match["meaning_ru"]})
            remaining = remaining[len(match["prefix"]):]

        suffixes: List[Dict[str, str]] = []
        for rule in sorted(suffix_rules, key=lambda r: len(r["suffix"]), reverse=True):
            if remaining.endswith(rule["suffix"]) and len(remaining) > len(rule["suffix"]) + 1:
                suffixes.append({"form": rule["suffix"], "meaning_ru": rule["meaning_ru"]})
                remaining = remaining[:-len(rule["suffix"])]
                if len(suffixes) >= 2:
                    break

        base_lemma = remaining or base
        if not prefixes and not suffixes:
            return None

        summary_parts = []
        if prefixes:
            summary_parts.append(
                "Приставки: " + ", ".join([f"{p['form']} ({p['meaning_ru']})" for p in prefixes])
            )
        if suffixes:
            summary_parts.append(
                "Суффиксы: " + ", ".join([f"{s['form']} ({s['meaning_ru']})" for s in suffixes])
            )
        summary_parts.append(f"Основа: {base_lemma}")

        return {
            "prefixes": prefixes,
            "suffixes": suffixes,
            "base_lemma": base_lemma,
            "summary_ru": ". ".join(summary_parts),
        }

    def _split_clitic(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.match(r"^([מ ס ד נ כ ב ל])['’ʼʹ′＇\u05f3](.+)$", text)
        if not match:
            return None
        remainder = match.group(2).strip()
        if len(remainder) < 2:
            return None
        return remainder

    def _apply_final_form(self, word: str) -> str:
        final_map = {
            "\u05db": "\u05da",  # kaf -> final kaf
            "\u05de": "\u05dd",  # mem -> final mem
            "\u05e0": "\u05df",  # nun -> final nun
            "\u05e4": "\u05e3",  # pe -> final pe
            "\u05e6": "\u05e5",  # tsadi -> final tsadi
        }
        if not word:
            return word
        last = word[-1]
        return word[:-1] + final_map.get(last, last)

    def _strip_diacritics(self, text: str) -> str:
        if not text:
            return text
        # Hebrew combining marks (vowels, cantillation)
        return re.sub(r"[\u0591-\u05C7]", "", text)

    def _wiktionary_cache_key(self, title: str) -> str:
        return f"wiktionary:yi:wikitext:{title.lower()}"

    async def _fetch_wikitext(self, title: str) -> Optional[str]:
        """
        Fetch raw wikitext for a page title from en.wiktionary, with Redis cache.
        """
        if not title:
            return None
        cache_key = self._wiktionary_cache_key(title)
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    logger.info("Wiktionary wikitext cache HIT", extra={"title": title})
                    return cached
            except Exception as exc:
                logger.warning("Redis read failed for Wiktionary cache", extra={"error": str(exc)})

        url = (
            "https://en.wiktionary.org/w/api.php?"
            f"action=query&prop=revisions&rvslots=*&rvprop=content&formatversion=2&format=json&redirects=1&titles={quote(title)}"
        )
        headers = {"User-Agent": self.user_agent}
        try:
            resp = await self.http_client.get(url, headers=headers, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                return None
            rev = (pages[0].get("revisions") or [{}])[0]
            content = rev.get("slots", {}).get("main", {}).get("content")
            if not content:
                content = rev.get("content")
            if not content:
                rest_url = f"https://en.wiktionary.org/api/rest_v1/page/source/{quote(title)}"
                rest_resp = await self.http_client.get(rest_url, headers=headers, timeout=15.0)
                if rest_resp.status_code == 200:
                    content = rest_resp.text
            if content and self.redis_client:
                try:
                    await self.redis_client.setex(cache_key, self.wiktionary_cache_ttl, content)
                    logger.info("Wiktionary wikitext cache WRITE", extra={"title": title})
                except Exception as exc:
                    logger.warning("Redis write failed for Wiktionary cache", extra={"error": str(exc)})
            return content
        except Exception as exc:
            logger.error("Wiktionary fetch failed", extra={"title": title, "error": str(exc)})
            return None

    async def _search_wiktionary_title(self, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        headers = {"User-Agent": self.user_agent}
        for candidate in candidates[:8]:
            if not candidate:
                continue
            for query in (f'intitle:"{candidate}"', f'{candidate} insource:"==Yiddish=="', candidate):
                url = (
                    "https://en.wiktionary.org/w/api.php?"
                    f"action=query&list=search&format=json&srsearch={quote(query)}&srlimit=5"
                )
                try:
                    resp = await self.http_client.get(url, headers=headers, timeout=10.0)
                    resp.raise_for_status()
                    data = resp.json()
                    results = data.get("query", {}).get("search", [])
                    for hit in results:
                        title = hit.get("title")
                        if not title:
                            continue
                        wikitext = await self._fetch_wikitext(title)
                        if not wikitext:
                            continue
                        parsed = self._parse_yiddish_section(wikitext)
                        if parsed.get("has_yiddish") and (
                            parsed.get("pos_entries") or parsed.get("form_of_target")
                        ):
                            return title
                except Exception as exc:
                    logger.warning("Wiktionary search failed", extra={"query": query, "error": str(exc)})
        return None

    # -----------------------------
    # Parsing helpers
    # -----------------------------
    def _parse_yiddish_section(self, wikitext: str) -> Dict[str, Any]:
        """
        Extract the ==Yiddish== section and parse POS entries, glosses, examples, forms, etymology.
        Returns dict with keys: raw_section, pos_entries, etymologies, head_templates
        """
        if not wikitext:
            return {"raw_section": "", "pos_entries": [], "etymologies": []}

        # Locate Yiddish section
        yid_match = re.search(r"^==\s*Yiddish\s*==", wikitext, flags=re.MULTILINE | re.IGNORECASE)
        if not yid_match:
            return {"raw_section": "", "pos_entries": [], "etymologies": []}
        start = yid_match.end()
        next_section = re.search(r"^==[^=].*==\s*$", wikitext[start:], flags=re.MULTILINE)
        end = start + next_section.start() if next_section else len(wikitext)
        section = wikitext[start:end]

        lines = section.splitlines()
        pos_entries: List[Dict[str, Any]] = []
        etymologies: List[Dict[str, Any]] = []
        current_pos: Optional[str] = None
        current_glosses: List[Dict[str, Any]] = []
        current_examples: Dict[int, List[str]] = {}
        head_templates: List[str] = []
        current_etymology: List[str] = []
        form_of_target: Optional[str] = None

        pos_headings = {
            "noun": "NOUN",
            "verb": "VERB",
            "adjective": "ADJ",
            "adverb": "ADV",
            "preposition": "PREP",
            "conjunction": "CONJ",
            "particle": "PART",
            "pronoun": "PRON",
            "determiner": "DET",
        }

        def flush_pos():
            nonlocal current_pos, current_glosses, current_examples
            if current_pos:
                # Attach examples to glosses by index
                for idx, gloss in enumerate(current_glosses):
                    if idx in current_examples:
                        gloss["examples"] = current_examples[idx]
                pos_entries.append(
                    {
                        "pos": current_pos,
                        "gloss_en_list": [g["text"] for g in current_glosses if g.get("text")],
                        "gloss_items": current_glosses,
                    }
                )
            current_pos = None
            current_glosses = []
            current_examples = {}

        def flush_etymology():
            nonlocal current_etymology
            if current_etymology:
                text = " ".join(current_etymology).strip()
                if self._has_letters(text):
                    etymologies.append({"text": text, "path": self._parse_etymology_path(text)})
            current_etymology = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            heading_match = re.match(r"^===\s*(.+?)\s*===\s*$", line)
            if heading_match:
                heading_text = heading_match.group(1).strip()
                heading_lower = heading_text.lower()
                if heading_lower.startswith("etymology"):
                    flush_pos()
                    flush_etymology()
                    current_etymology = []
                    continue
                matched_pos = False
                for k, v in pos_headings.items():
                    if heading_lower.startswith(k):
                        flush_pos()
                        current_pos = v
                        matched_pos = True
                        break
                if not matched_pos:
                    # Unknown subsection; stop collecting POS glosses until a new POS is found.
                    flush_pos()
                    current_pos = None
                continue

            if line.startswith("{{") and line.endswith("}}"):
                head_templates.append(line)

            if line.startswith("#*"):
                if not current_pos:
                    continue
                ex_text = self._clean_wikicode(line[2:].strip())
                if not self._has_letters(ex_text):
                    continue
                idx = len(current_glosses) - 1
                current_examples.setdefault(idx, []).append(ex_text)
                continue

            if line.startswith("#"):
                target = self._extract_form_of_target(raw_line)
                if target and not form_of_target:
                    form_of_target = target
                if not current_pos:
                    continue
                gloss_text = self._clean_wikicode(line[1:].strip())
                if not self._has_letters(gloss_text):
                    continue
                current_glosses.append({"text": gloss_text})
                continue

            if current_etymology is not None and (line.startswith("{{inh") or "borrow" in line or "From" in line):
                current_etymology.append(self._clean_wikicode(line))
            elif current_etymology is not None and heading_match is None and line:
                # inside etymology block, accumulate free text
                current_etymology.append(self._clean_wikicode(line))

        flush_pos()
        flush_etymology()

        return {
            "raw_section": section,
            "pos_entries": pos_entries,
            "etymologies": etymologies,
            "head_templates": head_templates,
            "forms": self._parse_headword_forms(head_templates),
            "has_yiddish": True,
            "form_of_target": form_of_target,
        }

    def _parse_headword_forms(self, templates: List[str]) -> Dict[str, Any]:
        """
        Extract simple grammar hints (gender/plural/infinitive) from headword templates.
        """
        forms: Dict[str, Any] = {"noun": {}, "verb": {}, "adj": {}}
        for tpl in templates:
            inner = tpl.strip("{}")
            parts = inner.split("|")
            if not parts:
                continue
            name = parts[0].lower()
            kv = {}
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    kv[k.strip()] = v.strip()
            if name.startswith("yi-noun") or name == "head" and kv.get("1") == "yi" and kv.get("2") == "noun":
                gender = kv.get("g") or kv.get("gender")
                plural = kv.get("pl") or kv.get("plural")
                if gender:
                    forms["noun"]["gender"] = gender
                if plural:
                    forms["noun"]["plural"] = plural
            if name.startswith("yi-verb") or name == "head" and kv.get("2") == "verb":
                infinitive = kv.get("inf") or kv.get("infinitive")
                if infinitive:
                    forms["verb"]["infinitive"] = infinitive
            if name.startswith("yi-adj") or name == "head" and kv.get("2") == "adjective":
                forms["adj"]["head"] = kv.get("head") or kv.get("headword")
        return forms

    def _parse_etymology_path(self, text: str) -> List[Dict[str, str]]:
        """
        Very light parser for {{inh|yi|...}} / {{der|yi|...}} templates to build path steps.
        """
        path: List[Dict[str, str]] = []
        for tpl in re.findall(r"\{\{([^{}]+)\}\}", text):
            parts = tpl.split("|")
            if not parts:
                continue
            kind = parts[0]
            if kind in {"inh", "der", "bor"} and len(parts) >= 3:
                lang = parts[1]
                term = parts[2]
                path.append({"lang": lang, "term": term})
        return path

    def _clean_wikicode(self, text: str) -> str:
        """
        Strip common wikicode markers to plain text for LLM evidence.
        """
        if not text:
            return ""
        cleaned = re.sub(r"\{\{[^{}]+\}\}", "", text)
        cleaned = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", cleaned)
        cleaned = cleaned.replace("''", "")
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        return " ".join(cleaned.split())

    def _extract_form_of_target(self, raw_line: str) -> Optional[str]:
        """
        Detect templates like {{alternative spelling of|yi|TARGET}} or {{inflection of|yi|TARGET}}.
        """
        for tpl in re.findall(r"\{\{([^{}]+)\}\}", raw_line):
            parts = [p.strip() for p in tpl.split("|") if p.strip()]
            if not parts:
                continue
            name = parts[0].lower()
            if name in {
                "alternative spelling of",
                "alternative form of",
                "alt spelling of",
                "alt form of",
                "form of",
                "inflection of",
            }:
                if len(parts) >= 3 and parts[1] in {"yi", "ydd"}:
                    return parts[2]
                if len(parts) >= 2:
                    return parts[1]
        return None

    def _has_letters(self, text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text or ""))

    # -----------------------------
    # Evidence + LLM
    # -----------------------------
    def _build_evidence_payload(
        self,
        *,
        word_surface: str,
        lemma: str,
        pos_guess: Optional[str],
        context_sentence: Optional[str],
        parsed: Dict[str, Any],
        allow_guess: bool = False,
    ) -> Dict[str, Any]:
        return {
            "lemma": lemma,
            "word_surface": word_surface,
            "pos_guess": pos_guess,
            "context_sentence": context_sentence,
            "pos_entries": parsed.get("pos_entries", []),
            "etymologies": parsed.get("etymologies", []),
            "forms": parsed.get("forms", {}),
            "form_of_target": parsed.get("form_of_target"),
            "allow_guess": allow_guess,
            "has_yiddish_section": bool(parsed.get("has_yiddish")),
        }

    def _get_tool_schemas(self, names: set[str]) -> List[Dict[str, Any]]:
        if not self.tool_registry:
            return []
        schemas = []
        for schema in self.tool_registry.get_tool_schemas():
            func = schema.get("function") or {}
            if func.get("name") in names:
                schemas.append(schema)
        return schemas

    def _parse_tool_args(self, raw_args: str) -> Dict[str, Any]:
        if not raw_args:
            return {}
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(raw_args)
            return parsed

    async def _run_wordcard_agent(
        self,
        *,
        word_surface: str,
        lemma_guess: Optional[str],
        pos_guess: Optional[str],
        context_sentence: Optional[str],
        ui_lang: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        system_prompt = get_prompt("actions.yiddish_wordcard_agent_system")
        user_template = get_prompt("actions.yiddish_wordcard_agent_user")
        if not system_prompt or not user_template:
            raise HTTPException(status_code=500, detail="Yiddish wordcard agent prompts not configured")

        try:
            llm_client, model, reasoning_params, capabilities = get_llm_for_task("YIDDISH")
        except LLMConfigError as exc:
            raise HTTPException(status_code=500, detail=f"LLM not configured: {exc}")

        fmt_args = {
            "word_surface": word_surface,
            "lemma_guess": lemma_guess or "",
            "pos_guess": pos_guess or "",
            "context_sentence": context_sentence or "",
            "ui_lang": ui_lang,
            "evidence_json": json.dumps({}, ensure_ascii=False),
            "lemma": (lemma_guess or word_surface),
            "pos": (pos_guess or ""),
            "index": 1,
        }

        last_evidence: Dict[str, Any] = {}
        try:
            evidence_bundle = await self.get_evidence_for_tool(
                word_surface,
                lemma_guess=lemma_guess,
                pos_guess=pos_guess,
                context_sentence=context_sentence,
                ui_lang=ui_lang,
            )
            if isinstance(evidence_bundle, dict):
                data = evidence_bundle.get("data")
                if isinstance(data, dict):
                    last_evidence = data
                    fmt_args["evidence_json"] = json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Pre-fetch evidence failed", extra={"error": str(exc)})
        try:
            user_msg = user_template.format(**fmt_args)
        except KeyError as exc:
            logger.error(
                "Wordcard user prompt missing placeholder",
                extra={"missing": str(exc)},
            )
            user_msg = f"Evidence (JSON):\n{fmt_args['evidence_json']}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        tools = self._get_tool_schemas({"wiktionary_yiddish_evidence_lookup"})
        if not tools:
            raise HTTPException(status_code=500, detail="Yiddish evidence tool not registered")
        api_params: Dict[str, Any] = {**reasoning_params, "model": model, "messages": messages, "stream": False}
        api_params.update({"tools": tools, "tool_choice": "auto"})
        api_params["response_format"] = {"type": "json_object"}

        for _ in range(5):
            try:
                completion = await llm_client.chat.completions.create(**api_params)
            except Exception as exc:
                logger.error("LLM agent call failed", extra={"error": str(exc)}, exc_info=True)
                raise HTTPException(status_code=500, detail="LLM agent call failed") from exc
            message = completion.choices[0].message if completion and completion.choices else None
            if not message:
                raise HTTPException(status_code=500, detail="LLM returned empty response")

            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                assistant_msg = {"role": "assistant", "content": message.content or "", "tool_calls": []}
                for tc in tool_calls:
                    fn = tc.function
                    assistant_msg["tool_calls"].append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": fn.name,
                                "arguments": fn.arguments,
                            },
                        }
                    )
                messages.append(assistant_msg)

                for tc in tool_calls:
                    fn = tc.function
                    try:
                        args = self._parse_tool_args(fn.arguments)
                    except Exception as exc:
                        raise HTTPException(status_code=500, detail=f"Invalid tool arguments: {exc}")

                    if not self.tool_registry:
                        raise HTTPException(status_code=500, detail="Tool registry not configured")
                    result = await self.tool_registry.call(fn.name, **args)
                    if fn.name == "wiktionary_yiddish_evidence_lookup" and isinstance(result, dict):
                        data = result.get("data")
                        if isinstance(data, dict):
                            last_evidence = data

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": fn.name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                api_params["messages"] = messages
                if last_evidence:
                    final_params: Dict[str, Any] = {
                        **reasoning_params,
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "response_format": {"type": "json_object"},
                    }
                    try:
                        final_completion = await llm_client.chat.completions.create(**final_params)
                    except Exception as exc:
                        logger.error("LLM finalization failed", extra={"error": str(exc)}, exc_info=True)
                        raise HTTPException(status_code=500, detail="LLM finalization failed") from exc
                    final_message = final_completion.choices[0].message if final_completion and final_completion.choices else None
                    final_content = final_message.content if final_message else ""
                    if not final_content:
                        raise HTTPException(status_code=500, detail="LLM returned empty content")
                    try:
                        return json.loads(final_content), last_evidence
                    except json.JSONDecodeError as exc:
                        logger.error("LLM returned non-JSON", extra={"content": final_content[:400]})
                        raise HTTPException(status_code=500, detail="LLM did not return JSON") from exc
                continue

            content = message.content or ""
            if not content:
                raise HTTPException(status_code=500, detail="LLM returned empty content")
            try:
                return json.loads(content), last_evidence
            except json.JSONDecodeError as exc:
                logger.error("Agent returned non-JSON", extra={"content": content[:400]})
                raise HTTPException(status_code=500, detail="LLM did not return JSON") from exc

        logger.warning(
            "LLM tool loop exceeded, returning minimal fallback",
            extra={"word_surface": word_surface, "lemma_guess": lemma_guess},
        )
        fallback = {
            "schema": "astra.yiddish.wordcard.v1",
            "lang": "yi",
            "ui_lang": ui_lang,
            "word_surface": word_surface,
            "lemma": lemma_guess or word_surface,
            "translit_ru": "",
            "pos_default": pos_guess or "",
            "pos_ru_short": "",
            "pos_ru_full": "",
            "popup": {"gloss_ru_short_list": []},
            "senses": [],
            "flags": {"needs_review": True, "evidence_missing": True},
            "sources": [],
            "version": self.wordcard_version,
        }
        return fallback, last_evidence
    def _finalize_agent_wordcard(
        self,
        wordcard: Dict[str, Any],
        *,
        word_surface: str,
        lemma_fallback: str,
        ui_lang: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        evidence = evidence or {}

        pos_map_short = {
            "NOUN": "сущ.",
            "VERB": "глаг.",
            "ADJ": "прил.",
            "ADV": "нареч.",
            "PRON": "мест.",
            "CONJ": "союз",
            "PREP": "предлог",
            "PART": "частица",
            "DET": "опр.",
        }
        pos_map_full = {
            "NOUN": "существительное",
            "VERB": "глагол",
            "ADJ": "прилагательное",
            "ADV": "наречие",
            "PRON": "местоимение",
            "CONJ": "союз",
            "PREP": "предлог",
            "PART": "частица",
            "DET": "определитель",
        }

        wordcard["schema"] = "astra.yiddish.wordcard.v1"
        wordcard.setdefault("lang", "yi")
        wordcard.setdefault("ui_lang", ui_lang)
        if not wordcard.get("word_surface"):
            wordcard["word_surface"] = word_surface

        lemma = wordcard.get("lemma") or lemma_fallback or word_surface
        wordcard["lemma"] = lemma

        pos_entries = evidence.get("pos_entries") or []
        if not wordcard.get("pos_default"):
            if pos_entries and isinstance(pos_entries[0], dict):
                wordcard["pos_default"] = pos_entries[0].get("pos")
        elif pos_entries:
            valid_pos = [entry.get("pos") for entry in pos_entries if isinstance(entry, dict) and entry.get("pos")]
            if valid_pos and wordcard.get("pos_default") not in valid_pos:
                wordcard["pos_default"] = valid_pos[0]
        if wordcard.get("pos_default"):
            short_val = wordcard.get("pos_ru_short") or ""
            full_val = wordcard.get("pos_ru_full") or ""
            if short_val not in pos_map_short.values():
                wordcard["pos_ru_short"] = pos_map_short.get(wordcard["pos_default"], "")
            if full_val not in pos_map_full.values():
                wordcard["pos_ru_full"] = pos_map_full.get(wordcard["pos_default"], "")

        translit_ru = wordcard.get("translit_ru") or ""
        if re.search(r"[\u0590-\u05FF]", translit_ru):
            wordcard["translit_ru"] = ""

        version_val = wordcard.get("version")
        if not isinstance(version_val, int):
            wordcard["version"] = self.wordcard_version

        if "morphology" not in wordcard:
            morph = self._analyze_morphology(wordcard.get("word_surface") or word_surface)
            if morph:
                wordcard["morphology"] = morph

        if not isinstance(wordcard.get("popup"), dict):
            wordcard["popup"] = {"gloss_ru_short_list": []}
        elif "gloss_ru_short_list" not in wordcard["popup"]:
            wordcard["popup"]["gloss_ru_short_list"] = []

        wordcard.setdefault("senses", [])

        senses = wordcard.get("senses") or []
        if isinstance(senses, list) and senses:
            normalized_senses = []
            for idx, sense in enumerate(senses, start=1):
                if not isinstance(sense, dict):
                    continue
                gloss_ru_list = sense.get("gloss_ru_list")
                if isinstance(gloss_ru_list, list) and gloss_ru_list:
                    first_ru = next((g for g in gloss_ru_list if isinstance(g, str) and g.strip()), "")
                    if first_ru and not sense.get("gloss_ru_short"):
                        sense["gloss_ru_short"] = first_ru
                    if not sense.get("gloss_ru_full"):
                        sense["gloss_ru_full"] = "; ".join(
                            [g.strip() for g in gloss_ru_list if isinstance(g, str) and g.strip()]
                        )
                if "gloss_ru" in sense and not sense.get("gloss_ru_short"):
                    sense["gloss_ru_short"] = sense.get("gloss_ru") or ""
                if "gloss_ru" in sense and not sense.get("gloss_ru_full"):
                    sense["gloss_ru_full"] = sense.get("gloss_ru") or sense.get("gloss_ru_short") or ""
                gloss_en_list = sense.get("gloss_en_list")
                if isinstance(gloss_en_list, list) and gloss_en_list and not sense.get("source_gloss_en"):
                    first_en = next((g for g in gloss_en_list if isinstance(g, str) and g.strip()), "")
                    if first_en:
                        sense["source_gloss_en"] = first_en
                sense_id = sense.get("sense_id")
                if not isinstance(sense_id, str) or ":" not in sense_id:
                    pos_val = wordcard.get("pos_default") or "UNKNOWN"
                    sense["sense_id"] = f"{lemma}:{pos_val}:{idx}"
                normalized_senses.append(sense)
            wordcard["senses"] = normalized_senses

        # Fallback: if LLM returned no senses/glosses, try to seed from evidence.
        if not wordcard["senses"] and not wordcard["popup"].get("gloss_ru_short_list"):
            gloss_candidates: List[str] = []
            for val in evidence.get("hebrew_glosses") or []:
                if isinstance(val, str) and val.strip():
                    gloss_candidates.append(val.strip())
            for entry in evidence.get("pos_entries") or []:
                for val in entry.get("gloss_en_list") or []:
                    if isinstance(val, str) and val.strip():
                        gloss_candidates.append(val.strip())
            if gloss_candidates:
                uniq = []
                for g in gloss_candidates:
                    if g not in uniq:
                        uniq.append(g)
                wordcard["popup"]["gloss_ru_short_list"] = uniq[:3]
                pos_val = wordcard.get("pos_default") or ""
                senses = []
                for idx, gloss in enumerate(uniq[:2], start=1):
                    senses.append(
                        {
                            "sense_id": f"{lemma}:{pos_val or 'UNKNOWN'}:{idx}",
                            "gloss_ru_short": gloss,
                            "gloss_ru_full": gloss,
                            "source_gloss_en": gloss,
                            "confidence": 0.2,
                        }
                    )
                wordcard["senses"] = senses

        flags = wordcard.get("flags") if isinstance(wordcard.get("flags"), dict) else {}
        if "needs_review" not in flags:
            flags["needs_review"] = not bool(evidence.get("pos_entries"))
        if "evidence_missing" not in flags:
            flags["evidence_missing"] = not bool(evidence.get("pos_entries"))
        wordcard["flags"] = flags

        if flags.get("evidence_missing") and wordcard.get("popup"):
            glosses = wordcard["popup"].get("gloss_ru_short_list") or []
            if any(isinstance(g, str) and ("неизвест" in g.lower() or "нет слова" in g.lower()) for g in glosses):
                wordcard["popup"]["gloss_ru_short_list"] = []

        sources = wordcard.get("sources")
        if not sources or not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
            title = evidence.get("title_found") or lemma
            retrieved_at = evidence.get("retrieved_at") or now_iso
            wordcard["sources"] = [
                {
                    "type": "wiktionary",
                    "site": "en.wiktionary.org",
                    "title": title,
                    "retrieved_at": retrieved_at,
                }
            ]

        return wordcard

    # -----------------------------
    # Persistence
    # -----------------------------
    async def _load_cached_card(self, lemma: str, ui_lang: str) -> Optional[Dict[str, Any]]:
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                select(YiddishWordCard).where(
                    YiddishWordCard.lemma == lemma,
                    YiddishWordCard.ui_lang == ui_lang,
                    YiddishWordCard.source == "wiktionary",
                    YiddishWordCard.version == self.wordcard_version,
                )
            )
            card = result.scalar_one_or_none()
            if card and card.data:
                return card.data
        return None

    async def _load_cached_card_any(self, candidates: List[str], ui_lang: str) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                select(YiddishWordCard).where(
                    YiddishWordCard.lemma.in_(candidates),
                    YiddishWordCard.ui_lang == ui_lang,
                    YiddishWordCard.source == "wiktionary",
                    YiddishWordCard.version == self.wordcard_version,
                )
            )
            card = result.scalar_one_or_none()
            if card and card.data:
                return {"data": card.data, "evidence": card.evidence}
        return None

    async def _persist_wordcard(
        self,
        *,
        lemma: Optional[str] = None,
        lemma_key: Optional[str] = None,
        ui_lang: str,
        pos_default: Optional[str],
        evidence: Dict[str, Any],
        wordcard: Dict[str, Any],
    ) -> None:
        target_lemma = lemma_key or lemma or wordcard.get("lemma")
        if not target_lemma:
            return
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                select(YiddishWordCard).where(
                    YiddishWordCard.lemma == target_lemma,
                    YiddishWordCard.ui_lang == ui_lang,
                    YiddishWordCard.source == "wiktionary",
                    YiddishWordCard.version == self.wordcard_version,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                await session.execute(
                    update(YiddishWordCard)
                    .where(YiddishWordCard.id == existing.id)
                    .values(
                        word_surface=wordcard.get("word_surface"),
                        pos_default=pos_default,
                        data=wordcard,
                        evidence=evidence,
                        retrieved_at=datetime.now(timezone.utc),
                    )
                )
            else:
                session.add(
                    YiddishWordCard(
                        lemma=target_lemma,
                        lang="yi",
                        ui_lang=ui_lang,
                        source="wiktionary",
                        version=self.wordcard_version,
                        word_surface=wordcard.get("word_surface"),
                        pos_default=pos_default,
                        data=wordcard,
                        evidence=evidence,
                        retrieved_at=datetime.now(timezone.utc),
                    )
                )
