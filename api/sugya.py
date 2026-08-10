import asyncio
import json
import os
import re
import logging
from typing import Any, Dict, List, Optional
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.llm_config import LLMConfigError, get_llm_for_task
try:
    from core.database import async_session_factory
    from models.db import SugyaMapCache
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert
except Exception as _import_err:
    async_session_factory = None
    SugyaMapCache = None
    select = None
    pg_insert = None

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sugya_maps_cache_v4_parent_id.json")


def _load_disk_cache() -> Dict[str, Dict[str, Any]]:
    """Loads cached sugya maps from disk JSON file."""
    try:
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} cached v4 sugya maps from disk.")
                return data
    except Exception as err:
        logger.warning(f"Failed to load sugya maps disk cache: {err}")
    return {}


def _save_disk_cache():
    """Saves SUGYA_MAP_CACHE dictionary to disk JSON file."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(SUGYA_MAP_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as err:
        logger.warning(f"Failed to save sugya maps disk cache: {err}")


SUGYA_MAP_CACHE: Dict[str, Dict[str, Any]] = _load_disk_cache()


def strip_nikud_and_punct(text: str) -> str:
    """Strips Hebrew vowels (nikud), cantillation (taamim), HTML tags, and normalizes punctuation."""
    if not text:
        return ""
    text = re.sub(r"[\u0591-\u05C7]", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("״", '"').replace("׳", "'").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_segment_spans_continuous(he_text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Рассчитывает сплошные границы (без дыр и пробоин) для спанов сегмента.
    Конец каждого спана упирается в начало следующего спана.
    """
    if not he_text or not spans:
        return spans

    # Очистка от HTML перед разбивкой на слова для точного подсчета слов
    he_text_no_html = re.sub(r"<[^>]+>", " ", he_text)
    words = he_text_no_html.split()
    total_words = len(words)

    clean_he = strip_nikud_and_punct(he_text)
    last_found_idx = 0

    # 1. Находим стартовые позиции всех валидных спанов
    valid_spans: List[Dict[str, Any]] = []
    for i, span in enumerate(spans):
        start_quote = span.get("start_quote", "") or span.get("start_anchor", "")
        clean_start = strip_nikud_and_punct(start_quote)

        start_char_idx = -1
        if clean_start and len(clean_start) >= 2:
            # Ищем строго ПОСЛЕ предыдущего найденного спана
            start_char_idx = clean_he.find(clean_start, last_found_idx)
            if start_char_idx == -1:
                # Fallback по первому слову цитаты (длиной >= 2)
                start_words = [w for w in clean_start.split() if len(w) >= 2]
                if start_words:
                    start_char_idx = clean_he.find(start_words[0], last_found_idx)

        if start_char_idx != -1:
            if len(valid_spans) == 0 or (start_char_idx - last_found_idx >= 3):
                last_found_idx = start_char_idx
                span["_start_char"] = start_char_idx
                valid_spans.append(span)

    if not valid_spans:
        return spans

    spans = valid_spans

    # 2. Формируем непрерывные отрезки (Sliding Window)
    processed = []
    clean_words = clean_he.split()

    for i, span in enumerate(spans):
        start_char = span["_start_char"]
        
        # Конец текущего спана = начало следующего спана (или конец сегмента)
        if i + 1 < len(spans):
            end_char = spans[i + 1]["_start_char"]
            if end_char < start_char:
                end_char = len(clean_he)
        else:
            end_char = len(clean_he)

        # Перевод символьных позиций в word_idx
        start_w_idx = 0
        end_w_idx = total_words

        if clean_words:
            curr_pos = 0
            for w_i, w in enumerate(clean_words):
                w_len = len(w)
                if curr_pos <= start_char < curr_pos + w_len + 1:
                    start_w_idx = w_i
                    break
                curr_pos += w_len + 1

            curr_pos = 0
            for w_i, w in enumerate(clean_words):
                w_len = len(w)
                if curr_pos < end_char <= curr_pos + w_len + 1:
                    end_w_idx = w_i + 1
                    break
                curr_pos += w_len + 1

        if end_w_idx <= start_w_idx:
            end_w_idx = min(total_words, start_w_idx + 1)

        orig_words = words[start_w_idx:end_w_idx]
        start_anchor = " ".join(orig_words[:min(4, len(orig_words))]) if orig_words else span.get("start_quote", "")
        end_anchor = " ".join(orig_words[-min(4, len(orig_words)):]) if orig_words else start_anchor

        processed.append({
            "ref": span.get("ref"),
            "he_text": he_text,
            "sub_index": span.get("sub_index", i),
            "type": span.get("type", "Statement"),
            "start_quote": span.get("start_quote", ""),
            "end_quote": end_anchor,
            "title_ru": span.get("title_ru", ""),
            "speaker": span.get("speaker"),
            "start_anchor": start_anchor,
            "end_anchor": end_anchor,
            "start_word_idx": start_w_idx,
            "end_word_idx": end_w_idx,
        })

    return processed


def _clean_and_parse_json(cleaned: str) -> Dict[str, Any]:
    """Robustly parses JSON from LLM outputs with fallbacks."""
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        import json_repair
        repaired = json_repair.repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    fixed = re.sub(r",\s*([\}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        block = m.group(0)
        block_fixed = re.sub(r",\s*([\}\]])", r"\1", block)
        try:
            return json.loads(block_fixed)
        except Exception:
            pass

    return json.loads(cleaned)


class SugyaMapRequest(BaseModel):
    ref: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    force_recalculate: Optional[bool] = False


class SugyaNode(BaseModel):
    id: str
    level: Optional[int] = 1
    type: str  # Statement | Question | Attack | Defense | Proof | Answer
    title: str
    ref: Optional[str] = None
    sub_index: Optional[int] = None
    speaker: Optional[str] = None
    parent_id: Optional[str] = None
    relation_label: Optional[str] = None
    start_quote: Optional[str] = None
    end_quote: Optional[str] = None
    start_anchor: Optional[str] = None
    end_anchor: Optional[str] = None
    start_word_idx: Optional[int] = None
    end_word_idx: Optional[int] = None


class SugyaMapResponse(BaseModel):
    sugya_title: str
    version: int = 2
    mishnah_summary: Optional[str] = None
    markdown_tree: str
    nodes: List[SugyaNode]


def _filter_segments_by_paragraph_symbol(ref: Optional[str], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments:
        return []

    focus_idx = 0
    if ref:
        for idx, seg in enumerate(segments):
            if seg.get("ref") == ref:
                focus_idx = idx
                break

    start_idx = 0
    for i in range(focus_idx, -1, -1):
        he = segments[i].get("he_text") or segments[i].get("heText") or ""
        en = segments[i].get("en_text") or segments[i].get("enText") or segments[i].get("text") or ""
        if "§" in he or "§" in en:
            start_idx = i
            break

    end_idx = len(segments)
    for i in range(focus_idx + 1, len(segments)):
        he = segments[i].get("he_text") or segments[i].get("heText") or ""
        en = segments[i].get("en_text") or segments[i].get("enText") or segments[i].get("text") or ""
        if "§" in he or "§" in en:
            end_idx = i
            break

    if start_idx >= end_idx:
        end_idx = min(len(segments), start_idx + 1)
        for i in range(start_idx + 1, len(segments)):
            he = segments[i].get("he_text") or segments[i].get("heText") or ""
            en = segments[i].get("en_text") or segments[i].get("enText") or segments[i].get("text") or ""
            if "§" in he or "§" in en:
                end_idx = i
                break
            end_idx = i + 1

    return segments[start_idx:end_idx]


def _parse_talmud_ref(ref_str: str):
    m = re.match(r"^(.+?)\s+(\d+)([ab])(?::|\.|\s*(\d+))?", ref_str.strip(), re.IGNORECASE)
    if not m:
        return None
    book = m.group(1).strip()
    daf_num = int(m.group(2))
    amud = m.group(3).lower()
    segment_num = int(m.group(4)) if m.group(4) else None
    return book, daf_num, amud, segment_num


def _get_prev_amud_ref(ref_str: str) -> str | None:
    parsed = _parse_talmud_ref(ref_str)
    if not parsed:
        return None
    book, daf_num, amud, _ = parsed
    if amud == "b":
        return f"{book} {daf_num}a"
    elif amud == "a" and daf_num > 2:
        return f"{book} {daf_num - 1}b"
    return None


def _get_next_amud_ref(ref_str: str) -> str | None:
    parsed = _parse_talmud_ref(ref_str)
    if not parsed:
        return None
    book, daf_num, amud, _ = parsed
    if amud == "a":
        return f"{book} {daf_num}b"
    elif amud == "b":
        return f"{book} {daf_num + 1}a"
    return None


async def _fetch_sefaria_amud_segments(amud_ref: str) -> List[Dict[str, Any]]:
    sefaria_ref = amud_ref.strip().replace(" ", ".")
    url = f"https://www.sefaria.org/api/texts/{sefaria_ref}?commentary=0&context=0"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url)
            if res.status_code != 200:
                return []
            data = res.json()
            he = data.get("he", [])
            en = data.get("text", [])
            if isinstance(he, str): he = [he]
            if isinstance(en, str): en = [en]

            segments = []
            max_len = max(len(he), len(en))
            for i in range(max_len):
                h_text = he[i] if i < len(he) else ""
                e_text = en[i] if i < len(en) else ""
                seg_ref = f"{amud_ref}:{i+1}"
                segments.append({
                    "ref": seg_ref,
                    "he_text": h_text,
                    "en_text": e_text,
                })
            return segments
    except Exception as err:
        logger.warning(f"Failed to fetch Sefaria amud {amud_ref}: {err}")
        return []


async def _ensure_full_cross_page_context(ref: Optional[str], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ref:
        return segments or []

    working_segments = list(segments) if segments else []

    if not working_segments:
        parsed = _parse_talmud_ref(ref)
        if parsed:
            book, daf_num, amud, _ = parsed
            focus_amud = f"{book} {daf_num}{amud}"
            working_segments = await _fetch_sefaria_amud_segments(focus_amud)

    if not working_segments:
        return []

    focus_idx = 0
    for idx, seg in enumerate(working_segments):
        if seg.get("ref") == ref:
            focus_idx = idx
            break

    has_prev_section_symbol = any(
        "§" in (s.get("he_text") or s.get("heText") or "") or "§" in (s.get("en_text") or s.get("enText") or s.get("text") or "")
        for s in working_segments[:focus_idx + 1]
    )

    if not has_prev_section_symbol:
        prev_amud_ref = _get_prev_amud_ref(ref)
        if prev_amud_ref:
            logger.info(f"Sugya start not found on current amud. Fetching previous amud: {prev_amud_ref}")
            prev_segments = await _fetch_sefaria_amud_segments(prev_amud_ref)
            if prev_segments:
                working_segments = prev_segments + working_segments

    focus_idx = 0
    for idx, seg in enumerate(working_segments):
        if seg.get("ref") == ref:
            focus_idx = idx
            break

    has_next_section_symbol = any(
        "§" in (s.get("he_text") or s.get("heText") or "") or "§" in (s.get("en_text") or s.get("enText") or s.get("text") or "")
        for s in working_segments[focus_idx + 1:]
    )

    if not has_next_section_symbol:
        next_amud_ref = _get_next_amud_ref(ref)
        if next_amud_ref:
            logger.info(f"Sugya end not found on current amud. Fetching next amud: {next_amud_ref}")
            next_segments = await _fetch_sefaria_amud_segments(next_amud_ref)
            if next_segments:
                working_segments = working_segments + next_segments

    return working_segments


def _canonical_key(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    cleaned = ref.strip().replace(" ", ".").replace(":", ".")
    cleaned = re.sub(r"\.+", ".", cleaned)
    return cleaned


# --- STEP 2: PARALLEL SEGMENT SPAN EXTRACTION (LLM) ---

SPAN_EXTRACTION_PROMPT = """You are an expert Talmudic logic analyst.
Analyze the given single Gemara segment (Hebrew and English).
CRITICAL SEGMENTATION RULE (1 SEGMENT -> N SUB-SPANS):
Paragraph boundaries in publishers (like Sefaria) are arbitrary layout choices. A SINGLE paragraph/segment frequently contains MULTIPLE logical steps or DIFFERENT OPINIONS of various Sages (e.g., Chachamim opinion -> Rabba opinion -> Rabban Shimon ben Gamliel opinion).
Whenever the authority, speaker, rhetorical function, or opinion changes mid-paragraph, YOU MUST CREATE SEPARATE SUB-SPANS FOR EACH STEP/OPINION IN ORDER!

For EACH sub-span:
- "sub_index": 0-indexed integer order within this segment (0, 1, 2...)
- "type": Exactly one of ["Statement", "Question", "Attack", "Defense", "Proof", "Answer"]
- "start_quote": First 3-5 Hebrew words of this span in the original Hebrew text
- "title_ru": Concise explanation in Russian (WHO holds WHAT position and WHY). Explicitly state names of Sages when present.
- "speaker": Name of the Sage/Authority if explicitly mentioned (e.g., "Хахамим (Мудрецы)", "Раббан Шимон бен Гамлиэль"), or null if anonymous Gemara voice.

FEW-SHOT BENCHMARK EXAMPLE 1:
Input Ref: Gittin 2a:3
Input Hebrew: וַחֲכָמִים אוֹמְרִים: אֵינוֹ צָרִיךְ שֶׁיֹּאמַר ״בְּפָנַי נִכְתַּב וּבְפָנַי נֶחְתַּם״, אֶלָּא הַמֵּבִיא מִמְּדִינַת הַיָּם וְהַמּוֹלִיךְ. וְהַמֵּבִיא מִמְּדִינָה לִמְדִינָה בִּמְדִינַת הַיָּם, צָרִיךְ שֶׁיֹּאמַר ״בְּפָנַי נִכְתַּב וּבְפָנַי נֶחְתַּם״. רַבָּן שִׁמְעוֹן בֶּן גַּמְלִיאֵל אוֹמֵר: אֲפִילּוּ מֵהֶגְמוֹנְיָא לְהֶגְמוֹנְיָא.

Output JSON:
{
  "spans": [
    {
      "sub_index": 0,
      "type": "Statement",
      "start_quote": "וַחֲכָמִים אוֹמְרִים אֵינוֹ צָרִיךְ",
      "title_ru": "Мудрецы полагают: произносить формулу подтверждения нужно только при доставке из-за моря в Эрец-Исраэль и обратно.",
      "speaker": "Мудрецы (Хахамим)"
    },
    {
      "sub_index": 1,
      "type": "Statement",
      "start_quote": "וְהַמֵּבִיא מִמְּדִינָה לִמְדִינָה",
      "title_ru": "Мудрецы добавляют: при перевозке гет между разными регионами за границей тоже требуется произносить формулу.",
      "speaker": "Мудрецы (Хахамим)"
    },
    {
      "sub_index": 2,
      "type": "Statement",
      "start_quote": "רַבָּן שִׁמְעוֹן בֶּן גַּמְלִיאֵל",
      "title_ru": "Раббан Шимон бен Гамлиэль считает: формулу нужно говорить даже при доставке из одного округа (гегмония) в другой.",
      "speaker": "Раббан Шимон бен Гамлиэль"
    }
  ]
}

FEW-SHOT BENCHMARK EXAMPLE 2:
Input Ref: Chullin 87a:3
Input Hebrew: גְּמָ׳ תָּנוּ רַבָּנַן: ״וְשָׁפַךְ וְכִסָּה״ – מִי שֶׁשָּׁפַךְ יְכַסֶּה. שָׁחַט וְלֹא כִּסָּה, וְרָאָהוּ אַחֵר, מִנַּיִן שֶׁחַיָּיב לְכַסּוֹת? שֶׁנֶּאֱמַר: ״וָאֹמַר לִבְנֵי יִשְׂרָאֵל״ – אַזְהָרָה לְכׇל בְּנֵי יִשְׂרָאֵל.

Output JSON:
{
  "spans": [
    {
      "sub_index": 0,
      "type": "Statement",
      "start_quote": "גְּמָ׳ תָּנוּ רַבָּנַן",
      "title_ru": "Мудрецы учат в барайте: тот, кто пролил кровь при убое, должен ее покрыть.",
      "speaker": "Мудрецы (Барайта)"
    },
    {
      "sub_index": 1,
      "type": "Question",
      "start_quote": "שָׁחַט וְלֹא כִּסָּה",
      "title_ru": "Гемара спрашивает: если резник не покрыл кровь и другой это увидел, откуда известно, что другой обязан покрыть?",
      "speaker": null
    },
    {
      "sub_index": 2,
      "type": "Proof",
      "start_quote": "שֶׁנֶּאֱמַר ״וָאֹמַר לִבְנֵי",
      "title_ru": "Гемара доказывает из стиха 'Я сказал сынам Израиля': это предостережение для всех сынов Израиля.",
      "speaker": null
    }
  ]
}"""


async def _extract_segment_spans(
    llm_client: Any,
    model: str,
    reasoning_params: Dict[str, Any],
    capabilities: List[str],
    segment: Dict[str, Any],
) -> List[Dict[str, Any]]:
    seg_ref = segment.get("ref", "")
    he_text = segment.get("he_text") or segment.get("heText") or ""
    en_text = segment.get("en_text") or segment.get("enText") or segment.get("text") or ""

    user_content = f"Segment Ref: {seg_ref}\nHebrew: {he_text}\nEnglish: {en_text}"

    messages = [
        {"role": "system", "content": SPAN_EXTRACTION_PROMPT},
        {"role": "user", "content": user_content},
    ]

    req: Dict[str, Any] = {
        **reasoning_params,
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if "json_mode" in capabilities:
        req["response_format"] = {"type": "json_object"}

    try:
        completion = await llm_client.chat.completions.create(**req)
        content = completion.choices[0].message.content if completion and completion.choices else ""
        if content:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                elif "```" in cleaned:
                    cleaned = cleaned.rsplit("```", 1)[0]
            parsed = _clean_and_parse_json(cleaned.strip())
            spans = parsed.get("spans", [])
            if isinstance(spans, list) and spans:
                res = []
                for s in spans:
                    if isinstance(s, dict):
                        res.append({
                            "ref": seg_ref,
                            "he_text": he_text,
                            "sub_index": s.get("sub_index", 0),
                            "type": s.get("type", "Statement"),
                            "start_quote": s.get("start_quote", ""),
                            "end_quote": s.get("end_quote", ""),
                            "title_ru": s.get("title_ru") or s.get("title", ""),
                            "speaker": s.get("speaker"),
                        })
                if res:
                    # If LLM returned only 1 span but segment contains multiple opinion triggers, apply heuristic splitting
                    if len(res) == 1 and he_text:
                        clean_he_text = strip_nikud_and_punct(he_text)
                        trigger_pattern = r"(וחכמים\s+אומרים|רבן\s+[^\s]+\s+אומר|רבי\s+[^\s]+\s+אומר|אמר\s+רב|ואמר\s+רב|והמביא\s+ממדינה)"
                        matches = list(re.finditer(trigger_pattern, clean_he_text))
                        if len(matches) > 1:
                            split_spans = []
                            for m_idx in range(len(matches)):
                                start_p = matches[m_idx].start()
                                end_p = matches[m_idx + 1].start() if m_idx + 1 < len(matches) else len(he_text)
                                sub_str = he_text[start_p:end_p].strip()
                                words = sub_str.split()
                                if words:
                                    s_q = " ".join(words[:min(4, len(words))])
                                    e_q = " ".join(words[-min(4, len(words)):])
                                    split_spans.append({
                                        "ref": seg_ref,
                                        "he_text": he_text,
                                        "sub_index": m_idx,
                                        "type": "Statement",
                                        "start_quote": s_q,
                                        "end_quote": e_q,
                                        "title_ru": f"Мнение / Высказывание {m_idx + 1} в отрывке {seg_ref}",
                                        "speaker": words[0] if words else None,
                                    })
                            if split_spans:
                                return split_spans
                    return res
    except Exception as exc:
        logger.warning(f"Span extraction failed for segment {seg_ref}: {exc}")

    # Fallback if LLM call failed or returned empty: treat whole segment as 1 span
    first_words = " ".join(he_text.split()[:4]) if he_text else ""
    last_words = " ".join(he_text.split()[-4:]) if he_text else ""
    return [{
        "ref": seg_ref,
        "he_text": he_text,
        "sub_index": 0,
        "type": "Statement",
        "start_quote": first_words,
        "end_quote": last_words,
        "title_ru": f"Разбор отрывка {seg_ref}",
        "speaker": None,
    }]


# --- STEP 4: TREE RELATIONS (LLM + FALLBACK) ---

TREE_RELATIONS_PROMPT = """You are an expert Talmudic logical hierarchy builder.
Given a list of extracted logical nodes in a sugya sequence, construct the tree structure by assigning parent_id and relation_label.

Rules:
1. The primary premise/statement node (first main thesis) has parent_id: null and relation_label: null.
2. Every subsequent node MUST link to its logical parent node id (e.g., an Attack links to the Statement or Defense it challenges; a Defense links to the Attack it answers; a Proof links to the Statement it proves).
3. "relation_label": short Russian description of the logical connection (e.g. "Возражение на тезис", "Ответ на вопрос", "Доказательство мнения").

Return valid JSON matching this schema:
{
  "nodes_relations": [
    { "id": "node_1", "parent_id": null, "relation_label": null },
    { "id": "node_2", "parent_id": "node_1", "relation_label": "Возражение на тезис" }
  ]
}"""


async def _build_tree_relations(
    llm_client: Any,
    model: str,
    reasoning_params: Dict[str, Any],
    capabilities: List[str],
    raw_nodes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Calls LLM to populate parent_id and relation_label for nodes."""
    if not raw_nodes:
        return {}

    compact_list = [
        {"id": n["id"], "type": n["type"], "title": n["title_ru"], "ref": n["ref"]}
        for n in raw_nodes
    ]

    user_content = f"Sugya Nodes Sequence ({len(compact_list)} nodes):\n" + json.dumps(compact_list, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": TREE_RELATIONS_PROMPT},
        {"role": "user", "content": user_content},
    ]

    req: Dict[str, Any] = {
        **reasoning_params,
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if "json_mode" in capabilities:
        req["response_format"] = {"type": "json_object"}

    try:
        completion = await llm_client.chat.completions.create(**req)
        content = completion.choices[0].message.content if completion and completion.choices else ""
        if content:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                elif "```" in cleaned:
                    cleaned = cleaned.rsplit("```", 1)[0]
            parsed = _clean_and_parse_json(cleaned.strip())
            relations = parsed.get("nodes_relations", [])
            if isinstance(relations, list):
                rel_map = {}
                valid_ids = {n["id"] for n in raw_nodes}
                for item in relations:
                    if isinstance(item, dict) and "id" in item:
                        n_id = item["id"]
                        p_id = item.get("parent_id")
                        # Prevent self-loop or invalid parent_id
                        if p_id == n_id or p_id not in valid_ids:
                            p_id = None
                        rel_map[n_id] = {
                            "parent_id": p_id,
                            "relation_label": item.get("relation_label"),
                        }
                return rel_map
    except Exception as exc:
        logger.warning(f"Tree relations LLM call failed: {exc}. Using deterministic fallback.")

    # --- DETERMINISTIC FALLBACK FOR STEP 4 ---
    rel_map = {}
    for i, n in enumerate(raw_nodes):
        if i == 0:
            rel_map[n["id"]] = {"parent_id": None, "relation_label": None}
        else:
            # Connect to nearest preceding Statement / Question / Attack
            parent = None
            for j in range(i - 1, -1, -1):
                prev = raw_nodes[j]
                if prev["type"] in ["Statement", "Question", "Attack"]:
                    parent = prev["id"]
                    break
            if not parent:
                parent = raw_nodes[0]["id"]
            rel_map[n["id"]] = {
                "parent_id": parent,
                "relation_label": f"Связь с {parent}",
            }
    return rel_map


@router.post("/calculate-map", response_model=SugyaMapResponse)
@router.post("/calculate-map/", response_model=SugyaMapResponse)
async def calculate_sugya_map(payload: SugyaMapRequest):
    """
    Analyzes Talmudic sugya text with 5-stage architecture:
    1. Paragraph split
    2. Parallel span extraction (asyncio.gather)
    3. Fuzzy matching with nikud stripping
    4. Graph parent_id building with fallbacks
    5. Cache & return
    """
    if not payload.ref and not payload.segments:
        raise HTTPException(status_code=400, detail="Either 'ref' or 'segments' must be provided.")

    raw_key = payload.ref or (payload.segments[0].get("ref") if payload.segments else None)
    cache_key = _canonical_key(raw_key)

    if cache_key and not payload.force_recalculate:
        # 1. In-memory / disk cache check
        if cache_key in SUGYA_MAP_CACHE:
            logger.info(f"Returning cached sugya map from global registry for ref: {cache_key}")
            return SugyaMapResponse(**SUGYA_MAP_CACHE[cache_key])

        # 2. PostgreSQL DB lookup
        if async_session_factory and SugyaMapCache and select:
            try:
                async with async_session_factory() as session:
                    result = await session.execute(select(SugyaMapCache).where(SugyaMapCache.ref == cache_key))
                    db_obj = result.scalar_one_or_none()
                    if db_obj and db_obj.nodes:
                        cached_data = {
                            "sugya_title": db_obj.sugya_title,
                            "version": 2,
                            "mishnah_summary": db_obj.mishnah_summary,
                            "markdown_tree": db_obj.markdown_tree,
                            "nodes": db_obj.nodes,
                        }
                        SUGYA_MAP_CACHE[cache_key] = cached_data
                        logger.info(f"Loaded sugya map from PostgreSQL DB for ref: {cache_key}")
                        return SugyaMapResponse(**cached_data)
            except Exception as db_err:
                logger.warning(f"PostgreSQL sugya map lookup failed: {db_err}")

    try:
        llm_client, model, reasoning_params, capabilities = get_llm_for_task("SUGYA_MAP_GEN")
    except LLMConfigError as exc:
        logger.warning(f"LLM config error for SUGYA_MAP_GEN: {exc}. Falling back to STUDY task.")
        try:
            llm_client, model, reasoning_params, capabilities = get_llm_for_task("STUDY")
        except LLMConfigError:
            try:
                llm_client, model, reasoning_params, capabilities = get_llm_for_task("CHAT")
            except LLMConfigError as fallback_exc:
                raise HTTPException(status_code=500, detail=f"LLM configuration error: {fallback_exc}")

    if payload.model and payload.model.strip():
        model = payload.model.strip()

    # Step 1: Context building & Paragraph filtering
    full_context_segments = await _ensure_full_cross_page_context(payload.ref, payload.segments or [])
    target_segments = _filter_segments_by_paragraph_symbol(payload.ref, full_context_segments) if full_context_segments else []
    if not target_segments:
        target_segments = payload.segments or []

    if not target_segments:
        raise HTTPException(status_code=400, detail="No Talmud segments found to analyze.")

    # Step 2: Parallel span extraction for all target segments via asyncio.gather(...)
    tasks = [
        _extract_segment_spans(llm_client, model, reasoning_params, capabilities, seg)
        for seg in target_segments
    ]
    extracted_segment_spans = await asyncio.gather(*tasks)

    # Flatten extracted spans into chronological raw nodes
    raw_nodes: List[Dict[str, Any]] = []
    node_counter = 1
    for seg_spans in extracted_segment_spans:
        for span in seg_spans:
            raw_nodes.append({
                "id": f"node_{node_counter}",
                "ref": span["ref"],
                "he_text": span["he_text"],
                "sub_index": span["sub_index"],
                "type": span["type"],
                "start_quote": span["start_quote"],
                "end_quote": span["end_quote"],
                "title_ru": span["title_ru"],
                "speaker": span["speaker"],
            })
            node_counter += 1

    # Step 3: Continuous Sliding Window Processing per Segment
    processed_nodes: List[Dict[str, Any]] = []

    from collections import defaultdict
    spans_by_ref = defaultdict(list)
    for node_data in raw_nodes:
        spans_by_ref[node_data["ref"]].append(node_data)

    for seg_ref, seg_spans in spans_by_ref.items():
        he_text = seg_spans[0]["he_text"] if seg_spans else ""
        continuous_spans = process_segment_spans_continuous(he_text, seg_spans)
        processed_nodes.extend(continuous_spans)

    # Reassign sequential node IDs (node_1, node_2...)
    for idx, node_data in enumerate(processed_nodes, start=1):
        node_data["id"] = f"node_{idx}"

    # Step 4: Tree Relations (parent_id & relation_label)
    rel_map = await _build_tree_relations(
        llm_client, model, reasoning_params, capabilities, processed_nodes
    )

    # Construct final SugyaNode Pydantic objects & Markdown tree
    final_nodes: List[SugyaNode] = []
    markdown_lines = []
    sugya_title = payload.ref or "Карта Сугии Талмуда"

    if processed_nodes and processed_nodes[0].get("title_ru"):
        sugya_title = f"Сугия: {processed_nodes[0]['title_ru']}"

    for node_data in processed_nodes:
        n_id = node_data["id"]
        rel = rel_map.get(n_id, {})
        parent_id = rel.get("parent_id")
        relation_label = rel.get("relation_label")

        s_node = SugyaNode(
            id=n_id,
            level=1 if not parent_id else 2,
            type=node_data["type"],
            title=node_data["title_ru"],
            ref=node_data["ref"],
            sub_index=node_data["sub_index"],
            speaker=node_data.get("speaker"),
            parent_id=parent_id,
            relation_label=relation_label,
            start_quote=node_data.get("start_quote"),
            end_quote=node_data.get("end_quote"),
            start_anchor=node_data.get("start_anchor"),
            end_anchor=node_data.get("end_anchor"),
            start_word_idx=node_data.get("start_word_idx"),
            end_word_idx=node_data.get("end_word_idx"),
        )
        final_nodes.append(s_node)

        indent = "  " if parent_id else ""
        spk_str = f" [{s_node.speaker}]" if s_node.speaker else ""
        rel_str = f" ({relation_label})" if relation_label else ""
        markdown_lines.append(f"{indent}- **[{s_node.type}]**{spk_str} {s_node.title}{rel_str}")

    response_obj = SugyaMapResponse(
        sugya_title=sugya_title,
        version=2,
        mishnah_summary=None,
        markdown_tree="\n".join(markdown_lines),
        nodes=final_nodes,
    )
    data_dict = response_obj.model_dump()

    # Step 5: Save Cache to Memory, Disk and PostgreSQL DB
    refs_to_cache = set()
    if cache_key:
        refs_to_cache.add(cache_key)
    for node in response_obj.nodes:
        if node.ref:
            c_node_key = _canonical_key(node.ref)
            if c_node_key:
                refs_to_cache.add(c_node_key)

    for r_key in refs_to_cache:
        SUGYA_MAP_CACHE[r_key] = data_dict

    _save_disk_cache()

    if async_session_factory and SugyaMapCache and pg_insert:
        try:
            async with async_session_factory() as session:
                for r_key in refs_to_cache:
                    stmt = pg_insert(SugyaMapCache).values(
                        ref=r_key,
                        sugya_title=response_obj.sugya_title,
                        mishnah_summary=None,
                        markdown_tree=response_obj.markdown_tree,
                        nodes=[n.model_dump() for n in response_obj.nodes],
                    ).on_conflict_do_update(
                        index_elements=["ref"],
                        set_={
                            "sugya_title": response_obj.sugya_title,
                            "mishnah_summary": None,
                            "markdown_tree": response_obj.markdown_tree,
                            "nodes": [n.model_dump() for n in response_obj.nodes],
                        }
                    )
                    await session.execute(stmt)
                await session.commit()
                logger.info(f"Persisted sugya map v2 to PostgreSQL DB for {len(refs_to_cache)} refs.")
        except Exception as db_err:
            logger.warning(f"Failed to persist sugya map to PostgreSQL: {db_err}")

    return response_obj
