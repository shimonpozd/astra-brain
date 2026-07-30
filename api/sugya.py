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

CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sugya_maps_cache.json")


def _load_disk_cache() -> Dict[str, Dict[str, Any]]:
    """Loads cached sugya maps from disk JSON file."""
    try:
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} cached sugya maps from disk.")
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

SYSTEM_PROMPT = """You are an expert Talmudic logic analyst. Analyze the provided Gemara sugya sequence (Hebrew and English) representing the complete Talmudic sugya topic.
Deconstruct the logical hierarchy into a Markdown tree using headers (# H1, ## H2, ### H3, #### H4, ##### H5, ###### H6).
The sugya sequence may span multiple segments (up to 20+ segments). You must analyze all logical steps across the full sequence from start to finish.

CRITICAL DISCOURSE SEGMENTATION RULE (1 REF -> N LOGICAL NODES):
Paragraph boundaries and Sefaria ref boundaries (\n, ¶) are arbitrary publisher layout choices. A single source paragraph/ref can contain MULTIPLE logical steps (e.g., an Attack followed by a Defense). Whenever the rhetorical function changes mid-paragraph, you MUST create separate logical nodes for each step, sharing the same ref!

For EACH node in the hierarchy, assign exactly ONE of the following 6 canonical logical types:
- Statement (Mishna statement, Amoraic premise)
- Question (Informational or structural question)
- Attack (Kushya, contradiction, challenge)
- Defense (Tirutz, resolution to an attack)
- Proof (Ra'aya, proof from scripture or Tannaitic source)
- Answer (Response to a simple question)

FEW-SHOT BENCHMARK EXAMPLE:
Input Ref: Chullin 91a:11
Input Text HE: "וְאִי פְּשִׁיטָא לֵיהּ, אַמַּאי סוֹפֵג אַרְבָּעִים וְתוּ לָא? לִילְקֵי שְׁמוֹנִים! הָכָא בְּמַאי עָסְקִינַן – כְּגוֹן דְּלֵית בּוֹ כְּזַיִת..."
Input Text EN: "The Gemara seeks to clarify... Let him be flogged eighty times. The Gemara answers: Here we are dealing with..."

Output Nodes:
- Node 1 (Attack): "וְאִי פְּשִׁיטָא לֵיהּ, אַמַּאי סוֹפֵג אַרְבָּעִים וְתוּ לָא? לִילְקֵי שְׁמוֹנִים!" (ref: Chullin 91a:11, start_word_idx: 0, end_word_idx: 10)
- Node 2 (Defense): "הָכָא בְּמַאי עָסְקִינַן – כְּגוֹן דְּלֵית בּוֹ כְּזַיִת..." (ref: Chullin 91a:11, start_word_idx: 11, end_word_idx: 25)

LANGUAGE & DETAIL REQUIREMENTS (КРИТИЧЕСКИ ВАЖНО):
1. Provide "sugya_title", "mishnah_summary", and node "title" in Russian (Русский язык).
2. NO GENERIC TITLES: Never use vague titles like "Спор о жертвах", "Вопрос о глухонемом", or "Мнение Раבби Йоханана".
3. EXPLICIT SAGE OPINIONS & REASONING REQUIRED: In every title and in "mishnah_summary", you MUST explicitly state:
   - WHO holds which position (name of the Sage / Tanna / Amora).
   - WHAT their specific opinion or ruling is.
   - WHY (their reasoning, proof, or textual source).
   Good Example: "Рабби Хизкия считает, что запрет покрывать кровь действует даже на несъедобные жертвы, а Рабби Йоханан возражает, что несъедобное не считается едой."
   Bad Example: "Спор о применимости запрета к несъедобным жертвам."
4. The start_anchor and end_anchor MUST remain exact Hebrew/Aramaic substrings from the original Hebrew text. Provide start_word_idx and end_word_idx as 0-indexed word counts in the space-split Hebrew text.

OUTPUT FORMAT REQUIREMENTS:
Return valid JSON matching this schema:
{
  "sugya_title": "Короткий заголовок темы сугии на русском языке с указанием сути предмета спора",
  "mishnah_summary": "Детальное резюме/перевод исходной Мишны на русском языке с указанием мнений каждого Танная (например, 'Рабби Меир говорит X, а Мудрецы говорят Y')",
  "markdown_tree": "Markdown string formatted with H1-H6 headers",
  "nodes": [
    {
      "id": "node_1",
      "level": 1,
      "type": "Statement" | "Question" | "Attack" | "Defense" | "Proof" | "Answer",
      "title": "Детальное описание логического шага на русском языке (КТО считает ЧТО и ПОЧЕМУ)",
      "ref": "Sefaria segment reference (e.g. Chullin 89b:10)",
      "sub_index": 0,
      "start_anchor": "Exact first 2-4 Hebrew words of this block",
      "end_anchor": "Exact last 2-4 Hebrew words of this block",
      "start_word_idx": 0,
      "end_word_idx": 10
    }
  ]
}

CRITICAL: The start_anchor and end_anchor MUST be exact substrings from the Hebrew text. Ensure all string values inside JSON are properly escaped (do NOT put raw unescaped double quotes inside titles or sugya_title). Do NOT include trailing commas."""


def _extract_governing_mishnah(segments: List[Dict[str, Any]], focus_ref: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Scans segments to extract the full multi-segment Mishnah.
    Starts at segment containing 'MISHNAH' / 'Mishna' in English text.
    Collects all consecutive segments until 'GEMARA' / 'Gemara:' is reached.
    """
    if not segments:
        return None

    # 1. Find focus index
    focus_idx = 0
    if focus_ref:
        for idx, seg in enumerate(segments):
            if seg.get("ref") == focus_ref:
                focus_idx = idx
                break

    # 2. Find starting Mishnah segment (scan backward from focus_idx)
    mishnah_start_idx = -1
    for i in range(focus_idx, -1, -1):
        en_text = (segments[i].get("en_text") or segments[i].get("enText") or segments[i].get("text") or "").strip()
        if re.search(r"\bMISHNAH\b|\bMishna\b", en_text, re.IGNORECASE):
            mishnah_start_idx = i
            break

    # Fallback scan from beginning if not found backward
    if mishnah_start_idx == -1:
        for i in range(len(segments)):
            en_text = (segments[i].get("en_text") or segments[i].get("enText") or segments[i].get("text") or "").strip()
            if re.search(r"\bMISHNAH\b|\bMishna\b", en_text, re.IGNORECASE):
                mishnah_start_idx = i
                break

    if mishnah_start_idx == -1:
        return None

    he_parts = []
    en_parts = []

    for i in range(mishnah_start_idx, len(segments)):
        seg = segments[i]
        en_text = (seg.get("en_text") or seg.get("enText") or seg.get("text") or "").strip()
        he_text = (seg.get("he_text") or seg.get("heText") or "").strip()

        # Stop if we hit GEMARA on subsequent segment
        if i > mishnah_start_idx and re.search(r"\bGEMARA\b|\bGemara\b", en_text, re.IGNORECASE):
            break

        if he_text:
            he_parts.append(he_text)
        if en_text:
            en_parts.append(en_text)

        # Stop if GEMARA is explicitly inside the current segment text
        if re.search(r"\bGEMARA\b|\bGemara\b", en_text, re.IGNORECASE) and i > mishnah_start_idx:
            break

    if not he_parts and not en_parts:
        return None

    return {
        "he_text": " ".join(he_parts),
        "en_text": " ".join(en_parts),
    }


def _clean_and_parse_json(cleaned: str) -> Dict[str, Any]:
    """
    Robustly parses JSON from LLM outputs with fallbacks for trailing commas,
    unescaped quotes, or markdown codeblocks.
    """
    # 1. Direct standard json.loads
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Try json_repair package if installed
    try:
        import json_repair
        repaired = json_repair.repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    # 3. Fix trailing commas before closing brackets/braces
    fixed = re.sub(r",\s*([\}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # 4. Extract outer {...} JSON block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        block = m.group(0)
        block_fixed = re.sub(r",\s*([\}\]])", r"\1", block)
        try:
            return json.loads(block_fixed)
        except Exception:
            pass

    # Final attempt: re-raise original JSON error
    return json.loads(cleaned)


SUGYA_MAP_CACHE: Dict[str, Dict[str, Any]] = {}


class SugyaMapRequest(BaseModel):
    ref: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    force_recalculate: Optional[bool] = False


class SugyaNode(BaseModel):
    id: str
    level: int
    type: str  # Statement | Question | Attack | Defense | Proof | Answer
    title: str
    ref: Optional[str] = None
    start_anchor: Optional[str] = None
    end_anchor: Optional[str] = None


class SugyaMapResponse(BaseModel):
    sugya_title: str
    mishnah_summary: Optional[str] = Field(None, description="Краткое резюме Мишны на русском языке")
    markdown_tree: str
    nodes: List[SugyaNode]


def _filter_segments_by_paragraph_symbol(ref: Optional[str], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Finds the exact Sefaria Sugya topic range bounded by paragraph symbols '§'.
    Scans backward from focus ref to find the starting '§', and scans forward
    to find the next '§' (which marks the start of the next topic).
    Returns all segments from the previous '§' up to the segment before the next '§'.
    """
    if not segments:
        return []

    # 1. Find focus index matching the target reference
    focus_idx = 0
    if ref:
        for idx, seg in enumerate(segments):
            if seg.get("ref") == ref:
                focus_idx = idx
                break

    # 2. Scan BACKWARD from focus_idx to find segment containing the starting '§'
    start_idx = 0
    for i in range(focus_idx, -1, -1):
        he = segments[i].get("he_text") or segments[i].get("heText") or ""
        en = segments[i].get("en_text") or segments[i].get("enText") or segments[i].get("text") or ""
        if "§" in he or "§" in en:
            start_idx = i
            break

    # 3. Scan FORWARD from focus_idx + 1 to find segment containing the NEXT '§' (start of next sugya)
    end_idx = len(segments)
    for i in range(focus_idx + 1, len(segments)):
        he = segments[i].get("he_text") or segments[i].get("heText") or ""
        en = segments[i].get("en_text") or segments[i].get("enText") or segments[i].get("text") or ""
        if "§" in he or "§" in en:
            end_idx = i  # exclude segment starting the NEXT sugya topic
            break

    # Safeguard: if start_idx >= end_idx (e.g. focus_idx is on the next § itself), include up to next § or max 15
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
    """
    If the provided segments do NOT contain a starting '§' before the focus ref,
    fetches the previous page/amud from Sefaria and prepends its segments.
    If no ending '§' is found after the focus ref, fetches the next page/amud.
    """
    if not ref:
        return segments or []

    working_segments = list(segments) if segments else []

    # If working_segments is empty, fetch the focus amud first
    if not working_segments:
        parsed = _parse_talmud_ref(ref)
        if parsed:
            book, daf_num, amud, _ = parsed
            focus_amud = f"{book} {daf_num}{amud}"
            working_segments = await _fetch_sefaria_amud_segments(focus_amud)

    if not working_segments:
        return []

    # Check if there is a starting '§' before focus_idx
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

    # Check if there is a closing '§' after focus_idx
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


async def _build_user_prompt(ref: Optional[str], segments: Optional[List[Dict[str, Any]]]) -> str:
    parts = []
    if ref:
        parts.append(f"SUGYA FOCUS REFERENCE: {ref}")

    full_context_segments = await _ensure_full_cross_page_context(ref, segments or [])

    # Extract multi-segment Mishnah from 'MISHNAH' up to 'GEMARA'
    mishnah_data = _extract_governing_mishnah(full_context_segments, focus_ref=ref)
    if mishnah_data:
        parts.append("==================================================")
        parts.append("GOVERNING MISHNAH FOR THIS SUGYA (FULL MULTI-SEGMENT MISHNAH FROM 'MISHNAH' UNTIL 'GEMARA'):")
        if mishnah_data.get("he_text"):
            parts.append(f"Hebrew Mishnah Text:\n{mishnah_data['he_text']}")
        if mishnah_data.get("en_text"):
            parts.append(f"English Mishnah Text:\n{mishnah_data['en_text']}")
        parts.append("==================================================")
        parts.append("INSTRUCTION FOR 'mishnah_summary': Synthesize a detailed, explicit summary of this governing Mishnah in Russian. You MUST explicitly state WHICH Sage/Tanna holds WHICH opinion (e.g., 'Рабби Меир считает X, а Мудрецы считают Y').")
        parts.append("")

    target_segments = _filter_segments_by_paragraph_symbol(ref, full_context_segments) if full_context_segments else []

    if target_segments:
        parts.append(f"SUGYA FULL SEQUENCE ({len(target_segments)} segments):")
        for seg in target_segments:
            seg_ref = seg.get("ref", "")
            he = seg.get("he_text") or seg.get("heText") or ""
            en = seg.get("en_text") or seg.get("enText") or seg.get("text") or ""
            parts.append(f"--- Segment [{seg_ref}] ---")
            if he:
                parts.append(f"Hebrew/Aramaic: {he}")
            if en:
                parts.append(f"English: {en}")
    else:
        parts.append("Please analyze the sugya for the given reference.")

    return "\n".join(parts)


def _canonical_key(ref: Optional[str]) -> Optional[str]:
    """Normalizes Sefaria references to a uniform canonical key (e.g. 'Menachot 29b:2' -> 'Menachot.29b.2')."""
    if not ref:
        return None
    cleaned = ref.strip().replace(" ", ".").replace(":", ".")
    cleaned = re.sub(r"\.+", ".", cleaned)
    return cleaned


@router.post("/calculate-map", response_model=SugyaMapResponse)
@router.post("/calculate-map/", response_model=SugyaMapResponse)
async def calculate_sugya_map(payload: SugyaMapRequest):
    """
    Analyzes Talmudic sugya text and generates logical hierarchy (Sugya Mind Map).
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
                    if db_obj:
                        cached_data = {
                            "sugya_title": db_obj.sugya_title,
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

    user_prompt = await _build_user_prompt(payload.ref, payload.segments)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
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
    except Exception as exc:
        logger.error("LLM call failed for Sugya Map calculation", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM completion failed: {exc}")

    content = completion.choices[0].message.content if completion and completion.choices else ""
    if not content:
        raise HTTPException(status_code=502, detail="Empty response from LLM")

    # Clean markdown json codeblock wrappers if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        elif "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        data = _clean_and_parse_json(cleaned)
    except Exception as exc:
        logger.error(f"Failed to parse LLM response as JSON: {cleaned[:200]}...", exc_info=True)
        raise HTTPException(status_code=502, detail="LLM response was not valid JSON")

    try:
        response_obj = SugyaMapResponse(**data)
        data_dict = response_obj.model_dump()

        # Collect all refs in the sugya (from focus ref and all nodes in the tree)
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

        # Save to PostgreSQL DB for all refs in the sugya
        if async_session_factory and SugyaMapCache and pg_insert:
            try:
                async with async_session_factory() as session:
                    for r_key in refs_to_cache:
                        stmt = pg_insert(SugyaMapCache).values(
                            ref=r_key,
                            sugya_title=response_obj.sugya_title,
                            mishnah_summary=response_obj.mishnah_summary,
                            markdown_tree=response_obj.markdown_tree,
                            nodes=[n.model_dump() for n in response_obj.nodes],
                        ).on_conflict_do_update(
                            index_elements=["ref"],
                            set_={
                                "sugya_title": response_obj.sugya_title,
                                "mishnah_summary": response_obj.mishnah_summary,
                                "markdown_tree": response_obj.markdown_tree,
                                "nodes": [n.model_dump() for n in response_obj.nodes],
                            }
                        )
                        await session.execute(stmt)
                    await session.commit()
                    logger.info(f"Persisted sugya map to PostgreSQL DB for {len(refs_to_cache)} refs: {list(refs_to_cache)}")
            except Exception as db_err:
                logger.warning(f"Failed to persist sugya map to PostgreSQL: {db_err}")

        return response_obj
    except Exception as exc:
        logger.error(f"SugyaMapResponse validation error: {exc}", exc_info=True)
        # Fallback formatting if dict missing required keys
        nodes = []
        raw_nodes = data.get("nodes", [])
        if isinstance(raw_nodes, list):
            for i, n in enumerate(raw_nodes):
                if isinstance(n, dict):
                    nodes.append(SugyaNode(
                        id=str(n.get("id", f"node_{i+1}")),
                        level=int(n.get("level", 1)),
                        type=str(n.get("type", "Statement")),
                        title=str(n.get("title", f"Node {i+1}")),
                        ref=n.get("ref"),
                        start_anchor=n.get("start_anchor"),
                        end_anchor=n.get("end_anchor"),
                    ))

        fallback_obj = SugyaMapResponse(
            sugya_title=data.get("sugya_title") or payload.ref or "Сугия",
            markdown_tree=data.get("markdown_tree") or "",
            nodes=nodes,
        )
        if cache_key:
            SUGYA_MAP_CACHE[cache_key] = fallback_obj.model_dump()
        return fallback_obj
