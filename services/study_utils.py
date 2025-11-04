# brain_service/services/study_utils.py
import logging
import re
import time
import html
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from urllib.parse import quote
from itertools import zip_longest

from brain_service.services.sefaria_service import SefariaService
from brain_service.services.sefaria_index_service import SefariaIndexService
from .study_state import Bookshelf, BookshelfItem
from .sefaria_index import get_book_structure
from config import get_config_section

import logging_utils

logger = logging_utils.get_logger(__name__)

# --- Constants & Configuration ---
WINDOW_SIZE = 5
PREVIEW_MAX_LEN = 600

# --- Collection & Ref Parsing Logic ---

def detect_collection(ref: str) -> str:
    ref_lower = ref.lower()
    if ' on ' in ref_lower:
        return "Commentary"
    # This can be improved by using the categories from get_book_structure
    talmud_tractates = [
        'shabbat', 'berakhot', 'pesachim', 'ketubot', 'gittin', 'kiddushin', 'bava kamma', 'bava metzia', 'bava batra', 'sanhedrin', 'makkot',
        'zevachim', 'menachot', 'hullin', 'bekhorot', 'arakhin', 'temurah', 'keritot', 'meilah', 'tamid', 'middot', 'kinnim', 'niddah',
        'yoma', 'sukkah', 'beitza', 'rosh hashana', 'taanit', 'megillah', 'moed katan', 'chagigah', 'yevamot', 'nedarim', 'nazir', 'sotah',
        'horayot', 'shevuot', 'avodah zarah', 'avot', 'eduoyot', 'abodah zarah', 'pirkei avot', 'tosefta', 'jerusalem talmud'
    ]
    if any(tractate in ref_lower for tractate in talmud_tractates):
        return "Talmud"
    bible_books = [
        'genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy', 'joshua', 'judges', 'samuel', 'kings', 'isaiah', 'jeremiah', 'ezekiel',
        'psalms', 'proverbs', 'job', 'song of songs', 'ruth', 'lamentations', 'ecclesiastes', 'esther', 'daniel', 'ezra', 'nehemiah', 'chronicles',
        'hosea', 'joel', 'amos', 'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 'haggai', 'zechariah', 'malachi',
        'i samuel', 'ii samuel', 'i kings', 'ii kings', 'i chronicles', 'ii chronicles'
    ]
    if any(book in ref_lower for book in bible_books):
        return "Bible"
    if 'mishnah' in ref_lower:
        return "Mishnah"
    return "Unknown"

def _coerce_bible_ref_string(ref: str) -> str:
    """If ref looks like a Bible ref but has an accidental Talmud amud pattern (e.g., 'Exodus 16b.12'),
    coerce it to 'Exodus 16:12'. Keeps non-Bible refs untouched.
    """
    try:
        lowered = ref.lower()
        bible_books = [
            'genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy', 'joshua', 'judges', 'samuel', 'kings', 'isaiah', 'jeremiah', 'ezekiel',
            'psalms', 'proverbs', 'job', 'song of songs', 'ruth', 'lamentations', 'ecclesiastes', 'esther', 'daniel', 'ezra', 'nehemiah', 'chronicles',
            'hosea', 'joel', 'amos', 'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 'haggai', 'zechariah', 'malachi',
            'i samuel', 'ii samuel', 'i kings', 'ii kings', 'i chronicles', 'ii chronicles'
        ]
        if not any(book in lowered for book in bible_books):
            return ref
        # Match '<Book> <chapter>[ab][.:]<verse>'
        m = re.match(r"([\w\s'.]+) (\d+)[ab][\.:](\d+)$", ref, re.IGNORECASE)
        if m:
            return f"{m.group(1).strip()} {int(m.group(2))}:{int(m.group(3))}"
        return ref
    except Exception:
        return ref

def _parse_ref(ref: str) -> Optional[Dict[str, Any]]:
    # First, try to detect if this is likely Tanakh/Bible based on book names
    ref_lower = ref.lower()
    bible_books = [
        'genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy', 'joshua', 'judges', 'samuel', 'kings', 'isaiah', 'jeremiah', 'ezekiel',
        'psalms', 'proverbs', 'job', 'song of songs', 'ruth', 'lamentations', 'ecclesiastes', 'esther', 'daniel', 'ezra', 'nehemiah', 'chronicles',
        'hosea', 'joel', 'amos', 'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 'haggai', 'zechariah', 'malachi',
        'i samuel', 'ii samuel', 'i kings', 'ii kings', 'i chronicles', 'ii chronicles'
    ]
    
    # Check if this looks like a Bible book
    is_likely_bible = any(book in ref_lower for book in bible_books)
    
    logger.debug(f"[daily] PARSE_REF: '{ref}' -> is_likely_bible={is_likely_bible}")
    
    if is_likely_bible:
        # For Bible books, prioritize the Bible format
        match = re.match(r"([\w\s'.]+) (\d+):(\d+)", ref)
        if match:
            result = {"type": "bible", "book": match.group(1).strip(), "chapter": int(match.group(2)), "verse": int(match.group(3))}
            logger.debug(f"[daily] PARSE_REF: Bible format matched -> {result}")
            return result
        # Fallback to Talmud format if Bible format doesn't match
        match = re.match(r"([\w\s'.]+) (\d+)([ab])(?:[.:](\d+))?", ref)
        if match:
            result = {"type": "talmud", "book": match.group(1).strip(), "page": int(match.group(2)), "amud": match.group(3), "segment": int(match.group(4)) if match.group(4) else 1}
            logger.warning(f"[daily] PARSE_REF: Bible book but Talmud format matched -> {result}")
            return result
    else:
        # For non-Bible books, try Talmud format first
        match = re.match(r"([\w\s'.]+) (\d+)([ab])(?:[.:](\d+))?", ref)
        if match:
            result = {"type": "talmud", "book": match.group(1).strip(), "page": int(match.group(2)), "amud": match.group(3), "segment": int(match.group(4)) if match.group(4) else 1}
            logger.debug(f"[daily] PARSE_REF: Talmud format matched -> {result}")
            return result
        # Fallback to Bible format
        match = re.match(r"([\w\s'.]+) (\d+):(\d+)", ref)
        if match:
            result = {"type": "bible", "book": match.group(1).strip(), "chapter": int(match.group(2)), "verse": int(match.group(3))}
            logger.debug(f"[daily] PARSE_REF: Bible format matched (fallback) -> {result}")
            return result
    
    logger.warning(f"[daily] PARSE_REF: No format matched for '{ref}'")
    return None


def _should_delegate_to_modular(ref: str, data: Dict[str, Any]) -> bool:
    """Decide whether legacy daily flow should reuse the modular builder output."""

    segments = data.get("text_segments")
    if not isinstance(segments, list) or len(segments) <= 1:
        return False
    collection = detect_collection(ref)
    type_label = str(data.get("type", "") or "").lower()
    return collection == "Bible" or type_label in {"bible", "tanakh"}


def _flatten_talmud_lines(values: Any) -> List[str]:
    """Flatten nested lists of Talmud lines into a simple list of strings."""
    flattened: List[str] = []
    if isinstance(values, list):
        for entry in values:
            flattened.extend(_flatten_talmud_lines(entry))
    elif isinstance(values, str):
        if values.strip():
            flattened.append(values)
    elif values:
        flattened.append(str(values))
    return flattened


async def _collect_talmud_amud_segments(
    book: str,
    daf: str,
    side: str,
    sefaria_service: SefariaService,
    metadata_source: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch a single amud (e.g., Zevachim 53a) and convert its lines into segments.
    """
    amud_ref = f"{book} {daf}{side}"
    logger.debug("daily.talmud.amud.fetch", extra={"ref": amud_ref})

    try:
        response = await sefaria_service.get_text(amud_ref)
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning(
            "daily.talmud.amud.fetch_failed",
            extra={"ref": amud_ref, "error": str(exc)},
        )
        return []

    if not isinstance(response, dict) or not response.get("ok"):
        logger.debug(
            "daily.talmud.amud.empty_response",
            extra={"ref": amud_ref, "response_type": type(response)},
        )
        return []

    data = response.get("data") or {}
    text_candidates = data.get("text_segments") or data.get("text")
    he_candidates = data.get("he_segments") or data.get("he")

    text_lines = _flatten_talmud_lines(text_candidates)
    he_lines = _flatten_talmud_lines(he_candidates)
    total = max(len(text_lines), len(he_lines))

    if total == 0:
        logger.debug(
            "daily.talmud.amud.no_lines",
            extra={"ref": amud_ref, "text_type": type(text_candidates)},
        )
        return []

    base_title = (metadata_source or {}).get("title") or data.get("title") or amud_ref
    index_title = (
        (metadata_source or {}).get("indexTitle")
        or data.get("indexTitle")
        or (metadata_source or {}).get("book")
        or book
    )
    he_ref = data.get("heRef", "")

    segments: List[Dict[str, Any]] = []
    for idx in range(total):
        en_text = text_lines[idx] if idx < len(text_lines) else ""
        he_text = he_lines[idx] if idx < len(he_lines) else ""

        if not (en_text or he_text):
            continue

        clean_en = _clean_html_text(en_text)
        clean_he = _clean_html_text(he_text)

        segment_ref = f"{book} {daf}{side}:{idx + 1}"
        segments.append(
            {
                "ref": segment_ref,
                "text": clean_he or clean_en,
                "heText": clean_he or clean_en,
                "enText": clean_en,
                "metadata": {
                    "title": base_title,
                    "indexTitle": index_title,
                    "heRef": he_ref,
                    "page": daf,
                    "amud": side,
                    "segment": idx + 1,
                },
            }
        )

    return segments


async def _build_talmud_payload(
    ref: str,
    data: Optional[Dict[str, Any]],
    sefaria_service: SefariaService,
    *,
    session_id: Optional[str] = None,
    redis_client=None,
) -> Optional[Dict[str, Any]]:
    """
    Build a segment payload for a Babylonian Talmud daf.
    """
    parsed = _parse_ref(ref)
    book: Optional[str] = None
    daf_number: Optional[int] = None
    forced_side: Optional[str] = None

    if parsed and parsed.get("type") == "talmud":
        book = parsed.get("book")
        daf_number = parsed.get("page")
        forced_side = parsed.get("amud")
    else:
        match = re.match(r"([\w\s'.]+) (\d+)([ab])?$", ref)
        if match:
            book = match.group(1).strip()
            daf_number = int(match.group(2))
            forced_side = match.group(3)

    if not book or daf_number is None:
        return None

    daf_str = str(daf_number)
    sides = [forced_side] if forced_side in {"a", "b"} else ["a", "b"]

    segments: List[Dict[str, Any]] = []
    for side in sides:
        amud_segments = await _collect_talmud_amud_segments(
            book,
            daf_str,
            side,
            sefaria_service,
            metadata_source=data if isinstance(data, dict) else None,
        )
        if amud_segments:
            segments.extend(amud_segments)

    if not segments:
        return None

    total_segments = len(segments)
    for idx, segment in enumerate(segments):
        position = idx / (total_segments - 1) if total_segments > 1 else 0.5
        segment["position"] = float(position)

    if session_id and redis_client:
        try:
            count_key = f"daily:sess:{session_id}:total_segments"
            await redis_client.set(count_key, total_segments, ex=3600 * 24 * 7)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "daily.talmud.redis_total_failed",
                extra={"ref": ref, "session_id": session_id, "error": str(exc)},
            )

    return {
        "segments": segments,
        "focusIndex": 0,
        "totalLength": total_segments,
        "ref": ref,
        "loadedAt": str(int(time.time() * 1000)),
        "he_ref": segments[0]["metadata"].get("heRef") if segments else None,
    }



# --- Navigation & Windowing Logic ---

async def _ensure_chapter_length(
    book: str,
    chapter: int,
    book_structure: Optional[Dict[str, Any]],
    cache: Dict[int, Optional[int]],
    sefaria_service: SefariaService,
) -> Optional[int]:
    if chapter <= 0:
        return None
    if chapter in cache:
        return cache[chapter]

    length: Optional[int] = None
    if book_structure:
        lengths = book_structure.get("lengths")
        if isinstance(lengths, list) and 0 <= (chapter - 1) < len(lengths):
            length = lengths[chapter - 1]

    if not length:
        try:
            chapter_ref = f"{book} {chapter}"
            chapter_result = await sefaria_service.get_text(chapter_ref)
            if chapter_result.get("ok") and chapter_result.get("data"):
                data = chapter_result["data"]
                if isinstance(data, dict):
                    segment_list = data.get("text") or data.get("he")
                    if isinstance(segment_list, list):
                        length = len([item for item in segment_list if item])
                elif isinstance(data, list):
                    length = len([item for item in data if item])
        except Exception:  # pragma: no cover - network errors fallback to None
            length = None

    cache[chapter] = length
    return length


async def _generate_and_validate_refs(base_ref: str, collection: str, direction: str, count: int, sefaria_service: SefariaService, index_service: SefariaIndexService) -> List[Dict[str, str]]:
    """Generates and validates a list of previous/next references with page/chapter transitions."""
    if not base_ref:
        return []

    parsed_ref = _parse_ref(base_ref)
    if not parsed_ref:
        return []

    # Get TOC data from index service
    toc_data = index_service.toc if hasattr(index_service, 'toc') else None
    book_structure = get_book_structure(parsed_ref['book'], toc_data)
    chapter_length_cache: Dict[int, Optional[int]] = {}
    
    generated_refs = []
    current_ref_parts = parsed_ref.copy()

    # Try up to e.g., 20 times to find `count` valid references
    for _ in range(count * 5):
        if len(generated_refs) >= count:
            break

        delta = 1 if direction == 'next' else -1
        candidate_ref_str = None
        
        # --- 1. Generate a candidate reference string ---
        if current_ref_parts['type'] == 'talmud':
            current_segment = current_ref_parts.get('segment')
            if current_segment is None:
                current_segment = 1
            next_segment = current_segment + delta
            if next_segment > 0:
                candidate_ref_str = f"{current_ref_parts['book']} {current_ref_parts['page']}{current_ref_parts['amud']}:{next_segment}"
                current_ref_parts['segment'] = next_segment # Tentatively update
            # Backward page transition is too complex for now

        elif current_ref_parts['type'] == 'bible':
            chapter = current_ref_parts.get('chapter')
            verse = current_ref_parts.get('verse')
            if not chapter or not verse:
                break

            length = await _ensure_chapter_length(
                current_ref_parts['book'],
                chapter,
                book_structure,
                chapter_length_cache,
                sefaria_service,
            )
            if not length:
                break

            new_verse = verse + delta

            if 1 <= new_verse <= length:
                current_ref_parts['verse'] = new_verse
            elif new_verse > length and direction == 'next':
                next_chapter = chapter + 1
                next_length = await _ensure_chapter_length(
                    current_ref_parts['book'],
                    next_chapter,
                    book_structure,
                    chapter_length_cache,
                    sefaria_service,
                )
                if not next_length:
                    break
                # For Tanakh, when transitioning to a new chapter, load the full chapter
                # This will be handled by the range loading logic in get_text_with_window
                current_ref_parts['chapter'] = next_chapter
                current_ref_parts['verse'] = 1
            elif new_verse < 1 and direction == 'prev':
                prev_chapter = chapter - 1
                prev_length = await _ensure_chapter_length(
                    current_ref_parts['book'],
                    prev_chapter,
                    book_structure,
                    chapter_length_cache,
                    sefaria_service,
                )
                if not prev_length:
                    break
                # For Tanakh, when transitioning to a previous chapter, load the full chapter
                # This will be handled by the range loading logic in get_text_with_window
                current_ref_parts['chapter'] = prev_chapter
                current_ref_parts['verse'] = prev_length
            else:
                break

            candidate_ref_str = f"{current_ref_parts['book']} {current_ref_parts['chapter']}:{current_ref_parts['verse']}"

        if not candidate_ref_str:
            break

        # --- 2. Validate the candidate ---
        text_result = await sefaria_service.get_text(candidate_ref_str)
        if text_result.get("ok") and text_result.get("data"):
            generated_refs.append(text_result["data"])
        else:
            # Validation failed, if it's Talmud and we're going forward, try jumping page
            if current_ref_parts['type'] == 'talmud' and direction == 'next':
                if current_ref_parts['amud'] == 'a':
                    current_ref_parts['amud'] = 'b'
                    current_ref_parts['segment'] = 0 # Will be incremented to 1 at the start of the next loop
                else: # amud was 'b'
                    current_ref_parts['page'] += 1
                    current_ref_parts['amud'] = 'a'
                    current_ref_parts['segment'] = 0 # Will be incremented to 1
                continue # Retry loop with new page/amud settings
            else:
                # For other types or directions, stop if we hit a dead end
                break
    
    if direction == 'prev':
        generated_refs.reverse()

    return generated_refs

async def _generate_tanakh_chapter_refs(base_ref: str, direction: str, count: int, sefaria_service: SefariaService, index_service: SefariaIndexService) -> List[Dict[str, Any]]:
    """Generate and validate a list of previous/next Tanakh chapters."""
    if not base_ref:
        return []

    parsed_ref = _parse_ref(base_ref)
    if not parsed_ref or parsed_ref.get('type') != 'bible':
        return []

    # Get TOC data from index service
    toc_data = index_service.toc if hasattr(index_service, 'toc') else None
    book_structure = get_book_structure(parsed_ref['book'], toc_data)
    chapter_length_cache: Dict[int, Optional[int]] = {}
    
    generated_chapters = []
    current_chapter = parsed_ref['chapter']
    
    # Try up to e.g., 20 times to find `count` valid chapters
    for _ in range(count * 5):
        if len(generated_chapters) >= count:
            break

        delta = 1 if direction == 'next' else -1
        candidate_chapter = current_chapter + delta
        
        if candidate_chapter < 1:
            break
            
        # Check if the chapter exists by trying to get its length
        chapter_length = await _ensure_chapter_length(
            parsed_ref['book'],
            candidate_chapter,
            book_structure,
            chapter_length_cache,
            sefaria_service,
        )
        
        if not chapter_length:
            break
            
        # Try to load the full chapter
        chapter_ref = f"{parsed_ref['book']} {candidate_chapter}"
        chapter_result = await sefaria_service.get_text(chapter_ref)
        
        if chapter_result.get("ok") and chapter_result.get("data"):
            generated_chapters.append(chapter_result["data"])
            current_chapter = candidate_chapter
        else:
            break
    
    if direction == 'prev':
        generated_chapters.reverse()

    return generated_chapters

def containsHebrew(text: str) -> bool:
    if not text:
        return False
    # Unicode range for Hebrew characters
    for char in text:
        if '\u0590' <= char <= '\u05FF':
            return True
    return False

async def get_text_with_window(ref: str, sefaria_service: SefariaService, index_service: SefariaIndexService, window_size: int = WINDOW_SIZE) -> Optional[Dict[str, Any]]:
    # First, try to get the text to check if it's a range or complete unit
    # Sanitize accidental Talmud-like suffixes in Bible refs
    safe_ref = _coerce_bible_ref_string(ref)
    focus_result = await sefaria_service.get_text(safe_ref)
    if not focus_result.get("ok") or not (focus_data := focus_result.get("data")):
        return None

    # Check if this is a Tanakh reference and if we should load full chapter
    collection = detect_collection(ref)
    parsed_ref = _parse_ref(ref)
    
    # For Tanakh, load full chapter but with performance limits
    if collection == "Bible" and parsed_ref and parsed_ref.get('type') == 'bible':
            chapter_ref = _coerce_bible_ref_string(f"{parsed_ref['book']} {parsed_ref['chapter']}")
            
            # Load the full chapter
            chapter_result = await sefaria_service.get_text(chapter_ref)
            if chapter_result.get("ok") and chapter_result.get("data"):
                chapter_data = chapter_result["data"]
                logger.info(f"[daily] TANAKH FULL CHAPTER: Loading full chapter {chapter_ref}")
                logger.info(f"[daily] TANAKH CHAPTER DATA KEYS: {list(chapter_data.keys())}")
                logger.info(f"[daily] TANAKH CHAPTER TEXT TYPE: {type(chapter_data.get('text'))}")
                logger.info(f"[daily] TANAKH CHAPTER HE TYPE: {type(chapter_data.get('he'))}")
                
                # Create segments for each verse in the chapter
                formatted_segments = []
                
                # Try different field names that Sefaria might use
                text_data = chapter_data.get("text", []) or chapter_data.get("text_segments", [])
                he_data = chapter_data.get("he", []) or chapter_data.get("he_segments", []) or chapter_data.get("he_text", [])
                
                # If text_data is not a list, try to get individual verses
                if not isinstance(text_data, list) or len(text_data) == 0:
                    # Try to load individual verses - get real chapter length
                    logger.info(f"[daily] TANAKH: Chapter data doesn't have verse list, trying individual verses")
                    
                    # Get real chapter length using existing mechanism
                    chapter_length = await _ensure_chapter_length(
                        parsed_ref['book'],
                        parsed_ref['chapter'],
                        None,  # book_structure
                        {},    # cache
                        sefaria_service,
                    )
                    
                    # Use real chapter length when available; otherwise use a high safety guard
                    max_verses = chapter_length if chapter_length else 200
                    logger.debug(f"[daily] TANAKH: Loading up to {max_verses} verses for chapter {parsed_ref['chapter']}")
                    
                    for verse_num in range(1, max_verses + 1):
                        verse_ref = _coerce_bible_ref_string(f"{parsed_ref['book']} {parsed_ref['chapter']}:{verse_num}")
                        try:
                            verse_result = await sefaria_service.get_text(verse_ref)
                            if verse_result.get("ok") and verse_result.get("data"):
                                verse_data = verse_result["data"]
                                en_text = verse_data.get("en_text", "") or verse_data.get("text", "")
                                he_text = verse_data.get("he_text", "") or verse_data.get("he", "")
                                
                                if he_text:  # Only add if there's Hebrew text
                                    formatted_segments.append({
                                        "ref": verse_ref,
                                        "text": he_text,  # Only Hebrew text
                                        "heText": he_text,
                                        "position": 0,  # Will be normalized later
                                        "metadata": {
                                            "title": verse_data.get("title"),
                                            "indexTitle": verse_data.get("indexTitle"),
                                            "chapter": parsed_ref['chapter'],
                                            "verse": verse_num,
                                        }
                                    })
                            else:
                                break  # No more verses
                        except Exception as e:
                            logger.warning(f"Failed to load verse {verse_ref}: {e}")
                            break
                    
                    # If we still don't have segments, something is wrong - log and return None
                    if len(formatted_segments) == 0:
                        logger.error(f"[daily] TANAKH: No segments found for chapter {chapter_ref}")
                        return None
                else:
                    # Use the list data directly with its full length
                    max_verses = len(text_data)
                    logger.debug(f"[daily] TANAKH: Using chapter list data with {max_verses} verses")
                    for i, (en_text, he_text) in enumerate(zip_longest(text_data, (he_data or []), fillvalue="")):
                        verse_num = i + 1
                        segment_ref = _coerce_bible_ref_string(f"{parsed_ref['book']} {parsed_ref['chapter']}:{verse_num}")
                        
                        # Only add if there's Hebrew text
                        if he_text:
                            formatted_segments.append({
                                "ref": segment_ref,
                                "text": he_text,  # Only Hebrew text
                                "heText": he_text,
                                "position": 0,  # Will be normalized later
                                "metadata": {
                                    "title": chapter_data.get("title"),
                                    "indexTitle": chapter_data.get("indexTitle"),
                                    "chapter": parsed_ref['chapter'],
                                    "verse": verse_num,
                                }
                            })
                
                # Normalize positions post-factum
                n = len(formatted_segments)
                for j, seg in enumerate(formatted_segments):
                    seg["position"] = (j / (n - 1)) if n > 1 else 0.5
                
                # Find the focus index for the original verse
                original_verse = parsed_ref.get('verse', 1)
                focus_index = max(0, min(original_verse - 1, len(formatted_segments) - 1))
                
                return {
                    "segments": formatted_segments,
                    "focusIndex": focus_index,
                    "ref": ref,
                    "he_ref": chapter_data.get("heRef"),
                }

    # Regular study mode - use window logic for all references

    # 2. Fetch surrounding segments (original window logic)
    prev_refs_task = _generate_and_validate_refs(ref, collection, "prev", window_size, sefaria_service, index_service)
    next_refs_task = _generate_and_validate_refs(ref, collection, "next", window_size, sefaria_service, index_service)
    prev_segments, next_segments = await asyncio.gather(prev_refs_task, next_refs_task)

    # 3. Assemble the flat list of segments
    all_segments_data = prev_segments + [focus_data] + next_segments
    focus_index = len(prev_segments)

    # 4. Format segments for the frontend, renaming keys to match component props
    formatted_segments = []
    total_segments = len(all_segments_data)
    for i, seg_data in enumerate(all_segments_data):
        formatted_segments.append({
            "ref": seg_data.get("ref"),
            "text": getattr(seg_data, "he_text", "") or seg_data.get("he_text") or "",      # FocusReader показывает только иврит
            "heText": getattr(seg_data, "he_text", "") or seg_data.get("he_text") or "",    # Map he_text to heText
            "position": (i / (total_segments - 1)) if total_segments > 1 else 0.5,
            "metadata": {
                "title": seg_data.get("title"),
                "indexTitle": seg_data.get("indexTitle"),
                "chapter": seg_data.get("chapter"), # Assuming these might exist
                "verse": seg_data.get("verse"),
            }
        })

    # 5. Return the structure expected by the frontend
    return {
        "segments": formatted_segments,
        "focusIndex": focus_index,
        "ref": ref,
        "he_ref": focus_data.get("heRef"),
    }

def _should_load_full_range(ref: str, data: Dict[str, Any]) -> bool:
    """
    Determine if a reference should be loaded as a full range rather than using window logic.
    
    Args:
        ref: The reference string
        data: Sefaria API response data
        
    Returns:
        True if should load full range, False for window logic
    """
    # Check for spanning text (like Talmud daf)
    if data.get("isSpanning"):
        return True
    
    # Check for range references (contain "-" or ":")
    if "-" in ref or (":" in ref and ref.count(":") >= 1):
        # This is likely a range like "Genesis 1:1-10" or "Shabbat 21a:1-21b:5"
        return True
    
    # Check for complete chapter/daf references without specific verse/segment
    # e.g., "Genesis 25", "Zevachim 18"
    if ":" not in ref:
        # This is a complete unit reference - check if it has multiple segments
        text_data = data.get("text", [])
        if isinstance(text_data, list) and len(text_data) > 1:
            return True

    return False

def _extract_text_entry(text_data, index: Optional[int] = None) -> str:
    """Extract a single text string from Sefaria API responses (list or string)."""
    if isinstance(text_data, list) and len(text_data) > 0:
        if index is not None and isinstance(index, int):
            zero_based = max(0, min(len(text_data) - 1, index - 1 if index > 0 else index))
            return text_data[zero_based]
        return text_data[0]
    if isinstance(text_data, str):
        return text_data
    return ""


def _extract_hebrew_text(he_data, index: Optional[int] = None) -> str:
    """Helper function to extract Hebrew text from various data structures."""
    return _extract_text_entry(he_data, index)

def _clean_html_text(text: str) -> str:
    """Clean HTML entities and tags from text."""
    if not text:
        return text
    
    # Decode HTML entities like &nbsp; &amp; &lt; etc.
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up multiple spaces, tabs, and other whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

async def _try_load_range(sefaria_service: SefariaService, ref: str) -> Optional[Dict[str, Any]]:
    """Try to load a range from Sefaria API."""
    try:
        logger.info(f"[daily] TRY_LOAD_RANGE: calling sefaria_service.get_text for '{ref}'")
        result = await sefaria_service.get_text(ref)
        logger.info(f"[daily] TRY_LOAD_RANGE RESULT: ok={result.get('ok')}, has_data={bool(result.get('data'))}")
        logger.info(f"[daily] SEFARIA RESULT: ok={result.get('ok')}, has_data={bool(result.get('data'))}")
        
        if result.get("ok") and result.get("data"):
            return result["data"]
        else:
            logger.warning(f"[daily] FAILED TO LOAD: ref={ref}, result={result}")
            return None
    except Exception as e:
        logger.error(f"[daily] EXCEPTION IN GET_TEXT: ref={ref}, error={str(e)}", exc_info=True)
        return None

async def _handle_jerusalem_talmud_range(
    ref: str,
    sefaria_service: SefariaService,
    session_id: str = None,
    redis_client=None,
) -> Optional[Dict[str, Any]]:
    """Handle Jerusalem Talmud ranges with triple structure (chapter:mishnah:halakhah)."""
    logger.info(f"[daily] HANDLING JERUSALEM TALMUD RANGE: {ref}")

    if "-" not in ref:
        return None

    start_ref, end_ref = ref.split("-", 1)
    start_ref = start_ref.strip()
    end_ref = end_ref.strip()

    def _parse_triple(raw: str) -> Optional[Tuple[int, int, int]]:
        parts = raw.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
            if len(parts) == 2:
                return int(parts[0]), int(parts[1]), 1
            if len(parts) == 1:
                return int(parts[0]), 1, 1
        except ValueError:
            return None
        return None

    try:
        book_part, start_numbers = start_ref.rsplit(" ", 1)
    except ValueError:
        logger.warning("[daily] JERUSALEM RANGE START UNPARSEABLE", extra={"ref": ref})
        return None
    start_triple = _parse_triple(start_numbers)
    if not start_triple:
        logger.warning("[daily] JERUSALEM RANGE START UNPARSEABLE", extra={"ref": ref})
        return None

    if " " in end_ref:
        end_book_part, end_numbers = end_ref.rsplit(" ", 1)
    else:
        end_book_part, end_numbers = book_part, end_ref
    end_triple = _parse_triple(end_numbers)
    if not end_triple:
        logger.warning("[daily] JERUSALEM RANGE END UNPARSEABLE", extra={"ref": ref})
        return None

    book_name = end_book_part.strip()
    logger.info(
        "[daily] JERUSALEM TALMUD RANGE PARSED",
        extra={
            "ref": ref,
            "book": book_name,
            "start": start_triple,
            "end": end_triple,
        },
    )

    segments: List[Dict[str, Any]] = []

    def triple_leq(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
        return a <= b

    current = start_triple
    done = False

    while not done:
        chapter, mishnah, halakhah = current

        mishnah_ref = f"{book_name} {chapter}:{mishnah}"
        try:
            mishnah_result = await sefaria_service.get_text(mishnah_ref)
        except Exception as exc:
            logger.warning(
                "[daily] JERUSALEM TALMUD MISHNAH FETCH FAILED",
                extra={"ref": mishnah_ref, "error": str(exc)},
            )
            break

        if not (mishnah_result.get("ok") and mishnah_result.get("data")):
            logger.debug(
                "[daily] JERUSALEM TALMUD MISHNAH EMPTY",
                extra={"ref": mishnah_ref, "response": mishnah_result},
            )
            break

        data = mishnah_result["data"]
        text_values = data.get("text", []) or data.get("en_text", [])
        he_values = data.get("he", []) or data.get("he_text", [])

        if isinstance(text_values, str):
            text_values = [text_values]
        if isinstance(he_values, str):
            he_values = [he_values]

        max_halakhot = max(len(text_values), len(he_values), 1)

        for hal_index in range(1, max_halakhot + 1):
            triple = (chapter, mishnah, hal_index)
            if not triple_leq(start_triple, triple):
                continue
            if not triple_leq(triple, end_triple):
                done = True
                break

            en_text = _clean_html_text(_extract_text_entry(text_values, hal_index))
            he_text = _clean_html_text(_extract_hebrew_text(he_values, hal_index))

            if not (en_text or he_text):
                continue

            segment_ref = f"{book_name} {chapter}:{mishnah}:{hal_index}"
            metadata = {
                "title": data.get("title", ref),
                "indexTitle": data.get("indexTitle", ""),
                "heRef": data.get("heRef", ""),
                "chapter": chapter,
                "mishnah": mishnah,
                "halakhah": hal_index,
                "enText": en_text,
            }

            segments.append(
                {
                    "ref": segment_ref,
                    "text": he_text or en_text,
                    "heText": he_text or en_text,
                    "enText": en_text,
                    "metadata": metadata,
                }
            )
            logger.debug(
                "[daily] JERUSALEM TALMUD SEGMENT ADDED",
                extra={"segment_ref": segment_ref},
            )

        if done:
            break

        if triple_leq((chapter, mishnah + 1, 1), end_triple):
            current = (chapter, mishnah + 1, 1)
        else:
            current = (chapter + 1, 1, 1)

        if not triple_leq(current, end_triple):
            break

    if not segments:
        logger.warning("[daily] NO SEGMENTS LOADED FOR JERUSALEM TALMUD RANGE", extra={"ref": ref})
        return None

    total_segments = len(segments)
    for idx, segment in enumerate(segments):
        position = idx / (total_segments - 1) if total_segments > 1 else 0.5
        segment["position"] = float(position)

    if session_id and redis_client:
        try:
            count_key = f"daily:sess:{session_id}:total_segments"
            await redis_client.set(count_key, total_segments, ex=3600 * 24 * 7)
        except Exception as exc:
            logger.warning(
                "[daily] JERUSALEM TALMUD REDIS TOTAL FAILED",
                extra={"ref": ref, "session_id": session_id, "error": str(exc)},
            )

    return {
        "segments": segments,
        "focusIndex": 0,
        "totalLength": total_segments,
        "ref": ref,
        "loadedAt": str(int(time.time() * 1000)),
        "he_ref": segments[0]["metadata"].get("heRef") if segments else None,
    }

async def _handle_inter_chapter_range(ref: str, sefaria_service: SefariaService, session_id: str = None, redis_client = None) -> Optional[Dict[str, Any]]:
    """Handle inter-chapter ranges by loading each chapter separately."""
    logger.info(f"[daily] HANDLING INTER-CHAPTER RANGE: {ref}")
    
    # Parse the range
    start_ref, end_ref = ref.split("-", 1)
    start_chapter_verse = start_ref.split(":")
    end_chapter_verse = end_ref.split(":")
    
    # Extract book name and chapter numbers
    book_name = " ".join(start_chapter_verse[:-2])  # Everything except last two parts
    start_chapter = int(start_chapter_verse[-2])
    start_verse = int(start_chapter_verse[-1])
    end_chapter = int(end_chapter_verse[-2])
    end_verse = int(end_chapter_verse[-1])
    
    logger.info(f"[daily] INTER-CHAPTER PARSED: {book_name}, {start_chapter}:{start_verse} to {end_chapter}:{end_verse}")
    
    all_segments_data = []
    
    # Load verses from start chapter
    for verse_num in range(start_verse, 1000):  # Go up to 1000 verses per chapter
        verse_ref = f"{book_name} {start_chapter}:{verse_num}"
        try:
            verse_result = await sefaria_service.get_text(verse_ref)
            if verse_result.get("ok") and verse_result.get("data"):
                verse_data = verse_result["data"]
                en_text = verse_data.get("en_text", "") or verse_data.get("text", "")
                he_text = verse_data.get("he_text", "") or verse_data.get("he", "")
                
                if en_text or he_text:
                    english_text = _clean_html_text(_extract_text_entry(en_text, verse_num))
                    hebrew_text = _clean_html_text(_extract_hebrew_text(he_text, verse_num))
                    segment_data = {
                        "ref": verse_ref,
                        "text": hebrew_text,
                        "heText": hebrew_text,
                        "enText": english_text,
                        "title": verse_data.get("title", ref),
                        "indexTitle": verse_data.get("indexTitle", ""),
                        "heRef": verse_data.get("heRef", ""),
                        "chapter": start_chapter,
                        "verse": verse_num,
                        "metadata": {
                            "title": verse_data.get("title", ref),
                            "indexTitle": verse_data.get("indexTitle", ""),
                            "heRef": verse_data.get("heRef", ""),
                            "chapter": start_chapter,
                            "verse": verse_num,
                            "enText": english_text,
                        },
                    }
                    all_segments_data.append(segment_data)
                    logger.info(f"[daily] INTER-CHAPTER VERSE: {verse_ref}")
                else:
                    break  # No more verses in this chapter
            else:
                break  # No more verses in this chapter
        except Exception as e:
            logger.error(f"[daily] ERROR FETCHING INTER-CHAPTER VERSE {verse_ref}: {str(e)}")
            break
    
    # Load verses from end chapter
    if end_chapter > start_chapter:
        for verse_num in range(1, end_verse + 1):
            verse_ref = f"{book_name} {end_chapter}:{verse_num}"
            try:
                verse_result = await sefaria_service.get_text(verse_ref)
                if verse_result.get("ok") and verse_result.get("data"):
                    verse_data = verse_result["data"]
                    en_text = verse_data.get("en_text", "") or verse_data.get("text", "")
                    he_text = verse_data.get("he_text", "") or verse_data.get("he", "")
                    
                    if en_text or he_text:
                        english_text = _clean_html_text(_extract_text_entry(en_text, verse_num))
                        hebrew_text = _clean_html_text(_extract_hebrew_text(he_text, verse_num))
                        segment_data = {
                            "ref": verse_ref,
                            "text": hebrew_text,
                            "heText": hebrew_text,
                            "enText": english_text,
                            "title": verse_data.get("title", ref),
                            "indexTitle": verse_data.get("indexTitle", ""),
                            "heRef": verse_data.get("heRef", ""),
                            "chapter": end_chapter,
                            "verse": verse_num,
                            "metadata": {
                                "title": verse_data.get("title", ref),
                                "indexTitle": verse_data.get("indexTitle", ""),
                                "heRef": verse_data.get("heRef", ""),
                                "chapter": end_chapter,
                                "verse": verse_num,
                                "enText": english_text,
                            },
                        }
                        all_segments_data.append(segment_data)
                        logger.info(f"[daily] INTER-CHAPTER VERSE: {verse_ref}")
                    else:
                        break
                else:
                    break
            except Exception as e:
                logger.error(f"[daily] ERROR FETCHING INTER-CHAPTER VERSE {verse_ref}: {str(e)}")
                break
    
    if not all_segments_data:
        logger.warning(f"[daily] NO SEGMENTS LOADED FOR INTER-CHAPTER RANGE: {ref}")
        return None
    
    # Format segments for frontend
    formatted_segments = []
    total_segments = len(all_segments_data)
    for i, seg_data in enumerate(all_segments_data):
        formatted_segments.append({
            "ref": seg_data.get("ref"),
            "text": getattr(seg_data, "he_text", "") or seg_data.get("he_text") or "",
            "heText": getattr(seg_data, "he_text", "") or seg_data.get("he_text") or "",
            "position": (i / (total_segments - 1)) if total_segments > 1 else 0.5,
            "metadata": {
                "title": seg_data.get("title"),
                "indexTitle": seg_data.get("indexTitle"),
                "heRef": seg_data.get("heRef")
            }
        })
    
    logger.info(f"[daily] INTER-CHAPTER RANGE COMPLETE: loaded {len(formatted_segments)} segments")
    
    # Save total_segments to Redis if session_id and redis_client are provided
    if session_id and redis_client:
        try:
            count_key = f"daily:sess:{session_id}:total_segments"
            await redis_client.set(count_key, total_segments, ex=3600*24*7)  # 7 days TTL
            logger.info(f"[daily] SAVED TOTAL SEGMENTS: {total_segments} for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to save total_segments to Redis: {e}")
    
    return {
        "segments": formatted_segments,
        "focusIndex": 0,
        "ref": ref,
        "he_ref": all_segments_data[0].get("heRef") if all_segments_data else None,
    }

async def get_full_daily_text(ref: str, sefaria_service: SefariaService, index_service: SefariaIndexService, session_id: str = None, redis_client = None) -> Optional[Dict[str, Any]]:
    """
    Load the full daily text for a given reference and segment it properly.
    
    This function handles different types of daily study texts:
    - Talmud: "Zevachim 18" -> load entire daf (18a:1, 18a:2, ... 18b:4)
    - Tanakh: "Genesis 25" -> load entire chapter (25:1, 25:2, ... 25:34)
    - Ranges: "Deuteronomy 32:1-52" -> load specified range
    - Mishnah: "Mishnah Menachot 9:5-6" -> load specified mishnayot
    """
    logger.info(f"[daily] GET_FULL_DAILY_TEXT STARTED: ref='{ref}'")
    collection = detect_collection(ref)
    
    # Check if this is an inter-chapter range first
    if "-" in ref and ":" in ref:
        # Parse the range to check if it's inter-chapter
        start_ref, end_ref = ref.split("-", 1)
        if ":" in start_ref and ":" in end_ref:
            start_chapter_verse = start_ref.split(":")
            end_chapter_verse = end_ref.split(":")
            
            if len(start_chapter_verse) >= 2 and len(end_chapter_verse) >= 2:
                # Try to extract chapter numbers safely
                try:
                    start_chapter = int(start_chapter_verse[-2]) if len(start_chapter_verse) > 1 and start_chapter_verse[-2].isdigit() else None
                    end_chapter = int(end_chapter_verse[-2]) if len(end_chapter_verse) > 1 and end_chapter_verse[-2].isdigit() else None
                except (ValueError, IndexError):
                    start_chapter = None
                    end_chapter = None
                
                if start_chapter and end_chapter and start_chapter != end_chapter:
                    logger.info(f"[daily] DETECTED INTER-CHAPTER RANGE: {ref} ({start_chapter} -> {end_chapter})")
                    # Handle inter-chapter range directly without trying to load the full range
                    data = None
                else:
                    # Same chapter range, try to load normally
                    data = await _try_load_range(sefaria_service, ref)
            else:
                # Not a proper chapter:verse format, try to load normally
                data = await _try_load_range(sefaria_service, ref)
        else:
            # Not a verse range, try to load normally
            data = await _try_load_range(sefaria_service, ref)
    else:
        # Not a range, try to load normally
        data = await _try_load_range(sefaria_service, ref)
    
    if data is None:
        # Check if this is Jerusalem Talmud (triple structure: Chapter:Mishnah:Halakhah)
        if "jerusalem talmud" in ref.lower() and ":" in ref and "-" in ref:
            logger.info(f"[daily] HANDLING JERUSALEM TALMUD RANGE: {ref}")
            return await _handle_jerusalem_talmud_range(ref, sefaria_service, session_id, redis_client)
        
        talmud_payload = await _build_talmud_payload(
            ref, None, sefaria_service, session_id=session_id, redis_client=redis_client
        )
        if talmud_payload:
            return talmud_payload

        # Check if this is an inter-chapter range that we detected earlier
        if "-" in ref and ":" in ref:
            start_ref, end_ref = ref.split("-", 1)
            start_ref = start_ref.strip()
            end_ref = end_ref.strip()

            start_info = _parse_ref(start_ref)
            end_candidate = end_ref
            if start_info and end_ref and not any(ch.isalpha() for ch in end_ref.split(":", 1)[0]):
                book_name = (start_info.get("book") or "").strip()
                end_candidate = f"{book_name} {end_ref}".strip()
            end_info = _parse_ref(end_candidate)

            start_chapter = start_info.get("chapter") if start_info else None
            end_chapter = end_info.get("chapter") if end_info else None

            logger.debug(
                "daily.range.same_chapter.inspect",
                extra={
                    "ref": ref,
                    "start_ref": start_ref,
                    "end_ref": end_ref,
                    "start_info": start_info,
                    "end_candidate": end_candidate,
                    "end_info": end_info,
                },
            )

            if start_chapter is not None and end_chapter is not None:
                if start_chapter != end_chapter:
                    return await _handle_inter_chapter_range(ref, sefaria_service, session_id, redis_client)
                else:
                    same_chapter_payload = await _handle_same_chapter_range(
                        ref,
                        start_info,
                        end_info,
                        sefaria_service,
                        session_id=session_id,
                        redis_client=redis_client,
                    )
                    if same_chapter_payload:
                        return same_chapter_payload
        
        logger.warning(f"[daily] FAILED TO LOAD: ref={ref}")
        return None
    
    # Debug language content
    if _should_delegate_to_modular(ref, data):
        try:
            from brain_service.services.study.daily_text import build_full_daily_text as modular_builder
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("[daily] modular_builder_unavailable", extra={"error": str(exc)})
        else:
            modular_payload = await modular_builder(
                ref,
                sefaria_service,
                index_service,
                session_id=session_id,
                redis_client=redis_client,
            )
            if modular_payload:
                return modular_payload

    talmud_payload = await _build_talmud_payload(
        ref,
        data if isinstance(data, dict) else {},
        sefaria_service,
        session_id=session_id,
        redis_client=redis_client,
    )
    if talmud_payload:
        return talmud_payload

    text_data = data.get("text_segments") or data.get("text", [])
    he_data = data.get("he_segments") or data.get("he", [])
    en_text = getattr(data, "en_text", "") or data.get("en_text", "")
    he_text = getattr(data, "he_text", "") or data.get("he_text", "")
    logger.info(f"[daily] LANGUAGE DEBUG: ref={ref}")
    logger.info(f"[daily] TEXT TYPE: text={type(text_data)}, he={type(he_data)}")
    en_len = len(en_text) if isinstance(en_text, (str, list, tuple)) else 0
    he_len = len(he_text) if isinstance(he_text, (str, list, tuple)) else 0
    logger.info(f"[daily] COMPACT TEXT: en_text={bool(en_text)} (len={en_len}), he_text={bool(he_text)} (len={he_len})")
    if en_text:
        logger.info(f"[daily] EN_TEXT PREVIEW: {en_text[:100]}...")
    if he_text:
        logger.info(f"[daily] HE_TEXT PREVIEW: [Hebrew text - {he_len} chars]")
    else:
        logger.info(f"[daily] HE_TEXT IS EMPTY - checking he_data type: {type(he_data)}")
        if isinstance(he_data, list) and len(he_data) > 0:
            logger.info(f"[daily] HE_DATA LIST[0]: {he_data[0][:100] if he_data[0] else 'EMPTY'}...")
    if isinstance(text_data, list) and len(text_data) > 0:
        logger.info(f"[daily] SAMPLE TEXT[0]: en='{text_data[0][:100] if text_data[0] else 'EMPTY'}...'")
    if isinstance(he_data, list) and len(he_data) > 0:
        logger.info(f"[daily] SAMPLE HE[0]: he='{he_data[0][:100] if he_data[0] else 'EMPTY'}...'")
    elif isinstance(he_data, str):
        logger.info(f"[daily] SAMPLE HE STRING: he='{he_data[:100]}...'")
    elif isinstance(text_data, str):
        logger.info(f"[daily] SAMPLE TEXT STRING: en='{text_data[:100]}...'")
        he_string = data.get("he", "")
        logger.info(f"[daily] SAMPLE HE STRING: he='{he_string[:100] if he_string else 'EMPTY'}...'")
    else:
        logger.warning(f"[daily] UNEXPECTED DATA STRUCTURE: text={text_data}, he={he_data}")
    all_segments_data = []
    focus_index = 0
    
    # Check if this is a spanning text (covers multiple sections like 18a-18b)
    if data.get("isSpanning") or isinstance(data.get("text"), list) and len(data["text"]) > 0 and isinstance(data["text"][0], list):
        # This is structured text with nested arrays for different sections
        text_sections = data.get("text", [])
        he_sections = data.get("he", [])
        spanning_refs = data.get("spanningRefs", [])
        
        logger.info(f"[daily] SPANNING TEXT: {ref} has {len(text_sections)} sections, isSpanning: {data.get('isSpanning')}")
        logger.info(f"[daily] TEXT STRUCTURE: text={type(data.get('text'))}, he={type(data.get('he'))}")
        logger.info(f"[daily] SPANNING REFS: {spanning_refs}")
        
        for section_idx, (text_section, he_section) in enumerate(zip(text_sections, he_sections)):
            # Each section represents one amud (side) of the daf
            base_ref = spanning_refs[section_idx] if section_idx < len(spanning_refs) else f"{ref}#{section_idx}"
            
            logger.info(f"[daily] SECTION {section_idx}: base_ref={base_ref}, segments={len(text_section)}")
            
            # Process each segment within this section
            for segment_idx, (en_text, he_text) in enumerate(zip(text_section, he_section)):
                # Generate ref for this specific segment
                if ":" in base_ref:
                    # Already has segment notation
                    segment_ref = f"{base_ref.split(':')[0]}:{segment_idx + 1}"
                else:
                    # Add segment notation
                    segment_ref = f"{base_ref}:{segment_idx + 1}"
                
                segment_data = {
                    "ref": segment_ref,
                    "en_text": en_text,
                    "he_text": _extract_hebrew_text(he_text),
                    "title": data.get("title", ref),
                    "indexTitle": data.get("indexTitle", data.get("book", "")),
                    "heRef": data.get("heRef", "")
                }
                
                all_segments_data.append(segment_data)
                logger.info(f"[daily] SEGMENT: {segment_ref}, en_len={len(en_text) if en_text else 0}, he_len={len(he_text) if he_text else 0}")
                
                # Set focus to first segment
                if len(all_segments_data) == 1:
                    focus_index = 0
    
    else:
        # This is a simple text or range - check if it's an array of segments
        text_data = data.get("text", [])
        he_data = data.get("he", [])
        
        if isinstance(text_data, list) and len(text_data) > 0:
            logger.info(f"[daily] SIMPLE/RANGE TEXT: {ref} has {len(text_data)} segments")
            
            # Parse the base reference to generate individual segment refs
            base_ref = data.get("ref", ref)
            
            # Handle different reference formats
            if "-" in base_ref:
                # This is a range like "Deuteronomy 32:1-52" or "Arukh HaShulchan, Orach Chaim 162:28-164:3"
                # Since Sefaria returns range as single text, we need to fetch each verse individually
                start_ref, end_ref = base_ref.split("-", 1)  # Split only on first "-"
                
                # Extract book, chapter, start and end verses
                if ":" in start_ref:
                    # Handle verse ranges like "Deuteronomy 32:1-52" or "Arukh HaShulchan, Orach Chaim 162:28-164:3"
                    book_chapter = start_ref.rsplit(":", 1)[0]  # "Deuteronomy 32" or "Arukh HaShulchan, Orach Chaim 162"
                    start_verse = int(start_ref.split(":")[-1])  # 1 or 28
                    
                    if ":" in end_ref:
                        # Full reference in end_ref: "Deuteronomy 32:52" or "164:3"
                        # Check if this is an inter-chapter range by comparing chapter numbers
                        end_chapter_verse = end_ref.split(":")
                        end_chapter = int(end_chapter_verse[0])  # 164 or 77
                        start_chapter = int(book_chapter.split()[-1])  # 162 or 76
                        
                        if end_chapter != start_chapter:
                            # Inter-chapter range like "162:28-164:3" or "76:7-77:1"
                            end_verse = int(end_chapter_verse[1])    # 3 or 1
                            
                            # For inter-chapter ranges, we need to handle each chapter separately
                            
                            logger.info(f"[daily] INTER-CHAPTER RANGE: {start_chapter}:{start_verse} to {end_chapter}:{end_verse}")
                            
                            # Fetch verses from start chapter (162:28 to end of chapter)
                            for verse_num in range(start_verse, 1000):  # Go up to 1000 verses per chapter
                                verse_ref = f"{book_chapter}:{verse_num}"
                                try:
                                    verse_result = await sefaria_service.get_text(verse_ref)
                                    if verse_result.get("ok") and verse_result.get("data"):
                                        verse_data = verse_result["data"]
                                        en_text = verse_data.get("text", "")
                                        he_text = verse_data.get("he", "")
                                        
                                        if en_text or he_text:
                                            segment_data = {
                                                "ref": verse_ref,
                                                "text": _clean_html_text(_extract_hebrew_text(he_text, verse_num)),
                                                "heText": _clean_html_text(_extract_hebrew_text(he_text, verse_num)),
                                                "position": 0,  # normalize later
                                                "metadata": {
                                                    "title": data.get("title", ref),
                                                    "indexTitle": data.get("indexTitle", data.get("book", "")),
                                                    "heRef": verse_data.get("heRef", "")
                                                }
                                            }
                                            all_segments_data.append(segment_data)
                                            logger.info(f"[daily] INTER-CHAPTER VERSE: {verse_ref}, en_len={len(en_text)}, he_len={len(_extract_hebrew_text(he_text, verse_num))}")
                                        else:
                                            break  # No more verses in this chapter
                                    else:
                                        break  # No more verses in this chapter
                                except Exception as e:
                                    logger.error(f"[daily] ERROR FETCHING INTER-CHAPTER VERSE {verse_ref}: {str(e)}")
                                    break
                            
                            # Fetch verses from end chapter
                            if end_chapter > start_chapter:
                                end_book_chapter = book_chapter.rsplit(" ", 1)[0] + f" {end_chapter}"  # "Arukh HaShulchan, Orach Chaim 164"
                                for verse_num in range(1, end_verse + 1):
                                    verse_ref = f"{end_book_chapter}:{verse_num}"
                                    try:
                                        verse_result = await sefaria_service.get_text(verse_ref)
                                        if verse_result.get("ok") and verse_result.get("data"):
                                            verse_data = verse_result["data"]
                                            en_text = verse_data.get("text", "")
                                            he_text = verse_data.get("he", "")
                                            
                                            if en_text or he_text:
                                                segment_data = {
                                                    "ref": verse_ref,
                                                    "text": _clean_html_text(_extract_hebrew_text(he_text, verse_num)),
                                                    "heText": _clean_html_text(_extract_hebrew_text(he_text, verse_num)),
                                                    "position": 0,  # normalize later
                                                    "metadata": {
                                                        "title": data.get("title", ref),
                                                        "indexTitle": data.get("indexTitle", data.get("book", "")),
                                                        "heRef": verse_data.get("heRef", "")
                                                    }
                                                }
                                                all_segments_data.append(segment_data)
                                                logger.info(f"[daily] INTER-CHAPTER VERSE: {verse_ref}, en_len={len(en_text)}, he_len={len(_extract_hebrew_text(he_text, verse_num))}")
                                            else:
                                                break
                                        else:
                                            break
                                    except Exception as e:
                                        logger.error(f"[daily] ERROR FETCHING INTER-CHAPTER VERSE {verse_ref}: {str(e)}")
                                        break
                            
                            if len(all_segments_data) > 0:
                                focus_index = 0
                                return {
                                    "segments": all_segments_data,
                                    "focusIndex": focus_index,
                                    "totalLength": len(all_segments_data),
                                    "ref": ref,
                                    "loadedAt": str(int(time.time() * 1000))
                                }
                        else:
                            # Same chapter range: "Deuteronomy 32:52"
                            end_verse = int(end_ref.split(":")[-1])     # 52
                    else:
                        # Just verse number in end_ref: "52"
                        end_verse = int(end_ref.strip())            # 52
                    
                    logger.info(f"[daily] FETCHING RANGE INDIVIDUALLY: {book_chapter}, verses {start_verse}-{end_verse}")
                    
                    # Load first segments for immediate display, then load rest dynamically
                    total_segments = end_verse - start_verse + 1
                    
                    # For large ranges, load more segments initially for better UX
                    if collection == "Bible":
                        segments_to_load = total_segments  # Tanakh sessions need full chapter upfront
                    elif total_segments <= 10:
                        segments_to_load = total_segments  # Load all for small ranges
                    elif total_segments <= 50:
                        segments_to_load = 10  # Load 10 for medium ranges
                    else:
                        segments_to_load = 20  # Load 20 for large ranges
                    
                    logger.info(f"[daily] DAILY MODE: Preparing first {segments_to_load} segments from range {start_verse}-{end_verse} (total: {total_segments})")
                    
                    chapter_match = re.search(r"(\d+)$", book_chapter.strip())
                    chapter_num = int(chapter_match.group(1)) if chapter_match else parsed_ref['chapter']
                    
                    for i in range(segments_to_load):
                        verse_num = start_verse + i
                        if verse_num > end_verse:
                            break
                        relative_index = verse_num - start_verse
                        he_text_entry = _extract_hebrew_text(he_data, relative_index + 1)
                        en_text_entry = _extract_text_entry(text_data, relative_index + 1)

                        # If the API returned a blob instead of per-verse content, fetch the verse directly
                        needs_direct_fetch = (
                            isinstance(he_data, str)
                            or isinstance(text_data, str)
                            or (not he_text_entry and not en_text_entry)
                        )

                        verse_ref = f"{book_chapter}:{verse_num}"

                        if needs_direct_fetch:
                            try:
                                verse_result = await sefaria_service.get_text(verse_ref)
                                if verse_result.get("ok") and verse_result.get("data"):
                                    verse_data = verse_result["data"]
                                    ve_text = verse_data.get("text", "") or verse_data.get("en_text", "")
                                    ve_he = verse_data.get("he", "") or verse_data.get("he_text", "")
                                    if ve_text:
                                        en_text_entry = (
                                            _extract_text_entry(ve_text, None)
                                            if isinstance(ve_text, list)
                                            else ve_text
                                        )
                                    if ve_he:
                                        he_text_entry = (
                                            _extract_hebrew_text(ve_he, None)
                                            if isinstance(ve_he, list)
                                            else ve_he
                                        )
                                else:
                                    logger.debug(f"?? DAILY MODE: Direct fetch for {verse_ref} returned no data")
                            except Exception as fetch_exc:
                                logger.warning(f"?? DAILY MODE: Failed direct fetch for {verse_ref}: {fetch_exc}")

                        if not he_text_entry and not en_text_entry:
                            logger.debug(f"?? DAILY MODE: No content for verse {verse_ref}, stopping range expansion")
                            break
                        
                        hebrew_text = _clean_html_text(he_text_entry)
                        english_text = _clean_html_text(en_text_entry)
                        segment_data = {
                            "ref": verse_ref,
                            "text": hebrew_text,
                            "heText": hebrew_text,
                            "enText": english_text,
                            "position": 0,  # normalize later
                            "metadata": {
                                "title": data.get("title", ref),
                                "indexTitle": data.get("indexTitle", data.get("book", "")),
                                "heRef": data.get("heRef", ""),
                                "chapter": chapter_num,
                                "verse": verse_num,
                                "enText": english_text,
                            }
                        }
                        all_segments_data.append(segment_data)
                        logger.info(f"[daily] DAILY SEGMENT {i+1}/{segments_to_load}: {verse_ref}")
                    
                    # Normalize positions post-factum for the initially loaded subset
                    n_partial = len(all_segments_data)
                    for j, seg in enumerate(all_segments_data):
                        seg["position"] = (j / (n_partial - 1)) if n_partial > 1 else 0.5

                    logger.info(f"[daily] DAILY MODE COMPLETE: loaded {len(all_segments_data)} segments, total range: {end_verse - start_verse + 1} verses")
                        
                elif ":" not in start_ref and ":" not in end_ref:
                    # Range like "Genesis 1-3" (chapters)
                    book = start_ref.rsplit(" ", 1)[0]  # "Genesis"
                    start_chapter = int(start_ref.split(" ")[-1])  # 1
                    end_chapter = int(end_ref)  # 3
                    
                    # For chapter ranges, treat each text element as a verse within chapters
                    current_chapter = start_chapter
                    current_verse = 1
                    
                    for i, (en_text, he_text) in enumerate(zip(text_data, he_data)):
                        segment_ref = f"{book} {current_chapter}:{current_verse}"
                        
                        segment_data = {
                            "ref": segment_ref,
                            "en_text": en_text,
                            "he_text": _extract_hebrew_text(he_text),
                            "title": data.get("title", ref),
                            "indexTitle": data.get("indexTitle", data.get("book", "")),
                            "heRef": data.get("heRef", "")
                        }
                        all_segments_data.append(segment_data)
                        current_verse += 1
                        
            elif ":" not in base_ref and " " in base_ref:
                # Single chapter like "Genesis 25" or "Mishneh Torah, Creditor and Debtor 12"
                book_chapter = base_ref  # "Genesis 25" or "Mishneh Torah, Creditor and Debtor 12"
                
                # Check if this is Mishneh Torah (needs special handling)
                if "Mishneh Torah" in base_ref:
                    logger.info(f"[daily] DETECTED MISHNEH TORAH: {base_ref}")
                    # For Mishneh Torah, try to fetch individual halakhot (sub-chapters)
                    # Extract the chapter number
                    chapter_num = base_ref.split()[-1]  # "12"
                    book_part = " ".join(base_ref.split()[:-1])  # "Mishneh Torah, Creditor and Debtor"
                    
                    # Try to fetch individual halakhot (1, 2, 3, etc.)
                    for halakha_num in range(1, 20):  # Try up to 20 halakhot
                        halakha_ref = f"{book_part} {chapter_num}:{halakha_num}"
                        try:
                            halakha_result = await sefaria_service.get_text(halakha_ref)
                            if halakha_result.get("ok") and halakha_result.get("data"):
                                halakha_data = halakha_result["data"]
                                en_text = halakha_data.get("text", "")
                                he_text = halakha_data.get("he", "")
                                
                                if en_text or he_text:  # Only add if we got content
                                    segment_data = {
                                        "ref": halakha_ref,
                                        "en_text": en_text,
                                        "he_text": _extract_hebrew_text(he_text),
                                        "title": data.get("title", ref),
                                        "indexTitle": data.get("indexTitle", data.get("book", "")),
                                        "heRef": halakha_data.get("heRef", "")
                                    }
                                    all_segments_data.append(segment_data)
                                    logger.info(f"[daily] MISHNEH TORAH HALAKHA: {halakha_ref}, en_len={len(en_text)}, he_len={len(he_text) if he_text else 0}")
                                else:
                                    logger.info(f"[daily] MISHNEH TORAH HALAKHA EMPTY: {halakha_ref}")
                            else:
                                logger.info(f"[daily] MISHNEH TORAH HALAKHA NOT FOUND: {halakha_ref}")
                                break  # Stop if we can't find more halakhot
                        except Exception as e:
                            logger.error(f"[daily] ERROR FETCHING MISHNEH TORAH HALAKHA {halakha_ref}: {str(e)}")
                            break
                else:
                    # Regular chapter segmentation (like Genesis 25)
                    for i, (en_text, he_text) in enumerate(zip(text_data, he_data)):
                        verse_num = i + 1
                        segment_ref = f"{book_chapter}:{verse_num}"
                        
                        segment_data = {
                            "ref": segment_ref,
                            "en_text": en_text,
                            "he_text": _extract_hebrew_text(he_text),
                            "title": data.get("title", ref),
                            "indexTitle": data.get("indexTitle", data.get("book", "")),
                            "heRef": data.get("heRef", "")
                        }
                        all_segments_data.append(segment_data)
                    logger.info(f"[daily] CHAPTER SEGMENT: {segment_ref}")
                    
            else:
                # Fallback: use original ref for all segments
                for i, (en_text, he_text) in enumerate(zip(text_data, he_data)):
                    segment_ref = f"{base_ref}:{i + 1}" if ":" not in base_ref else base_ref
                    
                    segment_data = {
                        "ref": segment_ref,
                        "en_text": en_text,
                        "he_text": _extract_hebrew_text(he_text),
                        "title": data.get("title", ref),
                        "indexTitle": data.get("indexTitle", data.get("book", "")),
                        "heRef": data.get("heRef", "")
                    }
                    all_segments_data.append(segment_data)
                    
            focus_index = 0
            
        else:
            # Single text string - check if this should be a range or full tractate page
            logger.info(f"[daily] SINGLE TEXT: {ref}")
            
            # Check if we have CompactText data (en_text/he_text)
            en_len = len(en_text) if isinstance(en_text, (str, list, tuple)) else 0
            he_len = len(he_text) if isinstance(he_text, (str, list, tuple)) else 0
            logger.info(f"[daily] CHECKING COMPACT TEXT: en_text={bool(en_text)} (len={en_len}), he_text={bool(he_text)} (len={he_len})")
            
            # If CompactText data is available, use it
            if en_text or he_text:
                logger.info(f"[daily] COMPACT TEXT DATA FOUND: en_text={bool(en_text)}, he_text={bool(he_text)}")
                # Create single segment from CompactText data
                segment_data = {
                    "ref": ref,
                    "en_text": en_text,
                    "he_text": he_text,
                    "title": data.get("title", ref),
                    "indexTitle": data.get("indexTitle", data.get("book", "")),
                    "heRef": data.get("heRef", "")
                }
                
                # Format for frontend - FocusReader показывает только иврит
                formatted_segments = [{
                    "ref": ref,
                    "text": he_text or "",  # Основной текст на иврите
                    "heText": he_text or "",  # Дублируем для совместимости
                    "position": 0.5,
                    "metadata": {
                        "title": data.get("title", ref),
                        "indexTitle": data.get("indexTitle", data.get("book", "")),
                        "heRef": data.get("heRef", "")
                    }
                }]
                
                en_len = len(en_text) if isinstance(en_text, (str, list, tuple)) else 0
                he_len = len(he_text) if isinstance(he_text, (str, list, tuple)) else 0
                logger.info(f"[daily] COMPACT TEXT SEGMENT CREATED: {ref} - EN: {en_len} chars, HE: {he_len} chars")
                
                # Save total_segments to Redis if session_id and redis_client are provided
                if session_id and redis_client:
                    try:
                        count_key = f"daily:sess:{session_id}:total_segments"
                        await redis_client.set(count_key, 1, ex=3600*24*7)  # 7 days TTL
                        logger.info(f"[daily] SAVED TOTAL SEGMENTS: 1 for session {session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save total_segments to Redis: {e}")
                
                return {
                    "segments": formatted_segments,
                    "focusIndex": 0,
                    "ref": ref,
                    "he_ref": data.get("heRef", ""),
                }
            
            # If CompactText data is not available, try to process he_data as list
            elif isinstance(he_data, list) and len(he_data) > 0 and he_data[0]:
                logger.info(f"[daily] PROCESSING HE_DATA AS LIST: {len(he_data)} items")
                he_text_from_list = _clean_html_text(_extract_hebrew_text(he_data[0]))
                
                if he_text_from_list:
                    logger.info(f"[daily] HE_TEXT FROM LIST: {len(he_text_from_list)} chars")
                    # Format for frontend - FocusReader показывает только иврит
                    formatted_segments = [{
                        "ref": ref,
                        "text": he_text_from_list,  # Основной текст на иврите
                        "heText": he_text_from_list,  # Дублируем для совместимости
                        "position": 0.5,
                        "metadata": {
                            "title": data.get("title", ref),
                            "indexTitle": data.get("indexTitle", data.get("book", "")),
                            "heRef": data.get("heRef", "")
                        }
                    }]
                    
                    logger.info(f"[daily] HE_DATA SEGMENT CREATED: {ref} - HE: {len(he_text_from_list)} chars")
                    
                    # Save total_segments to Redis if session_id and redis_client are provided
                    if session_id and redis_client:
                        try:
                            count_key = f"daily:sess:{session_id}:total_segments"
                            await redis_client.set(count_key, 1, ex=3600*24*7)  # 7 days TTL
                            logger.info(f"[daily] SAVED TOTAL SEGMENTS: 1 for session {session_id}")
                        except Exception as e:
                            logger.warning(f"Failed to save total_segments to Redis: {e}")
                    
                    return {
                        "segments": formatted_segments,
                        "focusIndex": 0,
                        "ref": ref,
                        "he_ref": data.get("heRef", ""),
                    }
            
            logger.info(f"[daily] NO TEXT DATA AVAILABLE: en_text={bool(en_text)}, he_text={bool(he_text)}, he_data_type={type(he_data)}")
            
            # Check if this is Jerusalem Talmud (triple structure: Chapter:Mishnah:Halakhah)
            if "jerusalem talmud" in ref.lower() and ":" in ref:
                logger.info(f"[daily] DETECTED JERUSALEM TALMUD: {ref}")
                
                # Parse Jerusalem Talmud range like "Jerusalem Talmud Sotah 5:4:3-6:3"
                if "-" in ref:
                    start_ref, end_ref = ref.split("-", 1)
                    # Extract book name
                    book_parts = start_ref.split()
                    book_name = " ".join(book_parts[:-1])  # "Jerusalem Talmud Sotah"
                    
                    # Parse start and end references
                    start_parts = start_ref.split(":")[-3:]  # Get last 3 parts
                    end_parts = end_ref.split(":")[-3:]     # Get last 3 parts
                    
                    if len(start_parts) >= 3 and len(end_parts) >= 3:
                        start_chapter = int(start_parts[0])
                        start_mishnah = int(start_parts[1])
                        start_halakhah = int(start_parts[2])
                        end_chapter = int(end_parts[0])
                        end_mishnah = int(end_parts[1])
                        end_halakhah = int(end_parts[2])
                        
                        logger.info(f"[daily] JERUSALEM TALMUD RANGE: {start_chapter}:{start_mishnah}:{start_halakhah} to {end_chapter}:{end_mishnah}:{end_halakhah}")
                        
                        all_segments_data = []
                        
                        # Load segments from start chapter
                        current_chapter = start_chapter
                        current_mishnah = start_mishnah
                        current_halakhah = start_halakhah
                        
                        while current_chapter <= end_chapter:
                            max_mishnah = end_mishnah if current_chapter == end_chapter else 20  # Assume max 20 mishnayot per chapter
                            min_mishnah = current_mishnah if current_chapter == start_chapter else 1
                            
                            for mishnah in range(min_mishnah, max_mishnah + 1):
                                max_halakhah = end_halakhah if (current_chapter == end_chapter and mishnah == end_mishnah) else 20  # Assume max 20 halakhot per mishnah
                                min_halakhah = current_halakhah if (current_chapter == start_chapter and mishnah == start_mishnah) else 1
                                
                                for halakhah in range(min_halakhah, max_halakhah + 1):
                                    segment_ref = f"{book_name} {current_chapter}:{mishnah}:{halakhah}"
                                    try:
                                        segment_result = await sefaria_service.get_text(segment_ref)
                                        if segment_result.get("ok") and segment_result.get("data"):
                                            segment_data_item = segment_result["data"]
                                            en_text = segment_data_item.get("en_text", "") or segment_data_item.get("text", "")
                                            he_text = segment_data_item.get("he_text", "") or segment_data_item.get("he", "")
                                            
                                            if en_text or he_text:
                                                segment_data = {
                                                    "ref": segment_ref,
                                                    "en_text": en_text,
                                                    "he_text": _extract_hebrew_text(he_text),
                                                    "title": data.get("title", ref),
                                                    "indexTitle": data.get("indexTitle", data.get("book", "")),
                                                    "heRef": segment_data_item.get("heRef", "")
                                                }
                                                all_segments_data.append(segment_data)
                                                logger.info(f"[daily] JERUSALEM TALMUD SEGMENT: {segment_ref}")
                                            else:
                                                break  # No more halakhot in this mishnah
                                        else:
                                            break  # No more halakhot in this mishnah
                                    except Exception as e:
                                        logger.warning(f"[daily] FAILED TO FETCH JERUSALEM TALMUD SEGMENT {segment_ref}: {str(e)}")
                                        break
                                
                                current_halakhah = 1  # Reset for next mishnah
                            
                            current_mishnah = 1  # Reset for next chapter
                            current_chapter += 1
                        
                        if all_segments_data:
                            # Format segments for frontend
                            formatted_segments = []
                            total_segments = len(all_segments_data)
                            for i, seg_data in enumerate(all_segments_data):
                                formatted_segments.append({
                                    "ref": seg_data.get("ref"),
                                    "text": getattr(seg_data, "he_text", "") or seg_data.get("he_text") or "",
                                    "heText": getattr(seg_data, "he_text", "") or seg_data.get("he_text") or "",
                                    "position": (i / (total_segments - 1)) if total_segments > 1 else 0.5,
                                    "metadata": {
                                        "title": seg_data.get("title"),
                                        "indexTitle": seg_data.get("indexTitle"),
                                        "heRef": seg_data.get("heRef")
                                    }
                                })
                            
                            logger.info(f"[daily] JERUSALEM TALMUD COMPLETE: loaded {len(formatted_segments)} segments")
                            
                            # Save total_segments to Redis if session_id and redis_client are provided
                            if session_id and redis_client:
                                try:
                                    count_key = f"daily:sess:{session_id}:total_segments"
                                    await redis_client.set(count_key, total_segments, ex=3600*24*7)  # 7 days TTL
                                    logger.info(f"[daily] SAVED TOTAL SEGMENTS: {total_segments} for session {session_id}")
                                except Exception as e:
                                    logger.warning(f"Failed to save total_segments to Redis: {e}")
                            
                            return {
                                "segments": formatted_segments,
                                "focusIndex": 0,
                                "ref": ref,
                                "he_ref": all_segments_data[0].get("heRef") if all_segments_data else None,
                            }

            # Check if this is a Babylonian Talmud daf that should be segmented
            # Examples: "Zevachim 18", "Shabbat 21a", "Berakhot 2"
            elif (" " in ref and not ":" in ref and not "-" in ref and 
                any(word in ref.lower() for word in ["zevachim", "shabbat", "berakhot", "gittin", "ketubot", "baba", "sanhedrin", "yoma", "sukkah", "beitzah", "rosh", "taanit", "megillah", "moed", "pesachim", "yevamot", "nedarim", "nazir", "sotah", "kiddushin", "avodah", "horayot", "menachot", "hullin", "bekhorot", "arakhin", "temurah", "keritot", "meilah", "tamid", "middot", "kinnim", "niddah"])):
                
                logger.info(f"[daily] DETECTED TALMUD DAF: {ref}, checking if it can be segmented")
                logger.info(f"[daily] TALMUD DEBUG: isSpanning={data.get('isSpanning')}, text_data={text_data}, text_type={type(text_data)}")
                
                # Try to detect if this is actually spanning/segmented content
                if data.get("isSpanning") or (isinstance(text_data, str) and len(text_data) > 1000) or (text_data is None and data.get("isSpanning")):
                    logger.info(f"[daily] TALMUD DAF SEGMENTATION: {ref} appears to be a full daf, attempting to retrieve individual segments")
                    
                    # For Talmud, try to get individual segments by querying for 18a:1, 18a:2, 18b:1, 18b:2, etc.
                    base_daf = ref  # e.g., "Zevachim 18"
                    segment_count = 0
                    
                    # Extract daf number from ref (e.g., "Zevachim 18" -> "18")
                    daf_parts = base_daf.split()
                    if len(daf_parts) >= 2:
                        daf_num = daf_parts[-1]  # "18"
                        book_name = " ".join(daf_parts[:-1])  # "Zevachim"
                        
                        # Try to fetch segments from both sides (a and b)
                        for side in ['a', 'b']:
                            side_segments: list = []
                            for segment_num in range(1, 21):  # Up to 20 segments per side
                                segment_ref = f"{book_name} {daf_num}{side}:{segment_num}"
                                try:
                                    segment_result = await sefaria_service.get_text(segment_ref)
                                    if segment_result.get("ok") and segment_result.get("data"):
                                        segment_data_item = segment_result["data"]
                                        # For daily mode, data comes through CompactText, so use en_text/he_text
                                        en_text = getattr(segment_data_item, "en_text", "") or segment_data_item.get("en_text", "") or segment_data_item.get("text", "")
                                        he_text = getattr(segment_data_item, "he_text", "") or segment_data_item.get("he_text", "") or segment_data_item.get("he", "")
                                        
                                        if en_text or he_text:  # Only add if there's actual content
                                            segment_data = {
                                                "ref": segment_ref,
                                                "en_text": en_text,
                                                "he_text": _extract_hebrew_text(he_text),
                                                "title": data.get("title", ref),
                                                "indexTitle": data.get("indexTitle", data.get("book", "")),
                                                "heRef": segment_data_item.get("heRef", "")
                                            }
                                            side_segments.append(segment_data)
                                            segment_count += 1
                                            logger.info(f"[daily] TALMUD SEGMENT #{segment_count}: {segment_ref}, en_len={len(en_text)}, he_len={len(he_text) if he_text else 0}")
                                        else:
                                            # No content, probably reached the end of this side
                                            break
                                    else:
                                        # API call failed or no data, probably reached the end of this side
                                        break
                                except Exception as e:
                                    logger.warning(f"[daily] FAILED TO FETCH TALMUD SEGMENT {segment_ref}: {str(e)}")
                                    break

                            # Ensure :1 exists as a placeholder if the side has content but started after 1
                            if side_segments:
                                has_first = any(s.get("ref", "").endswith(f"{side}:1") or s.get("ref", "").endswith(f"{side}.1") for s in side_segments)
                                if not has_first:
                                    placeholder_ref = f"{book_name} {daf_num}{side}:1"
                                    placeholder_segment = {
                                        "ref": placeholder_ref,
                                        "en_text": "",
                                        "he_text": "",  # keep empty so UI shows segment number badge without content
                                        "title": data.get("title", ref),
                                        "indexTitle": data.get("indexTitle", data.get("book", "")),
                                        "heRef": ""
                                    }
                                    side_segments.insert(0, placeholder_segment)
                                    logger.info(f"✨ Inserted placeholder for missing first segment: {placeholder_ref}")

                                # Append collected side segments preserving order
                                all_segments_data.extend(side_segments)
                    
                    if segment_count > 0:
                        logger.info(f"[daily] TALMUD SEGMENTATION COMPLETED: {ref} -> {segment_count} segments")
                        focus_index = 0
                    # Normalize positions post-factum
                    n = len(all_segments_data)
                    for j, seg in enumerate(all_segments_data):
                        seg["position"] = (j / (n - 1)) if n > 1 else 0.5

                    return {
                        "segments": all_segments_data,
                        "focusIndex": focus_index,
                        "totalLength": len(all_segments_data),
                        "ref": ref,
                        "loadedAt": str(int(time.time() * 1000))
                    }
            
            # If this is actually a range but returned as single text, try individual fetching
            if "-" in ref and ":" in ref:
                logger.info(f"[daily] RANGE DETECTED IN SINGLE TEXT: {ref}, attempting individual fetching")
                start_ref, end_ref = ref.split("-", 1)
                logger.info(f"[daily] SPLIT RESULT: start_ref='{start_ref}', end_ref='{end_ref}'")
                
                if ":" in start_ref:
                    # Handle verse ranges like "Deuteronomy 32:1-52" or "Deuteronomy 32:1-Deuteronomy 32:52"
                    book_chapter = start_ref.rsplit(":", 1)[0]  # "Deuteronomy 32"
                    start_verse = int(start_ref.split(":")[-1])  # 1
                    
                    if ":" in end_ref:
                        # Full reference in end_ref: "Deuteronomy 32:52"
                        end_verse = int(end_ref.split(":")[-1])     # 52
                    else:
                        # Just verse number in end_ref: "52"
                        end_verse = int(end_ref.strip())            # 52
                    
                    logger.info(f"[daily] FETCHING SINGLE-TEXT RANGE: {book_chapter}, verses {start_verse}-{end_verse}")
                    logger.info(f"[daily] ABOUT TO LOOP: range({start_verse}, {end_verse + 1})")
                    
                    # Fetch each verse individually
                    verse_count = 0
                    for verse_num in range(start_verse, end_verse + 1):
                        verse_ref = f"{book_chapter}:{verse_num}"
                        verse_count += 1
                        logger.info(f"[daily] FETCHING VERSE #{verse_count}: {verse_ref}")
                        try:
                            verse_result = await sefaria_service.get_text(verse_ref)
                            if verse_result.get("ok") and verse_result.get("data"):
                                verse_data = verse_result["data"]
                                # For daily mode, data comes through CompactText, so use en_text/he_text
                                en_text = getattr(verse_data, "en_text", "") or verse_data.get("en_text", "") or verse_data.get("text", "")
                                he_text = getattr(verse_data, "he_text", "") or verse_data.get("he_text", "") or verse_data.get("he", "")
                                
                                # Debug Hebrew text availability for ranges
                                logger.info(f"[daily] RANGE VERSE DEBUG: {verse_ref}")
                                logger.info(f"[daily] EN available: {bool(en_text)}, HE available: {bool(he_text)}")
                                logger.info(f"[daily] Raw verse_data keys: {list(verse_data.keys())}")
                                if isinstance(he_text, str) and he_text:
                                    logger.info(f"[daily] HE text preview: '{he_text[:50]}...'")
                                elif isinstance(he_text, list):
                                    logger.info(f"[daily] HE text is list, length: {len(he_text)}")
                                    if he_text and he_text[0]:
                                        logger.info(f"[daily] HE[0] preview: '{he_text[0][:50]}...'")
                                else:
                                    logger.info(f"[daily] HE text type: {type(he_text)}, value: '{he_text}'")
                                
                                segment_data = {
                                    "ref": verse_ref,
                                    "en_text": en_text,
                                    "he_text": _extract_hebrew_text(he_text),
                                    "title": data.get("title", ref),
                                    "indexTitle": data.get("indexTitle", data.get("book", "")),
                                    "heRef": verse_data.get("heRef", "")
                                }
                                all_segments_data.append(segment_data)
                                logger.info(f"[daily] SINGLE-RANGE VERSE #{verse_count}: {verse_ref}, added to segments (total now: {len(all_segments_data)})")
                            else:
                                logger.warning(f"[daily] FAILED TO FETCH VERSE: {verse_ref}, result: {verse_result}")
                        except Exception as e:
                            logger.error(f"[daily] ERROR FETCHING SINGLE-RANGE VERSE {verse_ref}: {str(e)}", exc_info=True)
                            continue
                            
                    logger.info(f"[daily] LOOP COMPLETED: processed {verse_count} verses, segments: {len(all_segments_data)}")
                else:
                    logger.info(f"[daily] RANGE PARSING FAILED: Could not parse range format for {ref}")
                    # Fall back to single segment
                    segment_data = {
                        "ref": data.get("ref", ref),
                        "text": _clean_html_text(data.get("text", "") if isinstance(data.get("text"), str) else ""),
                        "heText": data.get("he", "") if isinstance(data.get("he"), str) else "",
                        "position": 1.0,
                        "metadata": {
                            "title": data.get("title", ref),
                            "indexTitle": data.get("indexTitle", data.get("book", "")),
                            "heRef": data.get("heRef", "")
                        }
                    }
                    all_segments_data.append(segment_data)
            else:
                # True single text
                segment_data = {
                    "ref": data.get("ref", ref),
                    "text": _clean_html_text(data.get("text", "") if isinstance(data.get("text"), str) else ""),
                    "heText": data.get("he", "") if isinstance(data.get("he"), str) else "",
                    "position": 1.0,
                    "metadata": {
                        "title": data.get("title", ref),
                        "indexTitle": data.get("indexTitle", data.get("book", "")),
                        "heRef": data.get("heRef", "")
                    }
                }
                all_segments_data.append(segment_data)
                
            focus_index = 0
    
    if not all_segments_data:
        logger.warning(f"No segments extracted for daily reference: {ref}")
        return None
    
    # Format segments for frontend
    formatted_segments = []
    total_segments = len(all_segments_data)
    for i, seg_data in enumerate(all_segments_data):
        raw_en_text = ""
        raw_he_text = ""
        if hasattr(seg_data, "he_text"):
            raw_he_text = getattr(seg_data, "he_text") or ""
        if hasattr(seg_data, "en_text"):
            raw_en_text = getattr(seg_data, "en_text") or ""
        if not raw_he_text and isinstance(seg_data, dict):
            raw_he_text = (
                seg_data.get("he_text")
                or seg_data.get("heText")
                or ""
            )
        if not raw_en_text and isinstance(seg_data, dict):
            raw_en_text = (
                seg_data.get("en_text")
                or seg_data.get("enText")
                or seg_data.get("text")
                or ""
            )
        if isinstance(raw_he_text, list):
            raw_he_text = " ".join(str(item) for item in raw_he_text if item)
        elif raw_he_text is None:
            raw_he_text = ""
        else:
            raw_he_text = str(raw_he_text)
        if isinstance(raw_en_text, list):
            raw_en_text = " ".join(str(item) for item in raw_en_text if item)
        elif raw_en_text is None:
            raw_en_text = ""
        else:
            raw_en_text = str(raw_en_text)

        metadata = {}
        if isinstance(seg_data, dict):
            metadata = seg_data.get("metadata") or {}

        formatted_segments.append({
            "ref": seg_data.get("ref") if isinstance(seg_data, dict) else getattr(seg_data, "ref", None),
            "text": raw_he_text,
            "heText": raw_he_text,
            "enText": raw_en_text,
            "position": (i / (total_segments - 1)) if total_segments > 1 else 0.5,
            "metadata": {
                "title": (seg_data.get("title") if isinstance(seg_data, dict) else getattr(seg_data, "title", None)) or metadata.get("title"),
                "indexTitle": (seg_data.get("indexTitle") if isinstance(seg_data, dict) else getattr(seg_data, "indexTitle", None)) or metadata.get("indexTitle"),
                "chapter": (seg_data.get("chapter") if isinstance(seg_data, dict) else getattr(seg_data, "chapter", None)) or metadata.get("chapter"),
                "verse": (seg_data.get("verse") if isinstance(seg_data, dict) else getattr(seg_data, "verse", None)) or metadata.get("verse"),
                "heRef": (seg_data.get("heRef") if isinstance(seg_data, dict) else getattr(seg_data, "heRef", None)) or metadata.get("heRef"),
                "enText": metadata.get("enText") if metadata.get("enText") else raw_en_text,
            }
        })
    
    logger.info(f"Loaded {len(formatted_segments)} segments for daily reference: {ref}")
    
    # Save total_segments to Redis if session_id and redis_client are provided
    if session_id and redis_client:
        try:
            count_key = f"daily:sess:{session_id}:total_segments"
            await redis_client.set(count_key, total_segments, ex=3600*24*7)  # 7 days TTL
            logger.info(f"[daily] SAVED TOTAL SEGMENTS: {total_segments} for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to save total_segments to Redis: {e}")
    
    return {
        "segments": formatted_segments,
        "focusIndex": focus_index,
        "ref": ref,
        "he_ref": all_segments_data[0].get("heRef") if all_segments_data else None,
    }

# --- Bookshelf Logic ---

def _get_commentator_priority(commentator: str, collection: str) -> int:
    base_priority = {"Rashi": 100, "Tosafot": 90, "Ramban": 80, "Ibn Ezra": 75, "Sforno": 70, "Shach": 85, "Taz": 85}.get(commentator, 50)
    if collection == "Talmud" and commentator in ["Rashi", "Tosafot"]: return base_priority + 20
    if collection == "Bible" and commentator in ["Rashi", "Ramban", "Ibn Ezra"]: return base_priority + 20
    return base_priority

async def get_bookshelf_for(ref: str, sefaria_service: SefariaService, index_service: SefariaIndexService, limit: int = 40, categories: Optional[List[str]] = None) -> Bookshelf:
    collection = detect_collection(ref)
    
    # If categories aren't specified by the caller, use a curated default list.
    if categories is None:
        categories = [
            "Commentary",
            "Talmud",
            "Halakhah",
            "Responsa",
            "Mishnah",
            "Midrash",
            "Jewish Thought",
            "Chasidut",
            "Kabbalah",
            "Modern Works",
            "Bible",
        ]

    # 1. Try the original ref
    links_result = await sefaria_service.get_related_links(ref=ref, categories=categories, limit=limit * 2)

    # 2. If no links, try raising the level by removing the last segment
    if not links_result.get("ok") or not links_result.get("data"):
        parent_ref = ":".join(ref.split(":")[:-1])
        if parent_ref and parent_ref != ref:
            logger.info(f"No links for '{ref}', trying parent '{parent_ref}'")
            links_result = await sefaria_service.get_related_links(ref=parent_ref, categories=categories, limit=limit * 2)

    if not links_result.get("ok") or not (items := links_result.get("data")):
        return Bookshelf(counts={}, items=[])

    for item in items:
        item["score"] = _get_commentator_priority(item.get("commentator", ""), collection)
    
    sorted_items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    preview_tasks = []
    for item in sorted_items[:20]: # Fetch full text for top 20
        async def fetch_full_text(item_ref):
            res = await sefaria_service.get_text(item_ref)
            if res.get("ok") and res.get("data"):
                data = res["data"]
                en_text = getattr(data, "en_text", "") or data.get("en_text", "") or ""
                he_text = getattr(data, "he_text", "") or data.get("he_text", "") or ""
                return (en_text, he_text)
            return ("", "")
        preview_tasks.append(fetch_full_text(item["ref"]))

    full_texts = await asyncio.gather(*preview_tasks)
    for i, item in enumerate(sorted_items[:20]):
        en_text, he_text = full_texts[i]
        item["text_full"] = en_text
        item["heTextFull"] = he_text
        # For backward compatibility, populate preview with a snippet
        item["preview"] = (en_text or he_text)[:PREVIEW_MAX_LEN]

    # For items beyond 20, ensure the fields exist but are empty to satisfy the model
    for item in sorted_items[20:]:
        item["text_full"] = ""
        item["heTextFull"] = ""
        item["preview"] = ""

    counts = {cat: 0 for cat in {item.get("category", "Unknown") for item in sorted_items}}
    for item in sorted_items:
        counts[item.get("category", "Unknown")] += 1

    # Convert dicts to BookshelfItem objects, filtering out any that fail validation
    valid_items = []
    for item_dict in sorted_items[:limit]:
        try:
            # The preview field might be missing for items > 20, provide a default
            if 'preview' not in item_dict:
                item_dict['preview'] = ''
            valid_items.append(BookshelfItem(**item_dict))
        except Exception as e:
            logger.warning(f"Skipping bookshelf item due to validation error: {e}. Item: {item_dict}")

    return Bookshelf(counts=counts, items=valid_items)


async def _load_remaining_segments_background(
    ref: str,
    sefaria_service: SefariaService,
    session_id: str,
    start_verse: int,
    end_verse: int,
    book_chapter: str,
    redis_client,
    already_loaded: int,
) -> None:
    """Legacy shim that delegates background loading to the modular daily loader."""
    try:
        from .study.config_schema import load_study_config
        from .study.daily_loader import DailyLoader
        from .study.redis_repo import StudyRedisRepository
    except ImportError as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to import modular study loader", exc_info=True)
        raise exc

    raw_config = get_config_section("study") or {}
    config = load_study_config(raw_config)
    redis_repo = StudyRedisRepository(redis_client)

    loader = DailyLoader(
        sefaria_service=sefaria_service,
        index_service=None,
        redis_repo=redis_repo,
        config=config,
    )

    total_segments = max(end_verse - start_verse + 1, 0)
    if total_segments <= 0:
        logger.debug(
            "No background segments planned",
            extra={
                "session_id": session_id,
                "ref": ref,
                "start_verse": start_verse,
                "end_verse": end_verse,
            },
        )
        return

    await loader.load_background(
        ref=ref,
        session_id=session_id,
        start_verse=start_verse,
        end_verse=end_verse,
        book_chapter=book_chapter,
        already_loaded=max(already_loaded, 0),
        total_segments=total_segments,
    )
