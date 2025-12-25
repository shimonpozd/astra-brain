import json
import logging
import random
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from brain_service.models.yiddish import (
    YiddishAttestation,
    YiddishQueueItem,
    YiddishVocabEntry,
    YiddishSichaProgress,
    YiddishWordCard,
)
from brain_service.core.database import session_scope

logger = logging.getLogger(__name__)


class YiddishService:
    """
    Service for Yiddish Mode data (progress, vocab, queue, attestations).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession], redis_client: Optional[redis.Redis] = None):
        self.session_factory = session_factory
        self.redis_client = redis_client
        # Use local brain_service/data/yiddish by default
        self._data_root = Path(__file__).resolve().parents[1] / "data" / "yiddish"

    def _load_static_sicha(self, sicha_id: str) -> Dict[str, Any]:
        file_path = self._data_root / f"{sicha_id}.json"
        if file_path.exists():
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if (data.get("meta", {}) or {}).get("lang") != "ru":
                return data

        for candidate in sorted(self._data_root.glob("*.json")):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("sicha_id") == sicha_id and (data.get("meta", {}) or {}).get("lang") != "ru":
                return data

        fallback = self._data_root / "page_0001.json"
        return json.loads(fallback.read_text(encoding="utf-8"))

    def _load_static_sicha_ru(self, sicha_id: str) -> Optional[Dict[str, Any]]:
        ru_path = self._data_root / f"{sicha_id}_ru.json"
        if ru_path.exists():
            return json.loads(ru_path.read_text(encoding="utf-8"))

        for candidate in sorted(self._data_root.glob("*.json")):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("sicha_id") == sicha_id and (data.get("meta", {}) or {}).get("lang") == "ru":
                return data
        return None

    def _ensure_tokens(self, data: Dict[str, Any]) -> None:
        """
        If tokens missing, generate simple tokens (letters/numbers/Hebrew) per paragraph with POS=HEB_LOAN.
        """
        if data.get("tokens"):
            return
        paragraphs = data.get("paragraphs") or []
        tokens: List[Dict[str, Any]] = []
        import re

        token_regex = re.compile(r"[\w\u0590-\u05FF]+", re.UNICODE)
        for p in paragraphs:
            text = p.get("text", "")
            for match in token_regex.finditer(text):
                surface = match.group(0)
                tokens.append({
                    "pid": p.get("pid"),
                    "start": match.start(),
                    "end": match.end(),
                    "surface": surface,
                    "lemma": surface,
                    "pos": "HEB_LOAN",
                    "confidence": 0.5,
                })
        data["tokens"] = tokens

    async def list_sichos(self, user_id: str) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for candidate in sorted(self._data_root.glob("*.json")):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if candidate.stem.endswith("_ru"):
                continue
            meta = data.get("meta", {}) or {}
            if meta.get("lang") == "ru":
                continue
            sicha_id = data.get("sicha_id") or candidate.stem
            title = data.get("title")
            if not title:
                work = meta.get("work") or "Likkutei Sichos"
                parsha = meta.get("parsha") or ""
                section = meta.get("section") or ""
                volume = meta.get("volume") or ""
                title = " ".join(str(v) for v in [work, volume, parsha, section] if v)
            items.append(
                {
                    "id": sicha_id,
                    "title": title,
                    "meta": meta,
                    "progress_read_pct": 0,
                    "progress_vocab": 0,
                    "last_opened_ts": None,
                }
            )
        return {"items": items}

    async def get_sicha(self, sicha_id: str, user_id: str) -> Dict[str, Any]:
        data = self._load_static_sicha(sicha_id)
        ru_data = self._load_static_sicha_ru(sicha_id)
        self._ensure_tokens(data)
        # Pull learned map from vocab
        learned_map: Dict[str, List[str]] = {}
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                select(YiddishVocabEntry.lemma, YiddishVocabEntry.sense_id).where(YiddishVocabEntry.user_id == user_id)
            )
            for lemma, sense_id in result.all():
                learned_map.setdefault(lemma, []).append(sense_id)

        response = {
            "id": data.get("sicha_id", sicha_id),
            "meta": data.get("meta", {}),
            "paragraphs": data.get("paragraphs", []),
            "tokens": data.get("tokens", []),
            "notes": data.get("notes", []),
            "learned_map": learned_map,
            "offsets_version": data.get("offsets_version", "static-demo"),
        }
        if ru_data:
            response["ru_paragraphs"] = ru_data.get("paragraphs", [])
            response["ru_available"] = True
        else:
            response["ru_available"] = False
        return response

    async def save_attestation(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with session_scope(self.session_factory) as session:
            att = YiddishAttestation(
                user_id=user_id,
                lemma=payload.get("lemma"),
                sense_id=payload.get("sense_id"),
                pid=payload.get("pid"),
                start=payload.get("start"),
                end=payload.get("end"),
                surface=payload.get("surface", ""),
                context_sentence=payload.get("context_sentence"),
                sicha_id=payload.get("sicha_id"),
            )
            session.add(att)
            await self._touch_vocab(session, user_id, payload.get("lemma"), payload.get("sense_id"))
        return {"ok": True, "learned": True}

    async def _touch_vocab(self, session: AsyncSession, user_id: str, lemma: str, sense_id: str):
        now = datetime.utcnow()
        result = await session.execute(
            select(YiddishVocabEntry).where(
                YiddishVocabEntry.user_id == user_id,
                YiddishVocabEntry.lemma == lemma,
                YiddishVocabEntry.sense_id == sense_id,
            )
        )
        vocab = result.scalar_one_or_none()
        if vocab:
            vocab.last_seen_at = now
            vocab.seen_count = (vocab.seen_count or 0) + 1
        else:
            session.add(
                YiddishVocabEntry(
                    user_id=user_id,
                    lemma=lemma,
                    sense_id=sense_id,
                    last_seen_at=now,
                    seen_count=1,
                )
            )

    async def update_queue(self, user_id: str, action: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        async with session_scope(self.session_factory) as session:
            if action == "add":
                item = YiddishQueueItem(
                    user_id=user_id,
                    lemma=entry.get("lemma"),
                    sense_id=entry.get("sense_id"),
                    source_pid=entry.get("source_pid"),
                )
                session.add(item)
            elif action == "remove":
                await session.execute(
                    delete(YiddishQueueItem).where(
                        YiddishQueueItem.user_id == user_id,
                        YiddishQueueItem.lemma == entry.get("lemma"),
                        YiddishQueueItem.sense_id == entry.get("sense_id"),
                    )
                )
        return {"ok": True, "queue": await self.get_queue(user_id)}

    async def get_queue(self, user_id: str) -> List[Dict[str, Any]]:
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                select(YiddishQueueItem.lemma, YiddishQueueItem.sense_id, YiddishQueueItem.source_pid).where(
                    YiddishQueueItem.user_id == user_id
                )
            )
            return [{"lemma": l, "sense_id": s, "source_pid": pid} for l, s, pid in result.all()]

    async def start_exam(self, user_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "exam_id": "exam-" + datetime.utcnow().isoformat(),
            "items": [
                {"type": "cloze", "payload": {"lemma": i.get("lemma"), "sense_id": i.get("sense_id"), "prompt": "Заполните пропуск"}}
                for i in items
            ],
        }

    async def get_vocab(self, user_id: str, lemma: str) -> Dict[str, Any]:
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                select(YiddishVocabEntry).where(
                    YiddishVocabEntry.user_id == user_id,
                    YiddishVocabEntry.lemma == lemma,
                )
            )
            senses = []
            for row in result.scalars().all():
                senses.append({"sense_id": row.sense_id, "gloss_ru": ""})
            return {
                "lemma": lemma,
                "pos": None,
                "senses": senses,
                "attestations": [],
            }

    async def generate_mahjong_exam(self, user_id: str, min_words: int = 8, max_words: int = 12) -> Dict[str, Any]:
        min_words = max(2, min_words)
        max_words = max(min_words, max_words)
        target_words = min(max_words, max(min_words, 10))

        async with session_scope(self.session_factory) as session:
            queue_rows = await session.execute(
                select(YiddishQueueItem.lemma)
                .where(YiddishQueueItem.user_id == user_id)
                .order_by(YiddishQueueItem.created_at.desc())
            )
            queue_lemmas = [row[0] for row in queue_rows.all() if row[0]]

            recent_rows = await session.execute(
                select(YiddishAttestation.lemma)
                .where(YiddishAttestation.user_id == user_id)
                .order_by(YiddishAttestation.created_at.desc())
                .limit(100)
            )
            recent_lemmas = [row[0] for row in recent_rows.all() if row[0]]

            if not recent_lemmas:
                vocab_rows = await session.execute(
                    select(YiddishVocabEntry.lemma)
                    .where(YiddishVocabEntry.user_id == user_id)
                    .order_by(YiddishVocabEntry.last_seen_at.desc().nullslast())
                    .limit(100)
                )
                recent_lemmas = [row[0] for row in vocab_rows.all() if row[0]]

            candidate_lemmas: List[str] = []
            seen = set()
            for lemma in queue_lemmas + recent_lemmas:
                if lemma in seen:
                    continue
                seen.add(lemma)
                candidate_lemmas.append(lemma)

            wordcard_map: Dict[str, Dict[str, Any]] = {}
            if candidate_lemmas:
                result = await session.execute(
                    select(YiddishWordCard)
                    .where(
                        YiddishWordCard.lemma.in_(candidate_lemmas),
                        YiddishWordCard.ui_lang == "ru",
                        YiddishWordCard.source == "wiktionary",
                        YiddishWordCard.version == 1,
                    )
                )
                for row in result.scalars().all():
                    data = row.data or {}
                    popup = data.get("popup") or {}
                    glosses = popup.get("gloss_ru_short_list") if isinstance(popup, dict) else None
                    if not glosses:
                        glosses = []
                        for sense in data.get("senses") or []:
                            gloss = sense.get("gloss_ru_short") or sense.get("gloss_ru_full")
                            if gloss:
                                glosses.append(gloss)
                    glosses = [g for g in glosses if isinstance(g, str) and g.strip()]
                    word_surface = row.word_surface or data.get("word_surface")
                    if not word_surface or not glosses:
                        continue
                    wordcard_map[row.lemma] = {
                        "lemma": row.lemma,
                        "word_surface": word_surface,
                        "gloss": glosses[0],
                        "pos": row.pos_default or data.get("pos_default"),
                    }

            selected_words: List[Dict[str, Any]] = []
            for lemma in candidate_lemmas:
                if len(selected_words) >= target_words:
                    break
                card = wordcard_map.get(lemma)
                if not card:
                    continue
                selected_words.append(card)

            if len(selected_words) < min_words:
                needed = min_words - len(selected_words)
                random_rows = await session.execute(
                    select(YiddishWordCard)
                    .where(
                        YiddishWordCard.ui_lang == "ru",
                        YiddishWordCard.source == "wiktionary",
                        YiddishWordCard.version == 1,
                    )
                    .order_by(func.random())
                    .limit(needed * 3)
                )
                for row in random_rows.scalars().all():
                    if len(selected_words) >= min_words:
                        break
                    if any(item["lemma"] == row.lemma for item in selected_words):
                        continue
                    data = row.data or {}
                    popup = data.get("popup") or {}
                    glosses = popup.get("gloss_ru_short_list") if isinstance(popup, dict) else None
                    if not glosses:
                        glosses = []
                        for sense in data.get("senses") or []:
                            gloss = sense.get("gloss_ru_short") or sense.get("gloss_ru_full")
                            if gloss:
                                glosses.append(gloss)
                    glosses = [g for g in glosses if isinstance(g, str) and g.strip()]
                    word_surface = row.word_surface or data.get("word_surface")
                    if not word_surface or not glosses:
                        continue
                    selected_words.append(
                        {
                            "lemma": row.lemma,
                            "word_surface": word_surface,
                            "gloss": glosses[0],
                            "pos": row.pos_default or data.get("pos_default"),
                        }
                    )

        tiles: List[Dict[str, Any]] = []
        for item in selected_words[:max_words]:
            match_id = item["lemma"]
            pos = item.get("pos")
            tiles.append(
                {
                    "id": f"t-{uuid.uuid4().hex}",
                    "match_id": match_id,
                    "type": "yi",
                    "content": item["word_surface"],
                    "pos": pos,
                }
            )
            tiles.append(
                {
                    "id": f"t-{uuid.uuid4().hex}",
                    "match_id": match_id,
                    "type": "ru",
                    "content": item["gloss"],
                    "pos": pos,
                }
            )

        random.shuffle(tiles)
        return {
            "exam_id": f"mahjong-{uuid.uuid4().hex}",
            "tiles": tiles,
        }
