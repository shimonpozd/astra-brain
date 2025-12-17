import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from brain_service.models.yiddish import (
    YiddishAttestation,
    YiddishQueueItem,
    YiddishVocabEntry,
    YiddishSichaProgress,
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
        if not file_path.exists():
            file_path = self._data_root / "page_0001.json"
        return json.loads(file_path.read_text(encoding="utf-8"))

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
        data = self._load_static_sicha("page_0001")
        return {
            "items": [
                {
                    "id": data.get("sicha_id", "ls10_miketz_b"),
                    "title": "Likkutei Sichos 10 · Miketz · B",
                    "meta": data.get("meta", {}),
                    "progress_read_pct": 0,
                    "progress_vocab": 0,
                    "last_opened_ts": None,
                }
            ]
        }

    async def get_sicha(self, sicha_id: str, user_id: str) -> Dict[str, Any]:
        data = self._load_static_sicha(sicha_id)
        self._ensure_tokens(data)
        # Pull learned map from vocab
        learned_map: Dict[str, List[str]] = {}
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                select(YiddishVocabEntry.lemma, YiddishVocabEntry.sense_id).where(YiddishVocabEntry.user_id == user_id)
            )
            for lemma, sense_id in result.all():
                learned_map.setdefault(lemma, []).append(sense_id)

        return {
            "id": data.get("sicha_id", sicha_id),
            "meta": data.get("meta", {}),
            "paragraphs": data.get("paragraphs", []),
            "tokens": data.get("tokens", []),
            "notes": data.get("notes", []),
            "learned_map": learned_map,
            "offsets_version": data.get("offsets_version", "static-demo"),
        }

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
                senses.append({"sense_id": row.sense_id, "gloss_ru": "нет данных"})
            return {
                "lemma": lemma,
                "pos": "NOUN",
                "senses": senses or [{"sense_id": f"{lemma}-1", "gloss_ru": "нет данных (демо)"}],
                "attestations": [],
            }
