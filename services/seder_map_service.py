from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from brain_service.models.db import (
    SederArticle,
    SederDefinition,
    SederDomain,
    SederGlossaryTerm,
    SederMapEdge,
    SederMapGroup,
    SederMapLayout,
    SederMapNode,
    SederMapNote,
    SederSegment,
    SederSegmentLink,
    SederSegmentVersion,
)


class SederMapService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def list_map(self) -> Dict[str, Any]:
        async with self._session_factory() as session:
            nodes = (await session.scalars(select(SederMapNode))).all()
            edges = (await session.scalars(select(SederMapEdge))).all()
            groups = (await session.scalars(select(SederMapGroup))).all()
            notes = (await session.scalars(select(SederMapNote))).all()
            return {
                "nodes": [self._node_to_dict(n) for n in nodes],
                "edges": [self._edge_to_dict(e) for e in edges],
                "groups": [self._group_to_dict(g) for g in groups],
                "notes": [self._note_to_dict(n) for n in notes],
            }

    async def get_node(self, node_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            node = await session.get(SederMapNode, node_id)
            if not node:
                return None
            return self._node_to_dict(node)

    async def create_node(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._session_factory() as session:
            async with session.begin():
                node = SederMapNode(
                    definition_id=payload.get("definition_id"),
                    domain_id=payload.get("domain_id"),
                    title_he=payload.get("title_he"),
                    title_ru=payload.get("title_ru"),
                    node_type=payload.get("node_type"),
                    phase=payload.get("phase"),
                    status=payload.get("status"),
                    context_tag=payload.get("context_tag"),
                    article_id=payload.get("article_id"),
                    spine_parent_id=payload.get("spine_parent_id"),
                    pos_x=payload.get("pos_x"),
                    pos_y=payload.get("pos_y"),
                    width=payload.get("width"),
                    height=payload.get("height"),
                )
                session.add(node)
                await session.flush()
                await session.refresh(node)
                return self._node_to_dict(node)

    async def update_node(self, node_id: uuid.UUID, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                node = await session.get(SederMapNode, node_id)
                if not node:
                    return None
                for field in [
                    "definition_id",
                    "domain_id",
                    "title_he",
                    "title_ru",
                    "node_type",
                    "phase",
                    "status",
                    "context_tag",
                    "article_id",
                    "spine_parent_id",
                    "pos_x",
                    "pos_y",
                    "width",
                    "height",
                ]:
                    if field in payload:
                        setattr(node, field, payload[field])
                await session.flush()
                await session.refresh(node)
                return self._node_to_dict(node)

    async def delete_node(self, node_id: uuid.UUID) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(SederMapEdge).where(
                        (SederMapEdge.source_id == node_id) | (SederMapEdge.target_id == node_id)
                    )
                )
                await session.execute(
                    select(SederMapNode).where(SederMapNode.spine_parent_id == node_id)
                )
                await session.execute(
                    SederMapNode.__table__.update()
                    .where(SederMapNode.spine_parent_id == node_id)
                    .values(spine_parent_id=None)
                )
                node = await session.get(SederMapNode, node_id)
                if not node:
                    return False
                await session.delete(node)
                return True

    async def create_edge(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._session_factory() as session:
            async with session.begin():
                edge = SederMapEdge(
                    source_id=payload["source_id"],
                    target_id=payload["target_id"],
                    connection_type=payload.get("connection_type"),
                    label_he=payload.get("label_he"),
                    label_ru=payload.get("label_ru"),
                    is_canonical=bool(payload.get("is_canonical", False)),
                )
                session.add(edge)
                await session.flush()
                await session.refresh(edge)
                return self._edge_to_dict(edge)

    async def create_note(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._session_factory() as session:
            async with session.begin():
                note = SederMapNote(
                    kind=payload.get("kind", "note"),
                    title_he=payload.get("title_he"),
                    title_ru=payload.get("title_ru"),
                    text_he=payload.get("text_he"),
                    text_ru=payload.get("text_ru"),
                    color=payload.get("color"),
                    domain_id=payload.get("domain_id"),
                    attached_node_id=payload.get("attached_node_id"),
                    attached_edge_id=payload.get("attached_edge_id"),
                    pos_x=payload.get("pos_x"),
                    pos_y=payload.get("pos_y"),
                    width=payload.get("width"),
                    height=payload.get("height"),
                )
                session.add(note)
                await session.flush()
                await session.refresh(note)
                return self._note_to_dict(note)

    async def update_note(self, note_id: uuid.UUID, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                note = await session.get(SederMapNote, note_id)
                if not note:
                    return None
                for field in [
                    "kind",
                    "title_he",
                    "title_ru",
                    "text_he",
                    "text_ru",
                    "color",
                    "domain_id",
                    "attached_node_id",
                    "attached_edge_id",
                    "pos_x",
                    "pos_y",
                    "width",
                    "height",
                ]:
                    if field in payload:
                        setattr(note, field, payload[field])
                await session.flush()
                await session.refresh(note)
                return self._note_to_dict(note)

    async def delete_note(self, note_id: uuid.UUID) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                note = await session.get(SederMapNote, note_id)
                if not note:
                    return False
                await session.delete(note)
                return True

    async def delete_edge(self, edge_id: uuid.UUID) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                edge = await session.get(SederMapEdge, edge_id)
                if not edge:
                    return False
                await session.delete(edge)
                return True

    async def create_article(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._session_factory() as session:
            async with session.begin():
                article = SederArticle(
                    title_he=payload.get("title_he"),
                    title_ru=payload.get("title_ru"),
                    text_he=payload.get("text_he"),
                    text_ru=payload.get("text_ru"),
                    source_type=payload.get("source_type", "internal"),
                    status_he=payload.get("status_he"),
                    status_ru=payload.get("status_ru"),
                )
                session.add(article)
                await session.flush()
                await session.refresh(article)
                return self._article_to_dict(article)

    async def get_article(self, article_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            article = await session.get(SederArticle, article_id)
            if not article:
                return None
            return self._article_to_dict(article)

    async def update_article(self, article_id: uuid.UUID, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                article = await session.get(SederArticle, article_id)
                if not article:
                    return None
                for field in [
                    "title_he",
                    "title_ru",
                    "text_he",
                    "text_ru",
                    "source_type",
                    "status_he",
                    "status_ru",
                ]:
                    if field in payload:
                        setattr(article, field, payload[field])
                await session.flush()
                await session.refresh(article)
                return self._article_to_dict(article)

    async def list_segments(self, article_id: uuid.UUID) -> List[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(SederSegment).where(SederSegment.article_id == article_id).order_by(SederSegment.order_index)
            )
            return [self._segment_to_dict(s) for s in result]

    async def create_segments(self, article_id: uuid.UUID, segments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        created: List[SederSegment] = []
        async with self._session_factory() as session:
            async with session.begin():
                for item in segments:
                    seg = SederSegment(
                        article_id=article_id,
                        order_index=item.get("order_index", 0),
                        text_he=item.get("text_he"),
                        text_ru=item.get("text_ru"),
                        status_he=item.get("status_he"),
                        status_ru=item.get("status_ru"),
                    )
                    session.add(seg)
                    created.append(seg)
                await session.flush()
                for seg in created:
                    await session.refresh(seg)
        return [self._segment_to_dict(s) for s in created]

    async def update_segment(
        self,
        *,
        segment_id: uuid.UUID,
        payload: Dict[str, Any],
        author_id: Optional[uuid.UUID],
    ) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                segment = await session.get(SederSegment, segment_id)
                if not segment:
                    return None
                expected_version = payload.get("version")
                if expected_version is None or segment.version != int(expected_version):
                    raise ValueError("version_conflict")

                text_he = payload.get("text_he")
                text_ru = payload.get("text_ru")
                status_he = payload.get("status_he")
                status_ru = payload.get("status_ru")
                comment = payload.get("comment")

                if text_he is not None and text_he != segment.text_he:
                    segment.text_he = text_he
                    session.add(
                        SederSegmentVersion(
                            segment_id=segment.id,
                            lang="he",
                            author_id=author_id,
                            text=text_he,
                            comment=comment,
                        )
                    )
                if text_ru is not None and text_ru != segment.text_ru:
                    segment.text_ru = text_ru
                    session.add(
                        SederSegmentVersion(
                            segment_id=segment.id,
                            lang="ru",
                            author_id=author_id,
                            text=text_ru,
                            comment=comment,
                        )
                    )
                if status_he is not None:
                    segment.status_he = status_he
                if status_ru is not None:
                    segment.status_ru = status_ru

                segment.version = segment.version + 1
                await session.flush()
                await session.refresh(segment)
                return self._segment_to_dict(segment)

    async def list_segment_versions(self, segment_id: uuid.UUID) -> List[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(SederSegmentVersion).where(SederSegmentVersion.segment_id == segment_id).order_by(
                    SederSegmentVersion.created_at.desc()
                )
            )
            return [self._segment_version_to_dict(v) for v in result]

    async def restore_segment_version(
        self, *, segment_id: uuid.UUID, version_id: uuid.UUID, author_id: Optional[uuid.UUID]
    ) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                segment = await session.get(SederSegment, segment_id)
                version = await session.get(SederSegmentVersion, version_id)
                if not segment or not version or version.segment_id != segment_id:
                    return None
                if version.lang == "he":
                    segment.text_he = version.text
                else:
                    segment.text_ru = version.text
                segment.version = segment.version + 1
                session.add(
                    SederSegmentVersion(
                        segment_id=segment.id,
                        lang=version.lang,
                        author_id=author_id,
                        text=version.text,
                        comment="restore",
                    )
                )
                await session.flush()
                await session.refresh(segment)
                return self._segment_to_dict(segment)

    async def upsert_segment_links(self, segment_id: uuid.UUID, links: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(SederSegmentLink).where(
                        (SederSegmentLink.hebrew_segment_id == segment_id)
                        | (SederSegmentLink.russian_segment_id == segment_id)
                    )
                )
                for link in links:
                    session.add(
                        SederSegmentLink(
                            hebrew_segment_id=link["hebrew_segment_id"],
                            russian_segment_id=link["russian_segment_id"],
                            weight=link.get("weight"),
                            is_primary=bool(link.get("is_primary", False)),
                        )
                    )
        return {"ok": True}

    async def list_definitions(self) -> List[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(select(SederDefinition))
            return [self._definition_to_dict(d) for d in result]

    async def list_domains(self) -> List[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(select(SederDomain))
            return [self._domain_to_dict(d) for d in result]

    async def update_domain(self, domain_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                domain = await session.get(SederDomain, domain_id)
                if not domain:
                    return None
                for field in ["title_he", "title_ru", "description", "rules_json", "pos_x", "pos_y", "width", "height"]:
                    if field in payload:
                        setattr(domain, field, payload[field])
                await session.flush()
                await session.refresh(domain)
                return self._domain_to_dict(domain)

    async def create_domain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._session_factory() as session:
            async with session.begin():
                domain = SederDomain(
                    id=str(payload["id"]),
                    title_he=payload.get("title_he"),
                    title_ru=payload.get("title_ru"),
                    description=payload.get("description"),
                    rules_json=payload.get("rules_json"),
                    pos_x=payload.get("pos_x"),
                    pos_y=payload.get("pos_y"),
                    width=payload.get("width"),
                    height=payload.get("height"),
                )
                session.add(domain)
                await session.flush()
                await session.refresh(domain)
                return self._domain_to_dict(domain)

    async def delete_domain(self, domain_id: str) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                # Detach nodes/notes from this domain before delete
                await session.execute(
                    SederMapNode.__table__.update()
                    .where(SederMapNode.domain_id == domain_id)
                    .values(domain_id=None)
                )
                await session.execute(
                    SederMapNote.__table__.update()
                    .where(SederMapNote.domain_id == domain_id)
                    .values(domain_id=None)
                )
                domain = await session.get(SederDomain, domain_id)
                if not domain:
                    return False
                await session.delete(domain)
                return True

    async def list_definition_instances(self, definition_id: uuid.UUID) -> List[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(SederMapNode).where(SederMapNode.definition_id == definition_id)
            )
            return [self._node_to_dict(n) for n in result]

    async def list_layouts(self) -> List[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(select(SederMapLayout))
            return [self._layout_to_dict(l) for l in result]

    async def create_layout(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._session_factory() as session:
            async with session.begin():
                layout = SederMapLayout(
                    name=payload.get("name"),
                    is_canonical=bool(payload.get("is_canonical", False)),
                    owner_user_id=payload.get("owner_user_id"),
                    layout_json=payload.get("layout_json"),
                )
                session.add(layout)
                await session.flush()
                await session.refresh(layout)
                return self._layout_to_dict(layout)

    async def update_layout(self, layout_id: uuid.UUID, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                layout = await session.get(SederMapLayout, layout_id)
                if not layout:
                    return None
                if "name" in payload:
                    layout.name = payload["name"]
                if "is_canonical" in payload:
                    layout.is_canonical = bool(payload["is_canonical"])
                if "layout_json" in payload:
                    layout.layout_json = payload["layout_json"]
                await session.flush()
                await session.refresh(layout)
                return self._layout_to_dict(layout)

    @staticmethod
    def _node_to_dict(node: SederMapNode) -> Dict[str, Any]:
        return {
            "id": str(node.id),
            "definition_id": str(node.definition_id) if node.definition_id else None,
            "domain_id": node.domain_id,
            "title_he": node.title_he,
            "title_ru": node.title_ru,
            "node_type": node.node_type,
            "phase": node.phase,
            "status": node.status,
            "context_tag": node.context_tag,
            "article_id": str(node.article_id) if node.article_id else None,
            "spine_parent_id": str(node.spine_parent_id) if node.spine_parent_id else None,
            "pos_x": node.pos_x,
            "pos_y": node.pos_y,
            "width": node.width,
            "height": node.height,
        }

    @staticmethod
    def _edge_to_dict(edge: SederMapEdge) -> Dict[str, Any]:
        return {
            "id": str(edge.id),
            "source_id": str(edge.source_id),
            "target_id": str(edge.target_id),
            "connection_type": edge.connection_type,
            "label_he": edge.label_he,
            "label_ru": edge.label_ru,
            "is_canonical": edge.is_canonical,
        }

    @staticmethod
    def _group_to_dict(group: SederMapGroup) -> Dict[str, Any]:
        return {
            "id": str(group.id),
            "title_he": group.title_he,
            "title_ru": group.title_ru,
            "phase": group.phase,
            "order_index": group.order_index,
            "pos_x": group.pos_x,
            "pos_y": group.pos_y,
            "width": group.width,
            "height": group.height,
        }

    @staticmethod
    def _article_to_dict(article: SederArticle) -> Dict[str, Any]:
        return {
            "id": str(article.id),
            "title_he": article.title_he,
            "title_ru": article.title_ru,
            "text_he": article.text_he,
            "text_ru": article.text_ru,
            "source_type": article.source_type,
            "status_he": article.status_he,
            "status_ru": article.status_ru,
        }

    @staticmethod
    def _segment_to_dict(segment: SederSegment) -> Dict[str, Any]:
        return {
            "id": str(segment.id),
            "article_id": str(segment.article_id),
            "order_index": segment.order_index,
            "text_he": segment.text_he,
            "text_ru": segment.text_ru,
            "status_he": segment.status_he,
            "status_ru": segment.status_ru,
            "version": segment.version,
        }

    @staticmethod
    def _segment_version_to_dict(version: SederSegmentVersion) -> Dict[str, Any]:
        return {
            "id": str(version.id),
            "segment_id": str(version.segment_id),
            "lang": version.lang,
            "author_id": str(version.author_id) if version.author_id else None,
            "text": version.text,
            "comment": version.comment,
            "created_at": version.created_at.isoformat() if version.created_at else None,
        }

    @staticmethod
    def _definition_to_dict(defn: SederDefinition) -> Dict[str, Any]:
        return {
            "id": str(defn.id),
            "term_he": defn.term_he,
            "term_ru": defn.term_ru,
            "translit": defn.translit,
            "description": defn.description,
            "tags": defn.tags,
        }

    @staticmethod
    def _domain_to_dict(domain: SederDomain) -> Dict[str, Any]:
        return {
            "id": domain.id,
            "title_he": domain.title_he,
            "title_ru": domain.title_ru,
            "description": domain.description,
            "rules_json": domain.rules_json,
            "pos_x": domain.pos_x,
            "pos_y": domain.pos_y,
            "width": domain.width,
            "height": domain.height,
        }

    @staticmethod
    def _note_to_dict(note: SederMapNote) -> Dict[str, Any]:
        return {
            "id": str(note.id),
            "kind": note.kind,
            "title_he": note.title_he,
            "title_ru": note.title_ru,
            "text_he": note.text_he,
            "text_ru": note.text_ru,
            "color": note.color,
            "domain_id": note.domain_id,
            "attached_node_id": str(note.attached_node_id) if note.attached_node_id else None,
            "attached_edge_id": str(note.attached_edge_id) if note.attached_edge_id else None,
            "pos_x": note.pos_x,
            "pos_y": note.pos_y,
            "width": note.width,
            "height": note.height,
        }

    @staticmethod
    def _layout_to_dict(layout: SederMapLayout) -> Dict[str, Any]:
        return {
            "id": str(layout.id),
            "name": layout.name,
            "is_canonical": layout.is_canonical,
            "owner_user_id": str(layout.owner_user_id) if layout.owner_user_id else None,
            "layout_json": layout.layout_json,
        }
