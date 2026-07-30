from __future__ import annotations

from typing import Any, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.dependencies import get_current_user, require_admin_user, get_seder_map_service
from brain_service.models.db import User
from brain_service.services.seder_map_service import SederMapService

router = APIRouter()


class NodeCreate(BaseModel):
    definition_id: Optional[uuid.UUID] = None
    domain_id: Optional[str] = None
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    node_type: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    context_tag: Optional[str] = None
    article_id: Optional[uuid.UUID] = None
    spine_parent_id: Optional[uuid.UUID] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class NodeUpdate(BaseModel):
    definition_id: Optional[uuid.UUID] = None
    domain_id: Optional[str] = None
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    node_type: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    context_tag: Optional[str] = None
    article_id: Optional[uuid.UUID] = None
    spine_parent_id: Optional[uuid.UUID] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class EdgeCreate(BaseModel):
    source_id: uuid.UUID
    target_id: uuid.UUID
    connection_type: Optional[str] = None
    label_he: Optional[str] = None
    label_ru: Optional[str] = None
    is_canonical: Optional[bool] = False


class ArticleCreate(BaseModel):
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    text_he: Optional[str] = None
    text_ru: Optional[str] = None
    source_type: Optional[str] = "internal"
    status_he: Optional[str] = None
    status_ru: Optional[str] = None


class ArticleUpdate(BaseModel):
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    text_he: Optional[str] = None
    text_ru: Optional[str] = None
    source_type: Optional[str] = None
    status_he: Optional[str] = None
    status_ru: Optional[str] = None


class SegmentCreate(BaseModel):
    order_index: int = 0
    source_ref: Optional[str] = None
    sub_index: int = 0
    role: Optional[str] = None
    start_anchor: Optional[str] = None
    end_anchor: Optional[str] = None
    start_word_idx: Optional[int] = None
    end_word_idx: Optional[int] = None
    text_he: Optional[str] = None
    text_ru: Optional[str] = None
    status_he: Optional[str] = None
    status_ru: Optional[str] = None


class SegmentUpdate(BaseModel):
    source_ref: Optional[str] = None
    sub_index: Optional[int] = None
    role: Optional[str] = None
    start_anchor: Optional[str] = None
    end_anchor: Optional[str] = None
    start_word_idx: Optional[int] = None
    end_word_idx: Optional[int] = None
    text_he: Optional[str] = None
    text_ru: Optional[str] = None
    status_he: Optional[str] = None
    status_ru: Optional[str] = None
    comment: Optional[str] = None
    version: int = Field(..., ge=1)


class SegmentLinkItem(BaseModel):
    hebrew_segment_id: uuid.UUID
    russian_segment_id: uuid.UUID
    weight: Optional[float] = None
    is_primary: Optional[bool] = False


class LayoutCreate(BaseModel):
    name: Optional[str] = None
    is_canonical: Optional[bool] = False
    owner_user_id: Optional[uuid.UUID] = None
    layout_json: Optional[dict] = None


class LayoutUpdate(BaseModel):
    name: Optional[str] = None
    is_canonical: Optional[bool] = None
    layout_json: Optional[dict] = None


class DomainUpdate(BaseModel):
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    description: Optional[str] = None
    rules_json: Optional[dict] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class DomainCreate(BaseModel):
    id: str
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    description: Optional[str] = None
    rules_json: Optional[dict] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class NoteCreate(BaseModel):
    kind: Optional[str] = "note"
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    text_he: Optional[str] = None
    text_ru: Optional[str] = None
    color: Optional[str] = None
    domain_id: Optional[str] = None
    attached_node_id: Optional[uuid.UUID] = None
    attached_edge_id: Optional[uuid.UUID] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class NoteUpdate(BaseModel):
    kind: Optional[str] = None
    title_he: Optional[str] = None
    title_ru: Optional[str] = None
    text_he: Optional[str] = None
    text_ru: Optional[str] = None
    color: Optional[str] = None
    domain_id: Optional[str] = None
    attached_node_id: Optional[uuid.UUID] = None
    attached_edge_id: Optional[uuid.UUID] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


@router.get("/seder/map")
async def get_map(
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.list_map()


@router.get("/seder/node/{node_id}")
async def get_node(
    node_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    node = await service.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.post("/seder/node", status_code=status.HTTP_201_CREATED)
async def create_node(
    payload: NodeCreate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.create_node(payload.model_dump(exclude_unset=True))


@router.patch("/seder/node/{node_id}")
async def update_node(
    node_id: uuid.UUID,
    payload: NodeUpdate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    node = await service.update_node(node_id, payload.model_dump(exclude_unset=True))
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.delete("/seder/node/{node_id}")
async def delete_node(
    node_id: uuid.UUID,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    ok = await service.delete_node(node_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"ok": True}


@router.post("/seder/edge", status_code=status.HTTP_201_CREATED)
async def create_edge(
    payload: EdgeCreate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.create_edge(payload.model_dump(exclude_unset=True))


@router.delete("/seder/edge/{edge_id}")
async def delete_edge(
    edge_id: uuid.UUID,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    ok = await service.delete_edge(edge_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Edge not found")
    return {"ok": True}


@router.post("/seder/note", status_code=status.HTTP_201_CREATED)
async def create_note(
    payload: NoteCreate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.create_note(payload.model_dump(exclude_unset=True))


@router.patch("/seder/note/{note_id}")
async def update_note(
    note_id: uuid.UUID,
    payload: NoteUpdate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    note = await service.update_note(note_id, payload.model_dump(exclude_unset=True))
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.delete("/seder/note/{note_id}")
async def delete_note(
    note_id: uuid.UUID,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    ok = await service.delete_note(note_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"ok": True}


@router.post("/seder/article", status_code=status.HTTP_201_CREATED)
async def create_article(
    payload: ArticleCreate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.create_article(payload.model_dump(exclude_unset=True))


@router.get("/seder/article/{article_id}")
async def get_article(
    article_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    article = await service.get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.patch("/seder/article/{article_id}")
async def update_article(
    article_id: uuid.UUID,
    payload: ArticleUpdate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    article = await service.update_article(article_id, payload.model_dump(exclude_unset=True))
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.get("/seder/article/{article_id}/segments")
async def get_article_segments(
    article_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.list_segments(article_id)


@router.post("/seder/article/{article_id}/segments", status_code=status.HTTP_201_CREATED)
async def create_article_segments(
    article_id: uuid.UUID,
    payload: List[SegmentCreate],
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.create_segments(article_id, [p.model_dump() for p in payload])


@router.patch("/seder/segment/{segment_id}")
async def update_segment(
    segment_id: uuid.UUID,
    payload: SegmentUpdate,
    admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    try:
        segment = await service.update_segment(
            segment_id=segment_id,
            payload=payload.model_dump(exclude_unset=True),
            author_id=admin.id,
        )
    except ValueError as exc:
        if str(exc) == "version_conflict":
            raise HTTPException(status_code=409, detail="Version conflict")
        raise
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    return segment


@router.get("/seder/segment/{segment_id}/versions")
async def get_segment_versions(
    segment_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.list_segment_versions(segment_id)


@router.post("/seder/segment/{segment_id}/restore/{version_id}")
async def restore_segment_version(
    segment_id: uuid.UUID,
    version_id: uuid.UUID,
    admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    segment = await service.restore_segment_version(
        segment_id=segment_id,
        version_id=version_id,
        author_id=admin.id,
    )
    if not segment:
        raise HTTPException(status_code=404, detail="Segment or version not found")
    return segment


@router.post("/seder/segment/{segment_id}/links")
async def upsert_segment_links(
    segment_id: uuid.UUID,
    payload: List[SegmentLinkItem],
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.upsert_segment_links(segment_id, [p.model_dump() for p in payload])


@router.get("/seder/definitions")
async def list_definitions(
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.list_definitions()


@router.get("/seder/definition/{definition_id}/instances")
async def list_definition_instances(
    definition_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.list_definition_instances(definition_id)


@router.get("/seder/layouts")
async def list_layouts(
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.list_layouts()


@router.get("/seder/domains")
async def list_domains(
    _user: User = Depends(get_current_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.list_domains()


@router.post("/seder/domains", status_code=status.HTTP_201_CREATED)
async def create_domain(
    payload: DomainCreate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    return await service.create_domain(payload.model_dump(exclude_unset=True))


@router.post("/seder/layouts", status_code=status.HTTP_201_CREATED)
async def create_layout(
    payload: LayoutCreate,
    admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    data = payload.model_dump(exclude_unset=True)
    if data.get("owner_user_id") is None:
        data["owner_user_id"] = admin.id
    return await service.create_layout(data)


@router.patch("/seder/layouts/{layout_id}")
async def update_layout(
    layout_id: uuid.UUID,
    payload: LayoutUpdate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    layout = await service.update_layout(layout_id, payload.model_dump(exclude_unset=True))
    if not layout:
        raise HTTPException(status_code=404, detail="Layout not found")
    return layout


@router.patch("/seder/domains/{domain_id}")
async def update_domain(
    domain_id: str,
    payload: DomainUpdate,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    domain = await service.update_domain(domain_id, payload.model_dump(exclude_unset=True))
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    return domain


@router.delete("/seder/domains/{domain_id}")
async def delete_domain(
    domain_id: str,
    _admin: User = Depends(require_admin_user),
    service: SederMapService = Depends(get_seder_map_service),
):
    ok = await service.delete_domain(domain_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"ok": True}
