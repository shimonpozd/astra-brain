from pydantic import BaseModel, Field
from typing import List, Literal, Union, Optional, Dict, Any

# Block Types
class HeadingBlock(BaseModel):
    type: Literal['heading']
    level: Literal[1, 2, 3, 4, 5, 6]
    text: str
    lang: Optional[str] = None
    dir: Optional[Literal['rtl', 'ltr', 'auto']] = None

class ParagraphBlock(BaseModel):
    type: Literal['paragraph']
    text: str
    lang: Optional[str] = None
    dir: Optional[Literal['rtl', 'ltr', 'auto']] = None

class QuoteBlock(BaseModel):
    type: Literal['quote']
    text: str
    source: Optional[str] = None
    lang: Optional[str] = None
    dir: Optional[Literal['rtl', 'ltr', 'auto']] = None

class ListBlock(BaseModel):
    type: Literal['list']
    ordered: Optional[bool] = False
    items: List[str]

class TermBlock(BaseModel):
    type: Literal['term']
    he: str
    ru: Optional[str] = None

class CalloutBlock(BaseModel):
    type: Literal['callout']
    variant: Literal['info', 'warn', 'success', 'danger']
    text: str

class ActionBlock(BaseModel):
    type: Literal['action']
    label: str
    actionId: str
    params: Optional[Dict[str, Any]] = None

class CodeBlock(BaseModel):
    type: Literal['code']
    lang: Optional[str] = None
    code: str

Block = Union[HeadingBlock, ParagraphBlock, QuoteBlock, ListBlock, TermBlock, CalloutBlock, ActionBlock, CodeBlock]

# Op Types
class LinksOp(BaseModel):
    op: Literal['links']
    tref: str
    cat: Optional[Literal['Commentary', 'All']] = None

class TextOp(BaseModel):
    op: Literal['text']
    tref: str

class CommentaryIndexOp(BaseModel):
    op: Literal['commentaryIndex']
    tref: str

class NavOp(BaseModel):
    op: Literal['nav']
    delta: Literal[1, -1]
    baseTref: str

class RecallOp(BaseModel):
    op: Literal['recall']
    tref: Optional[str] = None

Op = Union[LinksOp, TextOp, CommentaryIndexOp, NavOp, RecallOp]

# Main DocV1 Type
class DocV1(BaseModel):
    version: Literal['1.0'] = '1.0'
    ops: Optional[List[Op]] = None
    blocks: List[Block]

# ChatMessage
class ChatMessage(BaseModel):
    id: str
    role: Literal['user', 'assistant', 'system']
    created_at: str
    content_type: Literal['doc.v1', 'text.v1']
    content: Union[DocV1, str]
    meta: Optional[Dict[str, Any]] = None


































