"""
Block Stream Service for streaming doc.v1 content block by block.
"""

import json
import re
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BlockEvent:
    """Event for streaming blocks"""
    type: str  # 'block_start', 'block_delta', 'block_end'
    block_index: int
    block_type: str
    data: Dict[str, Any]
    timestamp: float

class BlockStreamService:
    """Service for streaming doc.v1 blocks in real-time"""
    
    def __init__(self):
        self.buffer = ""
        self.current_blocks = []
        self.block_index = 0
        self.in_code_block = False
        self.code_block_content = ""
        
    def _get_block_id(self, block_type: str, content: str) -> str:
        """Generate stable block ID"""
        return f"{block_type}_{hash(content) % 10000}"
    
    def _extract_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete JSON objects from text"""
        objects = []
        brace_count = 0
        start = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    try:
                        obj = json.loads(text[start:i+1])
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
        
        return objects
    
    def _extract_blocks_from_buffer(self) -> List[Dict[str, Any]]:
        """Extract blocks from markdown-like text"""
        blocks = []
        
        # Remove processed content
        remaining = self.buffer
        
        # Extract headings
        heading_pattern = r'^#+\s+(.+)$'
        for match in re.finditer(heading_pattern, remaining, re.MULTILINE):
            level = len(match.group(0).split()[0])
            text = match.group(1).strip()
            blocks.append({
                "type": "heading",
                "level": min(6, max(1, level)),
                "text": text
            })
        
        # Extract code blocks
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(code_pattern, remaining, re.DOTALL):
            lang = match.group(1) or ""
            code = match.group(2).strip()
            blocks.append({
                "type": "code",
                "lang": lang,
                "code": code
            })
        
        # Extract quotes
        quote_pattern = r'^>\s*(.+)$'
        for match in re.finditer(quote_pattern, remaining, re.MULTILINE):
            text = match.group(1).strip()
            blocks.append({
                "type": "quote",
                "text": text
            })
        
        # Extract lists
        list_pattern = r'^[\s]*[-*+]\s+(.+)$'
        list_items = []
        for match in re.finditer(list_pattern, remaining, re.MULTILINE):
            list_items.append(match.group(1).strip())
        
        if list_items:
            blocks.append({
                "type": "list",
                "ordered": False,
                "items": list_items
            })
        
        # Extract ordered lists
        ordered_list_pattern = r'^[\s]*\d+\.\s+(.+)$'
        ordered_items = []
        for match in re.finditer(ordered_list_pattern, remaining, re.MULTILINE):
            ordered_items.append(match.group(1).strip())
        
        if ordered_items:
            blocks.append({
                "type": "list",
                "ordered": True,
                "items": ordered_items
            })
        
        # Clean up processed content
        for pattern in [heading_pattern, code_pattern, quote_pattern, list_pattern, ordered_list_pattern]:
            remaining = re.sub(pattern, '', remaining, flags=re.MULTILINE | re.DOTALL)
        
        # Extract paragraphs from remaining text
        paragraphs = [p.strip() for p in remaining.split('\n\n') if p.strip()]
        for para in paragraphs:
            if para and not para.startswith('#') and not para.startswith('>') and not para.startswith('-') and not para.startswith('*') and not para.startswith('+'):
                blocks.append({
                    "type": "paragraph",
                    "text": para
                })
        
        return blocks
    
    async def stream_blocks_from_text(
        self, 
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream blocks from text input"""
        
        async for chunk in text_stream:
            self.buffer += chunk
            
            # Try to extract JSON objects first
            json_objects = self._extract_json_objects(self.buffer)
            
            if json_objects:
                # Process JSON objects
                for obj in json_objects:
                    if "blocks" in obj and isinstance(obj["blocks"], list):
                        # This is a doc.v1 object
                        for block in obj["blocks"]:
                            yield json.dumps({
                                "type": "block",
                                "data": {
                                    "block_index": self.block_index,
                                    "block_type": block.get("type", "unknown"),
                                    "content": block
                                }
                            }) + '\n'
                            self.block_index += 1
                        
                        # Remove processed JSON from buffer
                        try:
                            json_str = json.dumps(obj)
                            self.buffer = self.buffer.replace(json_str, "", 1)
                        except:
                            pass
                    else:
                        # Regular JSON object, treat as paragraph
                        yield json.dumps({
                            "type": "block",
                            "data": {
                                "block_index": self.block_index,
                                "block_type": "paragraph",
                                "content": {
                                    "type": "paragraph",
                                    "text": json.dumps(obj)
                                }
                            }
                        }) + '\n'
                        self.block_index += 1
            else:
                # Extract blocks from markdown-like text
                blocks = self._extract_blocks_from_buffer()
                
                for block in blocks:
                    yield json.dumps({
                        "type": "block",
                        "data": {
                            "block_index": self.block_index,
                            "block_type": block["type"],
                            "content": block
                        }
                    }) + '\n'
                    self.block_index += 1
                
                # Clean buffer after processing
                self.buffer = ""
        
        # Process any remaining content
        if self.buffer.strip():
            blocks = self._extract_blocks_from_buffer()
            for block in blocks:
                yield {
                    "type": "block",
                    "data": {
                        "block_index": self.block_index,
                        "block_type": block["type"],
                        "content": block
                    }
                }
                self.block_index += 1
    
    async def stream_blocks_from_json(
        self, 
        json_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream blocks from JSON input"""
        
        async for json_obj in json_stream:
            if "blocks" in json_obj and isinstance(json_obj["blocks"], list):
                for block in json_obj["blocks"]:
                    yield {
                        "type": "block",
                        "data": {
                            "block_index": self.block_index,
                            "block_type": block.get("type", "unknown"),
                            "content": block
                        }
                    }
                    self.block_index += 1
            else:
                # Treat as single block
                yield json.dumps({
                    "type": "block",
                    "data": {
                        "block_index": self.block_index,
                        "block_type": "paragraph",
                        "content": {
                            "type": "paragraph",
                            "text": json.dumps(json_obj)
                        }
                    }
                }) + '\n'
                self.block_index += 1