import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .parser import ParsedDocument

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source_name: str
    section_title: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SemanticTableChunker:
    """
    Table-aware, hierarchical semantic chunker.
    Preserves markdown table integrity and prepends section headers
    so chunks retain global semantic context.
    """

    def __init__(self, target_chunk_size: int = 500, overlap_size: int = 80):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        raw_text = doc.content
        # Split by markdown headers (# or ##) to create semantic sections
        sections = re.split(r'\n(?=#{1,3}\s)', raw_text)
        
        chunks: List[DocumentChunk] = []
        global_index = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            lines = section.splitlines()
            section_title = "General Overview"
            if lines and lines[0].startswith("#"):
                section_title = lines[0].lstrip("#").strip()

            # Identify if section has markdown tables
            table_blocks = self._extract_table_blocks(section)

            if table_blocks:
                # If there are tables, keep each table as its own coherent chunk
                for idx, block in enumerate(table_blocks):
                    chunk_text = f"[{doc.metadata.get('title', doc.source_name)} - {section_title}]\n\n{block}"
                    chunk_id = f"{doc.source_name}_sec{len(chunks)}_tbl{idx}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source_name=doc.source_name,
                        section_title=section_title,
                        chunk_index=global_index,
                        metadata={
                            "is_table": "|" in block,
                            "char_count": len(chunk_text),
                            "word_count": len(chunk_text.split())
                        }
                    ))
                    global_index += 1
            else:
                # Regular text block - use paragraph-based chunking with sliding context
                paras = [p.strip() for p in section.split("\n\n") if p.strip()]
                current_text = f"[{doc.metadata.get('title', doc.source_name)} - {section_title}]\n\n"
                
                for p in paras:
                    if len(current_text) + len(p) > self.target_chunk_size and len(current_text) > 100:
                        chunk_id = f"{doc.source_name}_chunk{global_index}"
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            text=current_text.strip(),
                            source_name=doc.source_name,
                            section_title=section_title,
                            chunk_index=global_index,
                            metadata={
                                "is_table": False,
                                "char_count": len(current_text),
                                "word_count": len(current_text.split())
                            }
                        ))
                        global_index += 1
                        current_text = f"[{section_title}]\n\n{p}\n\n"
                    else:
                        current_text += p + "\n\n"

                if current_text.strip():
                    chunk_id = f"{doc.source_name}_chunk{global_index}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=current_text.strip(),
                        source_name=doc.source_name,
                        section_title=section_title,
                        chunk_index=global_index,
                        metadata={
                            "is_table": False,
                            "char_count": len(current_text),
                            "word_count": len(current_text.split())
                        }
                    ))
                    global_index += 1

        return chunks

    def _extract_table_blocks(self, section_text: str) -> List[str]:
        """
        Splits section into narrative paragraphs vs complete markdown tables.
        """
        lines = section_text.splitlines()
        blocks = []
        current_block = []
        in_table = False

        for line in lines:
            is_table_line = bool(re.match(r'^\s*\|.*\|\s*$', line))
            if is_table_line:
                if not in_table:
                    if current_block:
                        text_blk = "\n".join(current_block).strip()
                        if text_blk:
                            blocks.append(text_blk)
                        current_block = []
                    in_table = True
                current_block.append(line)
            else:
                if in_table:
                    # Table ended
                    tbl_blk = "\n".join(current_block).strip()
                    if tbl_blk:
                        blocks.append(tbl_blk)
                    current_block = []
                    in_table = False
                current_block.append(line)

        if current_block:
            rem = "\n".join(current_block).strip()
            if rem:
                blocks.append(rem)

        return blocks if blocks else [section_text]
