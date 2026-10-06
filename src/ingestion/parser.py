from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ParsedDocument(BaseModel):
    source_name: str
    file_path: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentParser:
    """
    Robust document parser that extracts text, markdown tables, and metadata.
    """

    @staticmethod
    def parse_file(file_path: Path | str) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Extract title if present (first markdown header)
        lines = raw_text.splitlines()
        title = path.stem.replace("_", " ").title()
        for line in lines:
            if line.startswith("# "):
                title = line.replace("# ", "").strip()
                break

        return ParsedDocument(
            source_name=path.name,
            file_path=str(path.resolve()),
            content=raw_text,
            metadata={
                "title": title,
                "file_extension": path.suffix,
                "character_count": len(raw_text),
                "line_count": len(lines)
            }
        )

    @staticmethod
    def parse_directory(dir_path: Path | str) -> List[ParsedDocument]:
        directory = Path(dir_path)
        documents = []
        for file in directory.glob("*.*"):
            if file.suffix.lower() in [".md", ".txt", ".json", ".csv"]:
                documents.append(DocumentParser.parse_file(file))
        return documents
