"""Document loading and chunking without external langchain loaders.

Removes dependency on `langchain_community.document_loaders` and
`langchain_text_splitters` by using simple custom implementations so we
avoid installation conflicts (numpy<2 constraint) while still producing
`langchain.schema.Document` objects for downstream vector store usage.
"""

from pathlib import Path
from typing import List
from langchain.schema import Document
from math import ceil

try:
    from pypdf import PdfReader  # lightweight PDF extraction
except Exception:  # pragma: no cover
    PdfReader = None


class DocumentLoader:
    """Utility class for loading and chunking documents (txt, md, pdf)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        documents: List[Document] = []
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")

        supported = {".txt", ".md", ".pdf"}

        for file_path in directory.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in supported:
                continue
            try:
                suffix = file_path.suffix.lower()
                if suffix in {".txt", ".md"}:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    documents.append(
                        Document(page_content=text, metadata={"source": str(file_path)})
                    )
                elif suffix == ".pdf":
                    if not PdfReader:
                        print(f"pypdf not available, skipping PDF {file_path}")
                        continue
                    reader = PdfReader(str(file_path))
                    for page_index, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={
                                    "source": str(file_path),
                                    "page": page_index + 1,
                                    "total_pages": len(reader.pages),
                                },
                            )
                        )
            except Exception as e:  # pragma: no cover
                print(f"Error loading {file_path}: {e}")
                continue

        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using a simple overlapping character window."""
        if not documents:
            return []
        chunked: List[Document] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for doc in documents:
            text = doc.page_content
            length = len(text)
            if length <= self.chunk_size:
                chunked.append(doc)
                continue
            for start in range(0, length, step):
                end = min(start + self.chunk_size, length)
                segment = text[start:end]
                if not segment.strip():
                    continue
                meta = dict(doc.metadata)
                meta.update({"chunk_start": start, "chunk_end": end})
                chunked.append(Document(page_content=segment, metadata=meta))
                if end >= length:
                    break
        return chunked
    
    def load_and_chunk(self, directory_path: str) -> List[Document]:
        """
        Load documents from directory and chunk them
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_directory(directory_path)
        chunks = self.chunk_documents(documents)
        print(f"Loaded {len(documents)} documents, created {len(chunks)} chunks from {directory_path}")
        return chunks

