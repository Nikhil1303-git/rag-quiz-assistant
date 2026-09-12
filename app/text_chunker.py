"""Text chunking utilities for splitting documents."""

from typing import List
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Split text into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
    ):
        """Initialize text chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized TextChunker: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Uses a recursive character splitter that tries to preserve
        semantic boundaries (paragraphs, sentences, words).

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_text(text)
        logger.info(
            f"Split text into {len(chunks)} chunks "
            f"(avg size: {len(text) // len(chunks) if chunks else 0} chars)"
        )

        return chunks

    def chunk_texts(self, texts: List[str]) -> List[str]:
        """Split multiple texts into chunks.

        Args:
            texts: List of texts to chunk

        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        for text in texts:
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_with_metadata(
        self, text: str, source: str = None, page: int = None
    ) -> List[dict]:
        """Chunk text while preserving metadata.

        Args:
            text: Text to chunk
            source: Source document identifier
            page: Page number (if applicable)

        Returns:
            List of dicts with chunk text and metadata
        """
        chunks = self.chunk_text(text)
        return [
            {
                "text": chunk,
                "source": source,
                "page": page,
                "chunk_size": len(chunk),
            }
            for chunk in chunks
        ]
