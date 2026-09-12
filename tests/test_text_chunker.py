"""Unit tests for text chunking module."""

import pytest
from app.text_chunker import TextChunker


class TestTextChunker:
    """Test TextChunker functionality."""

    def test_initialization(self):
        """Test chunker initialization."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_invalid_chunk_size(self):
        """Test error with negative chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-100)

    def test_invalid_chunk_overlap(self):
        """Test error with negative overlap."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            TextChunker(chunk_size=500, chunk_overlap=-10)

    def test_overlap_exceeds_chunk_size(self):
        """Test error when overlap exceeds chunk size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less"):
            TextChunker(chunk_size=100, chunk_overlap=200)

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = TextChunker(chunk_size=500)
        text = "This is a short text."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        assert "short text" in chunks[0]

    def test_chunk_respects_size(self):
        """Test that chunks respect size limits."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a longer text. " * 20
        chunks = chunker.chunk_text(text)
        
        # Most chunks should be within size limit (some flexibility for words)
        for chunk in chunks:
            assert len(chunk) <= 150  # Allow some flexibility

    def test_chunk_with_metadata(self):
        """Test chunking with metadata."""
        chunker = TextChunker()
        text = "This is a test document."
        chunks_meta = chunker.chunk_with_metadata(
            text,
            source="test.pdf",
            page=1
        )
        
        assert len(chunks_meta) > 0
        for chunk_dict in chunks_meta:
            assert "text" in chunk_dict
            assert "source" in chunk_dict
            assert "page" in chunk_dict
            assert chunk_dict["source"] == "test.pdf"
            assert chunk_dict["page"] == 1

    def test_chunk_multiple_texts(self):
        """Test chunking multiple texts."""
        chunker = TextChunker(chunk_size=100)
        texts = [
            "First document. " * 10,
            "Second document. " * 10,
        ]
        chunks = chunker.chunk_texts(texts)
        assert len(chunks) > 0

    def test_chunk_preserves_content(self):
        """Test that chunking preserves original content."""
        chunker = TextChunker(chunk_size=100)
        original_text = "The quick brown fox jumps. " * 5
        chunks = chunker.chunk_text(original_text)
        combined = " ".join(chunks)
        
        # Original words should be present
        assert "quick" in combined
        assert "brown" in combined
        assert "fox" in combined
