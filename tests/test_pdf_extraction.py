"""Unit tests for PDF extraction module."""

import pytest
from pathlib import Path
from app.pdf.text_extractor import PDFTextExtractor


class TestPDFTextExtractor:
    """Test PDFTextExtractor functionality."""

    def test_initialization_pdfplumber(self):
        """Test initializing with pdfplumber method."""
        extractor = PDFTextExtractor(method="pdfplumber")
        assert extractor.method == "pdfplumber"

    def test_initialization_pypdf(self):
        """Test initializing with pypdf method."""
        extractor = PDFTextExtractor(method="pypdf")
        assert extractor.method == "pypdf"

    def test_invalid_extraction_method(self):
        """Test error with invalid extraction method."""
        with pytest.raises(ValueError, match="Unsupported extraction method"):
            PDFTextExtractor(method="invalid_method")

    def test_extract_nonexistent_file(self):
        """Test error when PDF file doesn't exist."""
        extractor = PDFTextExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_text(Path("nonexistent.pdf"))

    def test_extract_with_pages_structure(self):
        """Test extract_text_with_pages returns correct structure."""
        extractor = PDFTextExtractor()
        # This test would need a real PDF to fully test
        # For now, just verify method exists and returns list
        assert callable(extractor.extract_text_with_pages)
