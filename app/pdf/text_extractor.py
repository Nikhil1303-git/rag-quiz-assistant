"""PDF text extraction module.

Supports multiple extraction methods:
- pdfplumber: More robust and handles complex PDFs well
- pypdf: Lightweight alternative for simple PDFs
"""

from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extract text from PDF files."""

    def __init__(self, method: str = "pdfplumber"):
        """Initialize extractor.

        Args:
            method: Extraction method ("pdfplumber" or "pypdf")

        Raises:
            ValueError: If method is not supported
        """
        if method not in ["pdfplumber", "pypdf"]:
            raise ValueError(
                f"Unsupported extraction method: {method}. "
                "Must be 'pdfplumber' or 'pypdf'"
            )
        self.method = method
        logger.info(f"Initialized PDFTextExtractor with method: {method}")

    def extract_text(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page texts

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is corrupted or unreadable
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting text from {pdf_path} using {self.method}")

        if self.method == "pdfplumber":
            return self._extract_with_pdfplumber(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: Path) -> List[str]:
        """Extract text using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page texts
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for this extraction method. "
                "Install it with: pip install pdfplumber"
            )

        texts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        texts.append(text.strip())
                    else:
                        logger.warning(f"No text extracted from page {i + 1}")
        except Exception as e:
            raise ValueError(
                f"Failed to extract text from PDF: {str(e)}"
            ) from e

        logger.info(f"Extracted text from {len(texts)} pages")
        return texts

    @staticmethod
    def _extract_with_pypdf(pdf_path: Path) -> List[str]:
        """Extract text using pypdf.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page texts
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for this extraction method. "
                "Install it with: pip install pypdf"
            )

        texts = []
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    texts.append(text.strip())
                else:
                    logger.warning(f"No text extracted from page {i + 1}")
        except Exception as e:
            raise ValueError(
                f"Failed to extract text from PDF: {str(e)}"
            ) from e

        logger.info(f"Extracted text from {len(texts)} pages")
        return texts

    def extract_text_with_pages(
        self, pdf_path: Path
    ) -> List[dict]:
        """Extract text with page metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dicts with 'page_number', 'text', and 'length'
        """
        texts = self.extract_text(pdf_path)
        return [
            {
                "page_number": i + 1,
                "text": text,
                "length": len(text),
            }
            for i, text in enumerate(texts)
        ]
