"""PDF table extraction module.

Supports two approaches:
1. Direct extraction using pdfplumber
2. LLM-based extraction for complex tables
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class PDFTableExtractor:
    """Extract tables from PDF files."""

    def __init__(self, method: str = "pdfplumber"):
        """Initialize table extractor.

        Args:
            method: Extraction method ("pdfplumber" or "llm")
        """
        if method not in ["pdfplumber", "llm"]:
            raise ValueError(
                f"Unsupported extraction method: {method}. "
                "Must be 'pdfplumber' or 'llm'"
            )
        self.method = method
        logger.info(f"Initialized PDFTableExtractor with method: {method}")

    def extract_tables(self, pdf_path: Path) -> List[List[Dict[str, Any]]]:
        """Extract tables from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of tables, where each table is a list of row dictionaries

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(
            f"Extracting tables from {pdf_path} using {self.method}"
        )

        if self.method == "pdfplumber":
            return self._extract_with_pdfplumber(pdf_path)
        else:
            raise NotImplementedError(
                "LLM-based table extraction requires LLM client configuration"
            )

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: Path) -> List[List[Dict]]:
        """Extract tables using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of tables
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for table extraction. "
                "Install it with: pip install pdfplumber"
            )

        all_tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue

                        # Convert table to list of dicts
                        headers = table[0]
                        rows = []
                        for row in table[1:]:
                            row_dict = {
                                str(headers[i]): row[i]
                                for i in range(len(headers))
                            }
                            rows.append(row_dict)

                        if rows:
                            all_tables.append(rows)
                            logger.info(
                                f"Extracted table from page "
                                f"{page_num + 1}, table {table_idx + 1}: "
                                f"{len(rows)} rows"
                            )
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            raise ValueError(
                f"Failed to extract tables from PDF: {str(e)}"
            ) from e

        logger.info(f"Extracted {len(all_tables)} tables total")
        return all_tables

    def extract_tables_to_json(
        self, pdf_path: Path, output_path: Path = None
    ) -> str:
        """Extract tables and return as JSON.

        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save JSON output

        Returns:
            JSON string representation of tables
        """
        tables = self.extract_tables(pdf_path)
        json_str = json.dumps(tables, indent=2)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
            logger.info(f"Saved tables to {output_path}")

        return json_str
