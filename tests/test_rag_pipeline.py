"""Unit tests for RAG pipeline module."""

import pytest
from unittest.mock import Mock, MagicMock
from app.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    """Test RAGPipeline functionality."""

    def create_mock_pipeline(self):
        """Create a pipeline with mocked components."""
        mock_pdf_extractor = Mock()
        mock_text_chunker = Mock()
        mock_vector_store = Mock()
        mock_llm_client = Mock()

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf_extractor,
            text_chunker=mock_text_chunker,
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        return pipeline, (
            mock_pdf_extractor,
            mock_text_chunker,
            mock_vector_store,
            mock_llm_client,
        )

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline, _ = self.create_mock_pipeline()
        assert pipeline is not None

    def test_ingest_pdf(self):
        """Test PDF ingestion."""
        pipeline, mocks = self.create_mock_pipeline()
        pdf_extractor, text_chunker, vector_store, llm_client = mocks

        # Setup mocks
        pdf_extractor.extract_text.return_value = ["Page 1 text", "Page 2 text"]
        text_chunker.chunk_texts.return_value = [
            "Chunk 1",
            "Chunk 2",
            "Chunk 3",
        ]
        vector_store.create_collection.return_value = None
        vector_store.add_documents.return_value = None

        # Test
        result = pipeline.ingest_pdf("test.pdf")

        assert result["success"] is True
        assert result["pages_extracted"] == 2
        assert result["chunks_created"] == 3
        pdf_extractor.extract_text.assert_called_once()
        text_chunker.chunk_texts.assert_called_once()
        vector_store.add_documents.assert_called_once()

    def test_retrieve(self):
        """Test document retrieval."""
        pipeline, mocks = self.create_mock_pipeline()
        pdf_extractor, text_chunker, vector_store, llm_client = mocks

        # Setup mock
        vector_store.query.return_value = {
            "documents": [["Doc 1", "Doc 2", "Doc 3"]]
        }

        # Test
        documents = pipeline.retrieve("test query", n_results=3)

        assert len(documents) == 3
        assert documents[0] == "Doc 1"
        vector_store.query.assert_called_once()

    def test_generate_response(self):
        """Test response generation."""
        pipeline, mocks = self.create_mock_pipeline()
        pdf_extractor, text_chunker, vector_store, llm_client = mocks

        # Setup mock
        llm_client.generate.return_value = "Generated response"

        # Test
        response = pipeline.generate_response(
            query="test query",
            retrieved_documents=["Doc 1", "Doc 2"],
        )

        assert response == "Generated response"
        llm_client.generate.assert_called_once()
        
        # Check that prompt was constructed
        call_args = llm_client.generate.call_args
        assert "test query" in call_args.kwargs["prompt"]

    def test_rag_query(self):
        """Test complete RAG query."""
        pipeline, mocks = self.create_mock_pipeline()
        pdf_extractor, text_chunker, vector_store, llm_client = mocks

        # Setup mocks
        vector_store.query.return_value = {
            "documents": [["Doc 1", "Doc 2"]]
        }
        llm_client.generate.return_value = "Final response"

        # Test
        result = pipeline.rag_query("test query")

        assert result["query"] == "test query"
        assert result["n_documents_retrieved"] == 2
        assert result["response"] == "Final response"
        vector_store.query.assert_called_once()
        llm_client.generate.assert_called_once()

    def test_get_status(self):
        """Test getting pipeline status."""
        pipeline, mocks = self.create_mock_pipeline()
        pdf_extractor, text_chunker, vector_store, llm_client = mocks

        # Setup mocks
        pdf_extractor.method = "pdfplumber"
        text_chunker.chunk_size = 500
        text_chunker.chunk_overlap = 0
        vector_store.get_collection_info.return_value = {
            "name": "test_collection",
            "document_count": 10,
        }
        llm_client.get_model_info.return_value = {
            "provider": "groq",
            "model": "llama3-8b-8192",
        }

        # Test
        status = pipeline.get_status()

        assert status["llm_provider"] == "groq"
        assert status["pdf_extractor"] == "pdfplumber"
        assert status["chunk_size"] == 500
        assert status["vector_store"] == "chromadb"

    def test_retrieve_empty_results(self):
        """Test retrieval with no results."""
        pipeline, mocks = self.create_mock_pipeline()
        pdf_extractor, text_chunker, vector_store, llm_client = mocks

        # Setup mock for empty results
        vector_store.query.return_value = {"documents": [[]]}

        # Test
        documents = pipeline.retrieve("query")

        assert documents == []
