"""Unit tests for ChromaDB vector store module."""

import pytest
from app.embeddings.chroma_store import ChromaVectorStore


class TestChromaVectorStore:
    """Test ChromaVectorStore functionality."""

    def test_initialization(self):
        """Test vector store initialization."""
        try:
            store = ChromaVectorStore(
                collection_name="test_collection",
                persist_directory="./test_chroma",
            )
            assert store.collection_name == "test_collection"
            assert store.persist_directory == "./test_chroma"
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_add_documents_without_collection(self):
        """Test error when adding documents without collection."""
        try:
            store = ChromaVectorStore()
            with pytest.raises(ValueError, match="Collection not created"):
                store.add_documents(["test document"])
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_query_without_collection(self):
        """Test error when querying without collection."""
        try:
            store = ChromaVectorStore()
            with pytest.raises(ValueError, match="Collection not created"):
                store.query(["test query"])
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_create_collection(self):
        """Test collection creation."""
        try:
            store = ChromaVectorStore(collection_name="test_col_create")
            store.create_collection()
            assert store.collection is not None
            # Cleanup
            store.delete_collection()
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_add_and_retrieve_documents(self):
        """Test adding and retrieving documents."""
        try:
            store = ChromaVectorStore(collection_name="test_add_retrieve")
            store.create_collection()

            # Add documents
            texts = ["First document about cats.", "Second document about dogs."]
            store.add_documents(texts, ids=["1", "2"])

            # Query
            results = store.query(["cats"], n_results=1)
            assert results is not None
            assert len(results["documents"]) > 0

            # Cleanup
            store.delete_collection()
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_get_collection_info(self):
        """Test getting collection information."""
        try:
            store = ChromaVectorStore(collection_name="test_info")
            store.create_collection()

            info = store.get_collection_info()
            assert "name" in info
            assert "document_count" in info
            assert info["name"] == "test_info"

            # Cleanup
            store.delete_collection()
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_clear_collection(self):
        """Test clearing collection."""
        try:
            store = ChromaVectorStore(collection_name="test_clear")
            store.create_collection()

            # Add documents
            store.add_documents(["Document 1", "Document 2"])

            # Clear
            store.clear_collection()

            # Verify empty
            info = store.get_collection_info()
            assert info["document_count"] == 0

            # Cleanup
            store.delete_collection()
        except ImportError:
            pytest.skip("ChromaDB not installed")

    def test_delete_collection(self):
        """Test deleting collection."""
        try:
            store = ChromaVectorStore(collection_name="test_delete")
            store.create_collection()

            store.delete_collection()
            assert store.collection is None

        except ImportError:
            pytest.skip("ChromaDB not installed")
