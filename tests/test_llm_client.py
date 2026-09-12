"""Unit tests for LLM client module."""

import pytest
from app.llm.client import (
    LLMFactory,
    GroqClient,
    OpenAIClient,
    LaminiClient,
)


class TestLLMFactory:
    """Test LLMFactory functionality."""

    def test_factory_groq(self):
        """Test creating Groq client."""
        try:
            client = LLMFactory.create("groq", "test_key", "llama3-8b-8192")
            assert isinstance(client, GroqClient)
            assert client.provider == "groq"
        except ImportError:
            # Groq library not installed, skip test
            pytest.skip("Groq library not installed")

    def test_factory_openai(self):
        """Test creating OpenAI client."""
        try:
            client = LLMFactory.create("openai", "test_key", "gpt-3.5-turbo")
            assert isinstance(client, OpenAIClient)
            assert client.provider == "openai"
        except ImportError:
            pytest.skip("OpenAI library not installed")

    def test_factory_lamini(self):
        """Test creating Lamini client."""
        try:
            client = LLMFactory.create("lamini", "test_key", "test_model")
            assert isinstance(client, LaminiClient)
            assert client.provider == "lamini"
        except ImportError:
            pytest.skip("Lamini library not installed")

    def test_factory_invalid_provider(self):
        """Test error with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMFactory.create("invalid", "test_key", "test_model")

    def test_factory_case_insensitive(self):
        """Test that provider names are case-insensitive."""
        try:
            client = LLMFactory.create("GROQ", "test_key", "llama3-8b-8192")
            assert client.provider == "groq"
        except ImportError:
            pytest.skip("Groq library not installed")


class TestGroqClient:
    """Test GroqClient functionality."""

    def test_groq_initialization(self):
        """Test Groq client initialization."""
        try:
            client = GroqClient(api_key="test_key", model="llama3-8b-8192")
            assert client.model == "llama3-8b-8192"
            assert client.provider == "groq"
        except ImportError:
            pytest.skip("Groq library not installed")

    def test_groq_get_model_info(self):
        """Test getting Groq model info."""
        try:
            client = GroqClient(api_key="test_key", model="llama3-8b-8192")
            info = client.get_model_info()
            assert info["provider"] == "groq"
            assert info["model"] == "llama3-8b-8192"
        except ImportError:
            pytest.skip("Groq library not installed")


class TestOpenAIClient:
    """Test OpenAIClient functionality."""

    def test_openai_initialization(self):
        """Test OpenAI client initialization."""
        try:
            client = OpenAIClient(api_key="sk-test", model="gpt-3.5-turbo")
            assert client.model == "gpt-3.5-turbo"
            assert client.provider == "openai"
        except ImportError:
            pytest.skip("OpenAI library not installed")

    def test_openai_get_model_info(self):
        """Test getting OpenAI model info."""
        try:
            client = OpenAIClient(api_key="sk-test", model="gpt-3.5-turbo")
            info = client.get_model_info()
            assert info["provider"] == "openai"
            assert info["model"] == "gpt-3.5-turbo"
        except ImportError:
            pytest.skip("OpenAI library not installed")
