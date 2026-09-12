"""Unit tests for configuration module."""

import pytest
import os
from pathlib import Path
from app.config import (
    load_config,
    validate_config,
    LLMConfig,
    AppConfig,
)


class TestConfigLoading:
    """Test configuration loading from environment variables."""

    def test_load_config_with_groq(self, monkeypatch):
        """Test loading Groq configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_API_KEY", "test_groq_key")
        monkeypatch.setenv("LLM_MODEL", "llama3-8b-8192")

        config = load_config()

        assert config.llm.provider == "groq"
        assert config.llm.api_key == "test_groq_key"
        assert config.llm.model == "llama3-8b-8192"

    def test_load_config_with_openai(self, monkeypatch):
        """Test loading OpenAI configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "sk-test_openai_key")
        monkeypatch.setenv("LLM_MODEL", "gpt-3.5-turbo")

        config = load_config()

        assert config.llm.provider == "openai"
        assert config.llm.api_key == "sk-test_openai_key"

    def test_load_config_missing_api_key(self, monkeypatch):
        """Test error when API key is missing."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        with pytest.raises(ValueError, match="LLM_API_KEY"):
            load_config()

    def test_load_config_invalid_provider(self, monkeypatch):
        """Test error with invalid provider."""
        monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
        monkeypatch.setenv("LLM_API_KEY", "test_key")

        with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
            load_config()

    def test_load_config_default_model(self, monkeypatch):
        """Test default model selection per provider."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        monkeypatch.delenv("LLM_MODEL", raising=False)

        config = load_config()
        assert config.llm.model == "llama3-8b-8192"


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_valid_config(self, monkeypatch):
        """Test validation of valid configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_API_KEY", "test_key")

        config = load_config()
        assert validate_config(config) is True

    def test_validate_invalid_chunk_size(self, monkeypatch):
        """Test error with invalid chunk size."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        monkeypatch.setenv("PDF_CHUNK_SIZE", "-100")

        config = load_config()
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_config(config)

    def test_validate_chunk_overlap_too_large(self, monkeypatch):
        """Test error when overlap >= chunk_size."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        monkeypatch.setenv("PDF_CHUNK_SIZE", "100")
        monkeypatch.setenv("PDF_CHUNK_OVERLAP", "150")

        config = load_config()
        with pytest.raises(ValueError, match="chunk_overlap must be less"):
            validate_config(config)

    def test_validate_temperature_range(self, monkeypatch):
        """Test error with temperature out of range."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        monkeypatch.setenv("LLM_TEMPERATURE", "3.0")

        config = load_config()
        with pytest.raises(ValueError, match="temperature must be between"):
            validate_config(config)


class TestConfigDirectories:
    """Test directory creation and handling."""

    def test_create_input_output_directories(self, monkeypatch, tmp_path):
        """Test automatic directory creation."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        monkeypatch.setenv("INPUT_DIR", str(tmp_path / "input"))
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

        config = load_config()

        assert config.input_dir.exists()
        assert config.output_dir.exists()
