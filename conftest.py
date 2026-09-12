"""Pytest configuration."""

import os
import pytest


@pytest.fixture
def monkeypatch():
    """Provide monkeypatch fixture."""
    from _pytest.monkeypatch import MonkeyPatch
    m = MonkeyPatch()
    yield m
    m.undo()


def pytest_configure(config):
    """Configure pytest."""
    # Suppress warnings during tests
    os.environ.setdefault("LLM_PROVIDER", "groq")
    os.environ.setdefault("LLM_API_KEY", "test_key")
