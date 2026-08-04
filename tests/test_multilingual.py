"""
Tests for Multilingual AI Module.
"""

import pytest

from modules.multilingual import MultilingualAI


@pytest.fixture
def translator():
    return MultilingualAI()


def test_detect_english(translator):
    result = translator.detect_language(
        "Hello, how are you?"
    )

    assert result["success"] is True
    assert result["language"] == "en"


def test_detect_telugu(translator):
    result = translator.detect_language(
        "నువ్వు ఎలా ఉన్నావు?"
    )

    assert result["success"] is True


def test_empty_text(translator):
    result = translator.detect_language("")

    assert result["success"] is False


def test_supported_languages(translator):

    assert "en" in translator.supported_languages
    assert "te" in translator.supported_languages
    assert "hi" in translator.supported_languages
    assert "ta" in translator.supported_languages


def test_invalid_target_language(translator):

    result = translator.translate(
        "Hello",
        "xyz",
    )

    assert result["success"] is False