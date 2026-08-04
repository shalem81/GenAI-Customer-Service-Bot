"""
Tests for Multimodal AI Module.
"""

from pathlib import Path

import pytest

from modules.multimodal import MultimodalAI


@pytest.fixture
def bot():
    return MultimodalAI()


def test_api_status(bot):
    assert isinstance(bot.is_available(), bool)


def test_missing_image(bot):
    result = bot.analyze_image(
        "does_not_exist.jpg",
        "Describe this image."
    )

    assert result["success"] is False
    assert "exist" in result["error"].lower()


def test_invalid_extension(bot, tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Not an image")

    result = bot.analyze_image(
        str(file),
        "Describe this."
    )

    assert result["success"] is False


def test_supported_formats():
    from modules.config import SUPPORTED_IMAGE_TYPES

    assert ".jpg" in SUPPORTED_IMAGE_TYPES
    assert ".jpeg" in SUPPORTED_IMAGE_TYPES
    assert ".png" in SUPPORTED_IMAGE_TYPES


def test_default_model():
    from modules.config import DEFAULT_OLLAMA_VISION_MODEL

    assert isinstance(DEFAULT_OLLAMA_VISION_MODEL, str)
    assert DEFAULT_OLLAMA_VISION_MODEL