"""
Configuration Module
GenAI Customer Service Bot

Centralized configuration for:
- Environment variables
- Project paths
- Gemini API
- Supported languages
- Supported image formats
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

VECTOR_DB_DIR = BASE_DIR / "vector_db"

MODULES_DIR = BASE_DIR / "modules"

TESTS_DIR = BASE_DIR / "tests"

SCREENSHOTS_DIR = BASE_DIR / "screenshots"

REPORT_DIR = BASE_DIR / "report"

# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

APP_NAME = os.getenv(
    "APP_NAME",
    "GenAI Customer Service Bot",
)

APP_ENV = os.getenv(
    "APP_ENV",
    "development",
)

DEBUG = (
    os.getenv(
        "DEBUG",
        "False",
    ).lower()
    == "true"
)

# ---------------------------------------------------------
# Ollama Configuration
# ---------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

DEFAULT_OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3",
)

DEFAULT_OLLAMA_VISION_MODEL = os.getenv(
    "OLLAMA_VISION_MODEL",
    "llava",
)


# ---------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------

VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    str(VECTOR_DB_DIR),
)

# ---------------------------------------------------------
# Multilingual
# ---------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
}

DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------
# Multimodal
# ---------------------------------------------------------

SUPPORTED_IMAGE_TYPES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)

MAX_IMAGE_SIZE_MB = 10

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_directories():
    """
    Create project directories if they do not exist.
    """

    for directory in [
        DATA_DIR,
        VECTOR_DB_DIR,
        SCREENSHOTS_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )