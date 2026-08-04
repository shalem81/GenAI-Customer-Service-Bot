"""
Multilingual Support Module

Task 6

Features
--------
- Detect language
- Translate text
- Validate supported languages
"""

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

from modules.config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
)


class MultilingualAI:

    def __init__(self):

        self.supported_languages = SUPPORTED_LANGUAGES

        self.default_language = DEFAULT_LANGUAGE

    # ---------------------------------------------------------
    # Detect language
    # ---------------------------------------------------------

    def detect_language(self, text):

        if not text or not text.strip():

            return {
                "success": False,
                "language": None,
                "message": "Empty text.",
            }

        try:

            language = detect(text)

            return {
                "success": True,
                "language": language,
                "name": self.supported_languages.get(
                    language,
                    "Unknown",
                ),
            }

        except LangDetectException:

            return {
                "success": False,
                "language": None,
                "message": "Unable to detect language.",
            }

    # ---------------------------------------------------------
    # Translate
    # ---------------------------------------------------------

    def translate(
        self,
        text,
        target_language="en",
    ):

        if target_language not in self.supported_languages:

            return {
                "success": False,
                "message": "Unsupported language.",
            }

        try:

            detected = self.detect_language(text)

            source = (
                detected["language"]
                if detected["success"]
                else "auto"
            )

            translated = GoogleTranslator(
                source=source,
                target=target_language,
            ).translate(text)

            return {
                "success": True,
                "original": text,
                "translated": translated,
                "source_language": source,
                "target_language": target_language,
            }

        except Exception as error:

            return {
                "success": False,
                "message": str(error),
            }


if __name__ == "__main__":

    translator = MultilingualAI()

    print("=" * 60)
    print("MULTILINGUAL AI")
    print("=" * 60)

    while True:

        text = input("\nEnter text (exit to quit): ")

        if text.lower() == "exit":
            break

        target = input(
            "Target language (en/te/hi/ta): "
        ).strip()

        result = translator.translate(
            text,
            target,
        )

        print()

        print(result)