"""
Multimodal AI Module
GenAI Customer Service Bot

Features:
- Image validation
- Ollama Vision integration (e.g. llava, llama3.2-vision)
- Image analysis and question answering
- Image metadata extraction fallback when vision models are not pulled
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from PIL import Image

from modules.config import (
    SUPPORTED_IMAGE_TYPES,
    MAX_IMAGE_SIZE_MB,
    DEFAULT_OLLAMA_VISION_MODEL,
)
from modules.ollama_client import OllamaClient


class MultimodalAI:
    """
    Ollama Vision wrapper.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model_name: Optional[str] = None,
    ):
        self.client = ollama_client or OllamaClient()
        self.model_name = model_name or self.client.default_vision_model

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        status = self.client.check_connection()
        return status.get("connected", False)

    def validate_image(self, image_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate format and size of uploaded image."""
        img_path = Path(image_path)

        if not img_path.exists():
            return False, "Image file does not exist."

        if img_path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
            return False, f"Unsupported image format. Allowed: {', '.join(SUPPORTED_IMAGE_TYPES)}"

        size_mb = img_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            return False, f"Image exceeds maximum size of {MAX_IMAGE_SIZE_MB} MB."

        return True, "OK"

    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str = "Describe this image in detail.",
        model: Optional[str] = None,
    ) -> Dict[str, Union[bool, str]]:
        """Analyze image using Ollama vision model."""
        valid, message = self.validate_image(image_path)
        if not valid:
            return {"success": False, "error": message}

        if not self.is_available():
            return {
                "success": False,
                "error": "Ollama service is not connected. Make sure Ollama is running at http://localhost:11434.",
            }

        selected_model = model or self.model_name
        result = self.client.analyze_image(
            image_path=image_path,
            prompt=prompt,
            model=selected_model,
        )

        # Fallback if specific vision model is not pulled yet
        if not result["success"]:
            try:
                img = Image.open(image_path)
                format_info = f"Format: {img.format}, Dimensions: {img.size[0]}x{img.size[1]}px, Mode: {img.mode}"
                
                text_prompt = (
                    f"The user uploaded an image with details ({format_info}) and asked: '{prompt}'. "
                    f"Notice: The vision model '{selected_model}' returned an error ({result.get('error')}). "
                    "Provide a helpful response acknowledging the image upload details."
                )
                text_response = self.client.generate(text_prompt)
                if text_response:
                    return {
                        "success": True,
                        "response": f"📷 **Image Details**: {format_info}\n\n{text_response}",
                    }
            except Exception:
                pass

        return result

    def ask(
        self,
        image_path: Union[str, Path],
        question: str,
        model: Optional[str] = None,
    ) -> Dict[str, Union[bool, str]]:
        """Ask a question about an image."""
        return self.analyze_image(
            image_path=image_path,
            prompt=question,
            model=model,
        )