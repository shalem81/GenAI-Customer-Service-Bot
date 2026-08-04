"""
Ollama Client Module
GenAI Customer Service Bot

Provides a robust REST API client wrapper for Ollama local LLM service.
Endpoints used:
- GET  /api/tags     (List local models / health check)
- POST /api/generate (Text generation)
- POST /api/chat     (Conversational multi-turn chat)
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

from modules.config import (
    OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_VISION_MODEL,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with local Ollama API server.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_vision_model: Optional[str] = None,
    ):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.default_model = default_model or DEFAULT_OLLAMA_MODEL
        self.default_vision_model = default_vision_model or DEFAULT_OLLAMA_VISION_MODEL

    def check_connection(self) -> Dict[str, Union[bool, List[str], str]]:
        """
        Check if Ollama server is running and fetch available models.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return {
                    "connected": True,
                    "models": models,
                    "message": f"Connected to Ollama. {len(models)} model(s) available.",
                }
            return {
                "connected": False,
                "models": [],
                "message": f"Ollama returned status code {response.status_code}",
            }
        except requests.exceptions.RequestException as e:
            return {
                "connected": False,
                "models": [],
                "message": f"Could not connect to Ollama at {self.base_url}: {str(e)}",
            }

    def list_models(self) -> List[str]:
        """
        Return list of model names installed on the local Ollama instance.
        """
        status = self.check_connection()
        return status.get("models", [])

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Optional[str]:
        """
        Generate text response using /api/generate endpoint.
        """
        selected_model = model or self.default_model
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama generate error {response.status_code}: {response.text}")
                return None
        except Exception as error:
            logger.error(f"Failed to generate text from Ollama: {error}")
            return None

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """
        Multi-turn chat completion using /api/chat endpoint.
        messages format: [{'role': 'user'|'assistant'|'system', 'content': '...'}]
        """
        selected_model = model or self.default_model

        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        
        formatted_messages.extend(messages)

        payload = {
            "model": selected_model,
            "messages": formatted_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                result = response.json()
                msg = result.get("message", {})
                return msg.get("content", "").strip()
            else:
                logger.error(f"Ollama chat error {response.status_code}: {response.text}")
                return None
        except Exception as error:
            logger.error(f"Failed chat completion from Ollama: {error}")
            return None

    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str = "Describe this image in detail.",
        model: Optional[str] = None,
    ) -> Dict[str, Union[bool, str]]:
        """
        Analyze an image using Ollama Vision models (e.g., llava, llama3.2-vision).
        """
        image_file = Path(image_path)
        if not image_file.exists():
            return {"success": False, "error": "Image file does not exist."}

        try:
            with open(image_file, "rb") as img_f:
                base64_image = base64.b64encode(img_f.read()).decode("utf-8")

            selected_model = model or self.default_vision_model

            payload = {
                "model": selected_model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=90,
            )

            if response.status_code == 200:
                res_data = response.json()
                return {
                    "success": True,
                    "response": res_data.get("response", "").strip(),
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama vision model error ({response.status_code}): {response.text}",
                }
        except Exception as err:
            return {"success": False, "error": f"Vision processing failed: {str(err)}"}

    def generate_rag_response(
        self,
        query: str,
        context: str,
        domain: str = "Customer Support",
        model: Optional[str] = None,
    ) -> Optional[str]:
        """
        RAG helper to generate grounded answers using retrieved context.
        """
        system_prompt = (
            f"You are an expert AI assistant specializing in {domain}. "
            "Use the provided context information from our verified database to accurately answer the user query. "
            "If the context does not contain relevant information, state what you know from general knowledge while clarifying the source."
        )

        user_prompt = f"### Database Context:\n{context}\n\n### User Question:\n{query}"
        return self.generate(user_prompt, model=model, system=system_prompt)
