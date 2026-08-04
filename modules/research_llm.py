"""
Research LLM Module
GenAI Customer Service Bot

Generate research-grounded explanations using local Ollama LLM service.

Features:
- Answer research questions
- Use retrieved arXiv papers as context
- Explain complex concepts
- Summarize papers
- Maintain conversation history
- Support follow-up questions
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.research_expert import ResearchExpert
from modules.ollama_client import OllamaClient

load_dotenv(BASE_DIR / ".env")


class ResearchLLM:

    def __init__(
        self,
        expert: Optional[ResearchExpert] = None,
        ollama_client: Optional[OllamaClient] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the research assistant.
        """
        self.expert = expert if expert is not None else ResearchExpert()
        self.client = ollama_client if ollama_client is not None else OllamaClient()
        self.model_name = model_name or self.client.default_model
        self.history: List[Dict[str, str]] = []

    def is_available(self) -> bool:
        """Check whether Ollama service is connected."""
        status = self.client.check_connection()
        return status.get("connected", False)

    # =====================================================
    # HISTORY
    # =====================================================

    def add_to_history(
        self,
        user_message: str,
        assistant_message: str,
    ):
        """Store one conversation exchange."""
        self.history.append(
            {
                "user": user_message,
                "assistant": assistant_message,
            }
        )
        self.history = self.history[-6:]

    def get_history_text(self) -> str:
        """Convert recent history to prompt text."""
        if not self.history:
            return "No previous conversation."

        parts = []
        for exchange in self.history:
            parts.append(f"User: {exchange['user']}")
            parts.append(f"Assistant: {exchange['assistant']}")

        return "\n".join(parts)

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    # =====================================================
    # FALLBACK
    # =====================================================

    @staticmethod
    def fallback_answer(papers: List[Dict[str, Any]]) -> str:
        """
        Generate a simple answer when Ollama is unavailable.
        """
        if not papers:
            return "I could not find relevant research papers for this question."

        paper = papers[0]
        return f"Based on the paper '{paper['title']}', {paper['abstract']}"

    # =====================================================
    # RESEARCH QUESTION
    # =====================================================

    def ask(
        self,
        question: str,
        top_k: int = 3,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a research question using retrieved arXiv papers context + Ollama LLM.
        """
        question = str(question).strip()

        if not question:
            return {
                "answer": "Please enter a research question.",
                "papers": [],
                "used_llm": False,
            }

        papers = self.expert.search(question, top_k=top_k)

        if not papers:
            return {
                "answer": "I could not find relevant research papers for that question.",
                "papers": [],
                "used_llm": False,
            }

        context = self.expert.create_context(question, top_k=top_k)
        history = self.get_history_text()

        prompt = f"""
You are an expert computer science and cybersecurity research assistant.
Answer the user's question accurately using the supplied arXiv research context.

Rules:
1. Base the answer primarily on the provided research papers.
2. Explain technical concepts clearly with key insights.
3. Reference relevant paper titles when appropriate.
4. If the context is insufficient, state what is known while clarifying source limits.

Recent Conversation:
{history}

Research Context:
{context}

User Question:
{question}

Provide a structured, research-grounded response.
"""

        selected_model = model or self.model_name
        answer = self.client.generate(prompt, model=selected_model)

        used_llm = bool(answer)

        if not answer:
            answer = self.fallback_answer(papers)

        self.add_to_history(question, answer)

        return {
            "answer": answer,
            "papers": papers,
            "used_llm": used_llm,
        }

    def explain_concept(
        self,
        concept: str,
        top_k: int = 3,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explain a research concept using Ollama LLM."""
        concept = str(concept).strip()
        if not concept:
            return {
                "answer": "Please enter a concept to explain.",
                "papers": [],
                "used_llm": False,
            }

        question = (
            f"Explain the concept of {concept}, including how it works, "
            "why it matters, and relevant research applications."
        )
        return self.ask(question, top_k=top_k, model=model)

    def summarize_paper(
        self,
        paper_id: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Summarize a specific paper by ID using Ollama LLM."""
        paper = self.expert.get_paper_by_id(paper_id)
        if not paper:
            return {
                "success": False,
                "summary": "Paper not found in database.",
            }

        prompt = f"""
Summarize the following cybersecurity research paper in bullet points:

Title: {paper['title']}
Authors: {paper['authors']}
Abstract: {paper['abstract']}

Provide:
1. Executive Summary
2. Core Technical Method
3. Key Findings & Contributions
"""
        selected_model = model or self.model_name
        summary_text = self.client.generate(prompt, model=selected_model)

        if not summary_text:
            summary_text = f"Summary of '{paper['title']}': {paper['abstract']}"

        return {
            "success": True,
            "paper": paper,
            "summary": summary_text,
        }