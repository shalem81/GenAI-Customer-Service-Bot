# 🎓 INTERNSHIP PROJECT REPORT

## **Project Title**: GenAI Customer Service Bot  
**Domain**: Generative AI, Retrieval-Augmented Generation (RAG), Local LLM Systems & Full-Stack Web Development  
**UI Framework**: Streamlit (ChatGPT & Gemini AI Design Aesthetic)  
**LLM Engine**: Ollama Local REST API (`llama3`, `llava`, `mistral`)  

---

## 📑 TABLE OF CONTENTS
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Project Objectives](#2-problem-statement--project-objectives)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Core Modules & Technical Implementation](#5-core-modules--technical-implementation)
   - 5.1 Local Ollama API Integration Wrapper
   - 5.2 Sentiment-Aware Customer Chat Assistant
   - 5.3 Medical Question Answering (MedQuAD Dataset)
   - 5.4 Dynamic Knowledge Base & Vector Search (RAG)
   - 5.5 Cybersecurity Research Expert (arXiv Papers)
   - 5.6 Multimodal Vision AI (`llava` Integration)
   - 5.7 Multilingual Support & Translation Engine
6. [Hybrid Retrieval System (Database + Ollama AI)](#6-hybrid-retrieval-system-database--ollama-ai)
7. [User Interface Design (ChatGPT & Gemini Aesthetic)](#7-user-interface-design-chatgpt--gemini-aesthetic)
8. [Testing & Verification](#8-testing--verification)
9. [Conclusion & Future Scope](#9-conclusion--future-scope)

---

## 1. EXECUTIVE SUMMARY

The **GenAI Customer Service Bot** is an end-to-end, enterprise-grade AI customer intelligence platform designed to replace cloud-dependent API models (such as Google Gemini) with privacy-preserving, cost-effective, local Large Language Models via **Ollama**.

The system integrates **Retrieval-Augmented Generation (RAG)** across domain-specific datasets (MedQuAD Medical QA dataset, arXiv Cybersecurity Research paper dataset, and dynamic document repositories). Built with **Streamlit**, the application features a **ChatGPT and Gemini AI-inspired Web Interface** equipped with live server connection monitoring, dynamic model pickers, sentiment analysis, entity extraction, multimodal vision processing, and multilingual capabilities.

---

## 2. PROBLEM STATEMENT & PROJECT OBJECTIVES

### 2.1 Problem Statement
Traditional customer service bots often suffer from two major flaws:
1. **Cloud API Dependencies & Privacy Risks**: Relying on third-party cloud APIs (e.g., OpenAI, Google Gemini) incurs high latency, recurring subscription costs, and potential data privacy violations.
2. **Hallucination & Generic Answers**: Standard LLMs lack domain-specific data context and frequently produce inaccurate or unsupported answers.

### 2.2 Project Objectives
- **Complete Local LLM Migration**: Remove external API dependencies (Gemini API) and transition to local model execution using **Ollama REST API** (`llama3`, `llava`).
- **Hybrid Data Retrieval (RAG)**: Combine local dataset search (TF-IDF vector matching) with Ollama generative LLM capabilities for grounded, non-hallucinated responses.
- **Full Backend-Frontend Integration**: Connect 7 specialized backend engines (Sentiment, Medical QA, Knowledge Base, Research Expert, Multimodal, Multilingual, Ollama Client) to a unified Streamlit UI.
- **ChatGPT & Gemini UI Redesign**: Implement a state-of-the-art dark theme UI featuring glassmorphism cards, dynamic model dropdowns, live connection indicators, and prompt action cards.

---

## 3. SYSTEM ARCHITECTURE

```
                  ┌─────────────────────────────────────────┐
                  │    Streamlit Web Interface (App UI)     │
                  │ (ChatGPT / Gemini AI Design Aesthetic)   │
                  └────────────────────┬────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │        Modular Router & Controller         │
                └──────┬──────────────┬──────────────┬────────┘
                       │              │              │
       ┌───────────────▼──┐   ┌───────▼────────┐   ┌─▼────────────────┐
       │ Customer Chat    │   │ Medical QA     │   │ Knowledge Base   │
       │ & Sentiment      │   │ (MedQuAD DB)   │   │ RAG Vector Search│
       └───────┬──────────┘   └───────┬────────┘   └─┬────────────────┘
               │                      │              │
               └──────────────────────┼──────────────┘
                                      │
                 ┌────────────────────▼────────────────────┐
                 │     Local Ollama REST API Client        │
                 │   (HTTP endpoints: /api/generate, chat) │
                 └────────────────────┬────────────────────┘
                                      │
                 ┌────────────────────▼────────────────────┐
                 │     Local Models (llama3, llava, etc.)  │
                 └─────────────────────────────────────────┘
```

---

## 4. TECHNOLOGY STACK

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend Framework** | Streamlit, Custom HTML5/CSS3 | Web interface with ChatGPT/Gemini UI theme |
| **LLM Engine** | Ollama REST API (`llama3`, `llava`) | Local text generation, chat completion, & vision analysis |
| **NLP & Vector Search** | Scikit-learn (TF-IDF, Cosine Similarity) | Dataset indexing and semantic snippet retrieval |
| **Sentiment Analysis** | VADER (`vaderSentiment`) | Customer message polarity & confidence detection |
| **Multilingual Engine** | `deep-translator`, `langdetect` | Language identification & translation |
| **Vision & Image** | Pillow (`PIL`), Base64 Encoding | Image metadata extraction & vision model input |
| **Testing** | Pytest | Automated test suite execution |

---

## 5. CORE MODULES & TECHNICAL IMPLEMENTATION

### 5.1 Local Ollama API Integration Wrapper (`modules/ollama_client.py`)
Replaced `google-genai` with a custom HTTP client interacting with Ollama daemon (`http://localhost:11434`):
- `check_connection()`: Pings `/api/tags` to verify server connectivity and list installed models.
- `chat()`: Manages multi-turn conversation payloads with system role prompts.
- `generate()`: Single-turn completion generator.
- `analyze_image()`: Base64-encodes images and executes vision requests using models like `llava`.

### 5.2 Sentiment-Aware Customer Chat Assistant (`ui/customer_chat.py`)
Analyzes input text using VADER sentiment analysis (`Positive`, `Negative`, `Neutral` compound scores) and feeds sentiment context to Ollama LLM to adapt conversational response tone.

### 5.3 Medical Question Answering (`modules/medical_qa.py`)
Utilizes a processed dataset of **MedQuAD XML records** (`medquad.csv`). Performs TF-IDF search, extracts medical entities (diseases, symptoms, treatments), and generates empathetic medical explanations with mandatory disclaimers.

### 5.4 Dynamic Knowledge Base RAG (`modules/knowledge_base.py`)
Allows users to add document content dynamically. Chunks text, computes TF-IDF representations, and runs hybrid RAG synthesis through Ollama.

### 5.5 Cybersecurity Research Expert (`modules/research_expert.py` & `research_llm.py`)
Indexes an **arXiv Cybersecurity dataset** (`arxiv_cybersecurity.csv`). Provides paper relevance scoring, abstract retrieval, and Ollama paper summarization.

### 5.6 Multimodal Vision AI (`modules/multimodal.py`)
Enables drag-and-drop image uploads. Validates size/formats, extracts image metadata, and passes queries to Ollama Vision models.

### 5.7 Multilingual Support & Translation (`modules/multilingual.py`)
Detects input languages (English, Hindi, Telugu, Tamil, etc.) and translates content using `GoogleTranslator` with Ollama fallback.

---

## 6. HYBRID RETRIEVAL SYSTEM (DATABASE + OLLAMA AI)

To eliminate LLM hallucinations while preserving conversational quality, the platform provides 3 selectable query modes:

1. **Database Only Mode**: Directly retrieves exact TF-IDF / vector record matches from local CSV datasets without LLM overhead.
2. **Pure Ollama AI Mode**: Direct prompt generation from local LLM.
3. **Hybrid RAG Mode**: 
   - *Phase 1*: Top matching document blocks are retrieved from the database.
   - *Phase 2*: Retrieved snippets are formatted as structured context.
   - *Phase 3*: Ollama LLM synthesizes a polished, accurate answer referencing the underlying sources.

---

## 7. USER INTERFACE DESIGN (CHATGPT & GEMINI AESTHETIC)

The UI was completely overhauled ([`ui/theme.py`](file:///c:/GenAI-Customer-Service-Bot/ui/theme.py)):
- **Header Bar**: Displays current workspace title alongside live connection status badges (`🟢 Ollama Active` / `🔴 Ollama Offline`).
- **Dynamic Model Picker**: Selectbox populated with pulled local Ollama models.
- **Glassmorphism Metric Cards**: Styled statistics counters displaying dataset record counts and document inventory.
- **Custom Chat Bubbles**: Styled user and AI avatar badges (`🤖 Ollama AI`).

---

## 8. TESTING & VERIFICATION

A comprehensive test suite was executed across all modules using `pytest`:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
collected 66 items

tests/test_knowledge_base.py .................                           [ 25%]
tests/test_medical_qa.py ..............                                  [ 46%]
tests/test_multilingual.py .....                                         [ 54%]
tests/test_multimodal.py .....                                           [ 62%]
tests/test_research_expert.py ................                           [ 86%]
tests/test_sentiment.py .........                                        [100%]

============================= 66 passed in 14.18s =============================
```

All 66 test cases passed successfully.

---

## 9. CONCLUSION & FUTURE SCOPE

### 9.1 Conclusion
The **GenAI Customer Service Bot** successfully demonstrates a privacy-first, fully local, multi-functional AI assistant. By replacing cloud API dependencies with Ollama local LLMs and implementing Hybrid RAG retrieval, the system achieves fast, cost-effective, and highly reliable AI performance paired with a modern ChatGPT/Gemini UI.

### 9.2 Future Scope
- Integration of local vector databases (ChromaDB / FAISS) for dense semantic embeddings.
- Voice input/output integration using local Whisper speech recognition.
- Multi-agent orchestration for complex multi-step customer service workflows.

---
**Report Submitted By**: AI Development Intern  
**Project Repository**: [GitHub Repository Link](https://github.com/shalem81/GenAI-Customer-Service-Bot)
