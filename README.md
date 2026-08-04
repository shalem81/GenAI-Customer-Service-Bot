# 🤖 GenAI Customer Service Bot

An **AI-powered customer support platform** built with **Streamlit**, local **Ollama LLMs**, and a **ChatGPT / Gemini-inspired UI**. 

This application integrates **Hybrid Data Retrieval (RAG)**, combining local domain databases (MedQuAD, arXiv security papers, dynamic knowledge articles) with local generative AI to provide grounded responses.

---

## ✨ Key Features

- 💬 **Customer Chat Assistant**: Conversational AI powered by Ollama LLM integrated with real-time **VADER sentiment analysis**.
- 🩺 **Medical Question Answering**: Search the **MedQuAD** dataset with automated medical entity extraction and Ollama medical synthesis.
- 📚 **Dynamic Knowledge Base (RAG)**: Search local vector & TF-IDF document chunks and synthesize answers using Ollama LLM.
- 📄 **Cybersecurity Research Expert**: Search **arXiv research papers** and generate structured AI paper summaries.
- 🖼️ **Multimodal Vision AI**: Upload images and perform visual question answering using Ollama Vision models (`llava` / `llama3.2-vision`).
- 🌍 **Multilingual Assistant**: Automatic language detection and translation across multiple target languages.
- 🔀 **Hybrid Search Modes**: Toggle seamlessly between **Database Only**, **Pure Ollama AI**, and **Hybrid RAG Mode**.
- 🎨 **ChatGPT & Gemini-Inspired UI**: Sleek dark theme with live Ollama connection status badge, dynamic model selector, and metric cards.

---

## 🛠️ Technology Stack

- **Frontend UI**: Streamlit, Custom HTML/CSS (ChatGPT/Gemini theme)
- **AI Engine**: Ollama REST API (`llama3`, `llava`, `mistral`)
- **NLP & Search**: Scikit-learn (TF-IDF & Cosine Similarity), VADER Sentiment Analysis, Spacy / Langdetect
- **Translation**: Deep-Translator
- **Data & Storage**: Pandas, JSON vector store

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** and **Ollama** installed on your system.

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/shalem81/GenAI-Customer-Service-Bot.git
cd GenAI-Customer-Service-Bot
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull Ollama Models
Ensure Ollama is running (`ollama serve`), then pull the required models:
```bash
ollama pull llama3
ollama pull llava
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📁 Project Structure

```text
GenAI-Customer-Service-Bot/
├── app.py                     # Main application entry point & router
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables configuration
├── modules/                   # Core backend logic
│   ├── ollama_client.py       # Ollama REST API integration wrapper
│   ├── medical_qa.py          # MedQuAD search & medical entity extractor
│   ├── knowledge_base.py      # Dynamic RAG knowledge base module
│   ├── research_expert.py     # arXiv cybersecurity paper retrieval
│   ├── research_llm.py        # Ollama research summarization module
│   ├── multimodal.py          # Vision AI image analysis module
│   ├── sentiment.py           # VADER sentiment analysis module
│   └── multilingual.py       # Language detection & translation module
├── ui/                        # User interface modules
│   ├── theme.py               # ChatGPT & Gemini CSS design system
│   ├── dashboard_page.py      # Metrics dashboard & quick start workspace
│   ├── customer_chat.py       # Conversational chat UI
│   ├── medical_page.py        # Medical assistant UI
│   ├── knowledge_base_page.py # RAG knowledge base UI
│   ├── research_page.py       # arXiv paper search & summary UI
│   ├── multimodal_page.py     # Image upload & vision query UI
│   ├── multilingual_page.py   # Translation workspace UI
│   └── settings_page.py       # Ollama server diagnostic UI
├── data/                      # Dataset files (medquad.csv, arxiv_cybersecurity.csv)
└── tests/                     # Unit test suite
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
