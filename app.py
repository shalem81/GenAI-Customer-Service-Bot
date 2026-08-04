"""
GenAI Customer Service Bot
ChatGPT & Gemini AI Inspired Web Application powered by local Ollama LLM.

Features:
- Customer Chat (Sentiment-Aware + Ollama Conversational Chat)
- Medical Q&A (MedQuAD Dataset + Entity Extraction + Hybrid Ollama AI)
- Dynamic Knowledge Base (Vector/TF-IDF Search + Ollama RAG Synthesis)
- Cybersecurity Research Expert (arXiv Papers + Ollama Summarization)
- Multimodal Vision AI (Ollama Vision / LLava Image Analysis)
- Multilingual Assistant (Language Detection + Deep Translation + Ollama Fallback)
- Settings & Ollama Server Status Diagnostics
"""

import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="GenAI Customer Service Bot - Ollama AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.theme import apply_custom_theme
apply_custom_theme()

# ---------------------------------------------------------
# Backend Modules
# ---------------------------------------------------------
from modules.ollama_client import OllamaClient
from modules.medical_qa import MedicalQABot
from modules.knowledge_base import DynamicKnowledgeBase
from modules.research_expert import ResearchExpert
from modules.research_llm import ResearchLLM
from modules.multimodal import MultimodalAI
from modules.multilingual import MultilingualAI

# ---------------------------------------------------------
# UI Page Modules
# ---------------------------------------------------------
from ui.dashboard_page import render_dashboard_page
from ui.customer_chat import render_customer_chat_page
from ui.medical_page import render_medical_page
from ui.knowledge_base_page import render_knowledge_base_page
from ui.research_page import render_research_page
from ui.multimodal_page import render_multimodal_page
from ui.multilingual_page import render_multilingual_page
from ui.settings_page import render_settings_page

# ---------------------------------------------------------
# Cached Resource Initializer
# ---------------------------------------------------------

@st.cache_resource
def load_ollama_client():
    return OllamaClient()

@st.cache_resource
def load_medical_bot():
    return MedicalQABot()

@st.cache_resource
def load_knowledge_base():
    return DynamicKnowledgeBase()

@st.cache_resource
def load_research_expert():
    return ResearchExpert()

@st.cache_resource
def load_research_llm(_expert, _ollama_client):
    return ResearchLLM(expert=_expert, ollama_client=_ollama_client)

@st.cache_resource
def load_multimodal(_ollama_client):
    return MultimodalAI(ollama_client=_ollama_client)

@st.cache_resource
def load_translator():
    return MultilingualAI()

# Initialize Objects
ollama_client = load_ollama_client()
medical_bot = load_medical_bot()
knowledge_base = load_knowledge_base()
research_expert = load_research_expert()
research_llm = load_research_llm(research_expert, ollama_client)
multimodal_ai = load_multimodal(ollama_client)
translator = load_translator()

# ---------------------------------------------------------
# Session State & Navigation
# ---------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Check connection
ollama_status = ollama_client.check_connection()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <img src="https://img.icons8.com/color/96/chatbot.png" width="40" height="40" />
        <div>
            <div class="sidebar-brand-title">GenAI Bot</div>
            <div style="font-size: 0.75rem; color: #9CA3AF;">Ollama Local AI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Radio
pages = [
    "Dashboard",
    "Customer Chat",
    "Medical Q&A",
    "Knowledge Base",
    "Research Expert",
    "Multimodal AI",
    "Multilingual",
    "Settings",
]

default_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0

selected_page = st.sidebar.radio(
    "Navigation Workspace",
    pages,
    index=default_index,
)

st.session_state.page = selected_page

st.sidebar.markdown("---")

# Connection Status Badge in Sidebar
if ollama_status["connected"]:
    st.sidebar.success(f"🟢 **Ollama Connected**\n\n`{len(ollama_status['models'])}` local model(s) ready.")
else:
    st.sidebar.error("🔴 **Ollama Disconnected**\n\nStart server at `http://localhost:11434`.")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 GenAI Customer Service Bot")

# ---------------------------------------------------------
# Router
# ---------------------------------------------------------

if st.session_state.page == "Dashboard":
    render_dashboard_page(ollama_client, medical_bot, knowledge_base, research_expert)

elif st.session_state.page == "Customer Chat":
    render_customer_chat_page(ollama_client)

elif st.session_state.page == "Medical Q&A":
    render_medical_page(medical_bot, ollama_client)

elif st.session_state.page == "Knowledge Base":
    render_knowledge_base_page(knowledge_base, ollama_client)

elif st.session_state.page == "Research Expert":
    render_research_page(research_llm, research_expert, ollama_client)

elif st.session_state.page == "Multimodal AI":
    render_multimodal_page(multimodal_ai, ollama_client)

elif st.session_state.page == "Multilingual":
    render_multilingual_page(translator, ollama_client)

elif st.session_state.page == "Settings":
    render_settings_page(ollama_client)