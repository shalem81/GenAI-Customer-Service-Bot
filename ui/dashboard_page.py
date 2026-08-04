"""
Dashboard Page UI
GenAI Customer Service Bot
"""

import streamlit as st
from ui.theme import render_header


def render_dashboard_page(
    ollama_client,
    medical_bot,
    knowledge_base,
    research_expert,
):
    status = ollama_client.check_connection()
    models = status.get("models", [])

    render_header(
        title="🤖 AI Workspace Dashboard",
        subtitle="ChatGPT & Gemini-inspired customer intelligence hub with local Ollama AI",
        status=status,
    )

    # Top Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-glass-card">
            <div class="metric-label">Ollama Models</div>
            <div class="metric-value">{len(models)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-glass-card">
            <div class="metric-label">Medical Records</div>
            <div class="metric-value">{len(medical_bot.data) if medical_bot.data is not None else 0}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-glass-card">
            <div class="metric-label">Knowledge Base</div>
            <div class="metric-value">{len(knowledge_base.documents)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-glass-card">
            <div class="metric-label">arXiv Papers</div>
            <div class="metric-value">{len(research_expert.data)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # Quick Start Prompts Cards
    st.markdown("### ⚡ Quick Start Assistant Modes")
    st.markdown("Choose a feature workspace or launch a prompt from below:")

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)

    with pcol1:
        if st.button("💬 Customer Chat", use_container_width=True, key="dash_btn_chat"):
            st.session_state.page = "Customer Chat"
            st.rerun()
        st.markdown(
            """
        <div class="prompt-card">
            <div class="prompt-card-icon">💬</div>
            <div class="prompt-card-title">Customer Support</div>
            <div class="prompt-card-desc">Sentiment-aware customer chat powered by Ollama LLM.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with pcol2:
        if st.button("🩺 Medical Q&A", use_container_width=True, key="dash_btn_med"):
            st.session_state.page = "Medical Q&A"
            st.rerun()
        st.markdown(
            """
        <div class="prompt-card">
            <div class="prompt-card-icon">🩺</div>
            <div class="prompt-card-title">Medical Assistant</div>
            <div class="prompt-card-desc">Search MedQuAD database with entity recognition & AI explanations.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with pcol3:
        if st.button("📚 Knowledge Base", use_container_width=True, key="dash_btn_kb"):
            st.session_state.page = "Knowledge Base"
            st.rerun()
        st.markdown(
            """
        <div class="prompt-card">
            <div class="prompt-card-icon">📚</div>
            <div class="prompt-card-title">Knowledge Base RAG</div>
            <div class="prompt-card-desc">Dynamic document search combined with Ollama RAG synthesis.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with pcol4:
        if st.button("📄 Research Expert", use_container_width=True, key="dash_btn_research"):
            st.session_state.page = "Research Expert"
            st.rerun()
        st.markdown(
            """
        <div class="prompt-card">
            <div class="prompt-card-icon">📄</div>
            <div class="prompt-card-title">Cybersecurity Research</div>
            <div class="prompt-card-desc">Search arXiv security papers & generate deep AI summaries.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # System Status Summary
    st.markdown("### ⚙️ Engine Status & Configuration")
    scol1, scol2 = st.columns(2)

    with scol1:
        st.info(
            f"""
        **Ollama Engine**: `{ollama_client.base_url}`  
        **Default Model**: `{ollama_client.default_model}`  
        **Vision Model**: `{ollama_client.default_vision_model}`  
        **Status Message**: {status.get('message')}
        """
        )

    with scol2:
        if models:
            st.success(f"**Pulled Ollama Models ({len(models)})**:\n" + "\n".join([f"- `{m}`" for m in models[:5]]))
        else:
            st.warning("No models currently detected. Run `ollama pull llama3` in your command line.")
