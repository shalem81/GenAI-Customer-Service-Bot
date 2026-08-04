"""
Settings Page UI
Configure Ollama engine, view system status, and manage environment variables.
"""

import os
import streamlit as st
from ui.theme import render_header


def render_settings_page(ollama_client):
    status = ollama_client.check_connection()
    models = status.get("models", [])

    render_header(
        title="⚙️ Application Settings",
        subtitle="Configure Ollama REST API endpoint, models, and application settings",
        status=status,
    )

    tab1, tab2, tab3 = st.tabs(["🦙 Ollama AI Engine", "📱 Application Info", "🛠 System Diagnostic"])

    with tab1:
        st.subheader("Ollama Configuration")

        base_url = st.text_input("Ollama Server URL", value=ollama_client.base_url)
        
        col1, col2 = st.columns(2)
        with col1:
            default_model = st.selectbox(
                "Default Text Model",
                options=models if models else [ollama_client.default_model],
                index=0 if models else 0,
            )
        with col2:
            vision_model = st.selectbox(
                "Default Vision Model",
                options=[m for m in models if "llava" in m.lower() or "vision" in m.lower()] or models or [ollama_client.default_vision_model],
                index=0,
            )

        if st.button("🧪 Test Ollama Connection", use_container_width=True):
            test_client = type(ollama_client)(base_url=base_url)
            test_status = test_client.check_connection()
            if test_status["connected"]:
                st.success(f"Successfully connected to Ollama at {base_url}! {len(test_status['models'])} model(s) available.")
            else:
                st.error(test_status["message"])

        st.divider()

        st.markdown("### Pulled Local Models")
        if models:
            for m in models:
                st.markdown(f"- 📦 `{m}`")
        else:
            st.warning("No local models detected. Pull models using command: `ollama pull llama3` or `ollama pull llava`.")

    with tab2:
        st.subheader("Application Details")
        st.text_input("App Name", value=os.getenv("APP_NAME", "GenAI Customer Service Bot"), disabled=True)
        st.text_input("Environment", value=os.getenv("APP_ENV", "development"), disabled=True)
        st.text_input("Vector DB Directory", value=os.getenv("VECTOR_DB_PATH", "vector_db"), disabled=True)

    with tab3:
        st.subheader("System Information")
        st.json(
            {
                "ollama_base_url": ollama_client.base_url,
                "connection_status": status,
                "python_version": os.sys.version,
            }
        )