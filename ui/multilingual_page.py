"""
Multilingual AI UI Page
Language detection & translation powered by deep-translator & Ollama.
"""

import streamlit as st
from ui.theme import render_header


def render_multilingual_page(translator, ollama_client):
    status = ollama_client.check_connection()
    models = status.get("models", [])

    render_header(
        title="🌍 Multilingual AI Assistant",
        subtitle="Automatic language detection & multi-target translation",
        status=status,
    )

    tcol1, tcol2 = st.columns([2, 1])

    with tcol1:
        target_lang = st.selectbox(
            "Target Language",
            options=list(translator.supported_languages.keys()),
            format_func=lambda x: f"{translator.supported_languages[x]} ({x})",
            index=0,
        )

    with tcol2:
        selected_model = st.selectbox(
            "Select Ollama Model (Fallback / Refinement)",
            options=models if models else [ollama_client.default_model],
            key="trans_model_select",
        )

    input_text = st.text_area(
        "Enter text to detect and translate",
        height=150,
        placeholder="Type text in English, Telugu, Hindi, Tamil, or any language...",
    )

    if st.button("Detect & Translate", use_container_width=True, type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Detecting language & translating..."):
                det_res = translator.detect_language(input_text)
                trans_res = translator.translate(input_text, target_language=target_lang)

            st.markdown("### Detection & Translation Results")
            dcol1, dcol2 = st.columns(2)

            with dcol1:
                st.markdown("#### Original Text")
                st.info(input_text)
                if det_res["success"]:
                    st.caption(f"🌐 Detected Language: **{det_res['name']}** (`{det_res['language']}`)")

            with dcol2:
                st.markdown(f"#### Translated Text ({translator.supported_languages.get(target_lang)})")
                if trans_res["success"]:
                    st.success(trans_res["translated"])
                else:
                    # Fallback to Ollama LLM translation if deep-translator fails
                    prompt = f"Translate the following text into {translator.supported_languages.get(target_lang)}:\n\n{input_text}"
                    ollama_trans = ollama_client.generate(prompt, model=selected_model)
                    if ollama_trans:
                        st.success(ollama_trans)
                    else:
                        st.error(f"Translation error: {trans_res.get('message')}")
