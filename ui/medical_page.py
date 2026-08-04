"""
Medical Q&A UI Page
Search MedQuAD database with entity recognition & Ollama AI explanations.
"""

import streamlit as st
from ui.theme import render_header


def render_medical_page(medical_bot, ollama_client):
    status = ollama_client.check_connection()
    models = status.get("models", [])

    render_header(
        title="🩺 Medical Question Answering",
        subtitle="MedQuAD Medical Dataset & Ollama AI Assistant",
        status=status,
    )

    mcol1, mcol2 = st.columns([2, 1])

    with mcol1:
        search_mode = st.radio(
            "Search & Generation Mode",
            ["Hybrid (MedQuAD DB + Ollama AI)", "Database Only (MedQuAD)", "Pure Ollama AI"],
            horizontal=True,
            key="med_mode_radio",
        )

    with mcol2:
        selected_model = st.selectbox(
            "Select Ollama Model",
            options=models if models else [ollama_client.default_model],
            key="med_model_select",
        )

    question = st.text_area(
        "Enter your medical question",
        height=100,
        placeholder="e.g., What are the symptoms of diabetes? How is asthma treated?",
    )

    if st.button("Get Medical Answer", use_container_width=True, type="primary"):
        if not question.strip():
            st.warning("Please enter a medical question.")
        else:
            with st.spinner("Processing medical inquiry..."):
                if search_mode == "Database Only (MedQuAD)":
                    res = medical_bot.answer_question(question)
                    used_llm = False
                elif search_mode == "Pure Ollama AI":
                    system = "You are a professional medical assistant. Provide accurate medical information with a disclaimer."
                    answer = ollama_client.generate(question, model=selected_model, system=system)
                    res = {
                        "question": question,
                        "answer": answer if answer else "Could not reach Ollama model.",
                        "source": "Ollama LLM",
                        "category": "General Health",
                        "similarity": 1.0,
                        "entities": [],
                        "used_llm": True,
                    }
                else:  # Hybrid Mode
                    res = medical_bot.answer_hybrid(question, ollama_client=ollama_client, model=selected_model)

            st.success("Response Generated")

            # Main Answer Box
            st.markdown("### 🩺 Medical Explanation")
            st.info(res["answer"])

            # Entity Tags
            if res.get("entities"):
                st.markdown("#### Detected Medical Entities")
                tags_html = ""
                for ent in res["entities"]:
                    tags_html += f'<span class="entity-tag">🏷️ {ent["text"]} ({ent["type"]})</span>'
                st.markdown(tags_html, unsafe_allow_html=True)
                st.write("")

            # Metadata Badges
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                st.metric("Category", res.get("category") or "N/A")
            with bcol2:
                st.metric("DB Similarity", f"{res.get('similarity', 0.0):.2f}")
            with bcol3:
                st.metric("Source Engine", "Ollama LLM + DB" if res.get("used_llm") else "MedQuAD DB")

            st.divider()

            # Source Details
            if res.get("source"):
                with st.expander("📌 Reference Source Details"):
                    st.write(f"**Source Dataset**: {res['source']}")
                    if "search_results" in res and res["search_results"]:
                        st.markdown("**Retrieved MedQuAD Entries:**")
                        for idx, sres in enumerate(res["search_results"], 1):
                            st.markdown(f"**{idx}. Q:** {sres['question']}")
                            st.caption(f"**A:** {sres['answer'][:300]}...")

            st.warning(
                "⚠️ **Disclaimer**: This tool provides educational information only. Always consult a qualified healthcare professional before making medical decisions."
            )