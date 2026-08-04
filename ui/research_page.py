"""
Research Expert UI Page
Search arXiv Cybersecurity Papers & generate AI summaries using Ollama.
"""

import streamlit as st
from ui.theme import render_header


def render_research_page(research_llm, research_expert, ollama_client):
    status = ollama_client.check_connection()
    models = status.get("models", [])

    render_header(
        title="📄 Research Expert & arXiv Browser",
        subtitle="Search cybersecurity papers & generate deep research summaries with Ollama LLM",
        status=status,
    )

    rcol1, rcol2, rcol3 = st.columns([2, 1, 1])

    with rcol1:
        topic = st.text_input("Enter research topic or question", placeholder="e.g. Ransomware detection, zero trust architecture, malware analysis")

    with rcol2:
        top_k = st.slider("Top papers", 1, 10, 5)

    with rcol3:
        selected_model = st.selectbox(
            "Select Ollama Model",
            options=models if models else [ollama_client.default_model],
            key="research_model_select",
        )

    if st.button("Search Cybersecurity Papers", use_container_width=True, type="primary"):
        if not topic.strip():
            st.warning("Please enter a research topic.")
        else:
            with st.spinner("Searching arXiv security papers & analyzing with Ollama..."):
                res = research_llm.ask(question=topic, top_k=top_k, model=selected_model)

            st.markdown("### 🤖 Research Synthesized Answer")
            st.info(res["answer"])

            papers = res.get("papers", [])
            if papers:
                st.success(f"{len(papers)} paper(s) retrieved from dataset")

                for idx, paper in enumerate(papers, start=1):
                    with st.expander(f"{idx}. {paper['title']} (Score: {paper['score']})"):
                        st.markdown(f"**Authors**: {paper['authors']}")
                        st.markdown(f"**Categories**: `{paper['categories']}`")
                        st.markdown("**Abstract:**")
                        st.write(paper["abstract"])

                        if st.button(f"Generate AI Summary for Paper #{paper['id']}", key=f"sum_btn_{paper['id']}"):
                            with st.spinner("Generating detailed summary with Ollama..."):
                                sum_res = research_llm.summarize_paper(paper["id"], model=selected_model)
                                st.success("Summary Generated!")
                                st.markdown(sum_res["summary"])
            else:
                st.warning("No relevant research papers found.")
