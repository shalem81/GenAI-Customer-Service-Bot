"""
Knowledge Base UI Page
Dynamic knowledge base search, document addition, statistics, and Ollama RAG.
"""

import streamlit as st
from ui.theme import render_header


def render_knowledge_base_page(knowledge_base, ollama_client):
    status = ollama_client.check_connection()
    models = status.get("models", [])

    render_header(
        title="📚 Dynamic Knowledge Base RAG",
        subtitle="Vector & TF-IDF Search synthesized with Ollama Generative AI",
        status=status,
    )

    tab1, tab2, tab3 = st.tabs(["🔍 Search & RAG Synthesis", "➕ Add Knowledge", "📊 Statistics & Index"])

    # ---------------------------------------------------------
    # TAB 1: SEARCH & RAG
    # ---------------------------------------------------------
    with tab1:
        scol1, scol2 = st.columns([2, 1])

        with scol1:
            search_mode = st.radio(
                "RAG Mode",
                ["Hybrid RAG (Knowledge DB + Ollama)", "Raw Knowledge DB Search"],
                horizontal=True,
                key="kb_mode_radio",
            )

        with scol2:
            selected_model = st.selectbox(
                "Select Ollama Model",
                options=models if models else [ollama_client.default_model],
                key="kb_model_select",
            )

        query = st.text_input("Enter search query or question", placeholder="e.g. Return policy, server configuration, support hours")

        if st.button("Search Knowledge Base", use_container_width=True, type="primary"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Searching documents & generating answer..."):
                    if search_mode == "Raw Knowledge DB Search":
                        results = knowledge_base.search(query, top_k=5)
                        answer_data = {
                            "query": query,
                            "answer": None,
                            "results": results,
                            "used_llm": False,
                        }
                    else:
                        answer_data = knowledge_base.search_hybrid(
                            query=query,
                            ollama_client=ollama_client,
                            model=selected_model,
                            top_k=5,
                        )

                if answer_data.get("used_llm") and answer_data.get("answer"):
                    st.markdown("### 🤖 Synthesized AI Response")
                    st.info(answer_data["answer"])

                results = answer_data.get("results", [])
                if results:
                    st.success(f"Found {len(results)} matching document chunk(s)")
                    for idx, res in enumerate(results, start=1):
                        with st.expander(f"Result #{idx} | {res['title']} (Score: {res['score']})"):
                            st.markdown(f"**Source**: `{res['source']}`")
                            st.write(res["content"])
                else:
                    st.warning("No matching documents found in knowledge base.")

    # ---------------------------------------------------------
    # TAB 2: ADD KNOWLEDGE
    # ---------------------------------------------------------
    with tab2:
        title = st.text_input("Document Title", placeholder="e.g. Refund Policy 2026")
        source = st.text_input("Source Identifier / URL", placeholder="e.g. kb-refunds-v1")
        content = st.text_area("Document Content", height=200, placeholder="Paste knowledge article text here...")

        if st.button("Add Knowledge Article", use_container_width=True):
            if not content.strip():
                st.error("Content cannot be empty.")
            else:
                res = knowledge_base.add_document(text=content, source=source or "manual", title=title or "Untitled")
                if res["success"]:
                    st.success(res["message"])
                    st.metric("Chunks Added", res.get("chunks_added", 0))
                else:
                    st.error(res["message"])

    # ---------------------------------------------------------
    # TAB 3: STATISTICS
    # ---------------------------------------------------------
    with tab3:
        stats = knowledge_base.get_stats()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Documents", stats.get("documents", 0))
        with col2:
            st.metric("Total Chunks", stats.get("chunks", 0))
        with col3:
            st.metric("Unique Sources", stats.get("sources", 0))
        with col4:
            st.metric("Last Updated", stats.get("last_update") or "N/A")

        if knowledge_base.documents:
            st.markdown("### Document Chunks Inventory")
            st.dataframe(knowledge_base.documents, use_container_width=True)
