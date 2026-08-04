"""
Customer Chat UI Page
ChatGPT & Gemini style conversational chat interface with Sentiment Analysis and Ollama LLM.
"""

import streamlit as st
from modules.sentiment import analyze_sentiment
from ui.theme import render_header


def render_customer_chat_page(ollama_client):
    status = ollama_client.check_connection()
    models = status.get("models", [])
    
    render_header(
        title="💬 Customer Chat Assistant",
        subtitle="Conversational customer support with sentiment tracking & local Ollama LLM",
        status=status,
    )

    if "customer_chat_messages" not in st.session_state:
        st.session_state.customer_chat_messages = []

    # Controls Row
    ccol1, ccol2, ccol3 = st.columns([2, 2, 1])

    with ccol1:
        selected_model = st.selectbox(
            "Select Ollama Model",
            options=models if models else [ollama_client.default_model],
            index=0,
            key="chat_model_select",
        )

    with ccol2:
        chat_mode = st.selectbox(
            "Response Generation Mode",
            options=["Ollama AI + Sentiment", "Sentiment-Rule Fallback Only"],
            index=0,
            key="chat_mode_select",
        )

    with ccol3:
        st.write("")
        st.write("")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.customer_chat_messages = []
            st.rerun()

    st.divider()

    # Display Conversation
    if not st.session_state.customer_chat_messages:
        st.markdown(
            """
        <div style="text-align: center; padding: 40px 20px; color: #9CA3AF;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🤖</div>
            <h3>How can I help you today?</h3>
            <p>Type any customer inquiry, complaint, or question below. The assistant will detect your sentiment and reply intelligently.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.customer_chat_messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    if "sentiment" in msg:
                        sent = msg["sentiment"]
                        s_type = sent.get("sentiment", "Neutral")
                        s_conf = sent.get("confidence", 0.0)
                        
                        if s_type == "Positive":
                            st.caption(f"😊 Sentiment: Positive (Score: {s_conf})")
                        elif s_type == "Negative":
                            st.caption(f"😟 Sentiment: Negative (Score: {s_conf})")
                        else:
                            st.caption(f"😐 Sentiment: Neutral (Score: {s_conf})")

    # Chat Input
    prompt = st.chat_input("Message customer support assistant...")

    if prompt:
        st.session_state.customer_chat_messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing message and generating response..."):
            sentiment_res = analyze_sentiment(prompt)

            if chat_mode == "Ollama AI + Sentiment" and status.get("connected"):
                system_prompt = (
                    "You are a warm, highly professional customer support representative. "
                    f"The customer's message has a detected sentiment of '{sentiment_res['sentiment']}'. "
                    "Adapt your tone accordingly: if negative, be empathetic and supportive; if positive, express gratitude; if neutral, be helpful and clear."
                )
                
                # Send history
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.customer_chat_messages[-6:]
                ]
                
                bot_reply = ollama_client.chat(
                    messages=history_payload,
                    model=selected_model,
                    system=system_prompt,
                )

                if not bot_reply:
                    bot_reply = (
                        f"Thank you for contacting us! I noticed your sentiment is {sentiment_res['sentiment']}. "
                        "How can I further assist you?"
                    )
            else:
                if sentiment_res["sentiment"] == "Positive":
                    bot_reply = "Thank you so much for your kind words! We are glad to hear from you. How else can we help?"
                elif sentiment_res["sentiment"] == "Negative":
                    bot_reply = "I am deeply sorry to hear about your experience. Please let us know the issue in detail so we can resolve it immediately."
                else:
                    bot_reply = "Thank you for reaching out! Please share any further details so we can best assist you."

            st.session_state.customer_chat_messages.append(
                {
                    "role": "assistant",
                    "content": bot_reply,
                    "sentiment": sentiment_res,
                }
            )

        st.rerun()