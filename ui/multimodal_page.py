"""
Multimodal AI UI Page
Image uploading, analysis, and visual Q&A powered by Ollama Vision models (llava / llama3.2-vision).
"""

import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from ui.theme import render_header


def render_multimodal_page(multimodal_ai, ollama_client):
    status = ollama_client.check_connection()
    models = status.get("models", [])

    render_header(
        title="🖼️ Multimodal Vision AI",
        subtitle="Upload images and analyze visual content using Ollama Vision models",
        status=status,
    )

    vcol1, vcol2 = st.columns([2, 1])

    with vcol1:
        uploaded_file = st.file_uploader(
            "Upload Image (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload an image to inspect or ask questions about.",
        )

    with vcol2:
        vision_model = st.selectbox(
            "Select Vision Model",
            options=[m for m in models if "llava" in m.lower() or "vision" in m.lower()] or models or [ollama_client.default_vision_model],
            key="vision_model_select",
        )

    if uploaded_file is not None:
        # Display image preview
        col_img, col_info = st.columns([1, 1])

        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)

        with col_info:
            st.markdown("### Image Metadata")
            st.write(f"**Filename**: `{uploaded_file.name}`")
            st.write(f"**Dimensions**: `{image.size[0]} x {image.size[1]} px`")
            st.write(f"**Format**: `{image.format}`")
            st.write(f"**Mode**: `{image.mode}`")

        prompt = st.text_input(
            "Ask something about this image",
            value="Describe this image in detail and identify any key features or text.",
        )

        if st.button("Analyze Image with Ollama AI", use_container_width=True, type="primary"):
            with st.spinner("Processing image through Ollama Vision model..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                res = multimodal_ai.analyze_image(
                    image_path=tmp_path,
                    prompt=prompt,
                    model=vision_model,
                )

                Path(tmp_path).unlink(missing_ok=True)

            if res["success"]:
                st.success("Analysis Complete!")
                st.markdown("### 🔍 Vision Analysis Output")
                st.markdown(res["response"])
            else:
                st.error(f"Analysis failed: {res.get('error')}")
