"""
UI Theme Module
GenAI Customer Service Bot

Provides custom CSS and HTML rendering components to emulate modern, state-of-the-art
ChatGPT & Gemini AI user interfaces.
"""

import streamlit as st


def apply_custom_theme():
    """
    Inject custom CSS styling for a ChatGPT / Gemini AI aesthetic.
    """
    st.markdown(
        """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 8px;
        margin-bottom: 12px;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.2), rgba(147, 51, 234, 0.2));
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
    }

    .sidebar-brand-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Header Bar */
    .chatgpt-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        background: rgba(17, 24, 39, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid #1F2937;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }

    .header-title-box h2 {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-subtitle {
        font-size: 0.85rem;
        color: #9CA3AF;
        margin-top: 4px;
    }

    /* Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Glassmorphism Metric Cards */
    .metric-glass-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.6), rgba(17, 24, 39, 0.8));
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-glass-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #F9FAFB;
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Prompt Recommendation Cards */
    .prompt-card {
        background: rgba(31, 41, 55, 0.5);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.2s ease;
        height: 100%;
    }

    .prompt-card:hover {
        background: rgba(55, 65, 81, 0.7);
        border-color: #8B5CF6;
        transform: translateY(-2px);
    }

    .prompt-card-icon {
        font-size: 1.5rem;
        margin-bottom: 8px;
    }

    .prompt-card-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #F3F4F6;
        margin-bottom: 4px;
    }

    .prompt-card-desc {
        font-size: 0.8rem;
        color: #9CA3AF;
    }

    /* Custom Chat Container */
    .chat-bubble-user {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: #FFFFFF;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin-left: auto;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    .chat-bubble-ai {
        background: rgba(31, 41, 55, 0.8);
        border: 1px solid #374151;
        color: #E5E7EB;
        padding: 16px 20px;
        border-radius: 18px 18px 18px 4px;
        margin-right: auto;
        max-width: 85%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .entity-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 24, 39, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #1F2937;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        color: #9CA3AF;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #374151 !important;
        color: #F9FAFB !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, status: dict):
    """
    Render ChatGPT/Gemini style top header bar with live connection badge.
    """
    connected = status.get("connected", False)
    badge_class = "status-online" if connected else "status-offline"
    badge_icon = "🟢" if connected else "🔴"
    badge_text = "Ollama Active" if connected else "Ollama Offline"

    st.markdown(
        f"""
    <div class="chatgpt-header">
        <div class="header-title-box">
            <h2>{title}</h2>
            <div class="header-subtitle">{subtitle}</div>
        </div>
        <div>
            <span class="status-badge {badge_class}">
                {badge_icon} {badge_text}
            </span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
