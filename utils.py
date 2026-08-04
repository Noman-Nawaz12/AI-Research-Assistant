"""
utils.py
---------
Shared helper functions and error-handling utilities used across
the app (Streamlit-facing error display, validation helpers, etc.)
"""

import streamlit as st


def show_error(message: str):
    """Display a consistent error message in the Streamlit UI."""
    st.error(f"❌ {message}")


def show_warning(message: str):
    st.warning(f"⚠️ {message}")


def show_success(message: str):
    st.success(f"✅ {message}")


def show_info(message: str):
    st.info(f"ℹ️ {message}")


def validate_api_key(api_key: str) -> bool:
    """Basic sanity check on the Groq API key format before making calls."""
    if not api_key or not api_key.strip():
        return False
    if len(api_key.strip()) < 20:  # Groq keys are long; catches obvious typos/empty input
        return False
    return True


def validate_uploaded_files(uploaded_files: list) -> bool:
    """Check that at least one file was uploaded."""
    return bool(uploaded_files) and len(uploaded_files) > 0


def safe_run(func, *args, error_prefix: str = "An error occurred", **kwargs):
    """
    Wraps a function call in a try/except and shows a Streamlit error
    instead of crashing the app. Returns None on failure.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        show_error(f"{error_prefix}: {str(e)}")
        return None