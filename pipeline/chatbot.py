import streamlit as st
import pandas as pd
from io import StringIO

try:
    from google import genai
except ImportError:
    genai = None


# =========================================================
# GEMINI API KEY
# =========================================================

def _get_api_key():
    """Read Gemini API key from Streamlit secrets."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")

        if api_key:
            return api_key

        return None

    except Exception:
        return None


# =========================================================
# DATA CHATBOT ENGINE
# =========================================================

class DataChatbotEngine:

    def __init__(self, df: pd.DataFrame):

        self.df = df
        self.api_key = _get_api_key()
        self.client = None

        # Check Gemini package
        if genai is None:
            return

        # Check API key
        if not self.api_key:
            return

        # Create Gemini client
        try:
            self.client = genai.Client(
                api_key=self.api_key
            )

        except Exception as e:
            st.error(f"Gemini client error: {e}")
            self.client = None

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    def _build_history_context(self, max_turns: int = 4) -> str:

        history = st.session_state.get(
            "chat_history",
            []
        )

        recent = history[-(max_turns * 2):]

        lines = []

        for msg in recent:

            role = (
                "User"
                if msg["role"] == "user"
                else "Assistant"
            )

            lines.append(
                f"{role}: {msg['content']}"
            )

        return "\n".join(lines)

    # =====================================================
    # PROCESS USER QUERY
    # =====================================================

    def process_query(self, user_query: str) -> str:

        # Gemini not configured
        if not self.client:

            if genai is None:
                return (
                    "Gemini package is not installed. "
                    "Add `google-genai` to requirements.txt "
                    "and redeploy the app."
                )

            if not self.api_key:
                return (
                    "Gemini API key is not configured. "
                    "Add `GEMINI_API_KEY` to Streamlit Cloud Secrets."
                )

            return (
                "Gemini client could not be initialized. "
                "Check your API key and Streamlit Cloud logs."
            )

        try:

            # ---------------------------------------------
            # DATAFRAME INFORMATION
            # ---------------------------------------------

            buffer = StringIO()

            self.df.info(buf=buffer)

            df_info_str = buffer.getvalue()

            sample_data = self.df.head(3).to_string()

            # ---------------------------------------------
            # CHAT HISTORY
            # ---------------------------------------------

            conversation_context = (
                self._build_history_context()
            )

            # ---------------------------------------------
            # PROMPT
            # ---------------------------------------------

            prompt = f"""
You are an expert Data Analyst Python Assistant.

The user is asking a question about a pandas DataFrame
named `df`.

DataFrame information:
{df_info_str}

First 3 rows:
{sample_data}

Recent conversation:
{conversation_context}

Current user question:
"{user_query}"

Instructions:

1. Write short executable Python code.
2. Use only `df` and `pd`.
3. Do not import any modules.
4. Do not access files.
5. Do not use exec or eval.
6. Put the code strictly inside:
```python
...
