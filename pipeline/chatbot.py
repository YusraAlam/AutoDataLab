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

        if genai is None:
            return

        if not self.api_key:
            return

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
    # PROCESS QUERY
    # =====================================================

    def process_query(self, user_query: str) -> str:

        if not self.client:

            if genai is None:
                return (
                    "Gemini package is not installed. "
                    "Add google-genai to requirements.txt "
                    "and redeploy the app."
                )

            if not self.api_key:
                return (
                    "Gemini API key is not configured. "
                    "Add GEMINI_API_KEY to Streamlit Cloud Secrets."
                )

            return (
                "Gemini client could not be initialized. "
                "Check your API key and Streamlit Cloud logs."
            )

        try:

            # -------------------------------------------------
            # DATAFRAME INFORMATION
            # -------------------------------------------------

            buffer = StringIO()

            self.df.info(buf=buffer)

            df_info_str = buffer.getvalue()

            sample_data = self.df.head(3).to_string()

            # -------------------------------------------------
            # CHAT HISTORY
            # -------------------------------------------------

            conversation_context = (
                self._build_history_context()
            )

            # -------------------------------------------------
            # GEMINI PROMPT
            # -------------------------------------------------

            prompt = (
                "You are an expert Data Analyst Python Assistant.\n\n"
                "The user is asking a question about a pandas "
                "DataFrame named df.\n\n"

                "DataFrame information:\n"
                + df_info_str
                + "\n\n"

                "First 3 rows:\n"
                + sample_data
                + "\n\n"

                "Recent conversation:\n"
                + conversation_context
                + "\n\n"

                "Current user question:\n"
                + user_query
                + "\n\n"

                "Instructions:\n"
                "1. Write short executable Python code.\n"
                "2. Use only df and pd.\n"
                "3. Do not import any modules.\n"
                "4. Do not access files.\n"
                "5. Do not use exec or eval.\n"
                "6. Put the code inside a python code block.\n"
                "7. Store the final answer in a variable named result.\n"
                "8. result can be a string, number, or pandas DataFrame."
            )

            # -------------------------------------------------
            # GEMINI REQUEST
            # -------------------------------------------------

            response = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            generated_text = response.text

            # -------------------------------------------------
            # EXTRACT PYTHON CODE
            # -------------------------------------------------

            if "```python" in generated_text:

                code = (
                    generated_text
                    .split("```python", 1)[1]
                    .split("```", 1)[0]
                    .strip()
                )

                # -------------------------------------------------
                # EXECUTE GENERATED CODE
                # -------------------------------------------------

                execution_result = self._safe_execute(code)

                if (
                    isinstance(execution_result, str)
                    and execution_result.startswith("ERROR:")
                ):
                    return execution_result

                # -------------------------------------------------
                # EXPLANATION PROMPT
                # -------------------------------------------------

                explanation_prompt = (
                    "The user asked:\n"
                    + user_query
                    + "\n\n"

                    "The Python analysis produced this result:\n"
                    + str(execution_result)
                    + "\n\n"

                    "Give a short and professional answer "
                    "in 2-3 sentences.\n"
                    "Do not provide Python code.\n"
                    "Clearly explain the result to the user."
                )

                final_answer = (
                    self.client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=explanation_prompt
                    )
                )

                return final_answer.text

            else:

                return generated_text

        except Exception as e:

            return (
                f"ERROR: Gemini API request failed: {str(e)}"
            )

    # =====================================================
    # SAFE CODE EXECUTION
    # =====================================================

    def _safe_execute(self, code: str):

        forbidden = [
            "import",
            "open(",
            "exec(",
            "eval(",
            "__",
            "os.",
            "sys.",
            "subprocess"
        ]

        if any(
            token in code
            for token in forbidden
        ):

            return (
                "ERROR: Generated code contained "
                "a disallowed operation and was blocked."
            )

        safe_builtins = {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "sorted": sorted,
            "abs": abs
        }

        local_vars = {
            "df": self.df,
            "pd": pd
        }

        try:

            exec(
                code,
                {
                    "__builtins__": safe_builtins
                },
                local_vars
            )

            return local_vars.get(
                "result",
                "Code executed successfully."
            )

        except Exception as e:

            return (
                f"ERROR: Code execution error: {e}"
            )

    # =====================================================
    # CHAT INTERFACE
    # =====================================================

    def render_chat_interface(self):

        st.subheader(
            "Smart AI Data Chatbot"
        )

        st.caption(
            "Powered by Gemini AI - ask questions "
            "about your dataset in plain English."
        )

        # -------------------------------------------------
        # CHECK CONFIGURATION
        # -------------------------------------------------

        if genai is None:

            st.warning(
                "Gemini package is missing. "
                "Add google-genai to requirements.txt."
            )

        elif not self.api_key:

            st.warning(
                "Set GEMINI_API_KEY in Streamlit Cloud "
                "Secrets to enable the chatbot."
            )

        elif not self.client:

            st.warning(
                "Gemini client could not be initialized. "
                "Check the API key and app logs."
            )

        # -------------------------------------------------
        # INITIAL CHAT HISTORY
        # -------------------------------------------------

        if "chat_history" not in st.session_state:

            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I'm your AI Data Assistant. "
                        "Ask me anything about your dataset!"
                    )
                }
            ]

        # -------------------------------------------------
        # CLEAR CHAT
        # -------------------------------------------------

        col1, col2 = st.columns([5, 1])

        with col2:

            if st.button("Clear Chat"):

                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content": (
                            "Chat cleared. "
                            "Ask me anything about your dataset!"
                        )
                    }
                ]

                st.rerun()

        # -------------------------------------------------
        # DISPLAY CHAT HISTORY
        # -------------------------------------------------

        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        # -------------------------------------------------
        # CHAT INPUT
        # -------------------------------------------------

        prompt = st.chat_input(
            "Ex: What are the top 5 values in column X?"
        )

        if prompt:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            with st.chat_message("user"):

                st.markdown(prompt)

            with st.spinner(
                "Analyzing dataset with AI..."
            ):

                response_text = self.process_query(
                    prompt
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response_text
                }
            )

            with st.chat_message("assistant"):

                st.markdown(response_text)
