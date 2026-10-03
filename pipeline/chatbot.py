import streamlit as st
import pandas as pd
from io import StringIO

try:
    from google import genai
except ImportError:
    genai = None


def _get_api_key():
    """Read the Gemini API key from Streamlit secrets, never from source code."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


class DataChatbotEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.api_key = _get_api_key()
        self.client = None
        if genai is not None and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def _build_history_context(self, max_turns: int = 4) -> str:
        """Include the last few turns so the bot has conversational memory."""
        history = st.session_state.get("chat_history", [])
        recent = history[-(max_turns * 2):]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def process_query(self, user_query: str) -> str:
        if not self.client:
            return ("⚠️ Gemini API key not configured. Add it to `.streamlit/secrets.toml` as "
                    "`GEMINI_API_KEY = \"your-key\"` — never hardcode it in source files.")

        try:
            buffer = StringIO()
            self.df.info(buf=buffer)
            df_info_str = buffer.getvalue()
            sample_data = self.df.head(3).to_string()
            conversation_context = self._build_history_context()

            prompt = f"""
You are an expert Data Analyst Python Assistant.
The user is asking a question about a pandas DataFrame named `df`.

DataFrame info:
{df_info_str}

First 3 rows sample:
{sample_data}

Recent conversation (for context only):
{conversation_context}

Current user question: "{user_query}"

Instructions:
1. Write short, executable Python code using only `df` and `pd` to compute the answer.
2. Put the code strictly inside ```python ... ``` code blocks.
3. Assign the final result to a variable named `result` (string, number, or DataFrame).
4. Do not import any modules, access the filesystem, or use exec/eval yourself.
"""
            response = self.client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
            generated_text = response.text

            if "```python" in generated_text:
                code = generated_text.split("```python")[1].split("```")[0].strip()
                execution_result = self._safe_execute(code)

                if isinstance(execution_result, str) and execution_result.startswith("❌"):
                    return execution_result

                explanation_prompt = f"""
User asked: "{user_query}"
Data analysis result: {execution_result}

Provide a short, professional 2-3 sentence answer summarizing this finding clearly for the user.
"""
                final_answer = self.client.models.generate_content(model='gemini-3.6-flash',
                                                                      contents=explanation_prompt)
                return final_answer.text
            else:
                return generated_text

        except Exception as e:
            return f"❌ Error executing AI Query: `{str(e)}`"

    def _safe_execute(self, code: str):
        """Execute AI-generated code in a restricted namespace — no builtins, no imports,
        only `df` and `pd` exposed. This limits (but does not eliminate) the risk of running
        LLM-generated code; review generated code periodically if this is used beyond personal use."""
        forbidden = ["import", "open(", "exec(", "eval(", "__", "os.", "sys.", "subprocess"]
        if any(tok in code for tok in forbidden):
            return "❌ Generated code contained a disallowed operation and was blocked for safety."

        safe_builtins = {"len": len, "range": range, "min": min, "max": max, "sum": sum,
                          "round": round, "sorted": sorted, "abs": abs}
        local_vars = {"df": self.df, "pd": pd}
        try:
            exec(code, {"__builtins__": safe_builtins}, local_vars)
            return local_vars.get("result", "Code executed successfully.")
        except Exception as e:
            return f"❌ Code execution error: {e}"

    def render_chat_interface(self):
        st.subheader("🤖 Smart AI Data Chatbot")
        st.caption("Powered by Gemini AI — ask anything about your dataset in plain English.")

        if not self.client:
            st.warning("⚠️ Set `GEMINI_API_KEY` in `.streamlit/secrets.toml` to enable the chatbot.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant",
                 "content": "Hello! I'm your AI Data Assistant. Ask me anything about your dataset!"}
            ]

        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = [
                    {"role": "assistant", "content": "Chat cleared. Ask me anything about your dataset!"}
                ]
                st.rerun()

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: What are the top 5 values in column X?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Analyzing dataset with AI..."):
                response_text = self.process_query(prompt)

            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)
