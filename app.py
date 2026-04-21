import streamlit as st

import os
from pathlib import Path

from dotenv import load_dotenv

from ai_helper import CodeExplanationError, answer_question_about_code, explain_code
from embeddings import index_code, search_similar


_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)

st.set_page_config(page_title="AI Codebase Explainer", layout="wide")
st.title("AI Codebase Explainer")

st.sidebar.header("API key status")
_key = os.getenv("OPENAI_API_KEY", "")
if _key:
    st.sidebar.success(f"OPENAI_API_KEY loaded (…{_key[-4:]})")
else:
    st.sidebar.error("OPENAI_API_KEY not loaded")
    st.sidebar.caption(f"Expected at: {_DOTENV_PATH}")

st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader(
    "Upload a code file",
    type=None,
    accept_multiple_files=False,
)

st.sidebar.header("AI settings")
model = st.sidebar.text_input("Model (optional)", value="")
top_k = st.sidebar.slider("Retrieval chunks (top_k)", min_value=1, max_value=10, value=4)

code_text = ""
filename = None

if uploaded is not None:
    filename = uploaded.name
    raw = uploaded.getvalue()
    try:
        code_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        code_text = raw.decode("latin-1", errors="replace")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Uploaded code")
    if uploaded is None:
        st.info("Upload a file from the sidebar to preview it here.")
    else:
        st.caption(filename)
        st.code(code_text, language=None)

with col_right:
    st.subheader("Explanation / Chat")
    if uploaded is None:
        st.info("Upload a file to generate an explanation or ask questions.")
    else:
        # Index once per upload (per session) so retrieval is fast.
        file_key = f"{filename}:{len(code_text)}"
        if st.session_state.get("indexed_file_key") != file_key:
            with st.spinner("Indexing file for search…"):
                try:
                    n_chunks = index_code(code_text, filename=filename or "uploaded")
                except Exception as e:
                    st.error("Failed to index the file for embeddings.")
                    st.exception(e)
                    n_chunks = 0
                st.session_state["indexed_file_key"] = file_key
                st.session_state["indexed_chunks"] = n_chunks

        st.caption(f"Indexed chunks: {st.session_state.get('indexed_chunks', 0)}")

        tabs = st.tabs(["Explain", "Chat"])

        with tabs[0]:
            if st.button("Explain code", type="primary", use_container_width=True):
                with st.spinner("Calling OpenAI…"):
                    try:
                        explanation = explain_code(
                            code_text,
                            filename=filename,
                            model=model.strip() or None,
                        )
                    except CodeExplanationError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error("Unexpected error.")
                        st.exception(e)
                    else:
                        st.markdown(explanation)

        with tabs[1]:
            if "qa_history" not in st.session_state:
                st.session_state.qa_history = []

            question = st.text_input(
                "Ask a question about the uploaded code",
                key="qa_question",
                placeholder="e.g., What does the main function do?",
            )
            ask = st.button("Ask", type="primary")

            if ask:
                q = (question or "").strip()
                if not q:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Searching relevant code and asking OpenAI…"):
                        try:
                            hits = search_similar(q, top_k=top_k)
                            context_chunks = [doc for (doc, _meta, _dist) in hits]
                            a = answer_question_about_code(
                                question=q,
                                context_chunks=context_chunks,
                                filename=filename,
                                model=model.strip() or None,
                            ).text
                        except CodeExplanationError as e:
                            a = str(e)
                        except Exception as e:
                            st.error("Unexpected error.")
                            st.exception(e)
                            a = ""

                    st.session_state.qa_history.append({"q": q, "a": a})
                    st.session_state.qa_question = ""

            if st.session_state.qa_history:
                st.divider()
                for item in reversed(st.session_state.qa_history):
                    st.markdown(f"**You:** {item['q']}")
                    if item["a"]:
                        st.markdown(f"**AI:** {item['a']}")
                    else:
                        st.markdown("**AI:** (no response)")
                    st.divider()

