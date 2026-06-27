"""Streamlit chat and document upload interface."""

import os
import sys
import tempfile
from pathlib import Path

# Streamlit puts frontend/ on sys.path; ensure project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient

st.set_page_config(page_title="RAG Milvus Assistant", page_icon="📚", layout="wide")


def get_backend_client() -> BackendAPIClient:
    base_url = st.sidebar.text_input("Backend URL", value=st.session_state["backend_url"])
    st.session_state["backend_url"] = base_url
    return BackendAPIClient(base_url=base_url)


def initialize_state() -> None:
    default_backend_url = os.environ.get("STREAMLIT_BACKEND_URL", "http://localhost:8000")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploaded_documents", [])
    st.session_state.setdefault("backend_url", default_backend_url)


def render_sidebar(client: BackendAPIClient) -> None:
    st.sidebar.header("Controls")
    st.sidebar.caption("Upload PDFs, inspect service health, and tune retrieval depth.")
    top_k = st.sidebar.slider("Top K retrieval depth", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Check backend health"):
        try:
            health = client.health()
            st.sidebar.success(f"{health['status']} - {health['app_name']}")
        except APIClientError as exc:
            st.sidebar.error(str(exc))

    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None and st.sidebar.button("Send PDF to backend"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)
        try:
            with st.spinner("Indexing document..."):
                result = client.upload_pdf(temp_path)
            st.session_state["uploaded_documents"].append(result)
            st.sidebar.success(f"Indexed {result['filename']} with {result['chunk_count']} chunks")
        except APIClientError as exc:
            st.sidebar.error(str(exc))
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    return top_k


def render_uploaded_documents() -> None:
    if not st.session_state["uploaded_documents"]:
        st.info("No documents indexed yet.")
        return

    st.subheader("Indexed documents")
    for document in reversed(st.session_state["uploaded_documents"]):
        st.write(
            f"{document['filename']} | pages: {document['page_count']} | chunks: {document['chunk_count']} | id: {document['document_id']}"
        )


def render_chat(client: BackendAPIClient, top_k: int) -> None:
    st.subheader("Chat")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the uploaded PDFs")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and generating answer..."):
                response = client.query(question=question, top_k=top_k)
            answer = response["answer"]
            st.markdown(answer)
            if response.get("source_chunks"):
                with st.expander("Retrieved sources", expanded=False):
                    for chunk in response["source_chunks"]:
                        st.markdown(
                            f"**{chunk['source_filename']}** page {chunk['page_number']} | chunk {chunk['chunk_index']} | score {chunk['score']:.4f}"
                        )
                        st.write(chunk["chunk_text"])
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except APIClientError as exc:
            error_text = f"Request failed: {exc}"
            st.error(error_text)
            st.session_state["messages"].append({"role": "assistant", "content": error_text})


def main() -> None:
    initialize_state()
    st.title("RAG Milvus Assistant")
    st.caption("Upload PDFs, index them into Milvus, and query them with Gemini embeddings + Groq answers.")
    client = get_backend_client()
    top_k = render_sidebar(client)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_uploaded_documents()
    with col_right:
        render_chat(client, top_k)


if __name__ == "__main__":
    main()
