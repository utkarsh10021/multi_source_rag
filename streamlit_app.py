import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Multi-Source Intelligent RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Source Intelligent RAG Chatbot")
st.caption("Gemini + FAISS + BM25 + re-ranking + query rewriting")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge Base")

    url = st.text_input("Website URL")
    if st.button("Crawl website", use_container_width=True):
        if not url:
            st.warning("Enter a URL.")
        else:
            try:
                r = requests.post(
                    f"{API_URL}/crawl", json={"url": url}, timeout=180
                )
                if r.ok:
                    st.success(r.json()["message"])
                    st.info(f"Documents: {r.json()['documents']} | Chunks: {r.json()['chunks']}")
                else:
                    st.error(r.text)
            except Exception as exc:
                st.error(str(exc))

    uploaded = st.file_uploader(
        "Upload PDF / DOCX / CSV",
        type=["pdf", "docx", "csv"],
    )

    if st.button("Ingest file", use_container_width=True):
        if not uploaded:
            st.warning("Choose a file.")
        else:
            try:
                r = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    },
                    timeout=180,
                )
                if r.ok:
                    data = r.json()
                    st.success(data["message"])
                    st.info(f"Documents: {data['documents']} | Chunks: {data['chunks']}")
                else:
                    st.error(r.text)
            except Exception as exc:
                st.error(str(exc))

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        try:
            requests.post(f"{API_URL}/clear-memory", timeout=30)
        finally:
            st.session_state.messages = []
            st.rerun()

    if st.button("Reset knowledge base", use_container_width=True):
        try:
            requests.post(f"{API_URL}/reset", timeout=30)
        finally:
            st.session_state.messages = []
            st.rerun()

    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        st.write("**API:**", health.get("status"))
        st.write("**Chunks:**", health.get("chunks", 0))
    except Exception:
        st.warning("API is not reachable.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            for source in message.get("sources", []):
                with st.expander(
                    f"[Source {source['number']}] {source['title']} — {source['source_type']}"
                ):
                    st.write(source["source"])
                    st.caption(f"Score: {source['score']}")
                    st.write(source["content"])

question = st.chat_input("Ask a question about your sources...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=180,
        )
        if not response.ok:
            raise RuntimeError(response.text)

        data = response.json()
        with st.chat_message("assistant"):
            st.markdown(data["answer"])
            if data.get("rewritten_query"):
                with st.expander("Search query used"):
                    st.code(data["rewritten_query"])
            for source in data.get("sources", []):
                with st.expander(
                    f"[Source {source['number']}] {source['title']} — {source['source_type']}"
                ):
                    st.write(source["source"])
                    st.caption(f"Score: {source['score']}")
                    st.write(source["content"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "sources": data.get("sources", []),
        })

    except Exception as exc:
        error = f"Error: {exc}"
        with st.chat_message("assistant"):
            st.error(error)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error,
            "sources": [],
        })
