import os
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Source Intelligent RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# GEMINI API KEY
# ============================================================
# Streamlit Cloud stores the key in Secrets.
# We set the environment variable BEFORE importing RAGService
# because config.py reads GOOGLE_API_KEY during import.

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

elif "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]


# ============================================================
# IMPORT RAG SERVICE
# ============================================================

try:
    from rag_service import RAGService
except Exception as exc:
    st.error("Failed to load the RAG service.")
    st.exception(exc)
    st.stop()


# ============================================================
# CREATE ONE RAG SERVICE INSTANCE
# ============================================================

@st.cache_resource
def get_rag_service():
    return RAGService()


try:
    rag = get_rag_service()
except Exception as exc:
    st.error("Failed to initialize the RAG system.")
    st.exception(exc)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Multi-Source Intelligent RAG Chatbot")

st.caption(
    "Gemini + FAISS + BM25 + Re-ranking + Query Rewriting"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    # --------------------------------------------------------
    # WEBSITE CRAWLING
    # --------------------------------------------------------

    st.subheader("🌐 Website")

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
    )

    if st.button(
        "🔍 Crawl Website",
        use_container_width=True,
    ):

        if not url.strip():

            st.warning("Please enter a website URL.")

        else:

            with st.spinner("Crawling website and creating embeddings..."):

                try:

                    documents, chunks = rag.ingest_website(
                        url.strip()
                    )

                    st.success("Website successfully added.")

                    st.info(
                        f"Documents: {documents} | "
                        f"Chunks: {chunks}"
                    )

                except Exception as exc:

                    st.error(
                        f"Website ingestion failed: {exc}"
                    )


    st.divider()


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    st.subheader("📄 Documents")

    uploaded = st.file_uploader(
        "Upload PDF / DOCX / CSV",
        type=["pdf", "docx", "csv"],
    )

    if st.button(
        "📥 Ingest File",
        use_container_width=True,
    ):

        if uploaded is None:

            st.warning("Please select a file.")

        else:

            temp_path = None

            try:

                suffix = Path(uploaded.name).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:

                    temp_file.write(
                        uploaded.getvalue()
                    )

                    temp_path = temp_file.name

                with st.spinner(
                    "Processing document and creating embeddings..."
                ):

                    documents, chunks = rag.ingest_file(
                        temp_path
                    )

                st.success(
                    f"{uploaded.name} successfully added."
                )

                st.info(
                    f"Documents: {documents} | "
                    f"Chunks: {chunks}"
                )

            except Exception as exc:

                st.error(
                    f"File ingestion failed: {exc}"
                )

            finally:

                if temp_path and os.path.exists(temp_path):

                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass


    st.divider()


    # --------------------------------------------------------
    # KNOWLEDGE BASE STATUS
    # --------------------------------------------------------

    st.subheader("📊 Knowledge Base")

    try:

        chunks_count = len(rag.store.chunks)

        sources = rag.list_sources()

        st.metric(
            "Total Chunks",
            chunks_count,
        )

        st.metric(
            "Sources",
            len(sources),
        )

        if sources:

            with st.expander("View Sources"):

                for source in sources:

                    st.write(
                        f"**{source['title']}**"
                    )

                    st.caption(
                        f"{source['type']} — "
                        f"{source['source']}"
                    )

    except Exception as exc:

        st.warning(
            f"Could not read knowledge base status: {exc}"
        )


    st.divider()


    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True,
    ):

        try:

            rag.clear_memory()

        except Exception:
            pass

        st.session_state.messages = []

        st.rerun()


    # --------------------------------------------------------
    # RESET KNOWLEDGE BASE
    # --------------------------------------------------------

    if st.button(
        "🗑️ Reset Knowledge Base",
        use_container_width=True,
    ):

        try:

            rag.reset_knowledge_base()

            st.session_state.messages = []

            st.success(
                "Knowledge base reset successfully."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"Reset failed: {exc}"
            )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            if message.get("rewritten_query"):

                with st.expander(
                    "🔎 Search Query Used"
                ):

                    st.code(
                        message["rewritten_query"]
                    )

            for source in message.get(
                "sources",
                [],
            ):

                with st.expander(
                    f"[Source {source['number']}] "
                    f"{source['title']} — "
                    f"{source['source_type']}"
                ):

                    st.write(
                        f"**Source:** "
                        f"{source['source']}"
                    )

                    st.caption(
                        f"Score: {source['score']}"
                    )

                    st.write(
                        source["content"]
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your sources..."
)


# ============================================================
# ASK RAG
# ============================================================

if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # RAG RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching sources and generating answer..."
            ):

                response = rag.ask(
                    question
                )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.markdown(
                response.answer
            )


            # ------------------------------------------------
            # REWRITTEN QUERY
            # ------------------------------------------------

            if response.rewritten_query:

                with st.expander(
                    "🔎 Search Query Used"
                ):

                    st.code(
                        response.rewritten_query
                    )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            source_data = []

            for source in response.sources:

                source_item = {
                    "number": source.number,
                    "title": source.title,
                    "source": source.source,
                    "source_type": source.source_type,
                    "score": source.score,
                    "content": source.content,
                }

                source_data.append(
                    source_item
                )

                with st.expander(
                    f"[Source {source.number}] "
                    f"{source.title} — "
                    f"{source.source_type}"
                ):

                    st.write(
                        f"**Source:** "
                        f"{source.source}"
                    )

                    st.caption(
                        f"Score: {source.score}"
                    )

                    st.write(
                        source.content
                    )


            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": source_data,
                    "rewritten_query": response.rewritten_query,
                }
            )


        except Exception as exc:

            error_message = (
                f"RAG error: {exc}"
            )

            st.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "rewritten_query": "",
                }
            )