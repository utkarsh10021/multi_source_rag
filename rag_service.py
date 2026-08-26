import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import (
    GOOGLE_API_KEY, MODEL_NAME, EMBEDDING_MODEL, TEMPERATURE,
    MAX_OUTPUT_TOKENS, TOP_K, FINAL_K
)
from ingestion import crawl_website, load_file, create_chunks
from memory import ConversationMemory
from vector_store import PersistentHybridStore
from models import SourceInfo, AskResponse
from logging_config import setup_logging

logger = setup_logging()

PROMPT = PromptTemplate.from_template("""
You are a multi-source RAG assistant.

Previous conversation:
{chat_history}

Retrieved sources:
{context}

User question:
{question}

Rules:
1. Answer primarily from the retrieved sources.
2. Use conversation history only to resolve references such as "it", "they", or "that".
3. Do not invent facts or information that is not supported by the retrieved sources.
4. If the sources do not contain enough information, say:
"I don't have enough information from the provided sources to answer that question."
5. Keep the answer concise but complete.
6. Structure the answer with clear headings and bullet points.
7. Cite sources using [Source 1], [Source 2], etc.
8. Do not invent URLs.
9. If the sources are insufficient, explicitly say so.
10. Give a complete answer and do not stop mid-sentence.
11. Keep the answer under 500 words.
12. When information comes from multiple sources, clearly distinguish or combine the information from those sources.
13. For questions comparing or connecting multiple sources, use relevant information from each source rather than relying primarily on one source.
14. Make the answer directly relevant to the user's question.

Answer:
""")

REWRITE_PROMPT = PromptTemplate.from_template("""
Rewrite the user's latest question into a standalone search query.

Conversation history:
{history}

Latest question:
{question}

Return ONLY the rewritten search query. Do not answer it.
""")

class RAGService:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not configured.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            google_api_key=GOOGLE_API_KEY,
        )
        self.memory = ConversationMemory()
        self.store = PersistentHybridStore(self.embeddings)
        self.store.load()

    def ingest_documents(self, documents):
        chunks = create_chunks(documents)
        if not chunks:
            raise ValueError("No usable text was extracted.")
        if self.store.vectorstore is None:
            self.store.build(chunks)
        else:
            self.store.chunks.extend(chunks)
            self.store.vectorstore.add_documents(chunks)
            self.store._build_bm25()
            self.store.save()
        return len(documents), len(chunks)

    def ingest_website(self, url):
        docs = crawl_website(url)
        if not docs:
            raise ValueError("No text could be extracted from the website.")
        return self.ingest_documents(docs)

    def ingest_file(self, file_path):
        docs = load_file(file_path)
        if not docs:
            raise ValueError("No text could be extracted from the file.")
        return self.ingest_documents(docs)

    def rewrite_query(self, question):
        history = self.memory.as_text()
        if history == "No previous conversation.":
            return question
        try:
            result = self.llm.invoke(REWRITE_PROMPT.format(
                history=history, question=question
            ))
            rewritten = result.content
            if isinstance(rewritten, list):
                rewritten = " ".join(
                    x.get("text", "") for x in rewritten if isinstance(x, dict)
                )
            return str(rewritten).strip() or question
        except Exception as exc:
            logger.warning("Query rewriting failed: %s", exc)
            return question

    def ask(self, question):
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if not self.store.vectorstore:
            raise ValueError("Knowledge base is empty. Add a website or file first.")

        started = time.perf_counter()
        rewritten = self.rewrite_query(question)
        results = self.store.hybrid_search(
        rewritten,
        top_k=TOP_K,
        final_k=FINAL_K,
        original_query=question,
        )

        if not results:
            answer = "I don't have enough information from the provided sources to answer that question."
            self.memory.add(question, answer)
            return AskResponse(answer=answer, rewritten_query=rewritten)

        context_parts = []
        sources = []
        for i, (doc, distance, keyword_score, score) in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Type: {doc.metadata.get('type', 'unknown')}\n"
                f"Title: {doc.metadata.get('title', 'Unknown')}\n"
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Content:\n{doc.page_content}"
            )
            sources.append(SourceInfo(
                number=i,
                title=doc.metadata.get("title", "Unknown"),
                source=doc.metadata.get("source", "Unknown"),
                source_type=doc.metadata.get("type", "unknown"),
                score=round(float(score), 6),
                content=doc.page_content,
            ))

        prompt = PROMPT.format(
            chat_history=self.memory.as_text(),
            context="\n\n".join(context_parts),
            question=question,
        )

        response = self.llm.invoke(prompt)
        logger.info("LLM RESPONSE METADATA: %s", response.response_metadata)
        answer = response.content
        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", "") for item in answer
                if isinstance(item, dict)
            )
        answer = str(answer)

        self.memory.add(question, answer)
        elapsed = time.perf_counter() - started
        logger.info(
            "ask completed | original=%r | rewritten=%r | sources=%d | latency=%.3fs",
            question, rewritten, len(sources), elapsed
        )
        return AskResponse(
            answer=answer,
            sources=sources,
            rewritten_query=rewritten,
        )

    def clear_memory(self):
        self.memory.clear()

    def reset_knowledge_base(self):
        self.store.reset()
        self.memory.clear()

    def list_sources(self):
        unique = {}
        for d in self.store.chunks:
            src = d.metadata.get("source", "Unknown")
            unique[src] = {
                "source": src,
                "title": d.metadata.get("title", "Unknown"),
                "type": d.metadata.get("type", "unknown"),
            }
        return list(unique.values())
