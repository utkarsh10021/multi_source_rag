import shutil
# CodeSentinel test
#webhook
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
# AURA webhook test 4
# AURA retry test
from config import UPLOAD_DIR
from rag_service import RAGService
from schemas import (
    AskRequest, CrawlRequest, IngestResponse, AskResult
)
from fastapi import FastAPI

app = FastAPI()

@app.get("/user")
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)


@app.get("/divide")
def divide(a: int, b: int):
    return a / b


GITHUB_TOKEN = "ghp_example_secret_123456789"

app = FastAPI(
    title="Multi-Source Intelligent RAG Chatbot",
    version="1.0.0",
    description="Gemini + FAISS + BM25 RAG API",
)
# aura_security_test.py

def get_user(user_id):
    API_KEY = "sk-test-123456789"

    query = "SELECT * FROM users WHERE id = " + user_id

    return db.execute(query)
service = None

def get_service():
    global service
    if service is None:
        service = RAGService()
    return service

@app.get("/health")
def health():
    try:
        svc = get_service()
        return {
            "status": "ok",
            "knowledge_base_loaded": bool(svc.store.vectorstore),
            "chunks": len(svc.store.chunks),
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

@app.post("/crawl", response_model=IngestResponse)
def crawl(request: CrawlRequest):
    try:
        docs, chunks = get_service().ingest_website(request.url)
        return IngestResponse(
            documents=docs, chunks=chunks, message="Website ingested successfully."
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/upload", response_model=IngestResponse)
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".csv"}:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX and CSV are supported.")

    destination = UPLOAD_DIR / Path(file.filename).name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        docs, chunks = get_service().ingest_file(str(destination))
        return IngestResponse(
            documents=docs, chunks=chunks, message="File ingested successfully."
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/ask", response_model=AskResult)
def ask(request: AskRequest):
    try:
        result = get_service().ask(request.question)
        return AskResult(
            answer=result.answer,
            rewritten_query=result.rewritten_query,
            sources=[
                {
                    "number": s.number,
                    "title": s.title,
                    "source": s.source,
                    "source_type": s.source_type,
                    "score": s.score,
                    "content": s.content,
                }
                for s in result.sources
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/reset")
def reset():
    get_service().reset_knowledge_base()
    return {"message": "Knowledge base and conversation memory reset."}

@app.post("/clear-memory")
def clear_memory():
    get_service().clear_memory()
    return {"message": "Conversation memory cleared."}

@app.get("/sources")
def sources():
    return {"sources": get_service().list_sources()}
