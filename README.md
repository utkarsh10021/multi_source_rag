# Multi-Source Intelligent RAG Chatbot with Gemini

A complete multi-source RAG application using:

- Playwright website crawling and JavaScript rendering
- PDF, DOCX and CSV ingestion
- Recursive character chunking
- Gemini embeddings
- Persistent FAISS
- BM25 + FAISS hybrid retrieval
- Score-based re-ranking
- Gemini query rewriting
- Conversation memory
- Source citations
- Streamlit UI
- FastAPI backend
- RAG evaluation
- Logging and error handling
- Automated tests
- Docker and Docker Compose

## Architecture

```text
Website / PDF / DOCX / CSV
            |
            v
      Document Loaders
            |
            v
   Recursive Chunking
            |
            v
     Gemini Embeddings
            |
            v
     Persistent FAISS
            |
       +----+----+
       |         |
       v         v
    Semantic   BM25
       |         |
       +----+----+
            |
            v
       Hybrid Search
            |
            v
        Re-ranking
            |
            v
      Query Rewriting
            |
            v
       Gemini LLM
            |
            v
      Answer + Sources
            |
       +----+----+
       |         |
       v         v
   FastAPI   Streamlit
```

## 1. Setup

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`.

## 2. CLI

```powershell
python cli.py
```

Examples:

```text
add https://example.com
add C:\path\document.pdf
add C:\path\document.docx
add C:\path\data.csv
What is this document about?
```

## 3. FastAPI

```powershell
uvicorn api:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## 4. Streamlit

In another terminal:

```powershell
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## 5. Docker

Create `.env`, then:

```powershell
docker compose up --build
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

Streamlit:

```text
http://localhost:8501
```

## 6. Evaluation

Copy `evaluation_dataset.example.json` to `evaluation_dataset.json`, add questions and expected source URLs, then:

```powershell
python evaluation.py
```

## 7. Tests

```powershell
pytest -q
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health/status |
| POST | `/crawl` | Crawl and ingest website |
| POST | `/upload` | Upload PDF/DOCX/CSV |
| POST | `/ask` | Ask RAG question |
| POST | `/reset` | Reset knowledge base |
| POST | `/clear-memory` | Clear conversation memory |
| GET | `/sources` | List indexed sources |

## Persistent data

Docker volumes preserve:

- FAISS index
- uploaded files
- application logs

## Notes

The hybrid re-ranking stage combines semantic similarity and BM25 keyword scores. It is intentionally implemented without a separate cross-encoder so the full project can run with the Gemini + FAISS stack. A cross-encoder can be added later as an improvement phase.

The evaluation script uses retrieval/source metrics and latency without making additional LLM judge calls.
