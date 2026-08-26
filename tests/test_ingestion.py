from langchain_core.documents import Document
from ingestion import create_chunks

def test_recursive_chunking():
    docs = [Document(
        page_content="A " * 1000,
        metadata={"source": "test", "type": "website"}
    )]
    chunks = create_chunks(docs)
    assert chunks
    assert all("chunk_id" in c.metadata for c in chunks)
