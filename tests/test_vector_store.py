from rag_service import RAGService

def test_persistent_store_exists():
    service = RAGService()

    assert service.store.exists is True


def test_faiss_vectorstore_exists():
    service = RAGService()

    assert service.store.vectorstore is not None


def test_bm25_exists():
    service = RAGService()

    assert service.store.bm25 is not None


def test_chunks_exist():
    service = RAGService()

    assert isinstance(service.store.chunks, list)
    assert len(service.store.chunks) > 0