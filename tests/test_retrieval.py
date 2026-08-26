from rag_service import RAGService


def test_rag_service_initializes():
    service = RAGService()

    assert service is not None
    assert service.store is not None


def test_hybrid_search_returns_results():
    service = RAGService()

    results = service.store.hybrid_search(
        "What are the technical skills mentioned in my resume?",
        top_k=5
    )

    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_search_returns_documents():
    service = RAGService()

    results = service.store.hybrid_search(
        "What is FastAPI used for?",
        top_k=5
    )

    assert len(results) > 0

    document = results[0][0]

    assert hasattr(document, "page_content")
    assert hasattr(document, "metadata")


def test_retrieved_source_exists():
    service = RAGService()

    results = service.store.hybrid_search(
        "What are the technical skills mentioned in my resume?",
        top_k=5
    )

    sources = []

    for result in results:
        document = result[0]
        source = document.metadata.get("source")

        if source:
            sources.append(source)

    assert len(sources) > 0