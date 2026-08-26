"""
Retrieval-focused RAG evaluation.

This version does NOT call Gemini for query rewriting or answer generation.

It directly evaluates the existing FAISS/BM25 hybrid store.
"""

import json
import time
from pathlib import Path

from rag_service import RAGService


def normalize_source(source):
    """Normalize source names for comparison."""

    source = str(source).strip().lower()
    source = source.replace("\\", "/")

    # Convert Docker upload path to filename
    if "/uploads/" in source:
        source = source.split("/uploads/", 1)[1]

    return source


def source_matches(expected, retrieved):
    """Flexible source comparison."""

    expected = normalize_source(expected)
    retrieved = normalize_source(retrieved)

    return (
        expected == retrieved
        or expected in retrieved
        or retrieved in expected
    )

def get_source_from_document(doc):
    """Extract source from hybrid-search result."""

    # hybrid_search returns:
    # (Document, vector_score, bm25_score, final_score)

    if isinstance(doc, tuple):
        doc = doc[0]

    if hasattr(doc, "metadata"):
        return doc.metadata.get("source", "")

    if isinstance(doc, dict):
        return doc.get("source", "")

    return getattr(doc, "source", "")

def retrieve_documents(service, question):
    """
    Use the existing vector store directly.

    The project store contains the hybrid retrieval implementation.
    """

    store = service.store

    # Try the project's hybrid search method.
    if hasattr(store, "hybrid_search"):
        return store.hybrid_search(
            question,
            top_k=10
        )

    # Try common search method.
    if hasattr(store, "search"):
        return store.search(
            question,
            top_k=10
        )

    # Try similarity search.
    if hasattr(store, "similarity_search"):
        return store.similarity_search(
            question,
            k=10
        )

    raise RuntimeError(
        "Could not find a retrieval method in the project's "
        "vector store."
    )


def evaluate(dataset_path="evaluation_dataset.json"):

    dataset = json.loads(
        Path(dataset_path).read_text(
            encoding="utf-8"
        )
    )

    service = RAGService()

    results = []

    print()
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)
    print("Gemini answer generation: DISABLED")
    print("Gemini query rewriting: DISABLED")
    print("Evaluating retrieval only")
    print("=" * 70)
    print()

    for number, item in enumerate(dataset, start=1):

        question = item["question"]

        expected_sources = item.get(
            "expected_sources",
            []
        )

        print(
            f"Question {number}/{len(dataset)}"
        )

        print(
            f"Q: {question}"
        )

        started = time.perf_counter()

        try:

            documents = retrieve_documents(
                service,
                question
            )

            latency = (
                time.perf_counter()
                - started
            )

        except Exception as exc:

            print(
                f"Retrieval error: {exc}"
            )

            results.append(
                {
                    "question": question,
                    "retrieval_recall": 0.0,
                    "source_precision": 0.0,
                    "latency_seconds": 0.0,
                    "error": str(exc),
                }
            )

            continue

        retrieved_sources = []

        for document in documents:

            source = get_source_from_document(
                document
            )

            if source:
                retrieved_sources.append(
                    source
                )

        # Remove duplicate sources
        unique_sources = list(
            dict.fromkeys(
                retrieved_sources
            )
        )

        # --------------------------------
        # Recall
        # --------------------------------

        matched_expected = 0

        for expected in expected_sources:

            if any(
                source_matches(
                    expected,
                    retrieved
                )
                for retrieved in unique_sources
            ):
                matched_expected += 1

        recall = (
            matched_expected
            / len(expected_sources)
            if expected_sources
            else 0.0
        )

        # --------------------------------
        # Precision
        # --------------------------------

        matching_retrieved = 0

        for retrieved in unique_sources:

            if any(
                source_matches(
                    expected,
                    retrieved
                )
                for expected in expected_sources
            ):
                matching_retrieved += 1

        precision = (
            matching_retrieved
            / len(unique_sources)
            if unique_sources
            else 0.0
        )

        result = {
            "question": question,
            "retrieval_recall": round(
                recall,
                4
            ),
            "source_precision": round(
                precision,
                4
            ),
            "latency_seconds": round(
                latency,
                4
            ),
            "retrieved_sources": unique_sources,
        }

        results.append(result)

        print(
            f"Retrieved sources: "
            f"{len(unique_sources)}"
        )

        print(
            f"Recall: {recall:.2f}"
        )

        print(
            f"Precision: {precision:.2f}"
        )

        print(
            f"Latency: {latency:.4f}s"
        )

        print("-" * 70)

    # --------------------------------
    # Final metrics
    # --------------------------------

    successful_results = [
        r for r in results
        if "error" not in r
    ]

    if not successful_results:

        print(
            "No successful evaluation results."
        )

        return

    average_recall = (
        sum(
            r["retrieval_recall"]
            for r in successful_results
        )
        / len(successful_results)
    )

    average_precision = (
        sum(
            r["source_precision"]
            for r in successful_results
        )
        / len(successful_results)
    )

    average_latency = (
        sum(
            r["latency_seconds"]
            for r in successful_results
        )
        / len(successful_results)
    )

    print()
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Average retrieval recall: "
        f"{average_recall:.4f}"
    )

    print(
        f"Average source precision: "
        f"{average_precision:.4f}"
    )

    print(
        f"Average retrieval latency: "
        f"{average_latency:.4f}s"
    )

    print("=" * 70)

    output = {
        "average_retrieval_recall":
            round(
                average_recall,
                4
            ),

        "average_source_precision":
            round(
                average_precision,
                4
            ),

        "average_retrieval_latency_seconds":
            round(
                average_latency,
                4
            ),

        "questions": results,
    }

    Path(
        "evaluation_results.json"
    ).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print(
        "Results saved to "
        "evaluation_results.json"
    )


if __name__ == "__main__":
    evaluate()