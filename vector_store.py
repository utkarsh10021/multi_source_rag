import json
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import (
    FAISS_DIR,
    CHUNKS_FILE,
    SOURCES_FILE,
    FAISS_INDEX_NAME,
)


class PersistentHybridStore:
    """
    Persistent hybrid RAG store.

    Features:
    - FAISS semantic retrieval
    - BM25 keyword retrieval
    - Hybrid scoring
    - Multi-source retrieval
    - Source-aware ranking
    - Persistent FAISS index
    - Resume + website source balancing
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self.chunks = []
        self.bm25 = None

    # =========================================================
    # PERSISTENCE
    # =========================================================

    @property
    def exists(self):
        return (
            (FAISS_DIR / f"{FAISS_INDEX_NAME}.faiss").exists()
            and (FAISS_DIR / f"{FAISS_INDEX_NAME}.pkl").exists()
            and CHUNKS_FILE.exists()
        )

    def build(self, chunks: list[Document]):
        if not chunks:
            raise ValueError("No chunks supplied.")

        self.chunks = chunks

        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings,
        )

        self._build_bm25()
        self.save()

        return self.vectorstore

    def load(self):
        if not self.exists:
            return False

        self.vectorstore = FAISS.load_local(
            str(FAISS_DIR),
            self.embeddings,
            index_name=FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

        data = json.loads(
            CHUNKS_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.chunks = [
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in data
        ]

        self._build_bm25()

        return True

    def save(self):
        FAISS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.vectorstore is None:
            raise ValueError(
                "Cannot save because vectorstore is empty."
            )

        self.vectorstore.save_local(
            str(FAISS_DIR),
            index_name=FAISS_INDEX_NAME,
        )

        CHUNKS_FILE.write_text(
            json.dumps(
                [
                    {
                        "page_content": d.page_content,
                        "metadata": d.metadata,
                    }
                    for d in self.chunks
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        source_map = {}

        for d in self.chunks:
            source = d.metadata.get(
                "source",
                "unknown",
            )

            source_map[source] = {
                "title": d.metadata.get(
                    "title",
                    "Unknown",
                ),
                "type": d.metadata.get(
                    "type",
                    "unknown",
                ),
            }

        SOURCES_FILE.write_text(
            json.dumps(
                source_map,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def reset(self):
        self.vectorstore = None
        self.chunks = []
        self.bm25 = None

        for path in (
            FAISS_DIR / f"{FAISS_INDEX_NAME}.faiss",
            FAISS_DIR / f"{FAISS_INDEX_NAME}.pkl",
            CHUNKS_FILE,
            SOURCES_FILE,
        ):
            if path.exists():
                path.unlink()

    # =========================================================
    # BM25
    # =========================================================

    def _build_bm25(self):
        tokenized = [
            self._tokenize(d.page_content)
            for d in self.chunks
        ]

        self.bm25 = (
            BM25Okapi(tokenized)
            if tokenized
            else None
        )

    @staticmethod
    def _tokenize(text):
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    # =========================================================
    # METADATA HELPERS
    # =========================================================

    @staticmethod
    def _source_key(doc):
        return str(
            doc.metadata.get(
                "source",
                "unknown",
            )
        ).strip()

    @staticmethod
    def _source_type(doc):
        return str(
            doc.metadata.get(
                "type",
                "unknown",
            )
        ).lower()

    @staticmethod
    def _source_title(doc):
        return str(
            doc.metadata.get(
                "title",
                "",
            )
        ).lower()

    def _is_resume(self, doc):
        source = self._source_key(doc).lower()
        title = self._source_title(doc)
        source_type = self._source_type(doc)

        return (
            source.endswith(".pdf")
            or source.endswith(".docx")
            or "resume" in source
            or "resume" in title
            or source_type == "pdf"
        )

    def _is_website(self, doc):
        source = self._source_key(doc).lower()
        source_type = self._source_type(doc)

        return (
            source.startswith("http://")
            or source.startswith("https://")
            or source_type in {
                "website",
                "web",
            }
        )

    # =========================================================
    # QUERY DETECTION
    # =========================================================

    @staticmethod
    def _is_multi_source_query(query):
        q = query.lower()

        phrases = [
            "resume and",
            "resume with",
            "resume plus",
            "resume +",

            "my resume and",
            "my resume with",
            "my resume plus",
            "my resume +",

            "resume and the",
            "resume with the",

            "both sources",
            "both documents",
            "both source",
            "both document",

            "multiple sources",
            "multiple documents",

            "across the sources",
            "across sources",

            "from my resume and",
            "from resume and",

            "based on my resume",
            "using my resume",

            "resume as well as",
            "resume along with",

            "my skills and fastapi",
            "my experience and fastapi",

            "resume and fastapi",
            "resume with fastapi",

            "resume and the fastapi",
            "resume with the fastapi",
        ]

        return any(
            phrase in q
            for phrase in phrases
        )

    # =========================================================
    # MAIN SEARCH
    # =========================================================

    def hybrid_search(
        self,
        query,
        top_k=20,
        final_k=5,
        original_query=None,
    ):
        if not self.vectorstore or not self.chunks:
            return []

        

        source_query = original_query or query

        if self._is_multi_source_query(source_query):

            return self._multi_source_search(
                query=query,
                top_k=top_k,
                final_k=final_k,
            )

        return self._global_hybrid_search(
            query=query,
            top_k=top_k,
            final_k=final_k,
        )

    # =========================================================
    # GLOBAL HYBRID SEARCH
    # =========================================================

    def _global_hybrid_search(
        self,
        query,
        top_k,
        final_k,
    ):
        candidate_k = min(
            max(top_k, final_k * 5),
            len(self.chunks),
        )

        semantic_results = (
            self.vectorstore.similarity_search_with_score(
                query,
                k=candidate_k,
            )
        )

        keyword_scores = {}

        if self.bm25:

            scores = self.bm25.get_scores(
                self._tokenize(query)
            )

            ranked = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )

            for idx, score in ranked[:candidate_k]:
                keyword_scores[idx] = float(score)

        candidates = {}

        for doc, distance in semantic_results:

            idx = self._find_chunk_index(doc)

            if idx is None:
                continue

            candidates[idx] = {
                "doc": doc,
                "distance": float(distance),
            }

        for idx in keyword_scores:

            if idx not in candidates:

                candidates[idx] = {
                    "doc": self.chunks[idx],
                    "distance": float("inf"),
                }

        return self._rank_candidates(
            candidates,
            keyword_scores,
            final_k,
        )

    # =========================================================
    # MULTI-SOURCE SEARCH
    # =========================================================

    def _multi_source_search(
        self,
        query,
        top_k,
        final_k,
    ):
        """
        Multi-source retrieval.

        Resume and website documents are searched
        independently.

        This prevents a large website corpus from
        dominating a smaller resume corpus.
        """

        resume_indices = []
        website_indices = []
        other_indices = []

        for idx, doc in enumerate(self.chunks):

            if self._is_resume(doc):
                resume_indices.append(idx)

            elif self._is_website(doc):
                website_indices.append(idx)

            else:
                other_indices.append(idx)

        # -----------------------------------------------------
        # Search each source group independently
        # -----------------------------------------------------

        resume_results = self._search_subset(
            query=query,
            indices=resume_indices,
            top_k=top_k,
        )

        website_results = self._search_subset(
            query=query,
            indices=website_indices,
            top_k=top_k,
        )

        other_results = self._search_subset(
            query=query,
            indices=other_indices,
            top_k=top_k,
        )

        # -----------------------------------------------------
        # Combine
        # -----------------------------------------------------

        combined = (
            resume_results
            + website_results
            + other_results
        )

        if not combined:
            return []

        combined.sort(
            key=lambda x: x[3],
            reverse=True,
        )

        # -----------------------------------------------------
        # Guarantee source diversity
        # -----------------------------------------------------

        selected = []
        selected_keys = set()

        def add_best_from_group(results):
            if not results:
                return

            best = max(
                results,
                key=lambda x: x[3],
            )

            doc = best[0]

            key = (
                self._source_key(doc),
                doc.page_content,
            )

            if key not in selected_keys:
                selected.append(best)
                selected_keys.add(key)

        # Always try resume first
        add_best_from_group(
            resume_results
        )

        # Always try website second
        if len(selected) < final_k:
            add_best_from_group(
                website_results
            )

        # Other sources if present
        if len(selected) < final_k:
            add_best_from_group(
                other_results
            )

        # -----------------------------------------------------
        # Fill remaining slots by relevance
        # -----------------------------------------------------

        for item in combined:

            if len(selected) >= final_k:
                break

            doc = item[0]

            key = (
                self._source_key(doc),
                doc.page_content,
            )

            if key in selected_keys:
                continue

            selected.append(item)
            selected_keys.add(key)

        # -----------------------------------------------------
        # Final ranking
        # -----------------------------------------------------

        selected.sort(
            key=lambda x: x[3],
            reverse=True,
        )

        return selected[:final_k]

    # =========================================================
    # SOURCE-AWARE SUBSET SEARCH
    # =========================================================

    def _search_subset(
        self,
        query,
        indices,
        top_k,
    ):
        """
        Search only within the supplied chunk indices.

        IMPORTANT:
        FAISS is searched with a large candidate pool,
        then filtered to the requested source group.

        BM25 is searched directly within the source group.
        """

        if not indices:
            return []

        allowed = set(indices)

        # -----------------------------------------------------
        # FAISS semantic retrieval
        # -----------------------------------------------------

        candidate_k = min(
            len(self.chunks),
            max(
                top_k * 10,
                len(indices),
                50,
            ),
        )

        semantic_results = (
            self.vectorstore.similarity_search_with_score(
                query,
                k=candidate_k,
            )
        )

        semantic_candidates = {}

        for doc, distance in semantic_results:

            idx = self._find_chunk_index(doc)

            if idx is None:
                continue

            if idx not in allowed:
                continue

            semantic_candidates[idx] = {
                "doc": doc,
                "distance": float(distance),
            }

        # -----------------------------------------------------
        # BM25 source-specific retrieval
        # -----------------------------------------------------

        keyword_scores = {}

        if self.bm25:

            scores = self.bm25.get_scores(
                self._tokenize(query)
            )

            ranked = sorted(
                (
                    (
                        idx,
                        float(scores[idx]),
                    )
                    for idx in indices
                ),
                key=lambda x: x[1],
                reverse=True,
            )

            for idx, score in ranked[:top_k * 3]:

                keyword_scores[idx] = score

        # -----------------------------------------------------
        # Add BM25-only candidates
        # -----------------------------------------------------

        candidates = dict(
            semantic_candidates
        )

        for idx in keyword_scores:

            if idx not in candidates:

                candidates[idx] = {
                    "doc": self.chunks[idx],
                    "distance": float("inf"),
                }

        # -----------------------------------------------------
        # Rank
        # -----------------------------------------------------

        return self._rank_candidates(
            candidates,
            keyword_scores,
            top_k,
        )

    # =========================================================
    # RANK CANDIDATES
    # =========================================================

    def _rank_candidates(
        self,
        candidates,
        keyword_scores,
        final_k,
    ):
        if not candidates:
            return []

        # -----------------------------------------------------
        # Semantic normalization
        # -----------------------------------------------------

        semantic_distances = [
            item["distance"]
            for item in candidates.values()
            if item["distance"] != float("inf")
        ]

        if semantic_distances:

            min_distance = min(
                semantic_distances
            )

            max_distance = max(
                semantic_distances
            )

        else:

            min_distance = 0.0
            max_distance = 1.0

        # -----------------------------------------------------
        # BM25 normalization
        # -----------------------------------------------------

        bm25_values = list(
            keyword_scores.values()
        )

        if bm25_values:

            min_bm25 = min(
                bm25_values
            )

            max_bm25 = max(
                bm25_values
            )

        else:

            min_bm25 = 0.0
            max_bm25 = 1.0

        reranked = []

        # -----------------------------------------------------
        # Calculate hybrid score
        # -----------------------------------------------------

        for idx, item in candidates.items():

            doc = item["doc"]
            distance = item["distance"]

            # -------------------------------------------------
            # Semantic score
            # -------------------------------------------------

            if distance == float("inf"):

                semantic_score = 0.0

            elif max_distance == min_distance:

                semantic_score = 1.0

            else:

                semantic_score = (
                    max_distance - distance
                ) / (
                    max_distance - min_distance
                )

            # -------------------------------------------------
            # BM25 score
            # -------------------------------------------------

            raw_keyword = keyword_scores.get(
                idx,
                0.0,
            )

            if max_bm25 == min_bm25:

                keyword_score = (
                    1.0
                    if raw_keyword > 0
                    else 0.0
                )

            else:

                keyword_score = (
                    raw_keyword - min_bm25
                ) / (
                    max_bm25 - min_bm25
                )

            # -------------------------------------------------
            # Hybrid score
            # -------------------------------------------------

            final_score = (
                0.70 * semantic_score
                + 0.30 * keyword_score
            )

            reranked.append(
                (
                    doc,
                    distance,
                    keyword_score,
                    final_score,
                )
            )

        reranked.sort(
            key=lambda x: x[3],
            reverse=True,
        )

        return reranked[:final_k]

    # =========================================================
    # FIND CHUNK
    # =========================================================

    def _find_chunk_index(self, doc):

        for i, candidate in enumerate(
            self.chunks
        ):

            if (
                candidate.page_content
                == doc.page_content
                and candidate.metadata
                == doc.metadata
            ):
                return i

        return None