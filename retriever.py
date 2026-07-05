import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from config import (
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    TOP_K_RESULTS,
    NO_ANSWER_THRESHOLD,
    CANDIDATE_POOL_SIZE,
    HYBRID_ALPHA,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of scores into [0, 1]. Flat input -> all 0s."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# retriever
class PolicyRetriever:
    def __init__(self, vectorstore: Chroma = None):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        if vectorstore:
            self.vectorstore = vectorstore
        else:
            # Load existing (persisted) vectorstore from disk
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_configuration={"hnsw": {"space": "cosine"}},
            )

        self._build_bm25_index()

    def _build_bm25_index(self):
        """
        Build an in-memory BM25 index over every chunk currently in the
        collection. Cheap for the corpus sizes this project targets
        (hundreds of chunks, not millions) — rebuilt whenever a new
        PolicyRetriever is constructed, i.e. on startup and after /ingest.
        """
        raw = self.vectorstore._collection.get(include=["documents", "metadatas"])
        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []

        self._bm25_corpus = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(documents, metadatas)
        ]
        tokenized = [_tokenize(item["content"]) for item in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        # content -> bm25 corpus index, for matching against vector results
        self._content_to_bm25_idx = {
            item["content"]: i for i, item in enumerate(self._bm25_corpus)
        }

    def retrieve(self, query: str) -> list[dict]:
        # Pull a larger candidate pool by cosine similarity, then rerank
        # with BM25 keyword overlap before cutting down to TOP_K_RESULTS.
        # Widening the initial pool matters: the correct chunk can rank
        # below the final cutoff on cosine alone (as happened with the
        # "wear and tear" clause ranking 5th-9th on pure similarity) but
        # still surface once BM25 boosts it back up.
        pool_size = max(TOP_K_RESULTS, CANDIDATE_POOL_SIZE)
        vector_results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=pool_size,
        )

        if not vector_results:
            return []

        cosine_scores = [score for _, score in vector_results]

        bm25_scores = [0.0] * len(vector_results)
        if self._bm25 is not None:
            query_tokens = _tokenize(query)
            all_bm25_scores = self._bm25.get_scores(query_tokens)
            for i, (doc, _) in enumerate(vector_results):
                idx = self._content_to_bm25_idx.get(doc.page_content)
                if idx is not None:
                    bm25_scores[i] = all_bm25_scores[idx]

        norm_cosine = _normalize(cosine_scores)
        norm_bm25 = _normalize(bm25_scores)

        combined = [
            HYBRID_ALPHA * nc + (1 - HYBRID_ALPHA) * nb
            for nc, nb in zip(norm_cosine, norm_bm25)
        ]

        ranked = sorted(
            zip(vector_results, cosine_scores, bm25_scores, combined),
            key=lambda x: x[3],
            reverse=True,
        )[:TOP_K_RESULTS]

        chunks = []
        for (doc, _), cosine_score, bm25_score, combined_score in ranked:
            chunks.append({
                "content": doc.page_content,
                "score": round(combined_score, 4),
                "cosine_score": round(cosine_score, 4),
                "bm25_score": round(bm25_score, 4),
                "metadata": doc.metadata,
                "relevant": combined_score >= NO_ANSWER_THRESHOLD
            })

        return chunks

    def format_citations(self, chunks: list[dict]) -> str:
        citations = []
        seen = set()

        for chunk in chunks:
            if not chunk["relevant"]:
                continue

            meta = chunk["metadata"]
            doc_name = meta.get("doc_name", "Unknown")
            clause = meta.get("clause_number", "")
            section = meta.get("section", "")
            page = meta.get("page", "")

            # Clean up section name — remove generic "Section" placeholder
            if section.lower() == "section" or section.lower() == "general":
                section = ""

            # Build citation string — only include parts that have real values
            clause_part = f" §{clause}" if clause and clause != "" else ""
            section_part = f" ({section})" if section and len(section) > 8 else ""
            page_part = f", p.{page}" if page else ""

            citation = f"{doc_name}{clause_part}{section_part}{page_part}"

            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

        if not citations:
            return ""

        return "Sources: " + "; ".join(citations)

    def has_relevant_results(self, chunks: list[dict]) -> bool:
        return any(chunk["relevant"] for chunk in chunks)

    def format_context(self, chunks: list[dict]) -> str:
        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            if not chunk["relevant"]:
                continue

            meta = chunk["metadata"]
            header = f"[Chunk {i} | {meta.get('doc_name', '')} | " \
                     f"§{meta.get('clause_number', '')} | " \
                     f"p.{meta.get('page', '')}]"

            context_parts.append(f"{header}\n{chunk['content']}")

        return "\n\n".join(context_parts)