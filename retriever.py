from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
    TOP_K_RESULTS,
    NO_ANSWER_THRESHOLD,
)

# retriever
class PolicyRetriever:
    def __init__(self, vectorstore: Chroma = None):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            # Load existing vectorstore from disk
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=self.embeddings
            )

    def retrieve(self, query: str) -> list[dict]:
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=TOP_K_RESULTS
        )

        chunks = []
        for doc, score in results:
            chunks.append({
                "content": doc.page_content,
                "score": round(score, 4),
                "metadata": doc.metadata,
                "relevant": score >= NO_ANSWER_THRESHOLD
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