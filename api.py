import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from config import LLM_BACKEND, DOCS_DIR, NO_ANSWER_THRESHOLD
from ingestion import ingest_documents, load_existing_vectorstore
from retriever import PolicyRetriever
from chain import PolicyQAChain
from schemas import (
    AskRequest,
    AskResponse,
    IngestResponse,
    DocumentsResponse,
    HealthResponse,
    DebugChunk,
    RetrieveDebugResponse,
)
from logger.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── shared app state (set during lifespan, read by request handlers) ──
state: dict = {
    "vectorstore": None,
    "retriever": None,
    "chain": None,
}


def _build_pipeline(force_reingest: bool = False):
    """
    Load a persisted vectorstore if one exists, otherwise (or if forced)
    run full ingestion over DOCS_DIR. Rebuilds retriever + chain on top
    of whichever vectorstore is active.
    """
    vectorstore = None if force_reingest else load_existing_vectorstore()

    if vectorstore is None:
        logger.info("No usable persisted vectorstore found — running ingestion.")
        vectorstore = ingest_documents()
    else:
        logger.info("Loaded existing vectorstore from disk — skipping ingestion.")

    retriever = PolicyRetriever(vectorstore=vectorstore)
    chain = PolicyQAChain(retriever=retriever)

    state["vectorstore"] = vectorstore
    state["retriever"] = retriever
    state["chain"] = chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Policy Q&A API...")
    try:
        _build_pipeline(force_reingest=False)
        logger.info("Startup complete. System ready.")
    except FileNotFoundError as e:
        # No PDFs in docs/ yet — let the app come up anyway so /ingest
        # and /health remain usable; /ask will fail clearly until then.
        logger.warning("Startup ingestion skipped: %s", e)
    yield
    logger.info("Shutting down Policy Q&A API.")


app = FastAPI(
    title="Policy Q&A Bot",
    description="RAG-powered Q&A over insurance policy documents, with clause-level citations.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_chain() -> PolicyQAChain:
    if state["chain"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No policy documents are indexed yet. "
                f"Add PDFs to '{DOCS_DIR}/' and call POST /ingest."
            ),
        )
    return state["chain"]


def _collection_summary() -> tuple[list[str], int]:
    """Return (unique doc names, total chunk count) from the active vectorstore."""
    vectorstore = state["vectorstore"]
    if vectorstore is None:
        return [], 0

    raw = vectorstore._collection.get(include=["metadatas"])
    metadatas = raw.get("metadatas", []) or []
    doc_names = sorted({m.get("doc_name", "Unknown") for m in metadatas})
    return doc_names, len(metadatas)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    chain = _get_chain()
    result = chain.answer(payload.question)

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        grounded=result["relevant_chunks"] > 0,
        citations=result["citations"],
        chunks_retrieved=result["chunks_retrieved"],
        relevant_chunks=result["relevant_chunks"],
        backend=result["backend"],
    )


@app.post("/retrieve", response_model=RetrieveDebugResponse)
def retrieve(payload: AskRequest):
    """
    Debug endpoint: shows the raw retrieval result for a question — every
    chunk's actual cosine score and whether it passed NO_ANSWER_THRESHOLD —
    without going through the LLM at all. Use this to see exactly why a
    question came back grounded/ungrounded, and to calibrate the threshold
    against real documents instead of guessing.
    """
    if state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No policy documents are indexed yet. "
                f"Add PDFs to '{DOCS_DIR}/' and call POST /ingest."
            ),
        )

    chunks = state["retriever"].retrieve(payload.question)

    debug_chunks = [
        DebugChunk(
            score=c["score"],
            cosine_score=c["cosine_score"],
            bm25_score=c["bm25_score"],
            relevant=c["relevant"],
            doc_name=c["metadata"].get("doc_name", ""),
            section=c["metadata"].get("section", ""),
            clause_number=c["metadata"].get("clause_number", ""),
            page=str(c["metadata"].get("page", "")),
            content=c["content"],
        )
        for c in chunks
    ]

    return RetrieveDebugResponse(
        question=payload.question,
        threshold=NO_ANSWER_THRESHOLD,
        chunks=debug_chunks,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        _build_pipeline(force_reingest=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    doc_names, _ = _collection_summary()
    return IngestResponse(
        status="ok",
        message="Re-ingestion complete.",
        documents_indexed=doc_names,
    )


@app.get("/documents", response_model=DocumentsResponse)
def documents():
    doc_names, total_chunks = _collection_summary()
    return DocumentsResponse(documents=doc_names, total_chunks=total_chunks)


@app.get("/health", response_model=HealthResponse)
def health():
    doc_names, _ = _collection_summary()
    return HealthResponse(
        status="ok",
        vectorstore_loaded=state["vectorstore"] is not None,
        llm_backend=LLM_BACKEND,
        indexed_documents=len(doc_names),
    )