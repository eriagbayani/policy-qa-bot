from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the policy documents")


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: str
    chunks_retrieved: int
    relevant_chunks: int
    backend: str


class IngestResponse(BaseModel):
    status: str
    message: str
    documents_indexed: list[str] = []


class DocumentsResponse(BaseModel):
    documents: list[str]
    total_chunks: int


class DebugChunk(BaseModel):
    score: float
    cosine_score: float
    bm25_score: float
    relevant: bool
    doc_name: str
    section: str
    clause_number: str
    page: str
    content: str


class RetrieveDebugResponse(BaseModel):
    question: str
    threshold: float
    chunks: list[DebugChunk]


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    llm_backend: str
    indexed_documents: int