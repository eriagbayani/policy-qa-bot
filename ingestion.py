import os
import re
import pypdf
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tiktoken
from config import (
    DOCS_DIR,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    HEADING_PATTERNS,
)

# tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# heading detection
def is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line):
            return True
    return False

def extract_clause_number(heading: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)", heading.strip())
    if match:
        return match.group(1)
    match = re.match(r"^(SECTION|PART)\s+([A-Z0-9]+)", heading.strip())
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return ""

# pdf loader
def load_pdf(filepath: str) -> list[dict]:
    pages = []
    reader = pypdf.PdfReader(filepath)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "text": text,
            "page": page_num
        })
    return pages

# section aware chunker
def section_aware_chunk(pages: list[dict], doc_name: str) -> list[Document]:
    chunks = []
    current_section = "General"
    current_clause = ""
    current_heading_path = []
    current_text = ""
    current_page = 1
    chunk_start_page = 1

    def flush_chunk():
        nonlocal current_text
        if current_text.strip():
            heading_path = " > ".join(current_heading_path) if current_heading_path else current_section
            chunks.append(Document(
                page_content=current_text.strip(),
                metadata={
                    "doc_name": doc_name,
                    "section": current_section,
                    "clause_number": current_clause,
                    "heading_path": heading_path,
                    "page": chunk_start_page,
                }
            ))
        current_text = ""

    for page_data in pages:
        lines = page_data["text"].split("\n")
        current_page = page_data["page"]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if is_heading(stripped):
                # Flush previous chunk before starting new section
                if count_tokens(current_text) >= CHUNK_SIZE_TOKENS:
                    flush_chunk()

                # Update section tracking
                clause = extract_clause_number(stripped)
                if clause:
                    current_clause = clause
                current_section = stripped

                # Update heading path hierarchy
                if re.match(r"^\d+\.\d+\.\d+", stripped):
                    if len(current_heading_path) >= 3:
                        current_heading_path = current_heading_path[:2] + [stripped]
                    else:
                        current_heading_path.append(stripped)
                elif re.match(r"^\d+\.\d+", stripped):
                    if len(current_heading_path) >= 2:
                        current_heading_path = current_heading_path[:1] + [stripped]
                    else:
                        current_heading_path.append(stripped)
                else:
                    current_heading_path = [stripped]

                chunk_start_page = current_page
                current_text += f"\n{stripped}\n"

            else:
                current_text += f"{stripped} "

                # Flush if over token limit with overlap
                if count_tokens(current_text) >= CHUNK_SIZE_TOKENS:
                    flush_chunk()
                    # Keep overlap — last N tokens
                    words = current_text.split()
                    overlap_words = words[-CHUNK_OVERLAP_TOKENS:] if len(words) > CHUNK_OVERLAP_TOKENS else words
                    current_text = " ".join(overlap_words)
                    chunk_start_page = current_page

    # Flush remaining
    flush_chunk()
    return chunks

# ingest
def ingest_documents() -> Chroma:
    docs_path = Path(DOCS_DIR)
    all_chunks = []

    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DOCS_DIR}/")

    print(f"Found {len(pdf_files)} PDF file(s)")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        pages = load_pdf(str(pdf_file))
        chunks = section_aware_chunk(pages, pdf_file.name)
        print(f"  {len(chunks)} chunks created")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Creating embeddings and storing in ChromaDB...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION
    )

    print("Ingestion complete.")
    return vectorstore

if __name__ == "__main__":
    ingest_documents()