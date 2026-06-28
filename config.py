import os
from dotenv import load_dotenv

load_dotenv()

# LLm backend
# Options: "mock", "groq", choose either two
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector store
CHROMA_COLLECTION = "policy_docs"

# chunking
CHUNK_SIZE_TOKENS = 600       # target tokens per chunk
CHUNK_OVERLAP_TOKENS = 100    # overlap between chunks
TOP_K_RESULTS = 8             # number of chunks to retrieve

# docs
DOCS_DIR = "docs"

# heading patterns
# Regex patterns to detect section headings in policy docs
HEADING_PATTERNS = [
    r"^SECTION\s+[A-Z0-9]+",           # SECTION A, SECTION 1
    r"^PART\s+[A-Z0-9]+",              # PART 1, PART A
    r"^\d+\.\s+[A-Z]",                 # 1. DEFINITIONS
    r"^\d+\.\d+\s+[A-Z]",             # 1.1 General
    r"^\d+\.\d+\.\d+\s+",             # 1.1.1 Specific
    r"^[A-Z][A-Z\s]{4,}$",            # ALL CAPS HEADINGS
    r"^Schedule\s+\d+",               # Schedule 1
    r"^Endorsement\s+\d+",            # Endorsement 1
]

# format of the citation
CITATION_FORMAT = "{doc_name} §{clause_number} ({section}), p.{page}"

# threshold of no answer
# Minimum similarity score to consider a chunk relevant
NO_ANSWER_THRESHOLD = 0.1