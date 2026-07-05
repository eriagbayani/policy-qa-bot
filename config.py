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
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# chunking
CHUNK_SIZE_TOKENS = 600       # target tokens per chunk
CHUNK_OVERLAP_TOKENS = 100    # overlap between chunks
TOP_K_RESULTS = 8             # number of chunks returned to the LLM

# Hybrid retrieval: pure cosine similarity on generic sentence embeddings
# (all-MiniLM-L6-v2) tends to score any two insurance clauses similarly
# ("cover", "policy", "claim", "damage" appear everywhere), burying chunks
# that contain the actual query terms beneath ones that just share generic
# vocabulary. BM25 keyword matching is combined in to fix that: a chunk
# containing the literal words "wear" and "tear" gets rewarded even when
# its embedding similarity isn't top-ranked.
CANDIDATE_POOL_SIZE = 25       # chunks pulled by vector search before rerank
HYBRID_ALPHA = 0.5             # weight on cosine vs BM25 (0.5 = equal)

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

# Patterns above that match on a leading number are prone to false-positives
# — plenty of ordinary clause sentences start with "9. We will not cover...".
# These patterns get an extra length/shape check in ingestion.is_heading()
# so only short, title-like lines count as real headings.
NUMBERED_HEADING_PATTERNS = [
    r"^\d+\.\s+[A-Z]",
    r"^\d+\.\d+\s+[A-Z]",
    r"^\d+\.\d+\.\d+\s+",
]
MAX_HEADING_WORDS = 8          # headings longer than this are body text
MAX_HEADING_WORDS_WITH_PERIOD = 4  # a numbered line ending in "." is almost
                                    # always a full sentence unless very short

# format of the citation
CITATION_FORMAT = "{doc_name} §{clause_number} ({section}), p.{page}"

# threshold of no answer
# Minimum combined hybrid score (0-1) to consider a chunk relevant. The
# combined score is HYBRID_ALPHA * cosine + (1-HYBRID_ALPHA) * normalized
# BM25 — see retriever.py.
#
# 0.5 is based on one real measurement, not a large calibration set:
# on a genuine "wear and tear" query against real policy PDFs, combined
# scores ranged 0.38-0.81, with the correct clause landing at 0.79 and the
# weakest, least-relevant chunks down at 0.38-0.41. 0.5 sits in between —
# keeps clearly-relevant chunks, drops the weakest tail. The old 0.35
# passed every retrieved chunk (8/8) regardless of relevance, making the
# `grounded`/`relevant_chunks` fields meaningless. Recalibrate properly
# once you have more than one measured query (see test_set.py).
NO_ANSWER_THRESHOLD = 0.5