# Policy Q&A Bot

A RAG-powered question answering system for insurance policy documents, served as a FastAPI service. Ask questions in plain English and get grounded answers with clause-level citations — strictly from the policy wording.

## What It Does

Upload insurance policy PDFs and ask questions like:

- "Is wear and tear covered under the standard policy?"
- "What is the waiting period for accidental damage?"
- "How does the deductible apply to water damage?"
- "What definitions apply to Insured Person?"

Every answer includes citations in the format:
```
Sources: QBE_Policy.pdf §3.2 (Exclusions), p.12
```

If the answer is not in the policy documents, the system says so clearly and references the closest related clauses. Citations are computed deterministically from chunk metadata — never invented by the LLM — so the "Sources" line always matches what was actually retrieved (see [Key Design Decisions](#key-design-decisions)).

## Architecture

```
PDF Documents (docs/)
        ↓
Section-Aware Chunking (ingestion.py)
Preserves headings, clause numbers, page metadata
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
ChromaDB Vector Store — cosine similarity space, persisted to disk
        ↓
User Question
        ↓
Hybrid Retrieval (retriever.py)
Cosine similarity (candidate pool of 25) + BM25 keyword reranking → top 8
        ↓
Relevance Threshold Check (combined score >= 0.5)
        ↓
Pluggable LLM Backend (mock or Groq) — answers only, no self-cited sources
        ↓
Deterministic Citation Assembly (from chunk metadata, not the LLM)
        ↓
Grounded Answer + Clause Citations
        ↓
FastAPI (api.py) — served over HTTP
```

**Why hybrid retrieval:** pure cosine similarity on generic sentence embeddings tends to score any two insurance clauses similarly, since they share heavy boilerplate vocabulary ("cover," "policy," "claim," "damage"). This can bury the chunk that actually answers the question beneath chunks that just share generic wording. BM25 keyword matching is combined in specifically to reward literal term overlap (e.g. a chunk containing the actual words "wear" and "tear") — see `design_notes.md` for the before/after measurements that motivated this.

## Project Structure

```
policy-qa-bot/
├── docs/                ← place PDF policy documents here
├── config.py            ← all settings (chunk size, model, threshold, hybrid weighting)
├── ingestion.py         ← load, chunk, embed, store
├── retriever.py         ← hybrid search (cosine + BM25), citation formatting
├── chain.py             ← prompt construction, LLM backends, citation assembly
├── api.py               ← FastAPI app — /ask, /retrieve, /ingest, /documents, /health
├── schemas.py           ← Pydantic request/response models
├── main.py              ← CLI entry point
├── test_set.py          ← 10 Q&A test examples with automated pass/fail scoring
├── eval_results/        ← timestamped JSON output from each test_set.py run
├── design_notes.md      ← chunking strategy, retrieval design, and debugging history
├── pyproject.toml       ← dependencies (uv-managed)
├── uv.lock              ← locked dependency versions
├── Dockerfile           ← containerized deployment (uv-based)
├── docker-compose.yml   ← local container orchestration
├── .env.example         ← template for required environment variables
└── README.md
```

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core application logic |
| uv | Dependency management and virtual environments |
| FastAPI | HTTP API layer, request validation, auto-generated docs |
| LangChain | Document loading and embeddings pipeline |
| ChromaDB | Persisted vector database, cosine similarity space |
| rank-bm25 | Keyword-based reranking (hybrid retrieval) |
| HuggingFace | Embedding model (all-MiniLM-L6-v2) |
| Groq API | LLM inference (LLaMA 3.3 70b) — optional |
| tiktoken | Token-based chunk size measurement |
| pypdf | PDF text extraction |
| python-dotenv | Environment variable management |
| Docker | Containerized deployment |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/policy-qa-bot
cd policy-qa-bot
```

**2. Install dependencies with uv**
```bash
uv lock    # resolves and writes uv.lock (needs internet on first run)
uv sync    # creates .venv and installs everything into it
```

> **Note on torch:** `pyproject.toml` pins `torch` to the CPU-only PyTorch wheel index, since the default PyPI build pulls in several GB of CUDA packages that this project's embedding model doesn't need. If `uv lock` fails on your machine for any reason, the fallback is deleting the `[tool.uv.sources]` / `[[tool.uv.index]]` blocks and the `"torch"` line from `pyproject.toml` — you'll get the full CUDA build back (heavier, but works everywhere with zero config).

**3. Add policy documents**

Place your PDF policy documents in the `docs/` folder:
```
docs/
├── policy_a.pdf
├── policy_b.pdf
└── policy_c.pdf
```

**4. Configure environment**
```bash
cp .env.example .env
```
```
# LLM Backend: "mock" (no API needed) or "groq" (requires API key)
LLM_BACKEND=mock
GROQ_API_KEY=your_groq_key_here
```
Get a free Groq key at [console.groq.com](https://console.groq.com).

**5. Run the API**
```bash
uv run uvicorn api:app --reload
```
First run ingests everything in `docs/` and persists the vector store to `./chroma_db` — subsequent restarts skip re-ingestion unless you delete that folder or call `POST /ingest`.

**Or run the CLI** (unchanged, shares the same underlying modules):
```bash
uv run python main.py
uv run python main.py --demo   # 5 preset questions
```

**Or run the automated eval suite:**
```bash
uv run python test_set.py
```
Runs all 10 test questions, prints a PASS/FAIL verdict per question with reasoning, and saves a timestamped result file to `eval_results/` for comparing runs after future changes.

**Or run in Docker:**
```bash
docker compose up --build
```

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/ask` | POST | Ask a question, get a grounded answer with citations |
| `/retrieve` | POST | Debug endpoint — returns raw per-chunk cosine/BM25/combined scores for a question, without calling the LLM |
| `/ingest` | POST | Re-scan `docs/` and rebuild the vector store |
| `/documents` | GET | List currently indexed PDFs and total chunk count |
| `/health` | GET | Liveness check — vector store status, LLM backend, indexed document count |

### `/ask` request/response

```json
POST /ask
{
  "question": "Is wear and tear covered under the standard policy?"
}
```

```json
{
  "question": "Is wear and tear covered under the standard policy?",
  "answer": "...",
  "grounded": true,
  "citations": "Sources: QBE_Small_Business.pdf §87 (Portable Items Exclusions), p.88",
  "chunks_retrieved": 8,
  "relevant_chunks": 5,
  "backend": "groq"
}
```

### `/retrieve` — inspecting retrieval quality directly

```json
POST /retrieve
{
  "question": "Is wear and tear covered under the standard policy?"
}
```

Returns every retrieved chunk's `cosine_score`, `bm25_score`, combined `score`, whether it passed the relevance threshold, and its full content — useful for diagnosing why a question came back grounded or ungrounded, and for recalibrating `NO_ANSWER_THRESHOLD` / `HYBRID_ALPHA` against real documents instead of guessing.

## API Testing

The API auto-generates interactive documentation and an OpenAPI spec, both usable without writing any request by hand.

**Swagger UI** (built-in, no setup):
```
http://localhost:8000/docs
```
Every endpoint is listed with a "Try it out" button — fill in the request body and execute directly from the browser.

**Postman:**
1. Start the server: `uv run uvicorn api:app --reload`
2. In Postman: **File → Import**
3. Enter the URL: `http://localhost:8000/openapi.json`
4. Postman generates a full collection with all 5 endpoints, correct methods, and example schemas — ready to run immediately.

Suggested manual checks once imported:
- `GET /health` — confirm `vectorstore_loaded: true` before testing anything else
- `POST /ask` with an empty `question` — should return `422` (Pydantic validation)
- `POST /ask` with a genuinely unrelated question ("what's the weather today") — should return `grounded: false` and empty `citations`
- `POST /retrieve` with the same question you just asked via `/ask` — compare the raw scores against the `relevant_chunks` count you got back

## Citation Format

Every answer's citation is computed deterministically from chunk metadata — the LLM is explicitly instructed not to generate its own "Sources" line, and any citation-looking text it produces anyway is stripped before the real one is appended. This guarantees the citation always reflects what was actually retrieved, not what the model claims it read.

```
Sources: filename.pdf §clause_number (Section Name), p.page_number
```

Example:
```
Sources: QBE_Small_Business.pdf §3.2 (General Exclusions), p.108;
         QBE_Home_Policy.pdf §15 (Liability Exclusions), p.13
```

## No Answer Behaviour

When nothing clears the relevance threshold:

```
I cannot find a definitive answer in the provided policy wording.

The question may be out of scope or the topic may not be covered
in the uploaded policy documents.

Closest related clause found: policy.pdf §3.2 (Exclusions), p.12
```

When some content is retrieved but doesn't fully answer the question (a "near-miss"), the LLM is instructed to state clearly what it found and explicitly say it cannot give a definitive answer, rather than asserting a specific fact it doesn't have.

## Chunking Strategy

The system uses section-aware chunking — not standard character splitting:

- Detects headings by regex pattern (numbered sections, ALL CAPS, SECTION/PART prefixes)
- Numbered patterns (e.g. `9. We will not cover...`) are additionally checked for length and sentence shape, so ordinary numbered clause sentences aren't misdetected as section headings
- Flushes chunks at heading boundaries to preserve clause integrity
- Target chunk size: 600 tokens with 100 token overlap
- Each chunk carries metadata: doc_name, section, clause_number, page, heading_path

See `design_notes.md` for full technical details, including the specific false-positive cases found and fixed during development.

## Key Design Decisions

- **Section-aware over standard chunking** — preserves clause integrity and enables accurate citations
- **ChromaDB over FAISS** — supports metadata filtering and returns similarity scores for relevance thresholding; explicitly configured for cosine similarity space (`collection_configuration`) so those scores are actually bounded to [0, 1] and meaningful
- **Hybrid retrieval (cosine + BM25)** — pure embedding similarity clusters tightly across generic insurance vocabulary; BM25 keyword overlap is combined in 50/50 to reward literal term matches and pull the genuinely relevant chunk higher when embeddings alone can't discriminate
- **Relevance threshold (0.5)** — calibrated against measured combined-score distributions on real policy documents, not a guess; still a single data point, and a candidate for further tuning once more of `test_set.py`'s results accumulate
- **Deterministic, code-computed citations** — the LLM answers only; it is explicitly instructed not to cite sources, and any citation text it produces anyway is stripped before the real one (computed from actual chunk metadata) is appended. This was a deliberate fix after finding the LLM's self-generated citations could diverge from what was actually retrieved
- **Temperature 0.1** — near-deterministic responses for factual policy Q&A
- **Pluggable LLM** — system works offline with mock backend, production-ready with Groq
- **k=8 retrieval, 25-chunk candidate pool** — insurance questions often span multiple sections; the wider candidate pool gives BM25 reranking room to promote chunks that rank outside the top 8 on cosine alone
- **FastAPI wrapper over CLI-only** — same underlying modules (`ingestion.py`, `retriever.py`, `chain.py`) power both `main.py` (CLI) and `api.py` (HTTP), with the vector store built once at startup and persisted, not rebuilt per request

## Known Limitations

- **The relevance threshold alone cannot fully reject adversarial or off-topic queries.** A question with zero legitimate connection to insurance (e.g. asking about central bank interest rates) can still score above threshold on shared boilerplate vocabulary. In practice, the LLM's own instruction to "answer only from the provided context" is the layer actually preventing a wrong answer in these cases — the numeric threshold reduces noise but is not a hard guarantee of topical relevance on its own.
- **`test_set.py`'s pass/fail verdicts are a heuristic proxy, not a correctness grader.** It checks two testable behaviors (was retrieval "grounded," did the answer hedge appropriately) rather than judging factual accuracy. A test can pass this harness while still containing a subtly wrong detail, and can fail it for giving a correct, appropriately-scoped answer that the heuristic can't distinguish from evasiveness. Manual review of flagged failures (and periodic spot-checks of passes) is still necessary.
- **`0.5` and `HYBRID_ALPHA=0.5` are calibrated on a small number of real queries, not a systematic sweep.** Worth revisiting as `eval_results/` accumulates more runs across more diverse questions.

## Skills Demonstrated

`Python` `RAG` `FastAPI` `Docker` `uv` `LangChain` `ChromaDB` `Hybrid Retrieval (BM25 + Embeddings)` `Vector Embeddings` `HuggingFace` `Prompt Engineering` `Section-Aware Chunking` `Citation Extraction` `Pluggable Architecture` `Automated Evaluation Harness` `Root-Cause Debugging`