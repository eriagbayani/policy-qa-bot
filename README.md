# Policy Q&A Bot

A RAG-powered question answering system for insurance policy documents. Ask questions in plain English and get grounded answers with clause-level citations — strictly from the policy wording.

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

If the answer is not in the policy documents, the system says so clearly and references the closest related clauses.

## Architecture

```
PDF Documents (docs/)
        ↓
Section-Aware Chunking (ingestion.py)
Preserves headings, clause numbers, page metadata
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
ChromaDB Vector Store (in-memory)
        ↓
User Question
        ↓
Similarity Search — top 8 chunks
        ↓
Relevance Threshold Check (score >= 0.1)
        ↓
Pluggable LLM Backend (mock or Groq)
        ↓
Grounded Answer + Clause Citations
```

## Project Structure

```
policy-qa-bot/
├── docs/              ← place PDF policy documents here
├── config.py          ← all settings (chunk size, model, threshold)
├── ingestion.py       ← load, chunk, embed, store
├── retriever.py       ← search ChromaDB, format citations
├── chain.py           ← prompt construction, LLM backends
├── main.py            ← CLI entry point
├── test_set.py        ← 10 Q&A test examples
├── design_notes.md    ← chunking strategy and prompt design
├── .env               ← API keys and backend config
└── README.md
```

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core application logic |
| LangChain | Document loading and embeddings pipeline |
| ChromaDB | In-memory vector database |
| HuggingFace | Embedding model (all-MiniLM-L6-v2) |
| Groq API | LLM inference (LLaMA 3.3 70b) — optional |
| tiktoken | Token-based chunk size measurement |
| pypdf | PDF text extraction |
| python-dotenv | Environment variable management |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/policy-qa-bot
cd policy-qa-bot
```

**2. Install dependencies**

With uv (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

**3. Add policy documents**

Place your PDF policy documents in the `docs/` folder:
```
docs/
├── policy_a.pdf
├── policy_b.pdf
└── policy_c.pdf
```

**4. Configure environment**

Create a `.env` file:
```
# LLM Backend: "mock" (no API needed) or "groq" (requires API key)
LLM_BACKEND=mock
GROQ_API_KEY=your_groq_key_here
```

**5. Run the bot**

Interactive mode:
```bash
python main.py
```

Demo mode — runs 5 preset questions:
```bash
python main.py --demo
```

Run test set — 10 Q&A examples:
```bash
python test_set.py
```

## LLM Backends

The system supports pluggable LLM backends — switch with a single config change:

| Backend | Config | Requirements | Use case |
|---|---|---|---|
| Mock | `LLM_BACKEND=mock` | None | Testing, offline demo |
| Groq | `LLM_BACKEND=groq` | GROQ_API_KEY | Production quality answers |

Adding a new backend (OpenAI, Anthropic, local Ollama) requires implementing the `generate(question, context, citations)` interface in `chain.py`.

## Citation Format

Every answer ends with citations referencing the exact source:

```
Sources: filename.pdf §clause_number (Section Name), p.page_number
```

Example:
```
Sources: QBE_Small_Business.pdf §3.2 (General Exclusions), p.108;
         QBE_Home_Policy.pdf §15 (Liability Exclusions), p.13
```

## No Answer Behaviour

When the question cannot be answered from the policy documents:

```
I cannot find a definitive answer in the provided policy wording.

The question may be out of scope or the topic may not be covered
in the uploaded policy documents.

Closest related clause found: policy.pdf §3.2 (Exclusions), p.12
```

## Chunking Strategy

The system uses section-aware chunking — not standard character splitting:

- Detects headings by regex pattern (numbered sections, ALL CAPS, SECTION/PART prefixes)
- Flushes chunks at heading boundaries to preserve clause integrity
- Target chunk size: 600 tokens with 100 token overlap
- Each chunk carries metadata: doc_name, section, clause_number, page, heading_path

See `design_notes.md` for full technical details.

## Key Design Decisions

- **Section-aware over standard chunking** — preserves clause integrity and enables accurate citations
- **ChromaDB over FAISS** — supports metadata filtering and returns similarity scores for relevance thresholding
- **Relevance threshold (0.1)** — prevents irrelevant chunks from polluting the answer
- **Temperature 0.1** — near-deterministic responses for factual policy Q&A
- **Pluggable LLM** — system works offline with mock backend, production-ready with Groq
- **k=8 retrieval** — insurance questions often span multiple sections; more context improves grounding

## Skills Demonstrated

`Python` `RAG` `LangChain` `ChromaDB` `Vector Embeddings` `HuggingFace` `Prompt Engineering` `Section-Aware Chunking` `Citation Extraction` `Pluggable Architecture`s