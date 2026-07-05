# Design Notes — Policy Q&A Bot

## Overview

This document explains the key technical decisions made in building the Policy Q&A Bot — a RAG-based assistant for insurance policy document question answering with mandatory clause-level citations, served as a FastAPI API.

A meaningful part of this project's design was shaped by bugs found through actual testing against real policy PDFs, not assumptions made up front. Section 8 documents that debugging history in detail, since the reasoning behind each fix is arguably more informative than the final code.

---

## 1. Chunking Strategy

### Why Section-Aware Chunking Over Standard Chunking

Standard character or token-based chunking (e.g. `RecursiveCharacterTextSplitter`) splits text at arbitrary boundaries — it has no understanding of document structure. For insurance policy documents this is a significant problem because:

- A clause split across two chunks loses its meaning
- A citation pointing to "§3.2" becomes meaningless if the chunk doesn't contain the heading
- Retrieval quality drops when context is fragmented mid-sentence

**Section-aware chunking solves this by:**

- Detecting section headings using regex patterns before splitting
- Flushing the current chunk at heading boundaries — not mid-clause
- Attaching rich metadata per chunk: `doc_name`, `section`, `clause_number`, `page`, `heading_path`
- Preserving heading hierarchy — so a chunk from "3.2 Exclusions" knows it belongs under "3. General Conditions"

### Heading Detection Patterns

The chunker detects headings using a set of regex patterns covering common insurance document structures:

```
SECTION A, SECTION 1       → ^SECTION\s+[A-Z0-9]+
PART 1, PART A             → ^PART\s+[A-Z0-9]+
1. DEFINITIONS             → ^\d+\.\s+[A-Z]
1.1 General                → ^\d+\.\d+\s+[A-Z]
1.1.1 Specific             → ^\d+\.\d+\.\d+\s+
ALL CAPS HEADINGS          → ^[A-Z][A-Z\s]{4,}$
Schedule 1, Endorsement 1  → ^Schedule\s+\d+, ^Endorsement\s+\d+
```

**Numbered patterns get an additional shape check.** Real insurance PDFs contain plenty of ordinary clause sentences that happen to start with a number — `"9. We will not cover property not owned by your business."` matches the same regex as a real heading like `"9. EXCLUSIONS"`. A numbered match is only accepted as a real heading if:
- it's under 8 words, **and**
- if it ends in a period, it's under 4 words

Short headings without trailing punctuation (`"7. Inspection of property"`) pass; long, sentence-shaped matches ending in a period don't. This was found and fixed after real citations came back mislabeled — see Section 8.1.

### Chunk Size Decision

Target: **600 tokens** with **100 token overlap**

- 400 tokens minimum: too small — splits clauses mid-sentence
- 800 tokens maximum: too large — retrieval becomes less precise
- 600 tokens: balances completeness of clause content with retrieval precision
- 100 token overlap: prevents important information from being lost at chunk boundaries

**Trade-off observed in practice:** at 600 tokens, a single chunk can still span several distinct clauses (e.g. reinstatement/replacement settlement math and an unrelated exclusions list ended up in the same chunk in testing). This dilutes the chunk's embedding — cosine similarity ends up representing an average of several topics rather than one. This is the main reason hybrid retrieval (Section 3) was added rather than relying on cosine similarity alone.

### Known Limitations

- Cross-references ("see Section 3.2") are not explicitly extracted into metadata — they remain in chunk text and are retrievable but not structured
- Some PDF documents use non-standard formatting — headings may not be detected if they use unusual spacing or fonts. One real example found in testing: running page headers like `"Portable items section 87"` don't match any current heading pattern, so a chunk can carry a stale/incorrect `clause_number` from an earlier heading even though it discusses a different, later-numbered clause. Not yet fixed — flagged for future work.
- Scanned PDFs or image-based PDFs will not extract text correctly — requires OCR preprocessing
- Chunks can still span multiple clauses at the 600-token target size (see above) — mitigated by hybrid retrieval, not eliminated by chunking alone

---

## 2. Metadata Design

Each chunk carries the following metadata:

```python
{
    "doc_name": "QBE_Policy.pdf",        # Source filename
    "section": "3.2 General Exclusions", # Detected heading text
    "clause_number": "3.2",              # Extracted clause number
    "heading_path": "3. Conditions > 3.2 Exclusions",  # Full hierarchy
    "page": 12                           # PDF page number
}
```

This metadata enables precise citations in the format:
```
QBE_Policy.pdf §3.2 (General Exclusions), p.12
```

---

## 3. Retrieval Strategy

### Vector Store

ChromaDB was chosen over FAISS for the following reasons:

- In-process, no server required — appropriate for this scope
- Supports metadata filtering for future enhancements
- Returns similarity scores alongside documents — enables relevance thresholding
- Simple API consistent with LangChain ecosystem

**Persistence and distance metric.** The vector store is persisted to disk (`CHROMA_PERSIST_DIR`) so the API doesn't re-embed all documents on every restart — only on first run or after `POST /ingest`. It's also explicitly configured for cosine similarity space via `collection_configuration={"hnsw": {"space": "cosine"}}` at creation time. Without this, both the actual HNSW index and LangChain's relevance-score formula silently default to raw L2 distance, which produces scores outside `[0, 1]` (including negative values) and makes any relevance threshold meaningless. This was a real, non-obvious bug found during testing — see Section 8.2.

### Hybrid Retrieval — Cosine Similarity + BM25

Pure cosine similarity on generic sentence embeddings (`all-MiniLM-L6-v2`) tends to score most insurance clauses similarly to each other, because they share heavy boilerplate vocabulary — "cover," "policy," "claim," "damage," "excess," "schedule" appear in nearly every clause regardless of topic. In testing, this caused the chunk that actually answered a question to rank below several chunks that were only superficially related. See Section 8.4 for the measured example.

To address this, retrieval combines two signals:

1. **Cosine similarity** over a widened candidate pool (`CANDIDATE_POOL_SIZE = 25`, rather than retrieving only the final top-8 directly)
2. **BM25 keyword score** computed over the same candidate pool, rewarding literal term overlap with the query

Both are min-max normalized to `[0, 1]` and combined:
```
combined_score = HYBRID_ALPHA * normalized_cosine + (1 - HYBRID_ALPHA) * normalized_bm25
```
`HYBRID_ALPHA = 0.5` weights both signals equally. The top 8 chunks by combined score are returned to the LLM.

This is a standard mitigation for exactly this failure mode — semantic embeddings alone can miss chunks that are relevant primarily because of specific terminology, and BM25 recovers that signal cheaply (pure Python, no additional model, negligible latency at this corpus scale).

### Relevance Threshold

A minimum **combined hybrid score of 0.5** is used to determine whether a retrieved chunk is relevant enough to include in the answer.

This value was raised from an initial `0.1` (then briefly `0.35`) after measuring real combined-score distributions against actual policy documents: for a genuine "is wear and tear covered" query, combined scores ranged `0.38–0.81`, with the correct clause landing at `0.79` and the weakest, least-relevant retrieved chunks sitting at `0.38–0.41`. `0.5` sits between those clusters. This is based on a small number of real measurements, not a systematic sweep — see Section 7 (Known Limitations) and the `test_set.py` eval harness for ongoing calibration.

- Chunks below this threshold are retrieved but flagged as irrelevant
- If no chunks meet the threshold, the system returns the "cannot find" response
- The closest chunk is still cited as a reference point even in "cannot find" responses

**Important caveat found in testing:** the threshold reduces noise but does not fully guarantee topical relevance. A question with no legitimate connection to insurance at all can still retrieve chunks scoring above `0.5`, purely from shared boilerplate vocabulary. In practice, the LLM's own prompt instruction ("answer only from the provided context") is the layer actually preventing an incorrect answer in these cases — the threshold is a filter, not a hard guarantee. See Section 7.

### Top-K Retrieval

**k=8** chunks are returned to the LLM per query (drawn from the 25-chunk hybrid-reranked candidate pool) — higher than the typical k=4 default because:

- Insurance policy questions often span multiple sections
- Water damage, for example, appears in exclusions, conditions, and coverage sections simultaneously
- More context gives the LLM better grounding for complex multi-clause questions

### Debug Endpoint

`POST /retrieve` returns the raw `cosine_score`, `bm25_score`, and combined `score` for every retrieved chunk, along with full chunk content, without calling the LLM. This was built specifically to diagnose the threshold/hybrid-retrieval issues described in Section 8, and is the intended tool for any future threshold recalibration — inspect real scores before changing the number, rather than guessing.

---

## 4. Prompt Design

### System Prompt Philosophy

The system prompt is designed around three principles:

**Strict grounding** — The LLM is explicitly instructed to answer only from provided context and never use outside knowledge. This prevents hallucination of policy terms that don't exist in the document.

**Definitive language** — When the context contains explicit exclusion or inclusion language, the LLM is instructed to state it definitively — not hedge with "might" or "suggests." Insurance answers need to be precise.

**No self-generated citations** — The LLM is explicitly instructed *not* to produce a "Sources:" line, clause numbers, or page references at all. This is a deliberate change from an earlier design (see Section 8.3) — citations are now assembled entirely by code from actual chunk metadata, after the LLM's answer is generated.

### Citation Assembly — Single Source of Truth

Citations are **not** generated by the LLM. The flow is:

1. The LLM generates an answer from the provided context, with no citation instructions
2. As a safety net, any text the model still includes after the literal string `"Sources:"` is stripped (covers models that ignore the instruction)
3. The deterministic citation string — built from the actual metadata of chunks that passed the relevance threshold — is appended

```
Sources: filename.pdf §clause_number (Section Name), p.page_number
```

This guarantees the citation the user sees always matches what was actually retrieved, rather than a citation the model inferred or transcribed from reading the passage itself. An earlier design let the LLM generate its own citation from context text; testing found real cases where the LLM's self-reported citation and the code-computed one disagreed (see Section 8.3).

### Temperature Setting

Temperature is set to **0.1** — near-deterministic. Policy Q&A is a factual task where consistency and accuracy matter more than creativity. Lower temperature reduces the risk of the LLM paraphrasing policy language in a way that changes its meaning.

---

## 5. Negative Question Handling

### "I Cannot Find" Behaviour

The system handles out-of-scope and near-miss questions through two mechanisms:

**Relevance threshold** — if no retrieved chunks score above the combined hybrid threshold, the system bypasses the LLM entirely and returns the standard "cannot find" response programmatically. This avoids wasting LLM calls on clearly irrelevant queries.

**LLM-level instruction** — for borderline cases where chunks are retrieved but the answer isn't definitively there, the LLM is instructed to:
1. State it cannot find a definitive answer
2. Explain what related information was found
3. Reference the closest related clauses (without inventing a citation for them — see Section 4)

### Three Query Categories

| Category | Behaviour |
|---|---|
| In-domain | Definitive answer with clause citations |
| Near-miss | "Cannot find definitive answer" + closest related clauses |
| Out-of-scope | "Cannot find" + no related clauses found (ideally — see Section 7 caveat on threshold limits) |

---

## 6. Pluggable LLM Backend

The system is designed so the LLM backend is swappable with a single configuration change:

```python
# .env
LLM_BACKEND=mock    # No external API — works offline
LLM_BACKEND=groq    # Groq API — production quality answers
```

Both backends implement the same interface:
```python
def generate(self, question: str, context: str, citations: str) -> str
```

Adding a new backend (OpenAI, Anthropic, local Ollama) requires only implementing this interface and adding the option to the factory function in `chain.py`.

---

## 7. Known Limitations and Future Improvements

| Limitation | Proposed Fix |
|---|---|
| Relevance threshold alone can't fully reject adversarial/off-topic queries scoring above threshold on shared vocabulary | Currently mitigated by strict prompt instructions ("answer only from context"), not the threshold itself. Consider a lightweight topic/domain classifier as a pre-filter |
| Cross-references not structured in metadata | Extract "see Section X.X" patterns during chunking and store as `cross_refs` metadata field |
| Some PDF heading detection misses non-standard formatting (e.g. running page headers like "Portable items section 87") | Add font-size based heading detection using pdfplumber instead of pypdf |
| Large chunks (600 tokens) can still span multiple distinct clauses, diluting embeddings | Mitigated by hybrid retrieval; consider smaller chunk size or clause-boundary-only splitting as a further improvement |
| `test_set.py` verdicts are a heuristic proxy (grounded + hedge-phrase check), not a factual-correctness grader | Would need either human grading or an LLM-as-judge step to verify actual answer accuracy, not just behavioral pattern |
| `NO_ANSWER_THRESHOLD` (0.5) and `HYBRID_ALPHA` (0.5) are calibrated on a small number of real queries | Expand `test_set.py` and use `eval_results/` history to tune systematically as more runs accumulate |
| Single file type tested (PDF) | DOCX support is coded but requires additional testing |
| Rate limiting on Groq free tier | Add exponential backoff retry logic or switch to a local model for high-volume use |

---

## 8. Debugging History

This section documents specific bugs found and fixed during development against real policy PDFs, since the reasoning behind each is arguably more instructive than the final code.

### 8.1 Heading Regex False Positives

**Symptom:** citations came back with garbled section names, e.g. `§2 (2. The amount of each claim otherwise payable shall be reduced by the amount of the)` and `§9 (9. We will not cover property not owned by your business.)`.

**Cause:** the numbered heading pattern `^\d+\.\s+[A-Z]` matches any numbered sentence starting with a capital letter — not just actual section titles. Real insurance clauses are full of numbered body sentences that happen to fit this shape.

**Fix:** numbered pattern matches are additionally checked for length (reject if over 8 words) and sentence shape (reject if the line ends in a period and is over 4 words). Verified against the exact failing examples plus known-good short headings (`"7. Inspection of property"`, `"1. Reinstatement and replacement"`).

### 8.2 Relevance Scores Not Meaningful (Distance Metric Misconfiguration)

**Symptom:** the relevance threshold appeared to do nothing — every retrieved chunk was marked "relevant" regardless of actual topical relevance (`chunks_retrieved == relevant_chunks` on every query).

**Cause, found in two stages:**
1. ChromaDB defaults to raw L2 (Euclidean) distance unless explicitly told otherwise. LangChain's relevance-score conversion assumes cosine space, so scores came back outside `[0, 1]` (including negative values), making any threshold comparison meaningless.
2. The first fix attempt — `collection_metadata={"hnsw:space": "cosine"}` — did not actually work, because the installed `langchain_chroma`/`chromadb` version reads `collection.configuration.hnsw.space`, not `collection.metadata["hnsw:space"]`. The legacy key is silently ignored.

**Fix:** switched to `collection_configuration={"hnsw": {"space": "cosine"}}`, confirmed by directly inspecting `Chroma._select_relevance_score_fn()` to verify it now selects `_cosine_relevance_score_fn`, and by testing that a clearly relevant and clearly irrelevant document produced properly separated scores (`0.68` vs `0.32` in an isolated test) rather than both landing in the same narrow band.

### 8.3 Citation Duplication / Divergence

**Symptom:** an answer confidently cited `"§87 (Portable items), p.89"` inline, while the API's separate `citations` field listed five entirely different, unrelated clauses.

**Cause:** the system prompt instructed the LLM to generate its own `"Sources:"` line by reading the raw context it was given, completely independent of the code-computed `citations` string built from actual chunk metadata. A guard (`if "Sources:" not in answer: append citations`) meant the code-computed citation was only ever used when the LLM forgot to add one — which, given the prompt told it to always add one, was effectively never.

**Fix:** removed all citation-generation instructions from the prompt; the LLM now only answers. Any `"Sources:"` text it produces anyway is stripped as a safety net, and the deterministic, metadata-based citation is always appended afterward. Verified the LLM's self-reported quote (the "vibration, wear, tear" language) was in fact present in the actual retrieved chunk content — confirming this was a citation-attribution bug, not a hallucinated answer.

### 8.4 Retrieval Ranking — Cosine Similarity Burying the Correct Answer

**Symptom:** using `/retrieve` on the query "Is wear and tear covered under the standard policy?", the chunk containing the actual exclusion clause (literally: *"...vibration, wear, tear and/or depreciation"*) ranked **5th of 8** on pure cosine similarity, behind chunks about reinstatement/replacement settlement, breakage cost calculations, and property inspection rights — none of which mention wear and tear at all.

**Cause:** all retrieved chunks scored within a narrow band (`0.43–0.49`) because they share heavy insurance boilerplate vocabulary. Cosine similarity on generic sentence embeddings couldn't discriminate the specific relevant chunk from generic-but-unrelated ones.

**Fix:** added BM25 hybrid reranking (Section 3). Re-running the same query afterward moved the correct clause to **2nd of 8**, with its BM25 score (`14.97`) clearly the strongest keyword-match signal in the set, and produced real score separation (`0.38–0.81`) instead of a flat cluster — which is also what made the subsequent threshold recalibration (0.1 → 0.5) meaningful rather than arbitrary.