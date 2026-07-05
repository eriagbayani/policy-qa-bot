"""
Test set: 10 Q&A examples for Policy Q&A Bot
- 5 in-domain: questions answerable from QBE policy docs
- 3 near-miss: related but not directly answered
- 2 out-of-scope: completely outside the docs

Each test case now has an `expect_grounded` flag and gets an automatic
PASS/FAIL verdict, instead of just printing output for a human to eyeball.
The verdict logic is a proxy, not ground truth — see _verdict() below for
exactly what it checks and why. Run this any time you change config.py,
retriever.py, or ingestion.py, to catch regressions before they surface as
a surprise in Postman.

Results are saved to eval_results/<timestamp>.json so you can diff a
change's effect against a previous run instead of re-reading terminal
output from memory.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from chain import PolicyQAChain
from retriever import PolicyRetriever
from ingestion import ingest_documents, load_existing_vectorstore

# Phrases the system is instructed to use when it can't find a definitive
# answer (see chain.py SYSTEM_PROMPT). Used as a proxy for "did the system
# appropriately hedge instead of asserting something it doesn't have."
HEDGE_PHRASES = [
    "cannot find a definitive answer",
    "cannot find",
    "no definitive answer",
]

SLEEP_BETWEEN_CALLS_SECONDS = 3  # Groq free-tier rate limiting

# ── TEST CASES ─────────────────────────────────────────
TEST_CASES = [
    # ── IN-DOMAIN (5) ──────────────────────────────────
    {
        "id": 1,
        "type": "in-domain",
        "question": "Is wear and tear covered under the standard policy?",
        "expected": "Not covered — explicitly excluded",
        "expect_grounded": True,
        "notes": "Clear exclusion in policy wording"
    },
    {
        "id": 2,
        "type": "in-domain",
        "question": "What are the general exclusions?",
        "expected": "War, terrorism, unoccupancy, pollution, asbestos listed",
        "expect_grounded": True,
        "notes": "Covered explicitly in general exclusions section"
    },
    {
        "id": 3,
        "type": "in-domain",
        "question": "What happens if the business premises are unoccupied?",
        "expected": "Coverage excluded after 60 consecutive days unoccupied",
        "expect_grounded": True,
        "notes": "Unoccupancy clause in Small Business Insurance"
    },
    {
        "id": 4,
        "type": "in-domain",
        "question": "Are acts of terrorism covered?",
        "expected": "Not covered — explicitly excluded under general exclusions",
        "expect_grounded": True,
        "notes": "War and terrorism exclusion clause"
    },
    {
        "id": 5,
        "type": "in-domain",
        "question": "What does accidental damage cover?",
        "expected": "Covers sudden and unforeseen physical damage",
        "expect_grounded": True,
        "notes": "Accidental damage section in Contents Insurance"
    },

    # ── NEAR-MISS (3) ──────────────────────────────────
    {
        "id": 6,
        "type": "near-miss",
        "question": "What is the waiting period for accidental damage?",
        "expected": "Cannot find — policy covers accidental damage but no waiting period specified",
        "expect_grounded": None,  # retrieval may legitimately find related content
        "notes": "Accidental damage exists in docs but waiting period not mentioned"
    },
    {
        "id": 7,
        "type": "near-miss",
        "question": "How does the deductible apply to water damage?",
        "expected": "Cannot find definitive answer — water damage mentioned but deductible application not specified",
        "expect_grounded": None,
        "notes": "Water damage in docs but deductible mechanics not explicit"
    },
    {
        "id": 8,
        "type": "near-miss",
        "question": "What definitions apply to Insured Person?",
        "expected": "Cannot find — Insured defined but Insured Person not explicitly",
        "expect_grounded": None,
        "notes": "Near miss — Insured defined, Insured Person not"
    },

    # ── OUT-OF-SCOPE (2) ───────────────────────────────
    {
        "id": 9,
        "type": "out-of-scope",
        "question": "What is the premium payment schedule for monthly installments?",
        "expected": "Cannot find — premium payment schedules not in policy wording",
        "expect_grounded": False,
        "notes": "Premium schedules are in separate documents"
    },
    {
        "id": 10,
        "type": "out-of-scope",
        "question": "What is the current interest rate set by the Reserve Bank of Australia?",
        "expected": "Cannot find — monetary policy not in insurance documents",
        "expect_grounded": False,
        "notes": "Completely outside insurance domain"
    }
]


def _hedges(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in HEDGE_PHRASES)


def _verdict(test: dict, result: dict) -> tuple[str, str]:
    """
    Returns (verdict, reason). This is a heuristic proxy, not ground truth —
    it does not check factual correctness of the answer, only two testable
    behavioral properties:
      1. Did retrieval find enough to call the question "grounded"?
      2. Did the answer hedge ("cannot find...") rather than assert a
         specific fact?
    A human should still spot-check the actual answer text, especially for
    near-miss cases where "grounded" alone doesn't tell you much.
    """
    grounded = result["relevant_chunks"] > 0
    hedged = _hedges(result["answer"])
    expect_grounded = test["expect_grounded"]

    if test["type"] == "in-domain":
        if expect_grounded and grounded and not hedged:
            return "PASS", "grounded and answered confidently, as expected"
        if not grounded:
            return "FAIL", "expected grounded retrieval, got none — check threshold/chunking"
        if hedged:
            return "FAIL", "retrieval succeeded but answer hedged anyway — check prompt/context quality"
        return "FAIL", "unexpected combination"

    # near-miss and out-of-scope: the real requirement is "don't assert a
    # specific fact you don't have" — hedging is correct regardless of
    # whether some tangentially related content was retrieved.
    if hedged:
        return "PASS", "correctly hedged instead of asserting an unsupported fact"
    if test["type"] == "out-of-scope" and grounded:
        return "FAIL", "retrieved content for a question with no business being grounded — check threshold"
    return "FAIL", "answered without hedging on a question it shouldn't be confident about"


# ── RUNNER ─────────────────────────────────────────────
def run_test_set():
    print("=" * 60)
    print("POLICY Q&A BOT — TEST SET")
    print("=" * 60)

    print("\nLoading vectorstore (persisted if available, else ingesting)...")
    vectorstore = load_existing_vectorstore() or ingest_documents()
    retriever = PolicyRetriever(vectorstore=vectorstore)
    chain = PolicyQAChain(retriever=retriever)

    all_results = []

    for test in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"Test {test['id']} [{test['type'].upper()}]")
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected']}")
        print("-" * 60)

        result = chain.answer(test["question"])
        raw_chunks = retriever.retrieve(test["question"])
        time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)

        verdict, reason = _verdict(test, result)

        print(f"Answer:\n{result['answer']}")
        print(f"\nCitations: {result['citations']}")
        print(f"Relevant chunks: {result['relevant_chunks']}/{result['chunks_retrieved']}")
        if raw_chunks:
            print(f"Top chunk score: {raw_chunks[0]['score']} "
                  f"(cosine={raw_chunks[0].get('cosine_score')}, "
                  f"bm25={raw_chunks[0].get('bm25_score')})")
        print(f"Notes: {test['notes']}")
        print(f"VERDICT: {verdict} — {reason}")

        all_results.append({
            "id": test["id"],
            "type": test["type"],
            "question": test["question"],
            "expected": test["expected"],
            "answer": result["answer"],
            "citations": result["citations"],
            "relevant_chunks": result["relevant_chunks"],
            "chunks_retrieved": result["chunks_retrieved"],
            "top_chunk_score": raw_chunks[0]["score"] if raw_chunks else None,
            "top_chunk_cosine": raw_chunks[0].get("cosine_score") if raw_chunks else None,
            "top_chunk_bm25": raw_chunks[0].get("bm25_score") if raw_chunks else None,
            "verdict": verdict,
            "reason": reason,
        })

    # ── SUMMARY ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print("=" * 60)
    for t in ("in-domain", "near-miss", "out-of-scope"):
        subset = [r for r in all_results if r["type"] == t]
        passed = sum(1 for r in subset if r["verdict"] == "PASS")
        print(f"{t:14s}: {passed}/{len(subset)} passed")

    total_passed = sum(1 for r in all_results if r["verdict"] == "PASS")
    print(f"{'TOTAL':14s}: {total_passed}/{len(all_results)} passed")
    print("=" * 60)

    for r in all_results:
        if r["verdict"] == "FAIL":
            print(f"  FAIL — Test {r['id']}: {r['question']}  ({r['reason']})")

    # ── SAVE ───────────────────────────────────────────
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total": len(all_results),
            "passed": total_passed,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    return all_results


if __name__ == "__main__":
    run_test_set()