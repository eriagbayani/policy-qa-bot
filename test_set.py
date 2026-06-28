"""
Test set: 10 Q&A examples for Policy Q&A Bot
- 5 in-domain: questions answerable from QBE policy docs
- 3 near-miss: related but not directly answered
- 2 out-of-scope: completely outside the docs
"""

from chain import PolicyQAChain
from retriever import PolicyRetriever
from ingestion import ingest_documents
import time
# ── TEST CASES ─────────────────────────────────────────
TEST_CASES = [
    # ── IN-DOMAIN (5) ──────────────────────────────────
    {
        "id": 1,
        "type": "in-domain",
        "question": "Is wear and tear covered under the standard policy?",
        "expected": "Not covered — explicitly excluded",
        "notes": "Clear exclusion in policy wording"
    },
    {
        "id": 2,
        "type": "in-domain",
        "question": "What are the general exclusions?",
        "expected": "War, terrorism, unoccupancy, pollution, asbestos listed",
        "notes": "Covered explicitly in general exclusions section"
    },
    {
        "id": 3,
        "type": "in-domain",
        "question": "What happens if the business premises are unoccupied?",
        "expected": "Coverage excluded after 60 consecutive days unoccupied",
        "notes": "Unoccupancy clause in Small Business Insurance"
    },
    {
        "id": 4,
        "type": "in-domain",
        "question": "Are acts of terrorism covered?",
        "expected": "Not covered — explicitly excluded under general exclusions",
        "notes": "War and terrorism exclusion clause"
    },
    {
        "id": 5,
        "type": "in-domain",
        "question": "What does accidental damage cover?",
        "expected": "Covers sudden and unforeseen physical damage",
        "notes": "Accidental damage section in Contents Insurance"
    },

    # ── NEAR-MISS (3) ──────────────────────────────────
    {
        "id": 6,
        "type": "near-miss",
        "question": "What is the waiting period for accidental damage?",
        "expected": "Cannot find — policy covers accidental damage but no waiting period specified",
        "notes": "Accidental damage exists in docs but waiting period not mentioned"
    },
    {
        "id": 7,
        "type": "near-miss",
        "question": "How does the deductible apply to water damage?",
        "expected": "Cannot find definitive answer — water damage mentioned but deductible application not specified",
        "notes": "Water damage in docs but deductible mechanics not explicit"
    },
    {
        "id": 8,
        "type": "near-miss",
        "question": "What definitions apply to Insured Person?",
        "expected": "Cannot find — Insured defined but Insured Person not explicitly",
        "notes": "Near miss — Insured defined, Insured Person not"
    },

    # ── OUT-OF-SCOPE (2) ───────────────────────────────
    {
        "id": 9,
        "type": "out-of-scope",
        "question": "What is the premium payment schedule for monthly installments?",
        "expected": "Cannot find — premium payment schedules not in policy wording",
        "notes": "Premium schedules are in separate documents"
    },
    {
        "id": 10,
        "type": "out-of-scope",
        "question": "What is the current interest rate set by the Reserve Bank of Australia?",
        "expected": "Cannot find — monetary policy not in insurance documents",
        "notes": "Completely outside insurance domain"
    }
]

# ── RUNNER ─────────────────────────────────────────────
def run_test_set():
    print("=" * 60)
    print("POLICY Q&A BOT — TEST SET")
    print("=" * 60)

    print("\nIngesting documents...")
    vectorstore = ingest_documents()
    retriever = PolicyRetriever(vectorstore=vectorstore)
    chain = PolicyQAChain(retriever=retriever)

    results = {
        "in-domain": [],
        "near-miss": [],
        "out-of-scope": []
    }

    for test in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"Test {test['id']} [{test['type'].upper()}]")
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected']}")
        print("-" * 60)

        result = chain.answer(test["question"])
        time.sleep(3)

        print(f"Answer:\n{result['answer']}")
        print(f"\nCitations: {result['citations']}")
        print(f"Relevant chunks: {result['relevant_chunks']}/{result['chunks_retrieved']}")
        print(f"Notes: {test['notes']}")

        results[test["type"]].append({
            "id": test["id"],
            "question": test["question"],
            "answer": result["answer"],
            "citations": result["citations"],
            "relevant_chunks": result["relevant_chunks"]
        })

    # ── SUMMARY ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"In-domain (5):   {len(results['in-domain'])} run")
    print(f"Near-miss (3):   {len(results['near-miss'])} run")
    print(f"Out-of-scope (2): {len(results['out-of-scope'])} run")
    print(f"Total: {len(TEST_CASES)} tests completed")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_test_set()