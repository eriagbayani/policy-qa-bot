import sys
from ingestion import ingest_documents
from retriever import PolicyRetriever
from chain import PolicyQAChain

# ── DISPLAY ────────────────────────────────────────────
def print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"Question: {result['question']}")
    print("-" * 60)
    print(f"Answer:\n{result['answer']}")
    print("-" * 60)
    print(f"Backend: {result['backend']}")
    print(f"Chunks retrieved: {result['chunks_retrieved']} "
          f"({result['relevant_chunks']} relevant)")
    print("=" * 60)

# Demo mode, this has default questions already for testing.
def run_demo(chain: PolicyQAChain):
    demo_questions = [
        "What is the waiting period for accidental damage?",
        "Is wear and tear covered under the standard policy?",
        "How does the deductible apply to water damage?",
        "What definitions apply to Insured Person?",
        "What are the general exclusions?",
    ]

    print("\n" + "=" * 60)
    print("POLICY Q&A BOT — DEMO MODE")
    print(f"Running {len(demo_questions)} demo questions...")
    print("=" * 60)

    for question in demo_questions:
        result = chain.answer(question)
        print_result(result)

# interactive mode
def run_interactive(chain: PolicyQAChain):
    print("\n" + "=" * 60)
    print("POLICY Q&A BOT — INTERACTIVE MODE")
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        print()
        question = input("Question: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye.")
            break

        result = chain.answer(question)
        print_result(result)

# main func
def main():
    print("=" * 60)
    print("POLICY Q&A BOT")
    print("=" * 60)

    # Step 1 — Ingest documents
    print("\nStep 1: Ingesting policy documents...")
    vectorstore = ingest_documents()

    # Step 2 — Initialize retriever and chain
    print("\nStep 2: Initializing retriever and QA chain...")
    retriever = PolicyRetriever(vectorstore=vectorstore)
    chain = PolicyQAChain(retriever=retriever)

    print("\nSystem ready.")

    # Step 3 — Choose mode
    mode = "interactive"
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        mode = "demo"

    if mode == "demo":
        run_demo(chain)
    else:
        run_interactive(chain)

if __name__ == "__main__":
    main()