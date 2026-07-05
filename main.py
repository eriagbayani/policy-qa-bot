import sys
from ingestion import ingest_documents
from retriever import PolicyRetriever
from chain import PolicyQAChain
import logging
from logger.logging import setup_logging

logger = logging.getLogger(__name__)

# ── DISPLAY ────────────────────────────────────────────
def print_result(result: dict):
    logger.info(
    "\n%s\n"
    "Question: %s\n"
    "%s\n"
    "Answer:\n%s\n"
    "%s\n"
    "Backend: %s\n"
    "Chunks retrieved: %d (%d relevant)\n"
    "%s",
    "=" * 60,
    result["question"],
    "-" * 60,
    result["answer"],
    "-" * 60,
    result["backend"],
    result["chunks_retrieved"],
    result["relevant_chunks"],
    "=" * 60,
)

# Demo mode, this has default questions already for testing.
def run_demo(chain: PolicyQAChain):
    demo_questions = [
        "What is the waiting period for accidental damage?",
        "Is wear and tear covered under the standard policy?",
        "How does the deductible apply to water damage?",
        "What definitions apply to Insured Person?",
        "What are the general exclusions?",
    ]

    logger.info(
    "\n%s\n"
    "POLICY Q&A BOT — DEMO MODE\n"
    "Running %d demo questions...\n"
    "%s",
    "=" * 60,
    len(demo_questions),
    "=" * 60,
)

    for question in demo_questions:
        result = chain.answer(question)
        print_result(result)

# interactive mode
def run_interactive(chain: PolicyQAChain):

    logger.info(
    "\n%s\n"
    "POLICY Q&A BOT — INTERACTIVE MODE\n"
    "Type your question and press Enter.\n"
    "Type 'exit' to quit.\n"
    "%s",
    "=" * 60,
    "=" * 60,
)

    while True:
        question = input("\nQuestion: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            logger.info("Goodbye.")
            break

        result = chain.answer(question)
        print_result(result)

# main func
def main():
    setup_logging()
    logger.info(
    "\n%s\n"
    "POLICY Q&A BOT\n"
    "%s",
    "=" * 60,
    "=" * 60,
)

    logger.info("Step 1: Ingesting policy documents...")
    vectorstore = ingest_documents()

    logger.info("Step 2: Initializing retriever and QA chain...")
    retriever = PolicyRetriever(vectorstore=vectorstore)
    chain = PolicyQAChain(retriever=retriever)

    logger.info("System ready.")

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