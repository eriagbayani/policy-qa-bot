from groq import Groq
from config import (
    LLM_BACKEND,
    GROQ_API_KEY,
    GROQ_MODEL,
)

# PROMPT TEMPLATE -- adjust when needed
SYSTEM_PROMPT = """You are a precise insurance policy assistant.
Your job is to answer questions strictly based on the provided policy document excerpts.

Rules:
1. Answer ONLY from the provided context — never use outside knowledge
2. Do NOT include a "Sources:" line, citations, clause numbers, or page numbers
   in your answer — citations are generated separately by the system
3. Never include chunk headers like [Chunk 1 | ...] in your answer
4. If the context does not contain enough information to answer, say exactly:
   "I cannot find a definitive answer in the provided policy wording."
   Then briefly explain what related information was found, if any.
5. Be precise and concise — policy language matters
6. If multiple clauses are relevant, address all of them in your answer
7. Never speculate or infer beyond what is explicitly stated
8. If the context contains explicit exclusion language, state it definitively —
   do not hedge with "might" or "suggests". Quote the exact policy language."""


USER_PROMPT_TEMPLATE = """Policy Document Context:
{context}

Question: {question}

Instructions:
- Answer strictly from the context above
- Do not add a "Sources:" line or any citation — that is handled separately
- If you cannot find the answer, say so clearly

Answer:"""

# when using mock llm
class MockLLM:
    def generate(self, question: str, context: str, citations: str) -> str:
        if not context.strip():
            return (
                "I cannot find a definitive answer in the provided policy wording.\n\n"
                "No relevant clauses were found for this question. "
                "Please verify the question relates to the uploaded policy documents."
            )

        # Check if context is meaningful or just generic
        generic_phrases = [
            "how to contact", "phone number", "online portal",
            "website", "mobile app", "interest rate", "reserve bank"
        ]
        
        question_lower = question.lower()
        if any(phrase in question_lower for phrase in generic_phrases):
            return (
                "I cannot find a definitive answer in the provided policy wording.\n\n"
                "This question relates to operational or external information "
                "not typically contained in policy wording documents."
            )

        return (
            f"Based on the policy wording, the following information is relevant "
            f"to your question about '{question}':\n\n"
            f"The policy documents contain clauses that address this topic. "
            f"Please refer to the specific sections retrieved for detailed wording.\n\n"
            f"{citations}"
        )

# when using groq llm model
class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, question: str, context: str, citations: str) -> str:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1  # Low temperature for factual accuracy
        )

        answer = response.choices[0].message.content

        # Safety net: even with the prompt telling it not to, the model may
        # still tack on its own "Sources: ..." line (self-reported, not
        # verified against real chunk metadata). Strip anything from the
        # first "Sources:" onward so there is exactly one citation mechanism
        # — the deterministic one computed from actual chunk metadata below.
        marker = answer.find("Sources:")
        if marker != -1:
            answer = answer[:marker].rstrip()

        if citations:
            answer += f"\n\n{citations}"

        return answer

# llm factory
def get_llm():
    if LLM_BACKEND == "groq":
        return GroqLLM()
    return MockLLM()

# chain
class PolicyQAChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()

    def answer(self, question: str) -> dict:
        # Step 1 — Retrieve relevant chunks
        chunks = self.retriever.retrieve(question)

        # Step 2 — Check if relevant results exist
        has_relevant = self.retriever.has_relevant_results(chunks)

        # Step 3 — Format context and citations
        context = self.retriever.format_context(chunks)
        citations = self.retriever.format_citations(chunks)

        # Step 4 — Generate answer
        if not has_relevant:
            answer = (
                "I cannot find a definitive answer in the provided policy wording.\n\n"
                "The question may be out of scope or the topic may not be covered "
                "in the uploaded policy documents."
            )
            if chunks:
                closest = chunks[0]["metadata"]
                answer += (
                    f"\n\nClosest related clause found: "
                    f"{closest.get('doc_name', '')} "
                    f"§{closest.get('clause_number', '')} "
                    f"({closest.get('section', '')}), "
                    f"p.{closest.get('page', '')}"
                )
        else:
            answer = self.llm.generate(question, context, citations)

        return {
            "question": question,
            "answer": answer,
            "chunks_retrieved": len(chunks),
            "relevant_chunks": sum(1 for c in chunks if c["relevant"]),
            "citations": citations,
            "backend": LLM_BACKEND
        }