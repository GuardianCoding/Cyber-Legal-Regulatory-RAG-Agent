"""
Evaluation harness: 10 hardcoded Q&A pairs covering CYBR7003 topics.
Measures retrieval recall (keyword hit in any retrieved chunk) and grounding rate.

Run with: python src/eval.py
"""

from src.tools import hybrid_search
from src.agent import run_agent

QA_PAIRS = [
    {
        "question": "When must an organisation notify affected individuals under the NDB scheme?",
        "expected_keywords": ["notify", "notification", "30 days", "eligible data breach", "ndb"],
    },
    {
        "question": "What constitutes an eligible data breach under the Privacy Act 1988?",
        "expected_keywords": ["eligible data breach", "unauthorised access", "serious harm", "likely to result"],
    },
    {
        "question": "What are the ASD Essential Eight maturity level 1 requirements?",
        "expected_keywords": ["maturity level 1", "essential eight", "application control", "patch applications"],
    },
    {
        "question": "What obligations does APP 11 impose regarding the security of personal information?",
        "expected_keywords": ["app 11", "reasonable steps", "personal information", "security", "misuse"],
    },
    {
        "question": "What happened in McClure v Medibank Private Limited [2025] FCA 167?",
        "expected_keywords": ["mcclure", "medibank", "fca", "class action", "167"],
    },
    {
        "question": "What is the difference between a privacy breach and an eligible data breach?",
        "expected_keywords": ["eligible data breach", "serious harm", "privacy breach", "threshold"],
    },
    {
        "question": "What are the key differences between ISO 27001 and the ASD Essential Eight?",
        "expected_keywords": ["iso 27001", "essential eight", "risk", "certification", "framework"],
    },
    {
        "question": "What obligations does APP 1 impose on organisations handling personal information?",
        "expected_keywords": ["app 1", "privacy policy", "open and transparent", "personal information"],
    },
    {
        "question": "What data was compromised in the Medibank cyber incident?",
        "expected_keywords": ["medibank", "health information", "personal information", "data", "customer"],
    },
    {
        "question": "What is the role of the OAIC under the Privacy Act 1988?",
        "expected_keywords": ["oaic", "office of the australian information commissioner", "commissioner", "privacy act"],
    },
]

TRUNC = 55


def _retrieval_hit(question: str, keywords: list[str]) -> bool:
    chunks = hybrid_search(question)
    combined = " ".join(c["text"] for c in chunks).lower()
    return any(kw.lower() in combined for kw in keywords)


def _grounding_result(question: str) -> bool:
    result = run_agent(question)
    return result.get("grounded", False)


def main():
    results = []
    for i, pair in enumerate(QA_PAIRS, 1):
        q = pair["question"]
        print(f"[{i:02d}/10] {q[:TRUNC]}...", flush=True)
        retrieved = _retrieval_hit(q, pair["expected_keywords"])
        grounded = _grounding_result(q)
        results.append((i, q, retrieved, grounded))

    print()
    header = f"{'Q#':<4} {'Question':<{TRUNC+2}} {'Retrieved':<12} {'Grounded':<10}"
    print(header)
    print("-" * len(header))
    for i, q, retrieved, grounded in results:
        q_trunc = (q[:TRUNC] + "…") if len(q) > TRUNC else q
        r_mark = "✓" if retrieved else "✗"
        g_mark = "✓" if grounded else "✗"
        print(f"{i:<4} {q_trunc:<{TRUNC+2}} {r_mark:<12} {g_mark:<10}")

    retrieval_score = sum(1 for _, _, r, _ in results if r)
    grounding_score = sum(1 for _, _, _, g in results if g)
    print()
    print(f"Retrieval recall : {retrieval_score}/10")
    print(f"Grounding rate   : {grounding_score}/10")


if __name__ == "__main__":
    main()
