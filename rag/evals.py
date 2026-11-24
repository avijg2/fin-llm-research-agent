"""Eval harness: grounded Q&A over the sample docs.

Each case states the expected fact (a number or phrase that must appear
in the answer) and, for out-of-scope cases, that the agent must refuse
instead of answering. Metrics:

* ``answer_contains_fact`` -- the expected fact string appears in the answer.
* ``citation_precision`` -- fraction of returned citations whose chunk
  belongs to the expected document.
* ``refusal_correct`` -- out-of-scope questions are refused, in-scope
  questions are not.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.pipeline import ResearchAgent

INSUFFICIENT = "Insufficient evidence"


@dataclass
class EvalCase:
    question: str
    expected_fact: str  # substring that must appear in the answer, or "" for refusal cases
    expected_doc_id: str  # doc the citation should come from, or "" for refusal cases
    should_refuse: bool = False


# All facts below are taken verbatim from data/sample_docs/*.txt (fictional
# sample companies). No real filings are used.
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        question="What was Northwind Cloud Systems revenue in Q3 2026?",
        expected_fact="$412.8 million",
        expected_doc_id="northwind-q3-2026-earnings-call",
    ),
    EvalCase(
        question="How much did Northwind's cloud segment earn in Q3 2026?",
        expected_fact="$298.1 million",
        expected_doc_id="northwind-q3-2026-earnings-call",
    ),
    EvalCase(
        question="What is Northwind's revenue guidance for Q4 2026?",
        expected_fact="$430 million",
        expected_doc_id="northwind-q3-2026-earnings-call",
    ),
    EvalCase(
        question="What was Northwind's gross margin in Q3 2026?",
        expected_fact="71.2%",
        expected_doc_id="northwind-q3-2026-earnings-call",
    ),
    EvalCase(
        question="What was Bluepeak Semiconductor's total revenue in fiscal 2026?",
        expected_fact="$1,284.6 million",
        expected_doc_id="bluepeak-10k-fy2026-excerpt",
    ),
    EvalCase(
        question="How much did Bluepeak spend on research and development in fiscal 2026?",
        expected_fact="$212.3 million",
        expected_doc_id="bluepeak-10k-fy2026-excerpt",
    ),
    EvalCase(
        question="What was Harborline Retail Group's revenue in Q3 2026 and did it grow?",
        expected_fact="$856.4 million",
        expected_doc_id="harborline-q3-2026-earnings-call",
    ),
    EvalCase(
        question="What was Harborline's same-store sales change in Q3 2026?",
        expected_fact="1.8%",
        expected_doc_id="harborline-q3-2026-earnings-call",
    ),
    EvalCase(
        question="What was Tesla's revenue in Q3 2026?",
        expected_fact="",
        expected_doc_id="",
        should_refuse=True,
    ),
    EvalCase(
        question="What was Northwind's revenue in 1999?",
        expected_fact="",
        expected_doc_id="",
        should_refuse=True,
    ),
]


@dataclass
class EvalResult:
    total: int
    fact_accuracy: float
    citation_precision: float
    refusal_accuracy: float
    passed: int
    details: list[dict]


def run_evals(agent: ResearchAgent | None = None, cases: list[EvalCase] | None = None) -> EvalResult:
    """Run all eval cases against *agent* and return aggregate metrics."""
    agent = agent or ResearchAgent()
    cases = cases if cases is not None else EVAL_CASES
    fact_hits = 0
    fact_total = 0
    citation_hits = 0
    citation_total = 0
    refusal_hits = 0
    passed = 0
    details: list[dict] = []
    for case in cases:
        result = agent.ask(case.question)
        if case.should_refuse:
            ok_refusal = result.refused or INSUFFICIENT in result.answer
            refusal_hits += 1 if ok_refusal else 0
            passed_case = ok_refusal
            details.append(
                {
                    "question": case.question,
                    "expected": "refusal",
                    "refused": bool(result.refused),
                    "passed": passed_case,
                }
            )
            passed += 1 if passed_case else 0
            continue
        # In-scope case.
        fact_total += 1
        fact_ok = case.expected_fact in result.answer
        fact_hits += 1 if fact_ok else 0
        # Citation precision: cited chunks from the expected doc / total cited.
        cited_doc_ids = [c.doc_id for c in result.citations]
        citation_total += len(cited_doc_ids)
        citation_hits += sum(1 for d in cited_doc_ids if d == case.expected_doc_id)
        refusal_ok = not result.refused
        refusal_hits += 1 if refusal_ok else 0
        passed_case = bool(fact_ok and refusal_ok and cited_doc_ids and case.expected_doc_id in cited_doc_ids)
        passed += 1 if passed_case else 0
        details.append(
            {
                "question": case.question,
                "expected_fact": case.expected_fact,
                "fact_found": fact_ok,
                "cited_docs": cited_doc_ids,
                "passed": passed_case,
            }
        )
    # Refusal accuracy counts both in-scope (not refused) and out-of-scope.
    refusal_total = len(cases)
    return EvalResult(
        total=len(cases),
        fact_accuracy=(fact_hits / fact_total) if fact_total else 0.0,
        citation_precision=(citation_hits / citation_total) if citation_total else 0.0,
        refusal_accuracy=(refusal_hits / refusal_total) if refusal_total else 0.0,
        passed=passed,
        details=details,
    )


if __name__ == "__main__":
    import json

    result = run_evals()
    print(f"Eval cases: {result.total}")
    print(f"Passed: {result.passed}/{result.total}")
    print(f"Fact accuracy: {result.fact_accuracy:.2%}")
    print(f"Citation precision: {result.citation_precision:.2%}")
    print(f"Refusal accuracy: {result.refusal_accuracy:.2%}")
    for d in result.details:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"[{status}] {d['question']}")
    print(json.dumps(result.details, indent=2))
