"""CLI demo: ingest the sample docs and ask three questions.

Usage:
    python scripts/demo.py
    python scripts/demo.py --question "What was Northwind's gross margin?"

Runs fully offline in mock-extractive mode unless OPENAI_API_KEY is set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/demo.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import get_default_agent

DEMO_QUESTIONS = [
    "What was Northwind Cloud Systems revenue in Q3 2026?",
    "How much did Bluepeak spend on research and development in fiscal 2026?",
    "What was Tesla's revenue in Q3 2026?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fin LLM Research Agent demo")
    parser.add_argument("--question", help="Ask a single question instead of the demo set")
    args = parser.parse_args()

    agent = get_default_agent()
    print(f"Loaded {len(agent.documents)} documents, {len(agent.chunks)} chunks")
    print(f"Backend: {agent.backend}\n")

    questions = [args.question] if args.question else DEMO_QUESTIONS
    for question in questions:
        result = agent.ask(question)
        print(f"Q: {question}")
        print(f"Mode: {result.mode} | Confidence: {result.confidence} | Refused: {result.refused}")
        print(f"A: {result.answer}")
        if result.citations:
            print("Citations:")
            for c in result.citations:
                print(f"  - [{c.chunk_id}] {c.company} | {c.doc_type} | {c.date} | score={c.score}")
        print("-" * 80)


if __name__ == "__main__":
    main()
