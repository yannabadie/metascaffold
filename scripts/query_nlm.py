"""Query NotebookLM MetaScaffold_Core for improvement insights."""

import asyncio
import sys
import json

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from notebooklm import NotebookLMClient


async def query(question: str) -> str:
    client = await NotebookLMClient.from_storage(timeout=120.0)
    async with client:
        notebooks = await client.notebooks.list()
        matching = [nb for nb in notebooks if nb.title == "MetaScaffold_Core"]
        if not matching:
            return "ERROR: Notebook MetaScaffold_Core not found"
        result = await client.chat.ask(matching[0].id, question)
        return result.text


async def main():
    questions = [
        # 1. Classifier improvements
        "Based on the latest 2025-2026 research on metacognition in AI coding agents, what are the most effective signals and techniques for a System 1/System 2 classifier? How should it decide between fast heuristic processing and deliberate planning? What features matter most for routing accuracy? Include specific paper references.",

        # 2. Planner improvements
        "What are the state-of-the-art approaches (2025-2026) for task decomposition and planning in AI coding agents? How should a Planner module generate strategies with rollback plans? What techniques from papers like SWE-agent, Agentless, or AutoCodeRover are most effective?",

        # 3. Evaluator / self-reflection
        "What are the best 2025-2026 techniques for self-evaluation and auto-critique in AI agents? How should an Evaluator decide between pass/retry/backtrack/escalate verdicts? What patterns from Reflexion, LATS, or other papers improve correction loops?",

        # 4. Sandbox and execution
        "What are the latest best practices (2025-2026) for sandboxed code execution in AI coding agents? How do modern agents handle isolation, timeout, rollback, and git worktree management? What security and reliability patterns are recommended?",

        # 5. Overall architecture
        "Based on all sources in this notebook, what are the top 10 concrete improvements we should make to a metacognition plugin that implements System 1/2 dual-process cognitive architecture for Claude Code? The plugin has: Classifier, Planner, Sandbox, Evaluator, NotebookLM bridge, and Telemetry. Be very specific and actionable.",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*80}", flush=True)
        print(f"QUERY {i}/5", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Q: {q[:100]}...", flush=True)
        print(f"{'='*80}\n", flush=True)
        try:
            answer = await query(q)
            print(answer, flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        # Small delay between queries
        if i < len(questions):
            await asyncio.sleep(2)

    print(f"\n{'='*80}", flush=True)
    print("ALL QUERIES COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
