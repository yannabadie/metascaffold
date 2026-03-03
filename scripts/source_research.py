"""Source research papers and repos for MetaScaffold knowledge base.

Curates SOTA papers (2025-2026) on LLM metacognition, self-correction,
cognitive architectures, and dual-process thinking. Also includes foundational
papers (2022-2024) and defines Deep Research queries for AI Ultra.

Usage: uv run python scripts/source_research.py
"""

import json
from pathlib import Path

SOURCES = {
    # =====================================================================
    # FOUNDATIONAL PAPERS (2022-2024) — Classic references
    # =====================================================================
    "foundational_papers": [
        {
            "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
            "url": "https://arxiv.org/abs/2303.11366",
            "topics": ["reflexion", "self-correction", "verbal reinforcement"],
        },
        {
            "title": "Self-Refine: Iterative Refinement with Self-Feedback",
            "url": "https://arxiv.org/abs/2303.17651",
            "topics": ["self-refinement", "iterative feedback"],
        },
        {
            "title": "Tree of Thoughts: Deliberate Problem Solving with LLMs",
            "url": "https://arxiv.org/abs/2305.10601",
            "topics": ["system2 thinking", "deliberation", "backtracking"],
        },
        {
            "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
            "url": "https://arxiv.org/abs/2210.03629",
            "topics": ["reasoning-action loop", "agentic systems"],
        },
        {
            "title": "Language Agent Tree Search (LATS)",
            "url": "https://arxiv.org/abs/2310.04406",
            "topics": ["tree search", "planning", "self-evaluation"],
        },
        {
            "title": "Cognitive Architectures for Language Agents (CoALA)",
            "url": "https://arxiv.org/abs/2309.02427",
            "topics": ["cognitive architecture", "memory", "metacognition"],
        },
        {
            "title": "Self-RAG: Learning to Retrieve, Generate, and Critique",
            "url": "https://arxiv.org/abs/2310.11511",
            "topics": ["self-reflection", "RAG", "critique"],
        },
        {
            "title": "LLM Self-Correction Is Possible When Done Right",
            "url": "https://arxiv.org/abs/2406.01297",
            "topics": ["self-correction", "verification"],
        },
        {
            "title": "Dualformer: Controllable Fast and Slow Thinking",
            "url": "https://arxiv.org/abs/2410.09918",
            "topics": ["dual process", "auto-mode", "adaptive reasoning"],
        },
    ],

    # =====================================================================
    # SOTA PAPERS (2025-2026) — Cutting-edge metacognition research
    # =====================================================================
    "sota_papers": [
        # --- Metacognition & Self-Evaluation ---
        {
            "title": "Emergent Introspective Awareness in Large Language Models",
            "url": "https://arxiv.org/abs/2601.01828",
            "date": "2025-10",
            "topics": ["introspection", "self-awareness", "mechanistic"],
            "key_insight": "Claude-class models have genuine functional introspection via detectable internal circuits",
        },
        {
            "title": "LLMs Are Capable of Metacognitive Monitoring and Control of Internal Activations",
            "url": "https://arxiv.org/abs/2505.13763",
            "date": "2025-05",
            "topics": ["metacognitive space", "activation monitoring", "neurofeedback"],
            "key_insight": "Defines empirical limits of LLM metacognition - low-dimensional metacognitive space",
        },
        {
            "title": "Evidence for Limited Metacognition in LLMs",
            "url": "https://arxiv.org/abs/2509.21545",
            "date": "2025-09",
            "topics": ["calibration", "confidence", "behavioral tests"],
            "key_insight": "Post-training develops metacognitive abilities; raw self-reports unreliable",
        },
        {
            "title": "Fine-Tuning Language Models to Know What They Know (ESMA)",
            "url": "https://arxiv.org/abs/2602.02605",
            "date": "2026-02",
            "topics": ["metacognitive alignment", "dual-prompt", "self-knowledge"],
            "key_insight": "ESMA dual-prompt method binds internal knowledge to explicit behaviors",
        },
        {
            "title": "Gnosis: Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits",
            "url": "https://arxiv.org/abs/2512.20578",
            "date": "2025-12",
            "topics": ["failure prediction", "hidden state decoding", "lightweight"],
            "key_insight": "5M-param overlay outperforms 8B external judge; zero-shot to partial generations",
        },
        {
            "title": "Self-Interpretability: LLMs Can Describe Complex Internal Processes",
            "url": "https://arxiv.org/abs/2505.17120",
            "date": "2025-05",
            "topics": ["introspection", "self-reporting", "interpretability"],
            "key_insight": "Self-reporting channel has genuine informational content, not just confabulation",
        },
        {
            "title": "Feeling the Strength but Not the Source: Partial Introspection in LLMs",
            "url": "https://arxiv.org/abs/2512.12411",
            "date": "2025-12",
            "topics": ["partial introspection", "confidence magnitude", "causal limits"],
            "key_insight": "LLMs can sense confidence magnitude but not semantic source — design for this",
        },

        # --- Self-Correction ---
        {
            "title": "Self-Correction Bench: Uncovering the Self-Correction Blind Spot in LLMs",
            "url": "https://arxiv.org/abs/2507.02778",
            "date": "2025-07",
            "topics": ["self-correction blind spot", "external framing"],
            "key_insight": "LLMs cannot correct own errors but can correct identical errors from external sources",
        },
        {
            "title": "SELF-THOUGHT: Beyond Output Critique via Task Distillation",
            "url": "https://arxiv.org/abs/2602.00871",
            "date": "2026-01",
            "topics": ["task distillation", "abstraction before correction"],
            "key_insight": "Abstract task structure first, then evaluate — superior to naive self-critique",
        },
        {
            "title": "PAG: Policy as Generative Verifier — Multi-Turn Reinforced Self-Correction",
            "url": "https://arxiv.org/abs/2506.10406",
            "date": "2025-06",
            "topics": ["selective revision", "verify-then-revise", "RL"],
            "key_insight": "Revise only when verification detects error — prevents model collapse",
        },
        {
            "title": "SETS: Self-Enhanced Test-Time Scaling via Self-Verification",
            "url": "https://arxiv.org/abs/2501.19306",
            "date": "2025-01",
            "topics": ["test-time compute", "self-verification", "scaling"],
            "key_insight": "Unified sampling + verification + correction achieves superior scaling laws",
        },
        {
            "title": "ReVISE: Refine via Intrinsic Self-Verification",
            "url": "https://arxiv.org/abs/2502.14565",
            "date": "2025-02",
            "topics": ["confidence-aware decoding", "trajectory rethinking"],
            "key_insight": "Confidence-aware decoding mechanism for natural test-time scaling",
        },

        # --- Dual-Process / System 1-2 ---
        {
            "title": "SOFAI-LM: Language Models Coupled with Metacognition Can Outperform Reasoning Models",
            "url": "https://arxiv.org/abs/2508.17959",
            "date": "2025-08",
            "topics": ["SOFAI", "dual-process", "3-layer metacognition"],
            "key_insight": "3-layer metacognition (eval/correct/improve): 94% of LRM at 75% lower cost",
        },
        {
            "title": "DPT-Agent: Leveraging Dual Process Theory in Language Agent Framework",
            "url": "https://arxiv.org/abs/2502.11882",
            "date": "2025-02",
            "topics": ["dual process agent", "async reflection", "FSM"],
            "key_insight": "Asynchronous reflection — metacognition runs parallel to task execution",
        },
        {
            "title": "From System 1 to System 2: A Survey of Reasoning Large Language Models",
            "url": "https://arxiv.org/abs/2502.17419",
            "date": "2025-02",
            "topics": ["survey", "reasoning taxonomy", "System 1/2"],
            "key_insight": "Comprehensive taxonomy mapping S1/S2 transition techniques (o1, R1, etc.)",
        },
        {
            "title": "S1-Bench: Evaluating System 1 Thinking in Large Reasoning Models",
            "url": "https://arxiv.org/abs/2504.10368",
            "date": "2025-04",
            "topics": ["overthinking", "adaptive stopping", "S1-bench"],
            "key_insight": "Models produce 15.5x longer output than needed — need 'stop reasoning' signal",
        },

        # --- Self-Evolution & Agentic Reflection ---
        {
            "title": "Truly Self-Improving Agents Require Intrinsic Metacognitive Learning (ICML 2025)",
            "url": "https://arxiv.org/abs/2505.00020",
            "date": "2025-07",
            "topics": ["metacognitive learning", "self-assessment", "ICML"],
            "key_insight": "Three required components: metacognitive knowledge, planning, and evaluation",
        },
        {
            "title": "Towards Agentic Self-Learning LLMs (ICLR 2026)",
            "url": "https://arxiv.org/abs/2510.14253",
            "date": "2026-01",
            "topics": ["self-learning", "generative reward model", "closed-loop"],
            "key_insight": "Self-generated reward model enables zero-data self-improvement loop",
        },
        {
            "title": "MARS: Metacognitive Agent Reflective Self-improvement",
            "url": "https://arxiv.org/abs/2601.11974",
            "date": "2026-01",
            "topics": ["principle reflection", "procedural reflection", "single-cycle"],
            "key_insight": "Dual reflection (principle + procedural) in single cycle — extract rules + strategies",
        },
        {
            "title": "A Comprehensive Survey of Self-Evolving AI Agents",
            "url": "https://arxiv.org/abs/2508.07407",
            "date": "2025-08",
            "topics": ["self-evolution survey", "intra/inter test-time"],
            "key_insight": "Taxonomy: intra-test-time (within session) vs inter-test-time (across sessions)",
        },
        {
            "title": "SEAL: Self-Adapting Language Models",
            "url": "https://arxiv.org/abs/2506.10943",
            "date": "2025-06",
            "topics": ["self-adaptation", "self-generated training", "RL"],
            "key_insight": "LLMs generate own fine-tuning data for long-term metacognitive improvement",
        },

        # --- Confidence & Calibration ---
        {
            "title": "Uncertainty Quantification and Confidence Calibration in LLMs (KDD 2025)",
            "url": "https://arxiv.org/abs/2503.15850",
            "date": "2025-03",
            "topics": ["uncertainty taxonomy", "4-dimensional UQ", "KDD"],
            "key_insight": "4D uncertainty framework: input/reasoning/parameter/prediction — each needs different monitoring",
        },
        {
            "title": "The Confidence Dichotomy: Calibration in Tool-Use Agents",
            "url": "https://arxiv.org/abs/2601.07264",
            "date": "2026-01",
            "topics": ["tool-aware calibration", "overconfidence", "code verification"],
            "key_insight": "Evidence tools cause overconfidence, code interpreters mitigate it — tool-aware calibration needed",
        },
        {
            "title": "CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration?",
            "url": "https://arxiv.org/abs/2510.24505",
            "date": "2025-10",
            "topics": ["critique calibration", "natural language", "generalization"],
            "key_insight": "Natural language critique > numerical confidence for calibration — outperforms GPT-4o teacher",
        },
    ],

    # =====================================================================
    # GitHub repos — implementations
    # =====================================================================
    "github_repos": [
        {
            "name": "Reflexion",
            "url": "https://github.com/noahshinn/reflexion",
            "description": "Reference implementation of the Reflexion framework",
        },
        {
            "name": "Tree of Thoughts",
            "url": "https://github.com/princeton-nlp/tree-of-thought-llm",
            "description": "Official Tree of Thoughts implementation",
        },
        {
            "name": "Self-RAG",
            "url": "https://github.com/AkariAsai/self-rag",
            "description": "Self-Reflective RAG implementation",
        },
        {
            "name": "LATS",
            "url": "https://github.com/andyz245/LanguageAgentTreeSearch",
            "description": "Language Agent Tree Search implementation",
        },
        {
            "name": "LangGraph",
            "url": "https://github.com/langchain-ai/langgraph",
            "description": "Framework for building agentic workflows with cycles",
        },
    ],

    # =====================================================================
    # Deep Research queries — AI Ultra (mode=deep, source=web)
    # Optimized for SOTA March 2026 knowledge
    # =====================================================================
    "deep_research_queries": [
        {
            "query": "latest 2025 2026 research LLM metacognition self-awareness introspection emergent self-evaluation internal circuits Gnosis ESMA SOFAI-LM state of the art",
            "purpose": "SOTA 2025-2026 metacognition — introspection, self-awareness circuits, metacognitive alignment",
        },
        {
            "query": "latest 2025 2026 LLM self-correction limitations blind spot SELF-THOUGHT task distillation PAG selective revision verify-then-revise beyond DeepSeek-R1",
            "purpose": "SOTA 2025-2026 self-correction — overcoming blind spots, task abstraction, selective revision",
        },
        {
            "query": "2025 2026 dual process System 1 System 2 cognitive architecture AI agent SOFAI-LM DPT-Agent Dualformer S1-Bench adaptive reasoning overthinking",
            "purpose": "SOTA 2025-2026 dual-process — fast/slow switching, metacognitive routing, adaptive stopping",
        },
        {
            "query": "2025 2026 MARS metacognitive agent self-improvement self-evolution agentic self-learning ICLR 2026 ICML 2025 principle procedural reflection SEAL",
            "purpose": "SOTA 2025-2026 self-evolution — single-cycle improvement, dual reflection, closed-loop learning",
        },
        {
            "query": "2025 2026 LLM confidence calibration tool-use coding agents uncertainty quantification CritiCal critique-based calibration test-time compute scaling Claude",
            "purpose": "SOTA 2025-2026 confidence/uncertainty — tool-aware calibration for code agents",
        },
    ],
}


def save_source_list(output_path: Path = Path("docs/research_sources.json")) -> None:
    """Save the curated source list as JSON for reference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(SOURCES, f, indent=2, ensure_ascii=False)

    n_foundational = len(SOURCES["foundational_papers"])
    n_sota = len(SOURCES["sota_papers"])
    n_repos = len(SOURCES["github_repos"])
    n_queries = len(SOURCES["deep_research_queries"])

    print(f"Source list saved to {output_path}")
    print(f"  Foundational papers: {n_foundational}")
    print(f"  SOTA papers (2025-2026): {n_sota}")
    print(f"  GitHub repos: {n_repos}")
    print(f"  Deep Research queries: {n_queries}")
    print(f"  Total URL sources: {n_foundational + n_sota + n_repos}")


if __name__ == "__main__":
    save_source_list()
