# MetaScaffold v0.3 — Design Document

**Date**: 2026-03-03
**Status**: Approved
**Author**: Yann Abadie + Claude

## Goal

Evolve MetaScaffold from prompt-based confidence to **grounded, measurable** cognitive signals. Replace self-reported LLM confidence with entropy from token logprobs, add deterministic code verification before LLM-as-Judge, implement memory decay for Reflector rules, and introduce a 3-level adaptive compute system.

## Architectural Constraint

**Codex gpt-5.3 must be used for ALL LLM calls except when logprobs are needed.** Codex exec is the default backend (free via ChatGPT Plus). The OpenAI API is used only for the Classifier's entropy probe, via gpt-4.1-nano with `logprobs=True`.

## Four Improvement Axes

### Axis 1: Dual Backend LLM Client

**Problem**: v0.2's `LLMClient` only supports `codex exec`. We need direct OpenAI API access for logprobs (not available via codex exec).

**Solution**: Add `complete_with_logprobs()` to `LLMClient` using the `openai` Python SDK.

```
LLMClient
├── complete()              → codex exec (default, free, gpt-5.3)
│   ├── _complete_with_schema()  → --output-schema + -o
│   └── _complete_raw()          → stdout parsing
└── complete_with_logprobs() → OpenAI API (gpt-4.1-nano, logprobs=True)
    └── Returns LLMResponse + token_logprobs: list[dict]
```

**API key**: Read from `.env` file, env var `OPEN_API_KEY` (non-standard name, user's preference). Use `truststore` for corporate SSL proxy compatibility.

**Cost**: gpt-4.1-nano at $0.02/$0.15 per M tokens — negligible for short classifier prompts.

### Axis 2: Entropy-Based Routing

**Problem**: v0.2's classifier relies on LLM self-reported confidence, which is unreliable (models tend to be overconfident).

**Solution**: Use Shannon entropy from token logprobs to measure the model's **actual uncertainty** about the routing decision.

**How it works**:
1. Classifier sends task to gpt-4.1-nano via `complete_with_logprobs()`
2. Extract top-5 logprobs for the routing token (the token where the model decides "system1" vs "system2")
3. Compute Shannon entropy: `H = -sum(p * log2(p))` over the probability distribution
4. Low entropy (< threshold) = model is confident → trust its routing
5. High entropy (> threshold) = model is uncertain → escalate to System 2

**Key insight from research**: gpt-4.1-nano sometimes gives the wrong textual answer, but its entropy correctly signals uncertainty. A wrong answer with high entropy (p=0.73 system1 vs p=0.27 system2) is more informative than a "confident" self-report. The entropy signal matters more than textual correctness.

**Thresholds** (configurable):
- `entropy_threshold`: 0.5 (above = uncertain, force System 2)
- Falls back to codex exec heuristic classification if API call fails

### Axis 3: Deterministic Verifiers

**Problem**: v0.2's evaluator relies entirely on LLM-as-Judge or regex heuristics. LLMs can miss obvious errors that deterministic tools catch instantly.

**Solution**: Run 4 deterministic verification checks **before** LLM-as-Judge:

| Verifier | What it checks | When to run |
|----------|---------------|-------------|
| **AST parse** | Python syntax validity | Always (if .py files in output) |
| **Ruff** | Lint errors, style issues | Always (if .py files in output) |
| **pytest** | Test results | When test commands were executed |
| **mypy** | Type errors | On significant changes (System 2 only) |

**Integration point**: New `VerificationSuite` class in `verifiers.py`, called by `Evaluator.evaluate_async()` before the LLM evaluation step.

```
evaluate_async():
  1. Run deterministic verifiers → VerificationResult
  2. If any verifier fails with critical severity → return "backtrack" immediately
  3. If verifiers pass → proceed to LLM-as-Judge
  4. Merge verifier findings into LLM evaluation context
```

**Benefits**: Deterministic verifiers provide ground truth. If AST parse fails, there's no need to ask the LLM — the code is broken.

### Axis 4: Memory Management (Ebbinghaus Decay)

**Problem**: v0.2's Reflector accumulates rules/procedures indefinitely. Over time, irrelevant or outdated rules pollute the reflection context.

**Solution**: Add a `ReflectionMemory` with Ebbinghaus forgetting curve decay.

**Data model**:
```python
@dataclass
class ReflectionRule:
    content: str            # The rule text
    created_at: datetime    # When first learned
    last_reinforced: datetime  # Last time this rule was confirmed useful
    retention_strength: float  # 0.0 to 1.0, decays over time
    reinforcement_count: int   # How many times confirmed
    source_events: list[str]   # Telemetry event IDs that generated this rule
```

**Decay function**: `retention = e^(-t / (stability * reinforcement_factor))`
- `t` = time since last reinforcement (hours)
- `stability` = base half-life (configurable, default 168h = 1 week)
- `reinforcement_factor` = `1 + log(1 + reinforcement_count)`

**Operations**:
- **REINFORCE**: When a rule is confirmed useful, reset `last_reinforced` and increment count
- **PRUNE**: Remove rules with `retention_strength < 0.1`
- **MERGE**: Combine similar rules (via LLM) to reduce count
- **ARCHIVE**: Move low-retention rules to cold storage (SQLite) instead of deleting

**Storage**: JSON file `~/.metascaffold/reflection_memory.json` for hot rules, SQLite `cognitive.db` for archived rules.

### Axis 5: Adaptive Compute (System 1 / 1.5 / 2)

**Problem**: v0.2 has binary routing (System 1 or System 2). Some tasks need more than System 1 but less than full System 2.

**Solution**: Three-level compute allocation based on entropy + confidence:

| Level | When | Pipeline stages |
|-------|------|-----------------|
| **System 1** | Low entropy, high confidence | Classify → Execute → Evaluate → Reflect |
| **System 1.5** | Medium entropy or moderate complexity | Classify → Distill → Execute → Evaluate → Reflect |
| **System 2** | High entropy, low confidence, or forced | Classify → Distill → Plan → Execute → Evaluate → Reflect |

**Routing logic** (in Classifier):
```
if entropy > high_threshold:        → System 2
elif entropy > medium_threshold:    → System 1.5
else:                               → System 1

# Override: always_system2_tools → System 2 regardless
# Override: read-only tools → System 1 regardless
```

**Pipeline changes**: `PipelineState` gains a `compute_level` field (1, 1.5, or 2). Each stage checks compute_level to decide whether to run.

## Component Changes Summary

| File | Changes |
|------|---------|
| `llm_client.py` | Add `complete_with_logprobs()` using OpenAI SDK |
| `classifier.py` | Add entropy computation, 3-level routing |
| `evaluator.py` | Integrate `VerificationSuite` before LLM-as-Judge |
| `verifiers.py` | **New**: AST, Ruff, pytest, mypy verification suite |
| `reflector.py` | Use `ReflectionMemory` for decay-based rule management |
| `reflection_memory.py` | **New**: Ebbinghaus decay model and memory operations |
| `pipeline.py` | Add `compute_level` to `PipelineState`, System 1.5 logic |
| `config.py` | Add entropy thresholds, verifier settings, memory config |
| `server.py` | Wire new components, update hot-reload |
| `__init__.py` | Bump version to 0.3.0 |

## Non-Goals (v0.3)

- Multi-model evaluation ensemble (future)
- Automatic strategy selection from past plans (future)
- External tool integration beyond Ruff/mypy/pytest (future)
- Web-based dashboard for telemetry (future)

## Research References

- Shannon entropy for LLM uncertainty quantification
- Ebbinghaus forgetting curve (1885) adapted for AI memory systems
- SOFAI-LM dual-process architecture with adaptive compute
- Deterministic verification as ground truth anchor (2025 AI safety literature)
