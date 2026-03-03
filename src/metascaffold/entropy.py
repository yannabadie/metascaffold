"""Shannon entropy computation from token logprobs.

Pure-function utility module used by the Classifier to measure model
uncertainty at routing-decision token positions.  The OpenAI API returns
logprobs as natural logarithms; this module converts them to probabilities
and computes entropy in bits (log base 2).
"""

from __future__ import annotations

import math

# Keywords that signal a routing decision token.
_ROUTING_KEYWORDS = frozenset({"system", "system1", "system2", "simple", "complex"})


def compute_entropy(logprobs: list[dict]) -> float:
    """Compute Shannon entropy from a list of top-k logprob dicts.

    Parameters
    ----------
    logprobs:
        List of ``{"token": str, "logprob": float}`` dicts representing
        the top-k logprobs at a single token position.  Logprobs are in
        natural-log scale (as returned by the OpenAI API).

    Returns
    -------
    float
        Shannon entropy in **bits** (log base 2).  Returns 0.0 for empty
        input or when only a single valid token remains after filtering.
    """
    if not logprobs:
        return 0.0

    # Convert logprobs (natural log) to raw probabilities, skipping -inf.
    probs: list[float] = []
    for entry in logprobs:
        lp = entry.get("logprob", float("-inf"))
        if math.isinf(lp) and lp < 0:
            continue
        probs.append(math.exp(lp))

    if not probs:
        return 0.0

    # Normalize so probabilities sum to 1.0.
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]

    # Shannon entropy: H = -sum(p * log2(p)) for p > 0
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def find_routing_token_entropy(token_logprobs: list[dict]) -> float | None:
    """Scan a token stream for a routing-decision token and return its entropy.

    Parameters
    ----------
    token_logprobs:
        List of token entries as returned by ``complete_with_logprobs()``.
        Each entry has ``token``, ``logprob``, and ``top_logprobs`` keys.

    Returns
    -------
    float | None
        Shannon entropy (bits) at the first routing token found, or
        ``None`` if no routing keyword appears in the stream.
    """
    if not token_logprobs:
        return None

    for entry in token_logprobs:
        token_text = entry.get("token", "").lower().strip()
        if token_text in _ROUTING_KEYWORDS:
            top = entry.get("top_logprobs")
            if top:
                return compute_entropy(top)
            return None

    return None
