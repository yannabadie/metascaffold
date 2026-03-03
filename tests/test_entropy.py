"""Tests for Shannon entropy computation from token logprobs."""

import math

import pytest

from metascaffold.entropy import compute_entropy, find_routing_token_entropy


class TestComputeEntropy:
    def test_single_token_zero_entropy(self):
        """A single token with logprob=0.0 (p=1.0) has zero entropy."""
        logprobs = [{"token": "hello", "logprob": 0.0}]
        result = compute_entropy(logprobs)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_two_tokens_equal_probability(self):
        """Two tokens with equal probability yield entropy of 1.0 bit."""
        lp = math.log(0.5)  # natural log of 0.5
        logprobs = [
            {"token": "yes", "logprob": lp},
            {"token": "no", "logprob": lp},
        ]
        result = compute_entropy(logprobs)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_high_confidence_low_entropy(self):
        """p=0.95 vs p=0.05 should give low entropy (< 0.4 bits)."""
        logprobs = [
            {"token": "system1", "logprob": math.log(0.95)},
            {"token": "system2", "logprob": math.log(0.05)},
        ]
        result = compute_entropy(logprobs)
        assert result < 0.4

    def test_uncertain_high_entropy(self):
        """p=0.6 vs p=0.4 should give high entropy (> 0.9 bits)."""
        logprobs = [
            {"token": "system1", "logprob": math.log(0.6)},
            {"token": "system2", "logprob": math.log(0.4)},
        ]
        result = compute_entropy(logprobs)
        assert result > 0.9

    def test_empty_logprobs_returns_zero(self):
        """Empty logprobs list should return 0.0."""
        assert compute_entropy([]) == 0.0

    def test_handles_negative_infinity_logprob(self):
        """Tokens with -inf logprob should be skipped gracefully."""
        logprobs = [
            {"token": "hello", "logprob": 0.0},
            {"token": "world", "logprob": float("-inf")},
        ]
        result = compute_entropy(logprobs)
        # Only one valid token (p=1.0), so entropy should be 0.0
        assert result == pytest.approx(0.0, abs=1e-9)


class TestFindRoutingTokenEntropy:
    def test_finds_system_token(self):
        """Should find a 'system' routing token and compute its entropy."""
        token_logprobs = [
            {
                "token": "The",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "The", "logprob": -0.1},
                    {"token": "A", "logprob": -2.5},
                ],
            },
            {
                "token": "system",
                "logprob": math.log(0.7),
                "top_logprobs": [
                    {"token": "system", "logprob": math.log(0.7)},
                    {"token": "simple", "logprob": math.log(0.3)},
                ],
            },
        ]
        result = find_routing_token_entropy(token_logprobs)
        assert result is not None
        # H(0.7, 0.3) > 0.5 bits
        assert result > 0.5

    def test_returns_none_when_no_routing_token(self):
        """Should return None when no routing keyword is found."""
        token_logprobs = [
            {
                "token": "hello",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "hello", "logprob": -0.1},
                ],
            },
        ]
        result = find_routing_token_entropy(token_logprobs)
        assert result is None

    def test_returns_none_for_empty_list(self):
        """Should return None for an empty token list."""
        assert find_routing_token_entropy([]) is None
