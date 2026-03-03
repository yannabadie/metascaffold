"""Tests for the deterministic verification suite."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from metascaffold.verifiers import (
    VerifierResult,
    ast_verify,
    ruff_verify,
    pytest_verify,
    mypy_verify,
    VerificationSuite,
)


class TestAstVerify:
    """Tests for AST syntax verification (real code, no mocks)."""

    def test_valid_python_passes(self):
        result = ast_verify("def foo():\n    return 42\n")
        assert result.passed is True
        assert result.verifier == "ast"

    def test_syntax_error_fails(self):
        result = ast_verify("def foo(\n")
        assert result.passed is False
        assert result.severity == "critical"
        assert "SyntaxError" in result.detail

    def test_empty_string_passes(self):
        result = ast_verify("")
        assert result.passed is True


class TestRuffVerify:
    """Tests for Ruff linting verification (subprocess mocked)."""

    @patch("metascaffold.verifiers.subprocess.run")
    def test_ruff_clean_passes(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = ruff_verify("clean_file.py")
        assert result.passed is True
        assert result.verifier == "ruff"

    @patch("metascaffold.verifiers.subprocess.run")
    def test_ruff_errors_fail(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="clean_file.py:1:1: F401 'os' imported but unused",
            stderr="",
        )
        result = ruff_verify("clean_file.py")
        assert result.passed is False
        assert result.severity == "warning"
        assert "F401" in result.detail

    @patch("metascaffold.verifiers.subprocess.run")
    def test_ruff_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        result = ruff_verify("clean_file.py")
        assert result.passed is True
        assert result.skipped is True


class TestPytestVerify:
    """Tests for pytest verification (subprocess mocked)."""

    @patch("metascaffold.verifiers.subprocess.run")
    def test_all_tests_pass(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="5 passed in 1.23s", stderr=""
        )
        result = pytest_verify("tests/")
        assert result.passed is True
        assert result.verifier == "pytest"

    @patch("metascaffold.verifiers.subprocess.run")
    def test_test_failures(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="3 passed, 2 failed",
            stderr="FAILED test_auth - AssertionError",
        )
        result = pytest_verify("tests/")
        assert result.passed is False
        assert result.severity == "critical"


class TestMypyVerify:
    """Tests for mypy type-checking verification (subprocess mocked)."""

    @patch("metascaffold.verifiers.subprocess.run")
    def test_mypy_clean(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Success: no issues found", stderr=""
        )
        result = mypy_verify("module.py")
        assert result.passed is True
        assert result.verifier == "mypy"

    @patch("metascaffold.verifiers.subprocess.run")
    def test_mypy_errors(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="module.py:10: error: Incompatible types",
            stderr="",
        )
        result = mypy_verify("module.py")
        assert result.passed is False
        assert result.severity == "warning"


class TestVerificationSuite:
    """Tests for the high-level VerificationSuite orchestrator."""

    def test_suite_with_passing_code(self):
        suite = VerificationSuite()
        results = suite.verify_code("def foo():\n    return 42\n")
        assert len(results) >= 1
        assert results[0].verifier == "ast"
        assert results[0].passed is True

    def test_suite_with_broken_syntax(self):
        suite = VerificationSuite()
        results = suite.verify_code("def foo(\n")
        assert len(results) >= 1
        assert results[0].verifier == "ast"
        assert results[0].passed is False

    def test_suite_has_critical_failures(self):
        suite = VerificationSuite()
        results = suite.verify_code("def foo(\n")
        assert suite.has_critical_failures(results) is True

    def test_suite_no_critical_on_valid_code(self):
        suite = VerificationSuite()
        results = suite.verify_code("def foo():\n    return 42\n")
        assert suite.has_critical_failures(results) is False
