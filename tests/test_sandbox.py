"""Tests for the sandbox (worktree + subprocess isolation) module."""

import sys

import pytest

from metascaffold.sandbox import Sandbox, SandboxResult


class TestSandbox:
    def test_execute_simple_command(self, tmp_path):
        """Should execute a simple command and capture output."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute("echo hello world")
        assert isinstance(result, SandboxResult)
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_capture_stderr(self, tmp_path):
        """Should capture stderr output."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute(f'{sys.executable} -c "import sys; sys.stderr.write(\'err msg\')"')
        assert "err msg" in result.stderr

    def test_timeout_kills_process(self, tmp_path):
        """Commands exceeding timeout should be killed."""
        sandbox = Sandbox(work_dir=str(tmp_path), default_timeout_seconds=2)
        result = sandbox.execute(f'{sys.executable} -c "import time; time.sleep(30)"')
        assert result.exit_code != 0
        assert result.timed_out is True

    def test_failed_command_returns_nonzero(self, tmp_path):
        """Failed commands should return non-zero exit code."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute(f'{sys.executable} -c "raise ValueError(\'boom\')"')
        assert result.exit_code != 0
        assert "boom" in result.stderr

    def test_result_includes_duration(self, tmp_path):
        """Result should include execution duration in ms."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute("echo fast")
        assert result.duration_ms >= 0

    def test_result_serializes_to_dict(self, tmp_path):
        """SandboxResult should be serializable to dict."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute("echo test")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "exit_code" in d
        assert "stdout" in d
        assert "stderr" in d
        assert "duration_ms" in d
