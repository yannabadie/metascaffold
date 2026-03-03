"""Sandbox — isolated execution via restricted subprocesses.

Provides timeout, stderr/stdout capture, and duration tracking.
Git worktree integration is handled at the MCP server level.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    worktree_path: str | None = None
    worktree_branch: str | None = None

    def to_dict(self) -> dict:
        d = {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
        }
        if self.worktree_path:
            d["worktree_path"] = self.worktree_path
            d["worktree_branch"] = self.worktree_branch
        return d


class Sandbox:
    """Restricted subprocess executor with timeout and output capture."""

    def __init__(
        self,
        work_dir: str = ".",
        default_timeout_seconds: int = 30,
    ):
        self.work_dir = work_dir
        self.default_timeout_seconds = default_timeout_seconds

    def execute(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> SandboxResult:
        timeout = timeout_seconds or self.default_timeout_seconds
        start = time.monotonic()
        timed_out = False

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -1
            stdout = ""
            stderr = f"Command timed out after {timeout}s"

        duration_ms = int((time.monotonic() - start) * 1000)

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
