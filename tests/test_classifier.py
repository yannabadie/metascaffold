"""Tests for the System 1/2 classifier."""

from metascaffold.classifier import Classifier, ClassificationResult


class TestClassifier:
    def test_simple_read_is_system1(self):
        """Simple read operations should route to System 1."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Read",
            tool_input={"file_path": "/tmp/test.py"},
            context="Read a file to understand the code",
        )
        assert result.routing == "system1"
        assert result.confidence >= 0.8

    def test_multi_file_edit_is_system2(self):
        """Complex multi-file edits should route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Edit",
            tool_input={"file_path": "/src/auth.py"},
            context="Refactor the entire authentication system across 5 modules",
        )
        assert result.routing == "system2"
        assert result.confidence < 0.8

    def test_always_system2_tools(self):
        """Tools in always_system2_tools should always route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=["Write"])
        result = c.classify(
            tool_name="Write",
            tool_input={"file_path": "/tmp/new_file.py"},
            context="Create a small utility function",
        )
        assert result.routing == "system2"

    def test_destructive_bash_is_system2(self):
        """Destructive bash commands should route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Bash",
            tool_input={"command": "rm -rf /tmp/project"},
            context="Clean up temporary files",
        )
        assert result.routing == "system2"
        assert result.confidence < 0.8

    def test_classification_returns_all_fields(self):
        """ClassificationResult should have all required fields."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Bash",
            tool_input={"command": "ls"},
            context="List files",
        )
        assert isinstance(result, ClassificationResult)
        assert result.routing in ("system1", "system2")
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
        assert isinstance(result.signals, dict)

    def test_historical_success_rate_lowers_confidence(self):
        """Low historical success rate should lower confidence."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Edit",
            tool_input={"file_path": "/src/complex.py"},
            context="Fix the flaky test",
            historical_success_rate=0.3,
        )
        assert result.routing == "system2"
