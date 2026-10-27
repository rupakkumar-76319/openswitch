# Written by OpenSwitch: live smoke tests for the migrated LLM calls.
"""Smoke tests for the digest function after migration to Nemotron on Nebius."""
import pytest
from digest import digest


class TestDigestSmoke:
    """Live tests that exercise the real LLM pipeline."""

    def test_digest_crash_large_csv(self):
        """Digest a realistic bug report with comments."""
        result = digest(
            **{
                "issue": {
                    "title": "Crash when opening large CSV file",
                    "state": "open",
                    "labels": ["bug", "high-priority"],
                    "user": "alice_dev",
                    "created_at": "2024-09-10T14:32:00Z",
                    "body": "When I try to open a CSV larger than 200MB, the application throws an IndexOutOfRangeException and exits.",
                    "comments": [
                        {
                            "user": "bob_maintainer",
                            "created_at": "2024-09-11T09:05:00Z",
                            "body": "Can you share the exact file and stack trace?",
                        },
                        {
                            "user": "alice_dev",
                            "created_at": "2024-09-11T10:20:00Z",
                            "body": "Attached is the file and trace: ...",
                        },
                    ],
                }
            }
        )

        # Contract: non-empty string containing the four required headings
        assert isinstance(result, str), "digest must return a string"
        assert result.strip(), "digest output must not be empty"

        # Required level-2 headings (case-insensitive, flexible spacing)
        lower = result.lower()
        assert "## status" in lower or "## status" in lower.replace("**", ""), "Missing 'Status' heading"
        assert "## problem" in lower, "Missing 'Problem' heading"
        assert "## details" in lower, "Missing 'Details' heading"
        assert "## next step" in lower, "Missing 'Next step' heading"

        # Each section should have some content after its heading
        for heading in ["status", "problem", "details", "next step"]:
            idx = lower.find(f"## {heading}")
            assert idx >= 0, f"Heading '{heading}' not found"
            # Look for non-whitespace after the heading line
            after = result[idx:].split("\n", 1)
            if len(after) > 1:
                assert after[1].strip(), f"Section '{heading}' has no content"

    def test_digest_typo_documentation(self):
        """Digest a closed documentation issue with no comments."""
        result = digest(
            **{
                "issue": {
                    "title": "Typo in documentation",
                    "state": "closed",
                    "labels": ["documentation", "good-first-issue"],
                    "user": "doc_writer",
                    "created_at": "2024-08-01T08:00:00Z",
                    "body": "On page 12 of the README, the word 'recieve' should be 'receive'.",
                    "comments": [],
                }
            }
        )

        assert isinstance(result, str), "digest must return a string"
        assert result.strip(), "digest output must not be empty"

        lower = result.lower()
        assert "## status" in lower, "Missing 'Status' heading"
        assert "## problem" in lower, "Missing 'Problem' heading"
        assert "## details" in lower, "Missing 'Details' heading"
        assert "## next step" in lower, "Missing 'Next step' heading"

        # Verify state and labels appear in Status section
        assert "closed" in lower, "Status should reflect closed state"
        assert "documentation" in lower, "Status should include documentation label"
        assert "good-first-issue" in lower, "Status should include good-first-issue label"
        assert "doc_writer" in lower, "Status should include reporter"

        # Problem should be concise (one sentence, <= 40 words approx)
        problem_idx = lower.find("## problem")
        details_idx = lower.find("## details")
        problem_section = result[problem_idx:details_idx] if details_idx > problem_idx else result[problem_idx:]
        problem_words = problem_section.split()
        # Allow some flexibility but ensure it's not a wall of text
        assert len(problem_words) < 100, "Problem section should be concise"
