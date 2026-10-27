# Written by OpenSwitch: live smoke tests for the migrated LLM calls.
"""Smoke test for the LLM client after migration to Nemotron on Nebius.

This test performs a live call to the Nebius Token Factory API and validates the
Completion contract. It requires the NEBIUS_API_KEY environment variable to be set.
"""
import os
import json
import pytest
from llm_client import complete, Completion, CONFIG

# Skip the test if the API key is not configured, allowing the suite to run in CI without credentials.
@pytest.mark.skipif_no_nebius_key
def test_complete_live_smoke():
    """Verify that complete() returns a well‑formed Completion with non‑empty digest text."""
    # Use the exact model and system prompt the application uses in production.
    model = CONFIG.get("llm", "model")
    system = CONFIG.get("prompt", "system")

    # Realistic payload: two issues with titles, labels, bodies, and comments.
    issues = [
        {
            "issue": {
                "title": "Add dark mode toggle",
                "state": "open",
                "labels": ["enhancement", "ui"],
                "user": "carol_designer",
                "created_at": "2024-08-22T14:12:00Z",
                "body": "Users have asked for a dark mode option in the settings menu.",
                "comments": [],
            }
        },
        {
            "issue": {
                "title": "Fix login redirect loop",
                "state": "open",
                "labels": ["bug", "auth"],
                "user": "alice_dev",
                "created_at": "2024-09-25T08:30:00Z",
                "body": "After submitting credentials, the user is redirected back to login page indefinitely.",
                "comments": [
                    {"user": "bob_support", "created_at": "2024-09-25T09:00:00Z", "body": "Can you reproduce on incognito?"},
                    {"user": "alice_dev", "created_at": "2024-09-25T09:05:00Z", "body": "Yes, happens in Chrome and Firefox."},
                ],
            }
        },
    ]
    user = json.dumps(issues, ensure_ascii=False, indent=2)

    # Live call – no mocks.
    result = complete(model=model, system=system, user=user)

    # Contract assertions (robust to wording changes in the markdown digest).
    assert isinstance(result, Completion), "Expected a Completion dataclass instance"
    assert result.model == model, "Model name should match the configured model"
    assert isinstance(result.text, str) and result.text.strip(), "Digest text must be a non‑empty string"
    assert isinstance(result.input_tokens, int) and result.input_tokens > 0, "Input tokens should be a positive integer"
    assert isinstance(result.output_tokens, int) and result.output_tokens > 0, "Output tokens should be a positive integer"
    assert isinstance(result.seconds, float) and result.seconds > 0, "Latency should be a positive float"
    assert isinstance(result.raw, dict), "Raw response should be a dictionary"

    # Cost accounting – our model is in PRICES, so cost must be a non‑negative float.
    cost = result.cost
    assert isinstance(cost, float) and cost >= 0, "Cost should be a non‑negative float for known models"


# Helper marker: automatically skip if the environment variable is missing.
def pytest_configure(config):
    config.addinivalue_line("markers", "skipif_no_nebius_key: skip test when NEBIUS_API_KEY is not set")


@pytest.fixture(autouse=True)
def _skip_if_no_key(request):
    if request.node.get_closest_marker("skipif_no_nebius_key"):
        if not os.getenv("NEBIUS_API_KEY"):
            pytest.skip("NEBIUS_API_KEY not set – skipping live LLM call")
