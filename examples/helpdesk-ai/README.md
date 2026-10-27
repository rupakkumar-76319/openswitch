# HelpDesk AI

A small support console backend: routes tickets, extracts order data, applies the refund
policy, summarizes threads, drafts replies and translates them.

Runs on OpenAI (`gpt-4o`, `gpt-4o-mini`, `o3-mini`) and Anthropic (`claude-sonnet-4-5`).

    pip install -r requirements.txt
    export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...
    pytest
