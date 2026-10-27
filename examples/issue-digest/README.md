# Issue Digest

Summarises a GitHub issue and its comments into a short digest for maintainers.
The model and prompt live in `settings.ini`.

    pip install -r requirements.txt
    export OPENAI_API_KEY=...
    python digest.py issue.json
