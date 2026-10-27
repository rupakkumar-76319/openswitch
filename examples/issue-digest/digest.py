"""Turn a GitHub issue (title, body, comments) into a short markdown digest for maintainers."""
import sys

from llm_client import CONFIG, complete


def issue_to_text(issue: dict) -> str:
    lines = [f"Title: {issue['title']}", f"State: {issue['state']}", f"Labels: {', '.join(issue.get('labels', []))}",
             f"Reported by {issue['user']} at {issue['created_at']}", "", issue.get("body") or ""]
    for c in issue.get("comments", []):
        lines += ["", f"Comment by {c['user']} at {c['created_at']}:", c["body"]]
    return "\n".join(lines)


def digest(issue: dict) -> str:
    """Markdown digest with Status / Problem / Details / Next step sections."""
    result = complete(CONFIG.get("llm", "model"), CONFIG.get("prompt", "system"), issue_to_text(issue))
    return result.text


if __name__ == "__main__":
    import json
    print(digest(json.load(open(sys.argv[1], encoding="utf-8"))))
