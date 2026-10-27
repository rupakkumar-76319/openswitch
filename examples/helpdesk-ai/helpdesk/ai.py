"""LLM features for the HelpDesk AI support console.

Every function here is called from the ticket pipeline (helpdesk/pipeline.py).
"""
import json
import os

import anthropic
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

CATEGORIES = ["billing", "shipping", "technical", "account", "refund"]


def classify_ticket(text: str) -> str:
    """Route a ticket to one queue."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You route customer support tickets. "
             f"Answer with exactly one word from: {', '.join(CATEGORIES)}."},
            {"role": "user", "content": text},
        ],
    )
    label = resp.choices[0].message.content.strip().lower().strip(".")
    return label if label in CATEGORIES else "technical"


def extract_order(text: str) -> dict:
    """Pull structured order fields out of a free-text ticket."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Extract order details from the ticket. Return JSON with keys "
             "order_id (string or null), product (string or null), quantity (integer or null), "
             "order_date (YYYY-MM-DD or null). Do not guess values that are not in the text."},
            {"role": "user", "content": text},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def refund_amount(policy: str, ticket: str) -> float:
    """Apply the refund policy to a ticket and return the refund in dollars."""
    resp = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="medium",
        messages=[
            {"role": "user", "content": f"Refund policy:\n{policy}\n\nTicket:\n{ticket}\n\n"
             "Apply the policy exactly. Reply with only the refund amount in dollars as a number, e.g. 12.50"},
        ],
    )
    raw = resp.choices[0].message.content.strip().replace("$", "").replace(",", "")
    return round(float(raw.split()[0]), 2)


def summarize_thread(messages: list[str]) -> str:
    """Two-sentence handoff summary of a long ticket thread for the next agent."""
    thread = "\n".join(f"- {m}" for m in messages)
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[
            {"role": "system", "content": "Summarize this support thread in at most two sentences for the next "
             "agent. Include the customer's problem, what was already tried, and what is still open."},
            {"role": "user", "content": thread},
        ],
    )
    return resp.choices[0].message.content.strip()


def draft_reply(ticket: str, summary: str, tone: str = "friendly") -> str:
    """Draft a customer-facing reply an agent can send after review."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages=[
            {"role": "system", "content": f"You are a {tone} support agent for an online electronics store. "
             "Write a short reply (under 120 words). Never promise refunds or dates you were not given."},
            {"role": "user", "content": f"Ticket:\n{ticket}\n\nInternal summary:\n{summary}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def translate_reply(reply: str, language: str) -> str:
    """Translate an approved reply into the customer's language."""
    msg = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        system="Translate the support reply. Keep names, order numbers and amounts unchanged. "
               "Output only the translation.",
        messages=[{"role": "user", "content": f"Target language: {language}\n\n{reply}"}],
    )
    return msg.content[0].text.strip()
