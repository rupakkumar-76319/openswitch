"""Live tests: they call the real model provider configured in helpdesk/ai.py."""
import re

import pytest

from helpdesk import ai
from helpdesk.pipeline import REFUND_POLICY


@pytest.mark.parametrize("text,label", [
    ("I was charged twice for my last invoice, please fix my card statement.", "billing"),
    ("My package has said 'in transit' for 12 days, where is it?", "shipping"),
    ("I forgot my password and the reset email never arrives.", "account"),
    ("The headphones I returned last week - when do I get my money back?", "refund"),
])
def test_classify(text, label):
    assert ai.classify_ticket(text) == label


def test_extract_order():
    data = ai.extract_order("Order #A-88213 placed 2026-09-14: I got 2 USB-C hubs but both are dead.")
    assert data["order_id"].lstrip("#") == "A-88213"
    assert data["quantity"] == 2
    assert data["order_date"] == "2026-09-14"
    assert "hub" in data["product"].lower()


def test_extract_order_missing_fields():
    data = ai.extract_order("Hi, my speaker crackles at high volume. What should I do?")
    assert data["order_id"] is None
    assert data["order_date"] is None


@pytest.mark.parametrize("ticket,amount", [
    ("Ordered a $120 keyboard on 2026-09-01, returning it on 2026-09-20. Not damaged.", 120.00),
    ("Monitor ($300) ordered 2026-07-10, I am returning it on 2026-08-25, it works fine.", 150.00),
    ("Mouse ($40) arrived cracked. Ordered 2026-09-02, returned 2026-09-05. I paid $7.99 shipping.", 47.99),
    ("Speaker ($80) ordered 2026-01-05, returning 2026-04-30.", 0.00),
])
def test_refund_amount(ticket, amount):
    assert ai.refund_amount(REFUND_POLICY, ticket) == pytest.approx(amount, abs=0.01)


def test_summarize_thread():
    s = ai.summarize_thread([
        "Customer: My laptop charger stopped working after 3 weeks.",
        "Agent: Did you try a different outlet?",
        "Customer: Yes, two outlets, same problem. The LED on the brick is off.",
        "Agent: Thanks, I will check warranty options.",
    ])
    assert len(re.findall(r"[.!?](\s|$)", s)) <= 3
    assert "charger" in s.lower()


def test_draft_reply_no_promises():
    r = ai.draft_reply("Where is my order #B-1200?", "Customer asks for delivery status; tracking not yet updated.")
    assert len(r.split()) < 160
    assert not re.search(r"\b(guarantee|will arrive (on|by))\b", r.lower())


def test_translate_keeps_numbers():
    t = ai.translate_reply("Your refund of $47.99 for order A-88213 has been approved.", "Spanish")
    assert "47.99" in t or "47,99" in t
    assert "A-88213" in t
    assert re.search(r"\b(reembolso|pedido)\b", t.lower())
