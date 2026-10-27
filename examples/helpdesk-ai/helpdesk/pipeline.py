"""Ticket pipeline: route, extract, decide refund, summarize, reply, translate."""
from helpdesk import ai

REFUND_POLICY = """\
- Items returned within 30 days of the order date get a full refund of the item price.
- Items returned between 31 and 60 days get 50% of the item price.
- After 60 days: no refund.
- Shipping fees ($7.99) are refunded only if the item arrived damaged.
"""


def handle(ticket: str, thread: list[str], language: str = "English") -> dict:
    out = {"queue": ai.classify_ticket(ticket), "order": ai.extract_order(ticket)}
    if out["queue"] == "refund":
        out["refund"] = ai.refund_amount(REFUND_POLICY, ticket)
    out["summary"] = ai.summarize_thread(thread)
    reply = ai.draft_reply(ticket, out["summary"])
    out["reply"] = reply if language == "English" else ai.translate_reply(reply, language)
    return out
