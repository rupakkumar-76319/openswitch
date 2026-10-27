"""Live web research with Tavily: what the app pays today, what Nemotron costs on Nebius,
and the API differences the rewrite has to handle. Every fact keeps its source URL."""
from __future__ import annotations

import os

from tavily import TavilyClient

from . import llm

# Last-resort list prices (USD per 1M tokens) if search finds nothing for a model.
FALLBACK_SOURCE_PRICES = {
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60), "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40), "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5": (1.25, 10.00), "gpt-5-mini": (0.25, 2.00), "claude-sonnet-4-5": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00), "claude-haiku-4-5": (1.00, 5.00),
}


def _search(tv: TavilyClient, query: str, n: int = 5) -> list[dict]:
    r = tv.search(query=query, max_results=n, search_depth="advanced")
    return [{"url": x["url"], "title": x.get("title", ""), "content": x.get("content", "")[:1800]}
            for x in r.get("results", [])]


def _sources_block(results: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {r['url']}\n{r['content']}" for i, r in enumerate(results))


def run(source_models: list[str], providers: set[str], log=print) -> dict:
    tv = TavilyClient(os.environ["TAVILY_API_KEY"])
    out: dict = {"source_prices": {}, "nemotron_prices": {}, "notes": [], "queries": []}

    # 1. What does the app pay today?
    for model in source_models:
        q = f"{model} API pricing per 1M input tokens output tokens"
        out["queries"].append(q)
        res = _search(tv, q)
        log(f"tavily: {q}  ({len(res)} sources)")
        data = llm.ask_json([{"role": "user", "content":
            f"From these search results, find the current list price of the `{model}` API model "
            f"(standard tier, not batch or cached). Return JSON "
            '{"input": <USD per 1M input tokens>, "output": <USD per 1M output tokens>, "source": <index>} '
            'or {"input": null, "output": null, "source": null} if the exact model is not priced here.\n\n'
            + _sources_block(res)}], llm.SUPER, role="researcher")
        if data.get("input") and data.get("output"):
            src = res[int(data["source"])]["url"] if data.get("source") is not None else ""
            out["source_prices"][model] = {"input": float(data["input"]), "output": float(data["output"]), "url": src}
        elif model in FALLBACK_SOURCE_PRICES:
            i, o = FALLBACK_SOURCE_PRICES[model]
            out["source_prices"][model] = {"input": i, "output": o, "url": "fallback list price"}

    # 2. What does Nemotron cost on Nebius Token Factory?
    q = "Nebius Token Factory Nemotron 3 Nano Super Ultra price per 1M tokens"
    out["queries"].append(q)
    res = _search(tv, q, 6)
    log(f"tavily: {q}  ({len(res)} sources)")
    ids = list(llm.PRICES)
    data = llm.ask_json([{"role": "user", "content":
        "From these search results, find the Nebius Token Factory price (USD per 1M tokens) for each model id: "
        f"{ids}. Return JSON {{model_id: {{\"input\": x, \"output\": y, \"source\": index}}}} and leave out "
        "models you cannot find. Only use numbers that appear in the results.\n\n" + _sources_block(res)}],
        llm.SUPER, role="researcher")
    for mid, p in (data or {}).items():
        if mid in llm.PRICES and isinstance(p, dict) and p.get("input") is not None and p.get("output") is not None:
            llm.PRICES[mid] = (float(p["input"]), float(p["output"]))
            url = res[int(p["source"])]["url"] if p.get("source") is not None and int(p["source"]) < len(res) else ""
            out["nemotron_prices"][mid] = {"input": float(p["input"]), "output": float(p["output"]), "url": url}

    # 3. API differences the rewrite must respect.
    topics = ["Nebius Token Factory OpenAI compatible API base_url chat completions response_format json",
              "Nemotron 3 enable_thinking chat_template_kwargs reasoning on off"]
    if "anthropic" in providers:
        topics.append("convert Anthropic messages.create system parameter to OpenAI chat completions messages")
    if any(m.startswith(("o1", "o3", "o4")) for m in source_models):
        topics.append("OpenAI reasoning_effort parameter o3-mini chat completions")
    pool = []
    for q in topics:
        out["queries"].append(q)
        res = _search(tv, q, 4)
        log(f"tavily: {q}  ({len(res)} sources)")
        pool.extend(res)
    notes = llm.ask_json([{"role": "user", "content":
        "You are preparing a code migration from OpenAI/Anthropic SDK calls to NVIDIA Nemotron models served by "
        "Nebius Token Factory through the OpenAI-compatible API. From the search results, list the concrete facts "
        "a developer must apply (endpoint, auth, parameters that are or are not supported, how to switch Nemotron "
        "thinking on/off, JSON output, how Anthropic-only parameters map). Return JSON "
        '{"facts": [{"fact": "...", "source": <index>}]} with 4-10 items, only facts supported by the results.\n\n'
        + _sources_block(pool)}], llm.SUPER, role="researcher")
    if isinstance(notes, dict):
        notes = next((v for v in notes.values() if isinstance(v, list)), [])
    for n in notes if isinstance(notes, list) else []:
        try:
            url = pool[int(n["source"])]["url"]
        except (KeyError, ValueError, IndexError, TypeError):
            url = ""
        out["notes"].append({"fact": n.get("fact", ""), "url": url})
    return out
