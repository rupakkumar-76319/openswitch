"""Nebius Token Factory client with a per-role usage ledger.

Every call OpenSwitch makes is tagged with a role (planner, researcher, judge, ...),
so the final report can show exactly which Nemotron model did which job and what it cost.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"

SHORT = {NANO: "Nano 30B", LIGHTNING: "3.5 Lightning", SUPER: "Super 120B", ULTRA: "Ultra 550B"}

# USD per 1M tokens (input, output). Refreshed at runtime by research.py via Tavily;
# these are the fallbacks used when the web lookup fails.
PRICES = {
    NANO: (0.06, 0.24),
    LIGHTNING: (0.06, 0.24),
    SUPER: (0.30, 0.90),
    ULTRA: (1.00, 3.00),
}


@dataclass
class Call:
    role: str
    model: str
    thinking: bool
    prompt_tokens: int
    completion_tokens: int
    seconds: float

    @property
    def cost(self) -> float:
        pi, po = PRICES[self.model]
        return (self.prompt_tokens * pi + self.completion_tokens * po) / 1e6


@dataclass
class Ledger:
    calls: list[Call] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, call: Call):
        with self.lock:
            self.calls.append(call)

    def by_role(self) -> dict:
        out: dict = {}
        for c in self.calls:
            r = out.setdefault((c.role, c.model), {"calls": 0, "tokens": 0, "cost": 0.0})
            r["calls"] += 1
            r["tokens"] += c.prompt_tokens + c.completion_tokens
            r["cost"] += c.cost
        return out

    @property
    def total(self) -> float:
        return sum(c.cost for c in self.calls)


LEDGER = Ledger()
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
            api_key=os.environ["NEBIUS_API_KEY"],
            timeout=180,
            max_retries=3,
        )
    return _client


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    model: str
    thinking: bool

    @property
    def cost(self) -> float:
        pi, po = PRICES[self.model]
        return (self.prompt_tokens * pi + self.completion_tokens * po) / 1e6


def chat(messages: list[dict], model: str, role: str, thinking: bool = False,
         max_tokens: int = 4096, temperature: float | None = None, json_mode: bool = False) -> Reply:
    kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    t = time.time()
    r = client().chat.completions.create(**kwargs)
    dt = time.time() - t
    u = r.usage
    LEDGER.add(Call(role, model, thinking, u.prompt_tokens, u.completion_tokens, dt))
    return Reply(r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, dt, model, thinking)


def parse_json(text: str):
    """Parse the first JSON object/array in a model reply (tolerates code fences and prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def ask_json(messages: list[dict], model: str, role: str, thinking: bool = False,
             max_tokens: int = 8192, retries: int = 2):
    """JSON mode guarantees parseable output; an empty or broken reply is retried without thinking."""
    last = None
    for attempt in range(retries + 1):
        r = chat(messages, model, role, thinking=thinking and attempt == 0, max_tokens=max_tokens,
                 json_mode=True)
        try:
            return parse_json(r.text)
        except (json.JSONDecodeError, ValueError) as e:
            last = e
            messages = messages + [{"role": "assistant", "content": r.text},
                                   {"role": "user", "content": "That was not valid JSON. Reply with only the JSON."}]
    raise ValueError(f"model did not return JSON: {last}")
