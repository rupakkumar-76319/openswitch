"""Thin wrapper around the LLM provider so the rest of the app never touches the SDK."""
import configparser
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

CONFIG = configparser.ConfigParser()
CONFIG.read(Path(__file__).with_name("settings.ini"))

# USD per 1M tokens, from the provider's pricing page.
PRICES = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class Completion:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    raw: dict = field(default_factory=dict)

    @property
    def cost(self) -> float:
        p = PRICES.get(self.model)
        if p is None:
            return -1.0
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000


def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Set OPENAI_API_KEY (see README).")
    return OpenAI(api_key=key)


def complete(model: str, system: str, user: str) -> Completion:
    """Single chat completion with usage and cost accounting."""
    if not model.startswith("gpt"):
        raise ValueError(f"unsupported model {model}")
    start = time.time()
    resp = _client().chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=CONFIG.getint("llm", "max_output_tokens"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return Completion(model=model, text=resp.choices[0].message.content, input_tokens=resp.usage.prompt_tokens,
                      output_tokens=resp.usage.completion_tokens, seconds=time.time() - start,
                      raw=resp.model_dump())
