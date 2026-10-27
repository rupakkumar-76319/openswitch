"""Thin wrapper around the LLM provider so the rest of the app never touches the SDK."""
import configparser
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

CONFIG = configparser.ConfigParser()
CONFIG.read(Path(__file__).with_name("settings.ini"))

# USD per 1M tokens, from Nebius Token Factory pricing.
PRICES = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": {"input": 0.06, "output": 0.24},
    "nvidia/Nemotron-3_5-Lightning": {"input": 0.06, "output": 0.24},
    "nvidia/nemotron-3-super-120b-a12b": {"input": 0.3, "output": 0.9},
    "nvidia/Nemotron-3-Ultra-550b-a55b": {"input": 1.0, "output": 3.0},
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
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        raise EnvironmentError("Set NEBIUS_API_KEY (see README).")
    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
    return OpenAI(base_url=base_url, api_key=key)


def complete(model: str, system: str, user: str) -> Completion:
    """Single chat completion with usage and cost accounting."""
    start = time.time()
    resp = _client().chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=max(CONFIG.getint("llm", "max_output_tokens"), 2160),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    return Completion(model=model, text=resp.choices[0].message.content, input_tokens=resp.usage.prompt_tokens,
                      output_tokens=resp.usage.completion_tokens, seconds=time.time() - start,
                      raw=resp.model_dump())
