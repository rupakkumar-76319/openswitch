"""Run the app's own functions against a fake SDK.

capture(): call a function, intercept the first LLM request it makes and return the exact
           kwargs (real prompts, real parameters) without contacting any provider.
replay():  call the function again, but answer its LLM request with a given text, so the
           app's own post-processing (json.loads, float(), label checks...) runs on a
           candidate model's output. This is the functional check.
"""
from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


class Captured(Exception):
    def __init__(self, provider: str, kwargs: dict, callers: list[str] | None = None):
        self.provider, self.kwargs, self.callers = provider, kwargs, callers or []


def _callers() -> list[str]:
    f, names = sys._getframe(2), []
    while f is not None:
        names.append(f.f_code.co_name)
        f = f.f_back
    return names


_TARGETS = []  # (owner class, attribute, provider)


def _targets():
    if _TARGETS:
        return _TARGETS
    try:
        from openai.resources.chat.completions import Completions
        _TARGETS.append((Completions, "create", "openai"))
    except ImportError:
        pass
    try:
        from anthropic.resources.messages import Messages
        _TARGETS.append((Messages, "create", "anthropic"))
    except ImportError:
        pass
    return _TARGETS


@contextmanager
def _patched(handler):
    saved = [(cls, attr, getattr(cls, attr)) for cls, attr, _ in _targets()]
    try:
        for cls, attr, provider in _targets():
            setattr(cls, attr, (lambda p: lambda self, *a, **kw: handler(p, kw))(provider))
        yield
    finally:
        for cls, attr, fn in saved:
            setattr(cls, attr, fn)


def forget_modules(repo: Path):
    """Drop the repo's modules from the import cache so edited files are imported fresh."""
    root = str(repo.resolve()).lower()
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and str(Path(f).resolve()).lower().startswith(root):
            del sys.modules[name]
    importlib.invalidate_caches()


def load_module(repo: Path, rel_file: str):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEBIUS_API_KEY"):
        os.environ.setdefault(key, "openswitch-capture-placeholder")
    root = str(repo.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    mod_name = Path(rel_file).with_suffix("").as_posix().replace("/", ".")
    if mod_name.endswith(".__init__"):
        mod_name = mod_name[: -len(".__init__")]
    return importlib.import_module(mod_name)


def to_openai_messages(provider: str, kwargs: dict) -> list[dict]:
    """Normalise a captured request to OpenAI chat messages."""
    msgs = []
    if provider == "anthropic":
        system = kwargs.get("system")
        if isinstance(system, list):
            system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
        if system:
            msgs.append({"role": "system", "content": system})
    for m in kwargs.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
        role = "system" if m.get("role") == "developer" else m.get("role")
        msgs.append({"role": role, "content": content})
    return msgs


def capture(fn, args: dict) -> Captured:
    def handler(provider, kw):
        raise Captured(provider, kw, _callers())
    with _patched(handler):
        try:
            fn(**args)
        except Captured as c:
            return c
    raise RuntimeError(f"{fn.__name__} returned without calling an LLM")


def _fake_response(provider: str, text: str):
    """A real SDK response object, so code that calls .model_dump(), .usage, etc. keeps working."""
    if provider == "anthropic":
        try:
            from anthropic.types import Message, TextBlock, Usage
            return Message(id="msg_openswitch", type="message", role="assistant", model="replay",
                           content=[TextBlock(type="text", text=text)], stop_reason="end_turn",
                           usage=Usage(input_tokens=0, output_tokens=0))
        except Exception:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                                   usage=SimpleNamespace(input_tokens=0, output_tokens=0))
    try:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage
        return ChatCompletion(id="chatcmpl-openswitch", object="chat.completion", created=0, model="replay",
                              choices=[Choice(index=0, finish_reason="stop",
                                              message=ChatCompletionMessage(role="assistant", content=text))],
                              usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0))
    except Exception:
        msg = SimpleNamespace(content=text, role="assistant", tool_calls=None, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop", index=0)],
                               usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))


def count_calls(fn, args: dict, text: str = "{}") -> int:
    """How many LLM requests one call of fn makes (answers are dummies; errors after that are ignored)."""
    n = [0]

    def handler(provider, kw):
        n[0] += 1
        return _fake_response(provider, text)
    with _patched(handler):
        try:
            fn(**args)
        except Exception:
            pass
    return n[0]


def replay(fn, args: dict, text: str):
    """Return (ok, value_or_error) of running fn with the LLM answering `text`."""
    with _patched(lambda provider, kw: _fake_response(provider, text)):
        try:
            return True, fn(**args)
        except Exception as e:  # the app's own parsing rejected this output
            return False, f"{type(e).__name__}: {e}"
