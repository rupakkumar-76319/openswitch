"""Find every LLM call site in a Python repo (OpenAI and Anthropic SDKs) with the AST."""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "site-packages",
             "build", "dist", ".openswitch", ".tox", ".mypy_cache"}

CALL_SUFFIXES = {
    ("chat", "completions", "create"): "openai",
    ("responses", "create"): "openai-responses",
    ("completions", "create"): "openai-legacy",
    ("messages", "create"): "anthropic",
}
CLIENT_CTORS = {"OpenAI": "openai", "AsyncOpenAI": "openai", "AzureOpenAI": "openai",
                "Anthropic": "anthropic", "AsyncAnthropic": "anthropic"}


@dataclass
class CallSite:
    id: str
    file: str
    line: int
    end_line: int
    function: str | None
    provider: str
    model: str
    params: list[str]
    source: str
    function_source: str
    model_is_literal: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class ClientSite:
    file: str
    line: int
    provider: str
    source: str


def _chain(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _enclosing(tree: ast.AST) -> dict[int, ast.AST]:
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                if hasattr(n, "lineno"):
                    # innermost wins because inner functions are visited after outer ones
                    owner[id(n)] = fn
    return owner


def scan(repo: Path) -> tuple[list[CallSite], list[ClientSite]]:
    sites, clients = [], []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if any(p in SKIP_DIRS for p in rel.parts) or rel.parts[0] in ("tests", "test"):
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        owner = _enclosing(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _chain(node.func)
            name = chain[-1] if chain else None
            if name in CLIENT_CTORS:
                clients.append(ClientSite(rel.as_posix(), node.lineno, CLIENT_CTORS[name],
                                          ast.get_source_segment(src, node) or ""))
                continue
            for suffix, provider in CALL_SUFFIXES.items():
                if len(chain) >= len(suffix) and tuple(chain[-len(suffix):]) == suffix:
                    kw = {k.arg: k.value for k in node.keywords if k.arg}
                    m = kw.get("model")
                    model = m.value if isinstance(m, ast.Constant) else (ast.get_source_segment(src, m) if m else "?")
                    fn = owner.get(id(node))
                    sites.append(CallSite(
                        id=f"{rel.as_posix()}:{fn.name if fn else node.lineno}",
                        file=rel.as_posix(), line=node.lineno, end_line=node.end_lineno,
                        function=fn.name if fn else None, provider=provider, model=str(model),
                        params=sorted(kw), source=ast.get_source_segment(src, node) or "",
                        function_source=ast.get_source_segment(src, fn) if fn else "",
                        model_is_literal=isinstance(m, ast.Constant),
                    ))
                    break
    return sites, clients


def callers(repo: Path, function: str) -> list[tuple[str, str]]:
    """(file, function) pairs of functions that call `function` directly."""
    found = []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if any(p in SKIP_DIRS for p in rel.parts) or rel.parts[0] in ("tests", "test"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name != function:
                if any(isinstance(n, ast.Call) and (_chain(n.func) or [None])[-1] == function for n in ast.walk(fn)):
                    found.append((rel.as_posix(), fn.name))
    return found
