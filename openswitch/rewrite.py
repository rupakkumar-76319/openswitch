"""Rewrite the code, verify the rewrite mechanically, run the repo's tests, iterate."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import llm, scan

CONVENTIONS = """\
Target platform: NVIDIA Nemotron models on Nebius Token Factory, OpenAI-compatible API.
- Use the `openai` SDK for every LLM call. One module-level client:
      client = OpenAI(base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
                      api_key=os.environ.get("NEBIUS_API_KEY"))
- Every call is `client.chat.completions.create(...)` with
  extra_body={"chat_template_kwargs": {"enable_thinking": <True|False>}} exactly as decided below.
  If the original call had a string-literal model, put the new model id there as a literal. If the model came
  from a variable/config, keep that mechanism and change the config/default it reads so the decided Nemotron
  id is what reaches the call; also update model allow-lists, price tables and provider checks
  (e.g. model.startswith("gpt")) so the Nemotron id is accepted and priced (Nebius prices are given below).
- Credentials: read NEBIUS_API_KEY instead of OPENAI_API_KEY / ANTHROPIC_API_KEY, including error messages.
- Anthropic calls become chat completions: the `system=` text becomes the first {"role": "system"} message,
  and the reply text is `resp.choices[0].message.content`.
- Remove parameters Nemotron does not use (e.g. `reasoning_effort`); thinking replaces them.
- Keep every other original parameter (temperature, top_p, stop, ...) with its original value.
- Keep `response_format={"type": "json_object"}` where the original used it.
- Set max_tokens to at least the value given below for each call (thinking tokens count against it).
- Keep every function name, signature, docstring, return type and post-processing behaviour unchanged.
- Remove imports that become unused (e.g. `anthropic`). Do not add new dependencies.
"""


def _decisions_text(decisions: dict) -> str:
    return "\n".join(f"- {fn}: model=\"{d['model']}\", enable_thinking={d['thinking']}, max_tokens>={d['max_tokens']}"
                     for fn, d in decisions.items())


def _strip_fence(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def check_file(src: str, decisions: dict, literal: dict | None = None) -> list[str]:
    """Mechanical verification of a rewritten file against the decisions."""
    problems = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    if re.search(r"^\s*(import anthropic|from anthropic)", src, re.M):
        problems.append("still imports anthropic")
    if "api.openai.com" in src or "OPENAI_API_KEY" in src:
        problems.append("still points at OpenAI credentials")
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, d in decisions.items():
        fn = fns.get(name)
        if fn is None:
            problems.append(f"function {name} missing")
            continue
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and scan._chain(n.func)[-3:] == ["chat", "completions", "create"]]
        if len(calls) != 1:
            problems.append(f"{name}: expected one chat.completions.create call, found {len(calls)}")
            continue
        kw = {k.arg: k.value for k in calls[0].keywords if k.arg}
        m = kw.get("model")
        if (literal or {}).get(name, True):
            if not (isinstance(m, ast.Constant) and m.value == d["model"]):
                problems.append(f"{name}: model must be the literal \"{d['model']}\"")
        elif isinstance(m, ast.Constant):
            problems.append(f"{name}: the model was configurable (passed in / read from config); keep model= "
                            "non-literal and set the Nemotron id in the config or default it comes from")
        seg = ast.get_source_segment(src, calls[0]) or ""
        if not re.search(r"enable_thinking\"?'?\s*:\s*" + str(d["thinking"]), seg):
            problems.append(f"{name}: enable_thinking must be {d['thinking']}")
        mt = kw.get("max_tokens")
        if isinstance(mt, ast.Constant) and isinstance(mt.value, int) and mt.value < d["max_tokens"]:
            problems.append(f"{name}: max_tokens {mt.value} is below {d['max_tokens']}")
        # a max_tokens read from config is checked behaviourally (cli._verify_requests), not here
        if "reasoning_effort" in kw:
            problems.append(f"{name}: remove reasoning_effort")
        for p, v in d.get("keep", {}).items():
            got = kw.get(p)
            if not (isinstance(got, ast.Constant) and got.value == v):
                problems.append(f"{name}: keep the original {p}={v!r}")
    return problems


def _signatures(src: str) -> dict[str, list[str]]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    return {n.name: [a.arg for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs]
            for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _signature_changes(before: str, after: str) -> list[str]:
    old, new = _signatures(before), _signatures(after)
    return [f"function {name} changed its parameters {params} -> {new.get(name)}; keep the public API unchanged"
            for name, params in old.items() if not name.startswith("_") and new.get(name, params) != params]


FILE_RE = re.compile(r"^#{2,4} FILE: (\S+)\s*\n```[\w.+-]*\n(.*?)^```", re.S | re.M)


def _files_block(files: dict[str, str]) -> str:
    return "\n\n".join(f"#### FILE: {p}\n```\n{t}\n```" for p, t in files.items())


def rewrite_repo(repo: Path, decisions_by_file: dict, related: list[str], literal: dict, notes: list[dict],
                 prices: str, log=print, feedback: str = "", originals: dict | None = None,
                 verify=None) -> dict[str, str]:
    """Ultra rewrites the call-site files plus any related config files; the checker verifies the result."""
    originals = originals or {}
    paths = list(decisions_by_file) + [r for r in related if r not in decisions_by_file and (repo / r).is_file()]
    files = {p: originals.get(p) or (repo / p).read_text(encoding="utf-8") for p in paths}
    decisions = {k: v for d in decisions_by_file.values() for k, v in d.items()}
    facts = "\n".join(f"- {n['fact']} ({n['url']})" for n in notes)
    msgs = [{"role": "user", "content":
        f"Migrate this repository's LLM calls.\n\n{CONVENTIONS}\nDecisions per function (from measured evals):\n"
        f"{_decisions_text(decisions)}\n\nNebius Token Factory prices (USD per 1M tokens, input/output):\n{prices}\n\n"
        f"Facts from current documentation:\n{facts}\n\n{feedback}\nFiles:\n\n{_files_block(files)}\n\n"
        "Return every file you change, complete, each as `#### FILE: <path>` followed by one fenced code block. "
        "Leave out files that need no change. No other text."}]
    problems: list[str] = []
    for attempt in range(3):
        r = llm.chat(msgs, llm.ULTRA, role="rewriter", thinking=True, max_tokens=32000)
        text = re.sub(r"<think>.*?</think>", "", r.text, flags=re.S)
        changed = {m.group(1).strip("`"): m.group(2) for m in FILE_RE.finditer(text)}
        if not changed and len(files) == 1:  # a single bare code block
            changed = {paths[0]: _strip_fence(text)}
        changed = {p: t for p, t in changed.items() if p in files}
        merged = {**files, **changed}
        problems = []
        for f, d in decisions_by_file.items():
            problems += [f"{f}: {x}" for x in check_file(merged[f], d, literal)]
        for f in changed:
            if f.endswith(".py"):
                problems += [f"{f}: {x}" for x in _signature_changes(files[f], merged[f])]
        everything = "\n".join(merged.values())
        for fn, d in decisions.items():
            if not literal.get(fn, True) and d["model"] not in everything:
                problems.append(f"{fn}: the decided model id {d['model']} is not set anywhere (config/default)")
        if not problems and verify:
            for path, text in merged.items():  # the behavioural check runs the migrated code from disk
                (repo / path).write_text(text, encoding="utf-8")
            problems = verify()
        if not problems:
            return changed
        log(f"  checker rejected rewrite ({attempt + 1}/3): {'; '.join(problems)[:300]}")
        msgs += [{"role": "assistant", "content": r.text},
                 {"role": "user", "content": "The verifier found these problems, fix all of them and return every "
                  "changed file again in the same format:\n- " + "\n- ".join(problems)}]
    raise RuntimeError(f"rewrite still fails checks: {problems}")


def _module_name(rel_file: str) -> str:
    name = Path(rel_file).with_suffix("").as_posix().replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def write_smoke_tests(repo: Path, changed: dict[str, str], site_results: list[dict], feedback: str = "") -> Path:
    """The repo has no tests: Ultra writes live smoke tests from the eval inputs and the output contracts.
    The file must at least import and collect under pytest; otherwise Ultra gets the error and tries again."""
    cases = []
    for r in site_results:
        entry = r.get("entry") or {"file": r["file"], "function": r["function"]}
        probe = r.get("probe") or {}
        calls = "\n".join(f"    {entry['function']}(**{json.dumps(a, ensure_ascii=False, default=str)[:1200]})"
                          for a in r["inputs"][:2])
        cases.append(
            f"- function `{entry['function']}` (import with: `from {_module_name(entry['file'])} import "
            f"{entry['function']}`); it reaches the migrated LLM call `{r['function']}`.\n"
            f"  Example calls (keyword arguments exactly as shown):\n{calls}\n"
            f"  What the app's real code returned for one such call: {str(probe.get('value', ''))[:600]}\n"
            f"  Assert that shape (type, required parts), not the exact wording.")
    tests = repo / "tests"
    tests.mkdir(exist_ok=True)
    conftest = tests / "conftest.py"
    if not conftest.exists():
        conftest.write_text("import pathlib\nimport sys\n\nsys.path.insert(0, str(pathlib.Path(__file__).resolve()"
                            ".parents[1]))\n", encoding="utf-8")
    path = tests / "test_openswitch_smoke.py"
    msgs = [{"role": "user", "content":
        "Write a pytest file `tests/test_openswitch_smoke.py` for this repository after its LLM calls were migrated "
        "to Nemotron on Nebius. For each item below write one live test: call the named function with realistic "
        "arguments adapted from the examples and assert its output contract (types, required fields, non-empty "
        "text). Use exactly the import lines given; the repo root is on sys.path. Let the app pick its own model "
        "and settings (do not pass or hard-code model ids). Keep assertions robust to wording differences. No mocks."
        f"\n\n{feedback}Tests to write:\n" + "\n".join(cases)
        + f"\n\nMigrated code:\n{_files_block(changed)}\n\nReturn only the file in one ```python block."}]
    for _ in range(3):
        r = llm.chat(msgs, llm.ULTRA, role="test-writer", thinking=True, max_tokens=16000)
        path.write_text("# Written by OpenSwitch: live smoke tests for the migrated LLM calls.\n"
                        + _strip_fence(r.text), encoding="utf-8")
        ok, out, _ = run_tests(repo, ["--collect-only", "tests/test_openswitch_smoke.py"])
        if ok:
            break
        msgs += [{"role": "assistant", "content": r.text},
                 {"role": "user", "content": "pytest cannot collect this file:\n" + out[-2500:]
                  + "\nFix it and return the whole file again."}]
    return path


def run_tests(repo: Path, args: list[str] | None = None) -> tuple[bool, str, dict]:
    env = {**os.environ, "PYTHONUTF8": "1"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-rf", "--tb=short", "-p", "no:cacheprovider", *(args or [])]
    p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", out)}
    counts["no_tests"] = p.returncode == 5
    return p.returncode == 0, out, counts


def failing_tests(output: str) -> list[str]:
    return sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", output, re.M)))


def diagnose(test_output: str, files: dict[str, str], decisions: dict) -> dict:
    """Ultra reads failing tests and says whether to fix code or escalate a call site's model."""
    return llm.ask_json([{"role": "user", "content":
        "A migration agent moved this app's LLM calls to Nemotron models and the repo's tests now fail. "
        "Decide for each failure whether the migrated CODE is wrong (bug introduced by the rewrite) or the chosen "
        "MODEL config is too weak for that function, or the TEST itself is wrong. You may not change the repo's "
        "original tests. tests/test_openswitch_smoke.py was generated by the agent and can be wrong: if it calls a "
        "function with the wrong arguments or expects a different return type than the app's real code, put it "
        "under code_fixes as tests/test_openswitch_smoke.py. Never change application behaviour, return values or "
        "signatures to satisfy a test.\n\n"
        f"Current decisions:\n{_decisions_text(decisions)}\n\n{_files_block(files)}"
        f"\n\nTest output:\n```\n{test_output[-7000:]}\n```\n\n"
        'Return JSON {"escalate": [function names whose model is too weak], '
        '"code_fixes": {file path: "what to change"}, "explanation": "two sentences"}'}],
        llm.ULTRA, role="debugger", thinking=True, max_tokens=12000)
