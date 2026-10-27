"""Plan, build eval sets from the app's own prompts, run every Nemotron config, judge, choose."""
from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import capture, llm
from .scan import CallSite

# (model, thinking) configs tried for every call site, cheapest first.
CANDIDATES = [
    (llm.NANO, False), (llm.NANO, True),
    (llm.SUPER, False), (llm.SUPER, True),
    (llm.ULTRA, False), (llm.ULTRA, True),
]
REFERENCE = (llm.ULTRA, True)
# A config passes when every output survives the app's own parsing, its mean checklist score is within
# MARGIN of the best config (and at least PASS_FLOOR), and no single output scores below PASS_MIN.
PASS_FLOOR, MARGIN, PASS_MIN = 7.5, 0.75, 5.0
WORKERS = 8


def label(model: str, thinking: bool) -> str:
    return f"{llm.SHORT[model]} {'think' if thinking else 'fast'}"


@dataclass
class ConfigResult:
    model: str
    thinking: bool
    outputs: list = field(default_factory=list)   # per input: dict(text, ok, value, score, reason, tokens)
    threshold: float = PASS_FLOOR

    @property
    def name(self):
        return label(self.model, self.thinking)

    def stat(self, key):
        return [o[key] for o in self.outputs]

    @property
    def functional(self):
        return sum(o["ok"] for o in self.outputs) / max(1, len(self.outputs))

    @property
    def mean_score(self):
        s = self.stat("score")
        return sum(s) / max(1, len(s))

    @property
    def min_score(self):
        return min(self.stat("score") or [0])

    @property
    def avg_in(self):
        return sum(self.stat("prompt_tokens")) / max(1, len(self.outputs))

    @property
    def avg_out(self):
        return sum(self.stat("completion_tokens")) / max(1, len(self.outputs))

    @property
    def max_out(self):
        return max(self.stat("completion_tokens") or [0])

    @property
    def cost_per_1k(self):
        pi, po = llm.PRICES[self.model]
        return (self.avg_in * pi + self.avg_out * po) / 1e6 * 1000

    @property
    def avg_seconds(self):
        return sum(self.stat("seconds")) / max(1, len(self.outputs))

    def passes(self):
        return self.functional == 1.0 and self.mean_score >= self.threshold and self.min_score >= PASS_MIN

    def summary(self):
        return {"config": self.name, "model": self.model, "thinking": self.thinking,
                "functional": round(self.functional, 3), "mean_score": round(self.mean_score, 2),
                "min_score": self.min_score, "avg_in": round(self.avg_in), "avg_out": round(self.avg_out),
                "max_out": self.max_out, "cost_per_1k": round(self.cost_per_1k, 5),
                "avg_seconds": round(self.avg_seconds, 2), "passes": self.passes()}


def plan(sites: list[CallSite], repo_files: dict[str, str]) -> dict:
    """Ultra reads every call site in context and writes the migration plan."""
    blocks = "\n\n".join(
        f"### {s.id}  (provider={s.provider}, model={s.model}, params={s.params})\n```python\n{s.function_source}\n```"
        for s in sites)
    usage = "\n\n".join(f"--- {p}\n{t[:4000]}" for p, t in repo_files.items())
    return llm.ask_json([{"role": "user", "content":
        "You are the planner of a code-migration agent that moves an app's LLM calls to NVIDIA Nemotron models. "
        "For every call site below, work out what the call is for and what the surrounding code needs from it.\n\n"
        f"{blocks}\n\nOther repo files for context:\n{usage}\n\n"
        "Return JSON: {\"sites\": {<site id>: {\"purpose\": str, \"output_contract\": str (what downstream code "
        "parses or relies on), \"difficulty\": \"low\"|\"medium\"|\"high\", \"rubric\": [3-5 short checks a grader "
        "can apply to one output], \"risks\": [migration risks, e.g. unsupported params, provider-specific "
        "response shapes, token limits], \"effective_model\": the provider model id that reaches this call by "
        "default in production (resolve variables, config files and defaults), \"related_files\": [other repo "
        "files that must change so this call runs on a new provider/model: config files that set the model or key, "
        "model allow-lists, price tables, provider checks like model.startswith('gpt'), env/docs], "
        "\"eval_entrypoint\": \"<file path>:<function name>\" of the function to drive in evals: the outermost "
        "function that takes plain data (not prompts or model names) and ends up making this call, so the app "
        "builds its real production prompt itself; use the call site's own function if it already takes plain "
        "data}}, "
        "\"order\": [site ids, easiest first]}"}],
        llm.ULTRA, role="planner", thinking=True, max_tokens=16000)


def gen_inputs(site: CallSite, fn, plan_entry: dict, context: str, n: int) -> list[dict]:
    """Super invents realistic, varied inputs for the function (it never sees the repo's tests).
    Every input is tried against the app's own code; inputs the app rejects are sent back with the error."""
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = site.function_source
    msgs = [{"role": "user", "content":
        f"Function `{fn.__name__}{sig}`:\n```python\n{src}\n```\n"
        f"Purpose: {plan_entry.get('purpose', '')}\n\nRelated code and config:\n{context[:6000]}\n\n"
        f"Write {n} realistic and varied argument sets for this function as production traffic would send them. "
        "Match the exact data shapes the code expects (read how it indexes and joins its arguments). "
        "Include ordinary cases and 1-2 tricky edge cases (missing information, ambiguity, numbers that need care). "
        "Only use JSON-serialisable values; omit arguments that have good defaults. "
        'Return JSON {"inputs": [{"args": {<parameter name>: <value>, ...}}, ...]} where the keys of "args" are '
        f"the function's parameter names {params}."}]
    out: list[dict] = []
    for _ in range(2):
        try:
            data = llm.ask_json(msgs, llm.SUPER, role="eval-writer", thinking=True, max_tokens=12000)
        except ValueError:
            break  # keep the inputs we already have
        errors = []
        for item in data.get("inputs", []) if isinstance(data, dict) else []:
            args = item.get("args", item) if isinstance(item, dict) else item
            for cand in (args, {params[0]: args} if len(params) == 1 else None):
                if not isinstance(cand, dict):
                    continue
                try:
                    sig.bind(**cand)
                except TypeError:
                    continue
                try:
                    capture.capture(fn, cand)
                    out.append(cand)
                except Exception as e:
                    errors.append(f"{json.dumps(cand, default=str)[:300]} -> {type(e).__name__}: {e}")
                break
        if len(out) >= n or not errors:
            break
        msgs += [{"role": "assistant", "content": json.dumps(data, default=str)[:6000]},
                 {"role": "user", "content": "The app rejected these inputs before calling the LLM:\n"
                  + "\n".join(errors[:5]) + f"\nWrite {n - len(out)} more inputs in the shape the code expects."}]
    return out[:n]


def _run(msgs, model, thinking, kwargs):
    json_mode = (kwargs.get("response_format") or {}).get("type") == "json_object"
    temp = kwargs.get("temperature")
    try:
        r = llm.chat(msgs, model, role="candidate", thinking=thinking, max_tokens=8192,
                     temperature=temp, json_mode=json_mode)
        return {"text": r.text, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "seconds": r.seconds}
    except Exception as e:
        return {"text": "", "error": str(e)[:300], "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0}


def _norm(v):
    if isinstance(v, str):
        return " ".join(v.lower().split())
    return json.dumps(v, sort_keys=True, default=str)


CORRECTNESS = ("Every fact, number, label and extracted value is correct for this input "
               "(the reference is a hint from a strong model, but it can be wrong)")


def _checks(plan_entry: dict) -> list[str]:
    rubric = plan_entry.get("rubric") or []
    if isinstance(rubric, str):
        rubric = [rubric]
    return [CORRECTNESS] + [str(r) for r in rubric][:5]


def _judge(site, plan_entry, prompt_msgs, ref_value, cand_value) -> tuple[float, str]:
    """Checklist grading: each rubric check is pass/fail, score = 10 x share passed.
    Yes/no checks are far more repeatable between runs than one holistic 0-10 grade."""
    checks = _checks(plan_entry)
    convo = "\n".join(f"[{m['role']}] {m['content']}" for m in prompt_msgs)[:6000]
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(checks))
    data = llm.ask_json([{"role": "user", "content":
        f"You check one output of the app function `{site.function}`.\nPurpose: {plan_entry.get('purpose')}\n"
        f"Downstream contract: {plan_entry.get('output_contract')}\n\n"
        f"Request sent to the model:\n{convo}\n\n"
        f"Reference result from a strong model (a hint, not ground truth): "
        f"{json.dumps(ref_value, default=str, ensure_ascii=False)}\n"
        f"Output to check: {json.dumps(cand_value, default=str, ensure_ascii=False)}\n\n"
        f"Checks:\n{listed}\n\n"
        "Decide each check strictly but fairly: wording, length and style that differ from the reference are fine "
        "when the check is met; if the input is genuinely ambiguous, a defensible reading the instructions allow "
        'passes. Return JSON {"passed": [true/false for each check in order], "reason": "<one sentence on any '
        'failed check>"}'}],
        llm.ULTRA, role="judge", thinking=False, max_tokens=800)
    passed = [bool(x) for x in (data.get("passed") or [])][:len(checks)]
    passed += [False] * (len(checks) - len(passed))
    score = round(10 * sum(passed) / len(checks), 2)
    failed = [checks[i] for i, ok in enumerate(passed) if not ok]
    reason = str(data.get("reason", "")) if failed else "all checks passed"
    return score, reason


def comparable(value):
    """A stable fingerprint of a return value: JSON-native values as-is, other objects by type only
    (they may carry timings or raw responses that differ between runs)."""
    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
        return _norm(value)
    return f"<{type(value).__name__}>"


def _probe(configs) -> dict | None:
    """One real model answer and what the ORIGINAL app code returned for it. After the rewrite the same
    answer is replayed through the migrated code; the result must not change."""
    for c in configs:
        for i, o in enumerate(c.outputs):
            if o["ok"] and o.get("text"):
                return {"input_index": i, "text": o["text"], "value": comparable(o["value"])}
    return None


def evaluate_site(site: CallSite, fn, inputs: list[dict], plan_entry: dict, log=print) -> dict:
    caps, kept = [], []
    for a in inputs:
        try:
            caps.append(capture.capture(fn, a))
            kept.append(a)
        except Exception as e:  # generated input the app itself rejects before calling the LLM
            log(f"  dropped one generated input: {type(e).__name__}: {str(e)[:80]}")
    inputs = kept
    if not inputs:
        raise RuntimeError(f"no usable eval inputs for {site.function}")
    prompts = [capture.to_openai_messages(c.provider, c.kwargs) for c in caps]
    kw = caps[0].kwargs

    configs = [ConfigResult(m, t) for m, t in CANDIDATES]
    jobs = [(ci, ii) for ci in range(len(configs)) for ii in range(len(inputs))]
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda j: _run(prompts[j[1]], configs[j[0]].model, configs[j[0]].thinking, kw), jobs))
    for (ci, ii), r in zip(jobs, results):
        ok, value = capture.replay(fn, inputs[ii], r["text"]) if not r.get("error") else (False, r["error"])
        configs[ci].outputs.append({**r, "ok": ok, "value": value, "input": inputs[ii]})

    ref = next(c for c in configs if (c.model, c.thinking) == REFERENCE)
    ri = configs.index(ref)

    def grade(ci_ii):
        ci, ii = ci_ii
        o, ro = configs[ci].outputs[ii], ref.outputs[ii]
        if not o["ok"]:
            return 0.0, f"app rejected output: {o['value']}"
        return _judge(site, plan_entry, prompts[ii], ro["value"] if ro["ok"] and ci != ri else None, o["value"])

    # every config is graded, the reference included; identical outputs share one grading
    cache: dict = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        keyed = {}
        for ci, ii in jobs:
            o = configs[ci].outputs[ii]
            key = (ii, _norm(o["value"])) if o["ok"] else (ci, ii, "fail")
            keyed.setdefault(key, (ci, ii))
        for key, g in zip(keyed, ex.map(grade, keyed.values())):
            cache[key] = g
    for ci, ii in jobs:
        o = configs[ci].outputs[ii]
        key = (ii, _norm(o["value"])) if o["ok"] else (ci, ii, "fail")
        o["score"], o["reason"] = cache[key]

    best = max(c.mean_score for c in configs if c.functional == 1.0) if any(
        c.functional == 1.0 for c in configs) else max(c.mean_score for c in configs)
    threshold = max(PASS_FLOOR, best - MARGIN)
    for c in configs:
        c.threshold = threshold
    ranked = sorted(configs, key=lambda c: c.cost_per_1k)
    chosen = next((c for c in ranked if c.passes()),
                  max(configs, key=lambda c: (c.functional, c.mean_score, -c.cost_per_1k)))
    log(f"  {site.function}: chose {chosen.name} "
        f"(score {chosen.mean_score:.1f}, best {best:.1f}, ${chosen.cost_per_1k:.4f}/1K calls)")
    failures = [{"config": c.name, "input": o["input"], "value": o["value"], "score": o["score"], "reason": o["reason"]}
                for c in configs for o in c.outputs if o["score"] < 10]
    return {
        "site": site.id, "function": site.function, "file": site.file, "source_model": site.model,
        "provider": site.provider, "plan": plan_entry, "checks": _checks(plan_entry), "n_inputs": len(inputs),
        "inputs": inputs, "prompts": prompts, "configs": [c.summary() for c in configs],
        "chosen": chosen.summary(), "best_score": best, "threshold": threshold,
        # thinking length varies a lot between inputs; leave generous headroom so answers are never cut off
        "recommended_max_tokens": (max(2048, int(chosen.max_out * 3)) if chosen.thinking
                                   else max(512, int(chosen.max_out * 2))),
        "reference_out_tokens": ref.avg_out, "ultra_fast_out_tokens": next(
            c.avg_out for c in configs if (c.model, c.thinking) == (llm.ULTRA, False)),
        "avg_in_tokens": ref.avg_in, "failures": failures[:60], "probe": _probe(configs),
        "samples": [{"input": o["input"], "value": o["value"], "score": o["score"], "reason": o["reason"]}
                    for o in chosen.outputs][:6],
    }
