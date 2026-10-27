"""openswitch migrate <repo>: move a repo's LLM calls to NVIDIA Nemotron on Nebius Token Factory."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import capture, evals, llm, report, research, rewrite, scan

console = Console(highlight=False)


def stage(n: int, title: str, who: str = ""):
    console.rule(f"[bold]{n}. {title}[/bold]" + (f"  [dim]{who}[/dim]" if who else ""), align="left")


def log(msg: str):
    console.print(f"  [dim]{msg}[/dim]")


def git(repo: Path, *args) -> str:
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return p.stdout.strip()


def migrate(repo: Path, n_inputs: int, max_iters: int, reuse: bool):
    t0 = time.time()
    state_dir = repo / ".openswitch"
    state_dir.mkdir(exist_ok=True)
    cache = state_dir / "evals.json"

    stage(1, "Scan the repo for LLM call sites", "Python AST")
    sites, _ = scan.scan(repo)
    if not sites:
        console.print("[red]No OpenAI/Anthropic call sites found.[/red]")
        return 1
    t = Table(show_edge=False, pad_edge=False)
    for col in ("call site", "provider", "model", "params"):
        t.add_column(col)
    for s in sites:
        t.add_row(f"{s.file}:{s.line} {s.function}()", s.provider, s.model, ", ".join(p for p in s.params if p != "messages"))
    console.print(t)
    repo_files = _context_files(repo)

    if reuse and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        res, plan, site_results = saved["research"], saved["plan"], saved["sites"]
        for mid, p in res["nemotron_prices"].items():
            llm.PRICES[mid] = (p["input"], p["output"])
        console.print("  [yellow]reusing research + evals from .openswitch/evals.json[/yellow]")
    else:
        stage(2, "Plan the migration", "Nemotron Ultra, thinking on")
        plan = evals.plan(sites, repo_files)["sites"]
        for s in sites:
            p = plan.get(s.id) or next((v for k, v in plan.items() if k.endswith(s.function or "")), {})
            plan[s.id] = p
            console.print(f"  [bold]{s.function}[/bold] [{p.get('difficulty', '?')}] {p.get('purpose', '')[:110]}")
            if s.model_is_literal is False and p.get("effective_model"):
                log(f"model is set outside the call; resolved to {p['effective_model']}")

        stage(3, "Research pricing and API differences", "Tavily search + Nemotron Super")
        for s in sites:
            if not s.model_is_literal and plan[s.id].get("effective_model"):
                s.model = plan[s.id]["effective_model"]
        res = research.run(sorted({s.model for s in sites}), {s.provider for s in sites}, log=log)
        for m, p in res["source_prices"].items():
            log(f"{m}: ${p['input']} in / ${p['output']} out per 1M  ← {p['url'][:80]}")
        for n in res["notes"][:6]:
            log(f"• {n['fact'][:120]}")

        stage(4, "Build evals from the app's own prompts and measure every Nemotron config",
              "Super writes inputs · 6 configs run · Ultra judges")
        site_results = []
        for s in sites:
            fn, inputs = None, []
            for cand in _entrypoints(repo, s, plan[s.id])[:3]:
                ctx = _eval_context(repo_files, s, cand)
                cand_inputs = evals.gen_inputs(s, cand, plan[s.id], ctx, n_inputs)
                if not cand_inputs:  # an empty or rejected batch happens now and then; ask once more
                    cand_inputs = evals.gen_inputs(s, cand, plan[s.id], ctx, n_inputs)
                if not cand_inputs:
                    log(f"{cand.__name__}(): no usable eval inputs; trying the next entry point")
                    continue
                reason = "" if cand.__name__ == s.function else _entry_problem(cand, cand_inputs, s.function)
                if not reason:
                    fn, inputs, context = cand, cand_inputs, ctx
                    break
                log(f"{cand.__name__}(): {reason}; trying the next entry point")
            if fn is None:
                raise RuntimeError(f"could not build eval inputs for {s.function}")
            if fn.__name__ != s.function:
                log(f"{s.function}: evals drive {fn.__name__}() so the app builds its real prompts")
            log(f"{s.function}: {len(inputs)} inputs → {len(inputs) * len(evals.CANDIDATES)} runs")
            r = evals.evaluate_site(s, fn, inputs, plan[s.id], log=console.print)
            more = evals.gen_inputs(s, fn, plan[s.id], context, n_inputs) if _close_call(r) else []
            if more:
                log(f"{s.function}: a config ≥3x cheaper is within 1 point of the bar; "
                    f"adding {len(more)} inputs to decide on a bigger sample")
                r = evals.evaluate_site(s, fn, inputs + more, plan[s.id], log=console.print)
                r["adaptive"] = True
            r["entry"] = {"file": Path(fn.__code__.co_filename).resolve().relative_to(repo).as_posix(),
                          "function": fn.__name__}
            site_results.append(r)
        cache.write_text(json.dumps({"research": res, "plan": plan, "sites": site_results}, indent=2, default=str),
                         encoding="utf-8")

    t = Table(show_edge=False, pad_edge=False, title="mean judge score (✓ = passes)")
    t.add_column("function")
    for m, th in evals.CANDIDATES:
        t.add_column(evals.label(m, th), justify="center")
    for r in site_results:
        cells = []
        for c in r["configs"]:
            mark = "✓" if c["passes"] else " "
            style = "bold green" if c["config"] == r["chosen"]["config"] else ("green" if c["passes"] else "red")
            cells.append(f"[{style}]{c['mean_score']:.1f}{mark}[/{style}]")
        t.add_row(r["function"], *cells)
    console.print(t)

    decisions_by_file: dict[str, dict] = {}
    original_max, keep = {}, {}
    for s in sites:
        m = re.search(r"max_tokens\s*=\s*(\d+)", s.source)
        original_max[s.function] = int(m.group(1)) if m else None
        keep[s.function] = {k: float(v) if "." in v else int(v) for k, v in
                            re.findall(r"(temperature|top_p)\s*=\s*([0-9.]+)", s.source)}
    for r in site_results:
        c = r["chosen"]
        decisions_by_file.setdefault(r["file"], {})[r["function"]] = {
            "model": c["model"], "thinking": c["thinking"],
            "max_tokens": max(r["recommended_max_tokens"], original_max.get(r["function"]) or 0),
            "keep": keep.get(r["function"], {})}

    stage(5, "Rewrite the code", "Nemotron Ultra writes · AST checker verifies")
    branch = None
    if (repo / ".git").exists():
        branch = "openswitch/nemotron"
        git(repo, "checkout", "-B", branch)
    literal = {s.function: s.model_is_literal for s in sites}
    related = {m.group(0) for r in site_results for f in (r["plan"].get("related_files") or [])
               if isinstance(f, str) and (m := re.match(r"[\w./-]+", f.strip()))
               and (repo / m.group(0)).is_file() and not m.group(0).startswith(("tests/", "test/"))}
    related |= {p for p, t in repo_files.items() if not p.endswith(".py")
                and any(r["source_model"] in t for r in site_results)}
    related = sorted(related - set(decisions_by_file))
    prices = "\n".join(f"- {m}: {p[0]} / {p[1]}" for m, p in llm.PRICES.items())
    originals = {f: (repo / f).read_text(encoding="utf-8") for f in list(decisions_by_file) + related}

    def apply_rewrite(feedback: str = "") -> list[str]:
        for f, t in originals.items():
            (repo / f).write_text(t, encoding="utf-8")
        changed = rewrite.rewrite_repo(repo, decisions_by_file, related, literal, res["notes"], prices,
                                       log=log, feedback=feedback, originals=originals,
                                       verify=lambda: _verify_requests(repo, site_results, decisions_by_file))
        for f, t in changed.items():
            (repo / f).write_text(t, encoding="utf-8")
        return sorted(changed)

    changed = apply_rewrite()
    log(f"rewrote {', '.join(changed)}; checker passed for {sum(map(len, decisions_by_file.values()))} call sites")
    _update_project_files(repo)

    stage(6, "Run tests and iterate", "pytest · Nemotron Ultra debugs")
    history = []
    ok, out, counts = rewrite.run_tests(repo)
    wrote_tests = False
    if counts.get("no_tests"):
        path = rewrite.write_smoke_tests(repo, {f: (repo / f).read_text(encoding="utf-8") for f in changed},
                                         site_results)
        wrote_tests = True
        log(f"repo has no tests: wrote {path.relative_to(repo).as_posix()} (live smoke tests)")
        ok, out, counts = rewrite.run_tests(repo)
    summary = _summ(counts)
    history.append({"n": 1, "summary": summary,
                    **({"action": "wrote live smoke tests (repo had none)"} if wrote_tests else {})})
    console.print(f"  iteration 1: {summary}")
    it = 1
    ranked = {r["function"]: sorted(r["configs"], key=lambda c: c["cost_per_1k"]) for r in site_results}
    while not ok and it < max_iters:
        it += 1
        fails = rewrite.failing_tests(out)
        ok2, _, _ = rewrite.run_tests(repo, fails) if fails else (False, "", {})
        if ok2:
            history.append({"n": it, "summary": f"{len(fails)} failing test(s) passed on re-run (flaky)",
                            "action": "re-ran failures"})
            ok, out, counts = rewrite.run_tests(repo)
            summary = _summ(counts)
            history[-1]["summary"] += f"; full run {summary}"
            console.print(f"  iteration {it}: {history[-1]['summary']}")
            continue
        cur = {p: (repo / p).read_text(encoding="utf-8") for p in originals}
        if wrote_tests:
            cur["tests/test_openswitch_smoke.py"] = (repo / "tests/test_openswitch_smoke.py").read_text(encoding="utf-8")
        diag = rewrite.diagnose(out, cur, {k: v for d in decisions_by_file.values() for k, v in d.items()})
        actions = []
        for fn_name in diag.get("escalate", []):
            for f, d in decisions_by_file.items():
                if fn_name in d:
                    nxt = _next_config(ranked[fn_name], d[fn_name])
                    if nxt:
                        d[fn_name].update(model=nxt["model"], thinking=nxt["thinking"])
                        actions.append(f"escalated {fn_name} → {nxt['config']}")
        fixes = {k: v for k, v in (diag.get("code_fixes") or {}).items() if v}
        smoke_fix = next((v for k, v in fixes.items() if k.endswith("test_openswitch_smoke.py")), None)
        code_fixes = {k: v for k, v in fixes.items() if not k.endswith("test_openswitch_smoke.py")}
        if code_fixes or any(a.startswith("escalated") for a in actions):
            fb = ("Previous version failed tests. Required changes:\n" +
                  "\n".join(f"- {k}: {v}" for k, v in code_fixes.items()) + "\n") if code_fixes else ""
            good = {f: (repo / f).read_text(encoding="utf-8") for f in originals}
            try:
                changed = apply_rewrite(fb)
                actions += [f"fixed {k}: {str(v)[:120]}" for k, v in code_fixes.items()]
            except RuntimeError as e:  # the fix failed the checker: keep the last version that passed it
                for f, t in good.items():
                    (repo / f).write_text(t, encoding="utf-8")
                actions.append(f"proposed fix rejected by the checker, kept the previous version ({str(e)[:100]})")
        if smoke_fix and wrote_tests:  # only OpenSwitch's own tests may be rewritten, never the repo's
            rewrite.write_smoke_tests(repo, {f: (repo / f).read_text(encoding="utf-8") for f in changed},
                                      site_results, feedback=f"The previous version of this test was wrong: "
                                      f"{smoke_fix}\nTest output:\n{out[-2000:]}\n\n")
            actions.append(f"rewrote its own smoke test: {smoke_fix[:100]}")
        log(diag.get("explanation", ""))
        ok, out, counts = rewrite.run_tests(repo)
        summary = _summ(counts)
        history.append({"n": it, "summary": summary, "action": "; ".join(actions) or "no change proposed"})
        console.print(f"  iteration {it}: {summary}  [dim]{'; '.join(actions)[:200]}[/dim]")
        if not actions:
            break
    for r in site_results:  # reflect escalations in the report
        d = decisions_by_file[r["file"]][r["function"]]
        if (d["model"], d["thinking"]) != (r["chosen"]["model"], r["chosen"]["thinking"]):
            r["chosen"] = next(c for c in r["configs"] if (c["model"], c["thinking"]) == (d["model"], d["thinking"]))
            r["escalated"] = True

    limitations = [
        "Baseline cost is an estimate: source-model token counts are approximated with Nemotron's tokenizer, and for "
        "o-series models hidden reasoning is approximated with Nemotron Ultra's thinking length.",
        "The reference answers come from Nemotron 3 Ultra with thinking, not from the original provider (no "
        "OpenAI/Anthropic key is needed or used); the repo's own tests are the independent check.",
        f"Each function was evaluated on {site_results[0]['n_inputs']} generated inputs; rare inputs can still "
        "behave differently, so keep monitoring after merging.",
    ]
    if not ok:
        failing = rewrite.failing_tests(out)
        limitations.insert(0, f"Still failing after {it} iteration(s): {', '.join(failing) or 'see test log'}.")
    (state_dir / "pytest.log").write_text(out, encoding="utf-8")

    result = {"repo_name": repo.name, "sites": site_results, "research": res,
              "tests": {"ok": ok, "summary": summary, "iterations": it, "history": history, "written": wrote_tests},
              "limitations": limitations, "seconds": round(time.time() - t0)}
    stage(7, "Report and commit")
    path = report.write(repo, result)
    b = report.build(result)
    if branch:
        git(repo, "add", "-A")
        # The code in this commit was written by Nemotron 3 Ultra, so the commit is authored that way.
        git(repo, "-c", "user.name=OpenSwitch (NVIDIA Nemotron via Nebius Token Factory)",
            "-c", "user.email=openswitch@users.noreply.github.com",
            "commit", "-m", f"Migrate LLM calls to NVIDIA Nemotron on Nebius Token Factory\n\n"
            f"Estimated LLM cost -{b['saving_pct']:.0f}%; tests: {summary}. See OPENSWITCH_REPORT.md.\n\n"
            f"Code written by Nemotron 3 Ultra 550B; models chosen by measured evals on Nano/Super/Ultra.")
        log(f"committed on branch {branch}")
    console.print()
    console.print(f"[bold green]Done in {result['seconds']}s.[/bold green] "
                  f"Cost per 1K calls: ${b['total_before']:.3f} → ${b['total_after']:.3f} "
                  f"([bold]{b['saving_pct']:.0f}% lower[/bold]); tests: {summary}; agent spend ${llm.LEDGER.total:.3f}")
    console.print(f"Report: {path}")
    return 0 if ok else 2


def _entrypoints(repo: Path, site, plan_entry: dict) -> list:
    """Candidate functions for the evals, best first: the planner's pick, then the call site's callers
    (found with the AST), then the call site itself. Each must take plain data so the app builds its
    real prompts; _entry_problem() then confirms it reaches only this call site."""
    out, seen = [], set()

    def add(file: str, name: str):
        try:
            fn = getattr(capture.load_module(repo, file), name)
        except Exception:
            return
        if callable(fn) and (file, name) not in seen:
            seen.add((file, name))
            out.append(fn)

    ep = str(plan_entry.get("eval_entrypoint") or "")
    if ":" in ep:
        f, name = ep.rsplit(":", 1)
        add(f.strip(), name.strip().rstrip("()"))
    for file, name in scan.callers(repo, site.function):
        add(file, name)
    out = [f for f in out if f.__name__ != site.function]
    add(site.file, site.function)
    if not out or out[-1].__name__ != site.function:
        out.append(getattr(capture.load_module(repo, site.file), site.function))
    return out


def _eval_context(repo_files: dict, site, fn) -> str:
    return "\n\n".join(t for p, t in repo_files.items()
                       if not p.endswith(".py") or (p != site.file and (site.function in t or fn.__name__ in t)))


def _verify_requests(repo: Path, site_results: list[dict], decisions_by_file: dict) -> list[str]:
    """Run the migrated app against the fake SDK and check the request it would really send."""
    capture.forget_modules(repo)
    problems = []
    for r in site_results:
        d = decisions_by_file[r["file"]][r["function"]]
        entry = r.get("entry") or {"file": r["file"], "function": r["function"]}
        args = dict(r["inputs"][0])
        if entry["function"] == r["function"]:
            # evals drove the call site itself; if it takes the model as an argument, the caller now
            # passes the configured model, so replaying the old id from the eval input proves nothing
            for k, v in args.items():
                if v == r["source_model"]:
                    args[k] = _configured_model(repo, r, d)
        try:
            fn = getattr(capture.load_module(repo, entry["file"]), entry["function"])
            c = capture.capture(fn, args)
        except Exception as e:
            problems.append(f"{r['function']}: the migrated app failed before its LLM call: {type(e).__name__}: {e}")
            continue
        model = c.kwargs.get("model")
        thinking = ((c.kwargs.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking")
        if model != d["model"]:
            problems.append(f"{r['function']}: running the migrated app sends model={model!r}, expected "
                            f"{d['model']!r}; fix the config/default it is read from")
        if thinking is not d["thinking"]:
            problems.append(f"{r['function']}: running the migrated app sends enable_thinking={thinking!r}, "
                            f"expected {d['thinking']}")
        probe = r.get("probe")
        if probe:
            pargs = dict(r["inputs"][probe["input_index"]])
            for k, v in pargs.items():
                if v == r["source_model"] and entry["function"] == r["function"]:
                    pargs[k] = args.get(k, v)
            ok, value = capture.replay(fn, pargs, probe["text"])
            if not ok or evals.comparable(value) != probe["value"]:
                problems.append(f"{entry['function']}: behaviour changed. For the same model answer the original code "
                                f"returned {probe['value'][:160]!r}, the migrated code returns "
                                f"{(evals.comparable(value) if ok else value)[:160]!r}. Keep return values and "
                                "post-processing exactly as before")
        sent_max = c.kwargs.get("max_tokens")
        if isinstance(sent_max, int) and sent_max < d["max_tokens"]:
            problems.append(f"{r['function']}: running the migrated app sends max_tokens={sent_max}, below "
                            f"{d['max_tokens']} needed with thinking; raise it in the code or the config it reads")
    return problems


def _configured_model(repo: Path, r: dict, d: dict) -> str:
    """The model id the migrated app passes to this call site: the first known model id found in the
    repo's config files and callers. Returns the old id if nothing was changed, so the check fails."""
    known = [d["model"], r["source_model"], *llm.PRICES]
    for p, t in _context_files(repo).items():
        if p == r["file"]:
            continue
        for mid in known:
            if mid in t:
                return mid
    return r["source_model"]


def _close_call(r: dict) -> bool:
    """Is a much cheaper config only narrowly failing? Then more samples are worth their cost."""
    chosen = r["chosen"]
    return any(c["functional"] == 1.0 and not c["passes"] and c["cost_per_1k"] * 3 <= chosen["cost_per_1k"]
               and c["mean_score"] >= r["threshold"] - 1.0 for c in r["configs"])


def _entry_problem(fn, inputs: list[dict], site_function: str) -> str:
    """Why fn can't drive the evals for site_function ("" if it can): its one and only LLM request
    must come from the call site."""
    checked = 0
    for a in inputs[:3]:
        try:
            c = capture.capture(fn, a)
        except Exception:
            continue  # this generated input crashes before any LLM call; evaluate_site drops it
        if site_function not in c.callers:
            return f"its first LLM call comes from {next((x for x in c.callers if x != '<lambda>'), '?')}()"
        n = capture.count_calls(fn, a)
        if n != 1:
            return f"it makes {n} LLM calls per run"
        checked += 1
    return "" if checked else "none of its inputs reach an LLM call"


def _context_files(repo: Path) -> dict[str, str]:
    """Source plus small config/docs files the planner and rewriter may need to read."""
    out = {}
    for p in sorted(repo.rglob("*")):
        rel = p.relative_to(repo)
        if not p.is_file() or any(x in scan.SKIP_DIRS for x in rel.parts) or rel.parts[0] in ("tests", "test"):
            continue
        if p.suffix in (".py", ".ini", ".cfg", ".toml", ".yaml", ".yml", ".env.example") or p.name in (
                "requirements.txt", ".env.example"):
            if p.stat().st_size < 40_000 and p.name not in ("poetry.lock", "uv.lock"):
                out[rel.as_posix()] = p.read_text(encoding="utf-8", errors="replace")
    return out


def _summ(counts: dict) -> str:
    p, f = counts.get("passed", 0), counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    return f"{p}/{p + f} passed"


def _next_config(ranked: list[dict], cur: dict) -> dict | None:
    names = [(c["model"], c["thinking"]) for c in ranked]
    try:
        i = names.index((cur["model"], cur["thinking"]))
    except ValueError:
        return None
    cur_score = ranked[i]["mean_score"]
    for c in ranked[i + 1:]:
        if c["functional"] == 1.0 and c["mean_score"] >= cur_score:
            return c
    return ranked[-1] if i < len(ranked) - 1 else None


def _update_project_files(repo: Path):
    gi = repo / ".gitignore"
    have = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    missing = [x for x in (".openswitch/", "__pycache__/", ".env") if x not in have]
    if missing:
        gi.write_text("\n".join(have + missing) + "\n", encoding="utf-8")
    req = repo / "requirements.txt"
    still_anthropic = any("anthropic" in p.read_text(encoding="utf-8", errors="replace")
                          for p in repo.rglob("*.py") if not any(x in scan.SKIP_DIRS for x in p.parts)
                          and "tests" not in p.parts)
    if req.exists() and not still_anthropic:
        lines = [l for l in req.read_text(encoding="utf-8").splitlines() if not l.strip().startswith("anthropic")]
        req.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (repo / ".env.example").write_text(
        "NEBIUS_API_KEY=\nNEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1\n", encoding="utf-8")


def main(argv=None):
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser(prog="openswitch", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="migrate a repo to Nemotron")
    m.add_argument("repo", type=Path)
    m.add_argument("--inputs", type=int, default=6, help="eval inputs per call site")
    m.add_argument("--max-iters", type=int, default=3)
    m.add_argument("--reuse", action="store_true", help="reuse cached research + evals")
    a = ap.parse_args(argv)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    return migrate(a.repo.resolve(), a.inputs, a.max_iters, a.reuse)


if __name__ == "__main__":
    sys.exit(main())
