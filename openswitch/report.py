"""Cost/quality report: Markdown for the pull request, HTML for humans, JSON for machines."""
from __future__ import annotations

import html
import json
from pathlib import Path

from . import llm
from .evals import CANDIDATES, MARGIN, PASS_FLOOR, PASS_MIN, label


def source_cost_per_1k(site: dict, prices: dict) -> float | None:
    p = prices.get(site["source_model"])
    if not p:
        return None
    out = site["ultra_fast_out_tokens"]
    if site["source_model"].startswith(("o1", "o3", "o4")):  # hidden reasoning tokens are billed too
        out = site["reference_out_tokens"]
    return (site["avg_in_tokens"] * p["input"] + out * p["output"]) / 1e6 * 1000


def build(result: dict) -> dict:
    prices = result["research"]["source_prices"]
    rows = []
    for s in result["sites"]:
        before = source_cost_per_1k(s, prices)
        after = s["chosen"]["cost_per_1k"]
        rows.append({**s, "before": before, "after": after})
    tb = sum(r["before"] or 0 for r in rows)
    ta = sum(r["after"] for r in rows if r["before"] is not None)
    return {"rows": rows, "total_before": tb, "total_after": ta,
            "saving_pct": (1 - ta / tb) * 100 if tb else None}


def ledger_rows() -> list[dict]:
    rows = [{"role": r, "model": llm.SHORT.get(m, m), "calls": v["calls"], "tokens": v["tokens"], "cost": v["cost"]}
            for (r, m), v in llm.LEDGER.by_role().items()]
    return sorted(rows, key=lambda x: -x["cost"])


def markdown(result: dict, b: dict) -> str:
    L = ["# OpenSwitch: migrate LLM calls to NVIDIA Nemotron on Nebius Token Factory", ""]
    if b["saving_pct"] is not None:
        L += [f"**Estimated LLM cost: ${b['total_before']:.3f} → ${b['total_after']:.3f} per 1,000 calls of each "
              f"function ({b['saving_pct']:.0f}% lower).** Tests: {result['tests']['summary']}.", ""]
    L += ["| Function | Before | After (Nemotron) | Eval score | Functional | $/1K before | $/1K after |",
          "|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        c = r["chosen"]
        before = f"{r['before']:.4f}" if r["before"] is not None else "n/a"
        more = " ⚠ costs more" if r["before"] is not None and r["after"] > r["before"] else ""
        L.append(f"| `{r['function']}`{more} | {r['source_model']} | {c['config']} | {c['mean_score']:.1f}/10 "
                 f"(min {c['min_score']:.0f}) | {c['functional'] * 100:.0f}% | {before} | {c['cost_per_1k']:.4f} |")
    L += ["", "## How each model was chosen",
          f"Every function was run on {len(CANDIDATES)} Nemotron configs with {result['sites'][0]['n_inputs']} inputs "
          "generated from its own real prompts (captured by running the app's code against a fake SDK). Each output "
          "was fed back through the app's own post-processing (functional check) and checked by Nemotron 3 Ultra "
          "against a pass/fail checklist (correctness plus the call's rubric); score = share of checks passed. "
          "Every config is graded, the strongest one included. The cheapest config with 100% functional outputs, "
          f"a mean within {MARGIN} of the best config (and ≥ {PASS_FLOOR}) and no output below {PASS_MIN} was chosen.", ""]
    for r in b["rows"]:
        L.append(f"<details><summary><code>{r['function']}</code>: all configs</summary>\n")
        L.append("| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |")
        L.append("|---|---|---|---|---|---|---|")
        for c in r["configs"]:
            L.append(f"| {c['config']} | {c['functional'] * 100:.0f}% | {c['mean_score']:.1f} | {c['min_score']:.0f} | "
                     f"{c['avg_out']} | {c['cost_per_1k']:.4f} | {'✅' if c['passes'] else '❌'} |")
        L.append("\n</details>\n")
    L += ["## Tests", f"`pytest`: **{result['tests']['summary']}** after {result['tests']['iterations']} iteration(s)."]
    for it in result["tests"]["history"]:
        L.append(f"- Iteration {it['n']}: {it['summary']}" + (f": {it['action']}" if it.get("action") else ""))
    L += ["", "## Known failures and limits"]
    for r in b["rows"]:
        bad = [f for f in r["failures"] if f["config"] == r["chosen"]["config"]]
        for f in bad[:3]:
            L.append(f"- `{r['function']}` ({f['config']}): {f['reason']}")
    for lim in result["limitations"]:
        L.append(f"- {lim}")
    L += ["", "## Which Nemotron did what during this migration", "| Role | Model | Calls | Tokens | Cost |",
          "|---|---|---|---|---|"]
    for x in ledger_rows():
        L.append(f"| {x['role']} | {x['model']} | {x['calls']} | {x['tokens']:,} | ${x['cost']:.4f} |")
    L.append(f"\nTotal migration cost: **${llm.LEDGER.total:.3f}**.")
    L += ["", "## Sources (Tavily)"]
    for m, p in result["research"]["source_prices"].items():
        L.append(f"- {m}: ${p['input']}/${p['output']} per 1M tokens: {p['url']}")
    for m, p in result["research"]["nemotron_prices"].items():
        L.append(f"- {llm.SHORT.get(m, m)}: ${p['input']}/${p['output']} per 1M tokens: {p['url']}")
    for n in result["research"]["notes"]:
        L.append(f"- {n['fact']} ({n['url']})")
    L += ["", "_Generated by [OpenSwitch](https://github.com/rupakkumar-76319/openswitch)._"]
    return "\n".join(L)


CSS = """
:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaf0;--mute:#9aa3b2;--line:#262b36;--green:#76b900;--red:#ff5d5d;--blue:#4c8dff;--amber:#ffb020}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f6f7f9;--card:#fff;--ink:#14171f;--mute:#5b6372;--line:#e3e6ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 10px}
.sub{color:var(--mute);margin:0 0 22px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}.kpi b{display:block;font-size:28px}
.kpi span{color:var(--mute);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:600}.bar{height:10px;border-radius:5px;background:var(--red);display:inline-block;vertical-align:middle}
.bar.after{background:var(--green)}.pill{padding:2px 8px;border-radius:99px;font-size:12px;background:rgba(118,185,0,.15);color:var(--green)}
.cell{display:inline-block;min-width:44px;text-align:center;border-radius:6px;padding:2px 6px;font-weight:600}
.chosen{outline:2px solid var(--green)}.warn{margin-left:6px;padding:1px 7px;border-radius:99px;font-size:11px;background:rgba(255,176,32,.18);color:var(--amber)}.mute{color:var(--mute)}a{color:var(--blue)}code{font-size:13px}
ul{padding-left:18px}li{margin:4px 0}
"""


def _cell(c):
    s = c["mean_score"]
    col = "rgba(118,185,0,.25)" if c["passes"] else ("rgba(255,176,32,.22)" if s >= 6 else "rgba(255,93,93,.22)")
    return f'<span class="cell" style="background:{col}">{s:.1f}</span>'


def html_page(result: dict, b: dict) -> str:
    e = html.escape
    mx = max([r["before"] or 0 for r in b["rows"]] + [r["after"] for r in b["rows"]] + [1e-9])
    saving = f"{b['saving_pct']:.0f}%" if b["saving_pct"] is not None else "n/a"
    rows = ""
    for r in b["rows"]:
        bw = (r["before"] or 0) / mx * 260
        aw = r["after"] / mx * 260
        warn = (" <span class='warn' title='Cheaper Nemotron configs missed the quality bar on this call, so it "
                "costs more than before; see the matrix below.'>costs more</span>"
                if r["before"] is not None and r["after"] > r["before"] else "")
        rows += (f"<tr><td><code>{e(r['function'])}</code>{warn}</td><td>{e(r['source_model'])}</td>"
                 f"<td><span class='pill'>{e(r['chosen']['config'])}</span></td>"
                 f"<td>{r['chosen']['mean_score']:.1f}</td>"
                 f"<td><span class='bar' style='width:{bw:.0f}px'></span> ${(r['before'] or 0):.4f}<br>"
                 f"<span class='bar after' style='width:{max(aw, 2):.0f}px'></span> ${r['after']:.4f}</td></tr>")
    head = "".join(f"<th>{e(label(m, t))}</th>" for m, t in CANDIDATES)
    matrix = ""
    for r in b["rows"]:
        cells = ""
        for c in r["configs"]:
            cls = " chosen" if c["config"] == r["chosen"]["config"] else ""
            cells += f"<td class='{cls.strip()}'>{_cell(c)}<br><span class='mute'>${c['cost_per_1k']:.4f}</span></td>"
        matrix += f"<tr><td><code>{e(r['function'])}</code></td>{cells}</tr>"
    led = "".join(f"<tr><td>{e(x['role'])}</td><td>{e(x['model'])}</td><td>{x['calls']}</td><td>{x['tokens']:,}</td>"
                  f"<td>${x['cost']:.4f}</td></tr>" for x in ledger_rows())
    fails = ""
    for r in b["rows"]:
        for f in r["failures"][:4]:
            fails += (f"<li><code>{e(r['function'])}</code> · {e(f['config'])} · score {f['score']:.0f}: "
                      f"{e(str(f['reason'])[:220])}</li>")
    lims = "".join(f"<li>{e(x)}</li>" for x in result["limitations"])
    hist = "".join(f"<li>Iteration {it['n']}: {e(it['summary'])}{(': ' + e(it['action'])) if it.get('action') else ''}</li>"
                   for it in result["tests"]["history"])
    src = "".join(f"<li>{e(m)}: ${p['input']} / ${p['output']} per 1M · <a href='{e(p['url'])}'>{e(p['url'][:70])}</a></li>"
                  for m, p in result["research"]["source_prices"].items())
    src += "".join(f"<li>{e(n['fact'])} · <a href='{e(n['url'])}'>source</a></li>" for n in result["research"]["notes"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenSwitch Report</title><style>{CSS}</style></head>
<body><main>
<h1>OpenSwitch migration report</h1>
<p class="sub">{e(result['repo_name'])} · OpenAI / Anthropic → NVIDIA Nemotron on Nebius Token Factory</p>
<div class="kpis">
<div class="kpi"><b>{saving}</b><span>lower LLM cost (est. per 1K calls)</span></div>
<div class="kpi"><b>{e(result['tests']['summary'])}</b><span>{"live smoke tests written by OpenSwitch (repo had none)" if result['tests'].get('written') else "repo's own tests, unmodified"}</span></div>
<div class="kpi"><b>{len(b['rows'])}</b><span>call sites migrated</span></div>
<div class="kpi"><b>${llm.LEDGER.total:.2f}</b><span>spent by the agent to do it</span></div>
</div>
<h2>Per call site</h2><div class="card"><table><tr><th>Function</th><th>Before</th><th>Now runs on</th><th>Score</th><th>$ per 1K calls (red before, green after)</th></tr>{rows}</table></div>
<h2>Every config, measured</h2><p class="mute">Checklist score (0-10, share of pass/fail checks met) on inputs built from the app's own prompts; green = passes (100% functional, mean within {MARGIN} of the best config and ≥ {PASS_FLOOR}, no output below {PASS_MIN}); outlined = chosen (cheapest passing).</p>
<div class="card"><table><tr><th>Function</th>{head}</tr>{matrix}</table></div>
<h2>Tests</h2><div class="card"><ul>{hist}</ul></div>
<h2>Which Nemotron did what</h2><div class="card"><table><tr><th>Role</th><th>Model</th><th>Calls</th><th>Tokens</th><th>Cost</th></tr>{led}</table></div>
<h2>Failure cases we found</h2><div class="card"><ul>{fails or '<li>None below threshold.</li>'}</ul><ul>{lims}</ul></div>
<h2>Sources (live Tavily research)</h2><div class="card"><ul>{src}</ul></div>
</main></body></html>"""


def write(repo: Path, result: dict) -> Path:
    b = build(result)
    out = repo / ".openswitch"
    out.mkdir(exist_ok=True)
    md = markdown(result, b)
    (repo / "OPENSWITCH_REPORT.md").write_text(md, encoding="utf-8")
    (out / "report.html").write_text(html_page(result, b), encoding="utf-8")
    (out / "result.json").write_text(json.dumps({**result, "summary": {k: v for k, v in b.items() if k != "rows"},
                                                 "ledger": ledger_rows()}, indent=2, default=str), encoding="utf-8")
    return out / "report.html"
