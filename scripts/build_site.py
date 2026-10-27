"""Build the project page (docs/) from real OpenSwitch runs.

    python scripts/build_site.py <migrated repo> [<migrated repo> ...]

Copies each run's report into docs/reports/ and examples/reports/, and writes docs/index.html.
"""
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/rupakkumar-76319/openswitch"


def load(repo: Path) -> dict:
    d = json.loads((repo / ".openswitch" / "result.json").read_text(encoding="utf-8"))
    d["_repo"] = repo
    return d


def copy_reports(runs):
    (DOCS / "reports").mkdir(parents=True, exist_ok=True)
    for r in runs:
        name = r["repo_name"]
        shutil.copy(r["_repo"] / ".openswitch" / "report.html", DOCS / "reports" / f"{name}.html")
        dst = ROOT / "examples" / "reports" / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(r["_repo"] / "OPENSWITCH_REPORT.md", dst / "OPENSWITCH_REPORT.md")
        shutil.copy(r["_repo"] / ".openswitch" / "result.json", dst / "result.json")
        shutil.copy(r["_repo"] / ".openswitch" / "report.html", dst / "report.html")
        save_nemotron_code(r["_repo"], dst / "written-by-nemotron")


NEMOTRON_NOTE = """# Code written by NVIDIA Nemotron during this migration

`migration.diff` is the full change OpenSwitch committed on the `openswitch/nemotron` branch, and `files/`
holds the changed files after the migration.

- Application code, config and dependency changes, and `tests/test_openswitch_smoke.py` (if present), were
  written by **Nemotron 3 Ultra 550B** (thinking on) through Nebius Token Factory. That covers the rewriter,
  debugger and test-writer roles. Every one of these files passed OpenSwitch's AST checker and behavioural
  check before it was kept.
- `.env.example` and `tests/conftest.py` come from OpenSwitch's own templates, not from a model.
"""


def save_nemotron_code(repo: Path, dst: Path):
    """Keep the code Nemotron 3 Ultra wrote during the migration: the full diff and the changed files."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    diff = subprocess.run(["git", "diff", "main", "--", ".", ":(exclude)OPENSWITCH_REPORT.md"], cwd=repo,
                          capture_output=True, text=True, encoding="utf-8").stdout
    (dst / "migration.diff").write_text(diff, encoding="utf-8")
    names = subprocess.run(["git", "diff", "main", "--name-only"], cwd=repo, capture_output=True, text=True).stdout
    for name in names.split():
        if name == "OPENSWITCH_REPORT.md" or not (repo / name).is_file():
            continue
        out = dst / "files" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / name, out)
    (dst / "README.md").write_text(NEMOTRON_NOTE, encoding="utf-8")


def card(r) -> str:
    e = html.escape
    s = r["summary"]
    rows = ""
    for site in r["sites"]:
        c = site["chosen"]
        rows += (f"<tr><td><code>{e(site['function'])}</code></td><td>{e(site['source_model'])}</td>"
                 f"<td><span class='pill'>{e(c['config'])}</span></td><td>{c['mean_score']:.1f}</td></tr>")
    tests_note = "smoke tests written by OpenSwitch" if r["tests"].get("written") else "the repo's own tests, unmodified"
    return f"""
<section class="run">
  <div class="run-head"><h3>{e(r['repo_name'])}</h3>
  <a href="reports/{e(r['repo_name'])}.html">Full report →</a></div>
  <div class="kpis">
    <div class="kpi"><b>{s['saving_pct']:.0f}%</b><span>lower est. LLM cost<br>${s['total_before']:.2f} → ${s['total_after']:.2f} per 1K calls</span></div>
    <div class="kpi"><b>{e(r['tests']['summary'])}</b><span>{tests_note}</span></div>
    <div class="kpi"><b>${sum(x['cost'] for x in r['ledger']):.2f}</b><span>spent by the agent, in {r['seconds']}s</span></div>
  </div>
  <div class="tbl"><table><tr><th>Call site</th><th>Was</th><th>Now</th><th>Score</th></tr>{rows}</table></div>
</section>"""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenSwitch</title>
<meta name="description" content="A coding agent that migrates your app's OpenAI and Anthropic calls to NVIDIA Nemotron on Nebius Token Factory, and shows it still works.">
<style>
:root{--bg:#0e1013;--card:#161a20;--ink:#eef0f4;--mute:#9aa3b2;--line:#252b35;--green:#76b900;--blue:#5b9bff}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f7f8fa;--card:#fff;--ink:#12151b;--mute:#5b6372;--line:#e4e7ec}}
:root[data-theme="dark"]{--bg:#0e1013;--card:#161a20;--ink:#eef0f4;--mute:#9aa3b2;--line:#252b35}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 Inter,system-ui,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:40px 16px 80px}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.tag{display:inline-block;font-size:13px;color:var(--green);border:1px solid var(--green);border-radius:99px;padding:2px 10px}
h1{font-size:44px;line-height:1.1;margin:16px 0 12px;letter-spacing:-.02em}h2{font-size:24px;margin:56px 0 14px}h3{margin:0;font-size:18px}
.lead{font-size:19px;color:var(--mute);max-width:720px}.cta{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}
.btn{padding:10px 16px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);font-weight:600}
.btn.primary{background:var(--green);border-color:var(--green);color:#0b0d10}
pre{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;overflow-x:auto;font-size:14px}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;counter-reset:s}
.step{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.step b{display:block;margin-bottom:4px}.step b:before{counter-increment:s;content:counter(s) ". ";color:var(--green)}
.step p{margin:0;color:var(--mute);font-size:15px}.who{font-size:12px;color:var(--green)}
.run{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0}
.run-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:12px}
.kpi{border:1px solid var(--line);border-radius:10px;padding:12px}.kpi b{display:block;font-size:26px}.kpi span{color:var(--mute);font-size:13px}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:600}.pill{padding:2px 8px;border-radius:99px;font-size:12px;background:rgba(118,185,0,.15);color:var(--green)}
ul{padding-left:20px}li{margin:6px 0}.mute{color:var(--mute)}footer{margin-top:60px;color:var(--mute);font-size:14px}
</style></head><body><main>
<span class="tag">Nebius × NVIDIA Global AI Hackathon · Coding &amp; Agentic Engineering</span>
<h1>Switch your app to open models.<br>Keep the tests green.</h1>
<p class="lead">OpenSwitch is a coding agent that moves an app's OpenAI and Anthropic calls to NVIDIA Nemotron on
Nebius Token Factory. It measures every Nemotron size, with and without thinking, on the app's <em>own</em> prompts,
picks the cheapest configuration that keeps quality, rewrites the code, runs the tests and fixes what breaks.</p>
<div class="cta"><a class="btn primary" href="REPO_URL">GitHub repo</a>
<a class="btn" href="reports/helpdesk-ai.html">Sample report</a></div>
<pre>$ openswitch migrate ./your-app</pre>

<h2>Results on two apps</h2>
RUNS

<h2>All AI work runs on NVIDIA Nemotron</h2>
<p class="mute">Every model call OpenSwitch makes goes through a Nebius Token Factory API key to Nemotron. No OpenAI or
Anthropic key is used at any step. These are the totals from the ledgers of the two runs above.</p>
NEMOTRON

<h2>How it works</h2>
<div class="steps">
<div class="step"><b>Scan</b><p>The Python AST finds every OpenAI/Anthropic call, with its model and parameters.</p></div>
<div class="step"><b>Plan</b><span class="who">Nemotron 3 Ultra · thinking</span><p>Purpose, output contract, rubric and risks for every call. Resolves models set in config files.</p></div>
<div class="step"><b>Research</b><span class="who">Tavily + Nemotron 3 Super</span><p>Live prices for the current model and for Nemotron, plus the API differences that matter, with sources.</p></div>
<div class="step"><b>Evaluate</b><span class="who">Nano · Super · Ultra × thinking on/off</span><p>Captures the app's real prompts with a fake SDK, runs 6 configs, checks each answer through the app's own code, Ultra judges.</p></div>
<div class="step"><b>Rewrite</b><span class="who">Ultra writes · AST checker verifies</span><p>Model ids, thinking flags, token headroom, config files and price tables, all mechanically checked.</p></div>
<div class="step"><b>Test &amp; iterate</b><span class="who">pytest · Ultra debugs</span><p>Runs the repo's tests (or writes smoke tests). On a failure it fixes the code or moves that call to a stronger config.</p></div>
</div>

<h2>Why it matters</h2>
<ul>
<li><b>Measured, not guessed.</b> On refund-policy math, Nemotron Nano <em>with thinking</em> scored 10/10 while Super <em>without</em> thinking scored 1.7/10. Thinking is a choice per call, and OpenSwitch measures it.</li>
<li><b>Spends where it matters.</b> When a much cheaper config narrowly misses, OpenSwitch doubles the sample for that call before deciding. In the HelpDesk run, that moved <code>draft_reply</code> from Super to Nano (both with thinking): about 4x cheaper, still passing.</li>
<li><b>Honest output.</b> The report shows every configuration's score, flags calls that cost more after the switch, and lists the failure cases it found.</li>
<li><b>Nothing to trust blindly.</b> You get a branch to review: a normal diff, a PR description and the test log.</li>
</ul>
<footer>MIT licensed · <a href="REPO_URL">REPO_URL</a></footer>
</main></body></html>"""


ROLE_LABELS = {"judge": "Judge (checklist grading)", "candidate": "Candidates under test",
               "eval-writer": "Eval-input writer", "researcher": "Researcher (reads Tavily results)",
               "planner": "Planner", "rewriter": "Rewriter (writes the code changes)", "test-writer": "Test writer",
               "debugger": "Debugger"}


def nemotron_block(runs) -> str:
    """Role x model totals across all runs, straight from the ledgers."""
    agg: dict = {}
    for r in runs:
        for x in r["ledger"]:
            a = agg.setdefault((x["role"], x["model"]), [0, 0, 0.0])
            a[0] += x["calls"]
            a[1] += x["tokens"]
            a[2] += x["cost"]
    rows = "".join(
        f"<tr><td>{html.escape(ROLE_LABELS.get(role, role))}</td><td>Nemotron 3 {html.escape(model)}</td>"
        f"<td>{c:,}</td><td>{t:,}</td><td>${cost:.3f}</td></tr>"
        for (role, model), (c, t, cost) in sorted(agg.items(), key=lambda kv: -kv[1][1]))
    tc, tt, tcost = (sum(v[i] for v in agg.values()) for i in range(3))
    rows += f"<tr><th>Total</th><th></th><th>{tc:,}</th><th>{tt:,}</th><th>${tcost:.2f}</th></tr>"
    return (f'<div class="run"><div class="tbl"><table><tr><th>Role</th><th>Model</th><th>Calls</th><th>Tokens</th>'
            f"<th>Cost</th></tr>{rows}</table></div><p class=\"mute\">The code changes in each migration were written "
            f'by Nemotron 3 Ultra; see <a href="REPO_URL/tree/main/examples/reports">written-by-nemotron/</a> in the '
            f"repo.</p></div>")


def main():
    runs = [load(Path(p)) for p in sys.argv[1:]]
    copy_reports(runs)
    page = PAGE.replace("RUNS", "".join(card(r) for r in runs)).replace("NEMOTRON", nemotron_block(runs))
    page = page.replace("REPO_URL", REPO_URL)
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print("wrote", DOCS / "index.html")


if __name__ == "__main__":
    main()
