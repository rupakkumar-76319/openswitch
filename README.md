# OpenSwitch

**A coding agent that moves an app's OpenAI and Anthropic calls to NVIDIA Nemotron on Nebius Token Factory, then shows the app still works.**

Most teams want to try open models but don't, because swapping the model inside a working product is risky:
which calls will break, which ones need the big model, and is it actually cheaper? OpenSwitch answers those
questions with measurements on the app's own prompts. It then does the migration and runs the tests. You get
a branch to review that contains the new code, a cost/quality report and a list of known failure cases.

```
openswitch migrate path/to/your-repo
```

| | HelpDesk AI (6 call sites, OpenAI + Anthropic) | Issue Digest (config-driven model, no tests) |
|---|---|---|
| Estimated LLM cost per 1K calls | **$5.18 → $0.47 (91% lower)** | **$1.74 → $0.14 (92% lower)** |
| Tests after migration | **13/13 pass** (repo's own tests, unmodified) | **1/1 pass** (smoke tests written by OpenSwitch) |
| What the agent itself spent | $0.30, 5 min | $0.18, 2 min |

Sample reports: [`examples/reports/helpdesk-ai`](examples/reports/helpdesk-ai/OPENSWITCH_REPORT.md) and
[`examples/reports/issue-digest`](examples/reports/issue-digest/OPENSWITCH_REPORT.md).

## What it does

1. **Scan** (Python AST). Finds every `chat.completions.create`, `responses.create` and Anthropic `messages.create` call,
   with its model, parameters and enclosing function.
2. **Plan** (Nemotron 3 Ultra, thinking on). Reads each call in context and writes down:
   - the call's purpose
   - what downstream code relies on (the output contract)
   - a grading rubric
   - migration risks
   - the model that actually reaches the call (resolving config files and defaults)
   - which other files must change (config, price tables, `model.startswith("gpt")` checks)
   - an *eval entry point*: the outermost function that takes plain data, so the app builds its real prompt itself
3. **Research** (Tavily + Nemotron 3 Super). Live web search for:
   - what the current model costs
   - what Nemotron costs on Token Factory
   - the API differences that matter (endpoint, `response_format`, the thinking switch, how Anthropic parameters map)

   Every fact keeps its source URL, and the URLs appear in the report.
4. **Evaluate** (Nano, Super and Ultra, each with thinking on and off):
   - **Capture:** the agent runs the app's own functions against a fake SDK and captures the exact production
     request: real system prompt, real parameters, real response format. No OpenAI or Anthropic key is ever used.
   - **Entry point:** when the call sits inside a generic wrapper, the evals drive the outermost caller that
     builds the real prompt. The call stack confirms this caller's one and only LLM request is the target.
   - **Inputs:** Nemotron Super writes realistic inputs. Each one is first run through the app's code; inputs
     the app rejects go back to Super with the error.
   - **Run:** each captured prompt runs on 6 Nemotron configurations.
   - **Functional check:** each answer is fed back through the app's own post-processing (`json.loads`,
     `float()`, label checks...).
   - **Judge:** Nemotron Ultra marks each answer pass/fail on a checklist (correctness plus the call's rubric).
     Every configuration is graded, the strongest one included.
   - **Choose:** the cheapest configuration that has 100% functional answers, scores within 0.75 of the best
     configuration and has no answer below 5/10 wins.
   - **Adaptive sampling:** when a configuration at least 3x cheaper narrowly misses, OpenSwitch doubles the
     sample for that call and decides again. Extra spend only goes where the decision is close.
5. **Rewrite** (Ultra writes, an AST checker verifies). The checker mechanically confirms, for each call:
   - the chosen model id
   - the `enable_thinking` flag
   - enough `max_tokens` headroom for thinking
   - the original `temperature`/`top_p`
   - no leftover Anthropic import or OpenAI key

   If the model came from config, the checker requires that the config is changed and that the code doesn't
   hard-code the model. Then a behavioural check *runs* the migrated app against the fake SDK and confirms the
   request it would really send carries the decided model and thinking flag. A rejected rewrite goes back to
   Ultra with the list of problems.
6. **Test and iterate.** OpenSwitch runs `pytest`. If the repo has no tests, Ultra writes live smoke tests from
   the output contracts. When tests fail, Ultra diagnoses each failure as either a code bug (the file is
   rewritten with the fix) or a model that is too weak (that call moves up to the next measured configuration).
   It never edits the repo's own tests.
7. **Report and commit.** OpenSwitch creates a git branch `openswitch/nemotron` containing the migrated code,
   `OPENSWITCH_REPORT.md` (ready to use as a PR description), `.openswitch/report.html` and machine-readable
   `result.json`.

## How Nemotron is used (which model does what)

| Role | Model | Why this model |
|---|---|---|
| Planner | Nemotron 3 Ultra 550B, thinking on | Reads the whole repo and reasons about contracts and risks |
| Rewriter, debugger, test writer | Nemotron 3 Ultra 550B, thinking on | Writes code that must pass a strict checker and real tests |
| Judge | Nemotron 3 Ultra 550B, thinking off | Strongest grader; pass/fail checklists keep 200+ gradings cheap and repeatable |
| Researcher, eval-input writer | Nemotron 3 Super 120B | Long-context reading of search results, structured JSON |
| Candidates under test | Nano 30B, Super 120B, Ultra 550B, each with thinking on and off | The point of the project: measure, don't guess |

The reasoning switch (`chat_template_kwargs: {"enable_thinking": ...}`) is treated as a model choice of its own.
On HelpDesk's refund-policy math, Nemotron Nano *with* thinking scored 9.2/10 in the final run while Super
*without* thinking scored 3.3/10. Nano with thinking beat Super without thinking by more than 4 points in every
run we did. A small model with thinking can beat a bigger model without it. The migrated code sets this flag
explicitly per call, with enough `max_tokens` headroom for thinking.

Every call the agent makes goes through one ledger (`openswitch/llm.py`) tagged with its role, so the report shows
exactly which model did which job, how many tokens it used and what it cost.

## Where Nebius Token Factory helped

- **One OpenAI-compatible endpoint for every Nemotron size.** Nano, Super and Ultra share the same API, so trying
  6 configurations per call site is a loop, not 6 integrations. The migrated apps keep using the `openai` SDK and
  only change `base_url`, key and model.
- **Price per token low enough to evaluate before migrating.** A full migration of a 6-call-site app, including
  about 300 candidate runs, the judge calls, planning, research, rewriting and tests, cost the agent $0.30.
- **The thinking switch per request.** Turning reasoning on or off without changing models gives a cheaper path
  than moving up a model size, and OpenSwitch measures both.
- **Structured output (`response_format: json_object`)** works the same as on OpenAI, so JSON extraction calls
  migrate without code changes.

## Other Nebius services used

- **Tavily** (now part of Nebius): live web search during migration for current model prices and API
  differences, with source URLs carried into the report.
- Nebius AI Cloud (Serverless Jobs/Endpoints) is not used. OpenSwitch runs where the code lives. Running it as a
  Serverless Job against a repo URL is the obvious next step.

## Run it

Requires Python 3.10+.

```bash
git clone https://github.com/rupakkumar-76319/openswitch && cd openswitch
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[anthropic]"
cp .env.example .env                              # add NEBIUS_API_KEY and TAVILY_API_KEY
```

OpenSwitch edits the target repo and commits on a new branch `openswitch/nemotron` when the target is a git repo.
To try the examples without touching this checkout, copy one out first:

```bash
cp -r examples/helpdesk-ai /tmp/helpdesk-ai && cd /tmp/helpdesk-ai && git init -q && git add -A && git commit -qm before
openswitch migrate /tmp/helpdesk-ai
```

Options: `--inputs N` (eval inputs per call site, default 6), `--max-iters N` (test/fix iterations, default 3),
`--reuse` (reuse cached research and evals from `.openswitch/evals.json`).

OpenSwitch imports the target app's modules to capture its prompts, so only run it on code you trust, inside the
app's own Python environment.

## Limitations (measured, not hidden)

- **Python only**, and only the OpenAI and Anthropic Python SDKs. Streaming, tool calls and multi-call agent
  loops are not migrated yet.
- **No original-provider baseline.** Answers are compared with each other and with the call's rubric, not with
  the original OpenAI/Anthropic output (no such key is used). The repo's tests are the independent check.
  Judging uses the same model family, so it may favour Nemotron's style; pass/fail checklists and the app's
  own functional checks reduce that bias.
- **Evals are small (6 inputs per call site, 12 when the decision is close)**, so the scores carry noise.
  Between runs a call can land on a different configuration of similar cost. Across our final runs the total
  saving stayed above 90%, and the report shows every configuration's score so a reviewer can see how close
  each decision was.
- **Cheaper is not guaranteed for every call.** When no small configuration meets the quality bar, OpenSwitch
  picks a bigger one even if it costs more than the original (for example a `gpt-4o-mini` classifier). The report
  flags these calls as "costs more".
- **Baseline costs are estimates**: source-model token counts are approximated with Nemotron's tokenizer, and
  for o-series models the hidden reasoning is approximated with Ultra's thinking length.

## Repository layout

```
openswitch/        the agent
  scan.py          AST call-site finder
  capture.py       fake-SDK prompt capture and replay (functional checks)
  evals.py         planner, input writer, 6-config runs, judge, selection
  research.py      Tavily research
  rewrite.py       Ultra rewriter, AST checker, test writer, debugger
  report.py        Markdown/HTML/JSON report
  llm.py           Token Factory client + per-role cost ledger
examples/
  helpdesk-ai/     support-desk backend on gpt-4o, gpt-4o-mini, o3-mini and Claude, with live tests
  issue-digest/    GitHub issue digester whose model lives in settings.ini, no tests
  reports/         reports produced by OpenSwitch on both
```

## License

MIT
