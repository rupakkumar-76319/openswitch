# OpenSwitch: migrate LLM calls to NVIDIA Nemotron on Nebius Token Factory

**Estimated LLM cost: $1.845 → $0.058 per 1,000 calls of each function (97% lower).** Tests: 2/2 passed.

| Function | Before | After (Nemotron) | Eval score | Functional | $/1K before | $/1K after |
|---|---|---|---|---|---|---|
| `complete` | gpt-4o | Nano 30B fast | 8.3/10 (min 5) | 100% | 1.8446 | 0.0583 |

## How each model was chosen
Every function was run on 6 Nemotron configs with 6 inputs generated from its own real prompts (captured by running the app's code against a fake SDK). Each output was fed back through the app's own post-processing (functional check) and checked by Nemotron 3 Ultra against a pass/fail checklist (correctness plus the call's rubric); score = share of checks passed. Every config is graded, the strongest one included. The cheapest config with 100% functional outputs, a mean within 0.75 of the best config (and ≥ 7.5) and no output below 5.0 was chosen.

<details><summary><code>complete</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 8.3 | 5 | 181 | 0.0583 | ✅ |
| Nano 30B think | 100% | 8.6 | 8 | 507 | 0.1365 | ✅ |
| Super 120B fast | 100% | 8.1 | 7 | 125 | 0.1862 | ✅ |
| Super 120B think | 100% | 6.7 | 3 | 421 | 0.4527 | ❌ |
| Ultra 550B fast | 100% | 8.1 | 7 | 123 | 0.6150 | ✅ |
| Ultra 550B think | 100% | 8.6 | 7 | 356 | 1.3145 | ✅ |

</details>

## Tests
`pytest`: **2/2 passed** after 1 iteration(s).
- Iteration 1: 2/2 passed: wrote live smoke tests (repo had none)

## Known failures and limits
- `complete` (Nano 30B fast): The output adds an assignee not present in the input, the Problem sentence exceeds 40 words, the Details section includes code and stack trace not present in the issue, and the Next step describes two actions instead of one concrete step.
- `complete` (Nano 30B fast): All checks pass: the output contains exactly four ## headings in the correct order; the Problem section is a single sentence of 18 words; the Details section accurately notes that no error was reported and quotes the implementation suggestion; the Next step describes one concrete action; and no facts are hallucinated beyond the issue content.
- `complete` (Nano 30B fast): Problem section contains 17 words, exceeding the 40-word limit by a small margin but the instruction says 'at most 40 words' and this is within that; however the instruction also says 'one sentence' and the Problem section is one sentence, so this passes. Wait, recount: 'The UI intermittently freezes for a few seconds when the save button is clicked.' That's 14 words. So it's within 40 words. The failure is that Details section does not quote relevant code or error from the issue; it only summarizes observations. The instruction requires quoting relevant code or error, which is absent. Therefore check 3 fails.
- Baseline cost is an estimate: source-model token counts are approximated with Nemotron's tokenizer, and for o-series models hidden reasoning is approximated with Nemotron Ultra's thinking length.
- The reference answers come from Nemotron 3 Ultra with thinking, not from the original provider (no OpenAI/Anthropic key is needed or used); the repo's own tests are the independent check.
- Each function was evaluated on 6 generated inputs; rare inputs can still behave differently, so keep monitoring after merging.

## Which Nemotron did what during this migration
| Role | Model | Calls | Tokens | Cost |
|---|---|---|---|---|
| judge | Ultra 550B | 36 | 31,393 | $0.0352 |
| candidate | Ultra 550B | 12 | 5,831 | $0.0116 |
| rewriter | Ultra 550B | 1 | 3,080 | $0.0055 |
| test-writer | Ultra 550B | 1 | 2,818 | $0.0054 |
| planner | Ultra 550B | 1 | 2,460 | $0.0044 |
| candidate | Super 120B | 12 | 6,231 | $0.0038 |
| researcher | Super 120B | 3 | 10,904 | $0.0036 |
| eval-writer | Super 120B | 1 | 3,034 | $0.0023 |
| candidate | Nano 30B | 12 | 7,087 | $0.0012 |

Total migration cost: **$0.073**.

## Sources (Tavily)
- gpt-4o: $2.5/$10.0 per 1M tokens: https://pricepertoken.com/pricing-page/model/openai-gpt-4o
- Nano 30B: $0.06/$0.24 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- 3.5 Lightning: $0.06/$0.24 per 1M tokens: https://mastra.ai/models/providers/nebius
- Super 120B: $0.3/$0.9 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- Ultra 550B: $1.0/$3.0 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- Endpoint URL is https://api.studio.nebius.com/v1/ ()
- Authentication uses Authorization: Bearer <NEBIUS_API_KEY> ()
- Supported parameters include max_tokens, temperature, top_p, top_k, n, stream, stream_options, stop, presence_penalty, frequency_penalty, logit_bias, logprobs, top_logprobs, user, and extra_body ()
- JSON output can be enabled via response_format: { type: "json_object" } or response_format: { type: "json_schema", json_schema: <schema> } ()
- Nemotron thinking can be disabled by setting extra_body.chat_template_kwargs.enable_thinking: false ()
- For Nemotron models, when enable_thinking is false, force_nonempty_content may be needed to avoid empty content due to parser behavior ()
- Anthropic-specific parameters like thinking are not supported; use chat_template_kwargs instead for reasoning control ()
- The OpenAI-compatible API forwards chat_template_kwargs to vLLM for Nemotron models when properly configured ()

_Generated by [OpenSwitch](https://github.com/rupakkumar-76319/openswitch)._