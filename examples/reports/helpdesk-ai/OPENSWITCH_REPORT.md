# OpenSwitch: migrate LLM calls to NVIDIA Nemotron on Nebius Token Factory

**Estimated LLM cost: $7.450 → $0.390 per 1,000 calls of each function (95% lower).** Tests: 13/13 passed.

| Function | Before | After (Nemotron) | Eval score | Functional | $/1K before | $/1K after |
|---|---|---|---|---|---|---|
| `classify_ticket` ⚠ costs more | gpt-4o-mini | Super 120B fast | 9.3/10 (min 6) | 100% | 0.0102 | 0.0194 |
| `extract_order` | gpt-4o | Nano 30B fast | 9.7/10 (min 8) | 100% | 0.6725 | 0.0161 |
| `refund_amount` | o3-mini | Nano 30B think | 10.0/10 (min 10) | 100% | 4.1858 | 0.1865 |
| `summarize_thread` | gpt-4o | Super 120B fast | 9.7/10 (min 8) | 100% | 0.7354 | 0.1041 |
| `draft_reply` | gpt-4o | Nano 30B think | 9.5/10 (min 6) | 100% | 1.1321 | 0.0487 |
| `translate_reply` | claude-sonnet-4-5 | Nano 30B fast | 10.0/10 (min 10) | 100% | 0.7145 | 0.0148 |

## How each model was chosen
Every function was run on 6 Nemotron configs with 6 inputs generated from its own real prompts (captured by running the app's code against a fake SDK). Each output was fed back through the app's own post-processing (functional check) and checked by Nemotron 3 Ultra against a pass/fail checklist (correctness plus the call's rubric); score = share of checks passed. Every config is graded, the strongest one included. The cheapest config with 100% functional outputs, a mean within 0.75 of the best config (and ≥ 7.5) and no output below 5.0 was chosen.

<details><summary><code>classify_ticket</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 7.7 | 4 | 3 | 0.0041 | ❌ |
| Nano 30B think | 100% | 8.0 | 6 | 77 | 0.0219 | ❌ |
| Super 120B fast | 100% | 9.3 | 6 | 2 | 0.0194 | ✅ |
| Super 120B think | 100% | 10.0 | 10 | 123 | 0.1278 | ✅ |
| Ultra 550B fast | 100% | 9.3 | 6 | 3 | 0.0653 | ✅ |
| Ultra 550B think | 100% | 10.0 | 10 | 82 | 0.3033 | ✅ |

</details>

<details><summary><code>extract_order</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 9.7 | 8 | 44 | 0.0161 | ✅ |
| Nano 30B think | 100% | 10.0 | 10 | 239 | 0.0630 | ✅ |
| Super 120B fast | 100% | 10.0 | 10 | 38 | 0.0621 | ✅ |
| Super 120B think | 100% | 10.0 | 10 | 146 | 0.1593 | ✅ |
| Ultra 550B fast | 100% | 10.0 | 10 | 44 | 0.2252 | ✅ |
| Ultra 550B think | 100% | 10.0 | 10 | 209 | 0.7202 | ✅ |

</details>

<details><summary><code>refund_amount</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 5.7 | 4 | 5 | 0.0117 | ❌ |
| Nano 30B think | 100% | 10.0 | 10 | 733 | 0.1865 | ✅ |
| Super 120B fast | 17% | 1.7 | 0 | 746 | 0.7245 | ❌ |
| Super 120B think | 83% | 8.3 | 0 | 326 | 0.3467 | ❌ |
| Ultra 550B fast | 83% | 7.3 | 0 | 14 | 0.2200 | ❌ |
| Ultra 550B think | 100% | 10.0 | 10 | 1858 | 5.7520 | ✅ |

</details>

<details><summary><code>summarize_thread</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 7.0 | 0 | 63 | 0.0206 | ❌ |
| Nano 30B think | 100% | 8.7 | 4 | 233 | 0.0616 | ❌ |
| Super 120B fast | 100% | 9.7 | 8 | 85 | 0.1041 | ✅ |
| Super 120B think | 100% | 9.7 | 8 | 136 | 0.1505 | ✅ |
| Ultra 550B fast | 100% | 9.7 | 8 | 50 | 0.2438 | ✅ |
| Ultra 550B think | 100% | 8.7 | 4 | 142 | 0.5183 | ❌ |

</details>

<details><summary><code>draft_reply</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 8.7 | 4 | 85 | 0.0264 | ❌ |
| Nano 30B think | 100% | 9.5 | 6 | 178 | 0.0487 | ✅ |
| Super 120B fast | 100% | 9.5 | 6 | 107 | 0.1268 | ✅ |
| Super 120B think | 100% | 10.0 | 10 | 213 | 0.2216 | ✅ |
| Ultra 550B fast | 100% | 9.3 | 6 | 88 | 0.3648 | ✅ |
| Ultra 550B think | 100% | 10.0 | 10 | 251 | 0.8538 | ✅ |

</details>

<details><summary><code>translate_reply</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 10.0 | 10 | 46 | 0.0148 | ✅ |
| Nano 30B think | 100% | 9.3 | 6 | 220 | 0.0567 | ✅ |
| Super 120B fast | 100% | 8.7 | 4 | 35 | 0.0503 | ❌ |
| Super 120B think | 100% | 8.7 | 2 | 187 | 0.1873 | ❌ |
| Ultra 550B fast | 100% | 9.3 | 8 | 35 | 0.1678 | ✅ |
| Ultra 550B think | 100% | 8.7 | 2 | 413 | 1.3008 | ❌ |

</details>

## Tests
`pytest`: **13/13 passed** after 1 iteration(s).
- Iteration 1: 13/13 passed

## Known failures and limits
- `classify_ticket` (Super 120B fast): The output 'shipping' does not map to any keywords in the input (e.g., 'shipping', 'delivery', 'tracking'), and the request about a 'problem with purchase' generally maps to 'refund' or 'technical', making 'shipping' an incorrect classification.
- `extract_order` (Nano 30B fast): Check 0 failed: product value 'B' does not match the full product name 'Product B' stated in the input text, and quantity value 1 is hallucinated as the input contains no quantity information.
- `summarize_thread` (Super 120B fast): The output contains internal jargon and agent instructions by explicitly listing diagnostic categories ('hardware faults, software glitches, power issues, or compatibility problems') and prescribing next-step actions ('determine whether a repair, replacement, or refund is warranted'), which goes beyond summarizing the thread.
- `draft_reply` (Nano 30B think): The reply incorrectly asks the customer to provide the order number (#778899) which is already present in the internal summary and should be referenced directly by the agent.
- `draft_reply` (Nano 30B think): The output contains approximately 128 words, exceeding the 120-word limit.
- Baseline cost is an estimate: source-model token counts are approximated with Nemotron's tokenizer, and for o-series models hidden reasoning is approximated with Nemotron Ultra's thinking length.
- The reference answers come from Nemotron 3 Ultra with thinking, not from the original provider (no OpenAI/Anthropic key is needed or used); the repo's own tests are the independent check.
- Each function was evaluated on 6 generated inputs; rare inputs can still behave differently, so keep monitoring after merging.

## Which Nemotron did what during this migration
| Role | Model | Calls | Tokens | Cost |
|---|---|---|---|---|
| judge | Ultra 550B | 200 | 105,925 | $0.1221 |
| candidate | Ultra 550B | 96 | 32,114 | $0.0776 |
| candidate | Super 120B | 96 | 26,115 | $0.0179 |
| eval-writer | Super 120B | 13 | 20,312 | $0.0137 |
| rewriter | Ultra 550B | 1 | 4,988 | $0.0100 |
| planner | Ultra 550B | 1 | 5,090 | $0.0098 |
| researcher | Super 120B | 6 | 21,862 | $0.0070 |
| candidate | Nano 30B | 96 | 24,001 | $0.0041 |

Total migration cost: **$0.262**.

## Sources (Tavily)
- claude-sonnet-4-5: $3.0/$15.0 per 1M tokens: https://custom.typingmind.com/tools/estimate-llm-usage-costs/anthropic/claude-sonnet-4-5
- gpt-4o: $2.5/$10.0 per 1M tokens: https://pricepertoken.com/pricing-page/model/openai-gpt-4o
- gpt-4o-mini: $0.15/$0.6 per 1M tokens: https://explainx.ai/models/openai/gpt-4o-mini/cost
- o3-mini: $0.55/$2.2 per 1M tokens: https://pricepertoken.com/pricing-page/model/openai-o3-mini
- Nano 30B: $0.06/$0.24 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- 3.5 Lightning: $0.06/$0.24 per 1M tokens: https://mastra.ai/models/providers/nebius
- Super 120B: $0.3/$0.9 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- Ultra 550B: $1.0/$3.0 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- Endpoint: baseURL/base_url must be set to "https://api.studio.nebius.com/v1/". ()
- Authentication: Use Authorization header with Bearer token from NEBIUS_API_KEY environment variable. ()
- Supported parameters: max_tokens, temperature, top_p, top_k, n, stream, stream_options, stop, presence_penalty, frequency_penalty, logit_bias, logprobs, top_logprobs, user, response_format, extra_body. ()
- Unsupported parameters: Anthropic-specific parameters like thinking, output_config.effort, cache_control, tool.eager_input_streaming, service_tier, speed, context_management, fallbacks, diagnostics are not carried over and may cause 400 errors if passed. ()
- JSON output: Use response_format={"type": "json_object"} for unconstrained JSON or {"type": "json_schema", "json_schema": <schema>} for schema-constrained JSON via Pydantic or direct JSON Schema. ()
- Enable/disable Nemotron thinking: Pass chat_template_kwargs in extra_body with {"enable_thinking": true} to enable or {"enable_thinking": false} to disable thinking mode. ()
- For Nemotron models with disabled thinking, also set force_nonempty_content: true in chat_template_kwargs to avoid empty responses due to reasoning parser issues. ()
- Anthropic-only parameters mapping: When using Anthropic SDK with OpenAI-compatible backend via gateway, system_prompt maps to system role message, max_tokens maps to max_tokens, messages maps directly, stream maps to stream, but Anthropic-specific fields like thinking, output_config.effort, cache_control are ignored or refused. ()

_Generated by [OpenSwitch](https://github.com/rupakkumar-76319/openswitch)._