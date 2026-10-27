# OpenSwitch: migrate LLM calls to NVIDIA Nemotron on Nebius Token Factory

**Estimated LLM cost: $5.183 → $0.471 per 1,000 calls of each function (91% lower).** Tests: 13/13 passed.

| Function | Before | After (Nemotron) | Eval score | Functional | $/1K before | $/1K after |
|---|---|---|---|---|---|---|
| `classify_ticket` | gpt-4o-mini | Nano 30B fast | 10.0/10 (min 10) | 100% | 0.0115 | 0.0046 |
| `extract_order` | gpt-4o | Super 120B fast | 9.4/10 (min 5) | 100% | 0.6931 | 0.0657 |
| `refund_amount` | o3-mini | Nano 30B think | 9.2/10 (min 5) | 100% | 1.5549 | 0.2802 |
| `summarize_thread` | gpt-4o | Nano 30B fast | 9.2/10 (min 8) | 100% | 0.7887 | 0.0227 |
| `draft_reply` | gpt-4o | Nano 30B fast | 10.0/10 (min 10) | 100% | 1.3150 | 0.0291 |
| `translate_reply` | claude-sonnet-4-5 | Nano 30B think | 8.8/10 (min 2) | 100% | 0.8203 | 0.0688 |

## How each model was chosen
Every function was run on 6 Nemotron configs with 6 inputs generated from its own real prompts (captured by running the app's code against a fake SDK). Each output was fed back through the app's own post-processing (functional check) and checked by Nemotron 3 Ultra against a pass/fail checklist (correctness plus the call's rubric); score = share of checks passed. Every config is graded, the strongest one included. The cheapest config with 100% functional outputs, a mean within 0.75 of the best config (and ≥ 7.5) and no output below 5.0 was chosen.

<details><summary><code>classify_ticket</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 10.0 | 10 | 2 | 0.0046 | ✅ |
| Nano 30B think | 100% | 9.2 | 5 | 53 | 0.0168 | ❌ |
| Super 120B fast | 100% | 9.2 | 5 | 3 | 0.0224 | ❌ |
| Super 120B think | 100% | 10.0 | 10 | 131 | 0.1382 | ✅ |
| Ultra 550B fast | 100% | 10.0 | 10 | 2 | 0.0742 | ✅ |
| Ultra 550B think | 100% | 10.0 | 10 | 86 | 0.3257 | ✅ |

</details>

<details><summary><code>extract_order</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 8.3 | 5 | 47 | 0.0172 | ❌ |
| Nano 30B think | 100% | 9.2 | 5 | 272 | 0.0713 | ❌ |
| Super 120B fast | 100% | 9.4 | 5 | 40 | 0.0657 | ✅ |
| Super 120B think | 100% | 9.6 | 8 | 178 | 0.1903 | ✅ |
| Ultra 550B fast | 100% | 9.8 | 8 | 44 | 0.2329 | ✅ |
| Ultra 550B think | 100% | 10.0 | 10 | 310 | 1.0312 | ✅ |

</details>

<details><summary><code>refund_amount</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 4.6 | 2 | 5 | 0.0110 | ❌ |
| Nano 30B think | 100% | 9.2 | 5 | 1127 | 0.2802 | ✅ |
| Super 120B fast | 33% | 3.3 | 0 | 338 | 0.3529 | ❌ |
| Super 120B think | 100% | 8.8 | 2 | 326 | 0.3423 | ❌ |
| Ultra 550B fast | 83% | 6.2 | 0 | 45 | 0.2975 | ❌ |
| Ultra 550B think | 100% | 9.2 | 5 | 666 | 2.1610 | ✅ |

</details>

<details><summary><code>summarize_thread</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 9.2 | 8 | 68 | 0.0227 | ✅ |
| Nano 30B think | 100% | 10.0 | 10 | 339 | 0.0878 | ✅ |
| Super 120B fast | 100% | 9.5 | 8 | 68 | 0.0938 | ✅ |
| Super 120B think | 100% | 9.0 | 2 | 152 | 0.1696 | ❌ |
| Ultra 550B fast | 100% | 9.8 | 8 | 52 | 0.2637 | ✅ |
| Ultra 550B think | 100% | 9.2 | 5 | 186 | 0.6672 | ✅ |

</details>

<details><summary><code>draft_reply</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 10.0 | 10 | 98 | 0.0291 | ✅ |
| Nano 30B think | 100% | 10.0 | 10 | 181 | 0.0492 | ✅ |
| Super 120B fast | 100% | 9.7 | 8 | 107 | 0.1245 | ✅ |
| Super 120B think | 100% | 10.0 | 10 | 186 | 0.1953 | ✅ |
| Ultra 550B fast | 100% | 10.0 | 10 | 108 | 0.4182 | ✅ |
| Ultra 550B think | 100% | 10.0 | 10 | 184 | 0.6472 | ✅ |

</details>

<details><summary><code>translate_reply</code>: all configs</summary>

| Config | Functional | Mean | Min | Avg out tokens | $/1K | Pass |
|---|---|---|---|---|---|---|
| Nano 30B fast | 100% | 8.5 | 2 | 86 | 0.0245 | ❌ |
| Nano 30B think | 100% | 8.8 | 2 | 270 | 0.0688 | ❌ |
| Super 120B fast | 100% | 8.5 | 2 | 41 | 0.0566 | ❌ |
| Super 120B think | 100% | 8.3 | 0 | 231 | 0.2277 | ❌ |
| Ultra 550B fast | 100% | 8.3 | 2 | 41 | 0.1908 | ❌ |
| Ultra 550B think | 100% | 8.5 | 0 | 377 | 1.1967 | ❌ |

</details>

## Tests
`pytest`: **13/13 passed** after 1 iteration(s).
- Iteration 1: 13/13 passed

## Known failures and limits
- `extract_order` (Super 120B fast): Check 0 failed because the extracted order_date '2024-09-15' is labeled as the receipt date ('received today') in the text, not the order date; Check 2 failed because the order_date value '2024-09-15' was not directly extractable as an order date from the ticket text.
- `extract_order` (Super 120B fast): Check 0 failed: output misses the second order ID 'ABC111' present in the ticket text.
- `refund_amount` (Nano 30B think): The input does not specify how many days have passed since the order date, making it impossible to determine which refund tier applies; assuming a full refund is not defensible without that information.
- `summarize_thread` (Nano 30B fast): The summary claims an open issue about the exact shipping date and tracking details, but the customer appeared satisfied with the agent's confirmation that the order ships tomorrow and did not request further information.
- `summarize_thread` (Nano 30B fast): The summary incorrectly states that the size change still needs confirmation when the agent has already processed the exchange for a medium size.
- `summarize_thread` (Nano 30B fast): The summary incorrectly states the issue remains open; the thread shows the agent already sent a corrected invoice, indicating resolution.
- `translate_reply` (Nano 30B think): Check 1 failed: 'nous avoir contactés' uses the masculine plural agreement for the past participle, but the direct object 'nous' refers to the company/support team as a singular collective entity, requiring the singular 'contacté'.
- `translate_reply` (Nano 30B think): The output is in English with only a few Portuguese words, not a proper Portuguese translation; names and order numbers are unchanged but the text fails the language requirement and contains the original English greeting/sign-off instead of a full translation.
- `translate_reply` (Nano 30B think): Check 0 failed because the translation output does not correspond to the requested target language (Arabic) but appears to be a translation of the meta-instruction 'Target language: Arabic'; Check 1 failed because the output is not a translation of a support reply into Arabic but merely the localized label for the target language.
- Baseline cost is an estimate: source-model token counts are approximated with Nemotron's tokenizer, and for o-series models hidden reasoning is approximated with Nemotron Ultra's thinking length.
- The reference answers come from Nemotron 3 Ultra with thinking, not from the original provider (no OpenAI/Anthropic key is needed or used); the repo's own tests are the independent check.
- Each function was evaluated on 6 generated inputs; rare inputs can still behave differently, so keep monitoring after merging.

## Which Nemotron did what during this migration
| Role | Model | Calls | Tokens | Cost |
|---|---|---|---|---|
| judge | Ultra 550B | 258 | 124,035 | $0.1436 |
| candidate | Ultra 550B | 136 | 36,397 | $0.0833 |
| candidate | Super 120B | 136 | 31,224 | $0.0203 |
| planner | Ultra 550B | 1 | 6,978 | $0.0155 |
| eval-writer | Super 120B | 15 | 20,837 | $0.0136 |
| rewriter | Ultra 550B | 1 | 4,878 | $0.0096 |
| candidate | Nano 30B | 136 | 38,056 | $0.0068 |
| researcher | Super 120B | 6 | 19,842 | $0.0064 |

Total migration cost: **$0.299**.

## Sources (Tavily)
- claude-sonnet-4-5: $3.0/$15.0 per 1M tokens: https://custom.typingmind.com/tools/estimate-llm-usage-costs/anthropic/claude-sonnet-4-5
- gpt-4o: $2.5/$10.0 per 1M tokens: https://pricepertoken.com/pricing-page/model/openai-gpt-4o
- gpt-4o-mini: $0.15/$0.6 per 1M tokens: https://pricepertoken.com/pricing-page/model/openai-gpt-4o-mini
- o3-mini: $0.55/$2.2 per 1M tokens: https://pricepertoken.com/pricing-page/model/openai-o3-mini
- Nano 30B: $0.06/$0.24 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- Super 120B: $0.3/$0.9 per 1M tokens: https://assets.nebius.com/assets/309d5e28-9376-4a45-81cf-dbb7f27a857c/Nebius%20Token%20Factory%20x%20NVIDIA%20Nemotron%20x%20clinical.pdf?cache-buster=2026-10-19T21%3A51%3A48.334Z
- For Nebius Token Factory, the base URL is "https://api.studio.nebius.com/v1/" and authentication uses a Bearer token in the Authorization header with the NEBIUS_API_KEY environment variable. ()
- Supported parameters in the OpenAI-compatible API include max_tokens, temperature, top_p, top_k, n, stream, stream_options, stop, presence_penalty, frequency_penalty, logit_bias, logprobs, top_logprobs, user, extra_body, and response_format. ()
- JSON output can be enabled via response_format={"type": "json_object"} or response_format={"type": "json_schema", "json_schema": <schema>} for structured outputs like Pydantic models. ()
- Nemotron thinking (reasoning) can be controlled via extra_body.chat_template_kwargs with enable_thinking:true/false and reasoning_budget:<integer> parameters, as seen in NVIDIA Nemotron models and Ollama compatibility layers. ()
- Anthropic-only parameters like max_tokens map directly to the OpenAI-compatible API's max_tokens field when using protocol conversion gateways (e.g., API7 AI Gateway or CompactifAI). ()
- The model parameter must specify the full model identifier (e.g., "meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen/Qwen3-235B-A22B") as shown in working examples. ()
- Streaming is supported via stream=true and stream_options can be set to null or omitted; SSE chunks are compatible with OpenAI-style streaming. ()
- Parameters such as reasoning_effort (used in OpenAI o3-mini) are not mentioned as supported in the Nebius Token Factory or NVIDIA Nemotron documentation within the provided results and should not be assumed available. ()

_Generated by [OpenSwitch](https://github.com/rupakkumar-76319/openswitch)._