# Code written by NVIDIA Nemotron during this migration

`migration.diff` is the full change OpenSwitch committed on the `openswitch/nemotron` branch, and `files/`
holds the changed files after the migration.

- Application code, config and dependency changes, and `tests/test_openswitch_smoke.py` (if present), were
  written by **Nemotron 3 Ultra 550B** (thinking on) through Nebius Token Factory. That covers the rewriter,
  debugger and test-writer roles. Every one of these files passed OpenSwitch's AST checker and behavioural
  check before it was kept.
- `.env.example` and `tests/conftest.py` come from OpenSwitch's own templates, not from a model.
