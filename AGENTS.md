# Repository Guidelines

## Project Structure & Module Organization

- `doc/` — explanatory and user-facing documentation.
- `.codex/doc/` — phase notes, validation records, and other internal
  development documentation.
- `plan/` — implementation plans, refactor notes, and task breakdowns.
- `test/` — tests and validation scripts; name test files `test_*.py`.
- Python package source — organize the new workflow as importable modules rather
  than a single large script. Prefer small, focused modules for parsing, model
  setup, inference, artifact writing, and reporting.

Runtime outputs should keep a clear and predictable naming pattern, for example
`<name>_pred.cif` and `<name>_pred_artifacts/` for plots, HTML vieers, reports,
or other quality artifacts.


## Development Workflow

- Before each development task or implementation step, create or update a plan in
  `plan/` so the intended work, scope, and breakdown are documented first.
- After completing development work, write the corresponding internal
  development documentation in `.codex/doc/`, including what changed, how it
  was validated, and any follow-up notes that matter for future work. Keep
  `doc/` reserved for explanatory and user-facing documentation.
- After development is complete, commit the code promptly according to the
  repository's commit requirements and keep each commit focused on the completed
  change.

## Coding Style & Naming Conventions

Follow Google Python style: 4-space indentation, `snake_case` for functions and
variables, `PascalCase` for classes, clear type annotations where useful, and
Google-style docstrings for public modules, classes, and functions. Comments
should be detailed where they explain modeling assumptions, tensor shapes, GPU
placement, or non-obvious performance choices; avoid comments that restate simple
code.

Keep the new library lightweight and modular. Prefer explicit data structures and
small helper functions over framework-style abstractions. Avoid compatibility
shims for the original workflow unless they directly support the new CLI contract.

## Performance Guidelines

All main workflow execution should run in the GPU environment. Pay special
attention to multi-seed loops and repeated inference paths:

- Keep tensors and model inputs on GPU whenever possible.
- Avoid repeated CPU/GPU transfers inside loops.
- Cache reusable encoded inputs, chain metadata, masks, and model state.
- Batch or vectorize repeated work when it does not change numerical behavior.
- Move artifact serialization, plotting, and text reporting outside hot inference
  loops when practical.

Favor efficiency in loops over convenience, but keep the code readable and document
any non-obvious optimization.


## Commit & Pull Request Guidelines

This checkout may not expose usable Git history, so use concise imperative commit
messages such as `Refactor inference loop for multi-seed reuse`. When creating
Git commits, follow the requirements in `CONTRIBUTION.md`. Keep each commit
focused on one behavior change. Pull requests should include:

- a short summary of the workflow, packaging, or reporting change
- exact commands used for validation
- example input and output paths
- screenshots or sample plots when visualization output changes
- notes about GPU performance changes when inference loops are modified

## Security & Configuration Tips

Do not switch this workflow to external APIs. Keep model loading local, preserve
offline cache settings, and avoid committing large prediction outputs unless they
are necessary for review. Do not introduce network-dependent setup or downloads
without explicit user approval.
