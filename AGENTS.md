# AGENTS.md

Guidance for coding agents working in this repository.

## Project Snapshot

- Project: `temporal-awareness`
- Purpose: research code for detecting, analyzing, and steering temporal awareness / intertemporal preference in language models
- Main source tree: `src/`
- Experiments and entrypoints: `scripts/`
- Tests: `tests/`
- Data and generated artifacts: `data/`, `results/`

## Repository Layout

- `src/inference/`: model backends, model runner, captured internals, intervention plumbing
- `src/intertemporal/`: prompt formatting, datasets, experiments, SAE pipeline, visualizations
- `src/activation_patching/` and `src/attribution_patching/`: patching experiments and result types
- `src/common/`: shared schema, math, file IO, token/tree utilities, profiling helpers
- `scripts/`: runnable experiment, validation, probe-training, and analysis scripts
- `tests/`: unit and integration coverage for common, inference, and patching workflows
- `notebooks/`: research notebooks and figure reproduction

## Environment And Setup

Preferred install:

```bash
pip install -e ".[dev]"
```

There is also a `Makefile` with a few common tasks:

```bash
make install
make verify-quick
make verify-probes
make verify-steering
```

Notes:

- Python requirement is `>=3.10`
- Main dependencies include `torch`, `transformers`, `transformer_lens`, and a git dependency on `latents`
- Some verification and experiment scripts assume model access, GPUs, API keys, or existing cached artifacts

## Code Conventions

These conventions are already established in the repo and should be preserved.

- Keep imports at module top level. Avoid inline imports unless they are required to break a circular dependency.
- Use `src/common/base_schema.py`'s `BaseSchema` for dataclasses that need serialization, deterministic IDs, or `from_dict` / `to_dict` support.
- In `__init__.py` files, use the existing `auto_export` pattern instead of hand-maintained `__all__` lists.
- Prefer small, focused functions and modules over adding more branching into already large files.
- Do not leave dead code, commented-out code, or debug prints behind.
- Follow the repository’s existing style; if a file already has a local pattern, match it before introducing a new abstraction.

## Working Safely

- Treat `data/` and `results/` as research assets. Do not delete or regenerate large artifacts unless the task clearly requires it.
- Many scripts are expensive or hardware-dependent. Prefer the smallest validating command that proves your change.
- This repository may contain large model- or dataset-driven workflows. Be explicit about assumptions when you cannot run the full experiment locally.
- Avoid changing notebooks or generated result files unless the user asked for that specifically.

## Testing And Validation

Start with targeted tests for the area you changed:

```bash
pytest tests/inference -q
pytest tests/common -q
pytest tests/attribution_patching -q
```

For broader coverage:

```bash
pytest -q
```

Useful project verification commands:

```bash
make verify-quick
make verify
```

Style tools listed in project docs:

```bash
black .
ruff check .
```

If a command is too expensive, blocked by missing dependencies, or requires unavailable hardware, say so clearly and report what you did run instead.

## Change Guidance

- Prefer surgical fixes over broad refactors unless the task is explicitly architectural.
- When touching backend or intervention code, check for corresponding tests under `tests/inference/`.
- When changing shared schemas or utilities in `src/common/`, scan for downstream call sites because those modules are reused widely.
- When adding new public types or modules, keep exports consistent with the repo’s `auto_export` approach.
- When adding scripts, place them in the nearest existing domain folder under `scripts/`.

## Good Defaults For Agents

1. Read the local module and nearby tests before editing.
2. Make the smallest coherent change that satisfies the request.
3. Run the narrowest relevant validation.
4. Summarize any unrun checks, hardware constraints, or assumptions in the handoff.
