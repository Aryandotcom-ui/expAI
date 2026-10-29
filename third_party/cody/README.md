# Vendored CoDy (MIT)

Source: https://github.com/daniel-gomm/CoDy (branch `main`, fetched 2026-10-29), MIT licence, see
LICENSE.md. Only the files needed by `tgn_trace/explainers/cody_adapter.py` are included.

Patches:
- `cody/utils.py`: the `IPython` import is optional (it is only used to detect notebooks).

Nothing else is modified. TGN-Trace replaces CoDy's model evaluation (its `TGNWrapper` and its
approximate rollout) with its own exact, memory-aware replay; see `docs/cody_notes.md`.
