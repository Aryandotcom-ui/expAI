# CoDy: findings (Stage 3b)

CoDy, "Counterfactual Explainers for Models on Dynamic Graphs" (Gomm et al.), official code at
https://github.com/daniel-gomm/CoDy. Inspected on 2026-10-29 from branch `main`. Read through
raw.githubusercontent.com because github.com itself was blocked in the build environment.

## Licence and task
- **Licence:** MIT (`LICENSE.md`, © 2025 Daniel Gomm), so vendoring is allowed with attribution.
  The files we use are in `third_party/cody/`.
- **Task:** future link prediction. It explains the score of one event under TGN (and TTGN, a
  second packaged implementation). CoDy uses its own TGN fork (`daniel-gomm/tgn-packaged`, a git
  submodule) whose `compute_edge_probabilities` adds `perform_memory_update` and
  `result_as_logit`.
- **Counterfactual criterion:** the logit changes sign. That is the same as crossing our 0.5
  threshold.

## How it re-evaluates the model after removing events
This was the main question. Source: `cody/implementations/tgn.py`, `cody/explainer/base.py`.

1. **Memory is replayed, not reused stale, but only from just before the candidate set.**
   - Before each evaluation, `calculate_subgraph_prediction` calls
     `initialize(min(candidate_events) - 1, memory_label=...)`. This restores a backed-up
     memory state from just before the earliest candidate. It then rolls events forward with
     the removed ones dropped, through `compute_edge_probabilities_for_subgraph`.
   - It also rebuilds the neighbor finder without the removed edges (`to_data_object(...,
     edges_to_drop)`, followed by `get_neighbor_finder`).
   - Both parts match what TGN-Trace does.
2. **By default, memory is approximate.** `Explainer.__init__` has
   `approximate_predictions=True`. In that mode the rollout replays **only the candidate events**
   (`event_ids_to_rollout = candidate_events[~isin(...)]`). Every non-candidate event between the
   earliest candidate and the target is skipped, so those events never update memory.
   - A full rollout is computed **only when the approximate prediction is already
     counterfactual** (the sign flipped).
   - So the deltas that drive CoDy's search, and its reported event importances for
     non-flipping steps, come from a memory state that differs from the model's real state.
   - The "original" prediction is computed with a full rollout. Comparing the two therefore
     mixes the effect of the removal with the effect of the skipped events.
3. **Batching differs from the original stream.**
   - The rollout uses `batch_size=32` by default, not the 200 used for training and evaluation.
   - Batch boundaries are `arange(latest_event_id, ..., batch_size)`, anchored at the restore
     point rather than at the original stream's batches.
   - Removing an event shrinks its batch without shifting other events, which is good. But the
     batch grid itself is not the original one, so even with nothing removed the state is not
     the model's batch-200 state. With memory updated at batch start, batch composition changes
     which events see each other's updates.
4. **Selection policies.**
   - The constructor's default `selection_policy='recent'` is not handled by
     `Explainer._create_sampler`, which only knows `random`, `temporal`, `spatio-temporal` and
     `local-gradient`. That default would raise `NotImplementedError`, so the evaluation scripts
     must pass a policy explicitly.
   - `local-gradient` needs a separately trained embedding model.
   - We use `temporal`, which ranks newest first.

## What we built (`tgn_trace/explainers/cody_adapter.py`)
- CoDy's tree search runs unchanged (`CoDy.explain`, `CoDyTreeNode`, `TemporalSelectionPolicy`).
- We override only `initialize_explanation` (our candidate set and the baseline score's logit) and
  `calculate_subgraph_prediction`.
  - The latter calls `Replayer.checkpointed_replay`: exact memory and neighborhood replay, with
    the original batch-200 boundaries kept and every non-removed event replayed.
  - This is equivalent to running CoDy with `approximate_predictions=False` on a TGN wrapper
    that respects the original batching.
- Budget: `max_steps=60` model evaluations and a 120 s wall-clock timeout. Status is `ok`,
  `no_solution`, `timeout` or `not_applicable` (baseline below 0.5).

## Not done
- We have not run CoDy's own `TGNWrapper` with approximation on to measure how far its
  approximate deltas drift from exact replay. That would need its TGN fork, which was not
  fetched. It is the natural follow-up experiment: CoDy-approx vs CoDy-exact on the same
  targets.
