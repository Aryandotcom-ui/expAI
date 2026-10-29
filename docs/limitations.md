# Limitations

- **The numbers in this repo were measured on a synthetic stand-in, not on Wikipedia.**
  - The build environment could not reach snap.stanford.edu, so `data/wikipedia.csv` was never
    downloaded.
  - `synthetic/generate.py` produces a dataset with Wikipedia's schema (bipartite user→page
    edits, 172 edge features, JODIE CSV) and similar repeat-edit behavior, at about half its size.
  - The code is dataset-agnostic, and Wikipedia is the default (`TGN_TRACE_DATA=wikipedia`).
    Every Wikipedia number still has to be produced by running the pipeline (README, "Run on
    Wikipedia").
- **Intervention effects measure model sensitivity, not real-world causation.**
  - A replay delta says how *this trained model's* score changes when an event is deleted from
    its input history.
  - It says nothing about what would have happened in the world.
- **Signal C is a linearization of Signal B, not independent evidence.**
  - It is the exact structural term plus a first-order memory term, computed at the baseline
    memory.
  - Agreement between B and C only shows that the model is locally close to linear in memory.
  - The memory gradient treats the post-update memory as a leaf. It does not differentiate
    through `get_updated_memory` (GRU, message function).
- **Brute force is single-event only.**
  - It is our ground truth for single-event influence.
  - It misses interactions: two redundant events can each have a small delta while removing
    both flips the prediction. The greedy and CoDy searches look at sets; brute force does not.
- **The candidate set is a design choice:** the last K=20 events touching u or v.
  - Events further back, or 2-hop events that reach the target only through a neighbor's
    memory, are never candidates.
  - With `n_layer=1` the target's embedding reads only u's and v's own temporal neighbors.
  - The K newest events of the union favor the busier endpoint. On `synthwiki`, 298 of 300
    sampled test targets had an earlier event of u, but only 222 had one among their 20
    candidates. A popular page's recent edits crowd out the user's own history.
    `candidate_events` could take K per endpoint instead; we kept the plan's definition.
- **The baseline state is our own pass,** not the memory left at the end of training. See the
  README for the definition. AP/AUC are reported for this pass.
- **Wikipedia is not fraud data.** Nothing here is evidence about fraud detection, and we make no
  detection claim.
- **The Elliptic dataset was deliberately not used.** Its timestamps are 49 coarse snapshots,
  not continuous event times. Memory replay under batch-start updates is therefore ill-defined
  within a snapshot.
- **Flag thresholds:**
  - τ is chosen on validation-region targets as the 95th percentile of |delta| for control
    removals (recent events touching neither u nor v), floored at 0.001.
  - We read "the top-3 event has |delta| ≥ τ" as "at least 2 of the explainer's top 3 events
    have |delta| ≥ τ".
- **CoDy** runs through an adapter: its search, but our exact replay for model evaluation (see
  `docs/cody_notes.md`). We did not reproduce CoDy's own approximate evaluation.
- **Hardware:** everything was measured on 4 CPU cores, with no GPU. Latencies on the target
  RTX 3050 laptop will differ. Speed-ups are reported as measured, never as targets.
