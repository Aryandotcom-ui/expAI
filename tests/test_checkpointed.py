"""Stage 2 gate: checkpointed_replay == full_replay on >= 30 (target, removal) pairs."""
from tgn_trace.experiments.bench_replay import bench, make_pairs, summarize


def test_checkpointed_equals_full(replayer):
  pairs = make_pairs(replayer, n_targets=9, seed=7)
  assert len(pairs) >= 30
  df = bench(replayer, pairs)
  s = summarize(df)
  print(f"\n{s['n_pairs']} pairs ({df.kind.value_counts().to_dict()}), distance {df.distance_batches.min()}"
        f"-{df.distance_batches.max()} batches, max |ckpt - full| = {s['max_abs_err']:.2e}, "
        f"median speed-up {s['speedup']['median']:.1f}x")
  assert s['max_abs_err'] <= 1e-5
