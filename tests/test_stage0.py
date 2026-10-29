"""Stage 0 gate.

- Snapshots are deep copies (memory, last_update and every pending message).
- score_target on a single event == the batched score of the baseline pass (atol 1e-5), with the
  state advanced by `advance_state` (dummy negatives). This also checks that dummy negatives give
  the same state as the real eval loop, which used sampled negatives in the test region.
- Two baseline passes give identical scores.
"""
import json

import numpy as np
import torch

from tgn_trace.baseline import baseline_pass
from tgn_trace.replay import score_target


def test_snapshot_is_deep_copy(env):
  env.reset_state()
  for b in range(3):
    env.advance_state(env.batches[b], env.full_ngh_finder)
  snap = env.snapshot()
  ref = (snap[0].clone(), snap[1].clone(),
         {k: [(m.clone(), t.clone()) for m, t in v] for k, v in snap[2].items()})
  # Mutate the live state in place, everywhere
  env.model.memory.memory.data.add_(1.0)
  env.model.memory.last_update.data.add_(1.0)
  for v in env.model.memory.messages.values():
    for m, t in v:
      m.add_(1.0)
      t.add_(1.0)
  assert torch.equal(snap[0], ref[0]) and torch.equal(snap[1], ref[1])
  for k, v in ref[2].items():
    assert len(snap[2][k]) == len(v)
    for (a, ta), (b, tb) in zip(v, snap[2][k]):
      assert torch.equal(a, b) and torch.equal(ta, tb)
  # restore() must also copy: mutating after restore must not touch the snapshot
  env.restore(snap)
  env.model.memory.memory.data.add_(1.0)
  assert torch.equal(snap[0], ref[0])
  # pack / unpack round trip (on-disk format)
  restored = env.unpack(env.pack(snap))
  assert torch.equal(restored[0].cpu(), snap[0].cpu()) and torch.equal(restored[1].cpu(), snap[1].cpu())
  for k, v in snap[2].items():
    assert len(restored[2][k]) == len(v)
    for (a, ta), (b, tb) in zip(v, restored[2][k]):
      assert torch.equal(a.cpu(), b.cpu()) and torch.equal(ta.cpu(), tb.cpu())


def test_single_target_matches_batched(env, baseline_df):
  test = baseline_df[baseline_df.region == 'test']
  sample = test.sample(50, random_state=0)
  by_batch = {}
  for row in sample.itertuples():
    by_batch.setdefault(int(row.batch), []).append(row)
  env.reset_state()
  errors = []
  last = max(by_batch)
  for b in range(last + 1):
    for row in by_batch.get(b, []):
      s = score_target(env, int(row.edge_idx), env.full_ngh_finder)
      errors.append(abs(s - row.score))
    env.advance_state(env.batches[b], env.full_ngh_finder)
  print(f'\nmax |score_target - batched| over {len(errors)} test events = {max(errors):.3e}')
  assert len(errors) == 50
  assert max(errors) <= 1e-5


def test_baseline_pass_repeatable(env, baseline_df):
  df1, m1 = baseline_pass(env, progress=False)
  df2, m2 = baseline_pass(env, progress=False)
  assert np.array_equal(df1.score.values, df2.score.values)
  assert np.allclose(df1.score.values, baseline_df.score.values, atol=1e-6, rtol=0)
  print('\nbaseline test AP/AUC (ours):', json.dumps({k: round(m1[k], 5) for k in m1 if k.startswith('test_')}))
