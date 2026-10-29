"""Stage 1 gate: full_replay is the correctness reference."""
import numpy as np
import pytest

from tgn_trace.candidates import candidate_events


def _targets(baseline_df, n, seed=0):
  test = baseline_df[(baseline_df.region == 'test') & (baseline_df.score >= 0.5)]
  return [int(e) for e in test.sample(n, random_state=seed).edge_idx]


def test_empty_removal_reproduces_baseline(replayer, baseline_df):
  for t in _targets(baseline_df, 3):
    r = replayer.full_replay(t, [])
    print(f'\ntarget {t}: baseline={r["baseline_score"]:.6f} replay={r["cf_score"]:.6f} '
          f'replayed={r["replayed_events"]} latency={r["latency_ms"]:.0f}ms')
    assert abs(r['delta']) <= 1e-5


def test_repeat_identical_and_no_leak(replayer, baseline_df):
  t = _targets(baseline_df, 1, seed=1)[0]
  cands = candidate_events(replayer.env, t)
  r1 = replayer.full_replay(t, cands[:1])
  r2 = replayer.full_replay(t, cands[:1])
  assert r1['cf_score'] == r2['cf_score']
  r3 = replayer.full_replay(t, [])
  assert abs(r3['cf_score'] - r3['baseline_score']) <= 1e-5
  print(f'\ntarget {t}: remove {cands[0]} -> delta={r1["delta"]:+.6f}; empty again -> {r3["delta"]:+.2e}')


def _unreachable_event(env, target):
  """An event before the target's batch whose influence (conservative forward closure over
  shared endpoints, batch by batch) never reaches u, v or their sampled neighbors."""
  p = env.pos(target)
  u, v, t, tb = int(env.src[p]), int(env.dst[p]), env.ts[p], int(env.batch_of[p])
  ngh, _, _ = env.full_ngh_finder.get_temporal_neighbor(np.array([u, v]), np.array([t, t]),
                                                        env.cfg.n_degree)
  influence = {u, v} | set(int(x) for x in ngh.flatten() if x != 0)
  for b in range(tb - 1, max(tb - 20, 0), -1):
    for q in env.batches[b]:
      a, c = int(env.src[q]), int(env.dst[q])
      if a in influence or c in influence:
        continue
      contaminated, reached = {a, c}, False
      for bb in range(b, tb):
        for qq in env.batches[bb]:
          if qq == q:
            continue
          x, y = int(env.src[qq]), int(env.dst[qq])
          if x in contaminated or y in contaminated:
            contaminated |= {x, y}
        if contaminated & influence:
          reached = True
          break
      if not reached:
        return int(env.edge_idx[q]), tb - b
  return None, None


def test_unreachable_event_has_zero_delta(replayer, baseline_df):
  env = replayer.env
  found = 0
  for t in _targets(baseline_df, 10, seed=2):
    e, dist = _unreachable_event(env, t)
    if e is None:
      continue
    r = replayer.full_replay(t, [e])
    print(f'\ntarget {t}: unreachable event {e} ({dist} batches back) -> delta={r["delta"]:+.2e}')
    assert abs(r['delta']) <= 1e-6
    found += 1
    if found >= 2:
      break
  if not found:
    pytest.skip('no provably unreachable event found near the sampled targets')


def test_recent_event_of_u_changes_score(replayer, baseline_df):
  env = replayer.env
  nonzero, n = 0, 0
  sampled = _targets(baseline_df, 20, seed=3)
  for t in sampled:
    u, _ = env.endpoints(t)
    cands = [e for e in candidate_events(env, t) if u in env.endpoints(e)]
    if not cands:
      continue
    r = replayer.full_replay(t, [cands[0]])
    n += 1
    nonzero += abs(r['delta']) > 1e-6
  print(f'\nremoving the most recent event of u changed the score for {nonzero}/{n} targets '
        f'({len(sampled) - n} of {len(sampled)} sampled targets had no event of u among their K candidates)')
  assert nonzero > 0
