"""Stage 2 / E3 / E4: checkpointed replay vs full replay (correctness and latency).

For (target, removal set) pairs covering single and multi-event removals at different distances,
run both replays and record the removal distance (target batch - earliest removed batch), both
latencies, the speed-up and |checkpointed - full|.

Usage: python -m tgn_trace.experiments.bench_replay [--n_targets 12]
Writes results/<data>/replay_bench.csv and prints median / IQR.
"""
import argparse
import json

import numpy as np
import pandas as pd

from tgn_trace.candidates import candidate_events
from tgn_trace.config import TraceConfig
from tgn_trace.replay import Replayer
from tgn_trace.state import get_env


def make_pairs(replayer, n_targets, seed=0):
  env = replayer.env
  df = pd.read_csv(env.cfg.results_dir / 'baseline_scores.csv')
  df = df[(df.region == 'test') & (df.score >= env.cfg.threshold)]
  rng = np.random.RandomState(seed)
  pairs = []
  for t in df.sample(n_targets, random_state=seed).edge_idx:
    t = int(t)
    cands = candidate_events(env, t)
    if len(cands) < 4:
      continue
    tb = int(env.batch_of[env.pos(t)])
    dist = {e: tb - int(env.batch_of[env.pos(e)]) for e in cands}
    by_dist = sorted(cands, key=lambda e: dist[e])
    pairs.append((t, [by_dist[0]], 'single-nearest'))
    pairs.append((t, [by_dist[-1]], 'single-farthest'))
    pairs.append((t, [by_dist[len(by_dist) // 2]], 'single-middle'))
    pairs.append((t, [int(e) for e in rng.choice(cands, size=3, replace=False)], 'multi-3'))
  return pairs


def bench(replayer, pairs):
  env = replayer.env
  rows = []
  for t, removed, kind in pairs:
    tb = int(env.batch_of[env.pos(t)])
    full = replayer.full_replay(t, removed)
    ck = replayer.checkpointed_replay(t, removed, use_cache=False, use_prefix=False)
    rows.append(dict(target=t, kind=kind, removed=' '.join(map(str, removed)), n_removed=len(removed),
                     distance_batches=tb - min(int(env.batch_of[env.pos(e)]) for e in removed),
                     full_ms=full['latency_ms'], checkpointed_ms=ck['latency_ms'],
                     speedup=full['latency_ms'] / ck['latency_ms'],
                     full_replayed_events=full['replayed_events'],
                     checkpointed_replayed_events=ck['replayed_events'],
                     checkpoint_used=ck['checkpoint_used'], full_score=full['cf_score'],
                     checkpointed_score=ck['cf_score'], delta=full['delta'],
                     abs_err=abs(full['cf_score'] - ck['cf_score'])))
  return pd.DataFrame(rows)


def summarize(df):
  def q(col):
    return dict(median=float(df[col].median()), q25=float(df[col].quantile(0.25)),
                q75=float(df[col].quantile(0.75)))
  return dict(n_pairs=len(df), max_abs_err=float(df.abs_err.max()), mean_abs_err=float(df.abs_err.mean()),
              full_ms=q('full_ms'), checkpointed_ms=q('checkpointed_ms'), speedup=q('speedup'),
              distance_batches=q('distance_batches'))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n_targets', type=int, default=12)
  args = ap.parse_args()
  cfg = TraceConfig()
  env = get_env(cfg)
  replayer = Replayer(env)
  df = bench(replayer, make_pairs(replayer, args.n_targets))
  cfg.results_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(cfg.results_dir / 'replay_bench.csv', index=False)
  s = summarize(df)
  (cfg.results_dir / 'replay_bench_summary.json').write_text(json.dumps(s, indent=2))
  print(df[['target', 'kind', 'distance_batches', 'full_ms', 'checkpointed_ms', 'speedup', 'abs_err']]
        .to_string(index=False, float_format=lambda x: f'{x:.3g}'))
  print(json.dumps(s, indent=2))


if __name__ == '__main__':
  main()
