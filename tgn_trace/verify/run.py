"""Stage 4: verification over many targets.

1. Choose tau on VALIDATION-region targets only: tau = 95th percentile of |delta| for control
   removals (recent events that touch neither u nor v), floored at `--tau_floor`. Everything is
   logged to results/<data>/threshold_selection.json.
2. For TEST targets (baseline >= threshold): brute-force deltas (Signal B), linear estimates
   (Signal C), explainer ranks (Signal A), agreement, null percentile and flags.

Outputs (results/<data>/):
  verification.csv          one row per (target, explainer)
  verification_events.csv   one row per (target, candidate)
  verification/<id>.json    everything for one target (served by the API)
  verification_null.npz     pooled null agreement samples per explainer (for the figure)

Usage: python -m tgn_trace.verify.run [--n_test 100] [--n_val 20] [--no_c] [--with_cody]
       Parallel: --stage tau, then --stage targets --shard k --n_shards n (one process per shard,
       resumable), then --stage aggregate.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from tgn_trace.candidates import candidate_events
from tgn_trace.config import TraceConfig
from tgn_trace.explainers import bruteforce, greedy_cf, recency
from tgn_trace.replay import Replayer
from tgn_trace.state import get_env
from tgn_trace.verify.flags import flag
from tgn_trace.verify.null_baseline import null_percentile
from tgn_trace.verify.signals import LinearEstimator

EXPLAINERS = {'recency': recency.explain, 'greedy': greedy_cf.explain}


def pick_targets(replayer, region, n, seed, min_batch=None):
  env = replayer.env
  df = pd.read_csv(env.cfg.results_dir / 'baseline_scores.csv')
  df = df[(df.region == region) & (df.score >= env.cfg.threshold)]
  if min_batch is not None:
    df = df[df.batch >= min_batch]
  # A fixed shuffle, then a prefix: a larger n always extends a smaller n's target list, so
  # experiments with different sizes share targets (and the brute-force cache).
  ids = df.edge_idx.to_numpy()[np.random.RandomState(seed).permutation(len(df))]
  return [int(e) for e in ids[:n]]


def control_events(env, target, n, rng, window=10):
  p = env.pos(target)
  u, v, tb = env.src[p], env.dst[p], int(env.batch_of[p])
  pool = np.concatenate([env.batches[b] for b in range(max(0, tb - window), tb + 1)])
  pool = pool[(env.ts[pool] < env.ts[p]) & (env.src[pool] != u) & (env.dst[pool] != v)
              & (env.src[pool] != v) & (env.dst[pool] != u)]
  return [int(env.edge_idx[q]) for q in rng.choice(pool, size=min(n, len(pool)), replace=False)]


def select_tau(replayer, n_val, seed, tau_floor, n_control=5):
  env = replayer.env
  rng = np.random.RandomState(seed)
  first_ckpt = replayer.checkpoints.batches[0] if replayer.checkpoints.batches else 0
  targets = pick_targets(replayer, 'val', n_val, seed, min_batch=first_ckpt + 5)
  control, cand = [], []
  for t in targets:
    for e in control_events(env, t, n_control, rng):
      control.append(abs(replayer.checkpointed_replay(t, [e])['delta']))
    bf = bruteforce.explain(env, t, replayer)
    cand.extend(abs(d) for d in bf['deltas'].values())
  control, cand = np.array(control), np.array(cand)
  q95 = float(np.quantile(control, 0.95))
  tau = max(q95, tau_floor)
  grid = {}
  for g in sorted({tau_floor, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, q95, tau}):
    grid[f'{g:.5f}'] = dict(frac_control_above=float(np.mean(control >= g)),
                            frac_candidates_above=float(np.mean(cand >= g)))
  log = dict(rule='tau = max(95th percentile of |delta| over control removals, tau_floor)',
             region='val', n_val_targets=len(targets), val_targets=targets, n_control=len(control),
             control_abs_delta_quantiles={q: float(np.quantile(control, q)) for q in (0.5, 0.9, 0.95, 0.99)},
             candidate_abs_delta_quantiles={q: float(np.quantile(cand, q)) for q in (0.5, 0.9, 0.95, 0.99)},
             q95_control=q95, tau_floor=tau_floor, tau=tau, grid=grid)
  return tau, log


def verify_target(replayer, target, tau, with_c=True, n_perm=1000, explainers=None):
  explainers = explainers or EXPLAINERS
  env = replayer.env
  bf = bruteforce.explain(env, target, replayer)
  deltas = bf['deltas']
  cands = bf['candidates']
  neg_delta = [-deltas[e] for e in cands]

  lin = {}
  if with_c:
    est = LinearEstimator(replayer, target)
    for e in cands:
      lin[e] = est.estimate([e])

  per_expl, nulls = {}, {}
  for name, fn in explainers.items():
    ex = fn(env, target, replayer)
    imp = [ex['importance'].get(e, 0.0) for e in cands]
    obs, pct, null = null_percentile(imp, neg_delta, n_perm=n_perm, seed=target % (2 ** 31))
    f, reason = flag(ex['ranked'], deltas, tau)
    per_expl[name] = dict(ranked=ex['ranked'], importance=ex['importance'], agreement=obs,
                          null_percentile=pct, flag=f, reason=reason, status=ex['status'],
                          runtime_ms=ex['runtime_ms'], n_model_evals=ex['n_model_evals'],
                          extra={k: ex[k] for k in ('cf_set', 'success', 'final_score') if k in ex})
    nulls[name] = null
  events = []
  for e in cands:
    row = dict(env.event(e), signal_b_delta=deltas[e])
    for name in explainers:
      row[f'rank_{name}'] = per_expl[name]['ranked'].index(e) + 1
      row[f'importance_{name}'] = per_expl[name]['importance'].get(e, 0.0)
    row['rank_bruteforce'] = bf['ranked'].index(e) + 1
    if with_c:
      row.update({k: lin[e][k] for k in ('signal_c_linear_estimate', 'structural_term',
                                         'memory_linear_term', 'memory_exact_term', 'nonlinearity')})
      row['c_vs_b_replay_check'] = abs(lin[e]['signal_b_replay'] - deltas[e])
    events.append(row)
  return dict(target=env.event(target), baseline_score=bf['baseline_score'], tau=tau,
              candidates=cands, bruteforce=dict(ranked=bf['ranked'], deltas=deltas),
              explainers=per_expl, events=events), nulls


def aggregate(cfg, targets):
  """Build the CSVs and the pooled null from per-target JSONs."""
  out = cfg.results_dir
  rows, event_rows, pooled = [], [], {}
  for t in targets:
    path = out / 'verification' / f'{t}.json'
    if not path.exists():
      continue
    res = json.loads(path.read_text())
    deltas = {int(e): d for e, d in res['bruteforce']['deltas'].items()}
    for name, ex in res['explainers'].items():
      pooled.setdefault(name, []).append(np.asarray(ex.get('null', []), dtype=float))
      rows.append(dict(target_id=t, explainer=name, baseline_score=res['baseline_score'],
                       n_candidates=len(res['candidates']), agreement=ex['agreement'],
                       null_percentile=ex['null_percentile'], flag=ex['flag'], reason=ex['reason'],
                       status=ex['status'], n_model_evals=ex['n_model_evals'], runtime_ms=ex['runtime_ms'],
                       cf_success=ex['extra'].get('success'),
                       cf_size=len(ex['extra']['cf_set']) if 'cf_set' in ex['extra'] else None,
                       top3=' '.join(map(str, ex['ranked'][:3])),
                       top3_abs_delta=' '.join(f'{abs(deltas[int(e)]):.4f}' for e in ex['ranked'][:3]),
                       max_abs_delta=max(abs(d) for d in deltas.values()), tau=res['tau']))
    for ev in res['events']:
      event_rows.append(dict(target_id=t, **ev))
  pd.DataFrame(rows).to_csv(out / 'verification.csv', index=False)
  pd.DataFrame(event_rows).to_csv(out / 'verification_events.csv', index=False)
  np.savez(out / 'verification_null.npz', **{k: np.concatenate([x for x in v if len(x)]) if any(
    len(x) for x in v) else np.array([]) for k, v in pooled.items()})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n_test', type=int, default=100)
  ap.add_argument('--n_val', type=int, default=20)
  ap.add_argument('--tau_floor', type=float, default=1e-3)
  ap.add_argument('--no_c', action='store_true')
  ap.add_argument('--with_cody', action='store_true')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--shard', type=int, default=0)
  ap.add_argument('--n_shards', type=int, default=1)
  ap.add_argument('--stage', choices=['all', 'tau', 'targets', 'aggregate'], default='all')
  args = ap.parse_args()

  cfg = TraceConfig()
  env = get_env(cfg)
  replayer = Replayer(env)
  out = cfg.results_dir
  (out / 'verification').mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  tau_path = out / 'threshold_selection.json'

  if args.stage in ('all', 'tau'):
    tau, log = select_tau(replayer, args.n_val, args.seed, args.tau_floor)
    tau_path.write_text(json.dumps(log, indent=2))
    print(f'tau={tau:.5f} chosen on {log["n_val_targets"]} val targets ({time.perf_counter() - t0:.0f}s)')
  targets = pick_targets(replayer, 'test', args.n_test, args.seed + 1)

  if args.stage in ('all', 'targets'):
    tau = json.loads(tau_path.read_text())['tau']
    explainers = dict(EXPLAINERS)
    if args.with_cody:
      from tgn_trace.explainers import cody_adapter
      explainers['cody'] = cody_adapter.explain
    mine = targets[args.shard::args.n_shards]
    for i, t in enumerate(mine):
      path = out / 'verification' / f'{t}.json'
      if path.exists():
        continue  # resumable
      res, nulls = verify_target(replayer, t, tau, with_c=not args.no_c, explainers=explainers)
      for name in res['explainers']:
        res['explainers'][name]['null'] = [round(float(x), 4) for x in nulls[name]]
      path.write_text(json.dumps(res, default=float))
      print(f'[shard {args.shard}: {i + 1}/{len(mine)}] target {t}: ' + ', '.join(
        f'{n}={ex["flag"]}(rho={ex["agreement"]:.2f})' for n, ex in res['explainers'].items()) +
        f'  {time.perf_counter() - t0:.0f}s', flush=True)

  if args.stage in ('all', 'aggregate'):
    df = aggregate(cfg, targets)
    print(df.groupby(['explainer', 'flag']).size())


if __name__ == '__main__':
  main()
