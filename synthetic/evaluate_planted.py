"""Stage 7 (optional): do explainers find the planted trigger events?

For consequence events (test region, baseline >= 0.5) whose planted trigger is among the K
candidates:
  1. First check with brute force that the trained model is actually sensitive to the trigger
     (removing it lowers the score by at least tau). Report the confirmed fraction.
  2. Only on confirmed cases: hit@1 / hit@3 of the trigger in each explainer's ranking.
Ground truth comes from data/<name>_ground_truth.csv (0-based JODIE line index; edge_idx = +1)
and is never used as a feature.

Usage: TGN_TRACE_DATA=synthwiki python synthetic/evaluate_planted.py [--max_targets 40]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgn_trace.candidates import candidate_events  # noqa: E402
from tgn_trace.config import TraceConfig  # noqa: E402
from tgn_trace.explainers import bruteforce, greedy_cf, recency  # noqa: E402
from tgn_trace.replay import Replayer  # noqa: E402
from tgn_trace.state import get_env  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--max_targets', type=int, default=40)
  args = ap.parse_args()
  cfg = TraceConfig()
  env = get_env(cfg)
  replayer = Replayer(env)
  tau = json.loads((cfg.results_dir / 'threshold_selection.json').read_text())['tau']
  gt = pd.read_csv(cfg.data_dir / f'{cfg.data}_ground_truth.csv')
  gt['target'] = gt.consequence_idx + 1
  gt['trigger'] = gt.trigger_idx + 1

  eligible = []
  n_test_consequences = 0
  for row in gt.itertuples():
    t = int(row.target)
    if env.region[env.pos(t)] != 2:
      continue
    n_test_consequences += 1
    if replayer.baseline.get(t, 0.0) < cfg.threshold:
      continue
    if int(row.trigger) in candidate_events(env, t):
      eligible.append((t, int(row.trigger)))
  eligible = eligible[:args.max_targets]

  rows = []
  for i, (t, trig) in enumerate(eligible):
    bf = bruteforce.explain(env, t, replayer)
    d = bf['deltas'][trig]
    confirmed = d <= -tau
    row = dict(target=t, trigger=trig, trigger_delta=d, confirmed=confirmed,
               trigger_bf_rank=bf['ranked'].index(trig) + 1, n_candidates=len(bf['candidates']))
    if confirmed:
      row['trigger_recency_rank'] = recency.explain(env, t)['ranked'].index(trig) + 1
      g = greedy_cf.explain(env, t, replayer)
      row['trigger_greedy_rank'] = g['ranked'].index(trig) + 1
      row['trigger_in_greedy_set'] = trig in g['cf_set']
    rows.append(row)
    print(f'[{i + 1}/{len(eligible)}] {t}: trigger {trig} delta={d:+.4f} confirmed={confirmed}', flush=True)

  df = pd.DataFrame(rows)
  df.to_csv(cfg.results_dir / 'planted_triggers.csv', index=False)
  conf = df[df.confirmed]
  summary = dict(tau=tau, n_planted_total=int(len(gt)), n_planted_in_test=n_test_consequences,
                 n_eligible_evaluated=int(len(df)), n_confirmed=int(len(conf)),
                 frac_confirmed=float(df.confirmed.mean()) if len(df) else None)
  for name in ('bf', 'recency', 'greedy'):
    col = f'trigger_{name}_rank'
    if len(conf) and col in conf:
      summary[f'{name}_hit@1'] = float((conf[col] <= 1).mean())
      summary[f'{name}_hit@3'] = float((conf[col] <= 3).mean())
  if len(conf):
    summary['greedy_set_contains_trigger'] = float(conf.trigger_in_greedy_set.mean())
  (cfg.results_dir / 'planted_triggers_summary.json').write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
