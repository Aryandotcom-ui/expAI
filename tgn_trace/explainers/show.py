"""Stage 3 gate: for a few test targets, print each explainer's top-5 next to brute-force deltas.

Usage: python -m tgn_trace.explainers.show [--n 5] [--no_cody]
"""
import argparse

from tgn_trace.config import TraceConfig
from tgn_trace.explainers import bruteforce, greedy_cf, recency
from tgn_trace.replay import Replayer
from tgn_trace.state import get_env
from tgn_trace.verify.run import pick_targets


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=5)
  ap.add_argument('--no_cody', action='store_true')
  args = ap.parse_args()
  env = get_env(TraceConfig())
  replayer = Replayer(env)
  methods = {'recency': lambda t: recency.explain(env, t),
             'greedy': lambda t: greedy_cf.explain(env, t, replayer)}
  if not args.no_cody:
    from tgn_trace.explainers import cody_adapter
    methods['cody'] = lambda t: cody_adapter.explain(env, t, replayer)
  for t in pick_targets(replayer, 'test', args.n, seed=1):
    bf = bruteforce.explain(env, t, replayer)
    d = bf['deltas']
    u, v = env.endpoints(t)
    print(f'\n=== target {t}: {u} -> {v}, baseline {bf["baseline_score"]:.4f}, '
          f'{len(bf["candidates"])} candidates ===')
    print(f'  {"bruteforce":<10} ' + '  '.join(f'{e}({d[e]:+.4f})' for e in bf['ranked'][:5]))
    for name, fn in methods.items():
      ex = fn(t)
      extra = ''
      if 'cf_set' in ex:
        fs = ex.get('final_score')
        extra = (f'  [status={ex["status"]}, set size={len(ex["cf_set"])}, final score='
                 f'{fs:.4f}, evals={ex["n_model_evals"]}]' if fs is not None else f'  [status={ex["status"]}]')
      print(f'  {name:<10} ' + '  '.join(f'{e}({d[e]:+.4f})' for e in ex['ranked'][:5]) + extra)


if __name__ == '__main__':
  main()
