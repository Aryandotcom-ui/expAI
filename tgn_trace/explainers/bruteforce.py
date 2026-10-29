"""Brute force single-event removal: model-level ground truth for single-event influence.

For every candidate, replay with that one event removed and record delta = cf - baseline.
Candidates are ranked by -delta (the removal that lowers the score most comes first).
Results are cached to results/<data>/bruteforce/<target>.json.
"""
import json
import time

from tgn_trace.candidates import candidate_events


def cache_path(env, target_edge_idx):
  return env.cfg.results_dir / 'bruteforce' / f'{int(target_edge_idx)}.json'


def explain(env, target_edge_idx, replayer, k=None, use_cache=True):
  path = cache_path(env, target_edge_idx)
  cands = candidate_events(env, target_edge_idx, k)
  if use_cache and path.exists():
    cached = json.loads(path.read_text())
    if cached['candidates'] == cands:
      cached['importance'] = {int(e): v for e, v in cached['importance'].items()}
      cached['deltas'] = {int(e): v for e, v in cached['deltas'].items()}
      cached['from_cache'] = True
      return cached
  t0 = time.perf_counter()
  deltas, latencies = {}, []
  for e in cands:
    r = replayer.checkpointed_replay(target_edge_idx, [e])
    deltas[e] = r['delta']
    latencies.append(r['latency_ms'])
  ranked = sorted(cands, key=lambda e: (deltas[e], -e))
  out = dict(method='bruteforce', target_id=int(target_edge_idx), status='ok', ranked=ranked,
             candidates=cands, deltas=deltas, importance={e: -d for e, d in deltas.items()},
             baseline_score=replayer.baseline[int(target_edge_idx)],
             n_model_evals=len(cands), runtime_ms=(time.perf_counter() - t0) * 1000.0,
             replay_latency_ms=latencies)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(out))
  out['from_cache'] = False
  return out
