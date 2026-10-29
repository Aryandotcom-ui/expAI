"""Recency baseline: the newest candidate is the most important."""
import time

from tgn_trace.candidates import candidate_events


def explain(env, target_edge_idx, replayer=None, k=None):
  t0 = time.perf_counter()
  cands = candidate_events(env, target_edge_idx, k)
  n = len(cands)
  return dict(method='recency', target_id=int(target_edge_idx), status='ok', ranked=cands,
              importance={e: float(n - i) for i, e in enumerate(cands)},
              n_model_evals=0, runtime_ms=(time.perf_counter() - t0) * 1000.0)
