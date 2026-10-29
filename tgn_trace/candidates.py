"""Candidate events for a target (u, v, t): the last K events with t' < t touching u or v."""
import numpy as np

from tgn_trace.state import TraceEnv


def candidate_events(env: TraceEnv, target_edge_idx, k=None):
  """Edge ids of candidates, newest first (ties: larger edge_idx first)."""
  k = k or env.cfg.k_candidates
  p = env.pos(target_edge_idx)
  u, v, t = int(env.src[p]), int(env.dst[p]), env.ts[p]
  ngh = env.full_ngh_finder
  _, eu, _ = ngh.find_before(u, t)
  _, ev, _ = ngh.find_before(v, t)
  edges = np.unique(np.concatenate([eu[-k:], ev[-k:]]).astype(np.int64))
  pos = env.pos_of_edge[edges]
  order = np.lexsort((-edges, -env.ts[pos]))  # newest first, tie-break on edge_idx
  return [int(e) for e in edges[order][:k]]


def candidate_table(env: TraceEnv, target_edge_idx, k=None):
  """Candidates with time, endpoints, batch and how far back (in batches) they sit."""
  tb = int(env.batch_of[env.pos(target_edge_idx)])
  u, v = env.endpoints(target_edge_idx)
  out = []
  for rank, e in enumerate(candidate_events(env, target_edge_idx, k)):
    ev = env.event(e)
    ev['recency_rank'] = rank + 1
    ev['batch_distance'] = tb - ev['batch']
    ev['touches'] = 'both' if {ev['u'], ev['v']} == {u, v} else ('u' if u in (ev['u'], ev['v']) else 'v')
    out.append(ev)
  return out
