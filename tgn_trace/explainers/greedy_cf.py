"""Greedy minimal counterfactual search.

Repeatedly add the candidate whose removal, together with the already selected set, lowers the
score most. Stop when the score drops below the threshold or the set reaches `max_size`.
"""
import time

from tgn_trace.candidates import candidate_events


def explain(env, target_edge_idx, replayer, k=None, max_size=None):
  t0 = time.perf_counter()
  max_size = max_size or env.cfg.greedy_max_size
  thr = env.cfg.threshold
  cands = candidate_events(env, target_edge_idx, k)
  base = replayer.baseline[int(target_edge_idx)]
  selected, trajectory, evals, fresh = [], [], 0, 0
  last_round = {}
  score = base
  while len(selected) < max_size and score >= thr:
    remaining = [e for e in cands if e not in selected]
    if not remaining:
      break
    round_scores = {}
    for e in remaining:
      r = replayer.checkpointed_replay(target_edge_idx, selected + [e])
      evals += 1
      fresh += 0 if r.get('cached') else 1
      round_scores[e] = r['cf_score']
    best = min(remaining, key=lambda e: (round_scores[e], -e))
    selected.append(best)
    score = round_scores[best]
    last_round = round_scores
    trajectory.append(dict(added=best, score=score))
  # Full ranking: selected set in order, then the rest by their score in the last round evaluated
  rest = sorted((e for e in cands if e not in selected), key=lambda e: (last_round.get(e, base), -e))
  ranked = selected + rest
  n = len(ranked)
  success = bool(base >= thr and score < thr)
  return dict(method='greedy', target_id=int(target_edge_idx),
              status='ok' if success else ('not_applicable' if base < thr else 'no_solution'),
              ranked=ranked, cf_set=selected, success=success, final_score=score,
              baseline_score=base, trajectory=trajectory,
              importance={e: float(n - i) for i, e in enumerate(ranked)},
              n_model_evals=evals, n_fresh_replays=fresh, runtime_ms=(time.perf_counter() - t0) * 1000.0)
