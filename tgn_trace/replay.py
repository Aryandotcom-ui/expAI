"""Memory-aware counterfactual replay.

Removing past events changes two things the target's score depends on:
  1. memory / last_update / pending messages, through every later batch, and
  2. the temporal neighborhood read by the embedding module.
Replay therefore re-runs the original batches (same boundaries, removed events dropped) with a
MaskedNeighborFinder, then scores the target from the state at the start of its batch.

Sign convention: delta = cf_score - baseline_score.
"""
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from tgn_trace.checkpoints import CheckpointStore
from tgn_trace.masked_ngh import MaskedNeighborFinder
from tgn_trace.state import TraceEnv


def score_target(env: TraceEnv, target_edge_idx, ngh_finder=None, state=None):
  """Score one target with the state at the start of its original batch.

  If `state` is given it is restored first; otherwise the env must already be in that state.
  The neighbor cutoff is the target time t (find_before is strictly before t).
  """
  if state is not None:
    env.restore(state)
  ngh_finder = ngh_finder or env.full_ngh_finder
  return float(env.score_positions([env.pos(target_edge_idx)], ngh_finder)[0])


def load_baseline_scores(env: TraceEnv):
  path = env.cfg.results_dir / 'baseline_scores.csv'
  if not path.exists():
    raise FileNotFoundError(f'{path} missing: run `python -m tgn_trace.baseline` first')
  df = pd.read_csv(path)
  return dict(zip(df.edge_idx.astype(int), df.score.astype(float)))


class Replayer:
  def __init__(self, env: TraceEnv, checkpoints: CheckpointStore = None, baseline_scores=None,
               cache_size=4096):
    self.env = env
    self.checkpoints = checkpoints if checkpoints is not None else CheckpointStore(env)
    self.baseline = baseline_scores if baseline_scores is not None else load_baseline_scores(env)
    self._cache = OrderedDict()
    self.cache_size = cache_size
    self.n_model_runs = 0

  # ------------------------------------------------------------------ helpers
  def _prepare(self, target_edge_idx, removed_edge_idxs):
    env = self.env
    tpos = env.pos(target_edge_idx)
    removed = sorted({int(e) for e in removed_edge_idxs})
    removed_pos = {env.pos(e) for e in removed}  # raises KeyError on unknown ids
    for p in removed_pos:
      if env.ts[p] >= env.ts[tpos]:
        raise ValueError(f'event {int(env.edge_idx[p])} is not strictly before the target')
    ngh = MaskedNeighborFinder(env.full_ngh_finder, removed, env.endpoints) if removed \
      else env.full_ngh_finder
    return tpos, removed, removed_pos, ngh

  def _replay_batches(self, b_start, b_end, removed_pos, ngh):
    env = self.env
    n = 0
    for b in range(b_start, b_end):
      positions = env.batches[b]
      if removed_pos:
        positions = positions[[p not in removed_pos for p in positions]]
      env.advance_state(positions, ngh)
      n += len(positions)
    return n

  def _result(self, target_edge_idx, removed, cf_score, replayed, t0, checkpoint_used):
    base = self.baseline[int(target_edge_idx)]
    thr = self.env.cfg.threshold
    return dict(target_id=int(target_edge_idx), baseline_score=base, removed=removed,
                cf_score=cf_score, delta=cf_score - base,
                baseline_class=int(base >= thr), cf_class=int(cf_score >= thr),
                replayed_events=int(replayed), latency_ms=(time.perf_counter() - t0) * 1000.0,
                checkpoint_used=checkpoint_used)

  # ------------------------------------------------------------------ replays
  def full_replay(self, target_edge_idx, removed_edge_idxs=()):
    """Reference implementation: reset, replay everything before the target's batch, score."""
    t0 = time.perf_counter()
    env = self.env
    tpos, removed, removed_pos, ngh = self._prepare(target_edge_idx, removed_edge_idxs)
    env.reset_state()
    replayed = self._replay_batches(0, int(env.batch_of[tpos]), removed_pos, ngh)
    cf = score_target(env, target_edge_idx, ngh)
    self.n_model_runs += 1
    return self._result(target_edge_idx, removed, cf, replayed, t0, None)

  def checkpointed_replay(self, target_edge_idx, removed_edge_idxs=(), use_cache=True):
    """Restore the latest checkpoint at/before the earliest removed event's batch, replay forward."""
    key = (int(target_edge_idx), frozenset(int(e) for e in removed_edge_idxs))
    if use_cache and key in self._cache:
      self._cache.move_to_end(key)
      res = dict(self._cache[key])
      res['cached'] = True
      return res
    t0 = time.perf_counter()
    env = self.env
    tpos, removed, removed_pos, ngh = self._prepare(target_edge_idx, removed_edge_idxs)
    tb = int(env.batch_of[tpos])
    earliest = min((int(env.batch_of[p]) for p in removed_pos), default=tb)
    ckpt = self.checkpoints.latest_at_or_before(min(earliest, tb))
    self.checkpoints.restore(ckpt)
    replayed = self._replay_batches(ckpt, tb, removed_pos, ngh)
    cf = score_target(env, target_edge_idx, ngh)
    self.n_model_runs += 1
    res = self._result(target_edge_idx, removed, cf, replayed, t0, int(ckpt))
    res['cached'] = False
    if use_cache:
      self._cache[key] = dict(res)
      while len(self._cache) > self.cache_size:
        self._cache.popitem(last=False)
    return res

  def counterfactual_state(self, target_edge_idx, removed_edge_idxs=()):
    """Leave the env in the (counterfactual) state at the start of the target's batch.

    Returns the neighbor finder to use with it. Used by Signal C.
    """
    env = self.env
    tpos, removed, removed_pos, ngh = self._prepare(target_edge_idx, removed_edge_idxs)
    tb = int(env.batch_of[tpos])
    earliest = min((int(env.batch_of[p]) for p in removed_pos), default=tb)
    ckpt = self.checkpoints.latest_at_or_before(min(earliest, tb))
    self.checkpoints.restore(ckpt)
    self._replay_batches(ckpt, tb, removed_pos, ngh)
    return ngh
