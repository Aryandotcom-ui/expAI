"""CoDy (Gomm et al.) search, evaluated with TGN-Trace's exact replay.

We reuse CoDy's Monte-Carlo-style tree search unchanged (vendored in third_party/cody, MIT) and
override only the two places where it talks to the model:
  * `initialize_explanation`: the candidate "subgraph" is our candidate set (last K events of u or
    v), and the original prediction is the baseline score's logit;
  * `calculate_subgraph_prediction`: the score after removing a set of events comes from
    `Replayer.checkpointed_replay` (exact memory + neighborhood replay), not from CoDy's
    `TGNWrapper`, whose default `approximate_predictions=True` replays only candidate events.
CoDy works in logit space and calls a node counterfactual when the logit changes sign, which is
the same as crossing our 0.5 threshold.
"""
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tgn_trace.candidates import candidate_events
from tgn_trace.config import ROOT

_VENDOR = ROOT / 'third_party' / 'cody'
if str(_VENDOR) not in sys.path:
  sys.path.insert(0, str(_VENDOR))

from cody.constants import COL_ID, COL_SUBGRAPH_DISTANCE, COL_TIMESTAMP  # noqa: E402
from cody.explainer.cody import CoDy  # noqa: E402

_EPS = 1e-7


def _logit(p):
  p = min(max(p, _EPS), 1 - _EPS)
  return math.log(p / (1 - p))


class _Timeout(Exception):
  pass


class _NoModelWrapper:
  """The bits of CoDy's TGNNWrapper interface the search touches outside model evaluation."""
  num_hops = 1
  dataset = SimpleNamespace(directed=False, events=pd.DataFrame())

  def set_evaluation_mode(self, mode):
    pass

  def reset_model(self):
    pass

  def remove_memory_backup(self, label):
    pass


class ReplayCoDy(CoDy):
  def __init__(self, env, replayer, max_steps=60, timeout_s=120.0, selection_policy='temporal'):
    super().__init__(_NoModelWrapper(), candidates_size=env.cfg.k_candidates,
                     selection_policy=selection_policy, max_steps=max_steps,
                     approximate_predictions=False)
    self.env, self.replayer, self.timeout_s = env, replayer, timeout_s
    self.n_evals, self.deadline, self.single = 0, None, {}

  def initialize_explanation(self, explained_event_id):
    cands = candidate_events(self.env, explained_event_id)
    rows = [dict(**{COL_ID: e, COL_TIMESTAMP: self.env.event(e)['t'], COL_SUBGRAPH_DISTANCE: 1})
            for e in cands]
    subgraph = pd.DataFrame(rows).sort_values([COL_TIMESTAMP, COL_ID]).reset_index(drop=True)
    original = _logit(self.replayer.baseline[int(explained_event_id)])
    return original, self._create_sampler(subgraph, explained_event_id, original)

  def calculate_subgraph_prediction(self, candidate_events, cf_example_events, explained_event_id,
                                    candidate_event_id, original_prediction, memory_label=None):
    if time.perf_counter() > self.deadline:
      raise _Timeout()
    removed = list(cf_example_events) + [candidate_event_id]
    r = self.replayer.checkpointed_replay(int(explained_event_id), [int(e) for e in removed])
    self.n_evals += 1
    if len(removed) == 1:
      self.single[int(candidate_event_id)] = r['cf_score']
    return _logit(r['cf_score'])


def explain(env, target_edge_idx, replayer, max_steps=60, timeout_s=120.0):
  t0 = time.perf_counter()
  cody = ReplayCoDy(env, replayer, max_steps=max_steps, timeout_s=timeout_s)
  cody.deadline = t0 + timeout_s
  cands = candidate_events(env, target_edge_idx)
  base = replayer.baseline[int(target_edge_idx)]
  try:
    ex = cody.explain(int(target_edge_idx))
    cf_set = [int(e) for e in ex.event_ids]
    status = 'ok' if ex.achieves_counterfactual_explanation else 'no_solution'
    cf_score = 1 / (1 + math.exp(-ex.counterfactual_prediction)) if ex.counterfactual_prediction is not None else None
  except _Timeout:
    cf_set, status, cf_score = [], 'timeout', None
  if base < env.cfg.threshold:
    status = 'not_applicable' if status != 'timeout' else status
  # Full ranking: CoDy's set in order, then the rest by their single-removal score (when CoDy
  # evaluated it), then by recency.
  rest = [e for e in cands if e not in cf_set]
  rest.sort(key=lambda e: (cody.single.get(e, math.inf), cands.index(e)))
  ranked = cf_set + rest
  n = len(ranked)
  return dict(method='cody', target_id=int(target_edge_idx), status=status, ranked=ranked,
              cf_set=cf_set, success=status == 'ok', final_score=cf_score, baseline_score=base,
              importance={e: float(n - i) for i, e in enumerate(ranked)},
              n_model_evals=cody.n_evals, runtime_ms=(time.perf_counter() - t0) * 1000.0)
