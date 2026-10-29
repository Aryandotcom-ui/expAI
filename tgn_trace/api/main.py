"""TGN-Trace API. Every number comes from the model, the replay engine or their on-disk caches.

Run: uvicorn tgn_trace.api.main:app --port 8000   (then open http://localhost:8000/)
"""
import json
import threading
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from tgn_trace.api.schemas import (CounterfactualRequest, CounterfactualResponse, ExplainRequest,
                                   ExplainResponse, Health, Target, Trace, TraceEvent)
from tgn_trace.candidates import candidate_events, candidate_table
from tgn_trace.config import ROOT, TraceConfig
from tgn_trace.explainers import bruteforce, greedy_cf, recency
from tgn_trace.replay import Replayer
from tgn_trace.state import get_env

app = FastAPI(title='TGN-Trace', version='0.1')
# The model state is one shared mutable object: serialize every request that touches it.
_LOCK = threading.Lock()
_GREEDY = {}
N_DEMO_TARGETS = 30


@lru_cache(maxsize=1)
def ctx():
  cfg = TraceConfig()
  env = get_env(cfg)
  replayer = Replayer(env)
  return cfg, env, replayer


def _results():
  return ctx()[0].results_dir


def _tau():
  path = _results() / 'threshold_selection.json'
  return json.loads(path.read_text())['tau'] if path.exists() else None


def _verification(target_id):
  # verification/ holds the experiment set (verify/run.py); verification_api/ holds results the
  # API computed on demand, kept apart so they never mix into the experiments.
  for sub in ('verification', 'verification_api'):
    path = _results() / sub / f'{int(target_id)}.json'
    if path.exists():
      return json.loads(path.read_text())
  return None


def _check_target(target_id):
  cfg, env, replayer = ctx()
  if int(target_id) not in replayer.baseline:
    raise HTTPException(404, f'unknown target id {target_id} (must be a val/test event id)')
  return cfg, env, replayer


def _target(env, replayer, target_id):
  ev = env.event(target_id)
  return Target(id=int(target_id), u=ev['u'], v=ev['v'], t=ev['t'],
                baseline_score=replayer.baseline[int(target_id)],
                has_verification=(_results() / 'verification' / f'{int(target_id)}.json').exists())


@app.get('/health', response_model=Health)
def health():
  cfg, env, replayer = ctx()
  metrics_path = cfg.results_dir / 'baseline_metrics.json'
  return Health(status='ok', data=cfg.data, model_file=str(cfg.model_file), device=str(env.device),
                n_events=env.n_events, n_batches=env.n_batches,
                first_test_batch=env.first_test_batch, checkpoint_every=cfg.checkpoint_every,
                n_checkpoints=len(replayer.checkpoints.batches),
                checkpoint_disk_mb=replayer.checkpoints.disk_bytes() / 2 ** 20,
                baseline_metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
                tau=_tau())


@app.get('/api/targets', response_model=list[Target])
def targets():
  cfg, env, replayer = ctx()
  verified = sorted(int(p.stem) for p in (cfg.results_dir / 'verification').glob('*.json'))
  # Put targets with a successful greedy counterfactual first (most informative to replay)
  e2 = cfg.results_dir / 'e2_explainers.csv'
  if e2.exists():
    df = pd.read_csv(e2)
    flips = set(df[(df.method == 'greedy') & (df.cf_success == True)].target.astype(int))  # noqa: E712
    verified.sort(key=lambda t: (t not in flips, t))
  ids = verified[:N_DEMO_TARGETS]
  if len(ids) < N_DEMO_TARGETS:
    df = pd.read_csv(cfg.results_dir / 'baseline_scores.csv')
    df = df[(df.region == 'test') & (df.score >= cfg.threshold) & ~df.edge_idx.isin(ids)]
    ids += [int(e) for e in df.sample(min(len(df), N_DEMO_TARGETS - len(ids)), random_state=0).edge_idx]
  return [_target(env, replayer, t) for t in sorted(ids)]


@app.get('/api/targets/{target_id}/trace', response_model=Trace)
def trace(target_id: int):
  cfg, env, replayer = _check_target(target_id)
  with _LOCK:
    table = candidate_table(env, target_id)
    bf = bruteforce.explain(env, target_id, replayer)
  ranks = {'bruteforce': {e: i + 1 for i, e in enumerate(bf['ranked'])}}
  if target_id in _GREEDY:
    ranks['greedy'] = {e: i + 1 for i, e in enumerate(_GREEDY[target_id]['ranked'])}
  else:
    ver = _verification(target_id)
    if ver and 'greedy' in ver['explainers']:
      ranks['greedy'] = {int(e): i + 1 for i, e in enumerate(ver['explainers']['greedy']['ranked'])}
  events = []
  for ev in table:
    r = {'recency': ev['recency_rank']}
    r.update({name: m[ev['edge_idx']] for name, m in ranks.items() if ev['edge_idx'] in m})
    events.append(TraceEvent(edge_idx=ev['edge_idx'], u=ev['u'], v=ev['v'], t=ev['t'],
                             batch=ev['batch'], batch_distance=ev['batch_distance'],
                             touches=ev['touches'], ranks=r,
                             bruteforce_delta=bf['deltas'][ev['edge_idx']]))
  base = replayer.baseline[int(target_id)]
  return Trace(target=_target(env, replayer, target_id), threshold=cfg.threshold,
               baseline_class=int(base >= cfg.threshold), k=cfg.k_candidates, events=events)


@app.post('/api/explain', response_model=ExplainResponse)
def explain(req: ExplainRequest):
  cfg, env, replayer = _check_target(req.target_id)
  if req.method == 'cody':
    raise HTTPException(501, 'CoDy is not available in this build; see docs/cody_notes.md')
  with _LOCK:
    if req.method == 'recency':
      ex = recency.explain(env, req.target_id)
    elif req.method == 'bruteforce':
      ex = bruteforce.explain(env, req.target_id, replayer)
    else:
      ex = greedy_cf.explain(env, req.target_id, replayer)
      _GREEDY[req.target_id] = ex
  extra = {k: ex[k] for k in ('cf_set', 'success', 'final_score', 'deltas', 'trajectory',
                              'n_fresh_replays', 'from_cache') if k in ex}
  return ExplainResponse(target_id=req.target_id, method=req.method, status=ex['status'],
                         ranked=ex['ranked'], importance=ex['importance'],
                         runtime_ms=ex['runtime_ms'], n_model_evals=ex['n_model_evals'], extra=extra)


@app.post('/api/counterfactual', response_model=CounterfactualResponse)
def counterfactual(req: CounterfactualRequest):
  cfg, env, replayer = _check_target(req.target_id)
  removed = list(dict.fromkeys(int(e) for e in req.removed_edge_idxs))
  if len(removed) > cfg.max_removed:
    raise HTTPException(422, f'at most {cfg.max_removed} events can be removed (got {len(removed)})')
  with _LOCK:
    cands = set(candidate_events(env, req.target_id))
    bad = [e for e in removed if e not in cands]
    if bad:
      raise HTTPException(422, f'events {bad} are not candidates of target {req.target_id}')
    res = replayer.checkpointed_replay(req.target_id, removed)
  return CounterfactualResponse(**res)


@app.get('/api/targets/{target_id}/verification')
def verification(target_id: int):
  cfg, env, replayer = _check_target(target_id)
  ver = _verification(target_id)
  if ver is not None:
    return ver
  tau = _tau()
  if tau is None:
    raise HTTPException(409, 'no threshold yet: run `python -m tgn_trace.verify.run` first')
  from tgn_trace.verify.run import verify_target
  with _LOCK:
    ver, _ = verify_target(replayer, target_id, tau)
  path = cfg.results_dir / 'verification_api' / f'{int(target_id)}.json'
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(ver, default=float))
  return json.loads(path.read_text())


app.mount('/ui', StaticFiles(directory=ROOT / 'tgn_trace' / 'ui', html=True), name='ui')


@app.get('/', include_in_schema=False)
def index(request: Request):
  query = f'?{request.url.query}' if request.url.query else ''
  return RedirectResponse(f'/ui/{query}')
