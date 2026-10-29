"""Stage 0/2: the baseline pass.

Baseline state definition: reset memory, then one clean, deterministic, no_grad, eval-mode pass of
the final checkpoint over the whole stream (train -> val -> test regions, every event, original
batch boundaries). This differs slightly from the memory state at the end of training (training
skips edges of held-out "new" nodes and runs with dropout and gradient updates in between), so
the AP/AUC reported here is our own number, printed next to the training log's.

Outputs:
  results/<data>/baseline_scores.csv    score of every val/test event (batched, normal path)
  results/<data>/baseline_metrics.json  test AP/AUC (repo-style mean over batches, and pooled)
  checkpoints_state/<data>-C<C>/        state at the start of every C-th batch (--checkpoints)

Usage: python -m tgn_trace.baseline [--checkpoints]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.utils import RandEdgeSampler
from tgn_trace.checkpoints import CheckpointStore
from tgn_trace.config import TraceConfig
from tgn_trace.state import TraceEnv


def baseline_pass(env: TraceEnv, save_checkpoints=False, progress=True):
  env.reset_state()
  ngh = env.full_ngh_finder
  # Same negative sampler as the repo's test evaluation (seed=2, reset at the start of test)
  test_sampler = RandEdgeSampler(env.src, env.dst, seed=2)
  store = CheckpointStore(env) if save_checkpoints else None
  planned = set(store.planned_batches()) if store else set()

  rows, batch_ap, batch_auc, pooled_pos, pooled_neg = [], [], [], [], []
  t0 = time.perf_counter()
  for b, positions in enumerate(env.batches):
    if b in planned:
      store.save(b, env.snapshot())
    region = env.batch_region[b]
    if region == 2:
      if b == env.first_test_batch:
        test_sampler.reset_random_state()
      _, negatives = test_sampler.sample(len(positions))
    else:
      negatives = env.dst[positions]
    pos_prob, neg_prob = env.batched_probabilities(positions, negatives, ngh)
    if region >= 1:
      for p, s, ns in zip(positions, pos_prob, neg_prob):
        rows.append((int(env.edge_idx[p]), int(p), int(env.src[p]), int(env.dst[p]), float(env.ts[p]),
                     b, ['train', 'val', 'test'][region], float(s), float(ns) if region == 2 else np.nan))
    if region == 2:
      y = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
      s = np.concatenate([pos_prob, neg_prob])
      batch_ap.append(average_precision_score(y, s))
      batch_auc.append(roc_auc_score(y, s))
      pooled_pos.append(pos_prob)
      pooled_neg.append(neg_prob)
    if progress and b % 100 == 0:
      print(f'  batch {b}/{env.n_batches}  {time.perf_counter() - t0:.1f}s', flush=True)

  if store:
    store.finalize(sorted(planned))
  pp, pn = np.concatenate(pooled_pos), np.concatenate(pooled_neg)
  y = np.concatenate([np.ones(len(pp)), np.zeros(len(pn))])
  metrics = dict(test_ap_batch_mean=float(np.mean(batch_ap)), test_auc_batch_mean=float(np.mean(batch_auc)),
                 test_ap_pooled=float(average_precision_score(y, np.concatenate([pp, pn]))),
                 test_auc_pooled=float(roc_auc_score(y, np.concatenate([pp, pn]))),
                 n_test_events=int(len(pp)), n_batches=env.n_batches,
                 first_test_batch=env.first_test_batch, seconds=time.perf_counter() - t0,
                 frac_test_scores_ge_threshold=float(np.mean(pp >= env.cfg.threshold)))
  if store:
    metrics['checkpoint_batches'] = len(planned)
    metrics['checkpoint_disk_mb'] = store.disk_bytes() / 2 ** 20
  df = pd.DataFrame(rows, columns=['edge_idx', 'pos', 'u', 'v', 't', 'batch', 'region', 'score',
                                   'neg_score'])
  return df, metrics


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoints', action='store_true', help='also write state checkpoints')
  args = ap.parse_args()
  cfg = TraceConfig()
  env = TraceEnv(cfg)
  print(f'data={cfg.data} model={cfg.model_file} device={env.device} events={env.n_events} '
        f'batches={env.n_batches} first_test_batch={env.first_test_batch}')
  df, metrics = baseline_pass(env, save_checkpoints=args.checkpoints)
  cfg.results_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(cfg.results_dir / 'baseline_scores.csv', index=False)
  (cfg.results_dir / 'baseline_metrics.json').write_text(json.dumps(metrics, indent=2))
  print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
  main()
