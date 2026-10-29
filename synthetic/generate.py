"""Generate a Wikipedia-shaped temporal interaction dataset in JODIE CSV format.

Why this exists: the Wikipedia JODIE file (snap.stanford.edu/jodie/wikipedia.csv) could not be
downloaded in the environment this project was built in. This generator produces a stand-in with
the same schema (bipartite user -> page edits, a 0/1 state label, 172 edge features) so every
stage of TGN-Trace can be built and gated. Numbers measured on it are NOT Wikipedia numbers.

It also plants trigger events (Stage 7): for a few (trigger page a -> consequence page b) pairs,
a user who edits `a` edits `b` shortly after with high probability. The consequence/trigger edge
ids are written to a separate ground-truth file and are never used as features.

Output (run from the repo root):
  data/<name>.csv                 JODIE format, consumed by utils/preprocess_data.py
  data/<name>_ground_truth.csv    consequence_idx, trigger_idx (0-based JODIE line index)
"""
import argparse
import json
from pathlib import Path

import numpy as np

N_FEAT = 172


def generate(n_users, n_items, n_events, n_triggers, seed, span_seconds):
  rng = np.random.RandomState(seed)

  user_activity = rng.pareto(1.2, n_users) + 1.0
  user_activity /= user_activity.sum()
  item_pop = 1.0 / np.arange(1, n_items + 1) ** 0.9
  item_pop = item_pop[rng.permutation(n_items)]
  item_pop /= item_pop.sum()

  # Each page has a latent topic vector; edge features are a noisy projection of it, so features
  # carry information about the destination (like LIWC features of the edit text do).
  latent_dim = 16
  item_latent = rng.normal(size=(n_items, latent_dim))
  projection = rng.normal(size=(latent_dim, N_FEAT)) / np.sqrt(latent_dim)

  # Trigger pairs: trigger page a -> consequence page b, drawn from mid-popularity pages
  order = np.argsort(-item_pop)
  mid = order[n_items // 10: n_items // 2]
  chosen = rng.choice(mid, size=2 * n_triggers, replace=False)
  triggers = {int(a): int(b) for a, b in zip(chosen[:n_triggers], chosen[n_triggers:])}

  times = np.sort(rng.uniform(0, span_seconds, n_events))
  history = [[] for _ in range(n_users)]
  events = []  # (user, item, ts, trigger_event_position or -1)
  pending = []  # scheduled consequence events: (ts, user, item, trigger_position)

  t_ptr = 0
  while len(events) < n_events and (t_ptr < n_events or pending):
    next_base = times[t_ptr] if t_ptr < n_events else np.inf
    if pending and pending[0][0] <= next_base:
      ts, u, i, trig_pos = pending.pop(0)
    else:
      ts = next_base
      t_ptr += 1
      u = int(rng.choice(n_users, p=user_activity))
      trig_pos = -1
      if history[u] and rng.rand() < 0.75:
        # Revisit: recency-weighted choice from the user's own history
        hist = history[u][-10:]
        w = 0.5 ** np.arange(len(hist))[::-1]
        i = int(hist[rng.choice(len(hist), p=w / w.sum())])
      else:
        i = int(rng.choice(n_items, p=item_pop))
    events.append((u, i, ts, trig_pos))
    history[u].append(i)
    if trig_pos == -1 and i in triggers and rng.rand() < 0.8:
      delay = rng.uniform(600, 3 * 3600)
      pending.append((ts + delay, u, triggers[i], len(events) - 1))
      pending.sort(key=lambda x: x[0])

  # Stable sort by time (ties broken by generation order) and remap trigger positions
  order = sorted(range(len(events)), key=lambda k: (events[k][2], k))
  new_pos = {old: new for new, old in enumerate(order)}
  events = [events[k] for k in order]
  events = [(u, i, ts, new_pos[tp] if tp >= 0 else -1) for (u, i, ts, tp) in events]

  # Contiguous ids (preprocess_data.reindex asserts that every id in [min, max] is used)
  users = {u: k for k, u in enumerate(sorted({e[0] for e in events}))}
  items = {i: k for k, i in enumerate(sorted({e[1] for e in events}))}

  rows, feats, gt = [], [], []
  for pos, (u, i, ts, tp) in enumerate(events):
    label = 1 if rng.rand() < 0.002 else 0
    f = item_latent[i] @ projection + 0.5 * rng.normal(size=N_FEAT)
    rows.append((users[u], items[i], float(np.round(ts, 1)), label))
    feats.append(f)
    if tp >= 0:
      gt.append((pos, tp))
  return rows, np.asarray(feats, dtype=np.float32), gt, triggers


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--name', default='synthwiki')
  p.add_argument('--n_users', type=int, default=6000)
  p.add_argument('--n_items', type=int, default=1000)
  p.add_argument('--n_events', type=int, default=80000)
  p.add_argument('--n_triggers', type=int, default=20)
  p.add_argument('--span_days', type=float, default=30.0)
  p.add_argument('--seed', type=int, default=0)
  a = p.parse_args()

  rows, feats, gt, triggers = generate(a.n_users, a.n_items, a.n_events, a.n_triggers, a.seed,
                                       a.span_days * 86400)
  out = Path('data')
  out.mkdir(exist_ok=True)
  with open(out / f'{a.name}.csv', 'w') as f:
    f.write('user_id,item_id,timestamp,state_label,comma_separated_list_of_features\n')
    for (u, i, ts, label), feat in zip(rows, feats):
      f.write(f'{u},{i},{ts},{label},' + ','.join(f'{x:.4f}' for x in feat) + '\n')
  with open(out / f'{a.name}_ground_truth.csv', 'w') as f:
    f.write('consequence_idx,trigger_idx\n')
    for c, t in gt:
      f.write(f'{c},{t}\n')
  (out / f'{a.name}_meta.json').write_text(json.dumps(
    {**vars(a), 'n_rows': len(rows), 'n_planted': len(gt)}, indent=2))
  print(f'wrote {len(rows)} events, {len(gt)} planted trigger->consequence pairs')


if __name__ == '__main__':
  main()
