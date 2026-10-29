"""Periodic state checkpoints on disk.

A checkpoint for batch b is the model state at the START of batch b of the baseline pass (CPU
tensors, packed). Only the test region plus `checkpoint_lead_batches` batches before it are
covered densely (every C batches), the rest of the stream sparsely (every `checkpoint_sparse_every`
batches); batch 0 (the reset state) is always implicitly available.
"""
import json
from collections import OrderedDict

import torch

from tgn_trace.state import TraceEnv


class CheckpointStore:
  def __init__(self, env: TraceEnv):
    self.env = env
    self.dir = env.cfg.checkpoint_dir
    self._cache = OrderedDict()
    self.batches = []
    index = self.dir / 'index.json'
    if index.exists():
      meta = json.loads(index.read_text())
      if meta.get('n_events') == env.n_events and meta.get('model_file') == str(env.cfg.model_file):
        self.batches = sorted(meta['batches'])

  def planned_batches(self):
    cfg = self.env.cfg
    start = max(0, self.env.first_test_batch - cfg.checkpoint_lead_batches)
    dense = list(range(start, self.env.n_batches, cfg.checkpoint_every))
    sparse = list(range(cfg.checkpoint_sparse_every, start, cfg.checkpoint_sparse_every)) \
      if cfg.checkpoint_sparse_every else []
    return sorted(set(sparse) | set(dense))

  def path(self, b):
    return self.dir / f'state_b{b:05d}.pt'

  def save(self, b, snap):
    self.dir.mkdir(parents=True, exist_ok=True)
    torch.save(TraceEnv.pack(snap), self.path(b))

  def finalize(self, batches):
    self.batches = sorted(batches)
    meta = dict(batches=self.batches, n_events=self.env.n_events,
                model_file=str(self.env.cfg.model_file), checkpoint_every=self.env.cfg.checkpoint_every)
    (self.dir / 'index.json').write_text(json.dumps(meta))

  def disk_bytes(self):
    return sum(self.path(b).stat().st_size for b in self.batches if self.path(b).exists())

  def latest_at_or_before(self, b):
    """Largest checkpointed batch <= b, or 0 (the reset state) if none."""
    best = 0
    for c in self.batches:
      if c <= b:
        best = c
      else:
        break
    return best

  def restore(self, b):
    """Put the env's model into the state at the start of batch b (b must be 0 or checkpointed)."""
    if b == 0:
      self.env.reset_state()
      return
    packed = self._cache.get(b)
    if packed is None:
      packed = torch.load(self.path(b), map_location='cpu')
      self._cache[b] = packed
      while len(self._cache) > self.env.cfg.checkpoint_cache_size:
        self._cache.popitem(last=False)
    else:
      self._cache.move_to_end(b)
    self.env.restore(self.env.unpack(packed))
