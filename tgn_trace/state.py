"""Load the trained TGN and the event stream, and expose the model state as a value.

Model state = (memory.memory, memory.last_update, memory.messages). `Memory.backup_memory()` and
`restore_memory()` deep-copy all three; we use them for in-RAM snapshots and add a packed format
for writing snapshots to disk.

The event stream is split into the original train / val / test regions (same 0.70 / 0.85 time
quantiles as utils.data_processing.get_data) and each region is cut into batches of `batch_size`
events, exactly like the training loop and eval_edge_prediction do. Every event keeps that batch
id forever: replay with removals shrinks a batch, it never moves events between batches.
"""
import contextlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from model.tgn import TGN
from utils.data_processing import get_data, compute_time_statistics
from utils.utils import get_neighbor_finder

from tgn_trace.config import TraceConfig, set_determinism, ROOT


@contextlib.contextmanager
def _chdir(path: Path):
  # The upstream loaders read './data/...' relative to the working directory.
  old = os.getcwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(old)


class TraceEnv:
  """Everything replay needs: model, stream, batches, neighbor finder and state helpers."""

  def __init__(self, cfg: TraceConfig = None, quiet=True):
    self.cfg = cfg or TraceConfig()
    set_determinism(self.cfg.seed)
    self.device = self.cfg.torch_device()

    with _chdir(ROOT), open(os.devnull, 'w') as devnull, \
        (contextlib.redirect_stdout(devnull) if quiet else contextlib.nullcontext()):
      node_features, edge_features, full_data, *_ = get_data(self.cfg.data)

    # Stable order: by timestamp, ties broken by edge_idx
    order = np.lexsort((full_data.edge_idxs, full_data.timestamps))
    self.src = full_data.sources[order].astype(np.int64)
    self.dst = full_data.destinations[order].astype(np.int64)
    self.ts = full_data.timestamps[order].astype(np.float64)
    self.edge_idx = full_data.edge_idxs[order].astype(np.int64)
    self.labels = full_data.labels[order]
    self.n_events = len(self.src)
    self.pos_of_edge = np.full(self.edge_idx.max() + 1, -1, dtype=np.int64)
    self.pos_of_edge[self.edge_idx] = np.arange(self.n_events)

    self.val_time, self.test_time = list(np.quantile(full_data.timestamps, [0.70, 0.85]))
    self.region = np.where(self.ts <= self.val_time, 0, np.where(self.ts <= self.test_time, 1, 2))
    self._build_batches()

    self.full_ngh_finder = get_neighbor_finder(full_data, uniform=False)
    self.node_features = node_features
    self.edge_features = edge_features
    self.n_nodes = node_features.shape[0]

    m_src, s_src, m_dst, s_dst = compute_time_statistics(full_data.sources, full_data.destinations,
                                                         full_data.timestamps)
    c = self.cfg
    self.model = TGN(neighbor_finder=self.full_ngh_finder, node_features=node_features,
                     edge_features=edge_features, device=self.device, n_layers=c.n_layer,
                     n_heads=c.n_head, dropout=c.drop_out, use_memory=True,
                     message_dimension=c.message_dim, memory_dimension=c.memory_dim,
                     memory_update_at_start=True, embedding_module_type=c.embedding_module,
                     message_function=c.message_function, aggregator_type=c.aggregator,
                     memory_updater_type=c.memory_updater, n_neighbors=c.n_degree,
                     mean_time_shift_src=m_src, std_time_shift_src=s_src,
                     mean_time_shift_dst=m_dst, std_time_shift_dst=s_dst)
    state_dict = torch.load(self.cfg.model_file, map_location=self.device)
    self.model.load_state_dict(state_dict)
    self.model = self.model.to(self.device)
    self.model.eval()
    self._fast_ok = os.environ.get('TGN_TRACE_FAST_ADVANCE', '1') != '0'
    self.reset_state()

  # ------------------------------------------------------------------ stream / batches
  def _build_batches(self):
    bs = self.cfg.batch_size
    self.batches = []        # list of np.ndarray of stream positions
    self.batch_region = []
    for r in (0, 1, 2):
      positions = np.nonzero(self.region == r)[0]
      for s in range(0, len(positions), bs):
        self.batches.append(positions[s:s + bs])
        self.batch_region.append(r)
    self.batch_of = np.empty(self.n_events, dtype=np.int64)
    for b, positions in enumerate(self.batches):
      self.batch_of[positions] = b
    self.first_test_batch = self.batch_region.index(2)
    self.n_batches = len(self.batches)

  def pos(self, edge_idx):
    p = int(self.pos_of_edge[int(edge_idx)]) if 0 <= int(edge_idx) < len(self.pos_of_edge) else -1
    if p < 0:
      raise KeyError(f'unknown edge_idx {edge_idx}')
    return p

  def endpoints(self, edge_idx):
    p = self.pos(edge_idx)
    return int(self.src[p]), int(self.dst[p])

  def event(self, edge_idx):
    p = self.pos(edge_idx)
    return dict(edge_idx=int(edge_idx), u=int(self.src[p]), v=int(self.dst[p]), t=float(self.ts[p]),
                batch=int(self.batch_of[p]), region=['train', 'val', 'test'][self.region[p]])

  # ------------------------------------------------------------------ state
  def reset_state(self):
    self.model.memory.__init_memory__()

  def snapshot(self):
    """Deep copy of (memory, last_update, messages) via Memory.backup_memory()."""
    return self.model.memory.backup_memory()

  def restore(self, snap):
    self.model.memory.restore_memory(snap)

  @staticmethod
  def pack(snap):
    """Snapshot -> dict of a few contiguous CPU tensors (fast to torch.save)."""
    memory, last_update, messages = snap
    nodes, counts, msgs, times = [], [], [], []
    for node, lst in messages.items():
      if lst:
        nodes.append(int(node))
        counts.append(len(lst))
        msgs.extend(m for m, _ in lst)
        times.extend(t.reshape(()) for _, t in lst)
    return dict(memory=memory.cpu(), last_update=last_update.cpu(),
                nodes=torch.tensor(nodes, dtype=torch.long),
                counts=torch.tensor(counts, dtype=torch.long),
                msgs=torch.stack(msgs).cpu() if msgs else torch.zeros(0),
                times=torch.stack(times).cpu() if times else torch.zeros(0))

  def unpack(self, packed):
    d = self.device
    messages = defaultdict(list)
    msgs, times = packed['msgs'].to(d), packed['times'].to(d)
    k = 0
    for node, cnt in zip(packed['nodes'].tolist(), packed['counts'].tolist()):
      messages[np.int64(node)] = [(msgs[k + j], times[k + j]) for j in range(cnt)]
      k += cnt
    return packed['memory'].to(d), packed['last_update'].to(d), messages

  # ------------------------------------------------------------------ model calls
  def _arrays(self, positions):
    positions = np.asarray(positions, dtype=np.int64)
    return self.src[positions], self.dst[positions], self.ts[positions], self.edge_idx[positions]

  @torch.no_grad()
  def advance_state(self, positions, ngh_finder):
    """Process one (possibly shrunk) batch: same code path as eval, dummy negatives.

    Negatives never touch memory or messages (only positives do), so negatives=destinations
    advances the state exactly like the real eval loop.
    """
    if len(positions) == 0:
      return
    src, dst, ts, eidx = self._arrays(positions)
    self.model.set_neighbor_finder(ngh_finder)
    if self.fast_advance:
      self._advance_state_only(src, dst, ts, eidx)
    else:
      self.model.compute_temporal_embeddings(src, dst, dst, ts, eidx, self.cfg.n_degree)

  @property
  def fast_advance(self):
    """Skip the embedding computation when advancing state.

    With memory_update_at_start and messages built from raw memory (the default flags), the only
    state change in TGN.compute_temporal_embeddings is: update memory of the positives with their
    pending messages, clear those, store this batch's raw messages. Embeddings (and the all-node
    get_updated_memory, used only for embeddings and an assert) do not touch the state.
    tests/test_stage0.py checks bitwise equality with the full code path.
    """
    m = self.model
    return (self._fast_ok and m.memory_update_at_start and not m.dyrep
            and not m.use_source_embedding_in_message and not m.use_destination_embedding_in_message)

  def _advance_state_only(self, src, dst, ts, eidx):
    m = self.model
    positives = np.concatenate([src, dst])
    m.update_memory(positives, m.memory.messages)
    m.memory.clear_messages(positives)
    unique_src, src_msgs = m.get_raw_messages(src, None, dst, None, ts, eidx)
    unique_dst, dst_msgs = m.get_raw_messages(dst, None, src, None, ts, eidx)
    m.memory.store_raw_messages(unique_src, src_msgs)
    m.memory.store_raw_messages(unique_dst, dst_msgs)

  @torch.no_grad()
  def batched_probabilities(self, positions, negatives, ngh_finder):
    """The normal batched forward pass (advances the state)."""
    src, dst, ts, eidx = self._arrays(positions)
    self.model.set_neighbor_finder(ngh_finder)
    pos, neg = self.model.compute_edge_probabilities(src, dst, negatives, ts, eidx, self.cfg.n_degree)
    return pos.reshape(-1).cpu().numpy(), neg.reshape(-1).cpu().numpy()

  def updated_memory(self):
    """Memory as the embedding sees it at the start of a batch (pending messages applied).

    Pure: does not modify the stored state (TGN.get_updated_memory works on clones).
    """
    with torch.no_grad():
      return self.model.get_updated_memory(list(range(self.n_nodes)), self.model.memory.messages)

  def score_from_memory(self, memory, last_update, src, dst, ts, ngh_finder):
    """Decoder(embedding(u), embedding(v)) given an explicit memory tensor.

    Mirrors the first half of TGN.compute_temporal_embeddings + compute_edge_probabilities, without
    persisting anything. Gradients flow to `memory` if it requires grad (used by Signal C).
    """
    m = self.model
    src, dst, ts = np.atleast_1d(src), np.atleast_1d(dst), np.atleast_1d(ts).astype(np.float64)
    n = len(src)
    nodes = np.concatenate([src, dst])
    timestamps = np.concatenate([ts, ts])
    t = torch.LongTensor(ts).to(self.device)
    src_diff = ((t - last_update[src].long()) - m.mean_time_shift_src) / m.std_time_shift_src
    dst_diff = ((t - last_update[dst].long()) - m.mean_time_shift_dst) / m.std_time_shift_dst
    time_diffs = torch.cat([src_diff, dst_diff], dim=0)
    m.set_neighbor_finder(ngh_finder)
    emb = m.embedding_module.compute_embedding(memory=memory, source_nodes=nodes,
                                               timestamps=timestamps, n_layers=m.n_layers,
                                               n_neighbors=self.cfg.n_degree, time_diffs=time_diffs)
    score = m.affinity_score(emb[:n], emb[n:]).squeeze(dim=-1).sigmoid()
    return score

  @torch.no_grad()
  def score_positions(self, positions, ngh_finder):
    """Score events with the current state, which must be the start of their (shared) batch."""
    memory, last_update = self.updated_memory()
    src, dst, ts, _ = self._arrays(positions)
    return self.score_from_memory(memory, last_update, src, dst, ts, ngh_finder).cpu().numpy()


_ENV = {}


def get_env(cfg: TraceConfig = None) -> TraceEnv:
  """Process-wide cached environment (loading data + model takes a few seconds)."""
  cfg = cfg or TraceConfig()
  key = (cfg.data, str(cfg.model_file), cfg.device, cfg.checkpoint_every, cfg.k_candidates)
  if key not in _ENV:
    _ENV[key] = TraceEnv(cfg)
  return _ENV[key]
