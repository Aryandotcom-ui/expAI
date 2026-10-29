"""Central configuration for TGN-Trace.

All paths are pathlib-based so they work on Windows and Linux. Values can be overridden with
environment variables (TGN_TRACE_DATA, TGN_TRACE_MODEL, TGN_TRACE_DEVICE, ...) so the same code
runs on the real Wikipedia checkpoint and on the synthetic stand-in.
"""
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default, cast=str):
  value = os.environ.get(name)
  return cast(value) if value is not None else default


@dataclass
class TraceConfig:
  # Data / model
  data: str = field(default_factory=lambda: _env('TGN_TRACE_DATA', 'wikipedia'))
  model_prefix: str = field(default_factory=lambda: _env('TGN_TRACE_PREFIX', 'tgn-attn'))
  model_path: str = field(default_factory=lambda: _env('TGN_TRACE_MODEL', ''))
  device: str = field(default_factory=lambda: _env('TGN_TRACE_DEVICE', 'auto'))

  # TGN hyper-parameters. Must match the ones used for training (defaults of
  # train_self_supervised.py with --use_memory, i.e. the repo's Wikipedia TGN-attn command).
  n_layer: int = 1
  n_degree: int = 10
  n_head: int = 2
  drop_out: float = 0.1
  memory_dim: int = 172
  message_dim: int = 100
  message_function: str = 'identity'
  memory_updater: str = 'gru'
  aggregator: str = 'last'
  embedding_module: str = 'graph_attention'

  # Replay
  batch_size: int = 200
  checkpoint_every: int = field(default_factory=lambda: _env('TGN_TRACE_CKPT_EVERY', 10, int))
  checkpoint_lead_batches: int = 50  # batches before the first test batch that get checkpoints
  checkpoint_cache_size: int = 4      # checkpoints kept in RAM (each is tens of MB)

  # Explanation / verification
  k_candidates: int = field(default_factory=lambda: _env('TGN_TRACE_K', 20, int))
  threshold: float = 0.5
  greedy_max_size: int = 5
  max_removed: int = 10
  seed: int = 0

  @property
  def root(self) -> Path:
    return ROOT

  @property
  def data_dir(self) -> Path:
    return ROOT / 'data'

  @property
  def model_file(self) -> Path:
    if self.model_path:
      return Path(self.model_path)
    return ROOT / 'saved_models' / f'{self.model_prefix}-{self.data}.pth'

  @property
  def results_dir(self) -> Path:
    return ROOT / 'results' / self.data

  @property
  def figures_dir(self) -> Path:
    return ROOT / 'figures' / self.data

  @property
  def checkpoint_dir(self) -> Path:
    return ROOT / 'checkpoints_state' / f'{self.data}-C{self.checkpoint_every}'

  def torch_device(self) -> torch.device:
    if self.device == 'auto':
      return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(self.device)

  def to_dict(self):
    d = asdict(self)
    d['model_file'] = str(self.model_file)
    return d


def set_determinism(seed: int = 0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.backends.cudnn.benchmark = False
