from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Health(BaseModel):
  status: str
  data: str
  model_file: str
  device: str
  n_events: int
  n_batches: int
  first_test_batch: int
  checkpoint_every: int
  n_checkpoints: int
  checkpoint_disk_mb: float
  baseline_metrics: Optional[dict] = None
  tau: Optional[float] = None


class Target(BaseModel):
  id: int
  u: int
  v: int
  t: float
  baseline_score: float
  has_verification: bool = False


class TraceEvent(BaseModel):
  edge_idx: int
  u: int
  v: int
  t: float
  batch: int
  batch_distance: int
  touches: str
  ranks: Dict[str, int]
  bruteforce_delta: Optional[float] = None


class Trace(BaseModel):
  target: Target
  threshold: float
  baseline_class: int
  k: int
  events: List[TraceEvent]


class ExplainRequest(BaseModel):
  target_id: int
  method: Literal['recency', 'bruteforce', 'greedy', 'cody']


class ExplainResponse(BaseModel):
  target_id: int
  method: str
  status: str
  ranked: List[int]
  importance: Dict[int, float]
  runtime_ms: float
  n_model_evals: int
  extra: dict = Field(default_factory=dict)


class CounterfactualRequest(BaseModel):
  target_id: int
  removed_edge_idxs: List[int]


class CounterfactualResponse(BaseModel):
  target_id: int
  baseline_score: float
  removed: List[int]
  cf_score: float
  delta: float
  baseline_class: int
  cf_class: int
  replayed_events: int
  latency_ms: float
  checkpoint_used: Optional[int]
  cached: bool = False
