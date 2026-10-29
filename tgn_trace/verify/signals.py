"""Verification signals for one target.

Signal A: the explainer's rank and raw importance for each candidate.
Signal B: the actual replay delta when that single event is removed (brute-force cache).
Signal C: "linear estimate" of B. It is a first-order approximation of B around the baseline
          state, NOT independent evidence. With f_N(m) = score under neighborhood N and memory m:

            B = f_cf(m_cf) - f_base(m_base)
              = [f_cf(m_base) - f_base(m_base)]        structural term (exact, masked neighbors)
              + [f_cf(m_cf)  - f_cf(m_base)]           memory term
            C = structural term + grad_m f_cf(m_base) . (m_cf - m_base)

          m is the memory the embedding reads at the start of the target's batch (pending
          messages applied), treated as a leaf; the gradient goes through the embedding module
          and decoder only. Only u, v and their sampled neighbors get a non-zero gradient.
          |B - C| is reported as "model nonlinearity around this state".
          `memory_linear_term` alone (grad . dm) is the plan's literal definition; it ignores the
          change of neighborhood, so we report it separately.
"""
import numpy as np
import torch

from tgn_trace.replay import Replayer


class LinearEstimator:
  def __init__(self, replayer: Replayer, target_edge_idx):
    self.r = replayer
    self.env = replayer.env
    self.target = int(target_edge_idx)
    p = self.env.pos(target_edge_idx)
    self.u, self.v, self.t = self.env.src[p], self.env.dst[p], self.env.ts[p]
    self.r.counterfactual_state(self.target, [])
    self.m_base, self.lu_base = self.env.updated_memory()
    with torch.no_grad():
      self.f_base = float(self._score(self.m_base, self.lu_base, self.env.full_ngh_finder))

  def _score(self, memory, last_update, ngh):
    return self.env.score_from_memory(memory, last_update, [self.u], [self.v], [self.t], ngh)[0]

  def estimate(self, removed):
    ngh = self.r.counterfactual_state(self.target, removed)
    m_cf, lu_cf = self.env.updated_memory()
    with torch.no_grad():
      f_cf = float(self._score(m_cf, lu_cf, ngh))
    m = self.m_base.clone().requires_grad_(True)
    with torch.enable_grad():
      s = self._score(m, self.lu_base, ngh)
      grad, = torch.autograd.grad(s, m)
    structural = float(s.detach()) - self.f_base
    dm = m_cf - self.m_base
    memory_linear = float((grad * dm).sum())
    exact = f_cf - self.f_base
    linear = structural + memory_linear
    return dict(signal_b_replay=exact, signal_c_linear_estimate=linear,
                structural_term=structural, memory_linear_term=memory_linear,
                memory_exact_term=exact - structural, nonlinearity=abs(exact - linear),
                n_nodes_memory_changed=int((dm.abs().sum(dim=1) > 0).sum()),
                grad_norm=float(grad.norm()))
