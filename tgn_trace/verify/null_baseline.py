"""Agreement between an explainer and replay, and a permutation null for it."""
import numpy as np
from scipy.stats import spearmanr


def agreement(importance, neg_delta):
  """Spearman correlation between explainer importance and -delta over the candidates."""
  importance, neg_delta = np.asarray(importance, float), np.asarray(neg_delta, float)
  if len(importance) < 3 or np.all(importance == importance[0]) or np.all(neg_delta == neg_delta[0]):
    return float('nan')
  return float(spearmanr(importance, neg_delta).correlation)


def null_percentile(importance, neg_delta, n_perm=1000, seed=0):
  """Percentile of the observed agreement among random permutations of the explainer ranking.

  Returns (observed, percentile in [0, 100], null samples). Ties count half.
  """
  obs = agreement(importance, neg_delta)
  if np.isnan(obs):
    return obs, float('nan'), np.array([])
  rng = np.random.RandomState(seed)
  importance = np.asarray(importance, float)
  null = np.array([agreement(rng.permutation(importance), neg_delta) for _ in range(n_perm)])
  pct = 100.0 * (np.mean(null < obs) + 0.5 * np.mean(null == obs))
  return obs, float(pct), null
