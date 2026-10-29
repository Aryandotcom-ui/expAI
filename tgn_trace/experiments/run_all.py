"""Stage 7: experiments E1-E5 and figures. Fixed seeds; saves config + git hash with the results.

Prerequisites (each writes the inputs this script reads):
  python -m tgn_trace.baseline --checkpoints          E1 (baseline_metrics.json)
  python -m tgn_trace.experiments.bench_replay        E3/E4 (replay_bench.csv)
  python -m tgn_trace.verify.run                      E5 (verification*.csv)
This script runs E2 itself (explainers vs brute force) and then aggregates everything.

Usage: python -m tgn_trace.experiments.run_all [--n_targets 100] [--no_cody]
Outputs: results/<data>/e2_explainers.csv, results/<data>/summary.json, figures/<data>/*.png
"""
import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from tgn_trace.config import ROOT, TraceConfig
from tgn_trace.explainers import bruteforce, greedy_cf, recency
from tgn_trace.replay import Replayer
from tgn_trace.state import get_env
from tgn_trace.verify.run import pick_targets

KS = (1, 3, 5)
# Reference categorical palette (dataviz skill, light mode, first three slots)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'
METHOD_COLORS = {'recency': BLUE, 'greedy': ORANGE, 'cody': AQUA}


def git_hash():
  try:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
  except Exception:
    return None


def topk_metrics(ranked, bf_ranked, relevant, k):
  top = ranked[:k]
  hit_bf = len(set(top) & set(bf_ranked[:k]))
  hit_rel = len(set(top) & relevant)
  return dict(**{f'p@{k}_vs_bf': hit_bf / k,
                 f'p@{k}_vs_rel': hit_rel / k,
                 f'r@{k}_vs_rel': hit_rel / len(relevant) if relevant else np.nan})


def run_e2(env, replayer, targets, tau, methods):
  rows = []
  t0 = time.perf_counter()
  for i, t in enumerate(targets):
    bf = bruteforce.explain(env, t, replayer)
    relevant = {e for e, d in bf['deltas'].items() if d <= -tau}  # removal lowers score by >= tau
    for name, fn in methods.items():
      ex = fn(env, t, replayer) if name != 'recency' else fn(env, t)
      # For counterfactual methods, precision@k is computed on the explanation set they return
      # (padded with their ranking of the rest), i.e. on `ranked`.
      row = dict(target=t, method=name, status=ex['status'], n_candidates=len(bf['candidates']),
                 n_relevant=len(relevant), runtime_ms=ex['runtime_ms'], n_model_evals=ex['n_model_evals'],
                 cf_success=ex.get('success'), cf_size=len(ex['cf_set']) if 'cf_set' in ex else None,
                 baseline_score=bf['baseline_score'])
      for k in KS:
        row.update(topk_metrics(ex['ranked'], bf['ranked'], relevant, k))
      rows.append(row)
    print(f'[{i + 1}/{len(targets)}] {t}  {time.perf_counter() - t0:.0f}s', flush=True)
  return pd.DataFrame(rows)


def e5_summary(cfg):
  ver_path = cfg.results_dir / 'verification.csv'
  if not ver_path.exists():
    return None
  ver = pd.read_csv(ver_path)
  ev = pd.read_csv(cfg.results_dir / 'verification_events.csv')
  tau = float(ver.tau.iloc[0])
  out = dict(n_targets=int(ver.target_id.nunique()), tau=tau, per_explainer={})
  for name, g in ver.groupby('explainer'):
    out['per_explainer'][name] = dict(
      flags=g.flag.value_counts().to_dict(), agreement_median=float(g.agreement.median()),
      agreement_mean=float(g.agreement.mean()), null_percentile_median=float(g.null_percentile.median()),
      frac_null_percentile_ge_95=float((g.null_percentile >= 95).mean()),
      n_nan_agreement=int(g.agreement.isna().sum()))
  if 'signal_c_linear_estimate' in ev:
    b, c = ev.signal_b_delta, ev.signal_c_linear_estimate
    strong_b, strong_c = b.abs() >= tau, c.abs() >= tau
    out['signal_c'] = dict(
      n_events=int(len(ev)), median_abs_b_minus_c=float((b - c).abs().median()),
      p90_abs_b_minus_c=float((b - c).abs().quantile(0.9)),
      spearman_b_c=float(pd.Series(b).corr(pd.Series(c), method='spearman')),
      frac_sign_disagree_when_b_strong=float((np.sign(b[strong_b]) != np.sign(c[strong_b])).mean()) if strong_b.any() else None,
      frac_strong_label_changes_if_c_replaced_b=float((strong_b != strong_c).mean()),
      structural_share_median=float((ev.structural_term.abs() / (ev.structural_term.abs()
                                     + ev.memory_exact_term.abs()).replace(0, np.nan)).median()),
      max_c_vs_b_replay_check=float(ev.c_vs_b_replay_check.max()))
    # Ablation: flags computed with C in place of B (what you would conclude without replay)
    changed = 0
    total = 0
    from tgn_trace.verify.flags import flag
    for (tid, name), g in ver.groupby(['target_id', 'explainer']):
      e = ev[ev.target_id == tid]
      ranked = [int(x) for x in g.top3.iloc[0].split()]
      dc = dict(zip(e.edge_idx.astype(int), e.signal_c_linear_estimate))
      ranked_full = ranked + [x for x in dc if x not in ranked]
      f_c, _ = flag(ranked_full, dc, tau)
      changed += f_c != g.flag.iloc[0]
      total += 1
    out['ablation_flags_with_c_instead_of_b'] = dict(changed=int(changed), total=int(total))
  return out


def figures(cfg, e2):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                       'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                       'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                       'grid.linewidth': 0.8, 'axes.axisbelow': True})
  out = cfg.figures_dir
  out.mkdir(parents=True, exist_ok=True)
  written = []
  bench_path = cfg.results_dir / 'replay_bench.csv'
  if bench_path.exists():
    b = pd.read_csv(bench_path)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(b.distance_batches, b.full_ms, s=28, color=ORANGE, label='full replay', edgecolor='white', linewidth=1)
    ax.scatter(b.distance_batches, b.checkpointed_ms, s=28, color=BLUE, label='checkpointed replay',
               edgecolor='white', linewidth=1)
    ax.set_yscale('log')
    ax.set_xlabel('removal distance (batches before the target batch)')
    ax.set_ylabel('latency (ms, log scale)')
    ax.set_title(f'Replay latency vs removal distance ({cfg.data}, C={cfg.checkpoint_every})', loc='left', fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'latency_vs_distance.png', dpi=150)
    plt.close(fig)
    written.append('latency_vs_distance.png')
  if e2 is not None and len(e2):
    methods = [m for m in ('recency', 'greedy', 'cody') if m in set(e2.method)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
      g = e2[e2.method == m]
      vals = [g[f'p@{k}_vs_bf'].mean() for k in KS]
      xs = np.arange(len(KS)) + (j - (len(methods) - 1) / 2) * w
      ax.bar(xs, vals, width=w - 0.03, color=METHOD_COLORS[m], label=m)
      for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(KS)), [f'k={k}' for k in KS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('precision@k vs brute-force top-k')
    ax.set_title(f'Explainer precision@k (n={e2.target.nunique()} test targets, {cfg.data})', loc='left', fontsize=11)
    ax.legend(frameon=False, ncol=len(methods))
    fig.tight_layout()
    fig.savefig(out / 'precision_at_k.png', dpi=150)
    plt.close(fig)
    written.append('precision_at_k.png')
  ver_path = cfg.results_dir / 'verification.csv'
  null_path = cfg.results_dir / 'verification_null.npz'
  if ver_path.exists() and null_path.exists():
    ver = pd.read_csv(ver_path)
    nulls = np.load(null_path)
    names = [n for n in ('recency', 'greedy', 'cody') if n in set(ver.explainer)]
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    bins = np.linspace(-1, 1, 41)
    for ax, n in zip(axes, names):
      obs = ver[ver.explainer == n].agreement.dropna()
      ax.hist(nulls[n], bins=bins, density=True, color=GRID, edgecolor=MUTED, linewidth=0.5,
              label='null (random rankings)')
      ax.hist(obs, bins=bins, density=True, color=METHOD_COLORS[n], alpha=0.85, label=f'{n} (observed)')
      ax.set_xlabel('Spearman(importance, −Δ)')
      ax.set_title(f'{n}: median ρ={obs.median():.2f}', loc='left', fontsize=10)
      ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('density')
    fig.suptitle(f'Agreement with replay vs permutation null ({cfg.data})', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'agreement_vs_null.png', dpi=150)
    plt.close(fig)
    written.append('agreement_vs_null.png')
  return written


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n_targets', type=int, default=100)
  ap.add_argument('--no_cody', action='store_true')
  ap.add_argument('--skip_e2', action='store_true', help='reuse results/<data>/e2_explainers.csv')
  args = ap.parse_args()
  cfg = TraceConfig()
  env = get_env(cfg)
  replayer = Replayer(env)
  res = cfg.results_dir
  tau = json.loads((res / 'threshold_selection.json').read_text())['tau']

  methods = {'recency': recency.explain, 'greedy': greedy_cf.explain}
  if not args.no_cody:
    from tgn_trace.explainers import cody_adapter
    methods['cody'] = cody_adapter.explain
  if args.skip_e2 and (res / 'e2_explainers.csv').exists():
    e2 = pd.read_csv(res / 'e2_explainers.csv')
  else:
    targets = pick_targets(replayer, 'test', args.n_targets, seed=1)
    e2 = run_e2(env, replayer, targets, tau, methods)
    e2.to_csv(res / 'e2_explainers.csv', index=False)

  summary = dict(config=cfg.to_dict(), git_hash=git_hash(), data=cfg.data)
  summary['E1_baseline'] = json.loads((res / 'baseline_metrics.json').read_text())
  agg = {}
  for m, g in e2.groupby('method'):
    a = {f'{c}': float(g[c].mean()) for c in g.columns if c.startswith(('p@', 'r@'))}
    a.update(n_targets=int(g.target.nunique()), runtime_ms_median=float(g.runtime_ms.median()),
             model_evals_mean=float(g.n_model_evals.mean()), status=g.status.value_counts().to_dict())
    if g.cf_success.notna().any():
      succ = g[g.cf_success == True]  # noqa: E712
      a.update(cf_success_rate=float((g.cf_success == True).mean()),  # noqa: E712
               cf_mean_size_successful=float(succ.cf_size.mean()) if len(succ) else None)
    agg[m] = a
  summary['E2_explainers'] = dict(tau_for_relevant_set=tau, n_relevant_mean=float(e2.groupby('target').n_relevant.first().mean()),
                                  per_method=agg)
  bench = res / 'replay_bench_summary.json'
  if bench.exists():
    b = json.loads(bench.read_text())
    summary['E3_replay_correctness'] = dict(max_abs_err=b['max_abs_err'], mean_abs_err=b['mean_abs_err'], n_pairs=b['n_pairs'])
    summary['E4_replay_latency'] = {k: b[k] for k in ('full_ms', 'checkpointed_ms', 'speedup', 'distance_batches')}
  summary['E5_verification'] = e5_summary(cfg)
  summary['figures'] = figures(cfg, e2)
  (res / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
  print(json.dumps({k: v for k, v in summary.items() if k != 'config'}, indent=2, default=str))


if __name__ == '__main__':
  main()
