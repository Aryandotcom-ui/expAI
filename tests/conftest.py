import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tgn_trace.config import TraceConfig  # noqa: E402


@pytest.fixture(scope='session')
def env():
  from tgn_trace.state import get_env
  cfg = TraceConfig()
  if not cfg.model_file.exists():
    pytest.skip(f'no trained model at {cfg.model_file}')
  return get_env(cfg)


@pytest.fixture(scope='session')
def replayer(env):
  from tgn_trace.replay import Replayer
  if not (env.cfg.results_dir / 'baseline_scores.csv').exists():
    pytest.skip('run `python -m tgn_trace.baseline --checkpoints` first')
  return Replayer(env)


@pytest.fixture(scope='session')
def baseline_df(env):
  import pandas as pd
  path = env.cfg.results_dir / 'baseline_scores.csv'
  if not path.exists():
    pytest.skip('run `python -m tgn_trace.baseline --checkpoints` first')
  return pd.read_csv(path)
