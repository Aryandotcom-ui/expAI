"""Stage 5 gate: API happy path, bad id, empty removal, oversized removal."""
import pytest

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope='module')
def client(replayer):
  from tgn_trace.api.main import app
  return TestClient(app)


@pytest.fixture(scope='module')
def target(client):
  ts = client.get('/api/targets').json()
  assert ts, 'no demo targets'
  return ts[0]


def test_health(client):
  r = client.get('/health')
  assert r.status_code == 200
  h = r.json()
  assert h['n_events'] > 0 and h['n_checkpoints'] > 0


def test_happy_path(client, target):
  tr = client.get(f"/api/targets/{target['id']}/trace")
  assert tr.status_code == 200
  events = tr.json()['events']
  assert len(events) > 0 and all('bruteforce_delta' in e for e in events)
  ex = client.post('/api/explain', json={'target_id': target['id'], 'method': 'recency'})
  assert ex.status_code == 200 and ex.json()['status'] == 'ok'
  removed = [e['edge_idx'] for e in events[:2]]
  cf = client.post('/api/counterfactual', json={'target_id': target['id'], 'removed_edge_idxs': removed})
  assert cf.status_code == 200
  body = cf.json()
  assert sorted(body['removed']) == sorted(removed)
  assert abs(body['delta'] - (body['cf_score'] - body['baseline_score'])) < 1e-12
  # the single-removal delta must match the brute-force delta shown in the trace
  one = client.post('/api/counterfactual', json={'target_id': target['id'], 'removed_edge_idxs': removed[:1]})
  assert abs(one.json()['delta'] - events[0]['bruteforce_delta']) < 1e-9


def test_bad_ids(client, target):
  assert client.get('/api/targets/999999999/trace').status_code == 404
  assert client.post('/api/counterfactual', json={'target_id': 999999999, 'removed_edge_idxs': []}).status_code == 404
  r = client.post('/api/counterfactual', json={'target_id': target['id'], 'removed_edge_idxs': [1]})
  assert r.status_code == 422  # edge 1 exists but is not a candidate of a test target
  assert client.post('/api/explain', json={'target_id': target['id'], 'method': 'nope'}).status_code == 422


def test_empty_removal_returns_baseline(client, target):
  r = client.post('/api/counterfactual', json={'target_id': target['id'], 'removed_edge_idxs': []})
  assert r.status_code == 200
  body = r.json()
  assert abs(body['cf_score'] - body['baseline_score']) <= 1e-5
  assert body['baseline_class'] == body['cf_class']


def test_oversized_removal(client, target):
  r = client.post('/api/counterfactual', json={'target_id': target['id'], 'removed_edge_idxs': list(range(1, 12))})
  assert r.status_code == 422
  assert 'at most' in r.json()['detail']
