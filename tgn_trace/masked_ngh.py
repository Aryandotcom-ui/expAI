"""A NeighborFinder that behaves as if a set of events had never happened."""
import numpy as np

from utils.utils import NeighborFinder


class MaskedNeighborFinder(NeighborFinder):
  """Wraps a built NeighborFinder and hides the events in `excluded_edge_idxs`.

  Only the endpoints of excluded events have different adjacency lists, so those are filtered once
  at construction and every other node reads the base finder's arrays directly. Sampling logic
  (`get_temporal_neighbor`) is inherited unchanged, so with an empty mask the output is identical
  to the base finder's.
  """

  def __init__(self, base: NeighborFinder, excluded_edge_idxs, edge_endpoints):
    # Deliberately not calling NeighborFinder.__init__: we share the base finder's arrays.
    self.base = base
    self.uniform = base.uniform
    self.excluded = frozenset(int(e) for e in excluded_edge_idxs)
    self.node_to_neighbors = base.node_to_neighbors
    self.node_to_edge_idxs = base.node_to_edge_idxs
    self.node_to_edge_timestamps = base.node_to_edge_timestamps

    self._filtered = {}
    if self.excluded:
      excluded_arr = np.fromiter(self.excluded, dtype=np.int64)
      touched = set()
      for e in self.excluded:
        touched.update(edge_endpoints(e))
      for node in touched:
        keep = ~np.isin(base.node_to_edge_idxs[node], excluded_arr)
        self._filtered[node] = (base.node_to_neighbors[node][keep],
                                base.node_to_edge_idxs[node][keep],
                                base.node_to_edge_timestamps[node][keep])

  def find_before(self, src_idx, cut_time):
    arrays = self._filtered.get(int(src_idx))
    if arrays is None:
      return self.base.find_before(src_idx, cut_time)
    neighbors, edge_idxs, timestamps = arrays
    i = np.searchsorted(timestamps, cut_time)
    return neighbors[:i], edge_idxs[:i], timestamps[:i]
