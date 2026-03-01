"""
Custom HNSW (Hierarchical Navigable Small World) Graph Index
=============================================================
A from-scratch implementation of the HNSW algorithm for approximate
nearest neighbor search in high-dimensional vector spaces.

Reference: "Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)

Key design decisions:
  - Layered graph where higher layers act as express lanes for search
  - Each node exists on layer 0 and probabilistically on higher layers
  - Search starts at the top layer's entry point and greedily descends
  - At layer 0, ef_search candidates are explored for recall quality
"""

import math
import random
import heapq
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HNSWNode:
    """Fields for a single node in HNSW graph."""
    id: str
    vector: np.ndarray
    layer: int  # max layer this node exists on
    neighbors: dict = field(default_factory=dict)  # layer -> list of neighbor ids

    def __post_init__(self):
        for lvl in range(self.layer + 1):
            if lvl not in self.neighbors:
                self.neighbors[lvl] = []


class HNSWIndex:
    """
    Hierarchical Navigable Small World graph for approximate nearest neighbor search.

    Parameters
    ----------
    dim : int
        Dimensionality of vectors.
    M : int
        Max number of connections per node per layer (default 16).
        Higher M = better recall but more memory & slower insertion.
    ef_construction : int
        Size of the dynamic candidate list during index building (default 200).
        Higher = better index quality but slower build.
    ef_search : int
        Size of the dynamic candidate list during search (default 50).
        Higher = better recall but slower queries.
    ml : float
        Level generation factor. Controls the probability distribution
        of how many layers a node appears on. Default = 1/ln(M).
    metric : str
        Distance metric: 'cosine' or 'euclidean'.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: Optional[float] = None,
        metric: str = "cosine",
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M  # layer 0 gets double the connections
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml or (1.0 / math.log(M)) if M > 1 else 1.0
        self.metric = metric

        self.nodes: dict[str, HNSWNode] = {}
        self.entry_point: Optional[str] = None
        self.max_layer: int = -1

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))
        else:  # euclidean
            return float(np.linalg.norm(a - b))

    # ------------------------------------------------------------------
    # Layer assignment
    # ------------------------------------------------------------------

    def _random_layer(self) -> int:
        """
        Assign a random layer using an exponentially decaying distribution.
        Most nodes land on layer 0; few reach higher layers.
        This creates the hierarchical "express lane" structure.
        """
        return int(-math.log(random.random()) * self.ml)

    # ------------------------------------------------------------------
    # Core graph search (used by both insert and query)
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry_ids: list[str],
        ef: int,
        layer: int,
    ) -> list[tuple[float, str]]:
        """
        Greedy beam search on a single layer of the HNSW graph.

        Uses a min-heap (candidates) and max-heap (results) pattern:
          - candidates: unexplored nodes closest to query (pop nearest first)
          - results: best ef nodes found so far (pop farthest first for pruning)

        Returns list of (distance, node_id) sorted by distance ascending.
        """
        visited = set(entry_ids)

        # candidates: min-heap of (distance, id) — closest first
        candidates = []
        # results: max-heap of (-distance, id) — farthest first (for pruning)
        results = []

        for eid in entry_ids:
            dist = self._distance(query, self.nodes[eid].vector)
            heapq.heappush(candidates, (dist, eid))
            heapq.heappush(results, (-dist, eid))

        while candidates:
            # Get nearest unprocessed candidate
            c_dist, c_id = heapq.heappop(candidates)

            # Get the farthest node in our results
            f_dist = -results[0][0]

            # If nearest candidate is farther than our worst result, stop
            if c_dist > f_dist:
                break

            # Explore neighbors of this candidate on the given layer
            node = self.nodes[c_id]
            neighbor_ids = node.neighbors.get(layer, [])

            for n_id in neighbor_ids:
                if n_id in visited:
                    continue
                visited.add(n_id)

                n_dist = self._distance(query, self.nodes[n_id].vector)
                f_dist = -results[0][0]

                # Add if better than worst result or results not full
                if n_dist < f_dist or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(results, (-n_dist, n_id))

                    # Prune results to keep only ef best
                    if len(results) > ef:
                        heapq.heappop(results)

        # Convert max-heap to sorted list (ascending distance)
        out = [(-d, nid) for d, nid in results]
        out.sort(key=lambda x: x[0])
        return out

    # ------------------------------------------------------------------
    # Neighbor selection with heuristic pruning
    # ------------------------------------------------------------------

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: list[tuple[float, str]],
        M: int,
        layer: int,
    ) -> list[str]:
        """
        Select up to M neighbors using the heuristic from the HNSW paper.

        The heuristic prefers neighbors that are not only close to the query
        but also provide diverse directions (not redundant with already-selected
        neighbors). This improves graph navigability.
        """
        if len(candidates) <= M:
            return [nid for _, nid in candidates]

        # Sort by distance ascending
        working = sorted(candidates, key=lambda x: x[0])
        selected = []

        for dist, nid in working:
            if len(selected) >= M:
                break

            # Heuristic: only add if this node is closer to query
            # than to any already-selected neighbor. This ensures diversity.
            is_good = True
            for sel_id in selected:
                d_between = self._distance(
                    self.nodes[nid].vector, self.nodes[sel_id].vector
                )
                if d_between < dist:
                    is_good = False
                    break

            if is_good:
                selected.append(nid)

        # If heuristic was too aggressive, fill remaining slots naively
        if len(selected) < M:
            selected_set = set(selected)
            for _, nid in working:
                if nid not in selected_set:
                    selected.append(nid)
                    selected_set.add(nid)
                    if len(selected) >= M:
                        break

        return selected

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, id: str, vector: np.ndarray):
        """
        Insert a new vector into the HNSW index.

        Algorithm:
        1. Assign a random layer level to the new node.
        2. Starting from the entry point, greedily descend through layers
           above the new node's level (finding the closest single node).
        3. At the node's level and below, do a beam search with ef_construction
           to find good neighbors, then bidirectionally connect them.
        4. If the new node's level exceeds the current max, update entry point.
        """
        vector = np.asarray(vector, dtype=np.float32)
        assert vector.shape == (self.dim,), f"Expected dim {self.dim}, got {vector.shape}"

        new_layer = self._random_layer()
        new_node = HNSWNode(id=id, vector=vector, layer=new_layer)

        with self._lock:
            self.nodes[id] = new_node

            # First node — just set as entry point
            if self.entry_point is None:
                self.entry_point = id
                self.max_layer = new_layer
                return

            # Phase 1: Greedy descent from top layer to new_layer + 1
            # At these higher layers, we just find the single closest node
            curr_entry = self.entry_point
            curr_dist = self._distance(vector, self.nodes[curr_entry].vector)

            for layer in range(self.max_layer, new_layer, -1):
                changed = True
                while changed:
                    changed = False
                    for n_id in self.nodes[curr_entry].neighbors.get(layer, []):
                        d = self._distance(vector, self.nodes[n_id].vector)
                        if d < curr_dist:
                            curr_dist = d
                            curr_entry = n_id
                            changed = True

            # Phase 2: Insert at layers [min(new_layer, max_layer) ... 0]
            entry_points = [curr_entry]

            for layer in range(min(new_layer, self.max_layer), -1, -1):
                # Find ef_construction nearest neighbors at this layer
                candidates = self._search_layer(
                    vector, entry_points, self.ef_construction, layer
                )

                # Select best M neighbors (M_max0 for layer 0)
                M_layer = self.M_max0 if layer == 0 else self.M
                neighbor_ids = self._select_neighbors(
                    vector, candidates, M_layer, layer
                )

                # Bidirectional connections
                new_node.neighbors[layer] = neighbor_ids

                for n_id in neighbor_ids:
                    neighbor_node = self.nodes[n_id]
                    if layer not in neighbor_node.neighbors:
                        neighbor_node.neighbors[layer] = []

                    neighbor_node.neighbors[layer].append(id)

                    # Prune neighbor's connections if over capacity
                    M_max = self.M_max0 if layer == 0 else self.M
                    if len(neighbor_node.neighbors[layer]) > M_max:
                        # Re-select best neighbors for the overloaded node
                        n_candidates = [
                            (
                                self._distance(
                                    neighbor_node.vector,
                                    self.nodes[cid].vector,
                                ),
                                cid,
                            )
                            for cid in neighbor_node.neighbors[layer]
                        ]
                        neighbor_node.neighbors[layer] = self._select_neighbors(
                            neighbor_node.vector, n_candidates, M_max, layer
                        )

                # Use closest found nodes as entry points for the next layer down
                entry_points = [nid for _, nid in candidates[:1]] or entry_points

            # Update entry point if new node reaches a higher layer
            if new_layer > self.max_layer:
                self.entry_point = id
                self.max_layer = new_layer

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        vector: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Find the k approximate nearest neighbors of the query vector.

        Returns list of (node_id, distance) sorted by distance ascending.
        """
        if not self.nodes:
            return []

        vector = np.asarray(vector, dtype=np.float32)
        ef = ef or max(self.ef_search, k)

        # Phase 1: Greedy descent to layer 0
        curr_entry = self.entry_point
        curr_dist = self._distance(vector, self.nodes[curr_entry].vector)

        for layer in range(self.max_layer, 0, -1):
            changed = True
            while changed:
                changed = False
                for n_id in self.nodes[curr_entry].neighbors.get(layer, []):
                    d = self._distance(vector, self.nodes[n_id].vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_entry = n_id
                        changed = True

        # Phase 2: Search layer 0 with ef candidates
        results = self._search_layer(vector, [curr_entry], ef, layer=0)

        # Return top k
        return [(nid, dist) for dist, nid in results[:k]]

    # ------------------------------------------------------------------
    # Bulk operations & utilities
    # ------------------------------------------------------------------

    def bulk_insert(self, items: list[tuple[str, np.ndarray]]):
        """Insert multiple vectors. Items: list of (id, vector)."""
        for id, vector in items:
            self.insert(id, vector)

    def delete(self, id: str):
        """
        Soft-delete a node by removing all its connections.
        The node stays in memory but becomes unreachable.
        """
        with self._lock:
            if id not in self.nodes:
                return

            node = self.nodes[id]

            # Remove from all neighbors' adjacency lists
            for layer, neighbor_ids in node.neighbors.items():
                for n_id in neighbor_ids:
                    if n_id in self.nodes:
                        n_node = self.nodes[n_id]
                        if layer in n_node.neighbors:
                            n_node.neighbors[layer] = [
                                x for x in n_node.neighbors[layer] if x != id
                            ]

            del self.nodes[id]

            # If we deleted the entry point, pick a new one
            if self.entry_point == id:
                if self.nodes:
                    self.entry_point = next(iter(self.nodes))
                    self.max_layer = self.nodes[self.entry_point].layer
                else:
                    self.entry_point = None
                    self.max_layer = -1

    def __len__(self):
        return len(self.nodes)

    def stats(self) -> dict:
        """Return diagnostic statistics about the index."""
        if not self.nodes:
            return {"num_nodes": 0, "max_layer": -1, "entry_point": None}

        layer_counts = {}
        total_edges = 0
        for node in self.nodes.values():
            for layer, neighbors in node.neighbors.items():
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
                total_edges += len(neighbors)

        return {
            "num_nodes": len(self.nodes),
            "max_layer": self.max_layer,
            "entry_point": self.entry_point,
            "layer_distribution": dict(sorted(layer_counts.items())),
            "total_edges": total_edges,
            "avg_edges_per_node": total_edges / len(self.nodes) if self.nodes else 0,
        }
