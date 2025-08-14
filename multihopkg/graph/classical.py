from collections import deque
from typing import List, Set, Tuple, Union

import numpy as np
import torch

from multihopkg.utils.data_structures import Triplet_Int, Triplet_Str

MASK_PADDING_VALUE=-1

def generate_csr_representation(
    _triplets: List[Triplet_Int],
    num_nodes: int,
) -> Tuple[List, List, List]:
    assert len(_triplets) > 1, "Given triplets list is empty"

    count_head = [ 0 ] * (num_nodes + 1)
    for (head, _, _) in _triplets:
        count_head[head+1] += 1

    # We calculate positioning first
    indptr = torch.cumsum(torch.tensor(count_head), dim=0).tolist()
    # This indptr will have a bunch of 0s at the beginning of the first entity does not have anything

    # Then we just lay them out in here.
    sorted_triplets = sorted(_triplets, key=lambda x: x[0])
    tail_idxs = []
    rel_idxs = []
    for (head, relation, tail) in sorted_triplets:
        tail_idxs.append(tail)
        rel_idxs.append(relation)

    return indptr, tail_idxs, rel_idxs


def find_heads_with_connections(triplets: List[Triplet_Int], tail_csr_idxs) -> Set[int]:
    """
    Given a list of triplets, return the set of heads that have at least one degree.
    """
    degree_heads = set()
    for (head, _, _) in triplets:
        start_idx = tail_csr_idxs[head]
        end_idx = tail_csr_idxs[head + 1]
        if start_idx != end_idx:
            degree_heads.add(head)
    return degree_heads


def sample_paths_given_csr(
    indptr: torch.Tensor,  # [N+1] long
    tail_indices: torch.Tensor,  # [T]   long (neighbor node ids)
    rel_ids: torch.Tensor,  # [T]   long (edge relation ids)
    h_batch: torch.Tensor,  # [B]   long (heads)
    L: int,  # max hops
    beams: int,  # beams per example
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters:
        indptr (torch.Tensor): (idx, idx+1) points to the start and end of each row in the adjacency matrix
        indices: Where neighbors are clustered together in a single list
    Returns:
      paths_nodes: [B, max_paths, L+1] long
      paths_rels:  [B, max_paths, L]   long
    """

    # Ensure that the given starting points have neighborhood. We want atleast one hop.
    assert len(h_batch.shape) == 1
    degrees = indptr[h_batch + 1] - indptr[h_batch]
    assert (degrees > 1).all()

    device = indptr.device
    B = h_batch.numel()

    # Initialize beams: each example has 'beams' copies at step 0
    # curr_nodes_idx = h_batch.unsqueeze(1).expand(B, beams).reshape(-1)  # [B*beams]
    paths_nodes = torch.full((B * beams, L + 1), MASK_PADDING_VALUE, dtype=torch.long, device=device)
    paths_rels = torch.full((B * beams, L), MASK_PADDING_VALUE, dtype=torch.long, device=device)
    paths_nodes[:, 0] = h_batch.unsqueeze(1).expand(B, beams).reshape(-1)

    active_idxs = torch.full_like(paths_nodes[:,0], True,dtype=torch.bool)
    for step in range(L):
        curr_nodes = paths_nodes[active_idxs, step]
        deg = indptr[curr_nodes + 1] - indptr[curr_nodes]
        has_next_step = deg > 0

        # Guard: for zero-degree, create dummy samples that we will mask out
        pos = torch.zeros((curr_nodes.numel()), dtype=torch.long, device=device)
        if has_next_step.any():
            deg_valid = deg[has_next_step]
            # sample positions then modulo by deg to avoid per-row loops
            pos_valid = torch.rand(deg_valid.numel(), device=device)
            # Then we just multiply these by the max deg 
            pos_valid = torch.floor(pos_valid * deg_valid).to(torch.long)
            pos[has_next_step] = pos_valid
            pos[~has_next_step] = MASK_PADDING_VALUE

        start = indptr[curr_nodes]
        edge_idx = start + pos  # [B*beams, m]
        tail_samples = torch.where(pos == MASK_PADDING_VALUE, MASK_PADDING_VALUE, tail_indices[edge_idx]) 
        rel_samples = torch.where(pos == MASK_PADDING_VALUE, MASK_PADDING_VALUE, rel_ids[edge_idx])

        paths_nodes[active_idxs, step + 1] = tail_samples
        paths_rels[active_idxs, step] = rel_samples

        active_idxs = paths_nodes[:, step + 1] != MASK_PADDING_VALUE

    return paths_nodes, paths_rels 


def random_walk(
    adjacency_matrix: np.ndarray,
    start_node: int,
    walk_length: int,
    weighted: bool = True,
) -> List[int]:
    """
    Perform random walk on graph starting from given node.

    Args:
        adjacency_matrix: 2D numpy array representing graph adjacency
        start_node: Starting node index
        walk_length: Number of steps in the walk
        weighted: If True, use edge weights for sampling probability

    Returns:
        List of node indices representing the walk path

    Time Complexity: O(L) where L is walk_length
    """
    if start_node >= adjacency_matrix.shape[0]:
        raise ValueError("Start node index out of bounds")

    path = [start_node]
    current_node = start_node

    for _ in range(walk_length):
        # Get neighbors (non-zero connections)
        neighbors = np.nonzero(adjacency_matrix[current_node])[0]

        if len(neighbors) == 0:
            break  # Dead end, terminate walk

        # Sample next node
        if weighted and adjacency_matrix.dtype != bool:
            # Use edge weights as probabilities
            weights = adjacency_matrix[current_node, neighbors]
            probabilities = weights / weights.sum()
            next_node = np.random.choice(neighbors, p=probabilities)
        else:
            # Uniform sampling
            next_node = np.random.choice(neighbors)

        path.append(next_node)
        current_node = next_node

    return path


def are_nodes_connected(
    adjacency_matrix: np.ndarray,
    node_pairs: Union[Tuple[int, int], List[Tuple[int, int]]],
) -> Union[bool, List[bool]]:
    """
    Check connectivity between node pairs using BFS.

    Args:
        adjacency_matrix: 2D numpy array representing graph adjacency
        node_pairs: Single pair (source, target) or list of pairs

    Returns:
        Boolean or list of booleans indicating connectivity

    Time Complexity: O(V + E) per pair, where V=vertices, E=edges
    """

    def bfs_connected(source: int, target: int) -> bool:
        if source == target:
            return True

        if source >= adjacency_matrix.shape[0] or target >= adjacency_matrix.shape[0]:
            return False

        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            current = queue.popleft()

            # Get neighbors
            neighbors = np.nonzero(adjacency_matrix[current])[0]

            for neighbor in neighbors:
                if neighbor == target:
                    return True

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    # Handle single pair vs batch
    if isinstance(node_pairs, tuple):
        return bfs_connected(node_pairs[0], node_pairs[1])
    else:
        return [bfs_connected(source, target) for source, target in node_pairs]


def batch_random_walks(
    adjacency_matrix: np.ndarray,
    start_nodes: List[int],
    walk_length: int,
    num_walks_per_node: int = 1,
    weighted: bool = True,
) -> List[List[int]]:
    """
    Generate multiple random walks for batch processing.

    Time Complexity: O(N * W * L) where N=num_nodes, W=walks_per_node, L=walk_length
    """
    all_walks = []

    for start_node in start_nodes:
        for _ in range(num_walks_per_node):
            walk = random_walk(adjacency_matrix, start_node, walk_length, weighted)
            all_walks.append(walk)

    return all_walks


class GraphCache:
    """Cache for expensive graph operations to improve online training performance."""

    # NOTE: YOU ARE LIKELY NOT GOING TO GET BENEFITS FROM THIS UNLESS USING KINSHIP
    # This is, Obviously, O(n^3). Most of our datasets have N>1e4

    def __init__(self):
        self._connectivity_cache = {}
        self._transitive_closure = None

    def precompute_connectivity(self, adjacency_matrix: np.ndarray):
        """Precompute transitive closure for O(1) connectivity queries."""
        n = adjacency_matrix.shape[0]
        # Floyd-Warshall for transitive closure
        closure = adjacency_matrix.astype(bool)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])

        self._transitive_closure = closure

    def fast_connectivity_check(self, source: int, target: int) -> bool:
        """O(1) connectivity check using precomputed closure."""
        if self._transitive_closure is None:
            raise ValueError("Must call precompute_connectivity first")
        return self._transitive_closure[source, target]
