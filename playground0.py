# Minimal CSR neighbor sampling vs. list-of-lists
# Run on CPU or GPU (if available). Requires: torch>=2.0

import time
import math
import random
import torch

def build_random_graph(N: int, avg_deg: int, seed=0):
    random.seed(seed)
    edges = []
    for u in range(N):
        deg = max(1, int(random.gauss(avg_deg, avg_deg*0.2)))
        for _ in range(deg):
            v = random.randrange(N)
            edges.append((u, v))
    # Build list-of-lists adjacency
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
    # Build CSR
    # 1) sort edges by u
    edges.sort()
    indices = []
    indptr = [0]
    cur = 0
    for u in range(N):
        while cur < len(edges) and edges[cur][0] == u:
            indices.append(edges[cur][1])
            cur += 1
        indptr.append(len(indices))
    return adj, torch.tensor(indptr, dtype=torch.long), torch.tensor(indices, dtype=torch.long)

def sample_neighbors_list_of_lists(adj, nodes, fanout=4):
    # Python loop over nodes; returns a list of sampled neighbor lists
    out = []
    for u in nodes:
        nbrs = adj[u]
        if not nbrs:
            out.append([])
            continue
        picks = [nbrs[random.randrange(len(nbrs))] for _ in range(fanout)]
        out.append(picks)
    return out

@torch.no_grad()
def sample_neighbors_csr(indptr, indices, nodes_tensor, fanout=4):
    """
    Vectorized, batched neighbor sampling:
      - nodes_tensor: [B] (on CPU or CUDA)
      Returns next_nodes: [B, fanout]
    """
    device = nodes_tensor.device
    v = nodes_tensor
    deg = indptr[v+1] - indptr[v]          # [B]
    start = indptr[v]                       # [B]

    # Sample positions per row without Python loops:
    # Different rows have different degrees, so use modulo trick.
    max_deg = int(deg.max().item()) if deg.numel() > 0 else 0
    if max_deg == 0:
        return torch.empty((v.numel(), fanout), dtype=torch.long, device=device)

    pos = torch.randint(0, max_deg, (v.numel(), fanout), device=device)
    pos = torch.remainder(pos, torch.clamp(deg.unsqueeze(1), min=1))  # safe when deg=0

    edge_idx = start.unsqueeze(1) + pos      # [B, fanout]
    next_nodes = indices[edge_idx]           # [B, fanout]
    # Optionally mask rows with deg==0 (here we just return repeated junk; caller can mask)
    return next_nodes

def time_it(fn, *args, warmup=3, iters=10):
    # Simple timer that handles CUDA sync if needed
    device = None
    for x in args:
        if isinstance(x, torch.Tensor):
            device = x.device
            break

    def sync():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()

    # warmup
    for _ in range(warmup):
        _ = fn(*args)
        sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(*args)
        sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

if __name__ == "__main__":
    N = 200_000        # number of nodes
    avg_deg = 8        # average out-degree
    B = 32_000         # batch of nodes to sample from
    fanout = 4

    adj, indptr, indices = build_random_graph(N, avg_deg)
    nodes = torch.randint(0, N, (B,))

    # CPU timings
    t_list = time_it(sample_neighbors_list_of_lists, adj, nodes.tolist(), iters=5)
    t_csr_cpu = time_it(sample_neighbors_csr, indptr, indices, nodes, fanout, iters=20)

    print(f"List-of-lists (CPU, Python loop): {t_list*1e3:.2f} ms/batch")
    print(f"CSR (CPU, vectorized torch):     {t_csr_cpu*1e3:.2f} ms/batch")

    # GPU timings (if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        indptr_cuda  = indptr.to(device)
        indices_cuda = indices.to(device)
        nodes_cuda   = nodes.to(device)
        # warm up CUDA
        _ = sample_neighbors_csr(indptr_cuda, indices_cuda, nodes_cuda, fanout)
        torch.cuda.synchronize()
        t_csr_gpu = time_it(sample_neighbors_csr, indptr_cuda, indices_cuda, nodes_cuda, fanout, iters=50)
        print(f"CSR (CUDA, vectorized torch):   {t_csr_gpu*1e3:.2f} ms/batch")
