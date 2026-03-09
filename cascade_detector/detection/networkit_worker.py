#!/usr/bin/env python3
"""
Isolated subprocess worker for NetworKit network metrics.

Runs in a separate Python process to avoid the dual-libomp SIGSEGV crash
when PyTorch and NetworKit are both loaded in the same process.

Protocol: reads pickled input from stdin, writes pickled output to stdout.

Input dict:
    edges: List[Tuple[node_a, node_b, weight]]
    nodes: List[node_id]

Output dict:
    success: bool
    density: float
    modularity: float
    mean_degree: float
    n_components: int
"""

import sys
import os
import pickle


def compute_metrics(edges, nodes):
    """Compute 4 network metrics using NetworKit."""
    # Configure threading — use up to 4 threads in subprocess
    import multiprocessing as _mp
    max_threads = min(4, (_mp.cpu_count() or 4) // 2)
    os.environ['OMP_NUM_THREADS'] = str(max(1, max_threads))

    import networkit as nk
    nk.setNumberOfThreads(max(1, max_threads))

    # Build NetworKit graph
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    G = nk.Graph(n, weighted=True, directed=False)

    for a, b, w in edges:
        idx_a = node_to_idx.get(a)
        idx_b = node_to_idx.get(b)
        if idx_a is not None and idx_b is not None and idx_a != idx_b:
            if not G.hasEdge(idx_a, idx_b):
                G.addEdge(idx_a, idx_b, w)

    # 1. Density + Average clustering coefficient
    m = G.numberOfEdges()
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0.0
    try:
        if n >= 3:
            lcc = nk.centrality.LocalClusteringCoefficient(G, turbo=True)
            lcc.run()
            scores = lcc.scores()
            avg_clustering = sum(scores) / len(scores) if scores else 0.0
        else:
            avg_clustering = 0.0
    except Exception:
        avg_clustering = 0.0

    # 2. Modularity (Louvain / PLM)
    try:
        plm = nk.community.PLM(G, refine=True, gamma=1.0)
        plm.run()
        partition = plm.getPartition()
        modularity = nk.community.Modularity().getQuality(partition, G)
    except Exception:
        modularity = 0.0

    # 3. Mean degree centrality
    try:
        dc = nk.centrality.DegreeCentrality(G, normalized=True)
        dc.run()
        scores = dc.scores()
        mean_degree = sum(scores) / len(scores) if scores else 0.0
    except Exception:
        mean_degree = 0.0

    # 4. Connected components
    try:
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        n_components = cc.numberOfComponents()
    except Exception:
        n_components = 1

    return {
        'density': density,
        'avg_clustering': avg_clustering,
        'modularity': modularity,
        'mean_degree': mean_degree,
        'n_components': n_components,
        'n_nodes': G.numberOfNodes(),
    }


if __name__ == "__main__":
    try:
        input_data = pickle.load(sys.stdin.buffer)
        edges = input_data['edges']
        nodes = input_data['nodes']

        result = compute_metrics(edges, nodes)
        result['success'] = True
        pickle.dump(result, sys.stdout.buffer)

    except Exception as e:
        import traceback
        pickle.dump({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }, sys.stdout.buffer)
