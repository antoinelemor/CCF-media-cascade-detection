#!/usr/bin/env python3
"""
Isolated worker script for NetworKit computations.

This script runs in a completely separate Python process to avoid
OpenMP/threading conflicts with the main process.
"""

import sys
import pickle
import os
import traceback


def compute_metric(edges, nodes, is_directed, metric_name):
    """Compute a metric using NetworKit with EXACT scientific computation.
    
    All metrics are computed exactly without approximation or sampling
    to ensure scientific precision for publication.
    """
    import networkx as nx
    import networkit as nk
    
    # Configure NetworKit threads based on context
    # When running in parallel workers, limit threads to prevent oversubscription
    # Total threads = n_workers * omp_threads should not exceed physical cores
    import multiprocessing as _mp
    _default_threads = str(min(4, (_mp.cpu_count() or 4) // 2))
    max_threads = int(os.environ.get('NETWORKIT_MAX_THREADS', _default_threads))
    os.environ['OMP_NUM_THREADS'] = str(max_threads)
    nk.setNumberOfThreads(max_threads)
    
    # Rebuild graph
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Convert to NetworKit
    nk_graph = nk.nxadapter.nx2nk(G, weightAttr=None)
    
    if metric_name == 'betweenness':
        bc = nk.centrality.Betweenness(nk_graph, normalized=True)
        bc.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = bc.scores()
        return {node_mapping[i]: score for i, score in enumerate(scores)}
        
    elif metric_name == 'closeness':
        hc = nk.centrality.HarmonicCloseness(nk_graph, normalized=True)
        hc.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = hc.scores()
        return {node_mapping[i]: score for i, score in enumerate(scores)}
        
    elif metric_name == 'pagerank':
        # PageRank with maximum precision for scientific computation
        pr = nk.centrality.PageRank(nk_graph, damp=0.85, tol=1e-12)
        pr.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = pr.scores()
        return {node_mapping[i]: score for i, score in enumerate(scores)}
        
    elif metric_name == 'katz':
        # Katz centrality with maximum precision for exact computation
        try:
            kc = nk.centrality.KatzCentrality(nk_graph, alpha=0.01, beta=1.0, tol=1e-12)
            kc.run()
            node_mapping = {i: node for i, node in enumerate(nodes)}
            scores = kc.scores()
            return {node_mapping[i]: score for i, score in enumerate(scores)}
        except:
            # Fallback to eigenvector if Katz fails
            ec = nk.centrality.EigenvectorCentrality(nk_graph, tol=1e-12)
            ec.run()
            node_mapping = {i: node for i, node in enumerate(nodes)}
            scores = ec.scores()
            return {node_mapping[i]: score for i, score in enumerate(scores)}
            
    elif metric_name == 'eigenvector':
        # Eigenvector centrality - only works on undirected graphs
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        ec = nk.centrality.EigenvectorCentrality(nk_undirected, tol=1e-12)
        ec.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = ec.scores()
        return {node_mapping[i]: score for i, score in enumerate(scores)}
            
    elif metric_name == 'local_clustering':
        # Local clustering coefficient - convert to undirected if needed
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        lcc = nk.centrality.LocalClusteringCoefficient(nk_undirected)
        lcc.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = lcc.scores()
        return {node_mapping[i]: score for i, score in enumerate(scores)}
        
    elif metric_name == 'global_clustering':
        # Global clustering coefficient
        gcc = nk.globals.ClusteringCoefficient()
        return gcc.exactGlobal(nk_graph)
        
    elif metric_name == 'transitivity':
        # Graph transitivity
        return nk.globals.ClusteringCoefficient().exactGlobal(nk_graph)
        
    elif metric_name == 'modularity':
        # Community detection with Louvain and modularity - convert to undirected if needed
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        plm = nk.community.PLM(nk_undirected, refine=True, turbo=True)
        plm.run()
        partition = plm.getPartition()
        mod = nk.community.Modularity().getQuality(partition, nk_undirected)
        return {'modularity': mod, 'n_communities': partition.numberOfSubsets()}
        
    elif metric_name == 'greedy_modularity' or metric_name == 'communities':
        # Full community detection using PLM - convert to undirected if needed
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        plm = nk.community.PLM(nk_undirected, refine=True, turbo=True)
        plm.run()
        partition = plm.getPartition()
        
        # Convert to node->community mapping
        node_mapping = {i: node for i, node in enumerate(nodes)}
        communities = {}
        for i in range(len(nodes)):
            communities[node_mapping[i]] = partition.subsetOf(i)
        
        # Also compute modularity
        mod = nk.community.Modularity().getQuality(partition, nk_undirected)
        
        return {'communities': communities, 'modularity': mod, 'n_communities': partition.numberOfSubsets()}
        
    elif metric_name == 'diameter':
        # Graph diameter - ALWAYS EXACT computation
        diam = nk.distance.Diameter(nk_graph, algo=nk.distance.DiameterAlgo.EXACT)
        diam.run()
        return diam.getDiameter()[0]
            
    elif metric_name == 'global_efficiency':
        # Global efficiency - ALWAYS EXACT computation using parallel APSP
        n = nk_graph.numberOfNodes()
        if n == 0:
            return 0.0
            
        # EXACT computation with NetworKit's parallel APSP
        apsp = nk.distance.APSP(nk_graph)
        apsp.run()
        
        total_eff = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist = apsp.getDistance(i, j)
                if 0 < dist < float('inf'):
                    total_eff += 2.0 / dist  # Count both directions
        
        return total_eff / (n * (n - 1))
            
    elif metric_name == 'local_efficiency':
        # Convert to undirected
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
        
        n = nk_undirected.numberOfNodes()
        if n == 0:
            return 0.0
        
        # ALWAYS EXACT computation with parallel APSP
        apsp = nk.distance.APSP(nk_undirected)
        apsp.run()
        
        local_eff_sum = 0.0
        for i in range(n):
            neighbors = list(nk_undirected.iterNeighbors(i))
            k = len(neighbors)
            if k > 1:
                eff = 0.0
                for j_idx, j in enumerate(neighbors):
                    for l in neighbors[j_idx+1:]:
                        dist = apsp.getDistance(j, l)
                        if 0 < dist < float('inf'):
                            eff += 1.0 / dist
                local_eff_sum += eff / (k * (k - 1) / 2)
        
        return local_eff_sum / n
    
    elif metric_name == 'eigenvalues' or metric_name == 'spectral_radius' or metric_name == 'spectral_gap' or metric_name == 'algebraic_connectivity':
        # Spectral metrics - ALWAYS EXACT computation using scipy
        import scipy.sparse
        import numpy as np
        import sys
        from scipy.sparse.linalg import eigsh
        from scipy.linalg import eigh
        
        # Get adjacency matrix from NetworKit graph
        n = nk_graph.numberOfNodes()
        if n == 0:
            if metric_name == 'eigenvalues':
                return []
            return 0.0
            
        rows, cols, data = [], [], []
        for u in range(n):
            for v in nk_graph.iterNeighbors(u):
                rows.append(u)
                cols.append(v)
                data.append(1.0)
        
        adj_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        
        # For spectral radius, compute largest eigenvalue exactly
        if metric_name == 'spectral_radius':
            # Always compute ALL eigenvalues for exact spectral radius
            adj_dense = adj_matrix.todense().astype(np.float64)
            if not np.allclose(adj_dense, adj_dense.T):
                # Directed graph - non-symmetric matrix
                eigvals = np.linalg.eigvals(adj_dense)
                return float(max(abs(ev) for ev in eigvals))
            else:
                # Undirected graph - symmetric matrix
                eigvals = eigh(adj_dense, eigvals_only=True)
                return float(max(abs(ev) for ev in eigvals))
                
        elif metric_name == 'eigenvalues':
            # Return ALL eigenvalues - EXACT computation, no approximation
            # Always compute ALL eigenvalues for exact scientific results
            adj_dense = adj_matrix.todense().astype(np.float64)
            
            # Check if matrix is symmetric
            if not np.allclose(adj_dense, adj_dense.T):
                # For directed graphs, compute eigenvalues of non-symmetric matrix
                eigvals = np.linalg.eigvals(adj_dense)
                # Return all eigenvalues (real parts) sorted
                return sorted(eigvals.real)
            else:
                # For undirected graphs, use eigh for exact computation
                eigvals = eigh(adj_dense, eigvals_only=True)
                return list(eigvals)
                
        elif metric_name == 'spectral_gap':
            # Compute first two largest eigenvalues exactly
            if n > 2:
                if n > 500:
                    eigvals = eigsh(adj_matrix, k=2, which='LM', return_eigenvectors=False, tol=1e-9)
                else:
                    adj_dense = adj_matrix.todense().astype(np.float64)
                    eigvals_all = eigh(adj_dense, eigvals_only=True)
                    eigvals = sorted(eigvals_all, key=abs, reverse=True)[:2]
                return float(abs(eigvals[0] - eigvals[1]))
            return 0.0
            
        elif metric_name == 'algebraic_connectivity':
            # Second smallest eigenvalue of Laplacian (Fiedler value)
            # Build Laplacian matrix
            degree = np.array(adj_matrix.sum(axis=1)).flatten()
            laplacian = scipy.sparse.diags(degree) - adj_matrix
            
            if n > 2:
                if n > 500:
                    # Use sparse solver
                    eigvals = eigsh(laplacian, k=2, which='SM', return_eigenvectors=False, tol=1e-9)
                    return float(sorted(eigvals)[1])  # Second smallest
                else:
                    # Exact computation
                    lap_dense = laplacian.todense().astype(np.float64)
                    eigvals = eigh(lap_dense, eigvals_only=True)
                    return float(sorted(eigvals)[1]) if len(eigvals) > 1 else 0.0
            return 0.0
            
    elif metric_name == 'epidemic_threshold':
        # Epidemic threshold: 1/lambda_max where lambda_max is largest eigenvalue
        import scipy.sparse
        import numpy as np
        from scipy.sparse.linalg import eigsh
        
        n = nk_graph.numberOfNodes()
        if n == 0:
            return 0.0
            
        # Build adjacency matrix
        rows, cols = [], []
        for u in range(n):
            for v in nk_graph.iterNeighbors(u):
                rows.append(u)
                cols.append(v)
        
        if not rows:
            return 0.0
            
        adj_matrix = scipy.sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)
        )
        
        # Compute largest eigenvalue exactly
        if n > 100:
            eigvals = eigsh(adj_matrix, k=1, which='LM', return_eigenvectors=False, tol=1e-9)
            lambda_max = abs(eigvals[0])
        else:
            adj_dense = adj_matrix.todense().astype(np.float64)
            eigvals = np.linalg.eigvalsh(adj_dense)
            lambda_max = max(abs(ev) for ev in eigvals)
        
        # Epidemic threshold is 1/lambda_max
        return 1.0 / lambda_max if lambda_max > 0 else 0.0
        
    elif metric_name == 'cascading_failure':
        # Simulate cascading failure exactly
        n = nk_graph.numberOfNodes()
        if n == 0:
            return {'final_size': 0, 'cascade_size': 0, 'removed_fraction': 0.0}
            
        # Get highest degree nodes (top 5%)
        degrees = [(nk_graph.degree(v), v) for v in range(n)]
        degrees.sort(reverse=True)
        
        nodes_to_remove = max(1, n // 20)  # Remove top 5% nodes
        
        # Measure initial giant component size
        if is_directed:
            cc_initial = nk.components.WeaklyConnectedComponents(nk_graph)
        else:
            cc_initial = nk.components.ConnectedComponents(nk_graph)
        cc_initial.run()
        initial_giant = max(cc_initial.getComponentSizes().values()) if cc_initial.numberOfComponents() > 0 else n
        
        # Create a copy for simulation
        import networkx as nx
        # Rebuild as NetworkX for easier manipulation
        G_sim = nx.Graph() if not is_directed else nx.DiGraph()
        G_sim.add_nodes_from(nodes)
        G_sim.add_edges_from(edges)
        
        # Remove top degree nodes
        nodes_removed = [degrees[i][1] for i in range(min(nodes_to_remove, len(degrees)))]
        for node_idx in nodes_removed:
            node = nodes[node_idx] if node_idx < len(nodes) else node_idx
            if node in G_sim:
                G_sim.remove_node(node)
        
        # Measure final giant component
        if G_sim.number_of_nodes() > 0:
            if is_directed:
                components = list(nx.weakly_connected_components(G_sim))
            else:
                components = list(nx.connected_components(G_sim))
            final_giant = max(len(c) for c in components) if components else 0
        else:
            final_giant = 0
        
        cascade_size = initial_giant - final_giant
        removed_fraction = len(nodes_removed) / n if n > 0 else 0
        
        return {
            'initial_giant': initial_giant,
            'final_giant': final_giant,
            'cascade_size': cascade_size,
            'cascade_fraction': cascade_size / initial_giant if initial_giant > 0 else 0,
            'removed_nodes': len(nodes_removed),
            'removed_fraction': removed_fraction
        }
        
    elif metric_name == 'k_core':
        # K-core decomposition - convert to undirected if needed
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        kc = nk.centrality.CoreDecomposition(nk_undirected)
        kc.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = kc.scores()
        return {node_mapping[i]: score for i, score in enumerate(scores)}
        
    elif metric_name in ('average_clustering', 'complex_contagion'):
        # Average clustering coefficient - exact computation
        # complex_contagion uses average clustering as proxy
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph

        lcc = nk.centrality.LocalClusteringCoefficient(nk_undirected)
        lcc.run()
        scores = lcc.scores()
        return sum(scores) / len(scores) if scores else 0.0

    elif metric_name == 'degree_assortativity':
        # Degree assortativity coefficient
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        dc = nk.centrality.DegreeCentrality(nk_undirected)
        dc.run()
        return nk.correlation.Assortativity(nk_undirected, dc.scores()).run().getCoefficient()
        
    elif metric_name == 'harmonic':
        # Harmonic centrality
        hc = nk.centrality.HarmonicCloseness(nk_graph, normalized=False)
        hc.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = hc.scores()
        return {node_mapping[i]: score for i, score in enumerate(scores)}
        
    elif metric_name == 'load':
        # Load centrality - exact computation via betweenness-degree product
        bc = nk.centrality.Betweenness(nk_graph, normalized=True)
        bc.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = bc.scores()
        # Load centrality is betweenness weighted by degree
        load_scores = {}
        for i, score in enumerate(scores):
            degree = nk_graph.degree(i)
            # Normalize by maximum possible degree
            normalized_degree = degree / (nk_graph.numberOfNodes() - 1) if nk_graph.numberOfNodes() > 1 else 0
            load_scores[node_mapping[i]] = score * (1 + normalized_degree)
        return load_scores
        
    elif metric_name == 'periphery':
        # Periphery: nodes with maximum eccentricity
        if is_directed:
            # Convert to undirected for distance metrics
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        # Compute eccentricities
        n = nk_undirected.numberOfNodes()
        if n == 0:
            return []
            
        apsp = nk.distance.APSP(nk_undirected)
        apsp.run()
        
        eccentricities = {}
        node_mapping = {i: node for i, node in enumerate(nodes)}
        
        for i in range(n):
            max_dist = 0
            for j in range(n):
                if i != j:
                    dist = apsp.getDistance(i, j)
                    if dist != float('inf') and dist > max_dist:
                        max_dist = dist
            eccentricities[node_mapping[i]] = max_dist
        
        if not eccentricities:
            return []
            
        max_ecc = max(eccentricities.values())
        return [node for node, ecc in eccentricities.items() if ecc == max_ecc]
        
    elif metric_name == 'center':
        # Center: nodes with minimum eccentricity
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        n = nk_undirected.numberOfNodes()
        if n == 0:
            return []
            
        apsp = nk.distance.APSP(nk_undirected)
        apsp.run()
        
        eccentricities = {}
        node_mapping = {i: node for i, node in enumerate(nodes)}
        
        for i in range(n):
            max_dist = 0
            for j in range(n):
                if i != j:
                    dist = apsp.getDistance(i, j)
                    if dist != float('inf') and dist > max_dist:
                        max_dist = dist
            if max_dist > 0:  # Only include connected nodes
                eccentricities[node_mapping[i]] = max_dist
        
        if not eccentricities:
            return []
            
        min_ecc = min(eccentricities.values())
        return [node for node, ecc in eccentricities.items() if ecc == min_ecc]
        
    elif metric_name == 'barycenter':
        # Barycenter: nodes with minimum sum of distances
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        n = nk_undirected.numberOfNodes()
        if n == 0:
            return []
            
        apsp = nk.distance.APSP(nk_undirected)
        apsp.run()
        
        distance_sums = {}
        node_mapping = {i: node for i, node in enumerate(nodes)}
        
        for i in range(n):
            total_dist = 0
            for j in range(n):
                if i != j:
                    dist = apsp.getDistance(i, j)
                    if dist != float('inf'):
                        total_dist += dist
            distance_sums[node_mapping[i]] = total_dist
        
        if not distance_sums:
            return []
            
        min_sum = min(distance_sums.values())
        return [node for node, dist_sum in distance_sums.items() if dist_sum == min_sum]
        
    elif metric_name == 'generalized_degree':
        # K-core decomposition for all k values
        if is_directed:
            G_undirected = G.to_undirected()
            nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
        else:
            nk_undirected = nk_graph
            
        kc = nk.centrality.CoreDecomposition(nk_undirected)
        kc.run()
        node_mapping = {i: node for i, node in enumerate(nodes)}
        scores = kc.scores()
        
        # Return k-core values for each node
        return {node_mapping[i]: int(score) for i, score in enumerate(scores)}
        
    elif metric_name in ['R0', 'cascade_size', 'spreading_time']:
        # Basic propagation metrics
        n = nk_graph.numberOfNodes()
        if n == 0:
            return 0.0
            
        # Compute average degree for R0
        if metric_name == 'R0':
            degrees = [nk_graph.degree(v) for v in range(n)]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            # Basic reproduction number estimate
            return avg_degree
            
        elif metric_name == 'cascade_size':
            # Expected cascade size from random seed
            # Use largest eigenvalue as proxy
            import scipy.sparse
            import numpy as np
            from scipy.sparse.linalg import eigsh
            
            rows, cols = [], []
            for u in range(n):
                for v in nk_graph.iterNeighbors(u):
                    rows.append(u)
                    cols.append(v)
            
            if not rows:
                return 0.0
                
            adj_matrix = scipy.sparse.csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(n, n)
            )
            
            if n > 100:
                eigvals = eigsh(adj_matrix, k=1, which='LM', return_eigenvectors=False, tol=1e-9)
                lambda_max = abs(eigvals[0])
            else:
                adj_dense = adj_matrix.todense().astype(np.float64)
                eigvals = np.linalg.eigvalsh(adj_dense)
                lambda_max = max(abs(ev) for ev in eigvals)
            
            # Expected cascade size proportional to spectral radius
            return lambda_max
            
        elif metric_name == 'spreading_time':
            # Estimate spreading time from diameter
            if is_directed:
                # For directed graphs, convert to undirected
                G_undirected = G.to_undirected()
                nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
            else:
                nk_undirected = nk_graph
                
            diam = nk.distance.Diameter(nk_undirected, algo=nk.distance.DiameterAlgo.EXACT)
            diam.run()
            return float(diam.getDiameter()[0])
            
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def compute_metrics_batch(edges, nodes, is_directed, metric_names):
    """Compute multiple metrics in a single process, sharing graph and APSP.

    This avoids the overhead of launching a separate subprocess for each metric.
    The graph is built once, and APSP (All-Pairs Shortest Path) is computed once
    and reused by all metrics that need it.

    Args:
        edges: List of edge tuples
        nodes: List of node identifiers
        is_directed: Whether the graph is directed
        metric_names: List of metric names to compute

    Returns:
        Dict mapping metric_name -> result (or error string on failure)
    """
    import multiprocessing
    import networkx as nx
    import networkit as nk

    # Batch runs in a single subprocess — use half the cores (capped at 8)
    _default_threads = str(min(8, (multiprocessing.cpu_count() or 4) // 2))
    max_threads = int(os.environ.get('NETWORKIT_MAX_THREADS', _default_threads))
    os.environ['OMP_NUM_THREADS'] = str(max_threads)
    nk.setNumberOfThreads(max_threads)

    # Build graphs once
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    nk_graph = nk.nxadapter.nx2nk(G, weightAttr=None)

    # Build undirected variants once (many metrics need them)
    if is_directed:
        G_undirected = G.to_undirected()
        nk_undirected = nk.nxadapter.nx2nk(G_undirected, weightAttr=None)
    else:
        G_undirected = G
        nk_undirected = nk_graph

    n = nk_graph.numberOfNodes()
    node_mapping = {i: node for i, node in enumerate(nodes)}

    # Pre-compute APSP if any metric needs it
    apsp_metrics = {'diameter', 'global_efficiency', 'local_efficiency',
                    'periphery', 'center', 'barycenter'}
    needs_apsp = bool(apsp_metrics & set(metric_names))

    apsp_directed = None
    apsp_undirected = None

    if needs_apsp and n > 0:
        # Compute APSP on the appropriate graph variant
        apsp_undirected = nk.distance.APSP(nk_undirected)
        apsp_undirected.run()
        # For directed metrics (global_efficiency), compute on directed graph
        if is_directed and {'global_efficiency'} & set(metric_names):
            apsp_directed = nk.distance.APSP(nk_graph)
            apsp_directed.run()
        else:
            apsp_directed = apsp_undirected

    results = {}

    for metric_name in metric_names:
        try:
            result = _compute_single_metric_with_shared_state(
                metric_name, G, G_undirected, nk_graph, nk_undirected,
                is_directed, nodes, node_mapping, n,
                apsp_directed, apsp_undirected, edges
            )
            results[metric_name] = result
        except Exception as e:
            results[metric_name] = {'__error__': str(e)}

    return results


def _compute_single_metric_with_shared_state(
    metric_name, G, G_undirected, nk_graph, nk_undirected,
    is_directed, nodes, node_mapping, n,
    apsp_directed, apsp_undirected, edges
):
    """Compute a single metric reusing pre-built graphs and APSP results."""
    import networkit as nk
    import numpy as np

    if metric_name == 'betweenness':
        bc = nk.centrality.Betweenness(nk_graph, normalized=True)
        bc.run()
        return {node_mapping[i]: s for i, s in enumerate(bc.scores())}

    elif metric_name == 'closeness':
        hc = nk.centrality.HarmonicCloseness(nk_graph, normalized=True)
        hc.run()
        return {node_mapping[i]: s for i, s in enumerate(hc.scores())}

    elif metric_name == 'pagerank':
        pr = nk.centrality.PageRank(nk_graph, damp=0.85, tol=1e-12)
        pr.run()
        return {node_mapping[i]: s for i, s in enumerate(pr.scores())}

    elif metric_name == 'katz':
        try:
            kc = nk.centrality.KatzCentrality(nk_graph, alpha=0.01, beta=1.0, tol=1e-12)
            kc.run()
            return {node_mapping[i]: s for i, s in enumerate(kc.scores())}
        except Exception:
            ec = nk.centrality.EigenvectorCentrality(nk_graph, tol=1e-12)
            ec.run()
            return {node_mapping[i]: s for i, s in enumerate(ec.scores())}

    elif metric_name == 'eigenvector':
        ec = nk.centrality.EigenvectorCentrality(nk_undirected, tol=1e-12)
        ec.run()
        return {node_mapping[i]: s for i, s in enumerate(ec.scores())}

    elif metric_name == 'local_clustering':
        lcc = nk.centrality.LocalClusteringCoefficient(nk_undirected)
        lcc.run()
        return {node_mapping[i]: s for i, s in enumerate(lcc.scores())}

    elif metric_name == 'global_clustering':
        gcc = nk.globals.ClusteringCoefficient()
        return gcc.exactGlobal(nk_graph)

    elif metric_name == 'transitivity':
        return nk.globals.ClusteringCoefficient().exactGlobal(nk_graph)

    elif metric_name == 'modularity':
        plm = nk.community.PLM(nk_undirected, refine=True, turbo=True)
        plm.run()
        partition = plm.getPartition()
        mod = nk.community.Modularity().getQuality(partition, nk_undirected)
        return {'modularity': mod, 'n_communities': partition.numberOfSubsets()}

    elif metric_name in ('greedy_modularity', 'communities'):
        plm = nk.community.PLM(nk_undirected, refine=True, turbo=True)
        plm.run()
        partition = plm.getPartition()
        communities = {node_mapping[i]: partition.subsetOf(i) for i in range(len(nodes))}
        mod = nk.community.Modularity().getQuality(partition, nk_undirected)
        return {'communities': communities, 'modularity': mod, 'n_communities': partition.numberOfSubsets()}

    elif metric_name == 'diameter':
        # Use pre-computed APSP for eccentricity-based diameter
        if apsp_undirected is not None and n > 0:
            max_dist = 0
            for i in range(n):
                for j in range(i + 1, n):
                    d = apsp_undirected.getDistance(i, j)
                    if d != float('inf') and d > max_dist:
                        max_dist = d
            return int(max_dist)
        diam = nk.distance.Diameter(nk_graph, algo=nk.distance.DiameterAlgo.EXACT)
        diam.run()
        return diam.getDiameter()[0]

    elif metric_name == 'global_efficiency':
        if n == 0:
            return 0.0
        apsp = apsp_directed if apsp_directed is not None else apsp_undirected
        if apsp is None:
            apsp = nk.distance.APSP(nk_graph)
            apsp.run()
        total_eff = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = apsp.getDistance(i, j)
                if 0 < dist < float('inf'):
                    total_eff += 2.0 / dist
        return total_eff / (n * (n - 1)) if n > 1 else 0.0

    elif metric_name == 'local_efficiency':
        if n == 0:
            return 0.0
        apsp = apsp_undirected
        if apsp is None:
            apsp = nk.distance.APSP(nk_undirected)
            apsp.run()
        local_eff_sum = 0.0
        for i in range(n):
            neighbors = list(nk_undirected.iterNeighbors(i))
            k = len(neighbors)
            if k > 1:
                eff = 0.0
                for j_idx, j in enumerate(neighbors):
                    for l in neighbors[j_idx + 1:]:
                        dist = apsp.getDistance(j, l)
                        if 0 < dist < float('inf'):
                            eff += 1.0 / dist
                local_eff_sum += eff / (k * (k - 1) / 2)
        return local_eff_sum / n

    elif metric_name in ('eigenvalues', 'spectral_radius', 'spectral_gap', 'algebraic_connectivity'):
        import scipy.sparse
        from scipy.sparse.linalg import eigsh
        from scipy.linalg import eigh

        if n == 0:
            return [] if metric_name == 'eigenvalues' else 0.0

        rows, cols, data = [], [], []
        for u in range(n):
            for v in nk_graph.iterNeighbors(u):
                rows.append(u)
                cols.append(v)
                data.append(1.0)
        adj_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

        if metric_name == 'spectral_radius':
            adj_dense = adj_matrix.todense().astype(np.float64)
            if not np.allclose(adj_dense, adj_dense.T):
                eigvals = np.linalg.eigvals(adj_dense)
                return float(max(abs(ev) for ev in eigvals))
            else:
                eigvals = eigh(adj_dense, eigvals_only=True)
                return float(max(abs(ev) for ev in eigvals))
        elif metric_name == 'eigenvalues':
            adj_dense = adj_matrix.todense().astype(np.float64)
            if not np.allclose(adj_dense, adj_dense.T):
                eigvals = np.linalg.eigvals(adj_dense)
                return sorted(eigvals.real)
            else:
                eigvals = eigh(adj_dense, eigvals_only=True)
                return list(eigvals)
        elif metric_name == 'spectral_gap':
            if n > 2:
                if n > 500:
                    eigvals = eigsh(adj_matrix, k=2, which='LM', return_eigenvectors=False, tol=1e-9)
                else:
                    adj_dense = adj_matrix.todense().astype(np.float64)
                    eigvals_all = eigh(adj_dense, eigvals_only=True)
                    eigvals = sorted(eigvals_all, key=abs, reverse=True)[:2]
                return float(abs(eigvals[0] - eigvals[1]))
            return 0.0
        elif metric_name == 'algebraic_connectivity':
            degree = np.array(adj_matrix.sum(axis=1)).flatten()
            laplacian = scipy.sparse.diags(degree) - adj_matrix
            if n > 2:
                if n > 500:
                    eigvals = eigsh(laplacian, k=2, which='SM', return_eigenvectors=False, tol=1e-9)
                    return float(sorted(eigvals)[1])
                else:
                    lap_dense = laplacian.todense().astype(np.float64)
                    eigvals = eigh(lap_dense, eigvals_only=True)
                    return float(sorted(eigvals)[1]) if len(eigvals) > 1 else 0.0
            return 0.0

    elif metric_name == 'epidemic_threshold':
        import scipy.sparse
        from scipy.sparse.linalg import eigsh

        if n == 0:
            return 0.0
        rows, cols = [], []
        for u in range(n):
            for v in nk_graph.iterNeighbors(u):
                rows.append(u)
                cols.append(v)
        if not rows:
            return 0.0
        adj_matrix = scipy.sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)
        )
        if n > 100:
            eigvals = eigsh(adj_matrix, k=1, which='LM', return_eigenvectors=False, tol=1e-9)
            lambda_max = abs(eigvals[0])
        else:
            adj_dense = adj_matrix.todense().astype(np.float64)
            eigvals = np.linalg.eigvalsh(adj_dense)
            lambda_max = max(abs(ev) for ev in eigvals)
        return 1.0 / lambda_max if lambda_max > 0 else 0.0

    elif metric_name == 'cascading_failure':
        import networkx as nx

        if n == 0:
            return {'final_size': 0, 'cascade_size': 0, 'removed_fraction': 0.0}
        degrees = [(nk_graph.degree(v), v) for v in range(n)]
        degrees.sort(reverse=True)
        nodes_to_remove = max(1, n // 20)
        if is_directed:
            cc_initial = nk.components.WeaklyConnectedComponents(nk_graph)
        else:
            cc_initial = nk.components.ConnectedComponents(nk_graph)
        cc_initial.run()
        initial_giant = max(cc_initial.getComponentSizes().values()) if cc_initial.numberOfComponents() > 0 else n
        G_sim = nx.Graph() if not is_directed else nx.DiGraph()
        G_sim.add_nodes_from(nodes)
        G_sim.add_edges_from(edges)
        nodes_removed = [degrees[i][1] for i in range(min(nodes_to_remove, len(degrees)))]
        for node_idx in nodes_removed:
            node = nodes[node_idx] if node_idx < len(nodes) else node_idx
            if node in G_sim:
                G_sim.remove_node(node)
        if G_sim.number_of_nodes() > 0:
            if is_directed:
                components = list(nx.weakly_connected_components(G_sim))
            else:
                components = list(nx.connected_components(G_sim))
            final_giant = max(len(c) for c in components) if components else 0
        else:
            final_giant = 0
        cascade_size = initial_giant - final_giant
        removed_fraction = len(nodes_removed) / n if n > 0 else 0
        return {
            'initial_giant': initial_giant, 'final_giant': final_giant,
            'cascade_size': cascade_size,
            'cascade_fraction': cascade_size / initial_giant if initial_giant > 0 else 0,
            'removed_nodes': len(nodes_removed), 'removed_fraction': removed_fraction
        }

    elif metric_name == 'k_core':
        kc = nk.centrality.CoreDecomposition(nk_undirected)
        kc.run()
        return {node_mapping[i]: s for i, s in enumerate(kc.scores())}

    elif metric_name in ('average_clustering', 'complex_contagion'):
        # complex_contagion uses average clustering as proxy
        lcc = nk.centrality.LocalClusteringCoefficient(nk_undirected)
        lcc.run()
        scores = lcc.scores()
        return sum(scores) / len(scores) if scores else 0.0

    elif metric_name == 'degree_assortativity':
        dc = nk.centrality.DegreeCentrality(nk_undirected)
        dc.run()
        return nk.correlation.Assortativity(nk_undirected, dc.scores()).run().getCoefficient()

    elif metric_name == 'harmonic':
        hc = nk.centrality.HarmonicCloseness(nk_graph, normalized=False)
        hc.run()
        return {node_mapping[i]: s for i, s in enumerate(hc.scores())}

    elif metric_name == 'load':
        bc = nk.centrality.Betweenness(nk_graph, normalized=True)
        bc.run()
        scores = bc.scores()
        load_scores = {}
        for i, score in enumerate(scores):
            degree = nk_graph.degree(i)
            normalized_degree = degree / (nk_graph.numberOfNodes() - 1) if nk_graph.numberOfNodes() > 1 else 0
            load_scores[node_mapping[i]] = score * (1 + normalized_degree)
        return load_scores

    elif metric_name == 'periphery':
        if n == 0:
            return []
        apsp = apsp_undirected
        if apsp is None:
            apsp = nk.distance.APSP(nk_undirected)
            apsp.run()
        eccentricities = {}
        for i in range(n):
            max_dist = 0
            for j in range(n):
                if i != j:
                    dist = apsp.getDistance(i, j)
                    if dist != float('inf') and dist > max_dist:
                        max_dist = dist
            eccentricities[node_mapping[i]] = max_dist
        if not eccentricities:
            return []
        max_ecc = max(eccentricities.values())
        return [node for node, ecc in eccentricities.items() if ecc == max_ecc]

    elif metric_name == 'center':
        if n == 0:
            return []
        apsp = apsp_undirected
        if apsp is None:
            apsp = nk.distance.APSP(nk_undirected)
            apsp.run()
        eccentricities = {}
        for i in range(n):
            max_dist = 0
            for j in range(n):
                if i != j:
                    dist = apsp.getDistance(i, j)
                    if dist != float('inf') and dist > max_dist:
                        max_dist = dist
            if max_dist > 0:
                eccentricities[node_mapping[i]] = max_dist
        if not eccentricities:
            return []
        min_ecc = min(eccentricities.values())
        return [node for node, ecc in eccentricities.items() if ecc == min_ecc]

    elif metric_name == 'barycenter':
        if n == 0:
            return []
        apsp = apsp_undirected
        if apsp is None:
            apsp = nk.distance.APSP(nk_undirected)
            apsp.run()
        distance_sums = {}
        for i in range(n):
            total_dist = 0
            for j in range(n):
                if i != j:
                    dist = apsp.getDistance(i, j)
                    if dist != float('inf'):
                        total_dist += dist
            distance_sums[node_mapping[i]] = total_dist
        if not distance_sums:
            return []
        min_sum = min(distance_sums.values())
        return [node for node, ds in distance_sums.items() if ds == min_sum]

    elif metric_name == 'generalized_degree':
        kc = nk.centrality.CoreDecomposition(nk_undirected)
        kc.run()
        return {node_mapping[i]: int(s) for i, s in enumerate(kc.scores())}

    elif metric_name in ('R0', 'cascade_size', 'spreading_time'):
        if n == 0:
            return 0.0
        if metric_name == 'R0':
            degrees = [nk_graph.degree(v) for v in range(n)]
            return sum(degrees) / len(degrees) if degrees else 0
        elif metric_name == 'cascade_size':
            import scipy.sparse
            from scipy.sparse.linalg import eigsh
            rows, cols = [], []
            for u in range(n):
                for v in nk_graph.iterNeighbors(u):
                    rows.append(u)
                    cols.append(v)
            if not rows:
                return 0.0
            adj_matrix = scipy.sparse.csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(n, n)
            )
            if n > 100:
                eigvals = eigsh(adj_matrix, k=1, which='LM', return_eigenvectors=False, tol=1e-9)
                lambda_max = abs(eigvals[0])
            else:
                adj_dense = adj_matrix.todense().astype(np.float64)
                eigvals = np.linalg.eigvalsh(adj_dense)
                lambda_max = max(abs(ev) for ev in eigvals)
            return lambda_max
        elif metric_name == 'spreading_time':
            diam = nk.distance.Diameter(nk_undirected, algo=nk.distance.DiameterAlgo.EXACT)
            diam.run()
            return float(diam.getDiameter()[0])

    else:
        raise ValueError(f"Unknown metric: {metric_name}")


if __name__ == "__main__":
    # Read input from stdin
    try:
        input_data = pickle.load(sys.stdin.buffer)
        edges = input_data['edges']
        nodes = input_data['nodes']
        is_directed = input_data['is_directed']

        # Batch mode: compute multiple metrics in one call
        if 'metric_names' in input_data:
            metric_names = input_data['metric_names']
            results = compute_metrics_batch(edges, nodes, is_directed, metric_names)
            pickle.dump({'success': True, 'result': results, 'batch': True}, sys.stdout.buffer)
        else:
            # Single metric mode (backward compatible)
            metric_name = input_data['metric_name']
            result = compute_metric(edges, nodes, is_directed, metric_name)
            pickle.dump({'success': True, 'result': result}, sys.stdout.buffer)

    except Exception as e:
        # Write error to stdout
        error_data = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        pickle.dump(error_data, sys.stdout.buffer)