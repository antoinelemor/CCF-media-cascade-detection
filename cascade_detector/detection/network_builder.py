"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
network_builder.py

MAIN OBJECTIVE:
---------------
Build co-coverage networks for validated cascades and compute 4 interpretable
metrics. Uses NetworKit via subprocess isolation (avoids dual-libomp crash).
Falls back to NetworkX if NetworKit is unavailable.

Graph construction:
- Nodes: unique (journalist, media) pairs publishing in cascade window
- Edges: two actors connected if they published on the same day using the
  same frame. Weight = number of co-occurrence days.

4 metrics:
- Density: widespread simultaneous adoption
- Modularity (Louvain): LOW = cascade crossed media boundaries
- Mean degree centrality: broad co-coverage overlap
- Connected components: 1 = unified diffusion, not fragmented

Author:
-------
Antoine Lemor
"""

import logging
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import networkx as nx

from cascade_detector.core.models import CascadeResult
from cascade_detector.core.constants import FRAME_COLUMNS

logger = logging.getLogger(__name__)

# Path to the NetworKit subprocess worker
_WORKER_SCRIPT = str(Path(__file__).parent / 'networkit_worker.py')


class NetworkBuilder:
    """Builds adoption networks for validated cascades.

    Uses NetworKit via subprocess isolation for performance.
    Falls back to NetworkX if NetworKit is unavailable.
    """

    # Column name mappings (aggregate_by_article uses *_first suffixes)
    DATE_COLS = ['date', 'date_converted_first', 'date_converted']
    AUTHOR_COLS = ['author', 'author_first', 'author_clean_first', 'author_clean']
    MEDIA_COLS = ['media', 'media_first']

    def __init__(self, use_networkit: bool = True, timeout: int = 30):
        """
        Args:
            use_networkit: Whether to try NetworKit (via subprocess).
            timeout: Max seconds for subprocess call.
        """
        self.use_networkit = use_networkit
        self.timeout = timeout
        self._networkit_available = None  # lazy check

    @staticmethod
    def _resolve_col(df: pd.DataFrame, candidates: list) -> str:
        """Return the first column name from candidates that exists in df."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def build(self, cascade: CascadeResult,
              articles: pd.DataFrame) -> Dict[str, Any]:
        """Build co-coverage network and compute metrics.

        Args:
            cascade: Validated cascade.
            articles: Article-level DataFrame.

        Returns:
            Dict with: graph, density, modularity, mean_degree, n_components.
        """
        G = self._build_graph(cascade, articles)

        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return {
                'graph': G,
                'density': 0.0,
                'modularity': 0.0,
                'mean_degree': 0.0,
                'n_components': nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0,
                'n_nodes': G.number_of_nodes(),
            }

        # Try NetworKit subprocess, fall back to NetworkX
        if self.use_networkit:
            metrics = self._compute_networkit(G)
            if metrics is not None:
                metrics['graph'] = G
                return metrics

        # NetworkX fallback
        return self._compute_networkx(G)

    def _build_graph(self, cascade: CascadeResult,
                     articles: pd.DataFrame) -> nx.Graph:
        """Build undirected co-coverage graph.

        Nodes are (journalist, media) tuples. Edges connect pairs
        that published on the same day using the cascade frame.
        """
        G = nx.Graph()

        cascade_articles = self._get_cascade_articles(cascade, articles)
        if cascade_articles.empty:
            return G

        author_col = self._resolve_col(cascade_articles, self.AUTHOR_COLS)
        media_col = self._resolve_col(cascade_articles, self.MEDIA_COLS)
        date_col = self._resolve_col(cascade_articles, self.DATE_COLS)

        if not author_col or not media_col or not date_col:
            return G

        dates = pd.to_datetime(cascade_articles[date_col], errors='coerce')
        cascade_articles = cascade_articles.copy()
        cascade_articles['_date'] = dates.dt.normalize()

        # Group actors by date
        daily_actors = defaultdict(set)
        for _, row in cascade_articles.iterrows():
            author = row.get(author_col)
            media = row.get(media_col)
            date = row.get('_date')
            if pd.notna(author) and pd.notna(media) and pd.notna(date):
                actor = (str(author), str(media))
                daily_actors[date].add(actor)
                if actor not in G:
                    G.add_node(actor, journalist=str(author), media=str(media))

        # Edges: same-day co-coverage
        for date, actors in daily_actors.items():
            actors_list = list(actors)
            for i in range(len(actors_list)):
                for j in range(i + 1, len(actors_list)):
                    a, b = actors_list[i], actors_list[j]
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)

        return G

    def _get_cascade_articles(self, cascade: CascadeResult,
                               articles: pd.DataFrame) -> pd.DataFrame:
        date_col = self._resolve_col(articles, self.DATE_COLS)
        if not date_col:
            return pd.DataFrame()

        dates = pd.to_datetime(articles[date_col], errors='coerce')
        mask = (dates >= cascade.onset_date) & (dates <= cascade.end_date)

        frame_col = FRAME_COLUMNS.get(cascade.frame)
        if frame_col:
            possible_cols = [frame_col, f"{frame_col}_sum", f"{frame_col}_mean"]
            for col in possible_cols:
                if col in articles.columns:
                    mask = mask & (articles[col] > 0)
                    break

        return articles[mask]

    # =========================================================================
    # NetworKit via subprocess
    # =========================================================================

    def _compute_networkit(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute metrics via NetworKit subprocess.

        Returns None if NetworKit is unavailable or subprocess fails.
        """
        # One-time availability check (avoids repeated subprocess spawns)
        if self._networkit_available is False:
            return None
        if not os.path.isfile(_WORKER_SCRIPT):
            logger.debug("NetworKit worker script not found, using NetworkX")
            self._networkit_available = False
            return None

        # Serialize graph as edges + nodes for subprocess
        nodes = list(G.nodes())
        edges = []
        for a, b, data in G.edges(data=True):
            edges.append((a, b, data.get('weight', 1)))

        input_data = {'edges': edges, 'nodes': nodes}

        try:
            proc = subprocess.Popen(
                [sys.executable, _WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdout, stderr = proc.communicate(
                input=pickle.dumps(input_data),
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                err_msg = stderr.decode('utf-8', errors='replace')[:200]
                # Detect missing module → disable permanently
                if 'No module named' in err_msg:
                    logger.info(f"NetworKit not installed, using NetworkX fallback")
                    self._networkit_available = False
                else:
                    logger.warning(f"NetworKit subprocess failed (code {proc.returncode}): {err_msg}")
                return None

            result = pickle.loads(stdout)
            if not result.get('success'):
                err = result.get('error', 'unknown')
                if 'No module named' in err:
                    self._networkit_available = False
                    logger.info(f"NetworKit not installed, using NetworkX fallback")
                else:
                    logger.warning(f"NetworKit computation failed: {err}")
                return None

            self._networkit_available = True
            logger.debug(f"NetworKit metrics computed for {len(nodes)} nodes")
            return {
                'density': result['density'],
                'modularity': result['modularity'],
                'mean_degree': result['mean_degree'],
                'n_components': result['n_components'],
                'n_nodes': result.get('n_nodes', len(nodes)),
            }

        except subprocess.TimeoutExpired:
            proc.kill()
            logger.warning(f"NetworKit subprocess timed out ({self.timeout}s)")
            return None
        except Exception as e:
            logger.warning(f"NetworKit subprocess error: {e}")
            return None

    # =========================================================================
    # NetworkX fallback
    # =========================================================================

    def _compute_networkx(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute metrics using pure NetworkX (fallback)."""
        density = nx.density(G)
        avg_clustering = nx.average_clustering(G) if G.number_of_nodes() >= 3 else 0.0
        modularity = self._modularity_networkx(G)
        mean_degree = float(np.mean(list(nx.degree_centrality(G).values()))) if G.number_of_nodes() > 0 else 0.0
        n_components = nx.number_connected_components(G)

        return {
            'graph': G,
            'density': density,
            'avg_clustering': avg_clustering,
            'modularity': modularity,
            'mean_degree': mean_degree,
            'n_components': n_components,
            'n_nodes': G.number_of_nodes(),
        }

    def _modularity_networkx(self, G: nx.Graph) -> float:
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return 0.0

        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, random_state=42)
            communities = defaultdict(list)
            for node, comm in partition.items():
                communities[comm].append(node)
            return nx.algorithms.community.quality.modularity(G, list(communities.values()))
        except (ImportError, ZeroDivisionError):
            pass

        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G)
            return nx.algorithms.community.quality.modularity(G, communities)
        except Exception:
            return 0.0
