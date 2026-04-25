"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
paradigm_dominance.py

MAIN OBJECTIVE:
---------------
This script determines frame dominance using multiple rigorous methods:
information theory, network analysis, causality, and proportional analysis.
All methods have equal weight in determining paradigm composition.


Dependencies:
-------------
- numpy
- pandas
- scipy
- sklearn
- networkx
- statsmodels
- tqdm

MAIN FEATURES:
--------------
1) Information theory metrics (mutual information, conditional entropy)
2) Network temporal analysis (PageRank, eigenvector centrality)
3) Causality analysis (Granger causality, transfer entropy)
4) Proportional dominance analysis
5) Objective paradigm boundary determination
6) Analyzes all periods
7) Progress tracking with tqdm for long operations

Author:
-------
Antoine Lemor
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine, pdist, squareform
from scipy.stats import entropy, percentileofscore, gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, List, Tuple, Optional, Set
import warnings
import logging
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import itertools
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ParadigmDominanceAnalyzer:
    """Analyzes frame dominance using multiple scientific methods."""
    
    def __init__(self, frame_names: List[str] = None, n_workers: Optional[int] = None):
        """
        Initialize dominance analyzer.
        
        Args:
            frame_names: List of frame names to analyze
            n_workers: Number of parallel workers (None for auto-detect)
        """
        self.frame_names = frame_names or ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]
        self.n_frames = len(self.frame_names)
        self.n_workers = n_workers or min(cpu_count(), 12)  # Cap at 12 for efficiency
        
    def calculate_information_theory_metrics(self, data: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate information theory metrics for frame dominance.
        Handles small datasets gracefully.
        
        Args:
            data: DataFrame with frame proportions or counts
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with information theory metrics
        """
        results = pd.DataFrame(index=self.frame_names)
        
        # For very small datasets, use fewer bins
        n_bins = min(10, max(3, int(np.sqrt(len(data)))))
        
        # Convert proportions to discrete bins for mutual information
        discretized = pd.DataFrame()
        
        if show_progress:
            desc = "Discretizing frames for MI"
            frame_iter = tqdm(self.frame_names, desc=desc, leave=False)
        else:
            frame_iter = self.frame_names
            
        for frame in frame_iter:
            discretized[frame] = pd.cut(data[frame], bins=n_bins, labels=False)
        
        # Parallel computation of mutual information
        total_mi_calculations = len(self.frame_names)
        
        if show_progress:
            pbar = tqdm(total=total_mi_calculations, desc="Computing mutual information", leave=False)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            mi_futures = {}
            
            for frame in self.frame_names:
                future = executor.submit(self._calculate_frame_mutual_info, 
                                       frame, discretized, self.frame_names)
                mi_futures[frame] = future
            
            mutual_info_scores = []
            for frame in self.frame_names:
                mi_score = mi_futures[frame].result()
                mutual_info_scores.append(mi_score)
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        results['mutual_information'] = mutual_info_scores
        
        # Parallel computation of entropy reduction
        base_entropy = self._calculate_joint_entropy(data[self.frame_names])
        
        if show_progress:
            pbar = tqdm(total=len(self.frame_names), desc="Computing entropy reduction", leave=False)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            entropy_futures = {}
            
            for frame in self.frame_names:
                other_frames = [f for f in self.frame_names if f != frame]
                future = executor.submit(self._calculate_conditional_entropy,
                                       data[other_frames], data[frame])
                entropy_futures[frame] = future
            
            for frame in self.frame_names:
                conditional_entropy = entropy_futures[frame].result()
                results.loc[frame, 'entropy_reduction'] = (base_entropy - conditional_entropy) / base_entropy if base_entropy > 0 else 0
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # Information content (self-entropy)
        if show_progress:
            desc = "Computing information content"
            frame_iter = tqdm(self.frame_names, desc=desc, leave=False)
        else:
            frame_iter = self.frame_names
            
        for frame in frame_iter:
            frame_entropy = entropy(np.histogram(data[frame], bins=n_bins)[0] + 1)
            results.loc[frame, 'information_content'] = frame_entropy
        
        return results
    
    def _calculate_frame_mutual_info(self, frame: str, discretized: pd.DataFrame, 
                                   all_frames: List[str]) -> float:
        """Calculate mutual information for a single frame."""
        mi_sum = 0
        for other_frame in all_frames:
            if frame != other_frame:
                mi = mutual_info_score(discretized[frame], discretized[other_frame])
                mi_sum += mi
        return mi_sum / (len(all_frames) - 1) if len(all_frames) > 1 else 0
    
    def _calculate_joint_entropy(self, data: pd.DataFrame) -> float:
        """Calculate joint entropy of multiple variables."""
        # Discretize and calculate joint distribution
        n_bins = min(10, max(3, int(np.sqrt(len(data)))))
        discretized = data.apply(lambda x: pd.cut(x, bins=n_bins, labels=False))
        
        # Create joint distribution
        joint_counts = discretized.value_counts()
        joint_probs = joint_counts / len(data)
        
        # Calculate entropy
        return entropy(joint_probs[joint_probs > 0])
    
    def _calculate_conditional_entropy(self, X: pd.DataFrame, Y: pd.Series) -> float:
        """Calculate H(X|Y) - entropy of X given Y."""
        n_bins = min(10, max(3, int(np.sqrt(len(Y)))))
        
        # Discretize
        X_disc = X.apply(lambda x: pd.cut(x, bins=n_bins, labels=False))
        Y_disc = pd.cut(Y, bins=n_bins, labels=False)
        
        # Calculate conditional entropy
        conditional_entropy = 0
        Y_vals = Y_disc.unique()
        
        for y_val in Y_vals:
            mask = Y_disc == y_val
            p_y = mask.sum() / len(Y_disc)
            
            if p_y > 0:
                X_given_y = X_disc[mask]
                if len(X_given_y) > 0:
                    joint_counts = X_given_y.value_counts()
                    joint_probs = joint_counts / len(X_given_y)
                    h_x_given_y = entropy(joint_probs[joint_probs > 0])
                    conditional_entropy += p_y * h_x_given_y
        
        return conditional_entropy
    
    def calculate_network_metrics(self, data: pd.DataFrame, window_size: Optional[int] = None, 
                                show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate network-based dominance metrics using temporal windows.
        Adapts window size to data length.
        
        Args:
            data: DataFrame with frame data
            window_size: Size of temporal window (None for adaptive sizing)
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with network metrics
        """
        # Adaptive window size based on data length
        if window_size is None:
            data_length = len(data)
            if data_length < 20:
                window_size = data_length  # Use all data
            elif data_length < 52:
                window_size = max(12, data_length // 2)  # At least 12 weeks or half the data
            else:
                window_size = 52  # Default 1 year
        
        results = pd.DataFrame(index=self.frame_names)
        
        # Generate window indices with overlap
        window_indices = []
        if len(data) <= window_size:
            # Single window with all data
            window_indices.append((0, len(data)))
        else:
            # Multiple overlapping windows
            step_size = max(1, window_size // 2)
            for start_idx in range(0, len(data) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_indices.append((start_idx, end_idx))
        
        # Parallel processing of windows with progress bar
        if show_progress:
            pbar = tqdm(total=len(window_indices), desc="Processing network windows", leave=False)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            window_futures = []
            
            for start_idx, end_idx in window_indices:
                window_data = data.iloc[start_idx:end_idx]
                future = executor.submit(self._process_network_window, window_data)
                window_futures.append(future)
            
            # Collect results
            pagerank_scores = {frame: [] for frame in self.frame_names}
            eigenvector_scores = {frame: [] for frame in self.frame_names}
            
            for future in as_completed(window_futures):
                pr_scores, ec_scores = future.result()
                if pr_scores:
                    for frame, score in pr_scores.items():
                        pagerank_scores[frame].append(score)
                if ec_scores:
                    for frame, score in ec_scores.items():
                        eigenvector_scores[frame].append(score)
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # Average scores over time windows
        for frame in self.frame_names:
            if pagerank_scores[frame]:
                results.loc[frame, 'pagerank'] = np.mean(pagerank_scores[frame])
                results.loc[frame, 'pagerank_stability'] = 1 / (1 + np.std(pagerank_scores[frame]))
            else:
                results.loc[frame, 'pagerank'] = 0
                results.loc[frame, 'pagerank_stability'] = 0
                
            if eigenvector_scores[frame]:
                results.loc[frame, 'eigenvector_centrality'] = np.mean(eigenvector_scores[frame])
                results.loc[frame, 'eigenvector_stability'] = 1 / (1 + np.std(eigenvector_scores[frame]))
            else:
                results.loc[frame, 'eigenvector_centrality'] = 0
                results.loc[frame, 'eigenvector_stability'] = 0
        
        return results
    
    def _process_network_window(self, window_data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Process a single window for network metrics."""
        G = self._build_cooccurrence_network(window_data)
        
        pr_scores = {}
        ec_scores = {}
        
        if G.number_of_nodes() > 0:
            try:
                pr = nx.pagerank(G, weight='weight')
                pr_scores = {frame: pr.get(frame, 0) for frame in self.frame_names}
            except:
                pass
            
            try:
                ec = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                ec_scores = {frame: ec.get(frame, 0) for frame in self.frame_names}
            except:
                pass
        
        return pr_scores, ec_scores
    
    def _build_cooccurrence_network(self, data: pd.DataFrame) -> nx.Graph:
        """Build frame co-occurrence network from data."""
        G = nx.Graph()
        
        # Add nodes
        for frame in self.frame_names:
            G.add_node(frame)
        
        # Calculate co-occurrence strengths
        threshold = 0.1  # Minimum proportion to consider presence
        
        for i, frame1 in enumerate(self.frame_names):
            for j, frame2 in enumerate(self.frame_names):
                if i < j:
                    # Count co-occurrences
                    frame1_present = data[frame1] > threshold
                    frame2_present = data[frame2] > threshold
                    cooccurrence = (frame1_present & frame2_present).sum()
                    
                    if cooccurrence > 0:
                        weight = cooccurrence / len(data)
                        G.add_edge(frame1, frame2, weight=weight)
        
        return G
    
    def calculate_causality_metrics(self, data: pd.DataFrame, max_lag: Optional[int] = None, 
                                  show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate causality-based dominance metrics.
        Adapts lag to data length.
        
        Args:
            data: DataFrame with temporal frame data
            max_lag: Maximum lag for Granger causality (None for adaptive)
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with causality metrics
        """
        # Adaptive lag based on data length
        if max_lag is None:
            data_length = len(data)
            if data_length < 10:
                max_lag = 1
            elif data_length < 20:
                max_lag = 2
            elif data_length < 40:
                max_lag = 3
            else:
                max_lag = 4
        
        results = pd.DataFrame(index=self.frame_names)
        
        # Generate all frame pairs for parallel processing
        frame_pairs = [(f1, f2) for f1 in self.frame_names 
                      for f2 in self.frame_names if f1 != f2]
        
        # Parallel Granger causality tests
        if show_progress:
            pbar = tqdm(total=len(frame_pairs), desc="Computing Granger causality", leave=False)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            granger_futures = {}
            
            for frame1, frame2 in frame_pairs:
                future = executor.submit(self._test_granger_causality,
                                       data[frame1], data[frame2], max_lag)
                granger_futures[(frame1, frame2)] = future
            
            # Collect results
            granger_scores = {frame: 0 for frame in self.frame_names}
            
            for (frame1, frame2), future in granger_futures.items():
                significant = future.result()
                if significant:
                    granger_scores[frame1] += 1
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # Normalize Granger scores
        max_possible = self.n_frames - 1
        for frame in self.frame_names:
            results.loc[frame, 'granger_causality'] = granger_scores[frame] / max_possible if max_possible > 0 else 0
        
        # Parallel Transfer Entropy calculation
        if show_progress:
            pbar = tqdm(total=len(self.frame_names), desc="Computing transfer entropy", leave=False)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            te_futures = {}
            
            for frame in self.frame_names:
                future = executor.submit(self._calculate_transfer_entropy,
                                       data, frame, self.frame_names)
                te_futures[frame] = future
            
            transfer_entropy_scores = []
            for frame in self.frame_names:
                te_score = te_futures[frame].result()
                transfer_entropy_scores.append(te_score)
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        results['transfer_entropy'] = transfer_entropy_scores
        
        return results
    
    def _test_granger_causality(self, x: pd.Series, y: pd.Series, max_lag: int) -> bool:
        """Test if x Granger-causes y."""
        try:
            # Need at least 2 * max_lag + 1 observations
            if len(x) < 2 * max_lag + 1:
                return False
                
            test_data = pd.DataFrame({'x': y.values, 'y': x.values})
            test_result = grangercausalitytests(test_data, max_lag, verbose=False)
            
            # Get minimum p-value across lags
            p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            min_p = min(p_values)
            
            return min_p < 0.05
        except:
            return False
    
    def _calculate_transfer_entropy(self, data: pd.DataFrame, source: str, 
                                   targets: List[str], lag: int = 1) -> float:
        """Calculate transfer entropy from source to all targets."""
        # Check if we have enough data for the lag
        if len(data) <= lag:
            return 0
            
        te_sum = 0
        n_targets = 0
        
        for target in targets:
            if target != source:
                try:
                    # Discretize data with adaptive bins
                    n_bins = min(5, max(2, int(np.sqrt(len(data)))))
                    source_disc = pd.cut(data[source], bins=n_bins, labels=False)
                    target_disc = pd.cut(data[target], bins=n_bins, labels=False)
                    
                    # Calculate TE(source -> target)
                    # TE = H(target_future | target_past) - H(target_future | target_past, source_past)
                    
                    # Create lagged versions
                    target_future = target_disc.iloc[lag:].reset_index(drop=True)
                    target_past = target_disc.iloc[:-lag].reset_index(drop=True)
                    source_past = source_disc.iloc[:-lag].reset_index(drop=True)
                    
                    # Calculate conditional entropies
                    h1 = self._conditional_entropy_discrete(target_future, target_past)
                    h2 = self._conditional_entropy_discrete_2d(target_future, target_past, source_past)
                    
                    te = h1 - h2
                    if te > 0:
                        te_sum += te
                        n_targets += 1
                except:
                    pass
        
        return te_sum / n_targets if n_targets > 0 else 0
    
    def _conditional_entropy_discrete(self, X: pd.Series, Y: pd.Series) -> float:
        """Calculate H(X|Y) for discrete variables."""
        joint = pd.crosstab(X, Y)
        p_y = Y.value_counts(normalize=True)
        
        h_xy = 0
        for y_val in Y.unique():
            if y_val in p_y.index and p_y[y_val] > 0:
                mask = Y == y_val
                if mask.sum() > 0:
                    x_given_y = X[mask]
                    px_given_y = x_given_y.value_counts(normalize=True)
                    h_x_given_y = entropy(px_given_y)
                    h_xy += p_y[y_val] * h_x_given_y
        
        return h_xy
    
    def _conditional_entropy_discrete_2d(self, X: pd.Series, Y: pd.Series, Z: pd.Series) -> float:
        """Calculate H(X|Y,Z) for discrete variables."""
        # Create combined condition
        YZ = pd.Series([f"{y}_{z}" for y, z in zip(Y, Z)])
        return self._conditional_entropy_discrete(X, YZ)
    
    def calculate_proportional_dominance(self, data: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate dominance based on proportional presence and elevation above median.
        
        Args:
            data: DataFrame with frame proportions
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with proportional dominance metrics
        """
        results = pd.DataFrame(index=self.frame_names)
        
        if show_progress:
            pbar = tqdm(total=4, desc="Computing proportional dominance", leave=False)
        
        # 1. Mean proportion
        results['mean_proportion'] = data[self.frame_names].mean()
        if show_progress:
            pbar.update(1)
        
        # 2. Elevation above median
        median_props = data[self.frame_names].median()
        overall_median = median_props.median()
        
        for frame in self.frame_names:
            # How much above the overall median
            elevation = (results.loc[frame, 'mean_proportion'] - overall_median) / overall_median if overall_median > 0 else 0
            results.loc[frame, 'elevation_score'] = max(0, elevation)
        if show_progress:
            pbar.update(1)
        
        # 3. Consistency (inverse of coefficient of variation)
        means = data[self.frame_names].mean()
        stds = data[self.frame_names].std()
        cv_scores = stds / means
        cv_scores = cv_scores.fillna(0)  # Handle division by zero
        results['consistency'] = 1 / (1 + cv_scores)
        if show_progress:
            pbar.update(1)
        
        # 4. Percentile rank
        for frame in self.frame_names:
            percentile = percentileofscore(results['mean_proportion'], 
                                         results.loc[frame, 'mean_proportion'])
            results.loc[frame, 'percentile_rank'] = percentile / 100
        if show_progress:
            pbar.update(1)
            pbar.close()
        
        return results
    
    def calculate_dominance_scores(self, data: pd.DataFrame, 
                                 weekly_data: Optional[pd.DataFrame] = None,
                                 show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate comprehensive dominance scores using all methods with equal weights.
        
        Args:
            data: DataFrame with frame proportions
            weekly_data: Optional weekly data for network/causality analysis
            show_progress: Whether to show progress bars
            
        Returns:
            DataFrame with all dominance metrics and composite score
        """
        # Use weekly data if provided, otherwise use main data
        temporal_data = weekly_data if weekly_data is not None else data
        
        if show_progress:
            print(f"\nCalculating dominance scores using 4 methods on {len(temporal_data)} weeks of data:")
            pbar = tqdm(total=4, desc="Overall progress")
        
        # Calculate all metric categories
        info_metrics = self.calculate_information_theory_metrics(temporal_data, show_progress)
        if show_progress:
            pbar.update(1)
            
        network_metrics = self.calculate_network_metrics(temporal_data, show_progress=show_progress)
        if show_progress:
            pbar.update(1)
            
        causality_metrics = self.calculate_causality_metrics(temporal_data, show_progress=show_progress)
        if show_progress:
            pbar.update(1)
            
        prop_metrics = self.calculate_proportional_dominance(temporal_data, show_progress)
        if show_progress:
            pbar.update(1)
            pbar.close()
        
        # Combine all metrics
        all_metrics = pd.concat([info_metrics, network_metrics, 
                               causality_metrics, prop_metrics], axis=1)
        
        # Define metric groups with equal weight
        metric_groups = {
            'information_theory': ['mutual_information', 'entropy_reduction', 'information_content'],
            'network_analysis': ['pagerank', 'pagerank_stability', 
                               'eigenvector_centrality', 'eigenvector_stability'],
            'causality': ['granger_causality', 'transfer_entropy'],
            'proportional': ['mean_proportion', 'elevation_score', 'consistency', 'percentile_rank']
        }
        
        # Normalize metrics within each group
        normalized_metrics = pd.DataFrame(index=self.frame_names)
        
        if show_progress:
            norm_pbar = tqdm(metric_groups.items(), desc="Normalizing metrics", leave=False)
        else:
            norm_pbar = metric_groups.items()
            
        for group_name, metrics in norm_pbar:
            group_data = all_metrics[metrics].copy()
            
            # Normalize each metric to 0-1
            for metric in metrics:
                if metric in group_data.columns:
                    min_val = group_data[metric].min()
                    max_val = group_data[metric].max()
                    if max_val > min_val:
                        group_data[metric] = (group_data[metric] - min_val) / (max_val - min_val)
                    else:
                        group_data[metric] = 0.5
            
            # Calculate group score (mean of normalized metrics)
            normalized_metrics[f'{group_name}_score'] = group_data.mean(axis=1)
        
        # Calculate final dominance score (equal weight to each group)
        normalized_metrics['dominance_score'] = normalized_metrics[[
            'information_theory_score', 'network_analysis_score', 
            'causality_score', 'proportional_score'
        ]].mean(axis=1)
        
        # Add rank
        normalized_metrics['rank'] = normalized_metrics['dominance_score'].rank(ascending=False)
        
        # Combine with raw metrics
        results = pd.concat([all_metrics, normalized_metrics], axis=1)
        
        return results.sort_values('dominance_score', ascending=False)
    
    def determine_dominant_frames(self, dominance_scores: pd.DataFrame) -> Tuple[List[str], Dict]:
        """
        Determine which frames are dominant using objective statistical criteria.
        
        Args:
            dominance_scores: DataFrame with dominance scores
            
        Returns:
            Tuple of (dominant_frames, analysis_details)
        """
        scores = dominance_scores['dominance_score']
        
        # Method 1: Above median method (more appropriate for close scores)
        median_threshold = scores.median()
        median_frames = scores[scores > median_threshold].index.tolist()
        
        # Method 2: Natural break using gradient
        sorted_scores = scores.sort_values(ascending=False)
        gradients = np.diff(sorted_scores.values)
        if len(gradients) > 0:
            max_gradient_idx = np.argmax(np.abs(gradients)) + 1
            natural_break_threshold = sorted_scores.iloc[max_gradient_idx] if max_gradient_idx < len(sorted_scores) else sorted_scores.iloc[-1]
            natural_break_frames = scores[scores >= natural_break_threshold].index.tolist()
        else:
            natural_break_frames = median_frames
            natural_break_threshold = median_threshold
        
        # Method 3: Improved elbow method
        elbow_frames = self._improved_elbow_method(scores)
        
        # Method 4: Statistical significance (mean + 0.5*std)
        mean_score = scores.mean()
        std_score = scores.std()
        significance_threshold = mean_score + 0.5 * std_score
        significant_frames = scores[scores >= significance_threshold].index.tolist()
        
        # Consensus approach with validation
        all_candidates = set(median_frames + natural_break_frames + elbow_frames + significant_frames)
        
        # Count votes for each candidate
        frame_votes = {}
        for frame in all_candidates:
            votes = 0
            if frame in median_frames:
                votes += 1
            if frame in natural_break_frames:
                votes += 1
            if frame in elbow_frames:
                votes += 1
            if frame in significant_frames:
                votes += 1
            frame_votes[frame] = votes
        
        # A frame is dominant if selected by at least 3 methods
        dominant_frames = [frame for frame, votes in frame_votes.items() if votes >= 3]
        
        # Sort by score
        dominant_frames = sorted(dominant_frames, 
                               key=lambda x: scores[x], 
                               reverse=True)
        
        # Validation: ensure reasonable number of dominant frames
        # For small datasets, allow single dominant frame
        if len(dominant_frames) == 0:
            # Fallback to top frame
            dominant_frames = [scores.idxmax()]
        elif len(dominant_frames) > 6:
            # Too many dominant frames, use natural break
            dominant_frames = natural_break_frames[:6]
        
        # Calculate final threshold
        if dominant_frames:
            dominance_threshold = min(scores[f] for f in dominant_frames)
        else:
            dominance_threshold = median_threshold
        
        analysis_details = {
            'method': 'improved multi-criteria consensus',
            'median_threshold': median_threshold,
            'median_frames': median_frames,
            'natural_break_threshold': natural_break_threshold,
            'natural_break_frames': natural_break_frames,
            'significance_threshold': significance_threshold,
            'significant_frames': significant_frames,
            'elbow_frames': elbow_frames,
            'frame_votes': frame_votes,
            'n_dominant': len(dominant_frames),
            'dominance_threshold': dominance_threshold
        }
        
        return dominant_frames, analysis_details
    
    def _kde_clustering(self, scores: pd.Series) -> List[str]:
        """Use kernel density estimation to find natural clusters."""
        scores_array = scores.values.reshape(-1, 1)
        
        # Fit KDE
        kde = gaussian_kde(scores_array.T)
        
        # Find local minima in density
        x_range = np.linspace(scores.min(), scores.max(), 100)
        density = kde(x_range)
        
        # Find local minima
        minima = []
        for i in range(1, len(density) - 1):
            if density[i] < density[i-1] and density[i] < density[i+1]:
                minima.append(x_range[i])
        
        # Use the highest minimum as threshold
        if minima:
            threshold = max(minima)
            return scores[scores > threshold].index.tolist()
        else:
            # No clear clusters, use top tertile
            threshold = scores.quantile(0.67)
            return scores[scores > threshold].index.tolist()
    
    def _improved_elbow_method(self, scores: pd.Series) -> List[str]:
        """Improved elbow method using cumulative variance."""
        sorted_scores = scores.sort_values(ascending=False)
        
        if len(sorted_scores) <= 2:
            return sorted_scores.index.tolist()
        
        # Calculate cumulative proportion
        scores_array = sorted_scores.values
        cumsum = np.cumsum(scores_array) / np.sum(scores_array)
        
        # Find elbow point using distance to diagonal
        n_points = len(cumsum)
        distances = []
        
        for i in range(1, n_points - 1):
            # Point on the curve
            point = np.array([i / n_points, cumsum[i]])
            # Line from start to end
            line_start = np.array([0, 0])
            line_end = np.array([1, 1])
            # Distance from point to line
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            proj_len = np.dot(point_vec, line_vec) / line_len
            proj = line_start + (proj_len / line_len) * line_vec
            dist = np.linalg.norm(point - proj)
            distances.append(dist)
        
        if distances:
            elbow_idx = np.argmax(distances) + 1
            return sorted_scores.iloc[:elbow_idx].index.tolist()
        else:
            return sorted_scores.iloc[:3].index.tolist()  # Default to top 3
    
    def _elbow_method(self, scores: pd.Series) -> List[str]:
        """Find elbow point in sorted scores."""
        sorted_scores = scores.sort_values(ascending=False)
        
        if len(sorted_scores) <= 2:
            return [sorted_scores.index[0]]
        
        # Calculate second derivative
        first_diff = np.diff(sorted_scores.values)
        second_diff = np.diff(first_diff)
        
        # Find maximum curvature (elbow)
        if len(second_diff) > 0:
            elbow_idx = np.argmax(np.abs(second_diff)) + 1
            return sorted_scores.iloc[:elbow_idx].index.tolist()
        else:
            return sorted_scores.iloc[:1].index.tolist()
    
    def analyze_paradigm_composition(self, data: pd.DataFrame, 
                                   temporal_resolution: str = 'overall',
                                   show_progress: bool = True) -> Dict:
        """
        Analyze paradigm composition for a given dataset.
        
        Args:
            data: DataFrame with frame data
            temporal_resolution: 'overall', 'yearly', 'period', 'weekly'
            show_progress: Whether to show progress bars
            
        Returns:
            Dictionary with paradigm analysis
        """
        # Calculate dominance scores
        dominance_scores = self.calculate_dominance_scores(data, show_progress=show_progress)
        
        # Determine dominant frames
        dominant_frames, details = self.determine_dominant_frames(dominance_scores)
        
        # Calculate paradigm metrics
        n_dominant = len(dominant_frames)
        paradigm_type = self._classify_paradigm(n_dominant)
        
        # Shannon diversity
        shannon = data[self.frame_names].apply(
            lambda row: entropy(row[row > 0]) / np.log(len(row[row > 0])) 
            if len(row[row > 0]) > 1 else 0,
            axis=1
        ).mean()
        
        # Paradigm concentration
        paradigm_concentration = sum(data[dominant_frames].mean()) if dominant_frames else 0
        
        # Paradigm coherence (correlation among dominant frames)
        if len(dominant_frames) > 1:
            dominant_corr = data[dominant_frames].corr()
            coherence = dominant_corr.values[np.triu_indices_from(dominant_corr.values, k=1)].mean()
        else:
            coherence = 1.0
        
        return {
            'temporal_resolution': temporal_resolution,
            'n_dominant_frames': n_dominant,
            'dominant_frames': dominant_frames,
            'paradigm_type': paradigm_type,
            'dominance_scores': dominance_scores,
            'shannon_diversity': shannon,
            'paradigm_concentration': paradigm_concentration,
            'paradigm_coherence': coherence,
            'analysis_details': details
        }
    
    def _classify_paradigm(self, n_frames: int) -> str:
        """Classify paradigm based on number of dominant frames."""
        if n_frames == 0:
            return "No dominant paradigm"
        elif n_frames == 1:
            return "Mono-paradigm"
        elif n_frames == 2:
            return "Dual-paradigm"
        elif n_frames == 3:
            return "Triple-paradigm"
        elif n_frames == 4:
            return "Quad-paradigm"
        else:
            return f"Multi-paradigm ({n_frames} frames)"
    
    def calculate_weekly_paradigm_evolution(self, data: pd.DataFrame, 
                                          show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate paradigm composition for each week.
        Adaptive window size based on data availability.
        
        Args:
            data: DataFrame with weekly frame data
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with weekly paradigm composition
        """
        # Adaptive window size
        data_length = len(data)
        if data_length < 12:
            window_size = data_length  # Use all available data
        elif data_length < 52:
            window_size = 12  # 3 months
        else:
            window_size = 12  # Default 3 months
        
        # Generate window indices
        window_indices = []
        if data_length <= window_size:
            # Single window for very small datasets
            for i in range(data_length):
                window_indices.append((0, data_length, data.iloc[i]['date']))
        else:
            # Sliding window
            for i in range(window_size, data_length):
                window_indices.append((i-window_size, i, data.iloc[i]['date']))
        
        # Progress tracking setup
        if show_progress:
            print(f"\nProcessing {len(window_indices)} weekly windows (window size: {window_size})...")
            pbar = tqdm(total=len(window_indices), desc="Analyzing weekly paradigms")
        
        # Parallel processing of windows
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for start, end, week_date in window_indices:
                window_data = data.iloc[start:end]
                future = executor.submit(self._analyze_window_paradigm, 
                                       window_data, week_date, self.frame_names)
                futures.append(future)
            
            # Collect results
            weekly_results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    weekly_results.append(result)
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # Sort by date
        weekly_results.sort(key=lambda x: x['date'])
        
        return pd.DataFrame(weekly_results)
    
    def _analyze_window_paradigm(self, window_data: pd.DataFrame, 
                                week_date: pd.Timestamp, 
                                frame_names: List[str]) -> Dict:
        """Analyze paradigm for a single window."""
        try:
            # Create new analyzer instance for subprocess
            analyzer = ParadigmDominanceAnalyzer(frame_names=frame_names, n_workers=1)
            analysis = analyzer.analyze_paradigm_composition(window_data, 'weekly', show_progress=False)
            
            # Create paradigm vector
            paradigm_vector = np.zeros(len(frame_names))
            for j, frame in enumerate(frame_names):
                if frame in analysis['dominant_frames']:
                    paradigm_vector[j] = analysis['dominance_scores'].loc[frame, 'dominance_score']
            
            # Normalize
            if paradigm_vector.sum() > 0:
                paradigm_vector = paradigm_vector / paradigm_vector.sum()
            
            result = {
                'date': week_date,
                'n_dominant': analysis['n_dominant_frames'],
                'dominant_frames': ','.join(analysis['dominant_frames']),
                'paradigm_type': analysis['paradigm_type'],
                'paradigm_concentration': analysis['paradigm_concentration'],
                'paradigm_coherence': analysis['paradigm_coherence']
            }
            
            # Add paradigm vector components
            for j, frame in enumerate(frame_names):
                result[f'paradigm_{frame}'] = paradigm_vector[j]
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing window paradigm: {e}")
            return None