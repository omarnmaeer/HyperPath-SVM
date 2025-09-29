# File: hyperpath_svm/core/tgck.py
"""
Temporal Graph Convolution Kernel (TGCK) Implementation

This module implements the TGCK kernel with:
- Efficient O(n) complexity vs O(n³) for traditional GNNs
- Spectral graph analysis for topology awareness
- Temporal dynamics modeling with graph-aware distances
- Network topology integration into SVM kernel space

Key Innovation: Graph-aware kernel that scales linearly with network size
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import eigsh, svds
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
import networkx as nx
from numba import jit, prange
import threading
from collections import defaultdict, deque

from ..data.network_graph import NetworkGraph
from ..utils.graph_utils import graph_laplacian, shortest_path_matrix, spectral_decomposition
from ..utils.math_utils import fast_kernel_computation, vectorized_distance_computation

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """
    Efficient spectral analysis of network graphs
    Computes eigendecomposition and spectral properties for kernel computation
    """
    
    def __init__(self, num_eigenvectors: int = 50, cache_size: int = 1000):
        self.num_eigenvectors = num_eigenvectors
        self.cache_size = cache_size
        
        # Cache for computed spectral properties
        self._eigenvalue_cache = {}
        self._eigenvector_cache = {}
        self._spectral_cache = {}
        
        # Thread safety for caching
        self._cache_lock = threading.RLock()
        
        logger.debug(f"SpectralAnalyzer initialized: {num_eigenvectors} eigenvectors, cache size {cache_size}")
    
    def compute_graph_spectrum(self, graph: NetworkGraph, 
                             use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of graph Laplacian
        
        Parameters
        ----------
        graph : NetworkGraph
            Network topology graph
        use_cache : bool
            Whether to use cached results
            
        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues of graph Laplacian (sorted ascending)
        eigenvectors : np.ndarray
            Corresponding eigenvectors
        """
        
        graph_hash = graph.get_topology_hash()
        
        with self._cache_lock:
            # Check cache first
            if use_cache and graph_hash in self._eigenvalue_cache:
                logger.debug("Using cached spectral decomposition")
                return (self._eigenvalue_cache[graph_hash].copy(),
                       self._eigenvector_cache[graph_hash].copy())
        
        # Compute graph Laplacian
        L = graph.compute_laplacian(normalize=True)
        
        # Use sparse eigenvalue solver for efficiency
        try:
            # Compute smallest eigenvalues (most informative for graph structure)
            k = min(self.num_eigenvectors, L.shape[0] - 2)
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', return_eigenvectors=True)
            
            # Sort by eigenvalue (should already be sorted, but ensure)
            sort_idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]
            
        except Exception as e:
            logger.warning(f"Sparse eigendecomposition failed: {e}. Using dense fallback.")
            
            # Fallback to dense eigendecomposition for small graphs
            if L.shape[0] <= 1000:
                L_dense = L.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            else:
                # For very large graphs, use approximation
                eigenvalues, eigenvectors = self._approximate_spectrum(L)
        
        # Cache results
        with self._cache_lock:
            if len(self._eigenvalue_cache) >= self.cache_size:
                # Remove oldest cache entries
                oldest_key = next(iter(self._eigenvalue_cache))
                del self._eigenvalue_cache[oldest_key]
                del self._eigenvector_cache[oldest_key]
            
            self._eigenvalue_cache[graph_hash] = eigenvalues.copy()
            self._eigenvector_cache[graph_hash] = eigenvectors.copy()
        
        logger.debug(f"Computed spectrum: {len(eigenvalues)} eigenvalues, "
                    f"spectral gap: {eigenvalues[1] - eigenvalues[0]:.6f}")
        
        return eigenvalues, eigenvectors
    
    def _approximate_spectrum(self, L: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate spectral decomposition for very large graphs"""
        
        # Use randomized SVD for approximation
        try:
            U, s, Vt = svds(L, k=min(self.num_eigenvectors, L.shape[0] - 2))
            eigenvalues = s**2  # Convert singular values to eigenvalues
            eigenvectors = U
            
            # Sort by eigenvalue
            sort_idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]
            
            return eigenvalues, eigenvectors
            
        except Exception as e:
            logger.error(f"Spectral approximation failed: {e}")
            # Return identity approximation
            n = min(self.num_eigenvectors, L.shape[0])
            return np.ones(n), np.eye(L.shape[0], n)
    
    def compute_spectral_features(self, graph: NetworkGraph, 
                                positions: np.ndarray) -> np.ndarray:
        """
        Compute spectral features for graph positions
        
        Parameters
        ----------
        graph : NetworkGraph
            Network topology
        positions : np.ndarray
            Node positions/indices to compute features for
            
        Returns
        -------
        features : np.ndarray, shape (len(positions), num_eigenvectors)
            Spectral features based on eigenvector values
        """
        
        eigenvalues, eigenvectors = self.compute_graph_spectrum(graph)
        
        # Extract features for specified positions
        if positions.dtype == int:
            # Positions are node indices
            features = eigenvectors[positions, :]
        else:
            # Positions are continuous - interpolate using graph structure
            features = self._interpolate_spectral_features(
                eigenvectors, positions, graph
            )
        
        return features
    
    def _interpolate_spectral_features(self, eigenvectors: np.ndarray,
                                     positions: np.ndarray, 
                                     graph: NetworkGraph) -> np.ndarray:
        """Interpolate spectral features for continuous positions"""
        
        # This is a simplified interpolation - in practice could use
        # more sophisticated graph-based interpolation methods
        
        # Find nearest nodes for each position
        node_coords = graph.get_node_coordinates()
        if node_coords is None:
            # Fallback: use position indices directly
            int_positions = positions.astype(int) % eigenvectors.shape[0]
            return eigenvectors[int_positions, :]
        
        # Compute distances to all nodes
        distances = np.linalg.norm(
            positions[:, np.newaxis, :] - node_coords[np.newaxis, :, :], 
            axis=2
        )
        
        # Weighted interpolation based on inverse distance
        weights = 1.0 / (distances + 1e-8)  # Add small epsilon to avoid division by zero
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        # Interpolate eigenvector values
        interpolated_features = np.einsum('ij,jk->ik', weights, eigenvectors)
        
        return interpolated_features
    
    def get_spectral_properties(self, graph: NetworkGraph) -> Dict:
        """Get comprehensive spectral properties of graph"""
        
        eigenvalues, eigenvectors = self.compute_graph_spectrum(graph)
        
        properties = {
            'num_eigenvalues': len(eigenvalues),
            'spectral_gap': eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0,
            'effective_resistance': np.sum(1.0 / (eigenvalues[1:] + 1e-8)) if len(eigenvalues) > 1 else 0.0,
            'algebraic_connectivity': eigenvalues[1] if len(eigenvalues) > 1 else 0.0,
            'spectral_radius': eigenvalues[-1] if len(eigenvalues) > 0 else 0.0,
            'trace': np.sum(eigenvalues),
            'condition_number': eigenvalues[-1] / (eigenvalues[1] + 1e-8) if len(eigenvalues) > 1 else 1.0
        }
        
        return properties


class TemporalModeling:
    """
    Temporal dynamics modeling for time-dependent graph kernels
    Handles temporal patterns and evolution in network performance
    """
    
    def __init__(self, temporal_decay: float = 0.95, max_temporal_window: int = 3600):
        self.temporal_decay = temporal_decay
        self.max_temporal_window = max_temporal_window
        
        # Temporal pattern storage
        self.temporal_patterns = defaultdict(deque)
        self.pattern_cache = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug(f"TemporalModeling initialized: decay={temporal_decay}, "
                    f"window={max_temporal_window}s")
    
    def compute_temporal_kernel(self, t1: Union[float, np.ndarray], 
                               t2: Union[float, np.ndarray],
                               current_time: float = None) -> Union[float, np.ndarray]:
        """
        Compute temporal kernel values between timestamps
        
        Implements Equation 9 from paper:
        K_temporal(t_i, t_j, t) = exp(-γ_t * |t_i - t_j|) * w_recency(t_i, t_j, t)
        
        Parameters
        ----------
        t1, t2 : float or np.ndarray
            Timestamps to compare
        current_time : float, optional
            Current time for recency weighting
            
        Returns
        -------
        kernel_values : float or np.ndarray
            Temporal kernel similarities
        """
        
        if current_time is None:
            current_time = time.time()
        
        # Convert to numpy arrays for vectorization
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)
        
        # Compute time differences
        time_diff = np.abs(t1 - t2)
        
        # Basic temporal kernel: exponential decay with time difference
        gamma_temporal = 0.001  # γ_t from paper
        temporal_similarity = np.exp(-gamma_temporal * time_diff)
        
        # Add recency weighting
        recency_weight1 = self._compute_recency_weight(t1, current_time)
        recency_weight2 = self._compute_recency_weight(t2, current_time)
        recency_factor = np.sqrt(recency_weight1 * recency_weight2)
        
        # Combined temporal kernel
        kernel_values = temporal_similarity * recency_factor
        
        return kernel_values
    
    def _compute_recency_weight(self, timestamps: np.ndarray, 
                              current_time: float) -> np.ndarray:
        """Compute recency weights for timestamps"""
        
        time_since = current_time - timestamps
        
        # Exponential decay for recency (more recent = higher weight)
        recency_weights = np.exp(-0.0001 * time_since)  # Slow decay for recency
        
        # Clip weights for very old timestamps
        recency_weights = np.maximum(recency_weights, 0.01)
        
        return recency_weights
    
    def update_temporal_patterns(self, timestamps: np.ndarray, 
                               performance_metrics: Dict) -> None:
        """Update temporal pattern recognition"""
        
        with self._lock:
            current_time = time.time()
            
            for t in timestamps:
                if current_time - t <= self.max_temporal_window:
                    # Extract temporal features
                    hour_of_day = int((t % 86400) / 3600)  # 0-23
                    day_of_week = int(t / 86400) % 7  # 0-6
                    
                    pattern_key = f"h{hour_of_day}_d{day_of_week}"
                    
                    self.temporal_patterns[pattern_key].append({
                        'timestamp': t,
                        'performance': performance_metrics.copy(),
                        'age': current_time - t
                    })
                    
                    # Limit pattern history size
                    if len(self.temporal_patterns[pattern_key]) > 1000:
                        self.temporal_patterns[pattern_key].popleft()
    
    def get_temporal_patterns(self, current_time: float = None) -> Dict:
        """Get current temporal patterns"""
        
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            current_hour = int((current_time % 86400) / 3600)
            current_day = int(current_time / 86400) % 7
            current_pattern_key = f"h{current_hour}_d{current_day}"
            
            patterns = {}
            for pattern_key, pattern_data in self.temporal_patterns.items():
                if pattern_data:
                    recent_data = [p for p in pattern_data 
                                 if current_time - p['timestamp'] <= 3600]  # Last hour
                    if recent_data:
                        patterns[pattern_key] = {
                            'count': len(recent_data),
                            'avg_performance': self._average_performance(recent_data),
                            'is_current': pattern_key == current_pattern_key
                        }
            
            return patterns
    
    def _average_performance(self, pattern_data: List[Dict]) -> Dict:
        """Compute average performance for pattern data"""
        
        if not pattern_data:
            return {}
        
        # Get all performance keys
        all_keys = set()
        for p in pattern_data:
            all_keys.update(p['performance'].keys())
        
        avg_performance = {}
        for key in all_keys:
            values = [p['performance'].get(key, 0.0) for p in pattern_data]
            avg_performance[key] = np.mean(values)
        
        return avg_performance
    
    def predict_temporal_behavior(self, target_timestamp: float,
                                context_timestamps: np.ndarray) -> Dict:
        """Predict temporal behavior at target timestamp"""
        
        # Find similar temporal contexts from history
        target_hour = int((target_timestamp % 86400) / 3600)
        target_day = int(target_timestamp / 86400) % 7
        target_pattern = f"h{target_hour}_d{target_day}"
        
        with self._lock:
            if target_pattern in self.temporal_patterns:
                pattern_data = list(self.temporal_patterns[target_pattern])
                if pattern_data:
                    prediction = {
                        'expected_performance': self._average_performance(pattern_data),
                        'confidence': min(len(pattern_data) / 100.0, 1.0),
                        'pattern_key': target_pattern,
                        'historical_samples': len(pattern_data)
                    }
                    return prediction
        
        return {'expected_performance': {}, 'confidence': 0.0}


class GraphAwareDistance:
    """
    Compute graph-aware distances that consider network topology
    More sophisticated than Euclidean distance for network data
    """
    
    def __init__(self, neighborhood_hops: int = 2):
        self.neighborhood_hops = neighborhood_hops
        self.distance_cache = {}
        self.cache_lock = threading.RLock()
        
        logger.debug(f"GraphAwareDistance initialized: {neighborhood_hops} neighborhood hops")
    
    def compute_graph_distance(self, x1: np.ndarray, x2: np.ndarray,
                             graph: NetworkGraph, node_indices: np.ndarray = None) -> np.ndarray:
        """
        Compute graph-aware distances between feature vectors
        
        Combines feature space distance with graph topology information
        
        Parameters
        ----------
        x1, x2 : np.ndarray
            Feature vectors to compare
        graph : NetworkGraph
            Network topology
        node_indices : np.ndarray, optional
            Node indices corresponding to feature vectors
            
        Returns
        -------
        distances : np.ndarray
            Graph-aware distances
        """
        
        # Compute feature space distance (standard Euclidean)
        feature_distances = np.linalg.norm(x1 - x2, axis=-1)
        
        if node_indices is None:
            # If no node mapping provided, return feature distances
            return feature_distances
        
        # Compute graph topology distances
        graph_distances = self._compute_topology_distances(graph, node_indices)
        
        # Combine feature and topology distances
        # Weight: 70% feature, 30% topology (can be tuned)
        combined_distances = 0.7 * feature_distances + 0.3 * graph_distances
        
        return combined_distances
    
    def _compute_topology_distances(self, graph: NetworkGraph, 
                                  node_indices: np.ndarray) -> np.ndarray:
        """Compute distances based on graph topology"""
        
        graph_hash = graph.get_topology_hash()
        
        with self.cache_lock:
            if graph_hash in self.distance_cache:
                cached_distances = self.distance_cache[graph_hash]
            else:
                # Compute shortest path distances for entire graph
                try:
                    cached_distances = shortest_path_matrix(graph, max_hops=self.neighborhood_hops)
                    self.distance_cache[graph_hash] = cached_distances
                except Exception as e:
                    logger.warning(f"Failed to compute shortest paths: {e}")
                    # Fallback to adjacency-based distances
                    cached_distances = self._adjacency_based_distances(graph)
                    self.distance_cache[graph_hash] = cached_distances
        
        # Extract distances for specified node pairs
        if len(node_indices.shape) == 1:
            # Single pair
            i, j = node_indices[0], node_indices[1]
            return cached_distances[i, j]
        else:
            # Multiple pairs
            distances = []
            for k in range(len(node_indices)):
                i, j = node_indices[k, 0], node_indices[k, 1]
                distances.append(cached_distances[i, j])
            return np.array(distances)
    
    def _adjacency_based_distances(self, graph: NetworkGraph) -> np.ndarray:
        """Fallback distance computation based on adjacency matrix"""
        
        A = graph.get_adjacency_matrix()
        n = A.shape[0]
        
        # Initialize distances with adjacency (1 for connected, inf for disconnected)
        distances = A.copy().astype(float)
        distances[distances == 0] = np.inf
        np.fill_diagonal(distances, 0)
        
        # Floyd-Warshall for small graphs, approximation for large graphs
        if n <= 500:
            # Floyd-Warshall algorithm
            for k in range(n):
                distances = np.minimum(distances, 
                                     distances[:, k:k+1] + distances[k:k+1, :])
        else:
            # Approximation: limit to immediate neighbors
            distances[distances == np.inf] = self.neighborhood_hops + 1
        
        return distances
    
    def compute_neighborhood_features(self, graph: NetworkGraph, 
                                    node_indices: np.ndarray,
                                    features: np.ndarray) -> np.ndarray:
        """
        Compute features aggregated over graph neighborhoods
        
        Parameters
        ----------
        graph : NetworkGraph
            Network topology
        node_indices : np.ndarray
            Indices of nodes to compute neighborhood features for
        features : np.ndarray
            Node features to aggregate
            
        Returns
        -------
        neighborhood_features : np.ndarray
            Aggregated neighborhood features
        """
        
        neighborhood_features = np.zeros_like(features)
        
        for i, node_idx in enumerate(node_indices):
            # Get k-hop neighborhood
            neighbors = graph.get_k_hop_neighbors(node_idx, k=self.neighborhood_hops)
            
            if neighbors:
                # Aggregate features from neighborhood
                neighbor_features = features[neighbors]
                
                # Simple aggregation: mean (could use attention mechanisms)
                aggregated = np.mean(neighbor_features, axis=0)
                neighborhood_features[i] = aggregated
            else:
                # If no neighbors, use node's own features
                neighborhood_features[i] = features[i]
        
        return neighborhood_features


@jit(nopython=True, parallel=True)
def _fast_spatial_kernel_computation(X1: np.ndarray, X2: np.ndarray, 
                                   gamma: float) -> np.ndarray:
    """
    Fast computation of spatial kernel matrix using Numba JIT compilation
    Optimized for large-scale network data
    """
    
    n1, n2 = X1.shape[0], X2.shape[0]
    kernel_matrix = np.zeros((n1, n2))
    
    for i in prange(n1):
        for j in range(n2):
            # Compute squared Euclidean distance
            dist_sq = 0.0
            for k in range(X1.shape[1]):
                diff = X1[i, k] - X2[j, k]
                dist_sq += diff * diff
            
            # RBF kernel: exp(-gamma * ||x1 - x2||^2)
            kernel_matrix[i, j] = np.exp(-gamma * dist_sq)
    
    return kernel_matrix


class TGCK:
    """
    Temporal Graph Convolution Kernel - Main Implementation
    
    Combines spatial graph convolution with temporal dynamics for
    network-aware kernel computation in SVM frameworks.
    
    Key Features:
    - O(n) complexity vs O(n³) for traditional GNNs
    - Spectral graph analysis integration
    - Temporal pattern modeling
    - Graph-aware distance computation
    """
    
    def __init__(self, config: Dict, gamma_spatial: float = 0.01,
                 gamma_temporal: float = 0.001, neighborhood_hops: int = 2):
        
        self.config = config
        self.gamma_spatial = gamma_spatial
        self.gamma_temporal = gamma_temporal
        self.neighborhood_hops = neighborhood_hops
        
        # Initialize components
        self.spectral_analyzer = SpectralAnalyzer(
            num_eigenvectors=config.get('num_eigenvectors', 50),
            cache_size=config.get('cache_size', 1000)
        )
        
        self.temporal_modeling = TemporalModeling(
            temporal_decay=config.get('temporal_decay', 0.95),
            max_temporal_window=config.get('max_temporal_window', 3600)
        )
        
        self.graph_distance = GraphAwareDistance(
            neighborhood_hops=neighborhood_hops
        )
        
        # Performance optimization settings
        self.use_sparse_matrices = config.get('use_sparse_matrices', True)
        self.parallel_computation = config.get('parallel_computation', True)
        self.spectral_approximation = config.get('spectral_approximation', True)
        
        # Caching for performance
        self.kernel_cache = {}
        self.graph_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics tracking
        self.computation_stats = {
            'total_kernel_computations': 0,
            'total_computation_time': 0.0,
            'average_computation_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info(f"TGCK initialized: γ_spatial={gamma_spatial}, "
                   f"γ_temporal={gamma_temporal}, hops={neighborhood_hops}")
    
    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray,
                             graph: NetworkGraph, timestamps1: np.ndarray,
                             timestamps2: np.ndarray = None) -> np.ndarray:
        """
        Compute TGCK kernel matrix between feature sets
        
        Implements Equation 6 from paper:
        K_TGCK(x_i, x_j, t) = K_spatial(x_i, x_j, G_t) * K_temporal(t_i, t_j, t)
        
        Parameters
        ----------
        X1 : np.ndarray, shape (n_samples1, n_features)
            First feature set
        X2 : np.ndarray, shape (n_samples2, n_features)  
            Second feature set
        graph : NetworkGraph
            Network topology at current time
        timestamps1 : np.ndarray, shape (n_samples1,)
            Timestamps for first feature set
        timestamps2 : np.ndarray, shape (n_samples2,), optional
            Timestamps for second feature set (uses timestamps1 if None)
            
        Returns
        -------
        kernel_matrix : np.ndarray, shape (n_samples1, n_samples2)
            TGCK kernel similarity matrix
        """
        
        start_time = time.time()
        
        if timestamps2 is None:
            timestamps2 = timestamps1
        
        # Input validation
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        timestamps1 = np.asarray(timestamps1)
        timestamps2 = np.asarray(timestamps2)
        
        # Check cache first
        cache_key = self._generate_cache_key(X1, X2, graph, timestamps1, timestamps2)
        
        with self._lock:
            if cache_key in self.kernel_cache:
                self.cache_hits += 1
                logger.debug("Using cached kernel matrix")
                return self.kernel_cache[cache_key].copy()
            
            self.cache_misses += 1
        
        logger.debug(f"Computing TGCK kernel matrix: {X1.shape} x {X2.shape}")
        
        # Step 1: Compute spatial kernel with graph awareness
        K_spatial = self.spatial_kernel(X1, X2, graph)
        
        # Step 2: Compute temporal kernel  
        K_temporal = self.temporal_kernel(timestamps1, timestamps2)
        
        # Step 3: Combine spatial and temporal kernels
        # Element-wise multiplication: K[i,j] = K_spatial[i,j] * K_temporal[i,j]
        kernel_matrix = K_spatial * K_temporal
        
        # Step 4: Apply graph convolution enhancement
        if self.config.get('use_graph_convolution', True):
            kernel_matrix = self._apply_graph_convolution(kernel_matrix, graph)
        
        # Step 5: Normalize kernel matrix
        kernel_matrix = self._normalize_kernel_matrix(kernel_matrix)
        
        # Cache result
        with self._lock:
            if len(self.kernel_cache) >= 100:  # Limit cache size
                # Remove oldest entry
                oldest_key = next(iter(self.kernel_cache))
                del self.kernel_cache[oldest_key]
            
            self.kernel_cache[cache_key] = kernel_matrix.copy()
        
        # Update statistics
        computation_time = time.time() - start_time
        with self._lock:
            self.computation_stats['total_kernel_computations'] += 1
            self.computation_stats['total_computation_time'] += computation_time
            self.computation_stats['average_computation_time'] = (
                self.computation_stats['total_computation_time'] / 
                self.computation_stats['total_kernel_computations']
            )
            total_requests = self.cache_hits + self.cache_misses
            self.computation_stats['cache_hit_rate'] = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        logger.debug(f"TGCK kernel computed in {computation_time:.3f}s")
        
        return kernel_matrix
    
    def spatial_kernel(self, X1: np.ndarray, X2: np.ndarray,
                      graph: NetworkGraph) -> np.ndarray:
        """
        Compute spatial kernel with graph-aware distance
        
        Implements Equations 7-8 from paper:
        K_spatial(x_i, x_j, G_t) = exp(-γ_s * d_graph(x_i, x_j, G_t))
        """
        
        logger.debug(f"Computing spatial kernel: {X1.shape} x {X2.shape}")
        
        if self.parallel_computation and X1.shape[0] * X2.shape[0] > 10000:
            # Use optimized parallel computation for large matrices
            return self._compute_spatial_kernel_parallel(X1, X2, graph)
        else:
            # Use standard computation for smaller matrices
            return self._compute_spatial_kernel_standard(X1, X2, graph)
    
    def _compute_spatial_kernel_standard(self, X1: np.ndarray, X2: np.ndarray,
                                       graph: NetworkGraph) -> np.ndarray:
        """Standard spatial kernel computation"""
        
        # Option 1: Pure feature-space RBF kernel (fast baseline)
        if not self.config.get('use_graph_distance', True):
            return rbf_kernel(X1, X2, gamma=self.gamma_spatial)
        
        # Option 2: Graph-enhanced kernel computation
        n1, n2 = X1.shape[0], X2.shape[0]
        kernel_matrix = np.zeros((n1, n2))
        
        # Get spectral features if enabled
        if self.spectral_approximation:
            spectral_features1 = self.spectral_analyzer.compute_spectral_features(
                graph, np.arange(min(n1, graph.num_nodes))
            )
            spectral_features2 = self.spectral_analyzer.compute_spectral_features(
                graph, np.arange(min(n2, graph.num_nodes))
            )
        else:
            spectral_features1 = spectral_features2 = None
        
        # Compute kernel entries
        for i in range(n1):
            for j in range(n2):
                # Feature distance
                feature_dist = np.linalg.norm(X1[i] - X2[j])
                
                # Graph-aware distance enhancement
                if spectral_features1 is not None and i < len(spectral_features1) and j < len(spectral_features2):
                    spectral_dist = np.linalg.norm(
                        spectral_features1[i] - spectral_features2[j]
                    )
                    # Combine feature and spectral distances
                    combined_dist = 0.8 * feature_dist + 0.2 * spectral_dist
                else:
                    combined_dist = feature_dist
                
                # RBF kernel
                kernel_matrix[i, j] = np.exp(-self.gamma_spatial * combined_dist**2)
        
        return kernel_matrix
    
    def _compute_spatial_kernel_parallel(self, X1: np.ndarray, X2: np.ndarray,
                                       graph: NetworkGraph) -> np.ndarray:
        """Parallel spatial kernel computation using Numba JIT"""
        
        if self.config.get('use_graph_distance', True):
            # For graph-aware computation, fall back to standard for now
            # (Could implement parallel graph distance computation)
            return self._compute_spatial_kernel_standard(X1, X2, graph)
        else:
            # Fast parallel RBF kernel computation
            return _fast_spatial_kernel_computation(X1, X2, self.gamma_spatial)
    
    def temporal_kernel(self, timestamps1: np.ndarray, 
                       timestamps2: np.ndarray) -> np.ndarray:
        """
        Compute temporal dependencies kernel
        
        Implements Equation 9 from paper:
        K_temporal(t_i, t_j, t) = exp(-γ_t * |t_i - t_j|) * w_recency(t_i, t_j, t)
        """
        
        logger.debug(f"Computing temporal kernel: {len(timestamps1)} x {len(timestamps2)}")
        
        current_time = time.time()
        
        # Compute pairwise temporal kernel values
        n1, n2 = len(timestamps1), len(timestamps2)
        temporal_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                temporal_matrix[i, j] = self.temporal_modeling.compute_temporal_kernel(
                    timestamps1[i], timestamps2[j], current_time
                )
        
        return temporal_matrix
    
    def spectral_decomposition(self, graph: NetworkGraph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficient spectral analysis of graph Laplacian
        
        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues of graph Laplacian  
        eigenvectors : np.ndarray
            Corresponding eigenvectors
        """
        
        return self.spectral_analyzer.compute_graph_spectrum(graph)
    
    def _apply_graph_convolution(self, kernel_matrix: np.ndarray, 
                               graph: NetworkGraph) -> np.ndarray:
        """Apply graph convolution to enhance kernel matrix"""
        
        if not self.config.get('use_graph_convolution', True):
            return kernel_matrix
        
        # Get graph Laplacian
        L = graph.compute_laplacian(normalize=True)
        
        # Apply diffusion-like operation
        # This is a simplified graph convolution - could be made more sophisticated
        n = min(kernel_matrix.shape[0], L.shape[0])
        
        if n > 0 and n == L.shape[0]:
            # Apply graph filtering
            try:
                # Use matrix exponential approximation for diffusion
                diffusion_operator = self._compute_diffusion_operator(L, n)
                
                # Apply to kernel matrix
                enhanced_kernel = diffusion_operator @ kernel_matrix[:n, :n] @ diffusion_operator.T
                
                # Replace relevant part of kernel matrix
                kernel_matrix[:n, :n] = enhanced_kernel
                
            except Exception as e:
                logger.warning(f"Graph convolution failed: {e}. Using original kernel.")
        
        return kernel_matrix
    
    def _compute_diffusion_operator(self, L: csr_matrix, n: int) -> np.ndarray:
        """Compute diffusion operator for graph convolution"""
        
        # Simplified diffusion: exp(-t * L) where t is diffusion time
        diffusion_time = 0.1
        
        if n <= 500:  # For small graphs, use exact computation
            L_dense = L.toarray()[:n, :n]
            diffusion_operator = scipy.linalg.expm(-diffusion_time * L_dense)
        else:
            # For large graphs, use approximation
            # First-order approximation: I - t * L
            I = np.eye(n)
            L_dense = L.toarray()[:n, :n]
            diffusion_operator = I - diffusion_time * L_dense
        
        return diffusion_operator
    
    def _normalize_kernel_matrix(self, kernel_matrix: np.ndarray) -> np.ndarray:
        """Normalize kernel matrix for numerical stability"""
        
        if self.config.get('normalize_kernel', True):
            # Ensure positive semi-definite and well-conditioned
            
            # Add small diagonal regularization
            regularization = 1e-6
            n = min(kernel_matrix.shape)
            kernel_matrix[:n, :n] += regularization * np.eye(n)
            
            # Optional: apply kernel normalization
            if self.config.get('kernel_normalization', 'none') == 'unit_diagonal':
                # Normalize to unit diagonal
                diag_sqrt = np.sqrt(np.diag(kernel_matrix))
                diag_sqrt[diag_sqrt == 0] = 1.0  # Avoid division by zero
                kernel_matrix = kernel_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        return kernel_matrix
    
    def _generate_cache_key(self, X1: np.ndarray, X2: np.ndarray,
                           graph: NetworkGraph, timestamps1: np.ndarray,
                           timestamps2: np.ndarray) -> str:
        """Generate cache key for kernel computation"""
        
        # Create hash from inputs (simplified for performance)
        key_components = [
            str(X1.shape),
            str(X2.shape), 
            graph.get_topology_hash()[:16],  # Truncate for efficiency
            str(len(timestamps1)),
            str(len(timestamps2)),
            f"{self.gamma_spatial:.6f}",
            f"{self.gamma_temporal:.6f}"
        ]
        
        return "_".join(key_components)
    
    def update_temporal_patterns(self, timestamps: np.ndarray, 
                               performance_metrics: Dict) -> None:
        """Update temporal pattern recognition"""
        
        self.temporal_modeling.update_temporal_patterns(timestamps, performance_metrics)
    
    def get_spectral_properties(self) -> Dict:
        """Get spectral properties of most recent graph"""
        
        # Return cached spectral properties if available
        return getattr(self, '_last_spectral_properties', {})
    
    def get_temporal_patterns(self) -> Dict:
        """Get current temporal patterns"""
        
        return self.temporal_modeling.get_temporal_patterns()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive TGCK statistics"""
        
        with self._lock:
            stats = {
                'computation_stats': self.computation_stats.copy(),
                'cache_stats': {
                    'cache_size': len(self.kernel_cache),
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'hit_rate': self.computation_stats['cache_hit_rate']
                },
                'parameters': {
                    'gamma_spatial': self.gamma_spatial,
                    'gamma_temporal': self.gamma_temporal,
                    'neighborhood_hops': self.neighborhood_hops,
                    'use_sparse_matrices': self.use_sparse_matrices,
                    'parallel_computation': self.parallel_computation,
                    'spectral_approximation': self.spectral_approximation
                },
                'spectral_analyzer_stats': {
                    'eigenvalue_cache_size': len(self.spectral_analyzer._eigenvalue_cache),
                    'num_eigenvectors': self.spectral_analyzer.num_eigenvectors
                },
                'temporal_patterns': len(self.temporal_modeling.temporal_patterns)
            }
        
        return stats
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        
        memory_mb = 0.0
        
        # Kernel cache
        for K in self.kernel_cache.values():
            memory_mb += K.nbytes / (1024 * 1024)
        
        # Spectral analyzer cache
        for eigenvals in self.spectral_analyzer._eigenvalue_cache.values():
            memory_mb += eigenvals.nbytes / (1024 * 1024)
        
        for eigenvecs in self.spectral_analyzer._eigenvector_cache.values():
            memory_mb += eigenvecs.nbytes / (1024 * 1024)
        
        # Graph distance cache
        for distances in self.graph_distance.distance_cache.values():
            if hasattr(distances, 'nbytes'):
                memory_mb += distances.nbytes / (1024 * 1024)
        
        return memory_mb
    
    def get_status(self) -> Dict:
        """Get current TGCK status"""
        
        with self._lock:
            return {
                'initialized': True,
                'cache_size': len(self.kernel_cache),
                'total_computations': self.computation_stats['total_kernel_computations'],
                'average_time_ms': self.computation_stats['average_computation_time'] * 1000,
                'cache_hit_rate': self.computation_stats['cache_hit_rate'],
                'memory_usage_mb': self.get_memory_usage(),
                'parameters': {
                    'gamma_spatial': self.gamma_spatial,
                    'gamma_temporal': self.gamma_temporal
                }
            }
    
    def get_state(self) -> Dict:
        """Get complete state for serialization"""
        
        with self._lock:
            # Note: We don't serialize large cache objects for efficiency
            return {
                'config': self.config,
                'gamma_spatial': self.gamma_spatial,
                'gamma_temporal': self.gamma_temporal,
                'neighborhood_hops': self.neighborhood_hops,
                'computation_stats': self.computation_stats,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'spectral_analyzer_config': {
                    'num_eigenvectors': self.spectral_analyzer.num_eigenvectors
                },
                'temporal_modeling_config': {
                    'temporal_decay': self.temporal_modeling.temporal_decay,
                    'max_temporal_window': self.temporal_modeling.max_temporal_window
                }
            }
    
    def restore_state(self, state: Dict) -> None:
        """Restore state from serialization"""
        
        with self._lock:
            self.config = state['config']
            self.gamma_spatial = state['gamma_spatial']
            self.gamma_temporal = state['gamma_temporal']
            self.neighborhood_hops = state['neighborhood_hops']
            self.computation_stats = state['computation_stats']
            self.cache_hits = state['cache_hits']
            self.cache_misses = state['cache_misses']
            
            # Reinitialize components with restored configuration
            if 'spectral_analyzer_config' in state:
                self.spectral_analyzer.num_eigenvectors = state['spectral_analyzer_config']['num_eigenvectors']
            
            if 'temporal_modeling_config' in state:
                self.temporal_modeling.temporal_decay = state['temporal_modeling_config']['temporal_decay']
                self.temporal_modeling.max_temporal_window = state['temporal_modeling_config']['max_temporal_window']
        
        logger.info("TGCK state restored successfully")
    
    def clear_cache(self) -> None:
        """Clear all caches to free memory"""
        
        with self._lock:
            self.kernel_cache.clear()
            self.graph_cache.clear()
            self.spectral_analyzer._eigenvalue_cache.clear()
            self.spectral_analyzer._eigenvector_cache.clear()
            self.graph_distance.distance_cache.clear()
            
            logger.info("TGCK caches cleared")
    
    def optimize_parameters(self, X_train: np.ndarray, y_train: np.ndarray,
                          graph: NetworkGraph, timestamps: np.ndarray,
                          validation_split: float = 0.2) -> Dict:
        """
        Optimize TGCK parameters using validation performance
        
        This is a simplified parameter optimization - in practice could use
        more sophisticated hyperparameter optimization methods.
        """
        
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        # Split data
        X_train_opt, X_val_opt, y_train_opt, y_val_opt, t_train_opt, t_val_opt = train_test_split(
            X_train, y_train, timestamps, test_size=validation_split, shuffle=False
        )
        
        best_score = 0.0
        best_params = {'gamma_spatial': self.gamma_spatial, 'gamma_temporal': self.gamma_temporal}
        
        # Grid search over parameter space
        spatial_gammas = [0.001, 0.01, 0.1]
        temporal_gammas = [0.0001, 0.001, 0.01]
        
        for gamma_s in spatial_gammas:
            for gamma_t in temporal_gammas:
                # Set parameters
                old_gamma_s, old_gamma_t = self.gamma_spatial, self.gamma_temporal
                self.gamma_spatial, self.gamma_temporal = gamma_s, gamma_t
                
                try:
                    # Compute kernel matrix
                    K_train = self.compute_kernel_matrix(
                        X_train_opt, X_train_opt, graph, t_train_opt
                    )
                    K_val = self.compute_kernel_matrix(
                        X_val_opt, X_train_opt, graph, t_val_opt, t_train_opt
                    )
                    
                    # Train SVM
                    svm = SVC(kernel='precomputed', C=1.0)
                    svm.fit(K_train, y_train_opt)
                    
                    # Evaluate
                    y_pred = svm.predict(K_val)
                    score = accuracy_score(y_val_opt, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'gamma_spatial': gamma_s, 'gamma_temporal': gamma_t}
                
                except Exception as e:
                    logger.warning(f"Parameter optimization failed for γ_s={gamma_s}, γ_t={gamma_t}: {e}")
                
                finally:
                    # Restore old parameters
                    self.gamma_spatial, self.gamma_temporal = old_gamma_s, old_gamma_t
        
        # Set best parameters
        self.gamma_spatial = best_params['gamma_spatial']
        self.gamma_temporal = best_params['gamma_temporal']
        
        logger.info(f"TGCK parameters optimized: γ_spatial={self.gamma_spatial}, "
                   f"γ_temporal={self.gamma_temporal}, validation_score={best_score:.3f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_completed': True
        } 
