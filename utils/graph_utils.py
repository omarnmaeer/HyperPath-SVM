# File: hyperpath_svm/utils/graph_utils.py

"""
Graph Processing and Spectral Analysis Utilities for HyperPath-SVM

This module provides comprehensive graph analysis capabilities specifically designed
for network topology processing, spectral graph theory, and routing optimization.

"""

import numpy as np
import scipy as sp
from scipy import sparse, linalg
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import time
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from .logging_utils import get_logger
from .math_utils import math_utils


@dataclass
class GraphMetrics:
    """Container for comprehensive graph metrics."""
    
    # Basic metrics
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    is_connected: bool = False
    
    # Centrality metrics
    degree_centrality: Optional[np.ndarray] = None
    betweenness_centrality: Optional[np.ndarray] = None
    closeness_centrality: Optional[np.ndarray] = None
    eigenvector_centrality: Optional[np.ndarray] = None
    pagerank: Optional[np.ndarray] = None
    
    # Structural metrics
    clustering_coefficient: float = 0.0
    average_path_length: float = 0.0
    diameter: int = 0
    radius: int = 0
    girth: int = 0  # Length of shortest cycle
    
    # Spectral metrics
    spectral_radius: float = 0.0
    algebraic_connectivity: float = 0.0
    num_connected_components: int = 0
    laplacian_eigenvalues: Optional[np.ndarray] = None
    
    # Network-specific metrics
    assortativity: float = 0.0  # Degree assortativity
    modularity: float = 0.0
    small_world_coefficient: float = 0.0


@dataclass
class SpectralProperties:
    """Container for spectral graph properties."""
    
    # Laplacian matrices
    laplacian: Optional[sparse.csr_matrix] = None
    normalized_laplacian: Optional[sparse.csr_matrix] = None
    signless_laplacian: Optional[sparse.csr_matrix] = None
    
    # Eigenvalues and eigenvectors
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    
    # Spectral properties
    spectral_gap: float = 0.0
    effective_resistance: Optional[np.ndarray] = None
    commute_times: Optional[np.ndarray] = None
    
    # Graph Fourier Transform
    graph_fourier_basis: Optional[np.ndarray] = None
    frequency_response: Optional[np.ndarray] = None


class GraphProcessor:
    """Main class for graph processing and analysis."""
    
    def __init__(self, cache_results: bool = True):
        self.logger = get_logger(__name__)
        self.cache_results = cache_results
        self._cache = {}
        
        # Numerical parameters
        self.eigenvalue_tolerance = 1e-10
        self.max_iterations = 1000
        self.convergence_tolerance = 1e-8
    
    # ==================== Graph Creation and Conversion ====================
    
    def adjacency_to_networkx(self, adjacency: Union[np.ndarray, sparse.csr_matrix],
                             node_attributes: Optional[Dict[int, Dict]] = None,
                             edge_attributes: Optional[Dict[Tuple[int, int], Dict]] = None) -> nx.Graph:
        
        try:
            # Convert sparse to dense if needed for NetworkX
            if sparse.issparse(adjacency):
                adj_dense = adjacency.toarray()
            else:
                adj_dense = adjacency
            
            # Create graph from adjacency matrix
            G = nx.from_numpy_array(adj_dense)
            
            # Add node attributes
            if node_attributes:
                for node_id, attrs in node_attributes.items():
                    if node_id in G.nodes():
                        G.nodes[node_id].update(attrs)
            
            # Add edge attributes
            if edge_attributes:
                for (u, v), attrs in edge_attributes.items():
                    if G.has_edge(u, v):
                        G.edges[u, v].update(attrs)
            
            return G
            
        except Exception as e:
            self.logger.error(f"Adjacency to NetworkX conversion failed: {str(e)}")
            return nx.empty_graph()
    
    def networkx_to_adjacency(self, G: nx.Graph, 
                             weight_attribute: str = 'weight',
                             sparse_format: bool = True) -> Union[np.ndarray, sparse.csr_matrix]:
   
        try:
            # Get adjacency matrix
            if weight_attribute in nx.get_edge_attributes(G, weight_attribute):
                adj_matrix = nx.adjacency_matrix(G, weight=weight_attribute)
            else:
                adj_matrix = nx.adjacency_matrix(G)
            
            if sparse_format:
                return adj_matrix.tocsr()
            else:
                return adj_matrix.toarray()
                
        except Exception as e:
            self.logger.error(f"NetworkX to adjacency conversion failed: {str(e)}")
            return sparse.csr_matrix((0, 0)) if sparse_format else np.array([])
    
    def create_graph_from_coordinates(self, coordinates: np.ndarray, 
                                    connection_radius: float,
       
        try:
            n_nodes = coordinates.shape[0]
            
            if periodic_boundary:
                # Compute distances with periodic boundaries
                distances = self._periodic_distances(coordinates)
            else:
                # Standard Euclidean distances
                distances = squareform(pdist(coordinates, metric='euclidean'))
            
            # Create adjacency matrix
            adjacency = (distances <= connection_radius) & (distances > 0)
            
            return sparse.csr_matrix(adjacency.astype(float))
            
        except Exception as e:
            self.logger.error(f"Graph creation from coordinates failed: {str(e)}")
            return sparse.csr_matrix((0, 0))
    
    def _periodic_distances(self, coordinates: np.ndarray, 
                           box_size: float = 1.0) -> np.ndarray:
       
        n_nodes, n_dims = coordinates.shape
        distances = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                diff = coordinates[i] - coordinates[j]
                # Apply periodic boundary conditions
                diff = diff - box_size * np.round(diff / box_size)
                distances[i, j] = distances[j, i] = np.linalg.norm(diff)
        
        return distances
    
    # ==================== Spectral Graph Analysis ====================
    
    def compute_laplacian_matrix(self, adjacency: Union[np.ndarray, sparse.csr_matrix],
                               laplacian_type: str = 'unnormalized') -> sparse.csr_matrix:
       
        try:
            if not sparse.issparse(adjacency):
                adjacency = sparse.csr_matrix(adjacency)
            
            # Degree matrix
            degrees = np.array(adjacency.sum(axis=1)).flatten()
            D = sparse.diags(degrees, format='csr')
            
            if laplacian_type == 'unnormalized':
                # L = D - A
                laplacian = D - adjacency
                
            elif laplacian_type == 'normalized':
                # L_norm = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
                # Handle zero degrees
                degrees_sqrt_inv = np.zeros_like(degrees)
                nonzero_mask = degrees > 0
                degrees_sqrt_inv[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
                
                D_sqrt_inv = sparse.diags(degrees_sqrt_inv, format='csr')
                laplacian = sparse.eye(adjacency.shape[0]) - D_sqrt_inv @ adjacency @ D_sqrt_inv
                
            elif laplacian_type == 'signless':
                # Q = D + A
                laplacian = D + adjacency
                
            else:
                raise ValueError(f"Unknown Laplacian type: {laplacian_type}")
            
            return laplacian.tocsr()
            
        except Exception as e:
            self.logger.error(f"Laplacian computation failed: {str(e)}")
            return sparse.csr_matrix((0, 0))
    
    def compute_spectral_properties(self, adjacency: Union[np.ndarray, sparse.csr_matrix],
                                  num_eigenvalues: Optional[int] = None) -> SpectralProperties:
        
        try:
            if not sparse.issparse(adjacency):
                adjacency = sparse.csr_matrix(adjacency)
            
            n_nodes = adjacency.shape[0]
            
            # Compute different Laplacian matrices
            laplacian = self.compute_laplacian_matrix(adjacency, 'unnormalized')
            normalized_laplacian = self.compute_laplacian_matrix(adjacency, 'normalized')
            signless_laplacian = self.compute_laplacian_matrix(adjacency, 'signless')
            
            # Determine number of eigenvalues
            if num_eigenvalues is None:
                num_eigenvalues = min(50, n_nodes - 1)  # Reasonable default
            num_eigenvalues = min(num_eigenvalues, n_nodes - 1)
            
            if num_eigenvalues <= 0:
                return SpectralProperties()
            
            # Compute eigenvalues and eigenvectors of normalized Laplacian
            try:
                eigenvalues, eigenvectors = eigsh(normalized_laplacian, 
                                                k=num_eigenvalues, 
                                                which='SM',  # Smallest magnitude
                                                tol=self.eigenvalue_tolerance)
                
                # Sort eigenvalues and eigenvectors
                sort_idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[sort_idx]
                eigenvectors = eigenvectors[:, sort_idx]
                
            except Exception as eig_error:
                self.logger.warning(f"Eigenvalue computation failed: {str(eig_error)}")
                # Fallback to dense computation for small graphs
                if n_nodes <= 500:
                    eigenvalues, eigenvectors = linalg.eigh(normalized_laplacian.toarray())
                else:
                    return SpectralProperties(
                        laplacian=laplacian,
                        normalized_laplacian=normalized_laplacian,
                        signless_laplacian=signless_laplacian
                    )
            
            # Compute spectral gap (difference between first two non-zero eigenvalues)
            nonzero_eigenvals = eigenvalues[eigenvalues > self.eigenvalue_tolerance]
            spectral_gap = nonzero_eigenvals[1] - nonzero_eigenvals[0] if len(nonzero_eigenvals) > 1 else 0.0
            
            # Compute effective resistance matrix
            effective_resistance = self._compute_effective_resistance(
                laplacian, eigenvalues, eigenvectors
            )
            
            # Create spectral properties object
            properties = SpectralProperties(
                laplacian=laplacian,
                normalized_laplacian=normalized_laplacian,
                signless_laplacian=signless_laplacian,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                spectral_gap=spectral_gap,
                effective_resistance=effective_resistance,
                graph_fourier_basis=eigenvectors
            )
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Spectral properties computation failed: {str(e)}")
            return SpectralProperties()
    
    def _compute_effective_resistance(self, laplacian: sparse.csr_matrix,
                                    eigenvalues: np.ndarray,
                                    eigenvectors: np.ndarray) -> np.ndarray:
        """Compute effective resistance matrix."""
        try:
            n_nodes = laplacian.shape[0]
            
            # Remove the zero eigenvalue (if present)
            nonzero_mask = eigenvalues > self.eigenvalue_tolerance
            if not np.any(nonzero_mask):
                return np.zeros((n_nodes, n_nodes))
            
            nonzero_eigenvals = eigenvalues[nonzero_mask]
            nonzero_eigenvecs = eigenvectors[:, nonzero_mask]
            
            # Compute pseudoinverse of Laplacian using eigendecomposition
            # L^+ = sum_{i: lambda_i > 0} (1/lambda_i) * v_i * v_i^T
            L_pinv = np.zeros((n_nodes, n_nodes))
            for i, eigenval in enumerate(nonzero_eigenvals):
                v_i = nonzero_eigenvecs[:, i]
                L_pinv += (1.0 / eigenval) * np.outer(v_i, v_i)
            
            # Effective resistance R_ij = L^+_ii + L^+_jj - 2*L^+_ij
            diagonal = np.diag(L_pinv)
            effective_resistance = (diagonal[:, np.newaxis] + 
                                  diagonal[np.newaxis, :] - 
                                  2 * L_pinv)
            
            return effective_resistance
            
        except Exception as e:
            self.logger.warning(f"Effective resistance computation failed: {str(e)}")
            return np.zeros((laplacian.shape[0], laplacian.shape[0]))
    
    def graph_fourier_transform(self, signal: np.ndarray, 
                               eigenvectors: np.ndarray) -> np.ndarray:
      
        try:
            # Graph Fourier Transform: F[s] = U^T * s
            # where U is the matrix of eigenvectors
            fourier_coefficients = eigenvectors.T @ signal
            return fourier_coefficients
            
        except Exception as e:
            self.logger.error(f"Graph Fourier Transform failed: {str(e)}")
            return np.zeros_like(signal)
    
    def inverse_graph_fourier_transform(self, fourier_coefficients: np.ndarray,
                                      eigenvectors: np.ndarray) -> np.ndarray:
        
        try:
            # Inverse Graph Fourier Transform: s = U * F[s]
            signal = eigenvectors @ fourier_coefficients
            return signal
            
        except Exception as e:
            self.logger.error(f"Inverse Graph Fourier Transform failed: {str(e)}")
            return np.zeros(eigenvectors.shape[0])
    
    # ==================== Graph Algorithms ====================
    
    def all_pairs_shortest_paths(self, adjacency: Union[np.ndarray, sparse.csr_matrix],
                                method: str = 'floyd_warshall') -> np.ndarray:
      
        try:
            if not sparse.issparse(adjacency):
                adjacency = sparse.csr_matrix(adjacency)
            
            n_nodes = adjacency.shape[0]
            
            if method == 'floyd_warshall':
                return self._floyd_warshall(adjacency)
            elif method == 'johnson':
                return self._johnson_algorithm(adjacency)
            elif method == 'dijkstra_all':
                return self._dijkstra_all_pairs(adjacency)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            self.logger.error(f"All-pairs shortest paths failed: {str(e)}")
            return np.full((adjacency.shape[0], adjacency.shape[0]), np.inf)
    
    def _floyd_warshall(self, adjacency: sparse.csr_matrix) -> np.ndarray:
      
        n_nodes = adjacency.shape[0]
        
        # Initialize distance matrix
        distances = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(distances, 0)
        
        # Set initial distances from adjacency matrix
        rows, cols = adjacency.nonzero()
        distances[rows, cols] = adjacency.data
        
        # Floyd-Warshall algorithm
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]
        
        return distances
    
    def _johnson_algorithm(self, adjacency: sparse.csr_matrix) -> np.ndarray:
       
        # This is a simplified version - full implementation would include
        # Bellman-Ford for reweighting and Dijkstra for each vertex
        return self._floyd_warshall(adjacency)  # Fallback
    
    def _dijkstra_all_pairs(self, adjacency: sparse.csr_matrix) -> np.ndarray:
        """Run Dijkstra's algorithm from all vertices."""
        n_nodes = adjacency.shape[0]
        all_distances = np.full((n_nodes, n_nodes), np.inf)
        
        for source in range(n_nodes):
            distances = self._dijkstra_single_source(adjacency, source)
            all_distances[source, :] = distances
        
        return all_distances
    
    def _dijkstra_single_source(self, adjacency: sparse.csr_matrix, 
                               source: int) -> np.ndarray:
        """Dijkstra's algorithm from single source."""
        n_nodes = adjacency.shape[0]
        distances = np.full(n_nodes, np.inf)
        distances[source] = 0
        visited = set()
        
        # Priority queue: (distance, node)
        pq = [(0, source)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Check neighbors
            start_idx = adjacency.indptr[current_node]
            end_idx = adjacency.indptr[current_node + 1]
            
            for idx in range(start_idx, end_idx):
                neighbor = adjacency.indices[idx]
                weight = adjacency.data[idx]
                
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return distances
    
    def compute_centrality_metrics(self, adjacency: Union[np.ndarray, sparse.csr_matrix]) -> Dict[str, np.ndarray]:
      
        try:
            if not sparse.issparse(adjacency):
                adjacency = sparse.csr_matrix(adjacency)
            
            n_nodes = adjacency.shape[0]
            centralities = {}
            
            # Degree centrality
            degrees = np.array(adjacency.sum(axis=1)).flatten()
            centralities['degree'] = degrees / (n_nodes - 1) if n_nodes > 1 else degrees
            
            # Eigenvector centrality (power iteration)
            try:
                eigenvals, eigenvecs = eigsh(adjacency.astype(float), k=1, which='LM')
                centralities['eigenvector'] = np.abs(eigenvecs[:, 0])
            except Exception:
                centralities['eigenvector'] = np.ones(n_nodes) / n_nodes
            
            # PageRank centrality
            centralities['pagerank'] = self._compute_pagerank(adjacency)
            
            # Betweenness centrality (approximate for large graphs)
            centralities['betweenness'] = self._compute_betweenness_centrality(adjacency)
            
            # Closeness centrality
            centralities['closeness'] = self._compute_closeness_centrality(adjacency)
            
            return centralities
            
        except Exception as e:
            self.logger.error(f"Centrality computation failed: {str(e)}")
            n_nodes = adjacency.shape[0]
            return {
                'degree': np.zeros(n_nodes),
                'eigenvector': np.ones(n_nodes) / n_nodes,
                'pagerank': np.ones(n_nodes) / n_nodes,
                'betweenness': np.zeros(n_nodes),
                'closeness': np.zeros(n_nodes)
            }
    
    def _compute_pagerank(self, adjacency: sparse.csr_matrix, 
                         damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
        """Compute PageRank centrality using power iteration."""
        n_nodes = adjacency.shape[0]
        
        # Create transition matrix
        out_degrees = np.array(adjacency.sum(axis=1)).flatten()
        
        # Handle nodes with zero out-degree
        out_degrees = np.where(out_degrees == 0, 1, out_degrees)
        
        # Normalize adjacency matrix
        D_inv = sparse.diags(1.0 / out_degrees, format='csr')
        transition_matrix = adjacency.T @ D_inv
        
        # Initialize PageRank vector
        pagerank = np.ones(n_nodes) / n_nodes
        
        # Power iteration
        for _ in range(max_iter):
            new_pagerank = (damping * transition_matrix @ pagerank + 
                          (1 - damping) / n_nodes)
            
            if np.linalg.norm(new_pagerank - pagerank) < self.convergence_tolerance:
                break
            
            pagerank = new_pagerank
        
        return pagerank
    
    def _compute_betweenness_centrality(self, adjacency: sparse.csr_matrix,
                                      sample_size: Optional[int] = None) -> np.ndarray:
        """Compute betweenness centrality (sampled for large graphs)."""
        n_nodes = adjacency.shape[0]
        betweenness = np.zeros(n_nodes)
        
        # For large graphs, sample a subset of source nodes
        if sample_size is None:
            sample_size = min(n_nodes, 100)  # Reasonable default
        
        sample_nodes = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
        
        for source in sample_nodes:
            # Single-source shortest paths with predecessor tracking
            distances, predecessors = self._dijkstra_with_predecessors(adjacency, source)
            
            # Accumulate betweenness scores
            for target in range(n_nodes):
                if target != source and np.isfinite(distances[target]):
                    # Count paths through each node
                    paths = self._count_paths(source, target, predecessors)
                    for intermediate in paths:
                        if intermediate != source and intermediate != target:
                            betweenness[intermediate] += 1.0
        
        # Normalize
        normalization = (n_nodes - 1) * (n_nodes - 2) / 2 if n_nodes > 2 else 1
        betweenness = betweenness / normalization
        
        return betweenness
    
    def _dijkstra_with_predecessors(self, adjacency: sparse.csr_matrix, 
                                   source: int) -> Tuple[np.ndarray, Dict]:
     
        n_nodes = adjacency.shape[0]
        distances = np.full(n_nodes, np.inf)
        distances[source] = 0
        predecessors = defaultdict(list)
        visited = set()
        
        pq = [(0, source)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Check neighbors
            start_idx = adjacency.indptr[current_node]
            end_idx = adjacency.indptr[current_node + 1]
            
            for idx in range(start_idx, end_idx):
                neighbor = adjacency.indices[idx]
                weight = adjacency.data[idx]
                
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = [current_node]
                        heapq.heappush(pq, (new_dist, neighbor))
                    elif new_dist == distances[neighbor]:
                        predecessors[neighbor].append(current_node)
        
        return distances, predecessors
    
    def _count_paths(self, source: int, target: int, 
                    predecessors: Dict) -> List[int]:
        """Count nodes on shortest paths (simplified)."""
        # This is a simplified version - full implementation would
        # properly count all shortest paths
        path = []
        current = target
        while current != source and current in predecessors:
            if predecessors[current]:
                current = predecessors[current][0]  # Take first predecessor
                path.append(current)
            else:
                break
        return path
    
    def _compute_closeness_centrality(self, adjacency: sparse.csr_matrix) -> np.ndarray:
        """Compute closeness centrality."""
        n_nodes = adjacency.shape[0]
        closeness = np.zeros(n_nodes)
        
        for source in range(n_nodes):
            distances = self._dijkstra_single_source(adjacency, source)
            finite_distances = distances[np.isfinite(distances) & (distances > 0)]
            
            if len(finite_distances) > 0:
                closeness[source] = len(finite_distances) / np.sum(finite_distances)
        
        return closeness
    
    # ==================== Graph Metrics and Properties ====================
    
    def compute_graph_metrics(self, adjacency: Union[np.ndarray, sparse.csr_matrix]) -> GraphMetrics:
       
        try:
            if not sparse.issparse(adjacency):
                adjacency = sparse.csr_matrix(adjacency)
            
            n_nodes, n_edges_double = adjacency.shape[0], adjacency.nnz
            # For undirected graphs, each edge is counted twice
            n_edges = n_edges_double // 2 if self._is_symmetric(adjacency) else n_edges_double
            
            # Basic metrics
            density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
            is_connected = self._check_connectivity(adjacency)
            
            # Compute centrality metrics
            centralities = self.compute_centrality_metrics(adjacency)
            
            # Compute spectral properties
            spectral_props = self.compute_spectral_properties(adjacency, num_eigenvalues=10)
            
            # Structural metrics
            clustering_coeff = self._compute_clustering_coefficient(adjacency)
            avg_path_length, diameter = self._compute_path_metrics(adjacency)
            
            # Network-specific metrics
            assortativity = self._compute_assortativity(adjacency)
            
            # Create metrics object
            metrics = GraphMetrics(
                num_nodes=n_nodes,
                num_edges=n_edges,
                density=density,
                is_connected=is_connected,
                degree_centrality=centralities.get('degree'),
                betweenness_centrality=centralities.get('betweenness'),
                closeness_centrality=centralities.get('closeness'),
                eigenvector_centrality=centralities.get('eigenvector'),
                pagerank=centralities.get('pagerank'),
                clustering_coefficient=clustering_coeff,
                average_path_length=avg_path_length,
                diameter=diameter,
                spectral_radius=np.max(spectral_props.eigenvalues) if spectral_props.eigenvalues is not None else 0,
                algebraic_connectivity=spectral_props.spectral_gap,
                laplacian_eigenvalues=spectral_props.eigenvalues,
                assortativity=assortativity
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Graph metrics computation failed: {str(e)}")
            return GraphMetrics()
    
    def _is_symmetric(self, adjacency: sparse.csr_matrix, tolerance: float = 1e-10) -> bool:
        """Check if adjacency matrix is symmetric."""
        try:
            diff = adjacency - adjacency.T
            return np.max(np.abs(diff.data)) < tolerance
        except Exception:
            return False
    
    def _check_connectivity(self, adjacency: sparse.csr_matrix) -> bool:
        """Check if graph is connected using BFS."""
        try:
            n_nodes = adjacency.shape[0]
            if n_nodes <= 1:
                return True
            
            visited = set()
            queue = deque([0])  # Start from node 0
            visited.add(0)
            
            while queue:
                current = queue.popleft()
                
                # Get neighbors
                start_idx = adjacency.indptr[current]
                end_idx = adjacency.indptr[current + 1]
                
                for idx in range(start_idx, end_idx):
                    neighbor = adjacency.indices[idx]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return len(visited) == n_nodes
            
        except Exception:
            return False
    
    def _compute_clustering_coefficient(self, adjacency: sparse.csr_matrix) -> float:
        """Compute average clustering coefficient."""
        try:
            n_nodes = adjacency.shape[0]
            clustering_coeffs = np.zeros(n_nodes)
            
            for node in range(n_nodes):
                # Get neighbors
                start_idx = adjacency.indptr[node]
                end_idx = adjacency.indptr[node + 1]
                neighbors = adjacency.indices[start_idx:end_idx]
                
                degree = len(neighbors)
                if degree < 2:
                    clustering_coeffs[node] = 0
                    continue
                
                # Count triangles
                triangles = 0
                for i, neighbor_i in enumerate(neighbors):
                    for neighbor_j in neighbors[i+1:]:
                        # Check if neighbor_i and neighbor_j are connected
                        if adjacency[neighbor_i, neighbor_j] > 0:
                            triangles += 1
                
                # Clustering coefficient for this node
                possible_triangles = degree * (degree - 1) / 2
                clustering_coeffs[node] = triangles / possible_triangles if possible_triangles > 0 else 0
            
            return np.mean(clustering_coeffs)
            
        except Exception as e:
            self.logger.warning(f"Clustering coefficient computation failed: {str(e)}")
            return 0.0
    
    def _compute_path_metrics(self, adjacency: sparse.csr_matrix) -> Tuple[float, int]:
        """Compute average path length and diameter."""
        try:
            # For large graphs, sample a subset
            n_nodes = adjacency.shape[0]
            if n_nodes > 1000:
                sample_size = min(100, n_nodes)
                sample_nodes = np.random.choice(n_nodes, sample_size, replace=False)
            else:
                sample_nodes = range(n_nodes)
            
            all_distances = []
            max_distance = 0
            
            for source in sample_nodes:
                distances = self._dijkstra_single_source(adjacency, source)
                finite_distances = distances[np.isfinite(distances) & (distances > 0)]
                
                if len(finite_distances) > 0:
                    all_distances.extend(finite_distances)
                    max_distance = max(max_distance, np.max(finite_distances))
            
            avg_path_length = np.mean(all_distances) if all_distances else 0
            diameter = int(max_distance) if np.isfinite(max_distance) else 0
            
            return avg_path_length, diameter
            
        except Exception as e:
            self.logger.warning(f"Path metrics computation failed: {str(e)}")
            return 0.0, 0
    
    def _compute_assortativity(self, adjacency: sparse.csr_matrix) -> float:
        """Compute degree assortativity coefficient."""
        try:
            degrees = np.array(adjacency.sum(axis=1)).flatten()
            
            # Get edge list
            rows, cols = adjacency.nonzero()
            
            # For undirected graphs, consider each edge only once
            if self._is_symmetric(adjacency):
                edge_mask = rows <= cols
                rows, cols = rows[edge_mask], cols[edge_mask]
            
            if len(rows) == 0:
                return 0.0
            
            # Degree of endpoints
            degree_i = degrees[rows]
            degree_j = degrees[cols]
            
            # Compute assortativity coefficient
            mean_degree = np.mean(degrees)
            
            numerator = np.sum((degree_i - mean_degree) * (degree_j - mean_degree))
            denominator = np.sum((degree_i - mean_degree)**2)
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            self.logger.warning(f"Assortativity computation failed: {str(e)}")
            return 0.0
    
    # ==================== Utility Functions ====================
    
    def cache_result(self, key: str, result: Any):
        """Cache computation result."""
        if self.cache_results:
            self._cache[key] = result
    
    def get_cached_result(self, key: str) -> Any:
        """Retrieve cached result."""
        return self._cache.get(key) if self.cache_results else None
    
    def clear_cache(self):
        """Clear computation cache."""
        self._cache.clear()


# Global instance for easy access
graph_processor = GraphProcessor()


# Convenience functions
def compute_laplacian(adjacency: Union[np.ndarray, sparse.csr_matrix], 
                     normalized: bool = True) -> sparse.csr_matrix:
    """Convenience function for computing Laplacian matrix."""
    laplacian_type = 'normalized' if normalized else 'unnormalized'
    return graph_processor.compute_laplacian_matrix(adjacency, laplacian_type)


def spectral_embedding(adjacency: Union[np.ndarray, sparse.csr_matrix], 
                      n_components: int = 2) -> np.ndarray:
    """Compute spectral embedding of graph."""
    try:
        spectral_props = graph_processor.compute_spectral_properties(adjacency, n_components + 1)
        
        if spectral_props.eigenvectors is not None:
            # Skip the first eigenvector (corresponding to eigenvalue 0)
            return spectral_props.eigenvectors[:, 1:n_components+1]
        else:
            n_nodes = adjacency.shape[0]
            return np.random.random((n_nodes, n_components))
            
    except Exception as e:
        graph_processor.logger.error(f"Spectral embedding failed: {str(e)}")
        n_nodes = adjacency.shape[0]
        return np.random.random((n_nodes, n_components))


def graph_signal_filter(signal: np.ndarray, eigenvalues: np.ndarray, 
                       eigenvectors: np.ndarray, filter_func: callable) -> np.ndarray:
    """Apply a filter to a graph signal using spectral domain."""
    try:
        # Transform to spectral domain
        fourier_coeffs = eigenvectors.T @ signal
        
        # Apply filter
        filtered_coeffs = filter_func(eigenvalues) * fourier_coeffs
        
        # Transform back to vertex domain
        filtered_signal = eigenvectors @ filtered_coeffs
        
        return filtered_signal
        
    except Exception as e:
        graph_processor.logger.error(f"Graph signal filtering failed: {str(e)}")
        return signal


if __name__ == "__main__":
    # Test graph utilities
    logger = get_logger(__name__)
    logger.info("Testing graph processing utilities...")
    
    # Create test graph
    np.random.seed(42)
    n_nodes = 20
    
    # Create random geometric graph
    coordinates = np.random.random((n_nodes, 2))
    adjacency = graph_processor.create_graph_from_coordinates(coordinates, connection_radius=0.3)
    
    logger.info(f"Created test graph: {n_nodes} nodes, {adjacency.nnz // 2} edges")
    
    # Compute graph metrics
    metrics = graph_processor.compute_graph_metrics(adjacency)
    logger.info(f"Graph metrics computed:")
    logger.info(f"  Density: {metrics.density:.4f}")
    logger.info(f"  Connected: {metrics.is_connected}")
    logger.info(f"  Clustering coefficient: {metrics.clustering_coefficient:.4f}")
    logger.info(f"  Average path length: {metrics.average_path_length:.4f}")
    
    # Compute spectral properties
    spectral_props = graph_processor.compute_spectral_properties(adjacency, num_eigenvalues=5)
    logger.info(f"Spectral properties:")
    logger.info(f"  Spectral gap: {spectral_props.spectral_gap:.4f}")
    logger.info(f"  Eigenvalues: {spectral_props.eigenvalues}")
    
    # Test spectral embedding
    embedding = spectral_embedding(adjacency, n_components=2)
    logger.info(f"Spectral embedding: {embedding.shape}")
    
    # Test graph signal processing
    signal = np.random.random(n_nodes)
    if spectral_props.eigenvalues is not None and spectral_props.eigenvectors is not None:
        # Apply low-pass filter
        low_pass_filter = lambda eigenvals: np.exp(-eigenvals)
        filtered_signal = graph_signal_filter(signal, spectral_props.eigenvalues, 
                                            spectral_props.eigenvectors, low_pass_filter)
        logger.info(f"Graph signal filtering: Original norm={np.linalg.norm(signal):.4f}, "
                   f"Filtered norm={np.linalg.norm(filtered_signal):.4f}")
    
    logger.info("Graph processing utilities testing completed!") 
