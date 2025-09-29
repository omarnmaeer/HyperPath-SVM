# File: hyperpath_svm/data/network_graph.py
"""
Network Graph Representation for HyperPath-SVM

This module implements dynamic network graph G_t = (V, E_t, W_t) with:
- Time-varying edges E_t with connectivity changes
- Feature vectors W_t: bandwidth, latency, loss, jitter, security
- Efficient graph operations for path analysis
- Spectral analysis and topology awareness
- Support for real-time network updates

Key Features: Dynamic topology with temporal evolution tracking
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import hashlib
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import shortest_path, connected_components
import pickle

logger = logging.getLogger(__name__)


@dataclass
class NetworkNode:
  
    
    # Node identification
    node_id: str
    node_type: str  # "core_router", "edge_router", "switch", "server", etc.
    
    # Geographic/logical position
    coordinates: Optional[Tuple[float, float]] = None
    region: str = field(default="unknown")
    zone: str = field(default="default")
    
    # Node capabilities
    processing_capacity: float = field(default=1.0)
    buffer_size: int = field(default=1000000)  # bytes
    max_connections: int = field(default=100)
    
    # Current state
    active: bool = field(default=True)
    load: float = field(default=0.0)  # 0.0 to 1.0
    last_seen: float = field(default_factory=time.time)
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields"""
        if self.coordinates is None:
            # Generate random coordinates if not provided
            self.coordinates = (np.random.uniform(-180, 180), np.random.uniform(-90, 90))
    
    def update_metrics(self, new_metrics: Dict[str, float], timestamp: float = None):
        """Update node metrics"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics.update(new_metrics)
        self.last_seen = timestamp
        
        # Update load if provided
        if 'cpu_load' in new_metrics:
            self.load = new_metrics['cpu_load']
    
    def is_active(self, timeout: float = 300) -> bool:
      
        if not self.active:
            return False
        
        time_since_seen = time.time() - self.last_seen
        return time_since_seen <= timeout


@dataclass
class NetworkEdge:
    """Represents a network link between two nodes"""
    
    # Edge identification
    source: str
    target: str
    edge_type: str = field(default="physical")  # "physical", "logical", "tunnel"
    bidirectional: bool = field(default=True)
    
    # Physical properties
    bandwidth_mbps: float = field(default=100.0)
    latency_ms: float = field(default=1.0)
    packet_loss_rate: float = field(default=0.0)
    jitter_ms: float = field(default=0.1)
    
    # Security and reliability
    security_score: float = field(default=5.0)  # 0-10 scale
    reliability: float = field(default=0.99)  # 0.0-1.0
    
    # Current state
    active: bool = field(default=True)
    utilization: float = field(default=0.0)  # 0.0-1.0
    last_updated: float = field(default_factory=time.time)
    
    # Dynamic metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_metrics(self, new_metrics: Dict[str, float], timestamp: float = None):
        """Update edge metrics"""
        if timestamp is None:
            timestamp = time.time()
        
        # Update core metrics if provided
        if 'bandwidth_mbps' in new_metrics:
            self.bandwidth_mbps = new_metrics['bandwidth_mbps']
        if 'latency_ms' in new_metrics:
            self.latency_ms = new_metrics['latency_ms']
        if 'packet_loss_rate' in new_metrics:
            self.packet_loss_rate = new_metrics['packet_loss_rate']
        if 'jitter_ms' in new_metrics:
            self.jitter_ms = new_metrics['jitter_ms']
        if 'utilization' in new_metrics:
            self.utilization = new_metrics['utilization']
        
        self.metrics.update(new_metrics)
        self.last_updated = timestamp
    
    def get_weight(self, weight_type: str = "latency") -> float:
        """Get edge weight for different routing metrics"""
        
        if weight_type == "latency":
            return self.latency_ms
        elif weight_type == "bandwidth":
            return 1.0 / (self.bandwidth_mbps + 1e-6)  # Inverse for shortest path
        elif weight_type == "loss":
            return self.packet_loss_rate
        elif weight_type == "composite":
            # Composite metric combining multiple factors
            normalized_latency = self.latency_ms / 100.0  # Normalize to ~0-1
            normalized_loss = self.packet_loss_rate * 1000  # Scale up
            normalized_util = self.utilization
            
            return 0.4 * normalized_latency + 0.3 * normalized_loss + 0.3 * normalized_util
        else:
            return 1.0  # Default uniform weight
    
    def is_active(self, timeout: float = 300) -> bool:
        """Check if edge is considered active"""
        if not self.active:
            return False
        
        time_since_update = time.time() - self.last_updated
        return time_since_update <= timeout
    
    def get_edge_id(self) -> str:
       
        if self.bidirectional:
            # Sort endpoints for consistent ID regardless of direction
            endpoints = sorted([self.source, self.target])
            return f"{endpoints[0]}-{endpoints[1]}"
        else:
            return f"{self.source}->{self.target}"


class NetworkTopology:
    """
    Manages network topology structure and operations
    Provides efficient graph operations and analysis
    """
    
    def __init__(self):
        # Graph storage using adjacency representation
        self._nodes: Dict[str, NetworkNode] = {}
        self._edges: Dict[str, NetworkEdge] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # Cached computations
        self._adjacency_matrix_cache: Optional[csr_matrix] = None
        self._laplacian_cache: Optional[csr_matrix] = None
        self._shortest_paths_cache: Optional[np.ndarray] = None
        self._cache_timestamp: float = 0.0
        self._cache_validity = 60.0  # Cache valid for 60 seconds
        
        # NetworkX graph for advanced operations
        self._nx_graph: Optional[nx.Graph] = None
        self._nx_graph_timestamp: float = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
    def add_node(self, node: NetworkNode) -> bool:
       
        with self._lock:
            if node.node_id in self._nodes:
                # Update existing node
                self._nodes[node.node_id] = node
            else:
                # Add new node
                self._nodes[node.node_id] = node
                self._adjacency[node.node_id] = set()
            
            self._invalidate_cache()
            return True
    
    def add_edge(self, edge: NetworkEdge) -> bool:
        
        with self._lock:
            # Ensure both endpoints exist
            if edge.source not in self._nodes or edge.target not in self._nodes:
                logger.warning(f"Cannot add edge {edge.get_edge_id()}: missing endpoints")
                return False
            
            # Add edge
            edge_id = edge.get_edge_id()
            self._edges[edge_id] = edge
            
            # Update adjacency
            self._adjacency[edge.source].add(edge.target)
            if edge.bidirectional:
                self._adjacency[edge.target].add(edge.source)
            
            self._invalidate_cache()
            return True
    
    def remove_node(self, node_id: str) -> bool:
        
        with self._lock:
            if node_id not in self._nodes:
                return False
            
            # Remove all edges connected to this node
            edges_to_remove = []
            for edge_id, edge in self._edges.items():
                if edge.source == node_id or edge.target == node_id:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                del self._edges[edge_id]
            
            # Remove from adjacency
            neighbors = list(self._adjacency[node_id])
            for neighbor in neighbors:
                self._adjacency[neighbor].discard(node_id)
            del self._adjacency[node_id]
            
            # Remove node
            del self._nodes[node_id]
            
            self._invalidate_cache()
            return True
    
    def remove_edge(self, source: str, target: str) -> bool:
        
        with self._lock:
            # Try both directions for edge ID
            edge_id1 = f"{min(source, target)}-{max(source, target)}"
            edge_id2 = f"{source}->{target}"
            edge_id3 = f"{target}->{source}"
            
            edge_removed = False
            
            for edge_id in [edge_id1, edge_id2, edge_id3]:
                if edge_id in self._edges:
                    del self._edges[edge_id]
                    edge_removed = True
                    break
            
            if edge_removed:
                # Update adjacency
                self._adjacency[source].discard(target)
                self._adjacency[target].discard(source)
                self._invalidate_cache()
                return True
            
            return False
    
    def get_node(self, node_id: str) -> Optional[NetworkNode]:
       
        return self._nodes.get(node_id)
    
    def get_edge(self, source: str, target: str) -> Optional[NetworkEdge]:
       
        
        # Try different edge ID formats
        edge_ids = [
            f"{min(source, target)}-{max(source, target)}",
            f"{source}->{target}",
            f"{target}->{source}"
        ]
        
        for edge_id in edge_ids:
            if edge_id in self._edges:
                return self._edges[edge_id]
        
        return None
    
    def get_neighbors(self, node_id: str) -> List[str]:
      
        return list(self._adjacency.get(node_id, set()))
    
    def get_k_hop_neighbors(self, node_id: str, k: int = 2) -> Set[str]:
       
        
        if node_id not in self._nodes:
            return set()
        
        visited = set()
        current_level = {node_id}
        
        for _ in range(k):
            next_level = set()
            for node in current_level:
                for neighbor in self._adjacency.get(node, set()):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            current_level = next_level
            
            if not current_level:
                break
        
        return visited
    
    def _invalidate_cache(self):
       
        self._adjacency_matrix_cache = None
        self._laplacian_cache = None
        self._shortest_paths_cache = None
        self._nx_graph = None
        self._cache_timestamp = time.time()
    
    def _is_cache_valid(self) -> bool:
       
        return time.time() - self._cache_timestamp < self._cache_validity
    
    def get_adjacency_matrix(self, weight_type: str = "uniform") -> csr_matrix:
       
        
        with self._lock:
            if self._adjacency_matrix_cache is not None and self._is_cache_valid():
                return self._adjacency_matrix_cache
            
            node_list = sorted(self._nodes.keys())
            n = len(node_list)
            node_to_idx = {node: i for i, node in enumerate(node_list)}
            
            # Build adjacency matrix
            adj_matrix = lil_matrix((n, n))
            
            for edge in self._edges.values():
                if edge.source in node_to_idx and edge.target in node_to_idx:
                    i = node_to_idx[edge.source]
                    j = node_to_idx[edge.target]
                    
                    weight = edge.get_weight(weight_type) if weight_type != "uniform" else 1.0
                    adj_matrix[i, j] = weight
                    
                    if edge.bidirectional:
                        adj_matrix[j, i] = weight
            
            self._adjacency_matrix_cache = adj_matrix.tocsr()
            return self._adjacency_matrix_cache
    
    def get_degree_matrix(self) -> csr_matrix:
      
        
        adj_matrix = self.get_adjacency_matrix()
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        degree_matrix = csr_matrix(np.diag(degrees))
        
        return degree_matrix
    
    def compute_laplacian(self, normalize: bool = True) -> csr_matrix:
       
        
        with self._lock:
            if self._laplacian_cache is not None and self._is_cache_valid():
                return self._laplacian_cache
            
            adj_matrix = self.get_adjacency_matrix()
            degree_matrix = self.get_degree_matrix()
            
            if normalize:
                # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
                degrees = np.array(degree_matrix.diagonal()).flatten()
                
                # Avoid division by zero
                degrees_sqrt_inv = np.zeros_like(degrees)
                nonzero_mask = degrees > 1e-10
                degrees_sqrt_inv[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
                
                D_sqrt_inv = csr_matrix(np.diag(degrees_sqrt_inv))
                I = csr_matrix(np.eye(adj_matrix.shape[0]))
                
                laplacian = I - D_sqrt_inv @ adj_matrix @ D_sqrt_inv
            else:
                # Unnormalized Laplacian: L = D - A
                laplacian = degree_matrix - adj_matrix
            
            self._laplacian_cache = laplacian
            return self._laplacian_cache
    
    def compute_shortest_paths(self, weight_type: str = "latency") -> np.ndarray:
        
        
        with self._lock:
            if self._shortest_paths_cache is not None and self._is_cache_valid():
                return self._shortest_paths_cache
            
            adj_matrix = self.get_adjacency_matrix(weight_type)
            
            # Use scipy's shortest path algorithm
            distances = shortest_path(adj_matrix, method='auto', directed=False)
            
            self._shortest_paths_cache = distances
            return self._shortest_paths_cache
    
    def get_path(self, source: str, target: str, weight_type: str = "latency") -> Optional[List[str]]:
       
        
        nx_graph = self._get_networkx_graph(weight_type)
        
        try:
            path = nx.shortest_path(nx_graph, source, target, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_all_paths(self, source: str, target: str, cutoff: int = None) -> List[List[str]]:
       
        
        nx_graph = self._get_networkx_graph()
        
        try:
            paths = list(nx.all_simple_paths(nx_graph, source, target, cutoff=cutoff))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def _get_networkx_graph(self, weight_type: str = "uniform") -> nx.Graph:
       
        
        current_time = time.time()
        
        if (self._nx_graph is not None and 
            current_time - self._nx_graph_timestamp < self._cache_validity):
            return self._nx_graph
        
        # Build NetworkX graph
        if any(not edge.bidirectional for edge in self._edges.values()):
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        for node_id, node in self._nodes.items():
            G.add_node(node_id, **{
                'type': node.node_type,
                'coordinates': node.coordinates,
                'active': node.active,
                'load': node.load
            })
        
        # Add edges
        for edge in self._edges.values():
            weight = edge.get_weight(weight_type)
            G.add_edge(edge.source, edge.target, 
                      weight=weight,
                      bandwidth=edge.bandwidth_mbps,
                      latency=edge.latency_ms,
                      loss=edge.packet_loss_rate,
                      active=edge.active)
        
        self._nx_graph = G
        self._nx_graph_timestamp = current_time
        
        return G
    
    def analyze_connectivity(self) -> Dict[str, Any]:
       
        
        G = self._get_networkx_graph()
        
        analysis = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G) if isinstance(G, nx.Graph) else nx.is_strongly_connected(G),
            'num_components': nx.number_connected_components(G) if isinstance(G, nx.Graph) else nx.number_strongly_connected_components(G)
        }
        
        # Additional metrics for connected graphs
        if analysis['is_connected']:
            try:
                analysis['diameter'] = nx.diameter(G)
                analysis['average_shortest_path_length'] = nx.average_shortest_path_length(G)
                analysis['clustering_coefficient'] = nx.average_clustering(G)
            except Exception as e:
                logger.warning(f"Could not compute advanced connectivity metrics: {e}")
        
        return analysis
    
    def get_central_nodes(self, centrality_type: str = "betweenness", top_k: int = 10) -> List[Tuple[str, float]]:
        
        
        G = self._get_networkx_graph()
        
        if centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(G)
        elif centrality_type == "degree":
            centrality = nx.degree_centrality(G)
        elif centrality_type == "eigenvector":
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                centrality = nx.degree_centrality(G)  # Fallback
        else:
            centrality = nx.degree_centrality(G)  # Default
        
        # Sort by centrality score
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:top_k]
    
    @property
    def num_nodes(self) -> int:
      
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
       
        return len(self._edges)
    
    @property
    def density(self) -> float:
      
        n = self.num_nodes
        if n <= 1:
            return 0.0
        
        max_edges = n * (n - 1) / 2
        return self.num_edges / max_edges if max_edges > 0 else 0.0
    
    def get_node_ids(self) -> List[str]:
      
        return list(self._nodes.keys())
    
    def get_edge_list(self) -> List[NetworkEdge]:
      
        return list(self._edges.values())


class NetworkGraph:
  
    
    def __init__(self, initial_nodes: List[NetworkNode] = None,
                 initial_edges: List[NetworkEdge] = None):
        
        # Core topology
        self.topology = NetworkTopology()
        
        # Temporal tracking
        self.timeline: deque = deque(maxlen=10000)  # Track topology changes
        self.creation_time = time.time()
        
        # Feature management
        self._node_features: Dict[str, np.ndarray] = {}
        self._edge_features: Dict[str, np.ndarray] = {}
        self._global_features: Dict[str, float] = {}
        
        # Performance caching
        self._topology_hash: Optional[str] = None
        self._hash_timestamp: float = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize with provided data
        if initial_nodes:
            for node in initial_nodes:
                self.add_node(node)
        
        if initial_edges:
            for edge in initial_edges:
                self.add_edge(edge)
        
        logger.info(f"NetworkGraph initialized: {self.num_nodes} nodes, {self.num_edges} edges")
    
    def add_node(self, node: NetworkNode, timestamp: float = None) -> bool:
        "
        
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            success = self.topology.add_node(node)
            
            if success:
                # Record topology change
                self.timeline.append({
                    'timestamp': timestamp,
                    'action': 'add_node',
                    'node_id': node.node_id,
                    'node_type': node.node_type
                })
                
                self._invalidate_hash()
                
                logger.debug(f"Added node {node.node_id} of type {node.node_type}")
            
            return success
    
    def add_edge(self, edge: NetworkEdge, timestamp: float = None) -> bool:
       
        
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            success = self.topology.add_edge(edge)
            
            if success:
                # Record topology change
                self.timeline.append({
                    'timestamp': timestamp,
                    'action': 'add_edge',
                    'source': edge.source,
                    'target': edge.target,
                    'edge_type': edge.edge_type
                })
                
                self._invalidate_hash()
                
                logger.debug(f"Added edge {edge.source} -> {edge.target}")
            
            return success
    
    def remove_edges(self, edge_list: List[Tuple[str, str]], timestamp: float = None) -> int:
        
        
        if timestamp is None:
            timestamp = time.time()
        
        removed_count = 0
        
        with self._lock:
            for source, target in edge_list:
                if self.topology.remove_edge(source, target):
                    removed_count += 1
                    
                    # Record topology change
                    self.timeline.append({
                        'timestamp': timestamp,
                        'action': 'remove_edge',
                        'source': source,
                        'target': target,
                        'reason': 'link_failure'
                    })
            
            if removed_count > 0:
                self._invalidate_hash()
                logger.info(f"Removed {removed_count} failed links")
        
        return removed_count
    
    def add_edges(self, edge_list: List[NetworkEdge], timestamp: float = None) -> int:

        
        if timestamp is None:
            timestamp = time.time()
        
        added_count = 0
        
        with self._lock:
            for edge in edge_list:
                if self.add_edge(edge, timestamp):
                    added_count += 1
        
        if added_count > 0:
            logger.info(f"Added {added_count} new links")
        
        return added_count
    
    def update_edge_metrics(self, edge_updates: Dict[Tuple[str, str], Dict[str, float]], 
                           timestamp: float = None) -> int:
      
        
        if timestamp is None:
            timestamp = time.time()
        
        updated_count = 0
        
        with self._lock:
            for (source, target), metrics in edge_updates.items():
                edge = self.topology.get_edge(source, target)
                if edge:
                    edge.update_metrics(metrics, timestamp)
                    updated_count += 1
        
        return updated_count
    
    def set_node_features(self, node_id: str, features: np.ndarray) -> bool:
        
        
        with self._lock:
            if node_id in self.topology._nodes:
                self._node_features[node_id] = features.copy()
                return True
            return False
    
    def get_node_features(self, node_id: str) -> Optional[np.ndarray]:
       
        return self._node_features.get(node_id)
    
    def set_edge_features(self, source: str, target: str, features: np.ndarray) -> bool:
       
        
        edge = self.topology.get_edge(source, target)
        if edge:
            edge_id = edge.get_edge_id()
            with self._lock:
                self._edge_features[edge_id] = features.copy()
            return True
        return False
    
    def get_edge_features(self, source: str, target: str) -> Optional[np.ndarray]:
        
        edge = self.topology.get_edge(source, target)
        if edge:
            edge_id = edge.get_edge_id()
            return self._edge_features.get(edge_id)
        return None
    
    def extract_paths(self, source: str, destination: str, 
                     max_paths: int = 10, max_length: int = None) -> List[List[str]]:
        
        
        # Get all simple paths up to certain length
        all_paths = self.topology.get_all_paths(source, destination, cutoff=max_length)
        
        # Filter and rank paths
        feasible_paths = []
        
        for path in all_paths:
            if len(feasible_paths) >= max_paths:
                break
            
            # Check path feasibility (all edges active)
            path_feasible = True
            for i in range(len(path) - 1):
                edge = self.topology.get_edge(path[i], path[i + 1])
                if not edge or not edge.is_active():
                    path_feasible = False
                    break
            
            if path_feasible:
                feasible_paths.append(path)
        
        return feasible_paths
    
    def compute_path_features(self, path: List[str]) -> Dict[str, float]:
       
        
        if len(path) < 2:
            return {}
        
        features = {
            'length': len(path) - 1,
            'total_latency': 0.0,
            'min_bandwidth': float('inf'),
            'total_loss': 0.0,
            'avg_jitter': 0.0,
            'min_security': 10.0,
            'avg_utilization': 0.0
        }
        
        edge_count = 0
        
        for i in range(len(path) - 1):
            edge = self.topology.get_edge(path[i], path[i + 1])
            if edge:
                features['total_latency'] += edge.latency_ms
                features['min_bandwidth'] = min(features['min_bandwidth'], edge.bandwidth_mbps)
                features['total_loss'] += edge.packet_loss_rate
                features['avg_jitter'] += edge.jitter_ms
                features['min_security'] = min(features['min_security'], edge.security_score)
                features['avg_utilization'] += edge.utilization
                edge_count += 1
        
        if edge_count > 0:
            features['avg_jitter'] /= edge_count
            features['avg_utilization'] /= edge_count
        
        # Avoid infinite bandwidth
        if features['min_bandwidth'] == float('inf'):
            features['min_bandwidth'] = 0.0
        
        return features
    
    def compute_laplacian(self, normalize: bool = True) -> csr_matrix:
        
        return self.topology.compute_laplacian(normalize)
    
    def get_adjacency_matrix(self, weight_type: str = "uniform") -> csr_matrix:
        
        return self.topology.get_adjacency_matrix(weight_type)
    
    def get_shortest_paths(self, weight_type: str = "latency") -> np.ndarray:
        
        return self.topology.compute_shortest_paths(weight_type)
    
    def get_topology_hash(self) -> str:
        
        
        current_time = time.time()
        
        with self._lock:
            if (self._topology_hash is not None and 
                current_time - self._hash_timestamp < 60.0):  # Cache for 1 minute
                return self._topology_hash
            
            # Generate hash from topology structure
            hash_data = {
                'nodes': sorted(self.topology._nodes.keys()),
                'edges': sorted([edge.get_edge_id() for edge in self.topology._edges.values()]),
                'timestamp': int(current_time / 60)  # Round to minute for stability
            }
            
            hash_str = str(hash_data)
            self._topology_hash = hashlib.md5(hash_str.encode()).hexdigest()
            self._hash_timestamp = current_time
            
            return self._topology_hash
    
    def _invalidate_hash(self):
       
        self._topology_hash = None
        self._hash_timestamp = 0.0
    
    def get_node_coordinates(self) -> Optional[np.ndarray]:
        
        
        nodes = sorted(self.topology._nodes.items())
        coordinates = []
        
        for node_id, node in nodes:
            if node.coordinates:
                coordinates.append(node.coordinates)
            else:
                coordinates.append((0.0, 0.0))  # Default coordinates
        
        return np.array(coordinates) if coordinates else None
    
    def analyze_graph_properties(self) -> Dict[str, Any]:
        
        
        properties = self.topology.analyze_connectivity()
        
        # Add temporal information
        properties['temporal_info'] = {
            'creation_time': self.creation_time,
            'age_seconds': time.time() - self.creation_time,
            'topology_changes': len(self.timeline),
            'last_change': self.timeline[-1]['timestamp'] if self.timeline else None
        }
        
        # Add feature information
        properties['features'] = {
            'nodes_with_features': len(self._node_features),
            'edges_with_features': len(self._edge_features),
            'global_features': len(self._global_features)
        }
        
        return properties
    
    def get_graph_statistics(self) -> Dict[str, Any]:
      
        
        with self._lock:
            stats = {
                'topology': self.analyze_graph_properties(),
                'centrality': {
                    'top_betweenness': self.topology.get_central_nodes('betweenness', 5),
                    'top_degree': self.topology.get_central_nodes('degree', 5)
                },
                'edge_metrics': self._compute_edge_statistics(),
                'node_metrics': self._compute_node_statistics(),
                'timeline_summary': self._get_timeline_summary()
            }
        
        return stats
    
    def _compute_edge_statistics(self) -> Dict[str, float]:
      
        
        edges = self.topology.get_edge_list()
        
        if not edges:
            return {}
        
        bandwidths = [e.bandwidth_mbps for e in edges if e.active]
        latencies = [e.latency_ms for e in edges if e.active]
        utilizations = [e.utilization for e in edges if e.active]
        
        stats = {}
        
        if bandwidths:
            stats['avg_bandwidth_mbps'] = np.mean(bandwidths)
            stats['min_bandwidth_mbps'] = np.min(bandwidths)
            stats['max_bandwidth_mbps'] = np.max(bandwidths)
        
        if latencies:
            stats['avg_latency_ms'] = np.mean(latencies)
            stats['min_latency_ms'] = np.min(latencies)
            stats['max_latency_ms'] = np.max(latencies)
        
        if utilizations:
            stats['avg_utilization'] = np.mean(utilizations)
            stats['max_utilization'] = np.max(utilizations)
        
        stats['active_edges'] = len([e for e in edges if e.active])
        stats['total_edges'] = len(edges)
        
        return stats
    
    def _compute_node_statistics(self) -> Dict[str, Any]:
        
        
        nodes = list(self.topology._nodes.values())
        
        if not nodes:
            return {}
        
        node_types = defaultdict(int)
        loads = []
        
        for node in nodes:
            node_types[node.node_type] += 1
            if node.active:
                loads.append(node.load)
        
        stats = {
            'node_types': dict(node_types),
            'active_nodes': len([n for n in nodes if n.active]),
            'total_nodes': len(nodes)
        }
        
        if loads:
            stats['avg_load'] = np.mean(loads)
            stats['max_load'] = np.max(loads)
            stats['overloaded_nodes'] = len([l for l in loads if l > 0.8])
        
        return stats
    
    def _get_timeline_summary(self) -> Dict[str, Any]:
        
        
        if not self.timeline:
            return {}
        
        actions = defaultdict(int)
        recent_changes = 0
        
        current_time = time.time()
        hour_ago = current_time - 3600
        
        for event in self.timeline:
            actions[event['action']] += 1
            if event['timestamp'] > hour_ago:
                recent_changes += 1
        
        return {
            'total_changes': len(self.timeline),
            'changes_last_hour': recent_changes,
            'action_breakdown': dict(actions),
            'first_change': self.timeline[0]['timestamp'],
            'last_change': self.timeline[-1]['timestamp']
        }
    
    @property
    def num_nodes(self) -> int:
       
        return self.topology.num_nodes
    
    @property
    def num_edges(self) -> int:
        """Get number of edges"""
        return self.topology.num_edges
    
    @property
    def density(self) -> float:
        
        return self.topology.density
    
    @property
    def diameter(self) -> int:
       
        try:
            nx_graph = self.topology._get_networkx_graph()
            if nx.is_connected(nx_graph):
                return nx.diameter(nx_graph)
            else:
                return float('inf')
        except Exception:
            return -1  # Could not compute
    
    @property
    def clustering_coefficient(self) -> float:
      
        try:
            nx_graph = self.topology._get_networkx_graph()
            return nx.average_clustering(nx_graph)
        except Exception:
            return 0.0
    
    def save_to_file(self, filepath: str) -> bool:
        """Save graph to file"""
        
        try:
            graph_data = {
                'nodes': [
                    {
                        'node_id': node.node_id,
                        'node_type': node.node_type,
                        'coordinates': node.coordinates,
                        'active': node.active,
                        'metrics': node.metrics
                    }
                    for node in self.topology._nodes.values()
                ],
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'edge_type': edge.edge_type,
                        'bandwidth_mbps': edge.bandwidth_mbps,
                        'latency_ms': edge.latency_ms,
                        'packet_loss_rate': edge.packet_loss_rate,
                        'active': edge.active
                    }
                    for edge in self.topology._edges.values()
                ],
                'timeline': list(self.timeline),
                'creation_time': self.creation_time
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(graph_data, f)
            
            logger.info(f"Graph saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['NetworkGraph']:
        """Load graph from file"""
        
        try:
            with open(filepath, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Create nodes
            nodes = []
            for node_data in graph_data['nodes']:
                node = NetworkNode(
                    node_id=node_data['node_id'],
                    node_type=node_data['node_type'],
                    coordinates=node_data.get('coordinates'),
                    active=node_data.get('active', True)
                )
                node.metrics = node_data.get('metrics', {})
                nodes.append(node)
            
            # Create edges
            edges = []
            for edge_data in graph_data['edges']:
                edge = NetworkEdge(
                    source=edge_data['source'],
                    target=edge_data['target'],
                    edge_type=edge_data.get('edge_type', 'physical'),
                    bandwidth_mbps=edge_data.get('bandwidth_mbps', 100.0),
                    latency_ms=edge_data.get('latency_ms', 1.0),
                    packet_loss_rate=edge_data.get('packet_loss_rate', 0.0),
                    active=edge_data.get('active', True)
                )
                edges.append(edge)
            
            # Create graph
            graph = cls(nodes, edges)
            graph.timeline = deque(graph_data.get('timeline', []), maxlen=10000)
            graph.creation_time = graph_data.get('creation_time', time.time())
            
            logger.info(f"Graph loaded from {filepath}")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return None
    
    def __repr__(self) -> str:
        """String representation of graph"""
        return (f"NetworkGraph(nodes={self.num_nodes}, edges={self.num_edges}, "
               f"density={self.density:.3f}, age={time.time() - self.creation_time:.1f}s)") 
