# File: hyperpath_svm/data/feature_extractor.py
"""
Network Path Feature Extraction

This module implements comprehensive feature extraction for network paths:
- Multi-dimensional network path characteristics
- Temporal aggregation and statistical features
- Graph-aware topological features
- Performance metric normalization
- Real-time feature computation

"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.fft import fft, fftfreq

from .network_graph import NetworkGraph, NetworkNode, NetworkEdge

logger = logging.getLogger(__name__)


@dataclass
class PathFeatures:
    
    
    # Core network metrics
    bandwidth_features: np.ndarray
    latency_features: np.ndarray
    loss_features: np.ndarray
    jitter_features: np.ndarray
    security_features: np.ndarray
    
    # Topological features
    path_length: int
    hop_count: int
    node_types: List[str]
    centrality_scores: np.ndarray
    
    # Temporal features
    temporal_patterns: np.ndarray
    trend_features: np.ndarray
    seasonality_features: np.ndarray
    
    # Statistical aggregations
    statistical_summary: Dict[str, float]
    
    # Combined feature vector
    feature_vector: np.ndarray
    feature_names: List[str]


class NetworkFeatureExtractor:
   
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Feature extraction parameters
        self.temporal_window = config.get('temporal_window', 60)  # seconds
        self.aggregation_methods = config.get('aggregation_methods', ['mean', 'std', 'max', 'min', 'median'])
        self.include_derivatives = config.get('include_derivatives', True)
        self.include_autocorr = config.get('include_autocorr', True)
        
        # Feature dimensions
        self.num_base_features = 5  # bandwidth, latency, loss, jitter, security
        self.num_statistical_features = len(self.aggregation_methods)
        self.num_temporal_features = 8
        self.num_topological_features = 6
        
        # Total feature vector size
        self.total_features = (self.num_base_features * self.num_statistical_features + 
                              self.num_temporal_features + self.num_topological_features)
        
        # Feature normalization parameters
        self.normalization_params = {
            'bandwidth': {'min': 0.1, 'max': 10000.0},  # Mbps
            'latency': {'min': 0.1, 'max': 1000.0},     # ms
            'loss': {'min': 0.0, 'max': 0.1},           # ratio
            'jitter': {'min': 0.0, 'max': 100.0},       # ms
            'security': {'min': 0.0, 'max': 10.0}       # score
        }
        
        # Feature caching for performance
        self.feature_cache = {}
        self.cache_timeout = config.get('cache_timeout', 300)  # 5 minutes
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"NetworkFeatureExtractor initialized: {self.total_features} total features")
    
    def extract_path_features(self, path_nodes: List[str], graph: NetworkGraph,
                            timestamps: np.ndarray, performance_data: Dict[str, np.ndarray]) -> PathFeatures:
       
        
        if len(path_nodes) < 2:
            return self._create_empty_features()
        
        # Extract different feature categories
        bandwidth_features = self._extract_bandwidth_features(performance_data.get('bandwidth', np.array([])))
        latency_features = self._extract_latency_features(performance_data.get('latency', np.array([])))
        loss_features = self._extract_loss_features(performance_data.get('loss', np.array([])))
        jitter_features = self._extract_jitter_features(performance_data.get('jitter', np.array([])))
        security_features = self._extract_security_features(path_nodes, graph)
        
        # Topological features
        topo_features = self._extract_topological_features(path_nodes, graph)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(timestamps, performance_data)
        
        # Statistical summary
        statistical_summary = self._compute_statistical_summary(performance_data)
        
        # Combine all features
        feature_vector, feature_names = self._combine_features(
            bandwidth_features, latency_features, loss_features,
            jitter_features, security_features, topo_features, temporal_features
        )
        
        path_features = PathFeatures(
            bandwidth_features=bandwidth_features,
            latency_features=latency_features,
            loss_features=loss_features,
            jitter_features=jitter_features,
            security_features=security_features,
            path_length=len(path_nodes),
            hop_count=len(path_nodes) - 1,
            node_types=[graph.topology.get_node(node).node_type if graph.topology.get_node(node) else 'unknown' 
                       for node in path_nodes],
            centrality_scores=topo_features[:4],  # First 4 are centrality scores
            temporal_patterns=temporal_features[:4],
            trend_features=temporal_features[4:6],
            seasonality_features=temporal_features[6:],
            statistical_summary=statistical_summary,
            feature_vector=feature_vector,
            feature_names=feature_names
        )
        
        return path_features
    
    def _extract_bandwidth_features(self, bandwidth_data: np.ndarray) -> np.ndarray:
       
        
        if len(bandwidth_data) == 0:
            return np.zeros(self.num_statistical_features)
        
        features = np.zeros(self.num_statistical_features)
        
        try:
            # Normalize bandwidth data
            normalized_data = self._normalize_metric(bandwidth_data, 'bandwidth')
            
            # Compute statistical aggregations
            if 'mean' in self.aggregation_methods:
                features[self.aggregation_methods.index('mean')] = np.mean(normalized_data)
            if 'std' in self.aggregation_methods:
                features[self.aggregation_methods.index('std')] = np.std(normalized_data)
            if 'max' in self.aggregation_methods:
                features[self.aggregation_methods.index('max')] = np.max(normalized_data)
            if 'min' in self.aggregation_methods:
                features[self.aggregation_methods.index('min')] = np.min(normalized_data)
            if 'median' in self.aggregation_methods:
                features[self.aggregation_methods.index('median')] = np.median(normalized_data)
            
        except Exception as e:
            logger.debug(f"Error extracting bandwidth features: {e}")
        
        return features
    
    def _extract_latency_features(self, latency_data: np.ndarray) -> np.ndarray:
        
        
        if len(latency_data) == 0:
            return np.zeros(self.num_statistical_features)
        
        features = np.zeros(self.num_statistical_features)
        
        try:
            normalized_data = self._normalize_metric(latency_data, 'latency')
            
            if 'mean' in self.aggregation_methods:
                features[self.aggregation_methods.index('mean')] = np.mean(normalized_data)
            if 'std' in self.aggregation_methods:
                features[self.aggregation_methods.index('std')] = np.std(normalized_data)
            if 'max' in self.aggregation_methods:
                features[self.aggregation_methods.index('max')] = np.max(normalized_data)
            if 'min' in self.aggregation_methods:
                features[self.aggregation_methods.index('min')] = np.min(normalized_data)
            if 'median' in self.aggregation_methods:
                features[self.aggregation_methods.index('median')] = np.median(normalized_data)
            
        except Exception as e:
            logger.debug(f"Error extracting latency features: {e}")
        
        return features
    
    def _extract_loss_features(self, loss_data: np.ndarray) -> np.ndarray:
        """Extract statistical features from packet loss measurements"""
        
        if len(loss_data) == 0:
            return np.zeros(self.num_statistical_features)
        
        features = np.zeros(self.num_statistical_features)
        
        try:
            normalized_data = self._normalize_metric(loss_data, 'loss')
            
            if 'mean' in self.aggregation_methods:
                features[self.aggregation_methods.index('mean')] = np.mean(normalized_data)
            if 'std' in self.aggregation_methods:
                features[self.aggregation_methods.index('std')] = np.std(normalized_data)
            if 'max' in self.aggregation_methods:
                features[self.aggregation_methods.index('max')] = np.max(normalized_data)
            if 'min' in self.aggregation_methods:
                features[self.aggregation_methods.index('min')] = np.min(normalized_data)
            if 'median' in self.aggregation_methods:
                features[self.aggregation_methods.index('median')] = np.median(normalized_data)
            
        except Exception as e:
            logger.debug(f"Error extracting loss features: {e}")
        
        return features
    
    def _extract_jitter_features(self, jitter_data: np.ndarray) -> np.ndarray:
      
        
        if len(jitter_data) == 0:
            return np.zeros(self.num_statistical_features)
        
        features = np.zeros(self.num_statistical_features)
        
        try:
            normalized_data = self._normalize_metric(jitter_data, 'jitter')
            
            if 'mean' in self.aggregation_methods:
                features[self.aggregation_methods.index('mean')] = np.mean(normalized_data)
            if 'std' in self.aggregation_methods:
                features[self.aggregation_methods.index('std')] = np.std(normalized_data)
            if 'max' in self.aggregation_methods:
                features[self.aggregation_methods.index('max')] = np.max(normalized_data)
            if 'min' in self.aggregation_methods:
                features[self.aggregation_methods.index('min')] = np.min(normalized_data)
            if 'median' in self.aggregation_methods:
                features[self.aggregation_methods.index('median')] = np.median(normalized_data)
            
        except Exception as e:
            logger.debug(f"Error extracting jitter features: {e}")
        
        return features
    
    def _extract_security_features(self, path_nodes: List[str], graph: NetworkGraph) -> np.ndarray:
        """Extract security-related features from path"""
        
        features = np.zeros(self.num_statistical_features)
        
        try:
            security_scores = []
            
            # Collect security scores from path edges
            for i in range(len(path_nodes) - 1):
                edge = graph.topology.get_edge(path_nodes[i], path_nodes[i + 1])
                if edge:
                    security_scores.append(edge.security_score)
                else:
                    security_scores.append(5.0)  # Default moderate security
            
            if security_scores:
                normalized_scores = self._normalize_metric(np.array(security_scores), 'security')
                
                if 'mean' in self.aggregation_methods:
                    features[self.aggregation_methods.index('mean')] = np.mean(normalized_scores)
                if 'std' in self.aggregation_methods:
                    features[self.aggregation_methods.index('std')] = np.std(normalized_scores)
                if 'max' in self.aggregation_methods:
                    features[self.aggregation_methods.index('max')] = np.max(normalized_scores)
                if 'min' in self.aggregation_methods:
                    features[self.aggregation_methods.index('min')] = np.min(normalized_scores)
                if 'median' in self.aggregation_methods:
                    features[self.aggregation_methods.index('median')] = np.median(normalized_scores)
        
        except Exception as e:
            logger.debug(f"Error extracting security features: {e}")
        
        return features
    
    def _extract_topological_features(self, path_nodes: List[str], graph: NetworkGraph) -> np.ndarray:
        """Extract topological features from graph structure"""
        
        features = np.zeros(self.num_topological_features)
        
        try:
            # Feature 0: Path length (normalized)
            features[0] = min(len(path_nodes) / 10.0, 1.0)  # Normalize by max expected path length
            
            # Feature 1: Average node degree
            degrees = []
            for node_id in path_nodes:
                neighbors = graph.topology.get_neighbors(node_id)
                degrees.append(len(neighbors))
            features[1] = np.mean(degrees) / 20.0 if degrees else 0.0  # Normalize by max expected degree
            
            # Feature 2: Path diversity (number of unique node types)
            node_types = set()
            for node_id in path_nodes:
                node = graph.topology.get_node(node_id)
                if node:
                    node_types.add(node.node_type)
            features[2] = len(node_types) / 5.0  # Normalize by max expected types
            
            # Feature 3: Centrality score (average betweenness centrality of path nodes)
            try:
                central_nodes = graph.topology.get_central_nodes('betweenness', top_k=50)
                central_dict = dict(central_nodes)
                centrality_scores = [central_dict.get(node, 0.0) for node in path_nodes]
                features[3] = np.mean(centrality_scores)
            except Exception:
                features[3] = 0.0
            
            # Feature 4: Graph density around path
            try:
                # Compute local density around path nodes
                local_edges = 0
                local_nodes = set()
                for node_id in path_nodes:
                    neighbors = graph.topology.get_neighbors(node_id)
                    local_nodes.update(neighbors)
                    local_edges += len(neighbors)
                
                if len(local_nodes) > 1:
                    max_edges = len(local_nodes) * (len(local_nodes) - 1) / 2
                    features[4] = local_edges / max_edges if max_edges > 0 else 0.0
                else:
                    features[4] = 0.0
            except Exception:
                features[4] = 0.0
            
            # Feature 5: Path redundancy (number of alternative paths)
            try:
                if len(path_nodes) >= 2:
                    alt_paths = graph.extract_paths(path_nodes[0], path_nodes[-1], max_paths=5)
                    features[5] = min(len(alt_paths) / 5.0, 1.0)  # Normalize by max paths considered
                else:
                    features[5] = 0.0
            except Exception:
                features[5] = 0.0
        
        except Exception as e:
            logger.debug(f"Error extracting topological features: {e}")
        
        return features
    
    def _extract_temporal_features(self, timestamps: np.ndarray, performance_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract temporal pattern features"""
        
        features = np.zeros(self.num_temporal_features)
        
        try:
            if len(timestamps) < 3:
                return features
            
            # Feature 0: Time span
            time_span = timestamps[-1] - timestamps[0]
            features[0] = min(time_span / 3600.0, 1.0)  # Normalize by 1 hour
            
            # Feature 1: Sampling rate consistency
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                features[1] = 1.0 / (1.0 + np.std(intervals))  # High when consistent
            
            # Features 2-3: Trend analysis on latency data
            if 'latency' in performance_data and len(performance_data['latency']) > 2:
                latency_data = performance_data['latency']
                
                # Linear trend slope
                if len(latency_data) == len(timestamps):
                    slope, _, r_value, _, _ = stats.linregress(timestamps, latency_data)
                    features[2] = np.tanh(slope * 1000)  # Scaled and bounded
                    features[3] = r_value ** 2  # R-squared value
            
            # Features 4-5: Periodicity detection
            if 'bandwidth' in performance_data and len(performance_data['bandwidth']) > 10:
                bw_data = performance_data['bandwidth']
                
                try:
                    # FFT for frequency domain analysis
                    fft_vals = fft(bw_data - np.mean(bw_data))
                    freqs = fftfreq(len(bw_data))
                    
                    # Find dominant frequency
                    dominant_freq_idx = np.argmax(np.abs(fft_vals[1:len(fft_vals)//2])) + 1
                    features[4] = np.abs(freqs[dominant_freq_idx])
                    features[5] = np.abs(fft_vals[dominant_freq_idx]) / len(bw_data)
                    
                except Exception:
                    features[4] = features[5] = 0.0
            
            # Features 6-7: Autocorrelation features
            if self.include_autocorr and 'latency' in performance_data:
                latency_data = performance_data['latency']
                
                if len(latency_data) > 5:
                    try:
                        # Autocorrelation at lag 1
                        autocorr_1 = np.corrcoef(latency_data[:-1], latency_data[1:])[0, 1]
                        features[6] = autocorr_1 if not np.isnan(autocorr_1) else 0.0
                        
                        # Autocorrelation at lag 2
                        if len(latency_data) > 6:
                            autocorr_2 = np.corrcoef(latency_data[:-2], latency_data[2:])[0, 1]
                            features[7] = autocorr_2 if not np.isnan(autocorr_2) else 0.0
                            
                    except Exception:
                        features[6] = features[7] = 0.0
        
        except Exception as e:
            logger.debug(f"Error extracting temporal features: {e}")
        
        return features
    
    def _normalize_metric(self, data: np.ndarray, metric_type: str) -> np.ndarray:
        """Normalize metric data to [0, 1] range"""
        
        if metric_type not in self.normalization_params:
            return np.clip(data, 0.0, 1.0)
        
        params = self.normalization_params[metric_type]
        min_val = params['min']
        max_val = params['max']
        
        # Min-max normalization
        normalized = (np.clip(data, min_val, max_val) - min_val) / (max_val - min_val)
        
        return normalized
    
    def _compute_statistical_summary(self, performance_data: Dict[str, np.ndarray]) -> Dict[str, float]:
      
        
        summary = {}
        
        for metric_name, data in performance_data.items():
            if len(data) > 0:
                summary[f'{metric_name}_mean'] = float(np.mean(data))
                summary[f'{metric_name}_std'] = float(np.std(data))
                summary[f'{metric_name}_min'] = float(np.min(data))
                summary[f'{metric_name}_max'] = float(np.max(data))
                
                if len(data) > 1:
                    # Coefficient of variation
                    cv = np.std(data) / (np.mean(data) + 1e-8)
                    summary[f'{metric_name}_cv'] = float(cv)
        
        return summary
    
    def _combine_features(self, bandwidth_features: np.ndarray, latency_features: np.ndarray,
                         loss_features: np.ndarray, jitter_features: np.ndarray,
                         security_features: np.ndarray, topo_features: np.ndarray,
                         temporal_features: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        
        # Concatenate all features
        feature_vector = np.concatenate([
            bandwidth_features,
            latency_features,
            loss_features,
            jitter_features,
            security_features,
            topo_features,
            temporal_features
        ])
        
        # Generate feature names
        feature_names = []
        
        # Bandwidth feature names
        for method in self.aggregation_methods:
            feature_names.append(f'bandwidth_{method}')
        
        # Latency feature names
        for method in self.aggregation_methods:
            feature_names.append(f'latency_{method}')
        
        # Loss feature names
        for method in self.aggregation_methods:
            feature_names.append(f'loss_{method}')
        
        # Jitter feature names
        for method in self.aggregation_methods:
            feature_names.append(f'jitter_{method}')
        
        # Security feature names
        for method in self.aggregation_methods:
            feature_names.append(f'security_{method}')
        
        # Topological feature names
        topo_names = ['path_length', 'avg_degree', 'path_diversity', 
                     'centrality_score', 'local_density', 'path_redundancy']
        feature_names.extend(topo_names)
        
        # Temporal feature names
        temporal_names = ['time_span', 'sampling_consistency', 'trend_slope', 
                         'trend_r2', 'dominant_frequency', 'frequency_amplitude',
                         'autocorr_lag1', 'autocorr_lag2']
        feature_names.extend(temporal_names)
        
        return feature_vector, feature_names
    
    def _create_empty_features(self) -> PathFeatures:
        """Create empty feature structure for invalid paths"""
        
        return PathFeatures(
            bandwidth_features=np.zeros(self.num_statistical_features),
            latency_features=np.zeros(self.num_statistical_features),
            loss_features=np.zeros(self.num_statistical_features),
            jitter_features=np.zeros(self.num_statistical_features),
            security_features=np.zeros(self.num_statistical_features),
            path_length=0,
            hop_count=0,
            node_types=[],
            centrality_scores=np.zeros(4),
            temporal_patterns=np.zeros(4),
            trend_features=np.zeros(2),
            seasonality_features=np.zeros(2),
            statistical_summary={},
            feature_vector=np.zeros(self.total_features),
            feature_names=[]
        )
    
    def extract_single_sample_features(self, sample_data: Dict) -> np.ndarray:

        
        try:
            # Extract basic metrics
            bandwidth = sample_data.get('bandwidth_mbps', 100.0)
            latency = sample_data.get('latency_ms', 10.0)
            loss = sample_data.get('packet_loss_rate', 0.001)
            jitter = sample_data.get('jitter_ms', 1.0)
            security = sample_data.get('security_score', 5.0)
            
            # Normalize metrics
            norm_bandwidth = self._normalize_metric(np.array([bandwidth]), 'bandwidth')[0]
            norm_latency = self._normalize_metric(np.array([latency]), 'latency')[0]
            norm_loss = self._normalize_metric(np.array([loss]), 'loss')[0]
            norm_jitter = self._normalize_metric(np.array([jitter]), 'jitter')[0]
            norm_security = self._normalize_metric(np.array([security]), 'security')[0]
            
            # Create simplified feature vector (for single samples)
            feature_vector = np.array([
                norm_bandwidth, norm_latency, norm_loss, norm_jitter, norm_security,
                # Add some derived features
                norm_bandwidth / (norm_latency + 0.01),  # Bandwidth-latency ratio
                norm_loss * norm_jitter,                  # Loss-jitter product
                (norm_bandwidth + norm_security) / 2,     # Quality score
                1.0 / (norm_latency + norm_loss + 0.01),  # Performance score
                sample_data.get('utilization', 0.5)       # Link utilization
            ])
            
            return feature_vector
            
        except Exception as e:
            logger.debug(f"Error extracting single sample features: {e}")
            return np.random.uniform(0.1, 0.9, 10)  # Fallback random features
    
    def get_feature_importance_weights(self) -> np.ndarray:
       
        
        weights = np.ones(self.total_features)
        
        # Higher weights for core network metrics
        base_features = self.num_base_features * self.num_statistical_features
        weights[:base_features] *= 1.5  # Core metrics more important
        
        # Moderate weights for topological features
        topo_start = base_features
        topo_end = topo_start + self.num_topological_features
        weights[topo_start:topo_end] *= 1.2
        
        # Standard weights for temporal features (already 1.0)
        
        # Normalize weights
        weights /= np.sum(weights)
        
        return weights
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        
        # Generate complete feature names list
        _, feature_names = self._combine_features(
            np.zeros(self.num_statistical_features),
            np.zeros(self.num_statistical_features),
            np.zeros(self.num_statistical_features),
            np.zeros(self.num_statistical_features),
            np.zeros(self.num_statistical_features),
            np.zeros(self.num_topological_features),
            np.zeros(self.num_temporal_features)
        )
        
        return feature_names 
