# File: hyperpath_svm/data/data_augmentation.py

"""
Data Augmentation Module for HyperPath-SVM

This module implements sophisticated data augmentation techniques specifically designed
for network routing data. It enhances training robustness by generating diverse
network scenarios while preserving realistic traffic patterns and topology constraints.

Key Features:
- Topology-aware augmentation with graph structure preservation
- Traffic pattern synthesis with realistic temporal dynamics
- Path perturbation with constraint preservation
- Multi-scale noise injection (link, node, and global levels)
- Temporal augmentation with seasonality modeling
- Load balancing scenario generation
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from scipy.stats import norm, exponential, pareto
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.preprocessing import StandardScaler
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..utils.logging_utils import get_logger
from .network_graph import NetworkGraph


@dataclass
class AugmentationConfig:
    
    
    # Topology augmentation
    node_dropout_prob: float = 0.1
    edge_dropout_prob: float = 0.05
    edge_addition_prob: float = 0.03
    topology_noise_std: float = 0.02
    
    # Traffic augmentation
    traffic_scaling_range: Tuple[float, float] = (0.7, 1.5)
    temporal_shift_range: Tuple[int, int] = (-5, 5)
    burst_injection_prob: float = 0.2
    congestion_simulation_prob: float = 0.15
    
    # Path augmentation
    path_perturbation_prob: float = 0.1
    alternative_path_ratio: float = 0.3
    routing_policy_variation: float = 0.2
    
    # Noise injection
    gaussian_noise_std: float = 0.01
    measurement_error_prob: float = 0.05
    outlier_injection_prob: float = 0.02
    
    # Temporal augmentation
    seasonal_variation_amplitude: float = 0.3
    trend_injection_prob: float = 0.1
    time_warping_factor: float = 0.1
    
    # Multi-threading
    max_workers: int = 4
    batch_size: int = 1000


class BaseAugmenter(ABC):
    
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.rng = np.random.RandomState(42)
        self._lock = threading.Lock()
    
    @abstractmethod
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
      
        pass
    
    def _thread_safe_random(self) -> float:
       
        with self._lock:
            return self.rng.random()


class TopologyAugmenter(BaseAugmenter):
    
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        self.topology_cache = {}
    
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        try:
            network_graph = data.get('network_graph')
            if not isinstance(network_graph, NetworkGraph):
                raise ValueError("Invalid network graph in data")
            
            augmented_samples = []
            
            # Node dropout augmentation
            if self._thread_safe_random() < self.config.node_dropout_prob:
                augmented_samples.extend(self._apply_node_dropout(data))
            
            # Edge modification augmentation
            if self._thread_safe_random() < self.config.edge_dropout_prob:
                augmented_samples.extend(self._apply_edge_modifications(data))
            
            # Topology noise injection
            augmented_samples.extend(self._apply_topology_noise(data))
            
            return augmented_samples if augmented_samples else [data]
            
        except Exception as e:
            self.logger.error(f"Topology augmentation failed: {str(e)}")
            return [data]
    
    def _apply_node_dropout(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        network_graph = data['network_graph']
        graph = network_graph.graph.copy()
        
        # Identify non-critical nodes (not breaking connectivity)
        non_critical_nodes = []
        for node in graph.nodes():
            temp_graph = graph.copy()
            temp_graph.remove_node(node)
            if nx.is_connected(temp_graph):
                non_critical_nodes.append(node)
        
        if not non_critical_nodes:
            return [data]
        
        # Randomly select nodes to drop
        num_to_drop = max(1, int(len(non_critical_nodes) * 0.1))
        nodes_to_drop = self.rng.choice(non_critical_nodes, size=num_to_drop, replace=False)
        
        # Create augmented graph
        augmented_graph = graph.copy()
        augmented_graph.remove_nodes_from(nodes_to_drop)
        
        # Update network graph
        augmented_network_graph = NetworkGraph(
            adjacency_matrix=nx.adjacency_matrix(augmented_graph).toarray(),
            node_features=network_graph.node_features,
            edge_features=network_graph.edge_features,
            temporal_features=network_graph.temporal_features
        )
        
        augmented_data = data.copy()
        augmented_data['network_graph'] = augmented_network_graph
        augmented_data['augmentation_type'] = 'node_dropout'
        
        return [augmented_data]
    
    def _apply_edge_modifications(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        network_graph = data['network_graph']
        graph = network_graph.graph.copy()
        
        augmented_samples = []
        
        # Edge dropout
        edges_to_remove = []
        for edge in graph.edges():
            if self._thread_safe_random() < self.config.edge_dropout_prob:
                temp_graph = graph.copy()
                temp_graph.remove_edge(*edge)
                if nx.is_connected(temp_graph):
                    edges_to_remove.append(edge)
        
        if edges_to_remove:
            augmented_graph = graph.copy()
            augmented_graph.remove_edges_from(edges_to_remove)
            
            augmented_network_graph = NetworkGraph(
                adjacency_matrix=nx.adjacency_matrix(augmented_graph).toarray(),
                node_features=network_graph.node_features,
                edge_features=network_graph.edge_features,
                temporal_features=network_graph.temporal_features
            )
            
            augmented_data = data.copy()
            augmented_data['network_graph'] = augmented_network_graph
            augmented_data['augmentation_type'] = 'edge_dropout'
            augmented_samples.append(augmented_data)
        
        # Edge addition
        non_edges = list(nx.non_edges(graph))
        if non_edges and self._thread_safe_random() < self.config.edge_addition_prob:
            num_to_add = max(1, int(len(non_edges) * 0.02))
            edges_to_add = self.rng.choice(len(non_edges), size=min(num_to_add, len(non_edges)), replace=False)
            
            augmented_graph = graph.copy()
            for idx in edges_to_add:
                augmented_graph.add_edge(*non_edges[idx])
            
            augmented_network_graph = NetworkGraph(
                adjacency_matrix=nx.adjacency_matrix(augmented_graph).toarray(),
                node_features=network_graph.node_features,
                edge_features=network_graph.edge_features,
                temporal_features=network_graph.temporal_features
            )
            
            augmented_data = data.copy()
            augmented_data['network_graph'] = augmented_network_graph
            augmented_data['augmentation_type'] = 'edge_addition'
            augmented_samples.append(augmented_data)
        
        return augmented_samples
    
    def _apply_topology_noise(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        network_graph = data['network_graph']
        
        # Add noise to node features
        noisy_node_features = network_graph.node_features.copy()
        noise = self.rng.normal(0, self.config.topology_noise_std, noisy_node_features.shape)
        noisy_node_features += noise
        
        # Add noise to edge features
        noisy_edge_features = network_graph.edge_features.copy()
        edge_noise = self.rng.normal(0, self.config.topology_noise_std, noisy_edge_features.shape)
        noisy_edge_features += edge_noise
        
        # Create augmented network graph
        augmented_network_graph = NetworkGraph(
            adjacency_matrix=network_graph.adjacency_matrix,
            node_features=noisy_node_features,
            edge_features=noisy_edge_features,
            temporal_features=network_graph.temporal_features
        )
        
        augmented_data = data.copy()
        augmented_data['network_graph'] = augmented_network_graph
        augmented_data['augmentation_type'] = 'topology_noise'
        
        return [augmented_data]


class TrafficAugmenter(BaseAugmenter):
    
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        self.traffic_models = self._initialize_traffic_models()
    
    def _initialize_traffic_models(self) -> Dict[str, Any]:
        
        return {
            'diurnal_pattern': self._generate_diurnal_pattern(),
            'weekly_pattern': self._generate_weekly_pattern(),
            'burst_patterns': self._generate_burst_patterns(),
            'congestion_patterns': self._generate_congestion_patterns()
        }
    
    def _generate_diurnal_pattern(self) -> np.ndarray:
       
        hours = np.arange(24)
        # Typical internet traffic pattern with peaks at 9 AM and 8 PM
        pattern = (0.3 + 0.4 * np.sin(2 * np.pi * (hours - 6) / 24) + 
                  0.3 * np.sin(2 * np.pi * (hours - 20) / 24))
        return np.clip(pattern, 0.1, 1.0)
    
    def _generate_weekly_pattern(self) -> np.ndarray:
       
        days = np.arange(7)
        # Lower traffic on weekends
        pattern = np.array([0.9, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6])
        return pattern
    
    def _generate_burst_patterns(self) -> Dict[str, np.ndarray]:
        
        return {
            'flash_crowd': exponential.rvs(scale=2, size=100, random_state=self.rng.randint(1000)),
            'ddos_like': pareto.rvs(b=1.5, size=100, random_state=self.rng.randint(1000)),
            'streaming_event': norm.rvs(loc=5, scale=1, size=100, random_state=self.rng.randint(1000))
        }
    
    def _generate_congestion_patterns(self) -> Dict[str, np.ndarray]:
        
        return {
            'gradual_buildup': np.logspace(0, 2, 50),
            'sudden_spike': np.concatenate([np.ones(20), np.ones(10) * 10, np.ones(20)]),
            'oscillating': 2 + np.sin(np.linspace(0, 4*np.pi, 100))
        }
    
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        try:
            augmented_samples = []
            
            # Traffic scaling
            if self._thread_safe_random() < 0.8:  # High probability for traffic scaling
                augmented_samples.extend(self._apply_traffic_scaling(data))
            
            # Temporal shifts
            if self._thread_safe_random() < 0.6:
                augmented_samples.extend(self._apply_temporal_shifts(data))
            
            # Burst injection
            if self._thread_safe_random() < self.config.burst_injection_prob:
                augmented_samples.extend(self._apply_burst_injection(data))
            
            # Congestion simulation
            if self._thread_safe_random() < self.config.congestion_simulation_prob:
                augmented_samples.extend(self._apply_congestion_simulation(data))
            
            return augmented_samples if augmented_samples else [data]
            
        except Exception as e:
            self.logger.error(f"Traffic augmentation failed: {str(e)}")
            return [data]
    
    def _apply_traffic_scaling(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        scaling_factor = self.rng.uniform(*self.config.traffic_scaling_range)
        
        augmented_data = data.copy()
        network_graph = augmented_data['network_graph']
        
        # Scale temporal features (traffic volumes)
        scaled_temporal_features = network_graph.temporal_features * scaling_factor
        
        # Update network graph
        augmented_network_graph = NetworkGraph(
            adjacency_matrix=network_graph.adjacency_matrix,
            node_features=network_graph.node_features,
            edge_features=network_graph.edge_features,
            temporal_features=scaled_temporal_features
        )
        
        augmented_data['network_graph'] = augmented_network_graph
        augmented_data['augmentation_type'] = 'traffic_scaling'
        augmented_data['scaling_factor'] = scaling_factor
        
        return [augmented_data]
    
    def _apply_temporal_shifts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        shift_amount = self.rng.randint(*self.config.temporal_shift_range)
        
        augmented_data = data.copy()
        network_graph = augmented_data['network_graph']
        
        # Apply temporal shift
        shifted_features = np.roll(network_graph.temporal_features, shift_amount, axis=0)
        
        augmented_network_graph = NetworkGraph(
            adjacency_matrix=network_graph.adjacency_matrix,
            node_features=network_graph.node_features,
            edge_features=network_graph.edge_features,
            temporal_features=shifted_features
        )
        
        augmented_data['network_graph'] = augmented_network_graph
        augmented_data['augmentation_type'] = 'temporal_shift'
        augmented_data['shift_amount'] = shift_amount
        
        return [augmented_data]
    
    def _apply_burst_injection(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
      
        burst_type = self.rng.choice(list(self.traffic_models['burst_patterns'].keys()))
        burst_pattern = self.traffic_models['burst_patterns'][burst_type]
        
        augmented_data = data.copy()
        network_graph = augmented_data['network_graph']
        
        # Select random time window for burst injection
        temporal_length = network_graph.temporal_features.shape[0]
        burst_length = min(len(burst_pattern), temporal_length // 4)
        start_idx = self.rng.randint(0, temporal_length - burst_length)
        
        # Apply burst to temporal features
        augmented_temporal = network_graph.temporal_features.copy()
        burst_multiplier = burst_pattern[:burst_length].reshape(-1, 1)
        augmented_temporal[start_idx:start_idx+burst_length] *= burst_multiplier
        
        augmented_network_graph = NetworkGraph(
            adjacency_matrix=network_graph.adjacency_matrix,
            node_features=network_graph.node_features,
            edge_features=network_graph.edge_features,
            temporal_features=augmented_temporal
        )
        
        augmented_data['network_graph'] = augmented_network_graph
        augmented_data['augmentation_type'] = 'burst_injection'
        augmented_data['burst_type'] = burst_type
        
        return [augmented_data]
    
    def _apply_congestion_simulation(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        congestion_type = self.rng.choice(list(self.traffic_models['congestion_patterns'].keys()))
        congestion_pattern = self.traffic_models['congestion_patterns'][congestion_type]
        
        augmented_data = data.copy()
        network_graph = augmented_data['network_graph']
        
        # Apply congestion to edge features (link utilization)
        augmented_edge_features = network_graph.edge_features.copy()
        congestion_factor = self.rng.uniform(1.2, 2.0)
        
        # Select subset of edges to congest
        num_edges = augmented_edge_features.shape[0]
        congested_edges = self.rng.choice(num_edges, size=max(1, num_edges // 10), replace=False)
        
        augmented_edge_features[congested_edges] *= congestion_factor
        augmented_edge_features = np.clip(augmented_edge_features, 0, 1)  # Ensure valid range
        
        augmented_network_graph = NetworkGraph(
            adjacency_matrix=network_graph.adjacency_matrix,
            node_features=network_graph.node_features,
            edge_features=augmented_edge_features,
            temporal_features=network_graph.temporal_features
        )
        
        augmented_data['network_graph'] = augmented_network_graph
        augmented_data['augmentation_type'] = 'congestion_simulation'
        augmented_data['congestion_type'] = congestion_type
        
        return [augmented_data]


class PathAugmenter(BaseAugmenter):
    
    
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        try:
            augmented_samples = []
            
            # Path perturbation
            if self._thread_safe_random() < self.config.path_perturbation_prob:
                augmented_samples.extend(self._apply_path_perturbation(data))
            
            # Alternative path generation
            if self._thread_safe_random() < self.config.alternative_path_ratio:
                augmented_samples.extend(self._generate_alternative_paths(data))
            
            return augmented_samples if augmented_samples else [data]
            
        except Exception as e:
            self.logger.error(f"Path augmentation failed: {str(e)}")
            return [data]
    
    def _apply_path_perturbation(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        augmented_data = data.copy()
        
        # Add small random variations to path costs/weights
        if 'path_features' in data:
            path_features = data['path_features'].copy()
            perturbation = self.rng.normal(0, 0.01, path_features.shape)
            path_features += perturbation
            augmented_data['path_features'] = path_features
        
        augmented_data['augmentation_type'] = 'path_perturbation'
        return [augmented_data]
    
    def _generate_alternative_paths(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        augmented_data = data.copy()
        
        # Simulate alternative routing decisions
        if 'routing_decisions' in data:
            decisions = data['routing_decisions'].copy()
            # Randomly flip some routing decisions
            flip_mask = self.rng.random(decisions.shape) < 0.05
            decisions[flip_mask] = 1 - decisions[flip_mask]  # Flip binary decisions
            augmented_data['routing_decisions'] = decisions
        
        augmented_data['augmentation_type'] = 'alternative_paths'
        return [augmented_data]


class DataAugmentationPipeline:
   
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.logger = get_logger(__name__)
        
        # Initialize augmenters
        self.topology_augmenter = TopologyAugmenter(self.config)
        self.traffic_augmenter = TrafficAugmenter(self.config)
        self.path_augmenter = PathAugmenter(self.config)
        
        self.augmenters = [
            self.topology_augmenter,
            self.traffic_augmenter,
            self.path_augmenter
        ]
        
        # Performance tracking
        self.augmentation_stats = {
            'total_processed': 0,
            'total_generated': 0,
            'processing_time': 0.0,
            'augmentation_types': {}
        }
    
    def augment_batch(self, data_batch: List[Dict[str, Any]], 
                     augmentation_factor: int = 2) -> List[Dict[str, Any]]:
      
        start_time = time.time()
        
        try:
            augmented_batch = []
            
            # Process in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_sample = {
                    executor.submit(self._augment_single_sample, sample, augmentation_factor): i 
                    for i, sample in enumerate(data_batch)
                }
                
                for future in as_completed(future_to_sample):
                    sample_idx = future_to_sample[future]
                    try:
                        augmented_samples = future.result()
                        augmented_batch.extend(augmented_samples)
                        
                        # Update statistics
                        for sample in augmented_samples:
                            aug_type = sample.get('augmentation_type', 'original')
                            self.augmentation_stats['augmentation_types'][aug_type] = \
                                self.augmentation_stats['augmentation_types'].get(aug_type, 0) + 1
                                
                    except Exception as e:
                        self.logger.error(f"Error augmenting sample {sample_idx}: {str(e)}")
                        augmented_batch.append(data_batch[sample_idx])  # Add original on failure
            
            # Update global statistics
            processing_time = time.time() - start_time
            self.augmentation_stats['total_processed'] += len(data_batch)
            self.augmentation_stats['total_generated'] += len(augmented_batch)
            self.augmentation_stats['processing_time'] += processing_time
            
            self.logger.info(f"Augmented {len(data_batch)} samples to {len(augmented_batch)} "
                           f"samples in {processing_time:.2f}s")
            
            return augmented_batch
            
        except Exception as e:
            self.logger.error(f"Batch augmentation failed: {str(e)}")
            return data_batch
    
    def _augment_single_sample(self, sample: Dict[str, Any], 
                              augmentation_factor: int) -> List[Dict[str, Any]]:
        
        try:
            augmented_samples = [sample]  # Include original
            
            for _ in range(augmentation_factor):
                # Randomly select augmenters to apply
                selected_augmenters = random.sample(
                    self.augmenters, 
                    k=random.randint(1, len(self.augmenters))
                )
                
                current_sample = sample
                for augmenter in selected_augmenters:
                    augmented_results = augmenter.augment(current_sample)
                    if augmented_results:
                        current_sample = random.choice(augmented_results)
                
                augmented_samples.append(current_sample)
            
            return augmented_samples
            
        except Exception as e:
            self.logger.error(f"Single sample augmentation failed: {str(e)}")
            return [sample]
    
    def get_augmentation_statistics(self) -> Dict[str, Any]:
      
        stats = self.augmentation_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['augmentation_ratio'] = stats['total_generated'] / stats['total_processed']
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_processed']
        
        return stats
    
    def reset_statistics(self):
       
        self.augmentation_stats = {
            'total_processed': 0,
            'total_generated': 0,
            'processing_time': 0.0,
            'augmentation_types': {}
        }
        
        self.logger.info("Augmentation statistics reset")


def create_augmentation_pipeline(config_dict: Optional[Dict[str, Any]] = None) -> DataAugmentationPipeline:
   
    if config_dict:
        config = AugmentationConfig(**config_dict)
    else:
        config = AugmentationConfig()
    
    return DataAugmentationPipeline(config)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Test the augmentation pipeline
    logger = get_logger(__name__)
    logger.info("Testing data augmentation pipeline...")
    
    # Create test data
    test_network_graph = NetworkGraph(
        adjacency_matrix=np.random.random((10, 10)),
        node_features=np.random.random((10, 5)),
        edge_features=np.random.random((20, 3)),
        temporal_features=np.random.random((100, 10))
    )
    
    test_data = {
        'network_graph': test_network_graph,
        'path_features': np.random.random((5, 8)),
        'routing_decisions': np.random.randint(0, 2, (10,)),
        'labels': np.random.random((1,))
    }
    
    # Create and test pipeline
    pipeline = create_augmentation_pipeline()
    augmented_batch = pipeline.augment_batch([test_data], augmentation_factor=3)
    
    logger.info(f"Original samples: 1")
    logger.info(f"Augmented samples: {len(augmented_batch)}")
    logger.info(f"Augmentation statistics: {pipeline.get_augmentation_statistics()}")
    
    logger.info("Data augmentation pipeline test completed successfully!") 
