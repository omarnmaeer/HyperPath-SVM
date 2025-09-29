# File: tests/test_data_processing.py

"""
Comprehensive unit tests for data processing components.
Tests dataset loading, preprocessing, and data augmentation functionality.
"""

import unittest
import numpy as np
import pandas as pd
import json
import tempfile
import os
import sys
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperpath_svm.data.dataset_loader import DatasetLoader
from hyperpath_svm.data.data_augmentation import DataAugmentation


class TestDatasetLoader(unittest.TestCase):
    """Test cases for dataset loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset_loader = DatasetLoader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample datasets
        self.sample_caida_data = self._create_sample_caida_data()
        self.sample_mawi_data = self._create_sample_mawi_data()
        self.sample_umass_data = self._create_sample_umass_data()
        self.sample_wits_data = self._create_sample_wits_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_caida_data(self) -> Dict[str, Any]:
        """Create sample CAIDA dataset structure."""
        # Create CAIDA-style topology data
        topology_data = {
            'nodes': [],
            'links': [],
            'metadata': {
                'collection_date': '2024-01-01',
                'num_nodes': 100,
                'num_links': 250
            }
        }
        
        # Generate nodes
        for i in range(100):
            node = {
                'node_id': i,
                'asn': f"AS{65000 + i}",
                'country': np.random.choice(['US', 'EU', 'AS', 'CN']),
                'degree': np.random.randint(1, 20),
                'coordinates': [np.random.uniform(-180, 180), np.random.uniform(-90, 90)]
            }
            topology_data['nodes'].append(node)
        
        # Generate links
        for i in range(250):
            src = np.random.randint(0, 100)
            dst = np.random.randint(0, 100)
            if src != dst:
                link = {
                    'src_node': src,
                    'dst_node': dst,
                    'capacity': np.random.choice([10, 100, 1000, 10000]),  # Mbps
                    'delay': np.random.exponential(10),  # ms
                    'reliability': np.random.uniform(0.95, 0.999),
                    'timestamp': datetime.now().isoformat()
                }
                topology_data['links'].append(link)
        
        return topology_data
    
    def _create_sample_mawi_data(self) -> Dict[str, Any]:
        """Create sample MAWI dataset structure."""
        # Create MAWI-style traffic trace data
        traffic_data = {
            'flows': [],
            'packets': [],
            'metadata': {
                'capture_date': '2024-01-01',
                'duration_seconds': 3600,
                'total_bytes': 0,
                'total_packets': 0
            }
        }
        
        # Generate flow data
        total_bytes = 0
        total_packets = 0
        
        for i in range(1000):
            flow = {
                'flow_id': i,
                'src_ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'dst_ip': f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.choice([80, 443, 22, 21, 25]),
                'protocol': np.random.choice(['TCP', 'UDP', 'ICMP']),
                'bytes': np.random.exponential(1000),
                'packets': np.random.poisson(10),
                'duration': np.random.exponential(60),
                'start_time': np.random.uniform(0, 3600)
            }
            traffic_data['flows'].append(flow)
            total_bytes += flow['bytes']
            total_packets += flow['packets']
        
        traffic_data['metadata']['total_bytes'] = total_bytes
        traffic_data['metadata']['total_packets'] = total_packets
        
        return traffic_data
    
    def _create_sample_umass_data(self) -> Dict[str, Any]:
        """Create sample UMass dataset structure."""
        # Create UMass-style campus network data
        network_data = {
            'topology': {
                'buildings': [],
                'links': [],
                'access_points': []
            },
            'traffic': {
                'sessions': [],
                'bandwidth_usage': []
            },
            'metadata': {
                'network_type': 'campus',
                'collection_period': '2024-01-01 to 2024-01-31',
                'num_buildings': 20,
                'num_access_points': 100
            }
        }
        
        # Generate building data
        for i in range(20):
            building = {
                'building_id': i,
                'name': f"Building_{chr(65+i)}",
                'location': [np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                'capacity': np.random.choice([100, 1000, 10000]),  # Mbps
                'users': np.random.randint(10, 500)
            }
            network_data['topology']['buildings'].append(building)
        
        # Generate access point data
        for i in range(100):
            ap = {
                'ap_id': i,
                'building_id': np.random.randint(0, 20),
                'channel': np.random.choice([1, 6, 11]),
                'power': np.random.uniform(10, 20),  # dBm
                'connected_users': np.random.randint(0, 50)
            }
            network_data['topology']['access_points'].append(ap)
        
        # Generate traffic sessions
        for i in range(500):
            session = {
                'session_id': i,
                'user_id': np.random.randint(0, 1000),
                'ap_id': np.random.randint(0, 100),
                'start_time': np.random.uniform(0, 86400),  # seconds in day
                'duration': np.random.exponential(1800),  # 30 min average
                'bytes_up': np.random.exponential(1e6),
                'bytes_down': np.random.exponential(5e6)
            }
            network_data['traffic']['sessions'].append(session)
        
        return network_data
    
    def _create_sample_wits_data(self) -> Dict[str, Any]:
        """Create sample WITS dataset structure."""
        # Create WITS-style aggregated traffic data
        wits_data = {
            'time_series': [],
            'aggregated_flows': [],
            'metadata': {
                'aggregation_level': 'hourly',
                'measurement_points': 10,
                'time_range': '2024-01-01 to 2024-01-07'
            }
        }
        
        # Generate time series data
        for hour in range(24 * 7):  # One week of hourly data
            timestamp = datetime(2024, 1, 1) + timedelta(hours=hour)
            
            for point in range(10):  # 10 measurement points
                ts_entry = {
                    'timestamp': timestamp.isoformat(),
                    'measurement_point': point,
                    'total_bytes': np.random.lognormal(15, 2),  # Log-normal traffic
                    'total_packets': np.random.poisson(10000),
                    'active_flows': np.random.randint(100, 1000),
                    'utilization': np.random.uniform(0.1, 0.9)
                }
                wits_data['time_series'].append(ts_entry)
        
        return wits_data
    
    def _save_sample_datasets(self):
        """Save sample datasets to temporary files."""
        # Create dataset directories
        caida_dir = os.path.join(self.temp_dir, 'caida')
        mawi_dir = os.path.join(self.temp_dir, 'mawi')
        umass_dir = os.path.join(self.temp_dir, 'umass')
        wits_dir = os.path.join(self.temp_dir, 'wits')
        
        for dir_path in [caida_dir, mawi_dir, umass_dir, wits_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save CAIDA data
        with open(os.path.join(caida_dir, 'topology.json'), 'w') as f:
            json.dump(self.sample_caida_data, f, default=str)
        
        # Save MAWI data
        with open(os.path.join(mawi_dir, 'traffic_traces.json'), 'w') as f:
            json.dump(self.sample_mawi_data, f, default=str)
        
        # Save UMass data
        with open(os.path.join(umass_dir, 'campus_network.json'), 'w') as f:
            json.dump(self.sample_umass_data, f, default=str)
        
        # Save WITS data
        with open(os.path.join(wits_dir, 'aggregated_traffic.json'), 'w') as f:
            json.dump(self.sample_wits_data, f, default=str)
    
    def test_caida_dataset_loading(self):
        """Test CAIDA dataset loading."""
        self._save_sample_datasets()
        caida_dir = os.path.join(self.temp_dir, 'caida')
        
        # Load dataset
        dataset = self.dataset_loader.load_caida_dataset(
            data_dir=caida_dir,
            time_window=24
        )
        
        # Validate structure
        self.assertIn('topology', dataset)
        self.assertIn('samples', dataset)
        self.assertIn('metadata', dataset)
        
        # Check topology
        topology = dataset['topology']
        self.assertIsInstance(topology, np.ndarray)
        self.assertEqual(topology.shape, (100, 100))  # 100 nodes
        
        # Check samples
        samples = dataset['samples']
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
        
        # Validate sample structure
        sample = samples[0]
        self.assertIn('src', sample)
        self.assertIn('dst', sample)
        self.assertIn('features', sample)
        self.assertIn('timestamp', sample)
    
    def test_mawi_dataset_loading(self):
        """Test MAWI dataset loading."""
        self._save_sample_datasets()
        mawi_dir = os.path.join(self.temp_dir, 'mawi')
        
        # Load dataset with sampling
        dataset = self.dataset_loader.load_mawi_dataset(
            data_dir=mawi_dir,
            sample_rate=0.1  # Sample 10% of flows
        )
        
        # Validate structure
        self.assertIn('flows', dataset)
        self.assertIn('samples', dataset)
        self.assertIn('metadata', dataset)
        
        # Check flows
        flows = dataset['flows']
        self.assertIsInstance(flows, list)
        
        # Should have sampled roughly 10% of original flows
        original_flow_count = len(self.sample_mawi_data['flows'])
        expected_sampled = int(original_flow_count * 0.1)
        self.assertAlmostEqual(len(flows), expected_sampled, delta=20)
        
        # Validate flow structure
        if flows:
            flow = flows[0]
            self.assertIn('src_ip', flow)
            self.assertIn('dst_ip', flow)
            self.assertIn('bytes', flow)
            self.assertIn('packets', flow)
    
    def test_umass_dataset_loading(self):
        """Test UMass dataset loading."""
        self._save_sample_datasets()
        umass_dir = os.path.join(self.temp_dir, 'umass')
        
        # Load dataset
        dataset = self.dataset_loader.load_umass_dataset(
            data_dir=umass_dir,
            network_type='campus'
        )
        
        # Validate structure
        self.assertIn('topology', dataset)
        self.assertIn('traffic', dataset)
        self.assertIn('samples', dataset)
        self.assertIn('metadata', dataset)
        
        # Check topology
        self.assertIn('buildings', dataset['topology'])
        self.assertIn('access_points', dataset['topology'])
        
        # Check traffic
        self.assertIn('sessions', dataset['traffic'])
        
        # Validate samples
        samples = dataset['samples']
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
    
    def test_wits_dataset_loading(self):
        """Test WITS dataset loading."""
        self._save_sample_datasets()
        wits_dir = os.path.join(self.temp_dir, 'wits')
        
        # Load dataset
        dataset = self.dataset_loader.load_wits_dataset(
            data_dir=wits_dir,
            traffic_type='aggregated'
        )
        
        # Validate structure
        self.assertIn('time_series', dataset)
        self.assertIn('samples', dataset)
        self.assertIn('metadata', dataset)
        
        # Check time series data
        time_series = dataset['time_series']
        self.assertIsInstance(time_series, list)
        self.assertEqual(len(time_series), 24 * 7 * 10)  # Week * measurement points
        
        # Validate time series entry
        if time_series:
            entry = time_series[0]
            self.assertIn('timestamp', entry)
            self.assertIn('total_bytes', entry)
            self.assertIn('utilization', entry)
    
    def test_dataset_validation(self):
        """Test dataset validation functionality."""
        # Test valid dataset
        valid_data = {
            'topology': np.random.rand(10, 10),
            'samples': [{'src': 0, 'dst': 1, 'features': [1, 2, 3]}],
            'metadata': {'num_nodes': 10}
        }
        
        validation_result = self.dataset_loader.validate_dataset(valid_data)
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # Test invalid dataset - missing topology
        invalid_data = {
            'samples': [{'src': 0, 'dst': 1}],
            'metadata': {}
        }
        
        validation_result = self.dataset_loader.validate_dataset(invalid_data)
        self.assertFalse(validation_result['is_valid'])
        self.assertGreater(len(validation_result['errors']), 0)
    
    def test_dataset_preprocessing(self):
        """Test dataset preprocessing functionality."""
        # Create raw dataset
        raw_dataset = {
            'topology': np.random.rand(20, 20),
            'samples': []
        }
        
        # Generate samples with varying feature dimensions
        for i in range(100):
            sample = {
                'src': np.random.randint(0, 20),
                'dst': np.random.randint(0, 20),
                'features': np.random.randn(np.random.randint(5, 15)).tolist(),
                'timestamp': datetime.now().isoformat()
            }
            raw_dataset['samples'].append(sample)
        
        # Preprocess dataset
        processed_dataset = self.dataset_loader.preprocess_dataset(
            raw_dataset,
            normalize_features=True,
            standardize_feature_dim=True,
            target_feature_dim=10
        )
        
        # Validate preprocessing
        self.assertIn('X', processed_dataset)
        self.assertIn('y', processed_dataset)
        self.assertIn('metadata', processed_dataset)
        
        # Check feature standardization
        X = processed_dataset['X']
        self.assertEqual(X.shape[1], 10)  # Should be standardized to 10 dimensions
        
        # Check normalization (features should have reasonable range)
        self.assertLessEqual(np.max(np.abs(X)), 5.0)  # After normalization
    
    def test_batch_loading(self):
        """Test batch loading of multiple datasets."""
        self._save_sample_datasets()
        
        dataset_configs = [
            {'type': 'caida', 'path': os.path.join(self.temp_dir, 'caida')},
            {'type': 'mawi', 'path': os.path.join(self.temp_dir, 'mawi')},
        ]
        
        # Load multiple datasets
        datasets = self.dataset_loader.load_multiple_datasets(dataset_configs)
        
        self.assertEqual(len(datasets), 2)
        self.assertIn('caida', datasets)
        self.assertIn('mawi', datasets)
        
        # Each dataset should be properly loaded
        for dataset_name, dataset_data in datasets.items():
            self.assertIn('samples', dataset_data)
            self.assertIn('metadata', dataset_data)
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        self._save_sample_datasets()
        caida_dir = os.path.join(self.temp_dir, 'caida')
        
        dataset = self.dataset_loader.load_caida_dataset(caida_dir)
        stats = self.dataset_loader.compute_dataset_statistics(dataset)
        
        # Validate statistics
        self.assertIn('num_samples', stats)
        self.assertIn('num_nodes', stats)
        self.assertIn('feature_statistics', stats)
        self.assertIn('topology_statistics', stats)
        
        # Check values
        self.assertGreater(stats['num_samples'], 0)
        self.assertEqual(stats['num_nodes'], 100)
        
        # Feature statistics should include mean, std, min, max
        feat_stats = stats['feature_statistics']
        self.assertIn('mean', feat_stats)
        self.assertIn('std', feat_stats)
        self.assertIn('min', feat_stats)
        self.assertIn('max', feat_stats)
    
    def test_error_handling(self):
        """Test error handling in dataset loading."""
        # Test loading non-existent directory
        with self.assertRaises(FileNotFoundError):
            self.dataset_loader.load_caida_dataset('/non/existent/path')
        
        # Test loading with corrupted data file
        corrupt_dir = os.path.join(self.temp_dir, 'corrupt')
        os.makedirs(corrupt_dir, exist_ok=True)
        
        with open(os.path.join(corrupt_dir, 'topology.json'), 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(json.JSONDecodeError):
            self.dataset_loader.load_caida_dataset(corrupt_dir)
    
    def test_data_format_conversion(self):
        """Test data format conversion utilities."""
        # Test topology matrix conversion
        edge_list = [(0, 1, 0.5), (1, 2, 0.8), (0, 2, 0.3)]
        adjacency_matrix = self.dataset_loader.edge_list_to_adjacency_matrix(
            edge_list, num_nodes=3
        )
        
        self.assertEqual(adjacency_matrix.shape, (3, 3))
        self.assertEqual(adjacency_matrix[0, 1], 0.5)
        self.assertEqual(adjacency_matrix[1, 2], 0.8)
        self.assertEqual(adjacency_matrix[0, 2], 0.3)
        
        # Test back conversion
        recovered_edge_list = self.dataset_loader.adjacency_matrix_to_edge_list(
            adjacency_matrix
        )
        
        self.assertEqual(len(recovered_edge_list), 3)


class TestDataAugmentation(unittest.TestCase):
    """Test cases for data augmentation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_augmentation = DataAugmentation()
        
        # Create sample data
        self.sample_features = np.random.randn(100, 20)
        self.sample_labels = np.random.randint(0, 5, 100)
        self.sample_topology = self._create_sample_topology(50)
    
    def _create_sample_topology(self, num_nodes):
        """Create sample network topology."""
        topology = np.zeros((num_nodes, num_nodes))
        
        # Create random connections
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < 0.1:  # 10% connection probability
                    weight = np.random.uniform(0.1, 2.0)
                    topology[i, j] = topology[j, i] = weight
        
        return topology
    
    def test_topology_aware_augmentation(self):
        """Test topology-aware data augmentation."""
        # Perform augmentation
        augmented_X, augmented_y = self.data_augmentation.topology_aware_augmentation(
            self.sample_features, 
            self.sample_labels,
            self.sample_topology,
            noise_level=0.1,
            augmentation_ratio=0.5
        )
        
        # Check augmented data size
        expected_size = int(len(self.sample_features) * 1.5)  # 50% augmentation
        self.assertAlmostEqual(len(augmented_X), expected_size, delta=10)
        self.assertEqual(len(augmented_X), len(augmented_y))
        
        # Check that original data is preserved
        np.testing.assert_array_equal(
            augmented_X[:len(self.sample_features)], 
            self.sample_features
        )
        np.testing.assert_array_equal(
            augmented_y[:len(self.sample_labels)], 
            self.sample_labels
        )
        
        # Check that augmented data is different but similar
        augmented_portion = augmented_X[len(self.sample_features):]
        if len(augmented_portion) > 0:
            self.assertFalse(np.array_equal(
                augmented_portion[0], 
                self.sample_features[0]
            ))
    
    def test_traffic_pattern_augmentation(self):
        """Test traffic pattern augmentation."""
        scaling_factors = [0.5, 1.5, 2.0]
        
        augmented_X, augmented_y = self.data_augmentation.traffic_pattern_augmentation(
            self.sample_features,
            self.sample_labels,
            scaling_factors=scaling_factors
        )
        
        # Should have original + (len(scaling_factors) * original_size)
        expected_size = len(self.sample_features) * (1 + len(scaling_factors))
        self.assertEqual(len(augmented_X), expected_size)
        self.assertEqual(len(augmented_y), expected_size)
        
        # Check scaling effects
        original_portion = augmented_X[:len(self.sample_features)]
        scaled_portions = []
        
        start_idx = len(self.sample_features)
        for i, factor in enumerate(scaling_factors):
            end_idx = start_idx + len(self.sample_features)
            scaled_portion = augmented_X[start_idx:end_idx]
            scaled_portions.append(scaled_portion)
            start_idx = end_idx
        
        # Scaled portions should be different from original
        for scaled_portion in scaled_portions:
            self.assertFalse(np.allclose(original_portion, scaled_portion))
    
    def test_temporal_augmentation(self):
        """Test temporal data augmentation."""
        # Create temporal features (time-dependent)
        temporal_features = np.random.randn(100, 24)  # 24 time steps
        
        augmented_features = self.data_augmentation.temporal_augmentation(
            temporal_features,
            time_shift_range=(-5, 5),
            time_warp_factor=0.1,
            augmentation_ratio=1.0
        )
        
        # Should double the data
        expected_size = len(temporal_features) * 2
        self.assertEqual(len(augmented_features), expected_size)
        
        # Original data should be preserved
        np.testing.assert_array_equal(
            augmented_features[:len(temporal_features)],
            temporal_features
        )
        
        # Augmented portion should be different
        augmented_portion = augmented_features[len(temporal_features):]
        self.assertFalse(np.allclose(temporal_features, augmented_portion))
    
    def test_noise_injection(self):
        """Test noise injection augmentation."""
        # Test Gaussian noise
        noisy_data = self.data_augmentation.add_noise(
            self.sample_features,
            noise_type='gaussian',
            noise_level=0.1
        )
        
        self.assertEqual(noisy_data.shape, self.sample_features.shape)
        self.assertFalse(np.allclose(self.sample_features, noisy_data))
        
        # Noise should be relatively small
        noise_magnitude = np.mean(np.abs(noisy_data - self.sample_features))
        self.assertLess(noise_magnitude, 0.2)
        
        # Test uniform noise
        uniform_noisy_data = self.data_augmentation.add_noise(
            self.sample_features,
            noise_type='uniform',
            noise_level=0.05
        )
        
        self.assertEqual(uniform_noisy_data.shape, self.sample_features.shape)
        self.assertFalse(np.allclose(self.sample_features, uniform_noisy_data))
    
    def test_feature_dropout(self):
        """Test feature dropout augmentation."""
        dropout_data = self.data_augmentation.feature_dropout(
            self.sample_features,
            dropout_rate=0.1
        )
        
        self.assertEqual(dropout_data.shape, self.sample_features.shape)
        
        # Some features should be zeroed out
        zero_mask = (dropout_data == 0)
        original_zero_mask = (self.sample_features == 0)
        
        # Should have more zeros after dropout (unless original data was already very sparse)
        total_zeros_after = np.sum(zero_mask)
        total_zeros_before = np.sum(original_zero_mask)
        
        self.assertGreaterEqual(total_zeros_after, total_zeros_before)
    
    def test_synthetic_sample_generation(self):
        """Test synthetic sample generation."""
        # Generate synthetic samples based on existing data
        synthetic_X, synthetic_y = self.data_augmentation.generate_synthetic_samples(
            self.sample_features,
            self.sample_labels,
            num_synthetic=50,
            method='interpolation'
        )
        
        self.assertEqual(len(synthetic_X), 50)
        self.assertEqual(len(synthetic_y), 50)
        self.assertEqual(synthetic_X.shape[1], self.sample_features.shape[1])
        
        # Synthetic labels should be from the same set as original
        unique_original = set(self.sample_labels)
        unique_synthetic = set(synthetic_y)
        self.assertTrue(unique_synthetic.issubset(unique_original))
        
        # Test adversarial method
        adversarial_X, adversarial_y = self.data_augmentation.generate_synthetic_samples(
            self.sample_features,
            self.sample_labels,
            num_synthetic=30,
            method='adversarial'
        )
        
        self.assertEqual(len(adversarial_X), 30)
        self.assertEqual(len(adversarial_y), 30)
    
    def test_graph_augmentation(self):
        """Test graph-specific augmentation techniques."""
        # Test node feature permutation
        node_features = np.random.randn(50, 10)  # 50 nodes, 10 features each
        
        permuted_features = self.data_augmentation.permute_node_features(
            node_features,
            self.sample_topology,
            permutation_strength=0.1
        )
        
        self.assertEqual(permuted_features.shape, node_features.shape)
        self.assertFalse(np.allclose(node_features, permuted_features))
        
        # Test edge perturbation
        perturbed_topology = self.data_augmentation.perturb_edges(
            self.sample_topology,
            edge_add_prob=0.05,
            edge_remove_prob=0.05,
            weight_noise_level=0.1
        )
        
        self.assertEqual(perturbed_topology.shape, self.sample_topology.shape)
        # Should be symmetric if original was symmetric
        if np.allclose(self.sample_topology, self.sample_topology.T):
            self.assertTrue(np.allclose(perturbed_topology, perturbed_topology.T, atol=1e-10))
    
    def test_class_balancing(self):
        """Test class balancing through augmentation."""
        # Create imbalanced dataset
        imbalanced_y = np.concatenate([
            np.zeros(80),  # 80 samples of class 0
            np.ones(15),   # 15 samples of class 1  
            np.full(5, 2)  # 5 samples of class 2
        ]).astype(int)
        
        imbalanced_X = np.random.randn(100, 10)
        
        # Balance classes through augmentation
        balanced_X, balanced_y = self.data_augmentation.balance_classes(
            imbalanced_X,
            imbalanced_y,
            target_samples_per_class=80
        )
        
        # Check class distribution
        unique_classes, counts = np.unique(balanced_y, return_counts=True)
        
        # All classes should have approximately the target number of samples
        for count in counts:
            self.assertGreaterEqual(count, 70)  # Allow some variation
            self.assertLessEqual(count, 90)
    
    def test_augmentation_pipeline(self):
        """Test complete augmentation pipeline."""
        # Define augmentation pipeline
        pipeline_config = {
            'topology_aware': {
                'enabled': True,
                'noise_level': 0.1,
                'augmentation_ratio': 0.5
            },
            'traffic_pattern': {
                'enabled': True,
                'scaling_factors': [0.8, 1.2]
            },
            'noise_injection': {
                'enabled': True,
                'noise_type': 'gaussian',
                'noise_level': 0.05
            }
        }
        
        # Apply pipeline
        augmented_X, augmented_y = self.data_augmentation.apply_augmentation_pipeline(
            self.sample_features,
            self.sample_labels,
            self.sample_topology,
            pipeline_config
        )
        
        # Should have significantly more data after full pipeline
        self.assertGreater(len(augmented_X), len(self.sample_features) * 2)
        self.assertEqual(len(augmented_X), len(augmented_y))
        
        # Original data should still be present at the beginning
        np.testing.assert_array_equal(
            augmented_X[:len(self.sample_features)],
            self.sample_features
        )
    
    def test_augmentation_quality_metrics(self):
        """Test quality metrics for augmented data."""
        # Generate augmented data
        augmented_X, augmented_y = self.data_augmentation.topology_aware_augmentation(
            self.sample_features,
            self.sample_labels,
            self.sample_topology,
            noise_level=0.1,
            augmentation_ratio=1.0
        )
        
        # Compute quality metrics
        quality_metrics = self.data_augmentation.compute_augmentation_quality(
            self.sample_features,
            augmented_X,
            self.sample_labels,
            augmented_y
        )
        
        # Check metric structure
        self.assertIn('diversity_score', quality_metrics)
        self.assertIn('class_balance_score', quality_metrics)
        self.assertIn('feature_distribution_similarity', quality_metrics)
        self.assertIn('label_distribution_similarity', quality_metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(quality_metrics['diversity_score'], 0)
        self.assertLessEqual(quality_metrics['diversity_score'], 1)
        
        self.assertGreaterEqual(quality_metrics['class_balance_score'], 0)
        self.assertLessEqual(quality_metrics['class_balance_score'], 1)
    
    def test_adaptive_augmentation(self):
        """Test adaptive augmentation based on model performance."""
        # Simulate model performance feedback
        performance_feedback = {
            'accuracy': 0.85,
            'class_accuracies': {0: 0.9, 1: 0.8, 2: 0.7},
            'difficult_samples': [10, 25, 37, 42, 68]
        }
        
        # Apply adaptive augmentation
        adaptive_X, adaptive_y = self.data_augmentation.adaptive_augmentation(
            self.sample_features,
            self.sample_labels,
            performance_feedback,
            self.sample_topology
        )
        
        # Should focus on difficult samples and underperforming classes
        self.assertGreater(len(adaptive_X), len(self.sample_features))
        
        # Should have more samples for class 2 (lowest accuracy)
        original_class_counts = np.bincount(self.sample_labels)
        adaptive_class_counts = np.bincount(adaptive_y)
        
        if len(adaptive_class_counts) > 2:  # If class 2 exists
            class_2_ratio = (adaptive_class_counts[2] / adaptive_class_counts.sum()) / \
                           (original_class_counts[2] / original_class_counts.sum())
            self.assertGreater(class_2_ratio, 1.0)  # Should be augmented more


if __name__ == '__main__':
    # Set up test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDatasetLoader,
        TestDataAugmentation
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("DATA PROCESSING TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code) 
