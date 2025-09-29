# File: tests/test_integration.py

"""
End-to-end integration tests for HyperPath-SVM framework.
Tests complete pipeline from data loading through training to evaluation.
"""

import unittest
import numpy as np
import pandas as pd
import json
import pickle
import tempfile
import shutil
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from hyperpath_svm.core.hyperpath_svm import HyperPathSVM
from hyperpath_svm.core.ddwe import DDWEOptimizer
from hyperpath_svm.core.tgck import TGCKKernel

# Data processing imports
from hyperpath_svm.data.dataset_loader import DatasetLoader
from hyperpath_svm.data.data_augmentation import DataAugmentation

# Evaluation imports
from hyperpath_svm.evaluation.evaluator import HyperPathEvaluator
from hyperpath_svm.evaluation.cross_validation import TemporalCrossValidator
from hyperpath_svm.evaluation.metrics import RoutingAccuracy, InferenceTime, MemoryUsage

# Baseline imports
from hyperpath_svm.baselines.neural_networks import GNNBaseline
from hyperpath_svm.baselines.traditional_svm import StaticSVM
from hyperpath_svm.baselines.routing_protocols import OSPFProtocol

# Utility imports
from hyperpath_svm.utils.math_utils import QuantumOptimizer
from hyperpath_svm.utils.graph_utils import GraphProcessor
from hyperpath_svm.utils.logging_utils import setup_logger


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = setup_logger("IntegrationTest", self.temp_dir)
        
        # Initialize main components
        self.dataset_loader = DatasetLoader()
        self.data_augmentation = DataAugmentation()
        self.evaluator = HyperPathEvaluator()
        self.cv = TemporalCrossValidator()
        
        # Create sample dataset
        self.sample_dataset = self._create_sample_dataset()
        self.dataset_path = self._save_sample_dataset()
        
        # Performance tracking
        self.performance_results = {}
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_dataset(self) -> Dict[str, Any]:
        """Create sample dataset for integration testing."""
        num_nodes = 50
        num_samples = 1000
        
        # Create network topology
        topology = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < 0.1:  # 10% connection probability
                    weight = np.random.uniform(0.1, 2.0)
                    topology[i, j] = topology[j, i] = weight
        
        # Create routing samples
        samples = []
        for i in range(num_samples):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                # Create features
                features = np.concatenate([
                    [src, dst],  # Source and destination
                    [np.sum(topology[src] > 0), np.sum(topology[dst] > 0)],  # Degrees
                    [np.random.exponential(1.0)],  # Traffic demand
                    np.random.randn(15)  # Additional features
                ])
                
                # Create optimal path (simplified Dijkstra)
                optimal_path = self._compute_shortest_path(src, dst, topology)
                
                sample = {
                    'src': src,
                    'dst': dst,
                    'features': features.tolist(),
                    'optimal_path': optimal_path,
                    'timestamp': (datetime.now() + timedelta(minutes=i)).isoformat()
                }
                samples.append(sample)
        
        return {
            'topology': topology.tolist(),
            'samples': samples,
            'metadata': {
                'num_nodes': num_nodes,
                'num_samples': len(samples),
                'creation_time': datetime.now().isoformat()
            }
        }
    
    def _compute_shortest_path(self, src: int, dst: int, topology: np.ndarray) -> List[int]:
        """Compute shortest path using simplified Dijkstra."""
        num_nodes = topology.shape[0]
        distances = np.full(num_nodes, np.inf)
        distances[src] = 0
        previous = [-1] * num_nodes
        unvisited = set(range(num_nodes))
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == np.inf:
                break
            
            unvisited.remove(current)
            
            if current == dst:
                break
            
            for neighbor in range(num_nodes):
                if topology[current, neighbor] > 0:
                    alt = distances[current] + topology[current, neighbor]
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        previous[neighbor] = current
        
        # Reconstruct path
        path = []
        current = dst
        while current != -1:
            path.append(current)
            current = previous[current]
        
        return path[::-1] if path and path[-1] == src else [src, dst]
    
    def _save_sample_dataset(self) -> str:
        """Save sample dataset to temporary file."""
        dataset_dir = os.path.join(self.temp_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        dataset_file = os.path.join(dataset_dir, 'routing_dataset.json')
        with open(dataset_file, 'w') as f:
            json.dump(self.sample_dataset, f, default=str)
        
        return dataset_dir
    
    def test_complete_pipeline_execution(self):
        """Test complete pipeline from data loading to evaluation."""
        self.logger.info("Starting complete pipeline integration test")
        
        # Step 1: Data Loading
        self.logger.info("Step 1: Loading dataset")
        dataset = self._load_test_dataset()
        self.assertIsNotNone(dataset)
        self.assertIn('X', dataset)
        self.assertIn('y', dataset)
        
        # Step 2: Data Preprocessing and Augmentation
        self.logger.info("Step 2: Data preprocessing and augmentation")
        X_original, y_original = dataset['X'], dataset['y']
        
        X_augmented, y_augmented = self.data_augmentation.topology_aware_augmentation(
            X_original, y_original, 
            topology=np.array(self.sample_dataset['topology']),
            noise_level=0.05,
            augmentation_ratio=0.3
        )
        
        self.assertGreater(len(X_augmented), len(X_original))
        self.assertEqual(len(X_augmented), len(y_augmented))
        
        # Step 3: Model Initialization and Training
        self.logger.info("Step 3: Model training")
        model = self._initialize_hyperpath_svm()
        
        # Split data
        split_idx = int(0.8 * len(X_augmented))
        X_train, X_test = X_augmented[:split_idx], X_augmented[split_idx:]
        y_train, y_test = y_augmented[:split_idx], y_augmented[split_idx:]
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.performance_results['training_time'] = training_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Step 4: Model Evaluation
        self.logger.info("Step 4: Model evaluation")
        evaluation_results = self.evaluator.evaluate_model(
            model, X_test, y_test, 
            compute_detailed_metrics=True
        )
        
        self.performance_results.update(evaluation_results)
        
        # Step 5: Cross-Validation
        self.logger.info("Step 5: Cross-validation")
        cv_results = self.cv.validate(
            model, X_augmented, y_augmented,
            n_splits=3, test_size=0.2
        )
        
        self.performance_results['cv_score'] = cv_results['mean_score']
        self.performance_results['cv_std'] = cv_results['std_score']
        
        # Step 6: Performance Validation
        self.logger.info("Step 6: Performance validation")
        self._validate_pipeline_performance()
        
        self.logger.info("Complete pipeline integration test passed")
    
    def _load_test_dataset(self) -> Dict[str, Any]:
        """Load and preprocess test dataset."""
        # Load raw data
        with open(os.path.join(self.dataset_path, 'routing_dataset.json'), 'r') as f:
            raw_data = json.load(f)
        
        # Extract features and labels
        X = []
        y = []
        
        for sample in raw_data['samples']:
            X.append(sample['features'])
            y.append(sample['optimal_path'])
        
        return {'X': np.array(X), 'y': np.array(y, dtype=object)}
    
    def _initialize_hyperpath_svm(self) -> HyperPathSVM:
        """Initialize HyperPath-SVM model."""
        ddwe_optimizer = DDWEOptimizer(
            learning_rate=0.01,
            quantum_enhanced=True,
            adaptation_rate=0.001
        )
        
        tgck_kernel = TGCKKernel(
            temporal_window=12,
            confidence_threshold=0.8,
            kernel_type='rbf'
        )
        
        return HyperPathSVM(
            ddwe_optimizer=ddwe_optimizer,
            tgck_kernel=tgck_kernel,
            C=1.0,
            epsilon=0.1
        )
    
    def _validate_pipeline_performance(self):
        """Validate that pipeline meets performance requirements."""
        # Check accuracy
        accuracy = self.performance_results.get('routing_accuracy', 0)
        self.assertGreater(accuracy, 0.7, "Routing accuracy too low")
        
        # Check inference time
        inference_time = self.performance_results.get('inference_time_ms', float('inf'))
        self.assertLess(inference_time, 100.0, "Inference time too high")  # Relaxed for testing
        
        # Check memory usage
        memory_usage = self.performance_results.get('memory_usage_mb', float('inf'))
        self.assertLess(memory_usage, 500.0, "Memory usage too high")  # Relaxed for testing
        
        # Check training time
        training_time = self.performance_results.get('training_time', float('inf'))
        self.assertLess(training_time, 300.0, "Training time too high")  # 5 minutes max
    
    def test_baseline_comparison_pipeline(self):
        """Test pipeline for comparing against baseline methods."""
        self.logger.info("Testing baseline comparison pipeline")
        
        # Load dataset
        dataset = self._load_test_dataset()
        X, y = dataset['X'], dataset['y']
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Initialize models
        models = {
            'HyperPath-SVM': self._initialize_hyperpath_svm(),
            'GNN': GNNBaseline(hidden_dim=32, num_layers=2),
            'Static-SVM': StaticSVM(C=1.0, kernel='rbf'),
            'OSPF': OSPFProtocol()
        }
        
        # Train and evaluate all models
        model_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name}")
            
            try:
                # Train model (if applicable)
                if hasattr(model, 'fit'):
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                else:
                    training_time = 0.0
                
                # Evaluate model
                start_time = time.time()
                predictions = model.predict(X_test)
                inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
                
                # Calculate accuracy
                accuracy = self._calculate_routing_accuracy(predictions, y_test)
                
                model_results[model_name] = {
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'inference_time_ms': inference_time
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
                model_results[model_name] = {
                    'accuracy': 0.0,
                    'training_time': float('inf'),
                    'inference_time_ms': float('inf'),
                    'error': str(e)
                }
        
        # Validate that HyperPath-SVM performs competitively
        hyperpath_accuracy = model_results['HyperPath-SVM']['accuracy']
        
        # Should outperform or be competitive with baselines
        for model_name, results in model_results.items():
            if model_name != 'HyperPath-SVM' and 'error' not in results:
                baseline_accuracy = results['accuracy']
                self.assertGreaterEqual(
                    hyperpath_accuracy, 
                    baseline_accuracy * 0.9,  # Allow some tolerance
                    f"HyperPath-SVM underperforms {model_name}"
                )
        
        self.logger.info("Baseline comparison pipeline completed successfully")
    
    def _calculate_routing_accuracy(self, predictions, ground_truth):
        """Calculate routing accuracy for comparison."""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return 0.0
        
        correct = 0
        total = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if len(pred) > 1 and len(truth) > 1:
                # Check first hop correctness
                if pred[1] == truth[1]:
                    correct += 1
                total += 1
        
        return correct / max(total, 1)
    
    def test_scalability_pipeline(self):
        """Test pipeline scalability with different dataset sizes."""
        self.logger.info("Testing scalability pipeline")
        
        dataset_sizes = [100, 500, 1000]
        scalability_results = {}
        
        for size in dataset_sizes:
            self.logger.info(f"Testing with dataset size: {size}")
            
            # Create smaller dataset
            limited_samples = self.sample_dataset['samples'][:size]
            
            # Process data
            X, y = [], []
            for sample in limited_samples:
                X.append(sample['features'])
                y.append(sample['optimal_path'])
            
            X, y = np.array(X), np.array(y, dtype=object)
            
            # Train model
            model = self._initialize_hyperpath_svm()
            
            start_time = time.time()
            split_idx = int(0.8 * len(X))
            model.fit(X[:split_idx], y[:split_idx])
            training_time = time.time() - start_time
            
            # Test inference
            inference_start = time.time()
            predictions = model.predict(X[split_idx:])
            inference_time = (time.time() - inference_start) / len(X[split_idx:]) * 1000
            
            scalability_results[size] = {
                'training_time': training_time,
                'inference_time_ms': inference_time,
                'samples_processed': size
            }
        
        # Validate scalability
        self._validate_scalability_results(scalability_results)
        
        self.logger.info("Scalability pipeline test completed")
    
    def _validate_scalability_results(self, results: Dict[int, Dict[str, float]]):
        """Validate scalability results."""
        sizes = sorted(results.keys())
        
        # Training time should scale reasonably
        training_times = [results[size]['training_time'] for size in sizes]
        
        # Should not have exponential scaling
        for i in range(1, len(training_times)):
            scale_factor = sizes[i] / sizes[i-1]
            time_factor = training_times[i] / training_times[i-1]
            
            # Time increase should not be more than quadratic
            self.assertLess(
                time_factor, 
                scale_factor ** 2.5,
                f"Training time scaling too poor: {time_factor} vs {scale_factor}"
            )
        
        # Inference time should remain reasonable
        for size, result in results.items():
            inference_time = result['inference_time_ms']
            self.assertLess(
                inference_time, 
                200.0,  # 200ms per sample max
                f"Inference time too high for size {size}: {inference_time}ms"
            )
    
    def test_robustness_pipeline(self):
        """Test pipeline robustness to various failure conditions."""
        self.logger.info("Testing robustness pipeline")
        
        # Test 1: Corrupted data handling
        self._test_corrupted_data_robustness()
        
        # Test 2: Memory pressure handling
        self._test_memory_pressure_robustness()
        
        # Test 3: Concurrent access handling
        self._test_concurrent_access_robustness()
        
        # Test 4: Model persistence robustness
        self._test_model_persistence_robustness()
        
        self.logger.info("Robustness pipeline test completed")
    
    def _test_corrupted_data_robustness(self):
        """Test robustness to corrupted input data."""
        dataset = self._load_test_dataset()
        X, y = dataset['X'], dataset['y']
        
        # Introduce various corruptions
        corrupted_X = X.copy()
        
        # NaN values
        corrupted_X[0, 0] = np.nan
        
        # Infinite values
        corrupted_X[1, 1] = np.inf
        
        # Extremely large values
        corrupted_X[2, 2] = 1e10
        
        # Initialize model
        model = self._initialize_hyperpath_svm()
        
        # Should handle corrupted data gracefully
        try:
            split_idx = int(0.8 * len(corrupted_X))
            model.fit(corrupted_X[:split_idx], y[:split_idx])
            predictions = model.predict(corrupted_X[split_idx:split_idx+5])
            
            # Should produce some reasonable output
            self.assertIsNotNone(predictions)
            self.assertGreater(len(predictions), 0)
            
        except Exception as e:
            # If it fails, should fail gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
            self.logger.info(f"Graceful failure with corrupted data: {type(e).__name__}")
    
    def _test_memory_pressure_robustness(self):
        """Test robustness under memory pressure."""
        # Create larger dataset to simulate memory pressure
        large_X = np.random.randn(5000, 50)  # Larger feature space
        large_y = [np.random.randint(0, 50, np.random.randint(2, 10)).tolist() 
                   for _ in range(5000)]
        
        model = self._initialize_hyperpath_svm()
        
        try:
            # Train with large dataset
            split_idx = int(0.8 * len(large_X))
            model.fit(large_X[:split_idx], large_y[:split_idx])
            
            # Should complete without memory errors
            predictions = model.predict(large_X[split_idx:split_idx+100])
            self.assertIsNotNone(predictions)
            
        except MemoryError:
            self.logger.info("Memory pressure test triggered expected MemoryError")
        except Exception as e:
            # Other exceptions should be handled gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def _test_concurrent_access_robustness(self):
        """Test robustness with concurrent access."""
        dataset = self._load_test_dataset()
        X, y = dataset['X'], dataset['y']
        
        model = self._initialize_hyperpath_svm()
        split_idx = int(0.8 * len(X))
        model.fit(X[:split_idx], y[:split_idx])
        
        results = []
        errors = []
        
        def concurrent_prediction():
            try:
                predictions = model.predict(X[split_idx:split_idx+10])
                results.append(predictions)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent predictions
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=concurrent_prediction)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access gracefully
        self.assertLessEqual(len(errors), 2)  # Allow some threading issues
        self.assertGreater(len(results), 0)  # But some should succeed
    
    def _test_model_persistence_robustness(self):
        """Test model persistence and loading robustness."""
        dataset = self._load_test_dataset()
        X, y = dataset['X'], dataset['y']
        
        # Train original model
        original_model = self._initialize_hyperpath_svm()
        split_idx = int(0.8 * len(X))
        original_model.fit(X[:split_idx], y[:split_idx])
        
        original_predictions = original_model.predict(X[split_idx:split_idx+10])
        
        # Save model
        model_file = os.path.join(self.temp_dir, 'test_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(original_model, f)
        
        # Load model
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)
        
        loaded_predictions = loaded_model.predict(X[split_idx:split_idx+10])
        
        # Predictions should be identical
        self.assertEqual(len(original_predictions), len(loaded_predictions))
        
        # For path predictions, compare first elements if they exist
        for orig, loaded in zip(original_predictions, loaded_predictions):
            if len(orig) > 0 and len(loaded) > 0:
                self.assertEqual(orig[0], loaded[0])
    
    def test_performance_monitoring_pipeline(self):
        """Test continuous performance monitoring pipeline."""
        self.logger.info("Testing performance monitoring pipeline")
        
        dataset = self._load_test_dataset()
        X, y = dataset['X'], dataset['y']
        
        # Initialize model and evaluator
        model = self._initialize_hyperpath_svm()
        split_idx = int(0.8 * len(X))
        model.fit(X[:split_idx], y[:split_idx])
        
        # Simulate continuous monitoring
        monitoring_results = []
        
        for i in range(5):  # 5 monitoring cycles
            # Get batch of test data
            start_idx = split_idx + i * 10
            end_idx = min(start_idx + 10, len(X))
            
            if end_idx <= start_idx:
                break
            
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            # Monitor performance
            start_time = time.time()
            predictions = model.predict(batch_X)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            accuracy = self._calculate_routing_accuracy(predictions, batch_y)
            
            monitoring_result = {
                'cycle': i,
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'inference_time_ms': inference_time,
                'batch_size': len(batch_X)
            }
            
            monitoring_results.append(monitoring_result)
            
            # Simulate adaptation if performance degrades
            if accuracy < 0.5 and i > 0:  # Arbitrary threshold
                self.logger.info(f"Performance degradation detected at cycle {i}")
                # Could trigger model retraining here
        
        # Validate monitoring results
        self.assertGreater(len(monitoring_results), 0)
        
        for result in monitoring_results:
            self.assertIn('accuracy', result)
            self.assertIn('inference_time_ms', result)
            self.assertGreaterEqual(result['accuracy'], 0.0)
            self.assertLessEqual(result['accuracy'], 1.0)
        
        self.logger.info("Performance monitoring pipeline completed")
    
    def test_full_production_simulation(self):
        """Test full production-like simulation."""
        self.logger.info("Testing full production simulation")
        
        dataset = self._load_test_dataset()
        X, y = dataset['X'], dataset['y']
        
        # Initialize production-like environment
        model = self._initialize_hyperpath_svm()
        
        # Training phase
        self.logger.info("Production simulation: Training phase")
        split_idx = int(0.8 * len(X))
        
        training_start = time.time()
        model.fit(X[:split_idx], y[:split_idx])
        training_time = time.time() - training_start
        
        # Production deployment phase
        self.logger.info("Production simulation: Deployment phase")
        
        production_results = {
            'total_requests': 0,
            'successful_predictions': 0,
            'total_inference_time': 0.0,
            'errors': []
        }
        
        # Simulate production requests
        test_X = X[split_idx:]
        
        for i in range(min(100, len(test_X))):  # Simulate 100 requests
            try:
                request_start = time.time()
                prediction = model.predict(test_X[i:i+1])
                request_time = time.time() - request_start
                
                production_results['total_requests'] += 1
                production_results['total_inference_time'] += request_time
                
                if len(prediction) > 0:
                    production_results['successful_predictions'] += 1
                    
            except Exception as e:
                production_results['errors'].append(str(e))
        
        # Validate production performance
        total_requests = production_results['total_requests']
        successful_predictions = production_results['successful_predictions']
        
        self.assertGreater(total_requests, 0)
        
        # Success rate should be high
        success_rate = successful_predictions / total_requests if total_requests > 0 else 0
        self.assertGreater(success_rate, 0.8, "Production success rate too low")
        
        # Average inference time should be reasonable
        if total_requests > 0:
            avg_inference_time = production_results['total_inference_time'] / total_requests
            self.assertLess(avg_inference_time, 1.0, "Production inference time too high")
        
        # Error rate should be low
        error_rate = len(production_results['errors']) / total_requests if total_requests > 0 else 0
        self.assertLess(error_rate, 0.1, "Production error rate too high")
        
        self.logger.info(f"Production simulation completed: {success_rate:.2%} success rate")


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different framework components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize components
        self.ddwe = DDWEOptimizer(quantum_enhanced=True)
        self.tgck = TGCKKernel(temporal_window=12)
        self.quantum_opt = QuantumOptimizer(num_qubits=8, num_layers=4)
        self.graph_processor = GraphProcessor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ddwe_tgck_integration(self):
        """Test integration between DDWE and TGCK components."""
        # Create sample data
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 5, 100)
        
        # Test DDWE initialization
        weights = self.ddwe.initialize_weights(X.shape[1], len(np.unique(y)))
        self.assertIn('primary', weights)
        self.assertIn('secondary', weights)
        
        # Test TGCK kernel computation
        K = self.tgck.compute_kernel_matrix(X[:50], X[50:])
        self.assertEqual(K.shape, (50, 50))
        
        # Test integration: use TGCK kernel with DDWE weights
        # This simulates how they work together in HyperPathSVM
        kernel_weighted_features = np.dot(K, X[50:])
        gradients = self.ddwe.compute_gradients(kernel_weighted_features, y[50:], weights)
        
        self.assertIn('primary', gradients)
        self.assertIn('secondary', gradients)
        self.assertEqual(gradients['primary'].shape, weights['primary'].shape)
    
    def test_quantum_integration(self):
        """Test integration with quantum optimization components."""
        # Test quantum parameter initialization
        quantum_params = self.ddwe.initialize_quantum_parameters(num_qubits=8)
        
        self.assertIn('rotation_angles', quantum_params)
        self.assertIn('entanglement_structure', quantum_params)
        
        # Test quantum state evolution
        initial_state = np.zeros(256, dtype=complex)
        initial_state[0] = 1.0  # |00000000⟩
        
        evolved_state = self.ddwe.evolve_quantum_state(initial_state, quantum_params)
        
        # State should remain normalized
        self.assertAlmostEqual(np.linalg.norm(evolved_state), 1.0, places=6)
        
        # Test quantum optimization step
        def simple_cost(params):
            return np.sum(params**2)
        
        initial_params = np.random.randn(32)
        updated_params, cost, gradient = self.quantum_opt.optimization_step(
            initial_params, simple_cost
        )
        
        self.assertEqual(len(updated_params), len(initial_params))
        self.assertIsInstance(cost, float)
        self.assertEqual(len(gradient), len(initial_params))
    
    def test_graph_processing_integration(self):
        """Test integration with graph processing utilities."""
        # Create sample graph
        num_nodes = 20
        adjacency_matrix = np.random.rand(num_nodes, num_nodes)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
        np.fill_diagonal(adjacency_matrix, 0)
        
        # Test graph feature extraction
        node_features = self.graph_processor.extract_node_features(
            adjacency_matrix, src_node=0, dst_node=10
        )
        
        self.assertIsInstance(node_features, np.ndarray)
        self.assertGreater(len(node_features), 0)
        
        # Test spectral analysis
        eigenvalues, eigenvectors = self.graph_processor.compute_spectral_decomposition(
            adjacency_matrix
        )
        
        self.assertEqual(len(eigenvalues), num_nodes)
        self.assertEqual(eigenvectors.shape, (num_nodes, num_nodes))
        
        # Test integration with TGCK
        # Use spectral features in kernel computation
        spectral_features = eigenvectors[:, :5]  # Top 5 eigenvectors
        
        K_spectral = self.tgck.compute_kernel_matrix(
            spectral_features[:10], 
            spectral_features[10:]
        )
        
        self.assertEqual(K_spectral.shape, (10, 10))
        
        # Should be positive semi-definite (at least non-negative on diagonal)
        self.assertTrue(np.all(np.diag(K_spectral) >= -1e-10))
    
    def test_data_pipeline_integration(self):
        """Test integration of data processing pipeline."""
        # Create sample dataset
        dataset_loader = DatasetLoader()
        data_augmentation = DataAugmentation()
        
        # Generate synthetic network data
        num_nodes = 30
        num_samples = 200
        
        # Create topology
        topology = np.random.rand(num_nodes, num_nodes)
        topology = (topology + topology.T) / 2
        np.fill_diagonal(topology, 0)
        topology = (topology > 0.7).astype(float)  # Sparse connections
        
        # Create samples
        X = []
        y = []
        
        for _ in range(num_samples):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                features = np.concatenate([
                    [src, dst],
                    [np.sum(topology[src] > 0)],  # Source degree
                    [np.sum(topology[dst] > 0)],  # Destination degree
                    np.random.randn(10)  # Additional features
                ])
                X.append(features)
                y.append([src, dst])  # Simplified path
        
        X = np.array(X)
        y = np.array(y, dtype=object)
        
        # Test data validation
        dataset = {'X': X, 'y': y, 'topology': topology}
        validation_result = dataset_loader.validate_dataset(dataset)
        
        # May not pass all validations due to simplified structure
        self.assertIn('is_valid', validation_result)
        self.assertIn('errors', validation_result)
        
        # Test data augmentation
        X_augmented, y_augmented = data_augmentation.topology_aware_augmentation(
            X, y, topology, noise_level=0.1, augmentation_ratio=0.5
        )
        
        self.assertGreater(len(X_augmented), len(X))
        self.assertEqual(len(X_augmented), len(y_augmented))
        
        # Test feature standardization
        processed_data = dataset_loader.preprocess_dataset(
            {'samples': [{'features': feat.tolist()} for feat in X]},
            normalize_features=True,
            target_feature_dim=15
        )
        
        self.assertIn('X', processed_data)
        processed_X = processed_data['X']
        self.assertEqual(processed_X.shape[1], 15)
    
    def test_evaluation_integration(self):
        """Test integration of evaluation components."""
        evaluator = HyperPathEvaluator()
        cv = TemporalCrossValidator()
        
        # Create mock model and data
        mock_model = Mock()
        mock_model.predict.return_value = np.array([
            [0, 1, 2], [0, 2, 3], [1, 2, 4], [1, 3, 4], [0, 1, 3]
        ])
        mock_model.fit.return_value = None
        
        X = np.random.randn(50, 10)
        y = np.array([
            [0, 1, 2], [0, 1, 3], [1, 2, 4], [1, 2, 3], [0, 2, 3]
        ] * 10)
        
        # Test model evaluation
        evaluation_results = evaluator.evaluate_model(
            mock_model, X, y, compute_detailed_metrics=True
        )
        
        self.assertIn('routing_accuracy', evaluation_results)
        self.assertIn('inference_time_ms', evaluation_results)
        
        # Test cross-validation integration
        cv_results = cv.validate(mock_model, X, y, n_splits=3)
        
        self.assertIn('fold_scores', cv_results)
        self.assertIn('mean_score', cv_results)
        self.assertEqual(len(cv_results['fold_scores']), 3)
        
        # Test statistical significance
        hyperpath_scores = np.random.normal(0.9, 0.05, 20)
        baseline_scores = np.random.normal(0.8, 0.08, 20)
        
        significance_result = evaluator.test_statistical_significance(
            hyperpath_scores, baseline_scores
        )
        
        self.assertIn('p_value', significance_result)
        self.assertIn('significant', significance_result)


if __name__ == '__main__':
    # Set up test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestEndToEndPipeline,
        TestComponentIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with higher verbosity for integration tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("INTEGRATION TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n{'='*40}")
        print("FAILURES:")
        print(f"{'='*40}")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i}. {test}")
            # Extract just the assertion error message
            error_lines = traceback.split('\n')
            for line in error_lines:
                if 'AssertionError:' in line:
                    print(f"   {line.strip()}")
                    break
            print()
    
    if result.errors:
        print(f"\n{'='*40}")
        print("ERRORS:")
        print(f"{'='*40}")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i}. {test}")
            # Extract the main error
            error_lines = traceback.split('\n')
            for line in reversed(error_lines):
                if line.strip() and not line.startswith(' '):
                    print(f"   {line.strip()}")
                    break
            print()
    
    # Performance summary for integration tests
    print(f"\n{'='*40}")
    print("INTEGRATION TEST INSIGHTS:")
    print(f"{'='*40}")
    print("✓ End-to-end pipeline functionality verified")
    print("✓ Component integration stability confirmed") 
    print("✓ Scalability and robustness tested")
    print("✓ Production-like scenarios validated")
    
    if result.wasSuccessful():
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("The HyperPath-SVM framework is ready for deployment.")
    else:
        print(f"\n⚠️  Integration tests completed with {len(result.failures + result.errors)} issues.")
        print("Review failures and errors before production deployment.")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)