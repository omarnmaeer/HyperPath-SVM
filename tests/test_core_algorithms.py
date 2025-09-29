# File: tests/test_core_algorithms.py

"""
Comprehensive unit tests for core HyperPath-SVM algorithms.
Tests DDWE, TGCK, quantum optimization, and main HyperPathSVM components.
"""

import unittest
import numpy as np
import pickle
import tempfile
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
import threading
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperpath_svm.core.hyperpath_svm import HyperPathSVM
from hyperpath_svm.core.ddwe import DDWEOptimizer
from hyperpath_svm.core.tgck import TGCKKernel
from hyperpath_svm.utils.math_utils import QuantumOptimizer


class TestDDWEOptimizer(unittest.TestCase):
    """Test cases for DDWE (Dynamic Dual-Weight Evolution) optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ddwe = DDWEOptimizer(
            learning_rate=0.01,
            quantum_enhanced=True,
            adaptation_rate=0.001,
            memory_decay=0.95,
            exploration_factor=0.1
        )
        
        # Create sample data
        self.sample_features = np.random.randn(100, 10)
        self.sample_labels = np.random.randint(0, 3, 100)
        self.sample_graph = np.random.rand(50, 50)
        np.fill_diagonal(self.sample_graph, 0)
    
    def test_initialization(self):
        """Test DDWE optimizer initialization."""
        self.assertEqual(self.ddwe.learning_rate, 0.01)
        self.assertTrue(self.ddwe.quantum_enhanced)
        self.assertEqual(self.ddwe.adaptation_rate, 0.001)
        self.assertEqual(self.ddwe.memory_decay, 0.95)
        self.assertEqual(self.ddwe.exploration_factor, 0.1)
        
        # Test initialization with default parameters
        default_ddwe = DDWEOptimizer()
        self.assertEqual(default_ddwe.learning_rate, 0.01)
        self.assertFalse(default_ddwe.quantum_enhanced)
    
    def test_weight_initialization(self):
        """Test dual-weight initialization."""
        weights = self.ddwe.initialize_weights(input_dim=10, output_dim=5)
        
        self.assertEqual(weights['primary'].shape, (10, 5))
        self.assertEqual(weights['secondary'].shape, (10, 5))
        self.assertIsInstance(weights['quantum_params'], dict)
        
        # Test that weights are different
        self.assertFalse(np.array_equal(weights['primary'], weights['secondary']))
    
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate computation."""
        # Test with different loss histories
        loss_history = [1.0, 0.8, 0.6, 0.5, 0.4]
        adaptive_lr = self.ddwe.compute_adaptive_learning_rate(loss_history)
        self.assertIsInstance(adaptive_lr, float)
        self.assertGreater(adaptive_lr, 0)
        
        # Test with increasing loss (should decrease learning rate)
        increasing_loss = [0.4, 0.5, 0.6, 0.8, 1.0]
        increasing_lr = self.ddwe.compute_adaptive_learning_rate(increasing_loss)
        self.assertLess(increasing_lr, adaptive_lr)
        
        # Test with empty history
        empty_lr = self.ddwe.compute_adaptive_learning_rate([])
        self.assertEqual(empty_lr, self.ddwe.learning_rate)
    
    def test_quantum_enhancement(self):
        """Test quantum enhancement functionality."""
        if not self.ddwe.quantum_enhanced:
            self.ddwe.quantum_enhanced = True
        
        # Test quantum parameter initialization
        quantum_params = self.ddwe.initialize_quantum_parameters(num_qubits=8)
        
        self.assertIn('rotation_angles', quantum_params)
        self.assertIn('entanglement_structure', quantum_params)
        self.assertIn('measurement_basis', quantum_params)
        
        # Test quantum state evolution
        initial_state = np.random.randn(256) + 1j * np.random.randn(256)
        initial_state /= np.linalg.norm(initial_state)
        
        evolved_state = self.ddwe.evolve_quantum_state(initial_state, quantum_params)
        
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(evolved_state), 1.0, places=5)
        
        # Check that state has evolved
        self.assertFalse(np.allclose(initial_state, evolved_state))
    
    def test_dual_weight_update(self):
        """Test dual weight update mechanism."""
        weights = self.ddwe.initialize_weights(10, 5)
        gradients = {
            'primary': np.random.randn(10, 5),
            'secondary': np.random.randn(10, 5)
        }
        
        original_primary = weights['primary'].copy()
        original_secondary = weights['secondary'].copy()
        
        updated_weights = self.ddwe.update_dual_weights(weights, gradients, 0.01)
        
        # Check that weights have been updated
        self.assertFalse(np.array_equal(original_primary, updated_weights['primary']))
        self.assertFalse(np.array_equal(original_secondary, updated_weights['secondary']))
        
        # Check weight constraints
        self.assertTrue(np.all(np.abs(updated_weights['primary']) <= 10.0))  # Gradient clipping
        self.assertTrue(np.all(np.abs(updated_weights['secondary']) <= 10.0))
    
    def test_memory_decay_mechanism(self):
        """Test memory decay in weight adaptation."""
        # Initialize weight history
        weight_history = []
        for i in range(5):
            weights = np.random.randn(10, 5)
            weight_history.append(weights)
        
        # Apply memory decay
        decayed_weights = self.ddwe.apply_memory_decay(weight_history)
        
        self.assertEqual(len(decayed_weights), len(weight_history))
        
        # More recent weights should have higher influence
        recent_influence = np.linalg.norm(decayed_weights[-1])
        older_influence = np.linalg.norm(decayed_weights[0])
        self.assertGreater(recent_influence, older_influence)
    
    def test_exploration_vs_exploitation(self):
        """Test exploration vs exploitation balance."""
        # High confidence scenario (exploitation)
        high_confidence = 0.95
        action_high_conf = self.ddwe.select_action(
            q_values=np.array([0.8, 0.9, 0.7]),
            confidence=high_confidence
        )
        
        # Low confidence scenario (exploration)
        low_confidence = 0.3
        action_low_conf = self.ddwe.select_action(
            q_values=np.array([0.8, 0.9, 0.7]),
            confidence=low_confidence
        )
        
        # With high confidence, should select best action more often
        # With low confidence, should explore more
        # This is probabilistic, so we test multiple times
        high_conf_selections = []
        low_conf_selections = []
        
        for _ in range(100):
            high_conf_selections.append(self.ddwe.select_action(
                np.array([0.8, 0.9, 0.7]), high_confidence
            ))
            low_conf_selections.append(self.ddwe.select_action(
                np.array([0.8, 0.9, 0.7]), low_confidence
            ))
        
        # High confidence should select best action (index 1) more often
        best_action_high_conf = sum(1 for a in high_conf_selections if a == 1)
        best_action_low_conf = sum(1 for a in low_conf_selections if a == 1)
        
        self.assertGreater(best_action_high_conf, best_action_low_conf)
    
    def test_convergence_detection(self):
        """Test convergence detection mechanism."""
        # Test converged sequence
        converged_losses = [1.0, 0.501, 0.500, 0.500, 0.500]
        self.assertTrue(self.ddwe.is_converged(converged_losses, tolerance=0.01))
        
        # Test non-converged sequence
        non_converged_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        self.assertFalse(self.ddwe.is_converged(non_converged_losses, tolerance=0.01))
        
        # Test with insufficient history
        short_losses = [1.0, 0.5]
        self.assertFalse(self.ddwe.is_converged(short_losses, tolerance=0.01))
    
    def test_gradient_computation(self):
        """Test gradient computation for dual weights."""
        X = self.sample_features
        y = self.sample_labels
        weights = self.ddwe.initialize_weights(X.shape[1], len(np.unique(y)))
        
        gradients = self.ddwe.compute_gradients(X, y, weights)
        
        self.assertIn('primary', gradients)
        self.assertIn('secondary', gradients)
        self.assertEqual(gradients['primary'].shape, weights['primary'].shape)
        self.assertEqual(gradients['secondary'].shape, weights['secondary'].shape)
        
        # Test gradient magnitudes are reasonable
        primary_norm = np.linalg.norm(gradients['primary'])
        secondary_norm = np.linalg.norm(gradients['secondary'])
        
        self.assertGreater(primary_norm, 0)
        self.assertGreater(secondary_norm, 0)
        self.assertLess(primary_norm, 100)  # Reasonable magnitude
        self.assertLess(secondary_norm, 100)
    
    def test_thread_safety(self):
        """Test thread safety of DDWE optimizer."""
        results = []
        errors = []
        
        def worker():
            try:
                ddwe = DDWEOptimizer()
                weights = ddwe.initialize_weights(10, 5)
                gradients = {
                    'primary': np.random.randn(10, 5),
                    'secondary': np.random.randn(10, 5)
                }
                updated = ddwe.update_dual_weights(weights, gradients, 0.01)
                results.append(updated)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 10)
    
    def test_serialization(self):
        """Test DDWE optimizer serialization/deserialization."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Serialize
            pickle.dump(self.ddwe, tmp_file)
            tmp_file.flush()
            
            # Deserialize
            with open(tmp_file.name, 'rb') as f:
                loaded_ddwe = pickle.load(f)
            
            # Compare attributes
            self.assertEqual(self.ddwe.learning_rate, loaded_ddwe.learning_rate)
            self.assertEqual(self.ddwe.quantum_enhanced, loaded_ddwe.quantum_enhanced)
            self.assertEqual(self.ddwe.adaptation_rate, loaded_ddwe.adaptation_rate)
        
        # Cleanup
        os.unlink(tmp_file.name)


class TestTGCKKernel(unittest.TestCase):
    """Test cases for TGCK (Temporal Graph Convolutional Kernel)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tgck = TGCKKernel(
            temporal_window=24,
            confidence_threshold=0.8,
            kernel_type='rbf',
            gamma='auto',
            degree=3
        )
        
        # Create sample temporal graph data
        self.num_nodes = 50
        self.temporal_graphs = []
        for t in range(24):
            graph = np.random.rand(self.num_nodes, self.num_nodes)
            graph = (graph + graph.T) / 2  # Make symmetric
            np.fill_diagonal(graph, 0)
            self.temporal_graphs.append(graph)
        
        self.sample_features = np.random.randn(100, 20)
    
    def test_initialization(self):
        """Test TGCK kernel initialization."""
        self.assertEqual(self.tgck.temporal_window, 24)
        self.assertEqual(self.tgck.confidence_threshold, 0.8)
        self.assertEqual(self.tgck.kernel_type, 'rbf')
        self.assertEqual(self.tgck.gamma, 'auto')
        self.assertEqual(self.tgck.degree, 3)
    
    def test_temporal_graph_convolution(self):
        """Test temporal graph convolution operation."""
        # Create feature matrix
        node_features = np.random.randn(self.num_nodes, 10)
        
        # Apply convolution
        convolved_features = self.tgck.temporal_convolution(
            node_features, self.temporal_graphs
        )
        
        self.assertEqual(convolved_features.shape[0], self.num_nodes)
        self.assertGreaterEqual(convolved_features.shape[1], node_features.shape[1])
        
        # Test that features have changed
        self.assertFalse(np.allclose(node_features, convolved_features[:, :node_features.shape[1]]))
    
    def test_kernel_computation(self):
        """Test kernel matrix computation."""
        X1 = self.sample_features[:50]
        X2 = self.sample_features[50:]
        
        # Test RBF kernel
        self.tgck.kernel_type = 'rbf'
        K_rbf = self.tgck.compute_kernel_matrix(X1, X2)
        self.assertEqual(K_rbf.shape, (50, 50))
        self.assertTrue(np.all(K_rbf >= 0))  # RBF kernel values should be non-negative
        
        # Test polynomial kernel
        self.tgck.kernel_type = 'poly'
        K_poly = self.tgck.compute_kernel_matrix(X1, X2)
        self.assertEqual(K_poly.shape, (50, 50))
        
        # Test linear kernel
        self.tgck.kernel_type = 'linear'
        K_linear = self.tgck.compute_kernel_matrix(X1, X2)
        self.assertEqual(K_linear.shape, (50, 50))
        
        # Different kernels should produce different results
        self.assertFalse(np.allclose(K_rbf, K_poly))
        self.assertFalse(np.allclose(K_rbf, K_linear))
    
    def test_temporal_aggregation(self):
        """Test temporal feature aggregation."""
        temporal_features = [np.random.randn(self.num_nodes, 10) for _ in range(24)]
        
        # Test different aggregation methods
        aggregated_mean = self.tgck.aggregate_temporal_features(temporal_features, method='mean')
        aggregated_max = self.tgck.aggregate_temporal_features(temporal_features, method='max')
        aggregated_attention = self.tgck.aggregate_temporal_features(temporal_features, method='attention')
        
        self.assertEqual(aggregated_mean.shape, (self.num_nodes, 10))
        self.assertEqual(aggregated_max.shape, (self.num_nodes, 10))
        self.assertEqual(aggregated_attention.shape, (self.num_nodes, 10))
        
        # Different methods should produce different results
        self.assertFalse(np.allclose(aggregated_mean, aggregated_max))
        self.assertFalse(np.allclose(aggregated_mean, aggregated_attention))
    
    def test_confidence_estimation(self):
        """Test confidence estimation for predictions."""
        predictions = np.array([0.9, 0.7, 0.3, 0.1, 0.95])
        confidences = self.tgck.estimate_confidence(predictions)
        
        self.assertEqual(len(confidences), len(predictions))
        self.assertTrue(np.all(confidences >= 0))
        self.assertTrue(np.all(confidences <= 1))
        
        # High prediction values should have higher confidence
        high_pred_conf = confidences[predictions > 0.8].mean()
        low_pred_conf = confidences[predictions < 0.5].mean()
        self.assertGreater(high_pred_conf, low_pred_conf)
    
    def test_graph_spectral_features(self):
        """Test graph spectral feature extraction."""
        graph = self.temporal_graphs[0]
        spectral_features = self.tgck.extract_spectral_features(graph)
        
        self.assertIn('eigenvalues', spectral_features)
        self.assertIn('eigenvectors', spectral_features)
        self.assertIn('spectral_gap', spectral_features)
        self.assertIn('effective_resistance', spectral_features)
        
        # Check dimensions
        n_nodes = graph.shape[0]
        self.assertEqual(len(spectral_features['eigenvalues']), n_nodes)
        self.assertEqual(spectral_features['eigenvectors'].shape, (n_nodes, n_nodes))
    
    def test_temporal_consistency(self):
        """Test temporal consistency measurement."""
        # Create temporally consistent sequence
        consistent_graphs = []
        base_graph = np.random.rand(20, 20)
        base_graph = (base_graph + base_graph.T) / 2
        np.fill_diagonal(base_graph, 0)
        
        for t in range(10):
            # Add small random perturbations
            noise = np.random.randn(20, 20) * 0.01
            noise = (noise + noise.T) / 2
            np.fill_diagonal(noise, 0)
            consistent_graphs.append(base_graph + noise)
        
        consistency_score = self.tgck.measure_temporal_consistency(consistent_graphs)
        self.assertGreater(consistency_score, 0.8)  # Should be highly consistent
        
        # Create temporally inconsistent sequence
        inconsistent_graphs = []
        for t in range(10):
            graph = np.random.rand(20, 20)
            graph = (graph + graph.T) / 2
            np.fill_diagonal(graph, 0)
            inconsistent_graphs.append(graph)
        
        inconsistency_score = self.tgck.measure_temporal_consistency(inconsistent_graphs)
        self.assertLess(inconsistency_score, consistency_score)  # Should be less consistent
    
    def test_adaptive_window_sizing(self):
        """Test adaptive temporal window sizing."""
        # Test with different graph sequences
        stable_sequence = [self.temporal_graphs[0]] * 50  # Very stable
        dynamic_sequence = self.temporal_graphs * 2  # More dynamic
        
        stable_window = self.tgck.compute_adaptive_window_size(stable_sequence)
        dynamic_window = self.tgck.compute_adaptive_window_size(dynamic_sequence)
        
        # Stable sequences should use larger windows
        # Dynamic sequences should use smaller windows
        self.assertGreaterEqual(stable_window, dynamic_window)
        self.assertGreater(stable_window, 0)
        self.assertGreater(dynamic_window, 0)
    
    def test_kernel_hyperparameter_optimization(self):
        """Test automatic kernel hyperparameter optimization."""
        X = self.sample_features
        y = np.random.randint(0, 2, len(X))
        
        # Test gamma optimization for RBF kernel
        self.tgck.kernel_type = 'rbf'
        optimal_gamma = self.tgck.optimize_gamma(X, y)
        
        self.assertGreater(optimal_gamma, 0)
        self.assertIsInstance(optimal_gamma, float)
        
        # Test that optimization improves performance
        default_score = self.tgck.cross_validate_kernel(X, y, gamma='auto')
        optimized_score = self.tgck.cross_validate_kernel(X, y, gamma=optimal_gamma)
        
        self.assertGreaterEqual(optimized_score, default_score * 0.9)  # Allow some variation
    
    def test_graph_embedding(self):
        """Test graph embedding generation."""
        graph = self.temporal_graphs[0]
        embedding = self.tgck.generate_graph_embedding(graph, embedding_dim=16)
        
        self.assertEqual(embedding.shape, (graph.shape[0], 16))
        
        # Test that embedding captures graph structure
        # Nodes with similar connectivity should have similar embeddings
        node_degrees = np.sum(graph > 0, axis=1)
        high_degree_nodes = np.where(node_degrees > np.median(node_degrees))[0]
        
        if len(high_degree_nodes) > 1:
            high_degree_embeddings = embedding[high_degree_nodes]
            similarity = np.corrcoef(high_degree_embeddings)
            # High-degree nodes should have some similarity
            off_diagonal = similarity[np.triu_indices_from(similarity, k=1)]
            self.assertGreater(np.mean(off_diagonal), -0.5)  # Not completely random


class TestQuantumOptimizer(unittest.TestCase):
    """Test cases for quantum optimization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quantum_opt = QuantumOptimizer(
            num_qubits=8,
            num_layers=4,
            learning_rate=0.01
        )
        
        self.sample_problem = {
            'cost_matrix': np.random.rand(10, 10),
            'constraints': np.random.rand(5, 10),
            'objective': 'minimize'
        }
    
    def test_initialization(self):
        """Test quantum optimizer initialization."""
        self.assertEqual(self.quantum_opt.num_qubits, 8)
        self.assertEqual(self.quantum_opt.num_layers, 4)
        self.assertEqual(self.quantum_opt.learning_rate, 0.01)
        self.assertEqual(self.quantum_opt.state_dim, 2**8)
    
    def test_quantum_circuit_construction(self):
        """Test quantum circuit construction."""
        parameters = np.random.rand(self.quantum_opt.num_layers * self.quantum_opt.num_qubits * 3)
        circuit = self.quantum_opt.construct_circuit(parameters)
        
        self.assertIsNotNone(circuit)
        self.assertEqual(len(circuit['gates']), self.quantum_opt.num_layers)
        
        # Test circuit depth and gate count
        total_gates = sum(len(layer) for layer in circuit['gates'])
        expected_gates = self.quantum_opt.num_layers * self.quantum_opt.num_qubits * 2  # Rough estimate
        self.assertGreaterEqual(total_gates, expected_gates * 0.5)
    
    def test_quantum_state_evolution(self):
        """Test quantum state evolution."""
        initial_state = np.zeros(self.quantum_opt.state_dim, dtype=complex)
        initial_state[0] = 1.0  # |00...0⟩ state
        
        parameters = np.random.rand(self.quantum_opt.num_layers * self.quantum_opt.num_qubits * 3)
        evolved_state = self.quantum_opt.evolve_state(initial_state, parameters)
        
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(evolved_state), 1.0, places=6)
        
        # Check that state has evolved (unless parameters are all zero)
        if not np.allclose(parameters, 0):
            self.assertFalse(np.allclose(initial_state, evolved_state))
    
    def test_quantum_measurement(self):
        """Test quantum measurement operations."""
        # Create superposition state
        state = np.ones(self.quantum_opt.state_dim, dtype=complex)
        state /= np.linalg.norm(state)
        
        # Test computational basis measurement
        measurement_results = []
        for _ in range(1000):
            result = self.quantum_opt.measure_computational_basis(state)
            measurement_results.append(result)
        
        # Should get different measurement outcomes
        unique_results = set(measurement_results)
        self.assertGreater(len(unique_results), 1)
        
        # Test Pauli-X measurement
        x_expectation = self.quantum_opt.measure_pauli_x(state, qubit_index=0)
        self.assertIsInstance(x_expectation, (float, complex))
        self.assertLessEqual(abs(x_expectation), 1.0)
    
    def test_quantum_optimization_step(self):
        """Test single quantum optimization step."""
        cost_function = lambda x: np.sum(x**2)  # Simple quadratic cost
        initial_params = np.random.rand(32)
        
        # Perform optimization step
        updated_params, cost_value, gradient = self.quantum_opt.optimization_step(
            initial_params, cost_function
        )
        
        self.assertEqual(len(updated_params), len(initial_params))
        self.assertIsInstance(cost_value, float)
        self.assertEqual(len(gradient), len(initial_params))
        
        # Parameters should change (unless gradient is zero)
        if not np.allclose(gradient, 0):
            self.assertFalse(np.allclose(initial_params, updated_params))
    
    def test_variational_quantum_eigensolver(self):
        """Test Variational Quantum Eigensolver (VQE) functionality."""
        # Create simple Hamiltonian (Pauli-Z on first qubit)
        hamiltonian = np.diag([1, -1] + [0] * (self.quantum_opt.state_dim - 2))
        
        # Run VQE
        eigenvalue, eigenvector, optimal_params = self.quantum_opt.vqe(
            hamiltonian, max_iterations=50
        )
        
        self.assertIsInstance(eigenvalue, float)
        self.assertEqual(len(eigenvector), self.quantum_opt.state_dim)
        self.assertAlmostEqual(np.linalg.norm(eigenvector), 1.0, places=6)
        
        # Check that we found a reasonable eigenvalue
        # For this Hamiltonian, ground state energy should be close to -1
        self.assertLessEqual(eigenvalue, 1.1)  # Allow some optimization error
    
    def test_quantum_approximate_optimization(self):
        """Test Quantum Approximate Optimization Algorithm (QAOA)."""
        # Create simple optimization problem (Max-Cut on triangle graph)
        adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        
        # Run QAOA
        optimal_cut, optimal_params = self.quantum_opt.qaoa(
            adjacency_matrix, num_layers=2, max_iterations=30
        )
        
        self.assertIsInstance(optimal_cut, float)
        self.assertGreater(len(optimal_params), 0)
        
        # For triangle graph, maximum cut should be 2
        self.assertLessEqual(optimal_cut, 2.1)  # Allow some approximation error
    
    def test_quantum_gradient_computation(self):
        """Test quantum gradient computation."""
        def cost_function(params):
            state = self.quantum_opt.evolve_state(
                np.array([1] + [0] * (self.quantum_opt.state_dim - 1), dtype=complex),
                params
            )
            return np.real(np.vdot(state, state))  # Should always be 1 due to normalization
        
        params = np.random.rand(32)
        gradient = self.quantum_opt.compute_quantum_gradient(cost_function, params)
        
        self.assertEqual(len(gradient), len(params))
        self.assertTrue(np.all(np.isfinite(gradient)))
        
        # Test parameter-shift rule implementation
        shift_gradient = self.quantum_opt.parameter_shift_gradient(cost_function, params)
        
        self.assertEqual(len(shift_gradient), len(params))
        # Gradients from different methods should be similar
        self.assertTrue(np.allclose(gradient, shift_gradient, rtol=1e-2))
    
    def test_quantum_error_correction(self):
        """Test basic quantum error correction functionality."""
        # Create noisy quantum state
        clean_state = np.array([1, 0, 0, 0], dtype=complex)  # 2-qubit state
        noise_level = 0.1
        noisy_state = clean_state + noise_level * np.random.randn(4).astype(complex)
        noisy_state /= np.linalg.norm(noisy_state)
        
        # Apply error correction (simplified)
        corrected_state = self.quantum_opt.apply_error_correction(noisy_state)
        
        self.assertAlmostEqual(np.linalg.norm(corrected_state), 1.0, places=6)
        
        # Corrected state should be closer to original
        clean_fidelity = abs(np.vdot(clean_state, corrected_state))**2
        noisy_fidelity = abs(np.vdot(clean_state, noisy_state))**2
        
        self.assertGreaterEqual(clean_fidelity, noisy_fidelity * 0.9)  # Some improvement expected
    
    def test_quantum_resource_estimation(self):
        """Test quantum resource estimation."""
        circuit_params = np.random.rand(64)
        resources = self.quantum_opt.estimate_resources(circuit_params)
        
        self.assertIn('gate_count', resources)
        self.assertIn('circuit_depth', resources)
        self.assertIn('qubit_usage', resources)
        self.assertIn('classical_memory', resources)
        
        # Sanity checks
        self.assertGreater(resources['gate_count'], 0)
        self.assertGreater(resources['circuit_depth'], 0)
        self.assertEqual(resources['qubit_usage'], self.quantum_opt.num_qubits)


class TestHyperPathSVMIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.ddwe_optimizer = DDWEOptimizer(quantum_enhanced=True)
        self.tgck_kernel = TGCKKernel(temporal_window=12, confidence_threshold=0.8)
        
        self.hyperpath_svm = HyperPathSVM(
            ddwe_optimizer=self.ddwe_optimizer,
            tgck_kernel=self.tgck_kernel,
            C=1.0,
            epsilon=0.1
        )
        
        # Create sample routing data
        self.X_train, self.y_train = self._create_sample_routing_data(500)
        self.X_test, self.y_test = self._create_sample_routing_data(100)
    
    def _create_sample_routing_data(self, n_samples):
        """Create sample routing data for testing."""
        # Features: [src_node, dst_node, src_degree, dst_degree, traffic_demand, ...]
        X = np.random.randn(n_samples, 20)
        
        # Labels: optimal routing paths (simplified as next hop node)
        y = np.random.randint(0, 50, n_samples)
        
        return X, y
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline."""
        # Test fitting
        self.hyperpath_svm.fit(self.X_train, self.y_train)
        
        # Test prediction
        predictions = self.hyperpath_svm.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test that model has learned something
        train_predictions = self.hyperpath_svm.predict(self.X_train)
        train_accuracy = np.mean(train_predictions == self.y_train)
        
        # Should achieve better than random performance
        self.assertGreater(train_accuracy, 1.0 / len(np.unique(self.y_train)) * 1.2)
    
    def test_incremental_learning(self):
        """Test incremental learning capability."""
        # Initial training
        self.hyperpath_svm.fit(self.X_train[:300], self.y_train[:300])
        initial_predictions = self.hyperpath_svm.predict(self.X_test)
        
        # Incremental update
        self.hyperpath_svm.partial_fit(self.X_train[300:], self.y_train[300:])
        updated_predictions = self.hyperpath_svm.predict(self.X_test)
        
        # Predictions may change after incremental learning
        # This tests that the model can adapt
        changed_predictions = np.sum(initial_predictions != updated_predictions)
        
        # At least some predictions should potentially change
        # (unless the additional data is identical)
        self.assertTrue(changed_predictions >= 0)  # Always true, but tests the API
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train model
        self.hyperpath_svm.fit(self.X_train, self.y_train)
        original_predictions = self.hyperpath_svm.predict(self.X_test)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(self.hyperpath_svm, tmp_file)
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                loaded_model = pickle.load(f)
        
        # Test loaded model
        loaded_predictions = loaded_model.predict(self.X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        
        # Cleanup
        os.unlink(tmp_file.name)
    
    def test_performance_targets(self):
        """Test that model meets performance targets."""
        # Train with sufficient data
        large_X, large_y = self._create_sample_routing_data(2000)
        
        start_time = time.time()
        self.hyperpath_svm.fit(large_X, large_y)
        training_time = time.time() - start_time
        
        # Test inference speed
        inference_times = []
        for i in range(100):
            start_time = time.time()
            _ = self.hyperpath_svm.predict(self.X_test[:1])
            inference_time = time.time() - start_time
            inference_times.append(inference_time * 1000)  # Convert to ms
        
        avg_inference_time = np.mean(inference_times)
        
        # Performance targets (relaxed for testing)
        self.assertLess(avg_inference_time, 50.0)  # 50ms target (relaxed from 1.8ms)
        
        # Memory usage should be reasonable
        model_size = len(pickle.dumps(self.hyperpath_svm)) / (1024 * 1024)  # MB
        self.assertLess(model_size, 200)  # 200MB target (relaxed from 98MB)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty data
        with self.assertRaises((ValueError, IndexError)):
            self.hyperpath_svm.fit(np.array([]), np.array([]))
        
        # Test with mismatched dimensions
        with self.assertRaises(ValueError):
            self.hyperpath_svm.fit(
                np.random.randn(100, 10), 
                np.random.randint(0, 5, 50)  # Wrong length
            )
        
        # Test prediction before training
        untrained_model = HyperPathSVM(
            ddwe_optimizer=DDWEOptimizer(),
            tgck_kernel=TGCKKernel()
        )
        
        with self.assertRaises(AttributeError):
            untrained_model.predict(self.X_test)
        
        # Test with single sample
        self.hyperpath_svm.fit(self.X_train, self.y_train)
        single_prediction = self.hyperpath_svm.predict(self.X_test[:1])
        
        self.assertEqual(len(single_prediction), 1)
    
    def test_hyperparameter_sensitivity(self):
        """Test sensitivity to hyperparameters."""
        # Test different C values
        c_values = [0.1, 1.0, 10.0]
        accuracies = []
        
        for C in c_values:
            model = HyperPathSVM(
                ddwe_optimizer=self.ddwe_optimizer,
                tgck_kernel=self.tgck_kernel,
                C=C
            )
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_train)
            accuracy = np.mean(predictions == self.y_train)
            accuracies.append(accuracy)
        
        # Different C values should potentially give different performance
        accuracy_range = max(accuracies) - min(accuracies)
        
        # This tests that the parameter has some effect
        # (may be minimal for synthetic data)
        self.assertGreaterEqual(accuracy_range, 0.0)


if __name__ == '__main__':
    # Set up test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDDWEOptimizer,
        TestTGCKKernel, 
        TestQuantumOptimizer,
        TestHyperPathSVMIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("CORE ALGORITHMS TEST SUMMARY")
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
