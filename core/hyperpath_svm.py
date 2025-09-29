# File: hyperpath_svm/core/hyperpath_svm.py
"""
Main HyperPath-SVM Implementation

This module implements the core HyperPath-SVM algorithm with three key innovations:
1. Dynamic Discriminative Weight Evolution (DDWE)
2. Temporal Graph Convolution Kernel (TGCK)  
3. Quantum-Inspired Superposition Optimization

"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import warnings

from .ddwe import DDWE
from .tgck import TGCK
from .quantum_optimizer import QuantumOptimizer
from .support_vector_manager import SupportVectorManager
from .memory_hierarchy import MemoryHierarchy
from ..utils.math_utils import kernel_matrix_computation, eigen_decomposition
from ..utils.graph_utils import graph_laplacian, shortest_path_matrix
from ..data.network_graph import NetworkGraph

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HyperPathSVM(BaseEstimator, ClassifierMixin):
    
    def __init__(self, config: Dict, C: float = 10.0, epsilon: float = 0.001,
                 max_support_vectors: int = 5000, random_state: int = 42):
        
        # Store configuration
        self.config = config
        self.C = C
        self.epsilon = epsilon
        self.max_support_vectors = max_support_vectors
        self.random_state = random_state
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Initialize core components
        self._initialize_components()
        
        # Training state
        self.is_fitted_ = False
        self.training_stats_ = {}
        self.performance_history_ = []
        
        # Cache for performance optimization
        self._kernel_cache = {}
        self._graph_cache = {}
        
        logger.info("HyperPath-SVM initialized with quantum-enhanced adaptive learning")
    
    def _initialize_components(self):
        """Initialize all core algorithm components"""
        
        # Dynamic Discriminative Weight Evolution
        self.ddwe = DDWE(
            config=self.config.get('ddwe', {}),
            short_term_window=self.config.get('ddwe', {}).get('short_term_window', 300),
            medium_term_window=self.config.get('ddwe', {}).get('medium_term_window', 14400),
            long_term_window=self.config.get('ddwe', {}).get('long_term_window', 86400)
        )
        
        # Temporal Graph Convolution Kernel
        self.tgck = TGCK(
            config=self.config.get('tgck', {}),
            gamma_spatial=self.config.get('tgck', {}).get('gamma_spatial', 0.01),
            gamma_temporal=self.config.get('tgck', {}).get('gamma_temporal', 0.001)
        )
        
        # Quantum-Inspired Optimizer
        self.quantum_optimizer = QuantumOptimizer(
            config=self.config.get('quantum', {}),
            num_configurations=self.config.get('quantum', {}).get('num_configurations', 32),
            max_iterations=self.config.get('quantum', {}).get('max_iterations', 1000)
        )
        
        # Support Vector Manager for continuous learning
        self.sv_manager = SupportVectorManager(
            config=self.config.get('support_vectors', {}),
            max_support_vectors=self.max_support_vectors
        )
        
        # Hierarchical Memory System
        self.memory_hierarchy = MemoryHierarchy(
            config=self.config.get('memory', {}),
            levels=self.config.get('memory', {}).get('levels', 3)
        )
        
        logger.info("Core components initialized: DDWE, TGCK, Quantum Optimizer")
    
    def fit(self, X: np.ndarray, y: np.ndarray, graph: NetworkGraph, 
            timestamps: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'HyperPathSVM':
  
        
        logger.info(f"Training HyperPath-SVM on {len(X)} samples")
        start_time = time.time()
        
        # Input validation
        X, y = self._validate_training_data(X, y, timestamps)
        
        # Store training data references
        self.X_train_ = X
        self.y_train_ = y
        self.graph_ = graph
        self.timestamps_ = timestamps
        self.n_samples_, self.n_features_ = X.shape
        
        # Initialize sample weights
        if sample_weight is None:
            sample_weight = np.ones(self.n_samples_)
        
        # Step 1: Initialize temporal graph analysis
        logger.info("Step 1: Computing temporal graph convolution kernel")
        self._precompute_graph_structures(graph, timestamps)
        
        # Step 2: Compute initial kernel matrix with TGCK
        logger.info("Step 2: Computing TGCK kernel matrix")
        K_initial = self.tgck.compute_kernel_matrix(X, X, graph, timestamps)
        
        # Step 3: Initialize DDWE with uniform weights
        logger.info("Step 3: Initializing DDWE weight hierarchy")
        initial_weights = self.ddwe.initialize_weights(X, y, timestamps)
        
        # Step 4: Quantum-inspired optimization of kernel weights
        logger.info("Step 4: Quantum-inspired weight optimization")
        
        def objective_function(weights):
            """Objective function for quantum optimization"""
            weighted_kernel = self._apply_adaptive_weights(K_initial, weights, timestamps)
            svm_temp = SVC(C=self.C, kernel='precomputed', probability=False)
            svm_temp.fit(weighted_kernel, y, sample_weight=sample_weight)
            
            # Compute cross-validation score
            cv_score = self._cross_validation_score(weighted_kernel, y, folds=3)
            
            # Add regularization terms
            weight_regularization = self._weight_regularization(weights)
            
            return -(cv_score - 0.1 * weight_regularization)  # Negative for minimization
        
        # Optimize weights using quantum superposition
        optimal_weights = self.quantum_optimizer.optimize(objective_function, initial_weights)
        
        # Step 5: Train final SVM with optimized kernel
        logger.info("Step 5: Training final SVM with optimized weights")
        self.adaptive_weights_ = optimal_weights
        K_optimized = self._apply_adaptive_weights(K_initial, optimal_weights, timestamps)
        
        # Initialize base SVM
        self.svm_ = SVC(
            C=self.C,
            kernel='precomputed',
            probability=False,
            cache_size=self.config.get('kernel_cache_size', 1000),
            shrinking=self.config.get('shrinking', True)
        )
        
        # Fit the SVM
        self.svm_.fit(K_optimized, y, sample_weight=sample_weight)
        
        # Step 6: Initialize support vector management
        logger.info("Step 6: Initializing support vector management")
        self.sv_manager.initialize_support_vectors(
            self.svm_.support_vectors_,
            self.svm_.support_,
            self.svm_.dual_coef_[0],
            timestamps[self.svm_.support_]
        )
        
        # Step 7: Initialize memory hierarchy with training data
        logger.info("Step 7: Initializing memory hierarchy")
        self._initialize_memory_hierarchy(X, y, timestamps, sample_weight)
        
        # Compute training statistics
        self._compute_training_stats(time.time() - start_time)
        
        # Mark as fitted
        self.is_fitted_ = True
        
        logger.info(f"Training completed in {time.time() - start_time:.2f}s")
        logger.info(f"Support vectors: {len(self.svm_.support_)}/{len(X)} ({100*len(self.svm_.support_)/len(X):.1f}%)")
        
        return self
    
    def predict(self, X: np.ndarray, graph: NetworkGraph, 
                timestamps: np.ndarray) -> np.ndarray:
      
        
        # Check if model is fitted
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.debug(f"Making predictions for {len(X)} samples")
        start_time = time.time()
        
        # Input validation
        X = self._validate_prediction_data(X, timestamps)
        
        # Step 1: Get current adaptive weights from DDWE
        current_weights = self.ddwe.get_adaptive_weights(timestamps)
        
        # Step 2: Compute kernel matrix between test and training data
        K_test = self.tgck.compute_kernel_matrix(
            X, self.X_train_, graph, timestamps, self.timestamps_
        )
        
        # Step 3: Apply adaptive weights to kernel
        K_weighted = self._apply_adaptive_weights(K_test, current_weights, timestamps)
        
        # Step 4: Make predictions using weighted kernel
        predictions = self.svm_.predict(K_weighted)
        
        # Step 5: Store predictions in memory hierarchy for learning
        self._store_predictions_for_learning(X, predictions, timestamps)
        
        # Update inference statistics
        inference_time = time.time() - start_time
        self._update_inference_stats(inference_time, len(X))
        
        logger.debug(f"Predictions completed in {inference_time*1000:.1f}ms")
        
        return predictions
    
    def predict_proba(self, X: np.ndarray, graph: NetworkGraph, 
                      timestamps: np.ndarray) -> np.ndarray:
       
        
        if not hasattr(self.svm_, 'predict_proba'):
            # Enable probability estimation and retrain if necessary
            logger.warning("Probability estimation not enabled. Retraining with probability=True")
            self._enable_probability_estimation()
        
        # Follow same steps as predict() but return probabilities
        current_weights = self.ddwe.get_adaptive_weights(timestamps)
        K_test = self.tgck.compute_kernel_matrix(
            X, self.X_train_, graph, timestamps, self.timestamps_
        )
        K_weighted = self._apply_adaptive_weights(K_test, current_weights, timestamps)
        
        return self.svm_.predict_proba(K_weighted)
    
    def decision_function(self, X: np.ndarray, graph: NetworkGraph,
                          timestamps: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for interpretability
        
        Returns the signed distance to the separating hyperplane.
        """
        
        current_weights = self.ddwe.get_adaptive_weights(timestamps)
        K_test = self.tgck.compute_kernel_matrix(
            X, self.X_train_, graph, timestamps, self.timestamps_
        )
        K_weighted = self._apply_adaptive_weights(K_test, current_weights, timestamps)
        
        return self.svm_.decision_function(K_weighted)
    
    def adapt(self, feedback: Dict) -> None:
      
        
        logger.info("Adapting model to network feedback")
        adaptation_start = time.time()
        
        timestamp = feedback.get('timestamp', time.time())
        
        # Step 1: Update DDWE weights based on performance feedback
        if 'path_performance' in feedback:
            self.ddwe.update_weights(feedback['path_performance'], timestamp)
        
        # Step 2: Handle topology changes
        if 'link_failures' in feedback or 'link_additions' in feedback:
            self._adapt_to_topology_changes(feedback, timestamp)
        
        # Step 3: Update support vector importance based on feedback
        if 'path_performance' in feedback:
            self.sv_manager.update_support_vector_weights(
                feedback['path_performance'], timestamp
            )
        
        # Step 4: Store feedback in memory hierarchy
        self.memory_hierarchy.store_experience({
            'feedback': feedback,
            'timestamp': timestamp,
            'weights_before': self.adaptive_weights_.copy(),
        })
        
        # Step 5: Evolve memory hierarchy
        self.ddwe.evolve_memory_hierarchy(feedback, timestamp)
        
        # Step 6: Continuous learning with new support vectors if needed
        if 'new_samples' in feedback:
            self._incremental_learning(feedback['new_samples'], timestamp)
        
        # Record adaptation statistics
        adaptation_time = time.time() - adaptation_start
        self._record_adaptation_stats(adaptation_time, feedback)
        
        logger.info(f"Adaptation completed in {adaptation_time:.2f}s")
    
    def get_interpretability_info(self) -> Dict:
       
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to provide interpretability info")
        
        interpretability_info = {
            # Support vector analysis
            'support_vectors': {
                'indices': self.svm_.support_.tolist(),
                'coefficients': self.svm_.dual_coef_[0].tolist(),
                'importance_scores': self.sv_manager.get_support_vector_importance(),
                'total_count': len(self.svm_.support_),
                'memory_usage_mb': self._compute_sv_memory_usage()
            },
            
            # DDWE weight evolution
            'weight_evolution': {
                'current_weights': self.ddwe.get_current_weights(),
                'weight_history': self.ddwe.get_weight_history(),
                'memory_levels': self.ddwe.get_memory_level_info(),
                'adaptation_events': self.ddwe.get_adaptation_events()
            },
            
            # SVM decision boundary
            'decision_boundaries': {
                'intercept': float(self.svm_.intercept_[0]),
                'n_support_vectors_per_class': self.svm_.n_support_.tolist(),
                'decision_function_shape': self.svm_.decision_function_shape
            },
            
            # Graph topology influence
            'graph_influence': {
                'spectral_properties': self.tgck.get_spectral_properties(),
                'temporal_patterns': self.tgck.get_temporal_patterns(),
                'graph_statistics': self._compute_graph_influence_stats()
            },
            
            # Quantum optimization state
            'quantum_states': {
                'configuration_history': self.quantum_optimizer.get_configuration_history(),
                'convergence_path': self.quantum_optimizer.get_convergence_path(),
                'tunneling_events': self.quantum_optimizer.get_tunneling_events(),
                'measurement_outcomes': self.quantum_optimizer.get_measurement_outcomes()
            },
            
            # Performance statistics
            'performance_stats': self.training_stats_.copy(),
            
            # Model complexity metrics
            'complexity_metrics': {
                'effective_dimension': self._compute_effective_dimension(),
                'kernel_rank': self._compute_kernel_rank(),
                'generalization_bound': self._compute_generalization_bound()
            }
        }
        
        return interpretability_info
    
    def _validate_training_data(self, X: np.ndarray, y: np.ndarray, 
                               timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess training data"""
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        timestamps = np.asarray(timestamps)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        if len(X) != len(y) or len(X) != len(timestamps):
            raise ValueError("X, y, and timestamps must have same length")
        
        if len(np.unique(y)) != 2:
            raise ValueError("y must contain exactly 2 classes")
        
        # Sort by timestamps to ensure temporal order
        sort_idx = np.argsort(timestamps)
        X = X[sort_idx]
        y = y[sort_idx]
        timestamps = timestamps[sort_idx]
        
        logger.info(f"Training data validated: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def _validate_prediction_data(self, X: np.ndarray, 
                                 timestamps: np.ndarray) -> np.ndarray:
        """Validate prediction data"""
        
        X = np.asarray(X, dtype=np.float64)
        timestamps = np.asarray(timestamps)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}")
        
        if len(X) != len(timestamps):
            raise ValueError("X and timestamps must have same length")
        
        return X
    
    def _precompute_graph_structures(self, graph: NetworkGraph, 
                                   timestamps: np.ndarray) -> None:
        """Precompute graph structures for efficiency"""
        
        # Compute graph Laplacian and eigendecomposition
        L = graph.compute_laplacian()
        eigenvalues, eigenvectors = eigen_decomposition(L)
        
        # Store for TGCK kernel computation
        self.graph_eigenvalues_ = eigenvalues
        self.graph_eigenvectors_ = eigenvectors
        
        # Compute shortest path matrix for graph-aware distances
        self.shortest_paths_ = shortest_path_matrix(graph)
        
        # Store graph statistics
        self.graph_stats_ = {
            'num_nodes': graph.num_nodes,
            'num_edges': graph.num_edges,
            'density': graph.density,
            'diameter': graph.diameter,
            'clustering_coefficient': graph.clustering_coefficient
        }
        
        logger.info(f"Graph structures precomputed: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    def _apply_adaptive_weights(self, K: np.ndarray, weights: np.ndarray,
                               timestamps: np.ndarray) -> np.ndarray:
        """Apply DDWE adaptive weights to kernel matrix"""
        
        # Get time-dependent weights
        time_weights = self.ddwe.compute_temporal_weights(timestamps)
        
        # Combine spatial and temporal weights
        combined_weights = weights * time_weights
        
        # Apply weights to kernel matrix
        # K_weighted[i,j] = w(t_i, t_j) * K[i,j]
        weight_matrix = np.outer(combined_weights, combined_weights)
        K_weighted = K * weight_matrix
        
        return K_weighted
    
    def _cross_validation_score(self, K: np.ndarray, y: np.ndarray, 
                               folds: int = 5) -> float:
        """Compute cross-validation score for kernel evaluation"""
        
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score
        
        kf = KFold(n_splits=folds, shuffle=False)  # No shuffle for temporal data
        scores = []
        
        for train_idx, val_idx in kf.split(K):
            # Train on fold
            K_train = K[np.ix_(train_idx, train_idx)]
            y_train_fold = y[train_idx]
            
            svm_fold = SVC(C=self.C, kernel='precomputed')
            svm_fold.fit(K_train, y_train_fold)
            
            # Predict on validation
            K_val = K[np.ix_(val_idx, train_idx)]
            y_pred_fold = svm_fold.predict(K_val)
            
            score = accuracy_score(y[val_idx], y_pred_fold)
            scores.append(score)
        
        return np.mean(scores)
    
    def _weight_regularization(self, weights: np.ndarray) -> float:
        """Compute regularization term for weight optimization"""
        
        # L2 regularization on weights
        l2_reg = np.sum(weights ** 2)
        
        # Entropy regularization to encourage diversity
        weights_normalized = weights / np.sum(weights)
        entropy_reg = -np.sum(weights_normalized * np.log(weights_normalized + 1e-8))
        
        # Smoothness regularization (encourage smooth weight evolution)
        if hasattr(self, 'previous_weights_'):
            smoothness_reg = np.sum((weights - self.previous_weights_) ** 2)
        else:
            smoothness_reg = 0.0
        
        total_reg = 0.01 * l2_reg - 0.001 * entropy_reg + 0.005 * smoothness_reg
        
        return total_reg
    
    def _adapt_to_topology_changes(self, feedback: Dict, timestamp: float) -> None:
        """Adapt model to network topology changes"""
        
        # Handle link failures
        if 'link_failures' in feedback:
            failed_links = feedback['link_failures']
            self.graph_.remove_edges(failed_links, timestamp)
            self._recompute_affected_kernel_entries(failed_links)
        
        # Handle new links
        if 'link_additions' in feedback:
            new_links = feedback['link_additions']
            self.graph_.add_edges(new_links, timestamp)
            self._recompute_affected_kernel_entries(new_links)
        
        # Update graph-dependent computations
        if 'link_failures' in feedback or 'link_additions' in feedback:
            self._update_graph_structures()
    
    def _incremental_learning(self, new_samples: Dict, timestamp: float) -> None:
        """Add new support vectors for continuous learning"""
        
        X_new = new_samples['features']
        y_new = new_samples['labels']
        
        # Compute novelty of new samples
        novelty_scores = self.sv_manager.compute_novelty(X_new)
        
        # Add novel samples as support vectors
        novel_mask = novelty_scores > self.config.get('support_vectors', {}).get('novelty_threshold', 0.1)
        
        if np.any(novel_mask):
            X_novel = X_new[novel_mask]
            y_novel = y_new[novel_mask]
            
            # Add to support vector set
            self.sv_manager.add_support_vectors(X_novel, y_novel, timestamp)
            
            logger.info(f"Added {np.sum(novel_mask)} novel support vectors")
    
    def _initialize_memory_hierarchy(self, X: np.ndarray, y: np.ndarray,
                                   timestamps: np.ndarray, sample_weight: np.ndarray) -> None:
        """Initialize hierarchical memory with training data"""
        
        # Store training samples across memory levels based on recency
        current_time = timestamps[-1]
        
        for i, (x, label, t, weight) in enumerate(zip(X, y, timestamps, sample_weight)):
            
            time_diff = current_time - t
            
            # Determine memory level based on recency
            if time_diff <= self.ddwe.short_term_window:
                level = 'short_term'
            elif time_diff <= self.ddwe.medium_term_window:
                level = 'medium_term'
            else:
                level = 'long_term'
            
            # Store sample in appropriate memory level
            self.memory_hierarchy.store_sample(
                sample={
                    'features': x,
                    'label': label,
                    'timestamp': t,
                    'weight': weight,
                    'importance': weight
                },
                level=level
            )
    
    def _compute_training_stats(self, training_time: float) -> None:
        """Compute comprehensive training statistics"""
        
        self.training_stats_ = {
            # Timing statistics
            'training_time_seconds': training_time,
            'training_time_per_sample_ms': (training_time * 1000) / self.n_samples_,
            
            # Model complexity
            'num_support_vectors': len(self.svm_.support_),
            'support_vector_ratio': len(self.svm_.support_) / self.n_samples_,
            
            # Memory usage
            'memory_usage_mb': self._compute_memory_usage(),
            
            # DDWE statistics
            'ddwe_stats': self.ddwe.get_statistics(),
            
            # TGCK statistics  
            'tgck_stats': self.tgck.get_statistics(),
            
            # Quantum optimization statistics
            'quantum_stats': self.quantum_optimizer.get_statistics(),
            
            # Graph statistics
            'graph_stats': self.graph_stats_,
            
            # Training data characteristics
            'n_samples': self.n_samples_,
            'n_features': self.n_features_,
            'class_distribution': np.bincount(self.y_train_) / len(self.y_train_),
            
            # Convergence information
            'converged': True,  # Will be updated based on actual convergence
            'final_objective_value': self.quantum_optimizer.get_final_objective(),
            'num_iterations': self.quantum_optimizer.get_num_iterations()
        }
    
    def _update_inference_stats(self, inference_time: float, num_samples: int) -> None:
        """Update inference timing statistics"""
        
        time_per_sample = (inference_time * 1000) / num_samples  # Convert to ms
        
        if not hasattr(self, 'inference_times_'):
            self.inference_times_ = []
        
        self.inference_times_.extend([time_per_sample] * num_samples)
        
        # Keep only recent timing data (last 10,000 predictions)
        if len(self.inference_times_) > 10000:
            self.inference_times_ = self.inference_times_[-10000:]
    
    def _record_adaptation_stats(self, adaptation_time: float, feedback: Dict) -> None:
        """Record adaptation performance statistics"""
        
        adaptation_record = {
            'timestamp': feedback.get('timestamp', time.time()),
            'adaptation_time_seconds': adaptation_time,
            'feedback_type': list(feedback.keys()),
            'weights_changed': self._compute_weight_change(),
            'support_vectors_modified': self.sv_manager.get_recent_modifications()
        }
        
        self.performance_history_.append(adaptation_record)
        
        # Keep only recent adaptation history (last 1000 adaptations)
        if len(self.performance_history_) > 1000:
            self.performance_history_ = self.performance_history_[-1000:]
    
    def _compute_memory_usage(self) -> float:
        """Compute current memory usage in MB"""
        
        memory_mb = 0.0
        
        # Training data
        memory_mb += self.X_train_.nbytes / (1024 * 1024)
        memory_mb += self.y_train_.nbytes / (1024 * 1024)
        
        # Support vectors
        memory_mb += self.svm_.support_vectors_.nbytes / (1024 * 1024)
        memory_mb += self.svm_.dual_coef_.nbytes / (1024 * 1024)
        
        # Adaptive weights
        memory_mb += self.adaptive_weights_.nbytes / (1024 * 1024)
        
        # Component memory usage
        memory_mb += self.ddwe.get_memory_usage()
        memory_mb += self.tgck.get_memory_usage()
        memory_mb += self.quantum_optimizer.get_memory_usage()
        memory_mb += self.sv_manager.get_memory_usage()
        memory_mb += self.memory_hierarchy.get_memory_usage()
        
        return memory_mb
    
    def _compute_weight_change(self) -> float:
        """Compute magnitude of recent weight changes"""
        
        if hasattr(self, 'previous_weights_'):
            current_weights = self.ddwe.get_current_weights()
            weight_change = np.linalg.norm(current_weights - self.previous_weights_)
            self.previous_weights_ = current_weights.copy()
            return float(weight_change)
        else:
            self.previous_weights_ = self.ddwe.get_current_weights().copy()
            return 0.0
    
    def _compute_effective_dimension(self) -> int:
        """Compute effective dimension of the kernel space"""
        
        # Use eigenvalue decay to estimate effective dimension
        eigenvalues = self.graph_eigenvalues_
        cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        
        # Find dimension that captures 95% of variance
        effective_dim = np.argmax(cumsum >= 0.95) + 1
        
        return int(effective_dim)
    
    def _compute_kernel_rank(self) -> int:
        """Compute numerical rank of kernel matrix"""
        
        # Sample subset for efficiency
        if len(self.X_train_) > 1000:
            idx = np.random.choice(len(self.X_train_), 1000, replace=False)
            X_subset = self.X_train_[idx]
            timestamps_subset = self.timestamps_[idx]
        else:
            X_subset = self.X_train_
            timestamps_subset = self.timestamps_
        
        # Compute kernel matrix
        K_subset = self.tgck.compute_kernel_matrix(
            X_subset, X_subset, self.graph_, timestamps_subset
        )
        
        # Compute numerical rank
        rank = np.linalg.matrix_rank(K_subset, tol=1e-6)
        
        return int(rank)
    
    def _compute_generalization_bound(self) -> float:
        """Compute theoretical generalization bound"""
        
        # Rademacher complexity bound for kernel methods
        # Based on support vector ratio and kernel properties
        
        sv_ratio = len(self.svm_.support_) / self.n_samples_
        
        # Kernel trace (complexity measure)
        K_diag = np.diag(self.tgck.compute_kernel_matrix(
            self.X_train_, self.X_train_, self.graph_, self.timestamps_
        ))
        kernel_trace = np.mean(K_diag)
        
        # Generalization bound (simplified)
        bound = 2 * np.sqrt(sv_ratio * kernel_trace / self.n_samples_)
        
        return float(bound)
    
    def _store_predictions_for_learning(self, X: np.ndarray, predictions: np.ndarray,
                                       timestamps: np.ndarray) -> None:
        """Store recent predictions for continuous learning"""
        
        for x, pred, t in zip(X, predictions, timestamps):
            prediction_record = {
                'features': x,
                'prediction': pred,
                'timestamp': t,
                'confidence': None  # Will be updated with feedback
            }
            
            # Store in short-term memory for potential learning
            self.memory_hierarchy.store_sample(prediction_record, level='short_term')
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        if not self.is_fitted_:
            return {"error": "Model not fitted"}
        
        summary = {
            # Current performance metrics
            'accuracy_estimate': self._estimate_current_accuracy(),
            'inference_time_stats': self._get_inference_time_stats(),
            'memory_usage_mb': self._compute_memory_usage(),
            'adaptation_speed_stats': self._get_adaptation_speed_stats(),
            
            # Model state
            'num_support_vectors': len(self.svm_.support_),
            'num_adaptations': len(self.performance_history_),
            'time_since_training': time.time() - self.training_stats_.get('training_time_seconds', 0),
            
            # Component status
            'ddwe_status': self.ddwe.get_status(),
            'tgck_status': self.tgck.get_status(),
            'quantum_status': self.quantum_optimizer.get_status(),
            'sv_manager_status': self.sv_manager.get_status(),
            
            # Performance targets
            'targets_met': {
                'accuracy_target': self._estimate_current_accuracy() >= 0.965,
                'inference_time_target': self._get_mean_inference_time() <= 1.8,
                'memory_target': self._compute_memory_usage() <= 98,
                'adaptation_target': self._get_mean_adaptation_time() <= 138  # 2.3 min
            }
        }
        
        return summary
    
    def _estimate_current_accuracy(self) -> float:
        """Estimate current model accuracy based on recent feedback"""
        
        if not self.performance_history_:
            return 0.0
        
        # Use recent performance feedback to estimate accuracy
        recent_feedback = self.performance_history_[-10:]  # Last 10 adaptations
        
        if recent_feedback:
            accuracies = [fb.get('accuracy', 0.0) for fb in recent_feedback 
                         if 'accuracy' in fb]
            return np.mean(accuracies) if accuracies else 0.0
        
        return 0.0
    
    def _get_inference_time_stats(self) -> Dict:
        """Get inference timing statistics"""
        
        if not hasattr(self, 'inference_times_') or not self.inference_times_:
            return {}
        
        times = np.array(self.inference_times_)
        
        return {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times))
        }
    
    def _get_mean_inference_time(self) -> float:
        """Get mean inference time in milliseconds"""
        
        if hasattr(self, 'inference_times_') and self.inference_times_:
            return float(np.mean(self.inference_times_))
        return 0.0
    
    def _get_adaptation_speed_stats(self) -> Dict:
        """Get adaptation speed statistics"""
        
        if not self.performance_history_:
            return {}
        
        adaptation_times = [record['adaptation_time_seconds'] 
                          for record in self.performance_history_
                          if 'adaptation_time_seconds' in record]
        
        if not adaptation_times:
            return {}
        
        times_array = np.array(adaptation_times)
        
        return {
            'mean_seconds': float(np.mean(times_array)),
            'median_seconds': float(np.median(times_array)),
            'p95_seconds': float(np.percentile(times_array, 95)),
            'std_seconds': float(np.std(times_array)),
            'min_seconds': float(np.min(times_array)),
            'max_seconds': float(np.max(times_array))
        }
    
    def _get_mean_adaptation_time(self) -> float:
        """Get mean adaptation time in seconds"""
        
        if self.performance_history_:
            adaptation_times = [record.get('adaptation_time_seconds', 0) 
                              for record in self.performance_history_]
            return float(np.mean(adaptation_times))
        return 0.0
    
    def save_model(self, filepath: str) -> None:
        """Save complete model state for production deployment"""
        
        import pickle
        
        model_state = {
            'config': self.config,
            'hyperparameters': {
                'C': self.C,
                'epsilon': self.epsilon,
                'max_support_vectors': self.max_support_vectors
            },
            'training_data': {
                'X_train': self.X_train_,
                'y_train': self.y_train_,
                'timestamps': self.timestamps_
            },
            'svm_model': self.svm_,
            'adaptive_weights': self.adaptive_weights_,
            'graph_structures': {
                'eigenvalues': self.graph_eigenvalues_,
                'eigenvectors': self.graph_eigenvectors_,
                'shortest_paths': self.shortest_paths_
            },
            'component_states': {
                'ddwe_state': self.ddwe.get_state(),
                'tgck_state': self.tgck.get_state(),
                'quantum_state': self.quantum_optimizer.get_state(),
                'sv_manager_state': self.sv_manager.get_state(),
                'memory_hierarchy_state': self.memory_hierarchy.get_state()
            },
            'statistics': {
                'training_stats': self.training_stats_,
                'performance_history': self.performance_history_,
                'inference_times': getattr(self, 'inference_times_', [])
            },
            'metadata': {
                'version': '1.0.0',
                'timestamp': time.time(),
                'n_samples': self.n_samples_,
                'n_features': self.n_features_
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HyperPathSVM':
        """Load complete model state from file"""
        
        import pickle
        
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Recreate model instance
        model = cls(model_state['config'])
        
        # Restore hyperparameters
        model.C = model_state['hyperparameters']['C']
        model.epsilon = model_state['hyperparameters']['epsilon']
        model.max_support_vectors = model_state['hyperparameters']['max_support_vectors']
        
        # Restore training data
        model.X_train_ = model_state['training_data']['X_train']
        model.y_train_ = model_state['training_data']['y_train']  
        model.timestamps_ = model_state['training_data']['timestamps']
        model.n_samples_, model.n_features_ = model.X_train_.shape
        
        # Restore SVM model
        model.svm_ = model_state['svm_model']
        model.adaptive_weights_ = model_state['adaptive_weights']
        
        # Restore graph structures
        model.graph_eigenvalues_ = model_state['graph_structures']['eigenvalues']
        model.graph_eigenvectors_ = model_state['graph_structures']['eigenvectors']
        model.shortest_paths_ = model_state['graph_structures']['shortest_paths']
        
        # Restore component states
        model.ddwe.restore_state(model_state['component_states']['ddwe_state'])
        model.tgck.restore_state(model_state['component_states']['tgck_state'])
        model.quantum_optimizer.restore_state(model_state['component_states']['quantum_state'])
        model.sv_manager.restore_state(model_state['component_states']['sv_manager_state'])
        model.memory_hierarchy.restore_state(model_state['component_states']['memory_hierarchy_state'])
        
        # Restore statistics
        model.training_stats_ = model_state['statistics']['training_stats']
        model.performance_history_ = model_state['statistics']['performance_history']
        model.inference_times_ = model_state['statistics']['inference_times']
        
        # Mark as fitted
        model.is_fitted_ = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def __repr__(self) -> str:
        """String representation of the model"""
        
        if self.is_fitted_:
            return (f"HyperPathSVM(C={self.C}, epsilon={self.epsilon}, "
                   f"n_support_vectors={len(self.svm_.support_)}, "
                   f"memory_usage={self._compute_memory_usage():.1f}MB)")
        else:
            return f"HyperPathSVM(C={self.C}, epsilon={self.epsilon}, unfitted)" 
