 
# File: hyperpath_svm/experiments/scalability_test.py

"""
Scalability Testing and Performance Analysis for HyperPath-SVM

This module provides comprehensive scalability testing to evaluate how HyperPath-SVM
performs as network size, data volume, and complexity increase. It tests various
scalability dimensions and identifies performance bottlenecks.


"""

import numpy as np
import pandas as pd
import time
import psutil
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import gc
import pickle
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx

from ..utils.logging_utils import get_logger, get_hyperpath_logger
from ..core.hyperpath_svm import HyperPathSVM
from ..data.network_graph import NetworkGraph
from ..evaluation.metrics import ComprehensiveMetricsEvaluator
from ..utils.math_utils import math_utils
from ..utils.graph_utils import graph_processor

warnings.filterwarnings('ignore')


@dataclass
class ScalabilityConfig:
    """Configuration for scalability testing."""
    
    # Test dimensions
    test_network_scaling: bool = True
    test_data_scaling: bool = True
    test_feature_scaling: bool = True
    test_temporal_scaling: bool = True
    test_memory_constraints: bool = True
    test_distributed_processing: bool = False
    
    # Network size scaling parameters
    network_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])
    max_network_size: int = 50000
    
    # Data volume scaling parameters  
    data_sizes: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 50000, 100000])
    max_data_size: int = 1000000
    
    # Feature scaling parameters
    feature_dims: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    max_features: int = 10000
    
    # Temporal scaling parameters
    temporal_lengths: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    max_temporal_length: int = 5000
    
    # Performance constraints
    max_training_time_hours: float = 2.0
    max_memory_gb: float = 8.0
    timeout_seconds: float = 3600  # 1 hour timeout per test
    
    # Accuracy requirements
    min_accuracy_threshold: float = 0.85
    accuracy_degradation_threshold: float = 0.05
    
    # Resource monitoring
    monitor_interval_seconds: float = 1.0
    enable_profiling: bool = True
    
    # Output configuration
    results_dir: Path = Path("scalability_results")
    generate_plots: bool = True
    save_detailed_logs: bool = True


@dataclass
class ScalabilityResult:
    
    
    test_name: str = ""
    scale_parameter: str = ""  # 'network_size', 'data_size', etc.
    scale_value: int = 0
    
    # Performance metrics
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    accuracy: float = 0.0
    convergence_iterations: int = 0
    
    # Resource utilization
    peak_cpu_percent: float = 0.0
    peak_memory_percent: float = 0.0
    disk_io_mb: float = 0.0
    
    # Scalability metrics
    computational_complexity: float = 0.0  # Empirical O(n^x)
    memory_complexity: float = 0.0
    
    # Status
    success: bool = False
    error_message: str = ""
    timeout: bool = False
    
    # Detailed profiling data
    profiling_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalabilityAnalysis:
    
    
    config: ScalabilityConfig
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Test results organized by dimension
    network_scaling_results: List[ScalabilityResult] = field(default_factory=list)
    data_scaling_results: List[ScalabilityResult] = field(default_factory=list)
    feature_scaling_results: List[ScalabilityResult] = field(default_factory=list)
    temporal_scaling_results: List[ScalabilityResult] = field(default_factory=list)
    
    # Complexity analysis
    complexity_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Performance bounds
    scalability_limits: Dict[str, int] = field(default_factory=dict)
    recommended_configurations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Summary statistics
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    timeout_tests: int = 0


class ResourceMonitor:
    
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.monitoring = False
        self.monitor_thread = None
        self.logger = get_logger(__name__)
        
        # Resource tracking
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_io = []
        self.timestamps = []
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.debug("Resource monitoring started")
    
    def stop_monitoring(self):
        
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.debug("Resource monitoring stopped")
    
    def _monitor_loop(self):
       
        while self.monitoring:
            try:
                # Get system stats
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                with self._lock:
                    self.cpu_usage.append(cpu_percent)
                    self.memory_usage.append(memory.percent)
                    self.disk_io.append(disk_io.read_bytes + disk_io.write_bytes)
                    self.timestamps.append(time.time())
                
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.warning(f"Resource monitoring error: {str(e)}")
    
    def get_peak_stats(self) -> Dict[str, float]:
      
        with self._lock:
            if not self.cpu_usage:
                return {}
            
            return {
                'peak_cpu_percent': max(self.cpu_usage),
                'avg_cpu_percent': np.mean(self.cpu_usage),
                'peak_memory_percent': max(self.memory_usage),
                'avg_memory_percent': np.mean(self.memory_usage),
                'total_disk_io_mb': (max(self.disk_io) - min(self.disk_io)) / (1024**2) if len(self.disk_io) > 1 else 0
            }
    
    def reset(self):
       
        with self._lock:
            self.cpu_usage.clear()
            self.memory_usage.clear()
            self.disk_io.clear()
            self.timestamps.clear()


class ScalabilityTester:
   
    
    def __init__(self, config: Optional[ScalabilityConfig] = None):
        self.config = config or ScalabilityConfig()
        self.logger = get_logger(__name__)
        self.hyperpath_logger = get_hyperpath_logger(__name__)
        
        # Initialize components
        self.metrics_evaluator = ComprehensiveMetricsEvaluator()
        self.resource_monitor = ResourceMonitor(self.config.monitor_interval_seconds)
        
        # Setup output directory
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Scalability tester initialized")
    
    def run_comprehensive_scalability_analysis(self) -> ScalabilityAnalysis:
       
        try:
            self.logger.info("Starting comprehensive scalability analysis")
            
            analysis = ScalabilityAnalysis(config=self.config)
            
            # Test different scalability dimensions
            test_functions = []
            
            if self.config.test_network_scaling:
                test_functions.append(("network_scaling", self._test_network_scaling))
            
            if self.config.test_data_scaling:
                test_functions.append(("data_scaling", self._test_data_scaling))
            
            if self.config.test_feature_scaling:
                test_functions.append(("feature_scaling", self._test_feature_scaling))
            
            if self.config.test_temporal_scaling:
                test_functions.append(("temporal_scaling", self._test_temporal_scaling))
            
            # Execute tests
            for test_name, test_function in test_functions:
                self.logger.info(f"Running {test_name} tests")
                
                try:
                    results = test_function()
                    setattr(analysis, f"{test_name}_results", results)
                    
                    # Update counters
                    analysis.total_tests += len(results)
                    analysis.successful_tests += sum(1 for r in results if r.success)
                    analysis.failed_tests += sum(1 for r in results if not r.success and not r.timeout)
                    analysis.timeout_tests += sum(1 for r in results if r.timeout)
                    
                except Exception as e:
                    self.logger.error(f"Test {test_name} failed: {str(e)}")
            
            # Analyze results
            self.logger.info("Analyzing scalability patterns")
            self._analyze_complexity_patterns(analysis)
            self._identify_bottlenecks(analysis)
            self._determine_scalability_limits(analysis)
            
            # Generate reports and visualizations
            if self.config.generate_plots:
                self._generate_scalability_plots(analysis)
            
            self._generate_scalability_report(analysis)
            self._save_analysis_results(analysis)
            
            analysis.end_time = time.time()
            duration_hours = (analysis.end_time - analysis.start_time) / 3600
            
            self.logger.info(f"Scalability analysis completed in {duration_hours:.2f} hours")
            self.logger.info(f"Success rate: {analysis.successful_tests / max(analysis.total_tests, 1):.1%}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Scalability analysis failed: {str(e)}")
            raise
    
    def _test_network_scaling(self) -> List[ScalabilityResult]:
        
        results = []
        
        for network_size in self.config.network_sizes:
            if network_size > self.config.max_network_size:
                break
            
            result = ScalabilityResult(
                test_name="network_scaling",
                scale_parameter="network_size",
                scale_value=network_size
            )
            
            try:
                self.logger.info(f"Testing network size: {network_size} nodes")
                
                # Generate synthetic network data
                network_data = self._generate_synthetic_network(network_size)
                
                # Run scalability test
                test_result = self._run_single_scalability_test(
                    network_data, f"network_{network_size}"
                )
                
                # Update result
                result.success = test_result['success']
                result.training_time_seconds = test_result['training_time']
                result.inference_time_ms = test_result['inference_time_ms']
                result.memory_usage_mb = test_result['memory_usage_mb']
                result.accuracy = test_result.get('accuracy', 0.0)
                result.error_message = test_result.get('error', '')
                result.timeout = test_result.get('timeout', False)
                
                # Resource utilization
                if 'resource_stats' in test_result:
                    stats = test_result['resource_stats']
                    result.peak_cpu_percent = stats.get('peak_cpu_percent', 0)
                    result.peak_memory_percent = stats.get('peak_memory_percent', 0)
                    result.disk_io_mb = stats.get('total_disk_io_mb', 0)
                
            except Exception as e:
                self.logger.error(f"Network scaling test failed for size {network_size}: {str(e)}")
                result.success = False
                result.error_message = str(e)
            
            results.append(result)
            
            # Early termination if hitting limits
            if not result.success or result.timeout:
                self.logger.warning(f"Terminating network scaling tests at size {network_size}")
                break
        
        return results
    
    def _test_data_scaling(self) -> List[ScalabilityResult]:
       
        results = []
        
        # Use fixed moderate network size
        base_network_size = 1000
        network_data = self._generate_synthetic_network(base_network_size)
        
        for data_size in self.config.data_sizes:
            if data_size > self.config.max_data_size:
                break
            
            result = ScalabilityResult(
                test_name="data_scaling",
                scale_parameter="data_size", 
                scale_value=data_size
            )
            
            try:
                self.logger.info(f"Testing data size: {data_size} samples")
                
                # Scale up the data by replication and noise
                scaled_data = self._scale_network_data(network_data, target_size=data_size)
                
                test_result = self._run_single_scalability_test(
                    scaled_data, f"data_{data_size}"
                )
                
                # Update result
                result.success = test_result['success']
                result.training_time_seconds = test_result['training_time']
                result.inference_time_ms = test_result['inference_time_ms']
                result.memory_usage_mb = test_result['memory_usage_mb']
                result.accuracy = test_result.get('accuracy', 0.0)
                result.error_message = test_result.get('error', '')
                result.timeout = test_result.get('timeout', False)
                
                # Resource utilization
                if 'resource_stats' in test_result:
                    stats = test_result['resource_stats']
                    result.peak_cpu_percent = stats.get('peak_cpu_percent', 0)
                    result.peak_memory_percent = stats.get('peak_memory_percent', 0)
                    result.disk_io_mb = stats.get('total_disk_io_mb', 0)
                
            except Exception as e:
                self.logger.error(f"Data scaling test failed for size {data_size}: {str(e)}")
                result.success = False
                result.error_message = str(e)
            
            results.append(result)
            
            # Early termination if hitting limits
            if not result.success or result.timeout:
                self.logger.warning(f"Terminating data scaling tests at size {data_size}")
                break
        
        return results
    
    def _test_feature_scaling(self) -> List[ScalabilityResult]:
       
        results = []
        
        # Use moderate network and data sizes
        base_network_size = 1000
        base_data_size = 10000
        
        for num_features in self.config.feature_dims:
            if num_features > self.config.max_features:
                break
            
            result = ScalabilityResult(
                test_name="feature_scaling",
                scale_parameter="num_features",
                scale_value=num_features
            )
            
            try:
                self.logger.info(f"Testing feature dimensions: {num_features}")
                
                # Generate data with specified number of features
                network_data = self._generate_synthetic_network(
                    base_network_size, num_features=num_features
                )
                scaled_data = self._scale_network_data(network_data, target_size=base_data_size)
                
                test_result = self._run_single_scalability_test(
                    scaled_data, f"features_{num_features}"
                )
                
                # Update result
                result.success = test_result['success']
                result.training_time_seconds = test_result['training_time']
                result.inference_time_ms = test_result['inference_time_ms']
                result.memory_usage_mb = test_result['memory_usage_mb']
                result.accuracy = test_result.get('accuracy', 0.0)
                result.error_message = test_result.get('error', '')
                result.timeout = test_result.get('timeout', False)
                
            except Exception as e:
                self.logger.error(f"Feature scaling test failed for {num_features} features: {str(e)}")
                result.success = False
                result.error_message = str(e)
            
            results.append(result)
            
            # Early termination if hitting limits
            if not result.success or result.timeout:
                self.logger.warning(f"Terminating feature scaling tests at {num_features} features")
                break
        
        return results
    
    def _test_temporal_scaling(self) -> List[ScalabilityResult]:
       
        results = []
        
        # Use moderate network and data sizes
        base_network_size = 1000
        base_data_size = 10000
        
        for temporal_length in self.config.temporal_lengths:
            if temporal_length > self.config.max_temporal_length:
                break
            
            result = ScalabilityResult(
                test_name="temporal_scaling",
                scale_parameter="temporal_length",
                scale_value=temporal_length
            )
            
            try:
                self.logger.info(f"Testing temporal length: {temporal_length}")
                
                # Generate temporal data
                network_data = self._generate_temporal_network_data(
                    base_network_size, temporal_length
                )
                scaled_data = self._scale_network_data(network_data, target_size=base_data_size)
                
                test_result = self._run_single_scalability_test(
                    scaled_data, f"temporal_{temporal_length}"
                )
                
                # Update result
                result.success = test_result['success']
                result.training_time_seconds = test_result['training_time']
                result.inference_time_ms = test_result['inference_time_ms']
                result.memory_usage_mb = test_result['memory_usage_mb']
                result.accuracy = test_result.get('accuracy', 0.0)
                result.error_message = test_result.get('error', '')
                result.timeout = test_result.get('timeout', False)
                
            except Exception as e:
                self.logger.error(f"Temporal scaling test failed for length {temporal_length}: {str(e)}")
                result.success = False
                result.error_message = str(e)
            
            results.append(result)
            
            # Early termination if hitting limits
            if not result.success or result.timeout:
                self.logger.warning(f"Terminating temporal scaling tests at length {temporal_length}")
                break
        
        return results
    
    def _generate_synthetic_network(self, num_nodes: int, 
                                  num_features: int = 20) -> Dict[str, Any]:
       
        try:
            # Create network topology
            if num_nodes <= 10000:
                # Use Barabási-Albert model for scale-free networks
                G = nx.barabasi_albert_graph(num_nodes, m=3, seed=42)
                adjacency_matrix = nx.adjacency_matrix(G).toarray()
            else:
                # For very large networks, use sparse representation
                # Create adjacency matrix directly
                adjacency_matrix = self._generate_large_sparse_network(num_nodes)
            
            # Generate node features
            node_features = np.random.normal(0, 1, (num_nodes, min(num_features, 50)))
            
            # Generate edge features (simplified for large networks)
            num_edges = np.sum(adjacency_matrix > 0) // 2
            edge_features = np.random.normal(0, 1, (num_edges, min(10, num_features // 2)))
            
            # Generate temporal features
            temporal_features = np.random.normal(0, 1, (100, num_nodes))
            
            # Create network graph
            network_graph = NetworkGraph(
                adjacency_matrix=adjacency_matrix,
                node_features=node_features,
                edge_features=edge_features,
                temporal_features=temporal_features
            )
            
            # Generate sample data points
            num_samples = min(num_nodes * 10, 50000)  # Scale samples with network size
            
            # Features combine network properties and random features
            features = np.random.normal(0, 1, (num_samples, num_features))
            
            # Add network-derived features
            if num_features >= 5:
                # Add degree-based features
                degrees = np.sum(adjacency_matrix, axis=1)
                degree_features = np.random.choice(degrees, size=num_samples)
                features[:, 0] = degree_features
                
                # Add clustering coefficient features
                if num_nodes <= 5000:  # Only for reasonably sized networks
                    clustering_coeffs = nx.clustering(G) if 'G' in locals() else {}
                    if clustering_coeffs:
                        cc_values = list(clustering_coeffs.values())
                        cc_features = np.random.choice(cc_values, size=num_samples)
                        features[:, 1] = cc_features
            
            # Generate labels (routing decisions)
            labels = np.random.randint(0, min(3, num_nodes), size=num_samples)
            
            return {
                'network_graph': network_graph,
                'features': features,
                'labels': labels,
                'num_nodes': num_nodes,
                'num_samples': num_samples,
                'num_features': num_features
            }
            
        except Exception as e:
            self.logger.error(f"Synthetic network generation failed: {str(e)}")
            raise
    
    def _generate_large_sparse_network(self, num_nodes: int) -> np.ndarray:
        
        # Create sparse random network
        density = min(0.01, 1000.0 / num_nodes)  # Adaptive density
        
        # Generate random connections
        num_edges = int(num_nodes * (num_nodes - 1) * density / 2)
        
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        # Add random edges
        for _ in range(num_edges):
            i, j = np.random.randint(0, num_nodes, 2)
            if i != j:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Undirected
        
        return adjacency_matrix
    
    def _generate_temporal_network_data(self, num_nodes: int, 
                                      temporal_length: int) -> Dict[str, Any]:
        
        # Start with base network
        base_data = self._generate_synthetic_network(num_nodes)
        
        # Extend temporal features
        extended_temporal = np.random.normal(0, 1, (temporal_length, num_nodes))
        
        # Add temporal correlations
        for t in range(1, temporal_length):
            extended_temporal[t] = (0.7 * extended_temporal[t-1] + 
                                  0.3 * np.random.normal(0, 1, num_nodes))
        
        base_data['network_graph'].temporal_features = extended_temporal
        base_data['temporal_length'] = temporal_length
        
        return base_data
    
    def _scale_network_data(self, base_data: Dict[str, Any], 
                          target_size: int) -> Dict[str, Any]:
        
        current_size = len(base_data['features'])
        
        if target_size <= current_size:
            # Subsample
            indices = np.random.choice(current_size, target_size, replace=False)
            scaled_data = base_data.copy()
            scaled_data['features'] = base_data['features'][indices]
            scaled_data['labels'] = base_data['labels'][indices]
            scaled_data['num_samples'] = target_size
        else:
            # Upsample with noise
            scale_factor = target_size // current_size
            remainder = target_size % current_size
            
            # Repeat data
            scaled_features = np.tile(base_data['features'], (scale_factor, 1))
            scaled_labels = np.tile(base_data['labels'], scale_factor)
            
            # Add remainder with noise
            if remainder > 0:
                indices = np.random.choice(current_size, remainder, replace=False)
                extra_features = base_data['features'][indices] + np.random.normal(0, 0.1, (remainder, base_data['features'].shape[1]))
                extra_labels = base_data['labels'][indices]
                
                scaled_features = np.vstack([scaled_features, extra_features])
                scaled_labels = np.hstack([scaled_labels, extra_labels])
            
            scaled_data = base_data.copy()
            scaled_data['features'] = scaled_features
            scaled_data['labels'] = scaled_labels
            scaled_data['num_samples'] = target_size
        
        return scaled_data
    
    def _run_single_scalability_test(self, data: Dict[str, Any], 
                                   test_id: str) -> Dict[str, Any]:
        
        result = {
            'success': False,
            'training_time': 0.0,
            'inference_time_ms': 0.0,
            'memory_usage_mb': 0.0,
            'accuracy': 0.0,
            'timeout': False,
            'error': ''
        }
        
        try:
            # Start resource monitoring
            self.resource_monitor.reset()
            self.resource_monitor.start_monitoring()
            
            # Setup timeout
            start_time = time.time()
            timeout_time = start_time + self.config.timeout_seconds
            
            # Create model
            model = HyperPathSVM(
                C=1.0,
                kernel='rbf',  # Use simpler kernel for scalability
                use_ddwe=False,  # Disable complex features for pure scalability test
                quantum_optimization=False
            )
            
            # Prepare data
            X, y = data['features'], data['labels']
            
            # Train/test split
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Training with timeout check
            training_start = time.time()
            
            with self.hyperpath_logger.timer(f"scalability_train_{test_id}"):
                model.fit(X_train, y_train)
            
            training_end = time.time()
            
            # Check timeout
            if training_end > timeout_time:
                result['timeout'] = True
                result['error'] = 'Training timeout'
                return result
            
            # Inference timing
            inference_times = []
            for _ in range(10):  # Multiple runs for stable timing
                inf_start = time.time()
                _ = model.predict(X_test[:100])  # Test on 100 samples
                inf_end = time.time()
                inference_times.append((inf_end - inf_start) * 1000)  # Convert to ms
            
            # Accuracy evaluation
            predictions = model.predict(X_test)
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, predictions)
            
            # Stop monitoring and get stats
            self.resource_monitor.stop_monitoring()
            resource_stats = self.resource_monitor.get_peak_stats()
            
            # Memory usage
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Update result
            result['success'] = True
            result['training_time'] = training_end - training_start
            result['inference_time_ms'] = np.mean(inference_times)
            result['memory_usage_mb'] = current_memory_mb
            result['accuracy'] = accuracy
            result['resource_stats'] = resource_stats
            
            self.logger.debug(f"Test {test_id} completed: "
                            f"Training={result['training_time']:.2f}s, "
                            f"Accuracy={accuracy:.4f}")
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            result['error'] = str(e)
            self.logger.error(f"Scalability test {test_id} failed: {str(e)}")
        
        finally:
            # Cleanup
            gc.collect()
        
        return result
    
    def _analyze_complexity_patterns(self, analysis: ScalabilityAnalysis):
        
        complexity_analysis = {}
        
        # Analyze each scaling dimension
        scaling_results = [
            ('network_size', analysis.network_scaling_results),
            ('data_size', analysis.data_scaling_results),
            ('num_features', analysis.feature_scaling_results),
            ('temporal_length', analysis.temporal_scaling_results)
        ]
        
        for dimension, results in scaling_results:
            if not results:
                continue
            
            successful_results = [r for r in results if r.success]
            if len(successful_results) < 3:
                continue
            
            # Extract scale values and performance metrics
            scale_values = [r.scale_value for r in successful_results]
            training_times = [r.training_time_seconds for r in successful_results]
            memory_usage = [r.memory_usage_mb for r in successful_results]
            
            # Fit complexity curves (log-log regression)
            if len(scale_values) >= 3:
                log_scales = np.log(scale_values)
                log_times = np.log(training_times)
                log_memory = np.log(memory_usage)
                
                # Linear regression in log space: log(y) = a + b*log(x)
                # This gives us y = x^b * exp(a), so b is the complexity exponent
                
                time_coef = np.polyfit(log_scales, log_times, 1)
                memory_coef = np.polyfit(log_scales, log_memory, 1)
                
                complexity_analysis[dimension] = {
                    'time_complexity_exponent': time_coef[0],
                    'time_complexity_constant': np.exp(time_coef[1]),
                    'memory_complexity_exponent': memory_coef[0],
                    'memory_complexity_constant': np.exp(memory_coef[1]),
                    'time_r_squared': stats.pearsonr(log_scales, log_times)[0]**2,
                    'memory_r_squared': stats.pearsonr(log_scales, log_memory)[0]**2
                }
                
                self.logger.info(f"{dimension} complexity analysis:")
                self.logger.info(f"  Time complexity: O(n^{time_coef[0]:.2f})")
                self.logger.info(f"  Memory complexity: O(n^{memory_coef[0]:.2f})")
        
        analysis.complexity_analysis = complexity_analysis
    
    def _identify_bottlenecks(self, analysis: ScalabilityAnalysis):
        
        bottlenecks = {}
        
        # Find dimension with highest complexity exponent
        max_time_complexity = 0
        max_memory_complexity = 0
        critical_dimension_time = None
        critical_dimension_memory = None
        
        for dimension, complexity in analysis.complexity_analysis.items():
            time_exp = complexity.get('time_complexity_exponent', 0)
            memory_exp = complexity.get('memory_complexity_exponent', 0)
            
            if time_exp > max_time_complexity:
                max_time_complexity = time_exp
                critical_dimension_time = dimension
            
            if memory_exp > max_memory_complexity:
                max_memory_complexity = memory_exp
                critical_dimension_memory = dimension
        
        bottlenecks['critical_dimensions'] = {
            'time_bottleneck': critical_dimension_time,
            'memory_bottleneck': critical_dimension_memory,
            'worst_time_complexity': max_time_complexity,
            'worst_memory_complexity': max_memory_complexity
        }
        
        # Identify failure points
        failure_points = {}
        all_results = [
            ('network_scaling', analysis.network_scaling_results),
            ('data_scaling', analysis.data_scaling_results),
            ('feature_scaling', analysis.feature_scaling_results),
            ('temporal_scaling', analysis.temporal_scaling_results)
        ]
        
        for test_type, results in all_results:
            if not results:
                continue
            
            # Find first failure or timeout
            for result in results:
                if not result.success or result.timeout:
                    failure_points[test_type] = {
                        'failed_at_scale': result.scale_value,
                        'failure_reason': result.error_message if result.error_message else 'timeout'
                    }
                    break
        
        bottlenecks['failure_points'] = failure_points
        analysis.bottleneck_analysis = bottlenecks
    
    def _determine_scalability_limits(self, analysis: ScalabilityAnalysis):
        
        limits = {}
        
        # Define performance thresholds
        max_training_time = self.config.max_training_time_hours * 3600  # Convert to seconds
        max_memory = self.config.max_memory_gb * 1024  # Convert to MB
        min_accuracy = self.config.min_accuracy_threshold
        
        all_results = [
            ('network_size', analysis.network_scaling_results),
            ('data_size', analysis.data_scaling_results),
            ('num_features', analysis.feature_scaling_results),
            ('temporal_length', analysis.temporal_scaling_results)
        ]
        
        for dimension, results in all_results:
            if not results:
                continue
            
            # Find practical limits based on thresholds
            practical_limit = None
            
            for result in results:
                if (not result.success or 
                    result.training_time_seconds > max_training_time or
                    result.memory_usage_mb > max_memory or
                    result.accuracy < min_accuracy):
                    
                    # Practical limit is the previous successful scale
                    if result != results[0]:  # Not the first result
                        prev_idx = results.index(result) - 1
                        practical_limit = results[prev_idx].scale_value
                    break
            
            # If all tests passed, use the largest tested scale
            if practical_limit is None and results:
                last_result = results[-1]
                if last_result.success:
                    practical_limit = last_result.scale_value
            
            if practical_limit:
                limits[dimension] = practical_limit
        
        analysis.scalability_limits = limits
        
        self.logger.info("Practical scalability limits:")
        for dimension, limit in limits.items():
            self.logger.info(f"  {dimension}: {limit}")
    
    def _generate_scalability_plots(self, analysis: ScalabilityAnalysis):
       
        try:
            plots_dir = self.config.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Generate plots for each scaling dimension
            self._plot_scaling_curves(analysis, plots_dir)
            self._plot_complexity_analysis(analysis, plots_dir)
            self._plot_resource_utilization(analysis, plots_dir)
            
        except Exception as e:
            self.logger.error(f"Plot generation failed: {str(e)}")
    
    def _plot_scaling_curves(self, analysis: ScalabilityAnalysis, output_dir: Path):
        """Plot scaling curves for all dimensions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        scaling_data = [
            ('Network Size', 'network_scaling_results', 'nodes'),
            ('Data Volume', 'data_scaling_results', 'samples'),
            ('Feature Dimensions', 'feature_scaling_results', 'features'),
            ('Temporal Length', 'temporal_scaling_results', 'time steps')
        ]
        
        for i, (title, attr_name, unit) in enumerate(scaling_data):
            results = getattr(analysis, attr_name, [])
            if not results:
                continue
            
            ax = axes[i]
            
            # Extract successful results
            successful = [r for r in results if r.success]
            if not successful:
                continue
            
            scales = [r.scale_value for r in successful]
            times = [r.training_time_seconds for r in successful]
            accuracies = [r.accuracy for r in successful]
            
            # Plot training time
            ax2 = ax.twinx()
            
            line1 = ax.plot(scales, times, 'b-o', label='Training Time', linewidth=2, markersize=6)
            line2 = ax2.plot(scales, accuracies, 'r-s', label='Accuracy', linewidth=2, markersize=6)
            
            ax.set_xlabel(f'Scale ({unit})')
            ax.set_ylabel('Training Time (seconds)', color='blue')
            ax2.set_ylabel('Accuracy', color='red')
            
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f'{title} Scaling')
            ax.grid(True, alpha=0.3)
            
            # Add complexity annotation if available
            if hasattr(analysis, 'complexity_analysis'):
                dim_key = attr_name.replace('_scaling_results', '').replace('_scaling_results', '')
                if dim_key in analysis.complexity_analysis:
                    complexity = analysis.complexity_analysis[dim_key]
                    time_exp = complexity.get('time_complexity_exponent', 0)
                    ax.text(0.05, 0.95, f'Time Complexity: O(n^{time_exp:.2f})', 
                           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'scaling_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_complexity_analysis(self, analysis: ScalabilityAnalysis, output_dir: Path):
     
        if not analysis.complexity_analysis:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        dimensions = list(analysis.complexity_analysis.keys())
        time_complexities = [analysis.complexity_analysis[dim]['time_complexity_exponent'] for dim in dimensions]
        memory_complexities = [analysis.complexity_analysis[dim]['memory_complexity_exponent'] for dim in dimensions]
        
        # Time complexity comparison
        bars1 = ax1.bar(dimensions, time_complexities, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Complexity Exponent')
        ax1.set_title('Time Complexity by Scaling Dimension')
        ax1.set_xticklabels([d.replace('_', '\n') for d in dimensions])
        
        # Add value labels
        for bar, complexity in zip(bars1, time_complexities):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{complexity:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Memory complexity comparison  
        bars2 = ax2.bar(dimensions, memory_complexities, color='orange', alpha=0.8)
        ax2.set_ylabel('Complexity Exponent')
        ax2.set_title('Memory Complexity by Scaling Dimension')
        ax2.set_xticklabels([d.replace('_', '\n') for d in dimensions])
        
        # Add value labels
        for bar, complexity in zip(bars2, memory_complexities):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{complexity:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_utilization(self, analysis: ScalabilityAnalysis, output_dir: Path):
     
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Combine all results for resource analysis
        all_results = []
        for results_list in [analysis.network_scaling_results, analysis.data_scaling_results,
                           analysis.feature_scaling_results, analysis.temporal_scaling_results]:
            all_results.extend([r for r in results_list if r.success])
        
        if not all_results:
            return
        
        # CPU utilization vs scale
        scales = [r.scale_value for r in all_results]
        cpu_usage = [r.peak_cpu_percent for r in all_results]
        memory_usage = [r.peak_memory_percent for r in all_results]
        memory_mb = [r.memory_usage_mb for r in all_results]
        training_times = [r.training_time_seconds for r in all_results]
        
        ax1.scatter(scales, cpu_usage, alpha=0.6, c=training_times, cmap='viridis')
        ax1.set_xlabel('Scale Value')
        ax1.set_ylabel('Peak CPU Usage (%)')
        ax1.set_title('CPU Utilization vs Scale')
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(scales, memory_usage, alpha=0.6, c=training_times, cmap='viridis')
        ax2.set_xlabel('Scale Value')
        ax2.set_ylabel('Peak Memory Usage (%)')
        ax2.set_title('Memory Utilization vs Scale')
        ax2.grid(True, alpha=0.3)
        
        ax3.scatter(training_times, memory_mb, alpha=0.6, c=scales, cmap='plasma')
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory vs Training Time')
        ax3.grid(True, alpha=0.3)
        
        # Performance efficiency (accuracy per unit time)
        accuracies = [r.accuracy for r in all_results]
        efficiency = [acc / max(time, 0.001) for acc, time in zip(accuracies, training_times)]
        
        ax4.scatter(scales, efficiency, alpha=0.6, c=memory_mb, cmap='coolwarm')
        ax4.set_xlabel('Scale Value')
        ax4.set_ylabel('Accuracy / Training Time')
        ax4.set_title('Performance Efficiency vs Scale')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_scalability_report(self, analysis: ScalabilityAnalysis):
        """Generate comprehensive scalability report."""
        report_path = self.config.results_dir / "scalability_report.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write("# HyperPath-SVM Scalability Analysis Report\n\n")
                
                # Executive summary
                f.write("## Executive Summary\n\n")
                f.write(f"- **Total Tests:** {analysis.total_tests}\n")
                f.write(f"- **Success Rate:** {analysis.successful_tests / max(analysis.total_tests, 1):.1%}\n")
                f.write(f"- **Timeout Rate:** {analysis.timeout_tests / max(analysis.total_tests, 1):.1%}\n\n")
                
                # Scalability limits
                if analysis.scalability_limits:
                    f.write("## Practical Scalability Limits\n\n")
                    f.write("| Dimension | Limit | Unit |\n")
                    f.write("|-----------|-------|------|\n")
                    
                    limit_units = {
                        'network_size': 'nodes',
                        'data_size': 'samples', 
                        'num_features': 'features',
                        'temporal_length': 'time steps'
                    }
                    
                    for dim, limit in analysis.scalability_limits.items():
                        unit = limit_units.get(dim, 'units')
                        f.write(f"| {dim.replace('_', ' ').title()} | {limit:,} | {unit} |\n")
                    f.write("\n")
                
                # Complexity analysis
                if analysis.complexity_analysis:
                    f.write("## Computational Complexity Analysis\n\n")
                    f.write("| Dimension | Time Complexity | Memory Complexity | R² (Time) |\n")
                    f.write("|-----------|-----------------|-------------------|----------|\n")
                    
                    for dim, complexity in analysis.complexity_analysis.items():
                        time_exp = complexity.get('time_complexity_exponent', 0)
                        memory_exp = complexity.get('memory_complexity_exponent', 0)
                        r_squared = complexity.get('time_r_squared', 0)
                        
                        f.write(f"| {dim.replace('_', ' ').title()} | O(n^{time_exp:.2f}) | "
                               f"O(n^{memory_exp:.2f}) | {r_squared:.3f} |\n")
                    f.write("\n")
                
                # Bottleneck analysis
                if analysis.bottleneck_analysis:
                    f.write("## Bottleneck Analysis\n\n")
                    bottlenecks = analysis.bottleneck_analysis
                    
                    if 'critical_dimensions' in bottlenecks:
                        critical = bottlenecks['critical_dimensions']
                        f.write(f"- **Time Bottleneck:** {critical.get('time_bottleneck', 'N/A')}\n")
                        f.write(f"- **Memory Bottleneck:** {critical.get('memory_bottleneck', 'N/A')}\n")
                        f.write(f"- **Worst Time Complexity:** O(n^{critical.get('worst_time_complexity', 0):.2f})\n")
                        f.write(f"- **Worst Memory Complexity:** O(n^{critical.get('worst_memory_complexity', 0):.2f})\n\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                f.write("### Performance Optimization\n")
                f.write("1. **Memory Management:** Implement more efficient memory hierarchy for large networks\n")
                f.write("2. **Algorithmic Optimization:** Consider sparse matrix operations for large networks\n")
                f.write("3. **Parallel Processing:** Leverage multi-threading for independent computations\n\n")
                
                f.write("### Deployment Guidelines\n")
                f.write("1. **Production Limits:** Stay within tested scalability limits\n")
                f.write("2. **Resource Planning:** Allocate resources based on complexity analysis\n")
                f.write("3. **Monitoring:** Implement resource monitoring in production deployments\n")
                
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
    
    def _save_analysis_results(self, analysis: ScalabilityAnalysis):
        """Save complete scalability analysis results."""
        try:
            # Save as JSON
            json_path = self.config.results_dir / "scalability_results.json"
            
            # Convert to JSON-serializable format
            def convert_result(result: ScalabilityResult) -> Dict[str, Any]:
                return {
                    'test_name': result.test_name,
                    'scale_parameter': result.scale_parameter,
                    'scale_value': result.scale_value,
                    'training_time_seconds': result.training_time_seconds,
                    'inference_time_ms': result.inference_time_ms,
                    'memory_usage_mb': result.memory_usage_mb,
                    'accuracy': result.accuracy,
                    'success': result.success,
                    'timeout': result.timeout,
                    'error_message': result.error_message,
                    'peak_cpu_percent': result.peak_cpu_percent,
                    'peak_memory_percent': result.peak_memory_percent
                }
            
            results_dict = {
                'config': {
                    'network_sizes': analysis.config.network_sizes,
                    'data_sizes': analysis.config.data_sizes,
                    'feature_dims': analysis.config.feature_dims,
                    'temporal_lengths': analysis.config.temporal_lengths
                },
                'start_time': analysis.start_time,
                'end_time': analysis.end_time,
                'total_tests': analysis.total_tests,
                'successful_tests': analysis.successful_tests,
                'failed_tests': analysis.failed_tests,
                'timeout_tests': analysis.timeout_tests,
                'network_scaling_results': [convert_result(r) for r in analysis.network_scaling_results],
                'data_scaling_results': [convert_result(r) for r in analysis.data_scaling_results],
                'feature_scaling_results': [convert_result(r) for r in analysis.feature_scaling_results],
                'temporal_scaling_results': [convert_result(r) for r in analysis.temporal_scaling_results],
                'complexity_analysis': analysis.complexity_analysis,
                'bottleneck_analysis': analysis.bottleneck_analysis,
                'scalability_limits': analysis.scalability_limits
            }
            
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            # Save as pickle for Python access
            pickle_path = self.config.results_dir / "scalability_analysis.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(analysis, f)
            
            self.logger.info(f"Results saved to {json_path} and {pickle_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")


def run_scalability_analysis(config: Optional[ScalabilityConfig] = None) -> ScalabilityAnalysis:

    tester = ScalabilityTester(config)
    return tester.run_comprehensive_scalability_analysis()


if __name__ == "__main__":
    # Test scalability analysis
    logger = get_logger(__name__)
    logger.info("Testing scalability analysis framework...")
    
    try:
        # Create test configuration
        test_config = ScalabilityConfig(
            network_sizes=[100, 500, 1000],  # Smaller sizes for testing
            data_sizes=[1000, 5000],
            feature_dims=[10, 50],
            temporal_lengths=[10, 50],
            max_training_time_hours=0.1,  # Short timeout for testing
            results_dir=Path("test_scalability"),
            generate_plots=True
        )
        
        # Run scalability analysis
        analysis = run_scalability_analysis(test_config)
        
        logger.info("Scalability Analysis Results:")
        logger.info(f"Total tests: {analysis.total_tests}")
        logger.info(f"Success rate: {analysis.successful_tests / max(analysis.total_tests, 1):.1%}")
        logger.info(f"Scalability limits: {analysis.scalability_limits}")
        
        if analysis.complexity_analysis:
            logger.info("Complexity Analysis:")
            for dim, complexity in analysis.complexity_analysis.items():
                time_exp = complexity.get('time_complexity_exponent', 0)
                logger.info(f"  {dim}: Time complexity O(n^{time_exp:.2f})")
        
    except Exception as e:
        logger.error(f"Scalability analysis test failed: {str(e)}")
    
    logger.info("Scalability analysis framework testing completed!")