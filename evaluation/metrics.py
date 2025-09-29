# File: hyperpath_svm/evaluation/metrics.py

"""
Comprehensive Evaluation Metrics for HyperPath-SVM

This module implements all performance metrics required for evaluating the HyperPath-SVM
framework against baselines. It includes network-specific metrics alongside traditional
ML metrics, with special focus on routing quality and real-time performance.

"""

import numpy as np
import time
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import sys

from ..utils.logging_utils import get_logger


@dataclass
class PerformanceTarget:
   
    accuracy: float = 0.965  # 96.5%
    inference_time_ms: float = 1.8  # milliseconds
    memory_usage_mb: float = 98.0  # megabytes
    adaptation_time_min: float = 2.3  # minutes
    network_overhead_pct: float = 5.0  # percentage


@dataclass
class MetricResult:
    
    name: str
    value: float
    unit: str = ""
    target: Optional[float] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.target is not None:
            if self.name.endswith('_time') or 'latency' in self.name or 'overhead' in self.name:
                # Lower is better for time/latency/overhead metrics
                self.passed = self.value <= self.target
            else:
                # Higher is better for accuracy/performance metrics
                self.passed = self.value >= self.target


@dataclass
class EvaluationResults:
  
    model_name: str
    dataset_name: str
    timestamp: float = field(default_factory=time.time)
    metrics: List[MetricResult] = field(default_factory=list)
    confusion_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    detailed_stats: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, name: str) -> Optional[MetricResult]:
      
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
    
    def get_metric_value(self, name: str) -> Optional[float]:
       
        metric = self.get_metric(name)
        return metric.value if metric else None
    
    def passed_targets(self) -> bool:
        
        return all(metric.passed for metric in self.metrics if metric.target is not None)
    
    def summary_dict(self) -> Dict[str, Any]:
       
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'timestamp': self.timestamp,
            'total_metrics': len(self.metrics),
            'passed_targets': self.passed_targets(),
            'key_metrics': {metric.name: metric.value for metric in self.metrics[:5]}
        }


class BaseMetric(ABC):
   
    
    def __init__(self, name: str, unit: str = "", target: Optional[float] = None):
        self.name = name
        self.unit = unit
        self.target = target
        self.logger = get_logger(__name__)
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                **kwargs) -> MetricResult:
       
        pass
    
    def _create_result(self, value: float, metadata: Optional[Dict] = None) -> MetricResult:
     
        return MetricResult(
            name=self.name,
            value=value,
            unit=self.unit,
            target=self.target,
            metadata=metadata or {}
        )


class AccuracyMetric(BaseMetric):
   
    
    def __init__(self, target: float = 0.965):
        super().__init__("accuracy", "fraction", target)
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> MetricResult:
      
        try:
            accuracy = accuracy_score(y_true, y_pred)
            
            # Additional metadata
            metadata = {
                'num_samples': len(y_true),
                'num_correct': np.sum(y_true == y_pred),
                'error_rate': 1 - accuracy
            }
            
            return self._create_result(accuracy, metadata)
            
        except Exception as e:
            self.logger.error(f"Error computing accuracy: {str(e)}")
            return self._create_result(0.0, {'error': str(e)})


class PrecisionRecallF1Metric(BaseMetric):
   
    
    def __init__(self, average: str = 'weighted'):
        super().__init__("precision_recall_f1", "scores")
        self.average = average
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> List[MetricResult]:
       
        try:
            precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)
            recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=self.average, zero_division=0)
            
            results = []
            
            # Create individual metric results
            results.append(MetricResult("precision", precision, "fraction", metadata={'average': self.average}))
            results.append(MetricResult("recall", recall, "fraction", metadata={'average': self.average}))
            results.append(MetricResult("f1_score", f1, "fraction", metadata={'average': self.average}))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error computing precision/recall/F1: {str(e)}")
            error_result = [MetricResult(name, 0.0, "fraction", metadata={'error': str(e)}) 
                          for name in ["precision", "recall", "f1_score"]]
            return error_result


class InferenceTimeMetric(BaseMetric):
    
    
    def __init__(self, target_ms: float = 1.8):
        super().__init__("inference_time", "ms", target_ms)
        self.measurements = deque(maxlen=1000)  # Keep last 1000 measurements
    
    @contextmanager
    def measure(self):
       
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            self.measurements.append(inference_time_ms)
    
    def compute(self, model: Any, X: np.ndarray, **kwargs) -> MetricResult:
       
        try:
            times = []
            num_runs = kwargs.get('num_runs', 100)
            
            # Warm-up runs (not measured)
            for _ in range(10):
                _ = model.predict(X[:1])
            
            # Measured runs
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model.predict(X[:1])  # Single sample inference
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            mean_time = np.mean(times)
            
            metadata = {
                'num_runs': num_runs,
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'p95_time': np.percentile(times, 95),
                'p99_time': np.percentile(times, 99)
            }
            
            return self._create_result(mean_time, metadata)
            
        except Exception as e:
            self.logger.error(f"Error measuring inference time: {str(e)}")
            return self._create_result(float('inf'), {'error': str(e)})


class MemoryUsageMetric(BaseMetric):
    
    
    def __init__(self, target_mb: float = 98.0):
        super().__init__("memory_usage", "MB", target_mb)
        self.process = psutil.Process()
    
    def compute(self, model: Any, **kwargs) -> MetricResult:
        
        try:
            # Get memory info before and after model operations
            initial_memory = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
            # Force garbage collection to get accurate baseline
            import gc
            gc.collect()
            
            # Simulate model operations
            if hasattr(model, 'predict') and 'X_test' in kwargs:
                X_test = kwargs['X_test']
                _ = model.predict(X_test)
            
            current_memory = self.process.memory_info().rss / (1024 * 1024)
            model_memory = current_memory - initial_memory
            
            # Get additional memory statistics
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            metadata = {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'memory_percent': memory_percent,
                'baseline_memory_mb': initial_memory,
                'current_memory_mb': current_memory
            }
            
            return self._create_result(max(model_memory, current_memory), metadata)
            
        except Exception as e:
            self.logger.error(f"Error measuring memory usage: {str(e)}")
            return self._create_result(float('inf'), {'error': str(e)})


class AdaptationTimeMetric(BaseMetric):
  
    
    def __init__(self, target_min: float = 2.3):
        super().__init__("adaptation_time", "minutes", target_min)
    
    def compute(self, model: Any, X_new: np.ndarray, y_new: np.ndarray, 
                **kwargs) -> MetricResult:
        
        try:
            start_time = time.perf_counter()
            
            if hasattr(model, 'partial_fit'):
                # Online learning
                model.partial_fit(X_new, y_new)
            elif hasattr(model, 'fit'):
                # Batch retraining
                model.fit(X_new, y_new)
            else:
                raise ValueError("Model does not support adaptation")
            
            end_time = time.perf_counter()
            adaptation_time_min = (end_time - start_time) / 60.0
            
            metadata = {
                'adaptation_method': 'partial_fit' if hasattr(model, 'partial_fit') else 'fit',
                'num_samples': len(X_new),
                'adaptation_time_seconds': (end_time - start_time)
            }
            
            return self._create_result(adaptation_time_min, metadata)
            
        except Exception as e:
            self.logger.error(f"Error measuring adaptation time: {str(e)}")
            return self._create_result(float('inf'), {'error': str(e)})


class NetworkPerformanceMetric(BaseMetric):
    
    
    def __init__(self, target_overhead_pct: float = 5.0):
        super().__init__("network_overhead", "percent", target_overhead_pct)
    
    def compute(self, y_true_performance: np.ndarray, 
                y_pred_performance: np.ndarray, **kwargs) -> MetricResult:
        
        try:
            # Calculate relative performance change
            performance_ratio = y_pred_performance / (y_true_performance + 1e-10)  # Avoid division by zero
            overhead_pct = np.mean(np.maximum(0, (performance_ratio - 1) * 100))
            
            # Additional network metrics
            latency_increase = np.mean(np.maximum(0, y_pred_performance - y_true_performance))
            throughput_loss = np.mean(np.maximum(0, y_true_performance - y_pred_performance)) / np.mean(y_true_performance)
            
            metadata = {
                'avg_performance_ratio': np.mean(performance_ratio),
                'latency_increase': latency_increase,
                'throughput_loss_pct': throughput_loss * 100,
                'num_measurements': len(y_true_performance),
                'performance_std': np.std(performance_ratio)
            }
            
            return self._create_result(overhead_pct, metadata)
            
        except Exception as e:
            self.logger.error(f"Error computing network performance: {str(e)}")
            return self._create_result(float('inf'), {'error': str(e)})


class RoutingQualityMetric(BaseMetric):
    
    
    def __init__(self):
        super().__init__("routing_quality", "score")
    
    def compute(self, optimal_paths: List[List[int]], 
                predicted_paths: List[List[int]], **kwargs) -> MetricResult:
        
        try:
            if len(optimal_paths) != len(predicted_paths):
                raise ValueError("Number of optimal and predicted paths must match")
            
            path_similarities = []
            hop_accuracies = []
            
            for opt_path, pred_path in zip(optimal_paths, predicted_paths):
                # Path similarity (Jaccard index)
                opt_set = set(opt_path)
                pred_set = set(pred_path)
                similarity = len(opt_set & pred_set) / len(opt_set | pred_set) if opt_set | pred_set else 0
                path_similarities.append(similarity)
                
                # Hop-by-hop accuracy
                min_len = min(len(opt_path), len(pred_path))
                if min_len > 0:
                    hop_accuracy = sum(1 for i in range(min_len) if opt_path[i] == pred_path[i]) / min_len
                    hop_accuracies.append(hop_accuracy)
            
            # Overall routing quality score
            avg_similarity = np.mean(path_similarities) if path_similarities else 0
            avg_hop_accuracy = np.mean(hop_accuracies) if hop_accuracies else 0
            routing_quality = (avg_similarity + avg_hop_accuracy) / 2
            
            metadata = {
                'path_similarity': avg_similarity,
                'hop_accuracy': avg_hop_accuracy,
                'num_paths': len(optimal_paths),
                'similarity_std': np.std(path_similarities) if path_similarities else 0
            }
            
            return self._create_result(routing_quality, metadata)
            
        except Exception as e:
            self.logger.error(f"Error computing routing quality: {str(e)}")
            return self._create_result(0.0, {'error': str(e)})


class ConvergenceMetric(BaseMetric):
   
    
    def __init__(self):
        super().__init__("convergence_rate", "iterations")
    
    def compute(self, training_history: Dict[str, List[float]], **kwargs) -> MetricResult:
        
        try:
            # Determine convergence based on loss or accuracy
            if 'loss' in training_history:
                values = training_history['loss']
                # Find iteration where loss stabilizes (change < 1% for 10 consecutive iterations)
                threshold = 0.01
                stability_window = 10
            elif 'accuracy' in training_history:
                values = training_history['accuracy']
                threshold = 0.001  # Smaller threshold for accuracy
                stability_window = 10
            else:
                raise ValueError("Training history must contain 'loss' or 'accuracy'")
            
            if len(values) < stability_window + 1:
                return self._create_result(len(values), {'converged': False})
            
            # Find convergence point
            convergence_iteration = len(values)  # Default to end if no convergence
            converged = False
            
            for i in range(stability_window, len(values)):
                window_values = values[i-stability_window:i]
                if len(window_values) >= 2:
                    relative_changes = [abs(window_values[j] - window_values[j-1]) / (abs(window_values[j-1]) + 1e-10) 
                                      for j in range(1, len(window_values))]
                    if all(change < threshold for change in relative_changes):
                        convergence_iteration = i
                        converged = True
                        break
            
            # Calculate convergence rate (1/iterations for faster convergence = higher rate)
            convergence_rate = 1.0 / convergence_iteration if convergence_iteration > 0 else 0
            
            metadata = {
                'converged': converged,
                'convergence_iteration': convergence_iteration,
                'total_iterations': len(values),
                'final_value': values[-1] if values else None,
                'convergence_threshold': threshold
            }
            
            return self._create_result(convergence_rate, metadata)
            
        except Exception as e:
            self.logger.error(f"Error computing convergence: {str(e)}")
            return self._create_result(0.0, {'error': str(e)})


class RobustnessMetric(BaseMetric):
   
    
    def __init__(self):
        super().__init__("robustness_score", "fraction")
    
    def compute(self, model: Any, X_clean: np.ndarray, y_clean: np.ndarray,
                noise_levels: List[float] = None, **kwargs) -> MetricResult:
       
        try:
            if noise_levels is None:
                noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
            
            # Baseline accuracy on clean data
            y_pred_clean = model.predict(X_clean)
            clean_accuracy = accuracy_score(y_clean, y_pred_clean)
            
            # Test with different noise levels
            noisy_accuracies = []
            
            for noise_level in noise_levels:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, X_clean.shape)
                X_noisy = X_clean + noise
                
                y_pred_noisy = model.predict(X_noisy)
                noisy_accuracy = accuracy_score(y_clean, y_pred_noisy)
                noisy_accuracies.append(noisy_accuracy)
            
            # Robustness score: average relative performance retention
            relative_performances = [acc / clean_accuracy for acc in noisy_accuracies]
            robustness_score = np.mean(relative_performances)
            
            metadata = {
                'clean_accuracy': clean_accuracy,
                'noise_levels': noise_levels,
                'noisy_accuracies': noisy_accuracies,
                'relative_performances': relative_performances,
                'min_relative_performance': min(relative_performances),
                'std_relative_performance': np.std(relative_performances)
            }
            
            return self._create_result(robustness_score, metadata)
            
        except Exception as e:
            self.logger.error(f"Error computing robustness: {str(e)}")
            return self._create_result(0.0, {'error': str(e)})


class ComprehensiveMetricsEvaluator:
    
    
    def __init__(self, targets: Optional[PerformanceTarget] = None):
        self.targets = targets or PerformanceTarget()
        self.logger = get_logger(__name__)
        
        # Initialize metrics
        self.metrics = {
            'accuracy': AccuracyMetric(self.targets.accuracy),
            'precision_recall_f1': PrecisionRecallF1Metric(),
            'inference_time': InferenceTimeMetric(self.targets.inference_time_ms),
            'memory_usage': MemoryUsageMetric(self.targets.memory_usage_mb),
            'adaptation_time': AdaptationTimeMetric(self.targets.adaptation_time_min),
            'network_performance': NetworkPerformanceMetric(self.targets.network_overhead_pct),
            'routing_quality': RoutingQualityMetric(),
            'convergence': ConvergenceMetric(),
            'robustness': RobustnessMetric()
        }
        
        # Thread pool for parallel metric computation
        self.max_workers = 4
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "unknown", dataset_name: str = "unknown",
                      **kwargs) -> EvaluationResults:
        
        try:
            self.logger.info(f"Starting comprehensive evaluation of {model_name} on {dataset_name}")
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            results = EvaluationResults(
                model_name=model_name,
                dataset_name=dataset_name,
                timestamp=time.time()
            )
            
            # Compute metrics in parallel where possible
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_metric = {}
                
                # Basic ML metrics (can be computed in parallel)
                future_to_metric[executor.submit(
                    self.metrics['accuracy'].compute, y_test, y_pred
                )] = 'accuracy'
                
                future_to_metric[executor.submit(
                    self.metrics['precision_recall_f1'].compute, y_test, y_pred
                )] = 'precision_recall_f1'
                
                # Performance metrics (need model access)
                if hasattr(model, 'predict'):
                    future_to_metric[executor.submit(
                        self.metrics['inference_time'].compute, model, X_test
                    )] = 'inference_time'
                    
                    future_to_metric[executor.submit(
                        self.metrics['memory_usage'].compute, model, X_test=X_test
                    )] = 'memory_usage'
                
                # Robustness metric
                future_to_metric[executor.submit(
                    self.metrics['robustness'].compute, model, X_test, y_test
                )] = 'robustness'
                
                # Collect results
                for future in as_completed(future_to_metric):
                    metric_name = future_to_metric[future]
                    try:
                        result = future.result()
                        if isinstance(result, list):
                            results.metrics.extend(result)
                        else:
                            results.metrics.append(result)
                    except Exception as e:
                        self.logger.error(f"Error computing {metric_name}: {str(e)}")
            
            # Compute metrics that require special data
            self._compute_special_metrics(results, model, X_test, y_test, **kwargs)
            
            # Add confusion matrix
            if len(np.unique(y_test)) <= 10:  # Only for reasonable number of classes
                results.confusion_matrices['main'] = confusion_matrix(y_test, y_pred)
            
            # Generate detailed statistics
            results.detailed_stats = self._generate_detailed_stats(y_test, y_pred, results.metrics)
            
            self.logger.info(f"Evaluation completed. Computed {len(results.metrics)} metrics.")
            self.logger.info(f"Target compliance: {results.passed_targets()}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {str(e)}")
            # Return empty results with error
            return EvaluationResults(
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=[MetricResult("evaluation_error", 0.0, metadata={'error': str(e)})]
            )
    
    def _compute_special_metrics(self, results: EvaluationResults, model: Any,
                               X_test: np.ndarray, y_test: np.ndarray, **kwargs):
        
        try:
            # Adaptation time (if adaptation data provided)
            if 'X_adapt' in kwargs and 'y_adapt' in kwargs:
                adapt_result = self.metrics['adaptation_time'].compute(
                    model, kwargs['X_adapt'], kwargs['y_adapt']
                )
                results.metrics.append(adapt_result)
            
            # Network performance (if performance data provided)
            if 'y_true_performance' in kwargs and 'y_pred_performance' in kwargs:
                network_result = self.metrics['network_performance'].compute(
                    kwargs['y_true_performance'], kwargs['y_pred_performance']
                )
                results.metrics.append(network_result)
            
            # Routing quality (if path data provided)
            if 'optimal_paths' in kwargs and 'predicted_paths' in kwargs:
                routing_result = self.metrics['routing_quality'].compute(
                    kwargs['optimal_paths'], kwargs['predicted_paths']
                )
                results.metrics.append(routing_result)
            
            # Convergence (if training history provided)
            if 'training_history' in kwargs:
                convergence_result = self.metrics['convergence'].compute(
                    kwargs['training_history']
                )
                results.metrics.append(convergence_result)
                
        except Exception as e:
            self.logger.error(f"Error computing special metrics: {str(e)}")
    
    def _generate_detailed_stats(self, y_true: np.ndarray, y_pred: np.ndarray,
                               metrics: List[MetricResult]) -> Dict[str, Any]:
       
        try:
            # Basic classification stats
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            stats = {
                'num_samples': len(y_true),
                'num_classes': len(unique_classes),
                'class_distribution': {
                    str(cls): int(np.sum(y_true == cls)) for cls in unique_classes
                },
                'prediction_distribution': {
                    str(cls): int(np.sum(y_pred == cls)) for cls in unique_classes
                },
                'metrics_summary': {
                    'total_metrics': len(metrics),
                    'passed_targets': sum(1 for m in metrics if m.passed is True),
                    'failed_targets': sum(1 for m in metrics if m.passed is False),
                    'no_targets': sum(1 for m in metrics if m.passed is None)
                }
            }
            
            # Performance summary
            key_metrics = ['accuracy', 'inference_time', 'memory_usage', 'adaptation_time']
            for metric_name in key_metrics:
                for metric in metrics:
                    if metric.name == metric_name:
                        stats[f'{metric_name}_status'] = {
                            'value': metric.value,
                            'target': metric.target,
                            'passed': metric.passed,
                            'unit': metric.unit
                        }
                        break
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating detailed stats: {str(e)}")
            return {'error': str(e)}
    
    def compare_models(self, results_list: List[EvaluationResults]) -> Dict[str, Any]:
        
        if not results_list:
            return {}
        
        try:
            comparison = {
                'models': [r.model_name for r in results_list],
                'datasets': list(set(r.dataset_name for r in results_list)),
                'comparison_metrics': {}
            }
            
            # Get common metrics
            all_metric_names = set()
            for results in results_list:
                all_metric_names.update(m.name for m in results.metrics)
            
            # Compare each metric
            for metric_name in all_metric_names:
                metric_values = []
                for results in results_list:
                    value = results.get_metric_value(metric_name)
                    metric_values.append(value if value is not None else float('nan'))
                
                comparison['comparison_metrics'][metric_name] = {
                    'values': dict(zip([r.model_name for r in results_list], metric_values)),
                    'best_model': results_list[np.nanargmin(metric_values) if 'time' in metric_name or 'overhead' in metric_name 
                                            else np.nanargmax(metric_values)].model_name,
                    'ranking': sorted(zip([r.model_name for r in results_list], metric_values), 
                                    key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'),
                                    reverse=not ('time' in metric_name or 'overhead' in metric_name))
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return {'error': str(e)}


def create_evaluator(custom_targets: Optional[Dict[str, float]] = None) -> ComprehensiveMetricsEvaluator:
    
    if custom_targets:
        targets = PerformanceTarget(**custom_targets)
    else:
        targets = PerformanceTarget()
    
    return ComprehensiveMetricsEvaluator(targets)


if __name__ == "__main__":
    # Example usage and testing
    logger = get_logger(__name__)
    logger.info("Testing comprehensive metrics evaluation...")
    
    # Create test data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    
    # Mock model for testing
    class MockModel:
        def predict(self, X):
            return np.random.randint(0, 3, len(X))
    
    model = MockModel()
    X_test = np.random.random((1000, 10))
    
    # Create evaluator and run evaluation
    evaluator = create_evaluator()
    results = evaluator.evaluate_model(model, X_test, y_true, "TestModel", "TestDataset")
    
    # Print results
    logger.info(f"Evaluation Results:")
    logger.info(f"Model: {results.model_name}")
    logger.info(f"Dataset: {results.dataset_name}")
    logger.info(f"Total Metrics: {len(results.metrics)}")
    logger.info(f"Passed Targets: {results.passed_targets()}")
    
    for metric in results.metrics[:5]:  # Show first 5 metrics
        logger.info(f"{metric.name}: {metric.value:.4f} {metric.unit} "
                   f"(Target: {metric.target}, Passed: {metric.passed})")
    
    logger.info("Comprehensive metrics evaluation test completed successfully!") 
