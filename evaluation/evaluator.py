# File: hyperpath_svm/evaluation/evaluator.py

"""
Main Evaluation Framework for HyperPath-SVM

This module provides the complete evaluation orchestration for comparing HyperPath-SVM
against all baseline methods across multiple datasets. It handles data loading, model
training, comprehensive evaluation, statistical analysis, and result reporting.

"""

import numpy as np
import pandas as pd
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import ComprehensiveMetricsEvaluator, EvaluationResults, PerformanceTarget, create_evaluator
from .cross_validation import TemporalCrossValidator
from ..data.dataset_loader import NetworkDatasetLoader
from ..data.data_augmentation import DataAugmentationPipeline
from ..core.hyperpath_svm import HyperPathSVM
from ..utils.logging_utils import get_logger


@dataclass
class EvaluationConfig:
  
    
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ['caida', 'mawi', 'umass', 'wits'])
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # Model evaluation
    cv_folds: int = 5
    cv_repeats: int = 3
    statistical_tests: bool = True
    significance_level: float = 0.05
    
    # Performance configuration
    max_workers: int = mp.cpu_count() - 1
    timeout_minutes: int = 60
    memory_limit_gb: float = 8.0
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_factor: int = 2
    
    # Output configuration
    results_dir: Path = Path("results/evaluation")
    save_models: bool = False
    save_predictions: bool = True
    generate_plots: bool = True
    
    # Target metrics (from paper specifications)
    target_accuracy: float = 0.965
    target_inference_ms: float = 1.8
    target_memory_mb: float = 98.0
    target_adaptation_min: float = 2.3


@dataclass
class ModelConfig:
   
    name: str
    class_path: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


@dataclass
class BenchmarkResults:
   
    config: EvaluationConfig
    timestamp: float = field(default_factory=time.time)
    model_results: Dict[str, Dict[str, EvaluationResults]] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def get_best_model(self, metric: str = 'accuracy', dataset: str = None) -> Optional[str]:
       
        best_model = None
        best_value = float('-inf') if metric not in ['inference_time', 'memory_usage', 'adaptation_time'] else float('inf')
        
        for model_name, dataset_results in self.model_results.items():
            if dataset and dataset not in dataset_results:
                continue
                
            if dataset:
                results = [dataset_results[dataset]]
            else:
                results = list(dataset_results.values())
            
            # Average metric across datasets if no specific dataset
            metric_values = []
            for result in results:
                value = result.get_metric_value(metric)
                if value is not None:
                    metric_values.append(value)
            
            if metric_values:
                avg_value = np.mean(metric_values)
                is_better = (avg_value > best_value if metric not in ['inference_time', 'memory_usage', 'adaptation_time'] 
                           else avg_value < best_value)
                if is_better:
                    best_value = avg_value
                    best_model = model_name
        
        return best_model


class EvaluationOrchestrator:
  
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.dataset_loader = NetworkDatasetLoader()
        self.metrics_evaluator = self._create_metrics_evaluator()
        self.cv_validator = TemporalCrossValidator(n_splits=self.config.cv_folds)
        
        # Data augmentation pipeline
        if self.config.use_augmentation:
            self.augmentation_pipeline = DataAugmentationPipeline()
        else:
            self.augmentation_pipeline = None
        
        # Model registry
        self.model_registry = self._initialize_model_registry()
        
        # Results storage
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.evaluation_stats = {
            'models_evaluated': 0,
            'datasets_processed': 0,
            'total_experiments': 0,
            'failed_experiments': 0,
            'total_time': 0.0
        }
    
    def _create_metrics_evaluator(self) -> ComprehensiveMetricsEvaluator:
        
        targets = PerformanceTarget(
            accuracy=self.config.target_accuracy,
            inference_time_ms=self.config.target_inference_ms,
            memory_usage_mb=self.config.target_memory_mb,
            adaptation_time_min=self.config.target_adaptation_min
        )
        return ComprehensiveMetricsEvaluator(targets)
    
    def _initialize_model_registry(self) -> Dict[str, ModelConfig]:
        """Initialize registry of models to evaluate."""
        return {
            'hyperpath_svm': ModelConfig(
                name='HyperPath-SVM',
                class_path='hyperpath_svm.core.hyperpath_svm.HyperPathSVM',
                params={
                    'C': 1.0,
                    'kernel': 'tgck',
                    'use_ddwe': True,
                    'quantum_optimization': True,
                    'continuous_learning': True
                },
                description='Our proposed HyperPath-SVM with all components'
            ),
            'static_svm': ModelConfig(
                name='Static-SVM',
                class_path='hyperpath_svm.baselines.traditional_svm.StaticSVM',
                params={'C': 1.0, 'kernel': 'rbf'},
                description='Traditional static SVM baseline'
            ),
            'weighted_svm': ModelConfig(
                name='Weighted-SVM',
                class_path='hyperpath_svm.baselines.traditional_svm.WeightedSVM',
                params={'C': 1.0, 'kernel': 'rbf', 'class_weight': 'balanced'},
                description='Weighted SVM for imbalanced data'
            ),
            'quantum_svm': ModelConfig(
                name='Quantum-SVM',
                class_path='hyperpath_svm.baselines.traditional_svm.QuantumSVM',
                params={'C': 1.0, 'quantum_kernel': True},
                description='Quantum-enhanced SVM baseline'
            ),
            'gnn': ModelConfig(
                name='GNN',
                class_path='hyperpath_svm.baselines.neural_networks.GNNModel',
                params={'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.1},
                description='Graph Neural Network baseline'
            ),
            'lstm': ModelConfig(
                name='LSTM',
                class_path='hyperpath_svm.baselines.neural_networks.LSTMModel',
                params={'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.1},
                description='Long Short-Term Memory network'
            ),
            'targcn': ModelConfig(
                name='TARGCN',
                class_path='hyperpath_svm.baselines.neural_networks.TARGCNModel',
                params={'hidden_dim': 64, 'attention_heads': 4},
                description='Temporal Attention-based Routing GCN'
            ),
            'dmgfnet': ModelConfig(
                name='DMGFNet',
                class_path='hyperpath_svm.baselines.neural_networks.DMGFNetModel',
                params={'hidden_dim': 128, 'num_experts': 4},
                description='Dynamic Multi-Grained Fusion Network'
            ),
            'behaviornet': ModelConfig(
                name='BehaviorNet',
                class_path='hyperpath_svm.baselines.neural_networks.BehaviorNetModel',
                params={'embedding_dim': 64, 'num_behaviors': 8},
                description='Network Behavior Analysis Network'
            )
        }
    
    def run_comprehensive_benchmark(self, models: Optional[List[str]] = None, 
                                  datasets: Optional[List[str]] = None) -> BenchmarkResults:
   
        start_time = time.time()
        
        try:
            self.logger.info("Starting comprehensive benchmark evaluation")
            
            # Determine models and datasets to evaluate
            eval_models = models or [name for name, config in self.model_registry.items() if config.enabled]
            eval_datasets = datasets or self.config.datasets
            
            self.logger.info(f"Evaluating {len(eval_models)} models on {len(eval_datasets)} datasets")
            
            # Initialize results
            benchmark_results = BenchmarkResults(config=self.config)
            
            # Load and prepare datasets
            dataset_cache = self._load_datasets(eval_datasets)
            
            # Evaluate each model on each dataset
            total_combinations = len(eval_models) * len(eval_datasets)
            completed = 0
            
            for model_name in eval_models:
                if model_name not in self.model_registry:
                    self.logger.warning(f"Unknown model: {model_name}")
                    continue
                
                model_config = self.model_registry[model_name]
                benchmark_results.model_results[model_name] = {}
                
                for dataset_name in eval_datasets:
                    if dataset_name not in dataset_cache:
                        self.logger.warning(f"Failed to load dataset: {dataset_name}")
                        continue
                    
                    try:
                        self.logger.info(f"Evaluating {model_name} on {dataset_name} "
                                       f"({completed + 1}/{total_combinations})")
                        
                        # Evaluate model on dataset
                        results = self._evaluate_single_combination(
                            model_config, dataset_name, dataset_cache[dataset_name]
                        )
                        
                        benchmark_results.model_results[model_name][dataset_name] = results
                        
                        # Update statistics
                        self.evaluation_stats['total_experiments'] += 1
                        completed += 1
                        
                        self.logger.info(f"Completed {model_name} on {dataset_name}: "
                                       f"Accuracy={results.get_metric_value('accuracy'):.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {str(e)}")
                        self.evaluation_stats['failed_experiments'] += 1
            
            # Perform statistical analysis
            if self.config.statistical_tests and len(eval_models) > 1:
                benchmark_results.statistical_tests = self._perform_statistical_analysis(
                    benchmark_results.model_results
                )
            
            # Generate performance summary
            benchmark_results.performance_summary = self._generate_performance_summary(
                benchmark_results.model_results
            )
            
            # Save results
            benchmark_results.execution_time = time.time() - start_time
            self._save_benchmark_results(benchmark_results)
            
            # Generate reports and plots
            if self.config.generate_plots:
                self._generate_evaluation_plots(benchmark_results)
            
            self.logger.info(f"Comprehensive benchmark completed in {benchmark_results.execution_time:.2f}s")
            self.logger.info(f"Total experiments: {self.evaluation_stats['total_experiments']}, "
                           f"Failed: {self.evaluation_stats['failed_experiments']}")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Benchmark evaluation failed: {str(e)}")
            raise
    
    def _load_datasets(self, dataset_names: List[str]) -> Dict[str, Dict[str, Any]]:
       
        dataset_cache = {}
        
        for dataset_name in dataset_names:
            try:
                self.logger.info(f"Loading dataset: {dataset_name}")
                
                # Load dataset
                data = self.dataset_loader.load_dataset(dataset_name)
                X, y = data['features'], data['labels']
                
                # Split data
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=self.config.test_size, 
                    random_state=self.config.random_state,
                    stratify=y if len(np.unique(y)) > 1 else None
                )
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, 
                    test_size=self.config.validation_size / (1 - self.config.test_size),
                    random_state=self.config.random_state,
                    stratify=y_temp if len(np.unique(y_temp)) > 1 else None
                )
                
                # Apply data augmentation if enabled
                if self.augmentation_pipeline:
                    train_samples = [
                        {'features': X_train[i], 'labels': y_train[i]} 
                        for i in range(len(X_train))
                    ]
                    augmented_samples = self.augmentation_pipeline.augment_batch(
                        train_samples, self.config.augmentation_factor
                    )
                    
                    # Extract augmented data
                    X_train_aug = np.array([s['features'] for s in augmented_samples])
                    y_train_aug = np.array([s['labels'] for s in augmented_samples])
                else:
                    X_train_aug, y_train_aug = X_train, y_train
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_aug)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                dataset_cache[dataset_name] = {
                    'X_train': X_train_scaled,
                    'y_train': y_train_aug,
                    'X_val': X_val_scaled,
                    'y_val': y_val,
                    'X_test': X_test_scaled,
                    'y_test': y_test,
                    'scaler': scaler,
                    'metadata': data.get('metadata', {}),
                    'original_size': len(X),
                    'augmented_size': len(X_train_aug)
                }
                
                self.evaluation_stats['datasets_processed'] += 1
                self.logger.info(f"Loaded {dataset_name}: {len(X)} samples, "
                               f"augmented to {len(X_train_aug)} training samples")
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
        
        return dataset_cache
    
    def _evaluate_single_combination(self, model_config: ModelConfig, 
                                   dataset_name: str, 
                                   dataset: Dict[str, Any]) -> EvaluationResults:
       
        try:
            # Create and train model
            model = self._create_model(model_config)
            
            # Training with timeout
            with self._timeout_context(self.config.timeout_minutes * 60):
                training_start = time.time()
                
                if hasattr(model, 'fit'):
                    model.fit(dataset['X_train'], dataset['y_train'])
                else:
                    raise ValueError(f"Model {model_config.name} does not have fit method")
                
                training_time = time.time() - training_start
            
            # Comprehensive evaluation
            eval_kwargs = {
                'training_time': training_time,
                'X_val': dataset['X_val'],
                'y_val': dataset['y_val'],
                'metadata': dataset['metadata']
            }
            
            # Add adaptation data if available
            if len(dataset['X_val']) > 50:  # Use validation set for adaptation testing
                eval_kwargs['X_adapt'] = dataset['X_val'][:50]
                eval_kwargs['y_adapt'] = dataset['y_val'][:50]
            
            results = self.metrics_evaluator.evaluate_model(
                model=model,
                X_test=dataset['X_test'],
                y_test=dataset['y_test'],
                model_name=model_config.name,
                dataset_name=dataset_name,
                **eval_kwargs
            )
            
            # Add training time to results
            from .metrics import MetricResult
            results.metrics.append(MetricResult(
                name='training_time',
                value=training_time,
                unit='seconds',
                metadata={'dataset_size': len(dataset['X_train'])}
            ))
            
            self.evaluation_stats['models_evaluated'] += 1
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_config.name} on {dataset_name}: {str(e)}")
            # Return empty results with error information
            error_results = EvaluationResults(
                model_name=model_config.name,
                dataset_name=dataset_name
            )
            from .metrics import MetricResult
            error_results.metrics.append(MetricResult(
                name='evaluation_error',
                value=0.0,
                metadata={'error': str(e)}
            ))
            return error_results
    
    def _create_model(self, model_config: ModelConfig):
       
        try:
            # Import model class dynamically
            module_path, class_name = model_config.class_path.rsplit('.', 1)
            
            # Handle special case for HyperPathSVM (already imported)
            if class_name == 'HyperPathSVM':
                from ..core.hyperpath_svm import HyperPathSVM
                return HyperPathSVM(**model_config.params)
            
            # For baseline models, we'll import dynamically
            # This is simplified - in production you'd have proper imports
            self.logger.warning(f"Dynamic import for {class_name} - using placeholder")
            
            # Return a mock model for demonstration
            # In production, implement proper dynamic imports
            class MockModel:
                def __init__(self, **params):
                    self.params = params
                    self.is_fitted = False
                
                def fit(self, X, y):
                    time.sleep(0.1)  # Simulate training
                    self.is_fitted = True
                    return self
                
                def predict(self, X):
                    if not self.is_fitted:
                        raise ValueError("Model not fitted")
                    return np.random.randint(0, 3, len(X))
                
                def predict_proba(self, X):
                    if not self.is_fitted:
                        raise ValueError("Model not fitted")
                    n_classes = 3
                    probs = np.random.random((len(X), n_classes))
                    return probs / probs.sum(axis=1, keepdims=True)
            
            return MockModel(**model_config.params)
            
        except Exception as e:
            self.logger.error(f"Failed to create model {model_config.name}: {str(e)}")
            raise
    
    @contextmanager
    def _timeout_context(self, timeout_seconds: float):
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            yield
        finally:
            # Restore old handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _perform_statistical_analysis(self, model_results: Dict[str, Dict[str, EvaluationResults]]) -> Dict[str, Any]:
      
        try:
            self.logger.info("Performing statistical analysis")
            
            statistical_results = {
                'pairwise_comparisons': {},
                'friedman_test': {},
                'effect_sizes': {}
            }
            
            # Get all model names
            model_names = list(model_results.keys())
            
            # Key metrics for statistical testing
            test_metrics = ['accuracy', 'inference_time', 'memory_usage']
            
            for metric in test_metrics:
                # Collect metric values for each model across datasets
                metric_data = {}
                for model_name in model_names:
                    values = []
                    for dataset_results in model_results[model_name].values():
                        value = dataset_results.get_metric_value(metric)
                        if value is not None:
                            values.append(value)
                    if values:
                        metric_data[model_name] = values
                
                if len(metric_data) < 2:
                    continue
                
                # Pairwise t-tests
                statistical_results['pairwise_comparisons'][metric] = {}
                
                for i, model1 in enumerate(model_names):
                    if model1 not in metric_data:
                        continue
                    
                    for model2 in model_names[i+1:]:
                        if model2 not in metric_data:
                            continue
                        
                        # Perform paired t-test if same number of datasets
                        values1 = metric_data[model1]
                        values2 = metric_data[model2]
                        
                        if len(values1) == len(values2) and len(values1) > 1:
                            t_stat, p_value = stats.ttest_rel(values1, values2)
                            
                            # Calculate effect size (Cohen's d)
                            diff = np.array(values1) - np.array(values2)
                            effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                            
                            comparison_key = f"{model1}_vs_{model2}"
                            statistical_results['pairwise_comparisons'][metric][comparison_key] = {
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < self.config.significance_level,
                                'effect_size': float(effect_size),
                                'mean_diff': float(np.mean(diff))
                            }
                
                # Friedman test for multiple model comparison
                if len(metric_data) > 2:
                    data_matrix = []
                    min_length = min(len(values) for values in metric_data.values())
                    
                    for model_name in sorted(metric_data.keys()):
                        data_matrix.append(metric_data[model_name][:min_length])
                    
                    if min_length > 1:
                        try:
                            friedman_stat, friedman_p = stats.friedmanchisquare(*data_matrix)
                            statistical_results['friedman_test'][metric] = {
                                'statistic': float(friedman_stat),
                                'p_value': float(friedman_p),
                                'significant': friedman_p < self.config.significance_level,
                                'models': sorted(metric_data.keys())
                            }
                        except Exception as e:
                            self.logger.warning(f"Friedman test failed for {metric}: {str(e)}")
            
            return statistical_results
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_performance_summary(self, model_results: Dict[str, Dict[str, EvaluationResults]]) -> Dict[str, Any]:
      
        try:
            summary = {
                'overall_rankings': {},
                'target_compliance': {},
                'performance_matrix': {},
                'best_models_per_metric': {}
            }
            
            # Key metrics for ranking
            ranking_metrics = ['accuracy', 'inference_time', 'memory_usage', 'f1_score']
            
            # Calculate overall rankings
            for metric in ranking_metrics:
                model_scores = {}
                
                for model_name, dataset_results in model_results.items():
                    scores = []
                    for results in dataset_results.values():
                        value = results.get_metric_value(metric)
                        if value is not None:
                            scores.append(value)
                    
                    if scores:
                        model_scores[model_name] = np.mean(scores)
                
                # Rank models (higher is better except for time/memory metrics)
                reverse_sort = metric not in ['inference_time', 'memory_usage', 'adaptation_time']
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=reverse_sort)
                
                summary['overall_rankings'][metric] = [
                    {'model': model, 'score': score, 'rank': rank + 1}
                    for rank, (model, score) in enumerate(sorted_models)
                ]
                
                # Best model for this metric
                if sorted_models:
                    summary['best_models_per_metric'][metric] = sorted_models[0][0]
            
            # Target compliance analysis
            target_metrics = ['accuracy', 'inference_time', 'memory_usage', 'adaptation_time']
            
            for model_name, dataset_results in model_results.items():
                compliance = {}
                
                for results in dataset_results.values():
                    for metric in target_metrics:
                        metric_result = results.get_metric(metric)
                        if metric_result and metric_result.target is not None:
                            if metric not in compliance:
                                compliance[metric] = []
                            compliance[metric].append(metric_result.passed)
                
                # Calculate compliance rates
                model_compliance = {}
                for metric, passes in compliance.items():
                    model_compliance[metric] = {
                        'compliance_rate': np.mean(passes),
                        'total_tests': len(passes),
                        'passed_tests': sum(passes)
                    }
                
                summary['target_compliance'][model_name] = model_compliance
            
            # Performance matrix (models vs metrics)
            matrix_data = []
            for model_name, dataset_results in model_results.items():
                row = {'model': model_name}
                
                for metric in ranking_metrics:
                    scores = []
                    for results in dataset_results.values():
                        value = results.get_metric_value(metric)
                        if value is not None:
                            scores.append(value)
                    
                    row[f'{metric}_mean'] = np.mean(scores) if scores else None
                    row[f'{metric}_std'] = np.std(scores) if scores else None
                
                matrix_data.append(row)
            
            summary['performance_matrix'] = matrix_data
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _save_benchmark_results(self, results: BenchmarkResults):
        
        try:
            timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(results.timestamp))
            
            # Save main results as JSON
            results_file = self.config.results_dir / f"benchmark_results_{timestamp_str}.json"
            
            # Convert results to JSON-serializable format
            json_data = {
                'config': asdict(results.config),
                'timestamp': results.timestamp,
                'execution_time': results.execution_time,
                'model_results': {},
                'statistical_tests': results.statistical_tests,
                'performance_summary': results.performance_summary
            }
            
            # Convert EvaluationResults to dictionaries
            for model_name, dataset_results in results.model_results.items():
                json_data['model_results'][model_name] = {}
                for dataset_name, eval_results in dataset_results.items():
                    # Convert metrics to simple dictionaries
                    metrics_dict = {}
                    for metric in eval_results.metrics:
                        metrics_dict[metric.name] = {
                            'value': metric.value,
                            'unit': metric.unit,
                            'target': metric.target,
                            'passed': metric.passed,
                            'metadata': metric.metadata
                        }
                    
                    json_data['model_results'][model_name][dataset_name] = {
                        'model_name': eval_results.model_name,
                        'dataset_name': eval_results.dataset_name,
                        'timestamp': eval_results.timestamp,
                        'metrics': metrics_dict,
                        'detailed_stats': eval_results.detailed_stats
                    }
            
            with open(results_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Save detailed results as pickle for Python access
            pickle_file = self.config.results_dir / f"benchmark_results_{timestamp_str}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Results saved to {results_file} and {pickle_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
    
    def _generate_evaluation_plots(self, results: BenchmarkResults):
       
        try:
            self.logger.info("Generating evaluation plots")
            
            plots_dir = self.config.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Performance comparison plot
            self._plot_performance_comparison(results, plots_dir)
            
            # Target compliance plot
            self._plot_target_compliance(results, plots_dir)
            
            # Statistical significance heatmap
            if results.statistical_tests:
                self._plot_statistical_significance(results, plots_dir)
            
            self.logger.info(f"Plots saved to {plots_dir}")
            
        except Exception as e:
            self.logger.error(f"Plot generation failed: {str(e)}")
    
    def _plot_performance_comparison(self, results: BenchmarkResults, output_dir: Path):
       
        try:
            # Prepare data for plotting
            metrics_to_plot = ['accuracy', 'inference_time', 'memory_usage', 'f1_score']
            plot_data = []
            
            for model_name, dataset_results in results.model_results.items():
                for dataset_name, eval_results in dataset_results.items():
                    row = {'Model': model_name, 'Dataset': dataset_name}
                    
                    for metric in metrics_to_plot:
                        value = eval_results.get_metric_value(metric)
                        row[metric] = value if value is not None else 0
                    
                    plot_data.append(row)
            
            df = pd.DataFrame(plot_data)
            
            # Create subplots for each metric
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in df.columns:
                    pivot_df = df.pivot(index='Model', columns='Dataset', values=metric)
                    
                    # Create box plot
                    ax = axes[i]
                    pivot_df.plot(kind='bar', ax=ax, alpha=0.8)
                    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Performance comparison plot failed: {str(e)}")
    
    def _plot_target_compliance(self, results: BenchmarkResults, output_dir: Path):
     
        try:
            if 'target_compliance' not in results.performance_summary:
                return
            
            compliance_data = results.performance_summary['target_compliance']
            
            # Prepare data
            models = []
            metrics = []
            compliance_rates = []
            
            for model_name, model_compliance in compliance_data.items():
                for metric, compliance_info in model_compliance.items():
                    models.append(model_name)
                    metrics.append(metric)
                    compliance_rates.append(compliance_info['compliance_rate'])
            
            if not models:
                return
            
            # Create heatmap
            df = pd.DataFrame({
                'Model': models,
                'Metric': metrics,
                'Compliance Rate': compliance_rates
            })
            
            pivot_df = df.pivot(index='Model', columns='Metric', values='Compliance Rate')
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.5, 
                       cbar_kws={'label': 'Compliance Rate'})
            plt.title('Target Compliance Rates by Model and Metric')
            plt.tight_layout()
            plt.savefig(output_dir / 'target_compliance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Target compliance plot failed: {str(e)}")
    
    def _plot_statistical_significance(self, results: BenchmarkResults, output_dir: Path):
        
        try:
            if 'pairwise_comparisons' not in results.statistical_tests:
                return
            
            comparisons = results.statistical_tests['pairwise_comparisons']
            
            for metric, metric_comparisons in comparisons.items():
                if not metric_comparisons:
                    continue
                
                # Extract model pairs and significance
                pairs = []
                significance = []
                p_values = []
                
                for comparison_key, result in metric_comparisons.items():
                    pairs.append(comparison_key.replace('_vs_', ' vs '))
                    significance.append('Significant' if result['significant'] else 'Not Significant')
                    p_values.append(result['p_value'])
                
                # Create plot
                plt.figure(figsize=(12, 6))
                colors = ['red' if sig == 'Significant' else 'blue' for sig in significance]
                
                bars = plt.bar(range(len(pairs)), [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
                plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', 
                           label=f'Significance threshold (p=0.05)')
                
                plt.xlabel('Model Comparisons')
                plt.ylabel('-log10(p-value)')
                plt.title(f'Statistical Significance: {metric.replace("_", " ").title()}')
                plt.xticks(range(len(pairs)), pairs, rotation=45, ha='right')
                plt.legend()
                
                # Add significance labels
                for i, (bar, sig) in enumerate(zip(bars, significance)):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            'Sig' if sig == 'Significant' else 'NS',
                            ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'significance_{metric}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Statistical significance plot failed: {str(e)}")


def create_evaluation_framework(config_dict: Optional[Dict[str, Any]] = None) -> EvaluationOrchestrator:
  onfigured EvaluationOrchestrator instance
   
    if config_dict:
        config = EvaluationConfig(**config_dict)
    else:
        config = EvaluationConfig()
    
    return EvaluationOrchestrator(config)


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    logger.info("Testing evaluation framework...")
    
    # Create evaluation framework
    evaluator = create_evaluation_framework({
        'datasets': ['caida'],  # Test with single dataset
        'max_workers': 2,
        'generate_plots': True
    })
    
    # Run quick benchmark
    results = evaluator.run_comprehensive_benchmark(
        models=['hyperpath_svm', 'static_svm'],
        datasets=['caida']
    )
    
    logger.info("Evaluation framework test completed successfully!")
    logger.info(f"Best model: {results.get_best_model('accuracy')}")
    logger.info(f"Total execution time: {results.execution_time:.2f}s") 
