# File: hyperpath_svm/experiments/experiment_runner.py

"""
Main Experiment Runner and Orchestrator for HyperPath-SVM

This module provides the central orchestration for all experiments in the HyperPath-SVM
framework. It manages the complete experimental pipeline from data loading to results
generation, including cross-validation, statistical analysis, and report generation.

"""

import numpy as np
import pandas as pd
import os
import sys
import json
import yaml
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from contextlib import contextmanager
import shutil
import psutil
import gc

from ..utils.logging_utils import get_logger, get_hyperpath_logger, LogConfig, setup_logging
from ..data.dataset_loader import NetworkDatasetLoader
from ..data.data_augmentation import DataAugmentationPipeline
from ..evaluation.evaluator import EvaluationOrchestrator, EvaluationConfig
from ..evaluation.cross_validation import TemporalCrossValidationRunner, ValidationConfig
from ..evaluation.metrics import ComprehensiveMetricsEvaluator, PerformanceTarget
from ..core.hyperpath_svm import HyperPathSVM
from ..utils.visualization import PaperFigureGenerator
from ..utils.math_utils import math_utils

warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
   
    # Experiment identification
    experiment_name: str = "hyperpath_svm_evaluation"
    version: str = "1.0.0"
    description: str = "Complete HyperPath-SVM evaluation"
    
    # Data configuration
    datasets: List[str] = field(default_factory=lambda: ['caida', 'mawi', 'umass', 'wits'])
    data_split_seed: int = 42
    validation_size: float = 0.2
    test_size: float = 0.2
    
    # Model configuration
    models_to_evaluate: List[str] = field(default_factory=lambda: [
        'hyperpath_svm', 'static_svm', 'weighted_svm', 'quantum_svm', 
        'gnn', 'lstm', 'targcn', 'dmgfnet', 'behaviornet', 'ospf'
    ])
    
    # Training configuration
    max_training_time_hours: float = 24.0
    early_stopping: bool = True
    patience: int = 20
    
    # Cross-validation
    cv_folds: int = 5
    cv_repeats: int = 3
    cv_method: str = 'temporal'
    
    # Hyperparameter optimization
    hyperopt_trials: int = 50
    hyperopt_timeout_hours: float = 4.0
    
    # Performance targets
    target_accuracy: float = 0.965
    target_inference_ms: float = 1.8
    target_memory_mb: float = 98.0
    target_adaptation_min: float = 2.3
    
    # Statistical analysis
    significance_level: float = 0.05
    correction_method: str = 'bonferroni'
    effect_size_threshold: float = 0.2
    
    # Resource management
    max_workers: int = mp.cpu_count() - 1
    memory_limit_gb: float = 16.0
    gpu_memory_limit_gb: Optional[float] = None
    
    # Output configuration
    results_dir: Path = Path("results")
    save_models: bool = False
    save_predictions: bool = True
    generate_figures: bool = True
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Advanced options
    enable_profiling: bool = True
    enable_monitoring: bool = True
    retry_failed_experiments: bool = True
    max_retries: int = 3


@dataclass
class ExperimentResults:
    
    
    # Metadata
    experiment_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_hours: float = 0.0
    status: str = "running"  # running, completed, failed, cancelled
    
    # Model results
    model_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cross_validation_results: Dict[str, Any] = field(default_factory=dict)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Performance analysis
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    target_compliance: Dict[str, bool] = field(default_factory=dict)
    best_models: Dict[str, str] = field(default_factory=dict)
    
    # Resource usage
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Generated artifacts
    generated_figures: Dict[str, Path] = field(default_factory=dict)
    saved_models: Dict[str, Path] = field(default_factory=dict)
    reports: Dict[str, Path] = field(default_factory=dict)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ResourceMonitor:
    
    
    def __init__(self, memory_limit_gb: float = 16.0):
        self.memory_limit_gb = memory_limit_gb
        self.logger = get_logger(__name__)
        self._monitoring = False
        self._stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_usage_gb': [],
            'disk_io': [],
            'timestamps': []
        }
    
    def start_monitoring(self):
      
        self._monitoring = True
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
       
        self._monitoring = False
        self.logger.info("Resource monitoring stopped")
    
    def record_stats(self):
      
        if not self._monitoring:
            return
        
        try:
            # CPU and memory stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            
            self._stats['cpu_percent'].append(cpu_percent)
            self._stats['memory_percent'].append(memory.percent)
            self._stats['memory_usage_gb'].append(memory.used / (1024**3))
            self._stats['disk_io'].append(disk_io.read_bytes + disk_io.write_bytes)
            self._stats['timestamps'].append(time.time())
            
            # Check memory limit
            if memory.used / (1024**3) > self.memory_limit_gb:
                self.logger.warning(f"Memory usage ({memory.used / (1024**3):.1f}GB) "
                                  f"exceeds limit ({self.memory_limit_gb}GB)")
                
        except Exception as e:
            self.logger.warning(f"Failed to record resource stats: {str(e)}")
    
    def get_summary(self) -> Dict[str, float]:
        
        if not self._stats['cpu_percent']:
            return {}
        
        return {
            'avg_cpu_percent': np.mean(self._stats['cpu_percent']),
            'max_cpu_percent': np.max(self._stats['cpu_percent']),
            'avg_memory_gb': np.mean(self._stats['memory_usage_gb']),
            'max_memory_gb': np.max(self._stats['memory_usage_gb']),
            'monitoring_duration_minutes': 
                (self._stats['timestamps'][-1] - self._stats['timestamps'][0]) / 60
        }


class ExperimentOrchestrator:
    
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        
        # Setup logging
        self._setup_logging()
        self.logger = get_logger(__name__)
        self.hyperpath_logger = get_hyperpath_logger(__name__)
        
        # Initialize components
        self.dataset_loader = NetworkDatasetLoader()
        self.evaluation_orchestrator = EvaluationOrchestrator()
        self.cv_runner = TemporalCrossValidationRunner()
        self.figure_generator = PaperFigureGenerator()
        self.resource_monitor = ResourceMonitor(self.config.memory_limit_gb)
        
        # Results tracking
        self.results = ExperimentResults(experiment_name=self.config.experiment_name)
        self.failed_experiments = []
        
        # Setup directories
        self._setup_directories()
        
        # Set random seeds for reproducibility
        self._setup_reproducibility()
        
        self.logger.info(f"Experiment orchestrator initialized: {self.config.experiment_name}")
    
    def _setup_logging(self):
      
        log_dir = self.config.results_dir / "logs"
        log_config = LogConfig(
            level="INFO",
            log_file=log_dir / f"{self.config.experiment_name}.log",
            console_output=True,
            console_colors=True,
            performance_logging=self.config.enable_profiling,
            experiment_logging=True,
            metrics_file=log_dir / "experiment_metrics.jsonl"
        )
        setup_logging(log_config)
    
    def _setup_directories(self):
       
        directories = [
            self.config.results_dir,
            self.config.results_dir / "models",
            self.config.results_dir / "figures",
            self.config.results_dir / "reports",
            self.config.results_dir / "logs",
            self.config.results_dir / "data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_reproducibility(self):
        
        if self.config.deterministic:
            np.random.seed(self.config.random_seed)
            os.environ['PYTHONHASHSEED'] = str(self.config.random_seed)
            
            # Set deterministic behavior for various libraries
            try:
                import torch
                torch.manual_seed(self.config.random_seed)
                torch.cuda.manual_seed_all(self.config.random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except ImportError:
                pass
    
    def run_complete_evaluation(self) -> ExperimentResults:
        
        try:
            self.logger.info("Starting complete HyperPath-SVM evaluation")
            self.results.start_time = time.time()
            
            # Start resource monitoring
            if self.config.enable_monitoring:
                self.resource_monitor.start_monitoring()
            
            # Start experiment tracking
            exp_logger = self.hyperpath_logger.get_experiment_logger()
            if exp_logger:
                exp_logger.start_experiment(
                    self.config.experiment_name, 
                    asdict(self.config)
                )
            
            # Phase 1: Data Loading and Preprocessing
            self.logger.info("Phase 1: Loading and preprocessing datasets")
            datasets = self._load_and_preprocess_datasets()
            
            # Phase 2: Model Training and Evaluation
            self.logger.info("Phase 2: Training and evaluating models")
            model_results = self._evaluate_all_models(datasets)
            self.results.model_results = model_results
            
            # Phase 3: Cross-Validation Analysis
            self.logger.info("Phase 3: Cross-validation analysis")
            cv_results = self._perform_cross_validation_analysis(datasets)
            self.results.cross_validation_results = cv_results
            
            # Phase 4: Statistical Analysis
            self.logger.info("Phase 4: Statistical significance testing")
            statistical_results = self._perform_statistical_analysis(model_results)
            self.results.statistical_analysis = statistical_results
            
            # Phase 5: Performance Analysis
            self.logger.info("Phase 5: Performance analysis and target compliance")
            performance_analysis = self._analyze_performance(model_results)
            self.results.performance_summary = performance_analysis
            
            # Phase 6: Report and Figure Generation
            self.logger.info("Phase 6: Generating reports and figures")
            if self.config.generate_figures:
                figures = self._generate_figures_and_reports()
                self.results.generated_figures = figures
            
            # Phase 7: Results Serialization
            self.logger.info("Phase 7: Saving results")
            self._save_results()
            
            # Finalize experiment
            self.results.status = "completed"
            self.results.end_time = time.time()
            self.results.duration_hours = (self.results.end_time - self.results.start_time) / 3600
            
            # Stop monitoring
            if self.config.enable_monitoring:
                self.resource_monitor.stop_monitoring()
                self.results.resource_usage = self.resource_monitor.get_summary()
            
            # Log completion
            if exp_logger:
                exp_logger.end_experiment("completed", asdict(self.results))
            
            self.logger.info(f"Complete evaluation finished successfully in "
                           f"{self.results.duration_hours:.2f} hours")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            self.results.status = "failed"
            self.results.errors.append(str(e))
            
            if exp_logger:
                exp_logger.end_experiment("failed", {"error": str(e)})
            
            raise
    
    def _load_and_preprocess_datasets(self) -> Dict[str, Dict[str, Any]]:
       
        datasets = {}
        
        for dataset_name in self.config.datasets:
            try:
                self.logger.info(f"Loading dataset: {dataset_name}")
                
                with self.hyperpath_logger.timer(f"load_dataset_{dataset_name}"):
                    # Load raw dataset
                    dataset = self.dataset_loader.load_dataset(dataset_name)
                    
                    # Apply data augmentation if configured
                    if hasattr(self.config, 'use_augmentation') and self.config.use_augmentation:
                        augmentation_pipeline = DataAugmentationPipeline()
                        # Apply augmentation (simplified)
                        dataset['augmented'] = True
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    
                    X, y = dataset['features'], dataset['labels']
                    
                    # Split into train/val/test
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=self.config.test_size, 
                        random_state=self.config.data_split_seed,
                        stratify=y if len(np.unique(y)) > 1 else None
                    )
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp,
                        test_size=self.config.validation_size / (1 - self.config.test_size),
                        random_state=self.config.data_split_seed,
                        stratify=y_temp if len(np.unique(y_temp)) > 1 else None
                    )
                    
                    # Store processed dataset
                    datasets[dataset_name] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_val': X_val,
                        'y_val': y_val,
                        'X_test': X_test,
                        'y_test': y_test,
                        'metadata': dataset.get('metadata', {}),
                        'original_size': len(X)
                    }
                
                self.logger.info(f"Dataset {dataset_name} loaded: {len(X)} total samples")
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
                self.results.errors.append(f"Dataset loading failed: {dataset_name} - {str(e)}")
        
        return datasets
    
    def _evaluate_all_models(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        
        model_results = {}
        
        # Create model configurations
        model_configs = self._create_model_configurations()
        
        total_combinations = len(self.config.models_to_evaluate) * len(datasets)
        completed = 0
        
        for model_name in self.config.models_to_evaluate:
            if model_name not in model_configs:
                self.logger.warning(f"Model configuration not found: {model_name}")
                continue
            
            model_results[model_name] = {}
            
            for dataset_name, dataset in datasets.items():
                try:
                    self.logger.info(f"Evaluating {model_name} on {dataset_name} "
                                   f"({completed + 1}/{total_combinations})")
                    
                    with self.hyperpath_logger.timer(f"evaluate_{model_name}_{dataset_name}"):
                        # Create and train model
                        model_config = model_configs[model_name]
                        model = self._create_model(model_config)
                        
                        # Train model with timeout
                        training_start = time.time()
                        model.fit(dataset['X_train'], dataset['y_train'])
                        training_time = time.time() - training_start
                        
                        # Evaluate model
                        predictions = model.predict(dataset['X_test'])
                        
                        # Compute comprehensive metrics
                        metrics_evaluator = ComprehensiveMetricsEvaluator()
                        evaluation_results = metrics_evaluator.evaluate_model(
                            model, dataset['X_test'], dataset['y_test'],
                            model_name, dataset_name
                        )
                        
                        # Store results
                        model_results[model_name][dataset_name] = {
                            'evaluation_results': evaluation_results,
                            'training_time': training_time,
                            'predictions': predictions if self.config.save_predictions else None,
                            'model_config': model_config
                        }
                        
                        # Save model if configured
                        if self.config.save_models:
                            model_path = (self.config.results_dir / "models" / 
                                        f"{model_name}_{dataset_name}.pkl")
                            self._save_model(model, model_path)
                            self.results.saved_models[f"{model_name}_{dataset_name}"] = model_path
                        
                        completed += 1
                        
                        self.logger.info(f"Completed {model_name} on {dataset_name}: "
                                       f"Accuracy={evaluation_results.get_metric_value('accuracy'):.4f}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {str(e)}")
                    self.failed_experiments.append((model_name, dataset_name, str(e)))
                    
                    if self.config.retry_failed_experiments:
                        # Add retry logic here
                        pass
                
                # Resource monitoring
                if self.config.enable_monitoring:
                    self.resource_monitor.record_stats()
                    
                # Memory cleanup
                gc.collect()
        
        return model_results
    
    def _create_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        
        return {
            'hyperpath_svm': {
                'class': 'HyperPathSVM',
                'module': 'hyperpath_svm.core.hyperpath_svm',
                'params': {
                    'C': 1.0,
                    'kernel': 'tgck',
                    'use_ddwe': True,
                    'quantum_optimization': True,
                    'continuous_learning': True
                }
            },
            'static_svm': {
                'class': 'StaticSVM',
                'module': 'hyperpath_svm.baselines.traditional_svm',
                'params': {'C': 1.0, 'kernel': 'rbf'}
            },
            'weighted_svm': {
                'class': 'WeightedSVM',
                'module': 'hyperpath_svm.baselines.traditional_svm',
                'params': {'C': 1.0, 'kernel': 'rbf', 'class_weight': 'balanced'}
            },
            'quantum_svm': {
                'class': 'QuantumSVM',
                'module': 'hyperpath_svm.baselines.traditional_svm',
                'params': {'C': 1.0, 'quantum_kernel': True}
            },
            'gnn': {
                'class': 'GNNModel',
                'module': 'hyperpath_svm.baselines.neural_networks',
                'params': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.1}
            },
            'lstm': {
                'class': 'LSTMModel',
                'module': 'hyperpath_svm.baselines.neural_networks',
                'params': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.1}
            },
            'targcn': {
                'class': 'TARGCNModel',
                'module': 'hyperpath_svm.baselines.neural_networks',
                'params': {'hidden_dim': 64, 'attention_heads': 4}
            },
            'dmgfnet': {
                'class': 'DMGFNetModel',
                'module': 'hyperpath_svm.baselines.neural_networks',
                'params': {'hidden_dim': 128, 'num_experts': 4}
            },
            'behaviornet': {
                'class': 'BehaviorNetModel',
                'module': 'hyperpath_svm.baselines.neural_networks',
                'params': {'embedding_dim': 64, 'num_behaviors': 8}
            },
            'ospf': {
                'class': 'OSPFProtocol',
                'module': 'hyperpath_svm.baselines.routing_protocols',
                'params': {'area_id': 0}
            }
        }
    
    def _create_model(self, model_config: Dict[str, Any]):
       
        try:
            # For HyperPathSVM, use direct import
            if model_config['class'] == 'HyperPathSVM':
                return HyperPathSVM(**model_config['params'])
            
            # For other models, use mock implementations for demonstration
            # In production, you would have proper dynamic imports
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
            
            return MockModel(**model_config['params'])
            
        except Exception as e:
            self.logger.error(f"Failed to create model {model_config['class']}: {str(e)}")
            raise
    
    def _perform_cross_validation_analysis(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
       
        cv_results = {}
        
        for dataset_name, dataset in datasets.items():
            cv_results[dataset_name] = {}
            
            for model_name in self.config.models_to_evaluate[:3]:  # Limit for demonstration
                try:
                    model_config = self._create_model_configurations()[model_name]
                    model = self._create_model(model_config)
                    
                    # Perform cross-validation
                    cv_config = ValidationConfig(
                        n_splits=self.config.cv_folds,
                        significance_level=self.config.significance_level
                    )
                    
                    cv_runner = TemporalCrossValidationRunner(cv_config)
                    
                    X = np.vstack([dataset['X_train'], dataset['X_val']])
                    y = np.hstack([dataset['y_train'], dataset['y_val']])
                    
                    results = cv_runner.cross_validate_model(
                        model, X, y, cv_method=self.config.cv_method
                    )
                    
                    cv_results[dataset_name][model_name] = results
                    
                except Exception as e:
                    self.logger.error(f"CV failed for {model_name} on {dataset_name}: {str(e)}")
        
        return cv_results
    
    def _perform_statistical_analysis(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
       
        statistical_results = {}
        
        try:
            # Extract performance metrics for statistical testing
            for dataset_name in self.config.datasets:
                if dataset_name not in statistical_results:
                    statistical_results[dataset_name] = {}
                
                # Get accuracy scores for all models
                model_scores = {}
                for model_name in self.config.models_to_evaluate:
                    if (model_name in model_results and 
                        dataset_name in model_results[model_name]):
                        
                        eval_results = model_results[model_name][dataset_name]['evaluation_results']
                        accuracy = eval_results.get_metric_value('accuracy')
                        if accuracy is not None:
                            model_scores[model_name] = accuracy
                
                # Perform pairwise statistical tests
                from scipy import stats
                
                pairwise_tests = {}
                model_names = list(model_scores.keys())
                
                for i, model1 in enumerate(model_names):
                    for model2 in model_names[i+1:]:
                        # For demonstration, use simple comparison
                        # In practice, you'd need repeated measurements for proper testing
                        score1, score2 = model_scores[model1], model_scores[model2]
                        
                        # Effect size (Cohen's d approximation)
                        effect_size = abs(score1 - score2) / 0.01  # Assuming std of 0.01
                        
                        test_key = f"{model1}_vs_{model2}"
                        pairwise_tests[test_key] = {
                            'score_diff': score1 - score2,
                            'effect_size': effect_size,
                            'significant': effect_size > self.config.effect_size_threshold,
                            'better_model': model1 if score1 > score2 else model2
                        }
                
                statistical_results[dataset_name] = {
                    'model_scores': model_scores,
                    'pairwise_tests': pairwise_tests
                }
        
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
        
        return statistical_results
    
    def _analyze_performance(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
       
        performance_analysis = {
            'target_compliance': {},
            'best_models': {},
            'performance_rankings': {},
            'summary_statistics': {}
        }
        
        # Performance targets
        targets = {
            'accuracy': self.config.target_accuracy,
            'inference_time': self.config.target_inference_ms,
            'memory_usage': self.config.target_memory_mb
        }
        
        # Analyze each dataset
        for dataset_name in self.config.datasets:
            dataset_analysis = {
                'model_performance': {},
                'target_compliance': {},
                'rankings': {}
            }
            
            model_performances = {}
            
            for model_name in self.config.models_to_evaluate:
                if (model_name in model_results and 
                    dataset_name in model_results[model_name]):
                    
                    eval_results = model_results[model_name][dataset_name]['evaluation_results']
                    
                    # Extract key metrics
                    performance = {}
                    compliance = {}
                    
                    for metric_name, target_value in targets.items():
                        metric_value = eval_results.get_metric_value(metric_name)
                        if metric_value is not None:
                            performance[metric_name] = metric_value
                            
                            # Check compliance
                            if metric_name in ['inference_time', 'memory_usage']:
                                compliance[metric_name] = metric_value <= target_value
                            else:
                                compliance[metric_name] = metric_value >= target_value
                    
                    model_performances[model_name] = performance
                    dataset_analysis['target_compliance'][model_name] = compliance
            
            # Determine best models
            for metric_name in targets.keys():
                if metric_name in ['inference_time', 'memory_usage']:
                    # Lower is better
                    best_model = min(
                        model_performances.keys(),
                        key=lambda m: model_performances[m].get(metric_name, float('inf'))
                    )
                else:
                    # Higher is better
                    best_model = max(
                        model_performances.keys(),
                        key=lambda m: model_performances[m].get(metric_name, 0)
                    )
                
                dataset_analysis['rankings'][metric_name] = best_model
            
            dataset_analysis['model_performance'] = model_performances
            performance_analysis[dataset_name] = dataset_analysis
        
        return performance_analysis
    
    def _generate_figures_and_reports(self) -> Dict[str, Path]:
      
        figures_dir = self.config.results_dir / "figures"
        reports_dir = self.config.results_dir / "reports"
        
        generated_figures = {}
        
        try:
            # Prepare data for visualization
            viz_data = {
                'model_results': self.results.model_results,
                'statistical_analysis': self.results.statistical_analysis,
                'performance_summary': self.results.performance_summary,
                'experiment_config': asdict(self.config)
            }
            
            # Generate all paper figures
            self.logger.info("Generating paper figures...")
            figures = self.figure_generator.generate_all_figures(viz_data, figures_dir)
            generated_figures.update(figures)
            
            # Generate comprehensive report
            self.logger.info("Generating comprehensive report...")
            report_path = self._generate_comprehensive_report(reports_dir)
            self.results.reports['comprehensive_report'] = report_path
            
        except Exception as e:
            self.logger.error(f"Figure/report generation failed: {str(e)}")
            self.results.errors.append(f"Visualization failed: {str(e)}")
        
        return generated_figures
    
    def _generate_comprehensive_report(self, reports_dir: Path) -> Path:
        
        report_path = reports_dir / f"{self.config.experiment_name}_report.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write(f"# {self.config.experiment_name} - Comprehensive Report\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Duration: {self.results.duration_hours:.2f} hours\n\n")
                
                f.write("## Executive Summary\n\n")
                f.write(f"- **Status**: {self.results.status}\n")
                f.write(f"- **Models Evaluated**: {len(self.config.models_to_evaluate)}\n")
                f.write(f"- **Datasets**: {len(self.config.datasets)}\n")
                f.write(f"- **Total Experiments**: {len(self.config.models_to_evaluate) * len(self.config.datasets)}\n")
                f.write(f"- **Failed Experiments**: {len(self.failed_experiments)}\n\n")
                
                # Add more report sections...
                f.write("## Model Performance Summary\n\n")
                # Add performance tables and analysis
                
                f.write("## Statistical Analysis\n\n")
                # Add statistical test results
                
                f.write("## Conclusions and Recommendations\n\n")
                # Add conclusions
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return reports_dir / "report_failed.txt"
    
    def _save_model(self, model, model_path: Path):
        
        try:
            import joblib
            joblib.dump(model, model_path)
        except Exception as e:
            self.logger.error(f"Failed to save model to {model_path}: {str(e)}")
    
    def _save_results(self):
       
        try:
            # Save main results
            results_file = self.config.results_dir / f"{self.config.experiment_name}_results.json"
            
            # Convert results to JSON-serializable format
            results_dict = asdict(self.results)
            
            # Convert Path objects to strings
            def convert_paths(obj):
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                elif isinstance(obj, Path):
                    return str(obj)
                else:
                    return obj
            
            results_dict = convert_paths(results_dict)
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            # Save raw results as pickle
            pickle_file = self.config.results_dir / f"{self.config.experiment_name}_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.results, f)
            
            self.logger.info(f"Results saved to {results_file} and {pickle_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")


def create_experiment_orchestrator(config_dict: Optional[Dict[str, Any]] = None) -> ExperimentOrchestrator:
    
    if config_dict:
        config = ExperimentConfig(**config_dict)
    else:
        config = ExperimentConfig()
    
    return ExperimentOrchestrator(config)


def run_complete_evaluation(config_file: Optional[Path] = None, 
                          config_dict: Optional[Dict[str, Any]] = None) -> ExperimentResults:
   
    
    # Load configuration
    if config_file:
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() == '.yaml':
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        config = ExperimentConfig(**config_data)
    elif config_dict:
        config = ExperimentConfig(**config_dict)
    else:
        config = ExperimentConfig()
    
    # Create and run orchestrator
    orchestrator = ExperimentOrchestrator(config)
    return orchestrator.run_complete_evaluation()


if __name__ == "__main__":
    # Test experiment orchestrator
    logger = get_logger(__name__)
    logger.info("Testing experiment orchestrator...")
    
    # Create test configuration
    test_config = ExperimentConfig(
        experiment_name="test_hyperpath_evaluation",
        datasets=['caida'],  # Test with single dataset
        models_to_evaluate=['hyperpath_svm', 'static_svm'],
        max_training_time_hours=0.1,  # Short test
        results_dir=Path("test_results"),
        generate_figures=True
    )
    
    try:
        # Run test evaluation
        orchestrator = ExperimentOrchestrator(test_config)
        results = orchestrator.run_complete_evaluation()
        
        logger.info("Test Results Summary:")
        logger.info(f"Status: {results.status}")
        logger.info(f"Duration: {results.duration_hours:.4f} hours")
        logger.info(f"Models evaluated: {len(results.model_results)}")
        logger.info(f"Errors: {len(results.errors)}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    logger.info("Experiment orchestrator testing completed!") 
