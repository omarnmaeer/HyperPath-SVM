# File: hyperpath_svm/experiments/ablation_study.py

"""
Comprehensive Ablation Study Framework for HyperPath-SVM

This module provides systematic ablation analysis to understand the contribution
of each component in the HyperPath-SVM framework. It evaluates different
combinations of components to identify which are most critical for performance.


Ablation Strategies:
- Individual component removal (leave-one-out)
- Progressive component addition (build-up)
- Component interaction analysis
- Sensitivity analysis for hyperparameters
- Performance degradation quantification
"""

import numpy as np
import pandas as pd
import itertools
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..utils.logging_utils import get_logger, get_hyperpath_logger
from ..core.hyperpath_svm import HyperPathSVM
from ..evaluation.metrics import ComprehensiveMetricsEvaluator, EvaluationResults
from ..evaluation.cross_validation import TemporalCrossValidationRunner
from ..data.dataset_loader import NetworkDatasetLoader
from ..utils.visualization import PaperFigureGenerator


@dataclass
class ComponentConfig:
   
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    critical: bool = False  # Whether component is considered critical


@dataclass
class AblationExperiment:
   
    experiment_id: str
    components: Dict[str, bool]  # component_name -> enabled
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    memory_usage: float = 0.0
    convergence_iterations: int = 0
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None


@dataclass
class AblationResults:
    
    study_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Experiment results
    experiments: Dict[str, AblationExperiment] = field(default_factory=dict)
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    
    # Analysis results
    component_importance: Dict[str, float] = field(default_factory=dict)
    component_interactions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_degradation: Dict[str, float] = field(default_factory=dict)
    
    # Statistical analysis
    significance_tests: Dict[str, Any] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    
    # Summary statistics
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    
    def get_completion_rate(self) -> float:
        """Get experiment completion rate."""
        return self.successful_experiments / max(self.total_experiments, 1)


class AblationStudyOrchestrator:
   
    
    def __init__(self, output_dir: Path = Path("ablation_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(__name__)
        self.hyperpath_logger = get_hyperpath_logger(__name__)
        
        # Initialize evaluation components
        self.metrics_evaluator = ComprehensiveMetricsEvaluator()
        self.cv_runner = TemporalCrossValidationRunner()
        self.dataset_loader = NetworkDatasetLoader()
        
        # Define HyperPath-SVM components for ablation
        self.components = self._define_components()
        
        self.logger.info("Ablation study orchestrator initialized")
    
    def _define_components(self) -> Dict[str, ComponentConfig]:
      
        return {
            'ddwe': ComponentConfig(
                name="Dynamic Discriminative Weight Evolution",
                enabled=True,
                params={'learning_rate': 0.01, 'decay_rate': 0.95},
                description="Adaptive weight evolution mechanism",
                critical=True
            ),
            'tgck': ComponentConfig(
                name="Temporal Graph Convolution Kernel",
                enabled=True,
                params={'temporal_window': 10, 'graph_depth': 3},
                description="Temporal-aware kernel for graph data",
                critical=True
            ),
            'quantum_optimization': ComponentConfig(
                name="Quantum-Inspired Optimization",
                enabled=True,
                params={'num_qubits': 8, 'optimization_rounds': 10},
                description="Quantum optimization for parameter tuning",
                critical=False
            ),
            'memory_hierarchy': ComponentConfig(
                name="Support Vector Memory Hierarchy",
                enabled=True,
                params={'levels': 3, 'compression_ratio': 0.1},
                description="Hierarchical memory management for SVs",
                critical=False
            ),
            'continuous_learning': ComponentConfig(
                name="Continuous Learning Mechanism",
                enabled=True,
                params={'adaptation_rate': 0.1, 'forgetting_factor': 0.9},
                description="Online learning and adaptation",
                critical=False
            ),
            'feature_engineering': ComponentConfig(
                name="Advanced Feature Engineering",
                enabled=True,
                params={'polynomial_features': True, 'interaction_features': True},
                description="Automated feature engineering pipeline",
                critical=False
            ),
            'topology_awareness': ComponentConfig(
                name="Network Topology Awareness",
                enabled=True,
                params={'spectral_features': True, 'centrality_features': True},
                description="Network structure-aware features",
                critical=False
            ),
            'regularization': ComponentConfig(
                name="Advanced Regularization",
                enabled=True,
                params={'l1_ratio': 0.5, 'adaptive_penalty': True},
                description="Sophisticated regularization techniques",
                critical=False
            )
        }
    
    def run_comprehensive_ablation_study(self, datasets: List[str] = None,
                                       study_types: List[str] = None) -> AblationResults:
  
        try:
            self.logger.info("Starting comprehensive ablation study")
            
            # Default parameters
            if datasets is None:
                datasets = ['caida', 'mawi']  # Subset for efficiency
            
            if study_types is None:
                study_types = ['leave_one_out', 'build_up', 'interaction_analysis']
            
            # Initialize results
            results = AblationResults(
                study_name=f"hyperpath_svm_ablation_{int(time.time())}"
            )
            
            # Load datasets
            self.logger.info("Loading datasets for ablation study")
            dataset_data = self._load_datasets(datasets)
            
            # Establish baseline performance
            self.logger.info("Establishing baseline performance")
            baseline_performance = self._establish_baseline(dataset_data)
            results.baseline_performance = baseline_performance
            
            # Run different types of ablation studies
            all_experiments = []
            
            for study_type in study_types:
                self.logger.info(f"Running {study_type} ablation study")
                
                if study_type == 'leave_one_out':
                    experiments = self._generate_leave_one_out_experiments()
                elif study_type == 'build_up':
                    experiments = self._generate_build_up_experiments()
                elif study_type == 'interaction_analysis':
                    experiments = self._generate_interaction_experiments()
                elif study_type == 'sensitivity_analysis':
                    experiments = self._generate_sensitivity_experiments()
                else:
                    self.logger.warning(f"Unknown study type: {study_type}")
                    continue
                
                all_experiments.extend(experiments)
            
            # Execute all experiments
            results.total_experiments = len(all_experiments)
            self.logger.info(f"Executing {len(all_experiments)} ablation experiments")
            
            experiment_results = self._execute_experiments_parallel(
                all_experiments, dataset_data
            )
            
            # Store experiment results
            for exp_id, exp_result in experiment_results.items():
                results.experiments[exp_id] = exp_result
                if exp_result.status == "completed":
                    results.successful_experiments += 1
                else:
                    results.failed_experiments += 1
            
            # Analyze results
            self.logger.info("Analyzing ablation study results")
            self._analyze_component_importance(results)
            self._analyze_component_interactions(results)
            self._perform_statistical_analysis(results)
            
            # Generate visualizations and reports
            self.logger.info("Generating ablation study reports")
            self._generate_ablation_visualizations(results)
            self._generate_ablation_report(results)
            
            # Save results
            self._save_ablation_results(results)
            
            results.end_time = time.time()
            duration_hours = (results.end_time - results.start_time) / 3600
            
            self.logger.info(f"Ablation study completed in {duration_hours:.2f} hours")
            self.logger.info(f"Success rate: {results.get_completion_rate():.1%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ablation study failed: {str(e)}")
            raise
    
    def _load_datasets(self, dataset_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load and prepare datasets for ablation study."""
        datasets = {}
        
        for dataset_name in dataset_names:
            try:
                self.logger.info(f"Loading dataset: {dataset_name}")
                dataset = self.dataset_loader.load_dataset(dataset_name)
                
                # Split data for consistent evaluation
                from sklearn.model_selection import train_test_split
                
                X, y = dataset['features'], dataset['labels']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42,
                    stratify=y if len(np.unique(y)) > 1 else None
                )
                
                # Further split training into train/validation
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42,
                    stratify=y_train if len(np.unique(y_train)) > 1 else None
                )
                
                datasets[dataset_name] = {
                    'X_train': X_train_split,
                    'y_train': y_train_split,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'metadata': dataset.get('metadata', {})
                }
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
        
        return datasets
    
    def _establish_baseline(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
       
        baseline_performance = {}
        
        for dataset_name, dataset in datasets.items():
            try:
                # Create full HyperPath-SVM model
                model = HyperPathSVM(
                    C=1.0,
                    kernel='tgck',
                    use_ddwe=True,
                    quantum_optimization=True,
                    continuous_learning=True
                )
                
                # Train and evaluate
                model.fit(dataset['X_train'], dataset['y_train'])
                
                # Evaluate on validation set
                predictions = model.predict(dataset['X_val'])
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(dataset['y_val'], predictions)
                
                baseline_performance[dataset_name] = accuracy
                
                self.logger.info(f"Baseline performance on {dataset_name}: {accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to establish baseline for {dataset_name}: {str(e)}")
                baseline_performance[dataset_name] = 0.0
        
        return baseline_performance
    
    def _generate_leave_one_out_experiments(self) -> List[AblationExperiment]:
      
        experiments = []
        
        for component_name in self.components.keys():
            # Create experiment with one component disabled
            component_config = {name: True for name in self.components.keys()}
            component_config[component_name] = False
            
            experiment = AblationExperiment(
                experiment_id=f"leave_out_{component_name}",
                components=component_config
            )
            
            experiments.append(experiment)
        
        self.logger.info(f"Generated {len(experiments)} leave-one-out experiments")
        return experiments
    
    def _generate_build_up_experiments(self) -> List[AblationExperiment]:
      
        experiments = []
        
        # Start with no components and progressively add them
        component_names = list(self.components.keys())
        
        for i in range(1, len(component_names) + 1):
            # All combinations of i components
            for component_subset in itertools.combinations(component_names, i):
                component_config = {name: name in component_subset 
                                  for name in component_names}
                
                experiment = AblationExperiment(
                    experiment_id=f"build_up_{i}_{hash(tuple(sorted(component_subset))) % 10000}",
                    components=component_config
                )
                
                experiments.append(experiment)
        
        self.logger.info(f"Generated {len(experiments)} build-up experiments")
        return experiments
    
    def _generate_interaction_experiments(self) -> List[AblationExperiment]:
        
        experiments = []
        
        # Focus on critical components and their interactions
        critical_components = [name for name, config in self.components.items() 
                             if config.critical]
        
        # Pairwise interactions
        for comp1, comp2 in itertools.combinations(critical_components, 2):
            # Both enabled
            config_both = {name: name in [comp1, comp2] for name in self.components.keys()}
            experiments.append(AblationExperiment(
                experiment_id=f"interaction_{comp1}_{comp2}_both",
                components=config_both
            ))
            
            # Only first enabled
            config_first = {name: name == comp1 for name in self.components.keys()}
            experiments.append(AblationExperiment(
                experiment_id=f"interaction_{comp1}_{comp2}_first",
                components=config_first
            ))
            
            # Only second enabled
            config_second = {name: name == comp2 for name in self.components.keys()}
            experiments.append(AblationExperiment(
                experiment_id=f"interaction_{comp1}_{comp2}_second",
                components=config_second
            ))
        
        self.logger.info(f"Generated {len(experiments)} interaction experiments")
        return experiments
    
    def _generate_sensitivity_experiments(self) -> List[AblationExperiment]:
       
        experiments = []
        
        # Test different parameter settings for key components
        sensitivity_configs = {
            'ddwe': [
                {'learning_rate': 0.001, 'decay_rate': 0.9},
                {'learning_rate': 0.01, 'decay_rate': 0.95},
                {'learning_rate': 0.1, 'decay_rate': 0.99}
            ],
            'tgck': [
                {'temporal_window': 5, 'graph_depth': 2},
                {'temporal_window': 10, 'graph_depth': 3},
                {'temporal_window': 20, 'graph_depth': 4}
            ]
        }
        
        for component_name, param_configs in sensitivity_configs.items():
            for i, param_config in enumerate(param_configs):
                # Enable all components but vary parameters
                component_config = {name: True for name in self.components.keys()}
                
                experiment = AblationExperiment(
                    experiment_id=f"sensitivity_{component_name}_{i}",
                    components=component_config
                )
                
                experiments.append(experiment)
        
        self.logger.info(f"Generated {len(experiments)} sensitivity experiments")
        return experiments
    
    def _execute_experiments_parallel(self, experiments: List[AblationExperiment],
                                    datasets: Dict[str, Dict[str, Any]]) -> Dict[str, AblationExperiment]:
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all experiments
            future_to_experiment = {
                executor.submit(self._execute_single_experiment, experiment, datasets): experiment
                for experiment in experiments
            }
            
            # Collect results
            for future in as_completed(future_to_experiment):
                experiment = future_to_experiment[future]
                try:
                    completed_experiment = future.result()
                    results[experiment.experiment_id] = completed_experiment
                    
                    if completed_experiment.status == "completed":
                        self.logger.debug(f"Completed experiment: {experiment.experiment_id}")
                    else:
                        self.logger.warning(f"Failed experiment: {experiment.experiment_id}")
                        
                except Exception as e:
                    self.logger.error(f"Experiment {experiment.experiment_id} failed: {str(e)}")
                    experiment.status = "failed"
                    experiment.error_message = str(e)
                    results[experiment.experiment_id] = experiment
        
        return results
    
    def _execute_single_experiment(self, experiment: AblationExperiment,
                                 datasets: Dict[str, Dict[str, Any]]) -> AblationExperiment:
       
        try:
            experiment.status = "running"
            start_time = time.time()
            
            # Average performance across datasets
            dataset_performances = []
            
            for dataset_name, dataset in datasets.items():
                # Create model with specified component configuration
                model = self._create_ablated_model(experiment.components)
                
                # Train model
                model.fit(dataset['X_train'], dataset['y_train'])
                
                # Evaluate
                predictions = model.predict(dataset['X_val'])
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(dataset['y_val'], predictions)
                
                dataset_performances.append(accuracy)
            
            # Store results
            experiment.performance_metrics['accuracy'] = np.mean(dataset_performances)
            experiment.performance_metrics['accuracy_std'] = np.std(dataset_performances)
            experiment.training_time = time.time() - start_time
            experiment.status = "completed"
            
        except Exception as e:
            experiment.status = "failed"
            experiment.error_message = str(e)
        
        return experiment
    
    def _create_ablated_model(self, component_config: Dict[str, bool]) -> HyperPathSVM:
       
        # Create model with components enabled/disabled based on configuration
        model_params = {
            'C': 1.0,
            'kernel': 'tgck' if component_config.get('tgck', False) else 'rbf',
            'use_ddwe': component_config.get('ddwe', False),
            'quantum_optimization': component_config.get('quantum_optimization', False),
            'continuous_learning': component_config.get('continuous_learning', False),
            'memory_hierarchy': component_config.get('memory_hierarchy', False),
            'feature_engineering': component_config.get('feature_engineering', False),
            'topology_awareness': component_config.get('topology_awareness', False),
            'advanced_regularization': component_config.get('regularization', False)
        }
        
        return HyperPathSVM(**model_params)
    
    def _analyze_component_importance(self, results: AblationResults):
      
        # Calculate performance drop when each component is removed
        baseline_acc = np.mean(list(results.baseline_performance.values()))
        
        for component_name in self.components.keys():
            experiment_id = f"leave_out_{component_name}"
            
            if experiment_id in results.experiments:
                experiment = results.experiments[experiment_id]
                if experiment.status == "completed":
                    ablated_acc = experiment.performance_metrics.get('accuracy', 0)
                    importance = baseline_acc - ablated_acc
                    results.component_importance[component_name] = importance
                else:
                    results.component_importance[component_name] = 0.0
        
        # Sort components by importance
        sorted_importance = sorted(results.component_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        self.logger.info("Component Importance Ranking:")
        for component_name, importance in sorted_importance:
            self.logger.info(f"  {component_name}: {importance:.4f}")
    
    def _analyze_component_interactions(self, results: AblationResults):
     
        # Find interaction experiments and analyze synergies
        interaction_experiments = {
            exp_id: exp for exp_id, exp in results.experiments.items()
            if 'interaction' in exp_id and exp.status == "completed"
        }
        
        # Group by component pairs
        component_pairs = defaultdict(dict)
        
        for exp_id, experiment in interaction_experiments.items():
            if 'both' in exp_id:
                pair_key = exp_id.replace('interaction_', '').replace('_both', '')
                component_pairs[pair_key]['both'] = experiment.performance_metrics.get('accuracy', 0)
            elif 'first' in exp_id:
                pair_key = exp_id.replace('interaction_', '').replace('_first', '')
                component_pairs[pair_key]['first'] = experiment.performance_metrics.get('accuracy', 0)
            elif 'second' in exp_id:
                pair_key = exp_id.replace('interaction_', '').replace('_second', '')
                component_pairs[pair_key]['second'] = experiment.performance_metrics.get('accuracy', 0)
        
        # Calculate interaction effects
        for pair_key, performances in component_pairs.items():
            if all(key in performances for key in ['both', 'first', 'second']):
                # Interaction effect = P(both) - P(first) - P(second) + baseline_individual
                both_perf = performances['both']
                first_perf = performances['first']
                second_perf = performances['second']
                
                # Simplified interaction effect
                interaction_effect = both_perf - max(first_perf, second_perf)
                results.component_interactions[pair_key] = {'interaction_effect': interaction_effect}
    
    def _perform_statistical_analysis(self, results: AblationResults):
      
        # Collect performance values for statistical testing
        successful_experiments = [
            exp for exp in results.experiments.values()
            if exp.status == "completed" and 'accuracy' in exp.performance_metrics
        ]
        
        if len(successful_experiments) < 2:
            return
        
        # Test significance of component importance
        baseline_acc = np.mean(list(results.baseline_performance.values()))
        
        for component_name in self.components.keys():
            experiment_id = f"leave_out_{component_name}"
            
            if experiment_id in results.experiments:
                experiment = results.experiments[experiment_id]
                if experiment.status == "completed":
                    ablated_acc = experiment.performance_metrics.get('accuracy', 0)
                    
                    # Simple effect size calculation
                    effect_size = abs(baseline_acc - ablated_acc) / 0.01  # Assuming std of 0.01
                    results.effect_sizes[component_name] = effect_size
                    
                    # Significance test (simplified)
                    is_significant = effect_size > 0.2  # Cohen's d threshold
                    results.significance_tests[component_name] = {
                        'significant': is_significant,
                        'effect_size': effect_size,
                        'performance_drop': baseline_acc - ablated_acc
                    }
    
    def _generate_ablation_visualizations(self, results: AblationResults):
      
        try:
            figures_dir = self.output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            # Component importance plot
            if results.component_importance:
                self._plot_component_importance(results, figures_dir)
            
            # Performance degradation plot
            self._plot_performance_degradation(results, figures_dir)
            
            # Component interaction heatmap
            if results.component_interactions:
                self._plot_component_interactions(results, figures_dir)
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {str(e)}")
    
    def _plot_component_importance(self, results: AblationResults, output_dir: Path):
        """Plot component importance rankings."""
        if not results.component_importance:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        components = list(results.component_importance.keys())
        importances = list(results.component_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_components = [components[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        bars = ax.bar(range(len(sorted_components)), sorted_importances, 
                     color='steelblue', alpha=0.8)
        
        ax.set_xlabel('Components')
        ax.set_ylabel('Performance Drop (Importance)')
        ax.set_title('Component Importance in HyperPath-SVM')
        ax.set_xticks(range(len(sorted_components)))
        ax.set_xticklabels([comp.replace('_', '\n') for comp in sorted_components], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, importance in zip(bars, sorted_importances):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                   f'{importance:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'component_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_degradation(self, results: AblationResults, output_dir: Path):
        """Plot performance degradation analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Performance by number of components
        build_up_experiments = [
            exp for exp_id, exp in results.experiments.items()
            if 'build_up' in exp_id and exp.status == "completed"
        ]
        
        if build_up_experiments:
            num_components = []
            performances = []
            
            for exp in build_up_experiments:
                num_enabled = sum(exp.components.values())
                accuracy = exp.performance_metrics.get('accuracy', 0)
                num_components.append(num_enabled)
                performances.append(accuracy)
            
            # Group by number of components
            component_counts = sorted(set(num_components))
            avg_performances = []
            std_performances = []
            
            for count in component_counts:
                perfs = [performances[i] for i, nc in enumerate(num_components) if nc == count]
                avg_performances.append(np.mean(perfs))
                std_performances.append(np.std(perfs))
            
            ax1.errorbar(component_counts, avg_performances, yerr=std_performances,
                        marker='o', linewidth=2, markersize=8, capsize=5)
            ax1.set_xlabel('Number of Components Enabled')
            ax1.set_ylabel('Average Performance')
            ax1.set_title('Performance vs Number of Components')
            ax1.grid(True, alpha=0.3)
        
        # Component synergy analysis
        if results.component_interactions:
            interactions = list(results.component_interactions.keys())
            effects = [data.get('interaction_effect', 0) 
                      for data in results.component_interactions.values()]
            
            bars = ax2.bar(range(len(interactions)), effects, 
                          color=['green' if e > 0 else 'red' for e in effects], 
                          alpha=0.7)
            
            ax2.set_xlabel('Component Pairs')
            ax2.set_ylabel('Interaction Effect')
            ax2.set_title('Component Interaction Effects')
            ax2.set_xticks(range(len(interactions)))
            ax2.set_xticklabels([i.replace('_', ' vs ') for i in interactions], 
                               rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_degradation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_interactions(self, results: AblationResults, output_dir: Path):
        """Plot component interaction heatmap."""
        if not results.component_interactions:
            return
        
        # Create interaction matrix
        components = list(self.components.keys())
        interaction_matrix = np.zeros((len(components), len(components)))
        
        for pair_key, interaction_data in results.component_interactions.items():
            if '_' in pair_key:
                comp1, comp2 = pair_key.split('_', 1)
                if comp1 in components and comp2 in components:
                    i, j = components.index(comp1), components.index(comp2)
                    effect = interaction_data.get('interaction_effect', 0)
                    interaction_matrix[i, j] = effect
                    interaction_matrix[j, i] = effect  # Symmetric
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        im = ax.imshow(interaction_matrix, cmap='RdBu_r', aspect='equal')
        
        # Set ticks and labels
        ax.set_xticks(range(len(components)))
        ax.set_yticks(range(len(components)))
        ax.set_xticklabels([comp.replace('_', '\n') for comp in components], 
                          rotation=45, ha='right')
        ax.set_yticklabels([comp.replace('_', '\n') for comp in components])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Interaction Effect')
        
        # Add text annotations
        for i in range(len(components)):
            for j in range(len(components)):
                if i != j and abs(interaction_matrix[i, j]) > 1e-4:
                    ax.text(j, i, f'{interaction_matrix[i, j]:.3f}',
                           ha='center', va='center', fontsize=8)
        
        ax.set_title('Component Interaction Effects Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / 'component_interactions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_ablation_report(self, results: AblationResults):
      
        report_path = self.output_dir / "ablation_report.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write(f"# HyperPath-SVM Ablation Study Report\n\n")
                f.write(f"**Study Name:** {results.study_name}\n")
                f.write(f"**Completion Rate:** {results.get_completion_rate():.1%}\n")
                f.write(f"**Total Experiments:** {results.total_experiments}\n")
                f.write(f"**Successful:** {results.successful_experiments}\n")
                f.write(f"**Failed:** {results.failed_experiments}\n\n")
                
                # Component importance section
                f.write("## Component Importance Rankings\n\n")
                if results.component_importance:
                    sorted_importance = sorted(results.component_importance.items(), 
                                             key=lambda x: x[1], reverse=True)
                    f.write("| Rank | Component | Performance Drop | Description |\n")
                    f.write("|------|-----------|------------------|-------------|\n")
                    
                    for rank, (comp_name, importance) in enumerate(sorted_importance, 1):
                        description = self.components[comp_name].description
                        f.write(f"| {rank} | {comp_name.replace('_', ' ').title()} | "
                               f"{importance:.4f} | {description} |\n")
                
                # Statistical significance
                f.write("\n## Statistical Significance\n\n")
                if results.significance_tests:
                    f.write("| Component | Significant | Effect Size | Performance Drop |\n")
                    f.write("|-----------|-------------|-------------|------------------|\n")
                    
                    for comp_name, test_result in results.significance_tests.items():
                        f.write(f"| {comp_name.replace('_', ' ').title()} | "
                               f"{'✓' if test_result['significant'] else '✗'} | "
                               f"{test_result['effect_size']:.3f} | "
                               f"{test_result['performance_drop']:.4f} |\n")
                
                # Key findings
                f.write("\n## Key Findings\n\n")
                
                if results.component_importance:
                    most_important = max(results.component_importance.items(), key=lambda x: x[1])
                    least_important = min(results.component_importance.items(), key=lambda x: x[1])
                    
                    f.write(f"- **Most Critical Component:** {most_important[0].replace('_', ' ').title()} "
                           f"(performance drop: {most_important[1]:.4f})\n")
                    f.write(f"- **Least Critical Component:** {least_important[0].replace('_', ' ').title()} "
                           f"(performance drop: {least_important[1]:.4f})\n")
                
                # Recommendations
                f.write("\n## Recommendations\n\n")
                f.write("1. **Essential Components:** Maintain all components with performance drop > 0.01\n")
                f.write("2. **Optimization Targets:** Focus optimization efforts on most critical components\n")
                f.write("3. **Computational Trade-offs:** Consider disabling least critical components for faster inference\n")
                
            self.logger.info(f"Ablation report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
    
    def _save_ablation_results(self, results: AblationResults):
       
        try:
            # Save as JSON
            json_path = self.output_dir / "ablation_results.json"
            
            # Convert to JSON-serializable format
            results_dict = {
                'study_name': results.study_name,
                'start_time': results.start_time,
                'end_time': results.end_time,
                'total_experiments': results.total_experiments,
                'successful_experiments': results.successful_experiments,
                'failed_experiments': results.failed_experiments,
                'baseline_performance': results.baseline_performance,
                'component_importance': results.component_importance,
                'component_interactions': results.component_interactions,
                'significance_tests': results.significance_tests,
                'effect_sizes': results.effect_sizes,
                'experiments': {
                    exp_id: {
                        'experiment_id': exp.experiment_id,
                        'components': exp.components,
                        'performance_metrics': exp.performance_metrics,
                        'training_time': exp.training_time,
                        'status': exp.status,
                        'error_message': exp.error_message
                    }
                    for exp_id, exp in results.experiments.items()
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            # Save as pickle for Python access
            pickle_path = self.output_dir / "ablation_results.pkl"
            import pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Results saved to {json_path} and {pickle_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")


def run_ablation_study(output_dir: Path = Path("ablation_results"),
                      datasets: List[str] = None,
                      study_types: List[str] = None) -> AblationResults:
    """
    Run comprehensive ablation study for HyperPath-SVM.
    
    Args:
        output_dir: Directory to save results
        datasets: Datasets to use for evaluation
        study_types: Types of ablation studies to run
        
    Returns:
        Complete ablation study results
    """
    orchestrator = AblationStudyOrchestrator(output_dir)
    return orchestrator.run_comprehensive_ablation_study(datasets, study_types)


if __name__ == "__main__":
    # Test ablation study
    logger = get_logger(__name__)
    logger.info("Testing ablation study framework...")
    
    try:
        # Run test ablation study
        results = run_ablation_study(
            output_dir=Path("test_ablation"),
            datasets=['caida'],  # Single dataset for testing
            study_types=['leave_one_out']  # Single study type for testing
        )
        
        logger.info("Ablation Study Results:")
        logger.info(f"Completion rate: {results.get_completion_rate():.1%}")
        logger.info(f"Component importance: {results.component_importance}")
        
    except Exception as e:
        logger.error(f"Ablation study test failed: {str(e)}")
    
    logger.info("Ablation study framework testing completed!") 
