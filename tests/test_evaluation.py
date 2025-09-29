# File: tests/test_evaluation.py

"""
Comprehensive unit tests for evaluation framework components.
Tests metrics computation, evaluator functionality, and cross-validation.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperpath_svm.evaluation.metrics import (
    RoutingAccuracy, InferenceTime, MemoryUsage, PathOptimality, 
    ConvergenceRate, ThroughputMetric, PacketLossRate, NetworkStability,
    AdaptationTime
)
from hyperpath_svm.evaluation.evaluator import HyperPathEvaluator
from hyperpath_svm.evaluation.cross_validation import TemporalCrossValidator


class TestRoutingMetrics(unittest.TestCase):
    """Test cases for individual routing metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample predictions and ground truth
        self.sample_predictions = [
            [0, 1, 3, 5],    # Path 1
            [0, 2, 4, 5],    # Path 2  
            [0, 1, 4],       # Path 3
            [0, 3, 5],       # Path 4
            [0, 2, 3, 5]     # Path 5
        ]
        
        self.sample_ground_truth = [
            [0, 1, 3, 5],    # Exact match
            [0, 1, 4, 5],    # Different path
            [0, 1, 4],       # Exact match
            [0, 2, 5],       # Different path
            [0, 2, 3, 5]     # Exact match
        ]
        
        # Create sample network topology
        self.sample_topology = np.array([
            [0, 1, 1, 1, 0, 0],  # Node 0 connects to 1,2,3
            [1, 0, 0, 1, 1, 0],  # Node 1 connects to 0,3,4
            [1, 0, 0, 1, 1, 0],  # Node 2 connects to 0,3,4
            [1, 1, 1, 0, 0, 1],  # Node 3 connects to 0,1,2,5
            [0, 1, 1, 0, 0, 1],  # Node 4 connects to 1,2,5
            [0, 0, 0, 1, 1, 0]   # Node 5 connects to 3,4
        ])
        
        # Timing data for performance metrics
        self.sample_inference_times = [1.2, 1.5, 0.8, 2.1, 1.8]  # milliseconds
        self.sample_memory_usage = [45.2, 67.8, 52.1, 89.3, 71.5]  # MB
    
    def test_routing_accuracy_metric(self):
        """Test routing accuracy computation."""
        accuracy_metric = RoutingAccuracy()
        
        # Test exact path matching
        accuracy = accuracy_metric.compute(
            self.sample_predictions, 
            self.sample_ground_truth
        )
        
        # 3 out of 5 predictions are exact matches
        expected_accuracy = 3.0 / 5.0
        self.assertAlmostEqual(accuracy, expected_accuracy, places=3)
        
        # Test first hop accuracy
        first_hop_accuracy = accuracy_metric.compute_first_hop_accuracy(
            self.sample_predictions,
            self.sample_ground_truth
        )
        
        # All predictions have correct first hop (0)
        # Check second hop: [1,2,1,3,2] vs [1,1,1,2,2] -> 3/5 correct
        expected_first_hop = 3.0 / 5.0
        self.assertAlmostEqual(first_hop_accuracy, expected_first_hop, places=3)
        
        # Test with empty predictions
        empty_accuracy = accuracy_metric.compute([], [])
        self.assertEqual(empty_accuracy, 0.0)
    
    def test_path_optimality_metric(self):
        """Test path optimality computation."""
        optimality_metric = PathOptimality()
        
        # Compute path lengths
        pred_lengths = [len(path) for path in self.sample_predictions]
        truth_lengths = [len(path) for path in self.sample_ground_truth]
        
        optimality = optimality_metric.compute(
            self.sample_predictions,
            self.sample_ground_truth,
            topology=self.sample_topology
        )
        
        # Optimality should be between 0 and 1
        self.assertGreaterEqual(optimality, 0.0)
        self.assertLessEqual(optimality, 1.0)
        
        # Test with identical paths (should give optimality = 1.0)
        identical_optimality = optimality_metric.compute(
            self.sample_ground_truth,
            self.sample_ground_truth,
            topology=self.sample_topology
        )
        self.assertAlmostEqual(identical_optimality, 1.0, places=3)
    
    def test_inference_time_metric(self):
        """Test inference time measurement."""
        inference_metric = InferenceTime()
        
        # Test with sample timing data
        avg_time = inference_metric.compute(self.sample_inference_times)
        expected_avg = np.mean(self.sample_inference_times)
        self.assertAlmostEqual(avg_time, expected_avg, places=3)
        
        # Test percentile computation
        p95_time = inference_metric.compute_percentile(
            self.sample_inference_times, 
            percentile=95
        )
        expected_p95 = np.percentile(self.sample_inference_times, 95)
        self.assertAlmostEqual(p95_time, expected_p95, places=3)
        
        # Test target compliance
        target_compliance = inference_metric.check_target_compliance(
            self.sample_inference_times,
            target_ms=1.8
        )
        
        # 3 out of 5 samples are <= 1.8ms
        expected_compliance = 3.0 / 5.0
        self.assertAlmostEqual(target_compliance, expected_compliance, places=3)
    
    def test_memory_usage_metric(self):
        """Test memory usage measurement."""
        memory_metric = MemoryUsage()
        
        # Test basic statistics
        stats = memory_metric.compute_statistics(self.sample_memory_usage)
        
        self.assertIn('mean', stats)
        self.assertIn('max', stats)
        self.assertIn('std', stats)
        self.assertIn('p95', stats)
        
        expected_mean = np.mean(self.sample_memory_usage)
        self.assertAlmostEqual(stats['mean'], expected_mean, places=3)
        
        expected_max = np.max(self.sample_memory_usage)
        self.assertAlmostEqual(stats['max'], expected_max, places=3)
        
        # Test memory efficiency score
        efficiency_score = memory_metric.compute_efficiency_score(
            self.sample_memory_usage,
            target_mb=98.0
        )
        
        self.assertGreaterEqual(efficiency_score, 0.0)
        self.assertLessEqual(efficiency_score, 1.0)
    
    def test_convergence_rate_metric(self):
        """Test convergence rate computation."""
        convergence_metric = ConvergenceRate()
        
        # Create sample training history
        loss_history = [1.0, 0.8, 0.6, 0.5, 0.48, 0.47, 0.46, 0.45, 0.45, 0.45]
        
        convergence_info = convergence_metric.compute(loss_history)
        
        self.assertIn('convergence_epoch', convergence_info)
        self.assertIn('convergence_rate', convergence_info)
        self.assertIn('final_loss', convergence_info)
        
        # Should converge somewhere in the later epochs
        self.assertGreater(convergence_info['convergence_epoch'], 0)
        self.assertLess(convergence_info['convergence_epoch'], len(loss_history))
        
        # Test with non-converging sequence
        non_converging = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        non_conv_info = convergence_metric.compute(non_converging, tolerance=0.01)
        
        # Should not detect convergence
        self.assertEqual(non_conv_info['convergence_epoch'], -1)
    
    def test_throughput_metric(self):
        """Test throughput measurement."""
        throughput_metric = ThroughputMetric()
        
        # Sample data: decisions per second over time
        decisions_per_second = [850, 920, 1100, 950, 1050, 980, 1200, 900]
        
        throughput_stats = throughput_metric.compute(decisions_per_second)
        
        self.assertIn('mean_throughput', throughput_stats)
        self.assertIn('max_throughput', throughput_stats)
        self.assertIn('min_throughput', throughput_stats)
        self.assertIn('throughput_stability', throughput_stats)
        
        # Check values
        expected_mean = np.mean(decisions_per_second)
        self.assertAlmostEqual(throughput_stats['mean_throughput'], expected_mean, places=1)
        
        # Stability should be between 0 and 1 (lower is more stable)
        self.assertGreaterEqual(throughput_stats['throughput_stability'], 0.0)
        self.assertLessEqual(throughput_stats['throughput_stability'], 1.0)
    
    def test_packet_loss_rate_metric(self):
        """Test packet loss rate computation."""
        packet_loss_metric = PacketLossRate()
        
        # Sample data: packets sent and received
        packets_sent = [1000, 1500, 2000, 1200, 800]
        packets_received = [998, 1495, 1990, 1195, 799]
        
        loss_rate = packet_loss_metric.compute(packets_sent, packets_received)
        
        # Calculate expected loss rate
        total_sent = sum(packets_sent)
        total_received = sum(packets_received)
        expected_loss_rate = (total_sent - total_received) / total_sent
        
        self.assertAlmostEqual(loss_rate, expected_loss_rate, places=4)
        
        # Test edge cases
        perfect_delivery = packet_loss_metric.compute([1000], [1000])
        self.assertEqual(perfect_delivery, 0.0)
        
        complete_loss = packet_loss_metric.compute([1000], [0])
        self.assertEqual(complete_loss, 1.0)
    
    def test_network_stability_metric(self):
        """Test network stability measurement."""
        stability_metric = NetworkStability()
        
        # Sample topology snapshots over time
        topology_snapshots = []
        base_topology = self.sample_topology.copy()
        
        for i in range(10):
            # Add small random perturbations
            noise = np.random.randn(*base_topology.shape) * 0.01
            perturbed_topology = base_topology + noise
            perturbed_topology = np.maximum(0, perturbed_topology)  # Keep non-negative
            topology_snapshots.append(perturbed_topology)
        
        stability_score = stability_metric.compute(topology_snapshots)
        
        self.assertGreaterEqual(stability_score, 0.0)
        self.assertLessEqual(stability_score, 1.0)
        
        # Test with identical snapshots (perfect stability)
        identical_snapshots = [base_topology] * 10
        perfect_stability = stability_metric.compute(identical_snapshots)
        self.assertGreater(perfect_stability, stability_score)  # Should be more stable
    
    def test_adaptation_time_metric(self):
        """Test adaptation time measurement."""
        adaptation_metric = AdaptationTime()
        
        # Sample adaptation events with timestamps and convergence info
        adaptation_events = [
            {'start_time': 0.0, 'convergence_time': 2.5, 'performance_change': 0.05},
            {'start_time': 100.0, 'convergence_time': 1.8, 'performance_change': 0.03},
            {'start_time': 250.0, 'convergence_time': 3.2, 'performance_change': 0.08},
            {'start_time': 400.0, 'convergence_time': 2.1, 'performance_change': 0.04}
        ]
        
        adaptation_stats = adaptation_metric.compute(adaptation_events)
        
        self.assertIn('mean_adaptation_time', adaptation_stats)
        self.assertIn('adaptation_efficiency', adaptation_stats)
        self.assertIn('adaptation_frequency', adaptation_stats)
        
        # Check mean adaptation time
        times = [event['convergence_time'] for event in adaptation_events]
        expected_mean = np.mean(times)
        self.assertAlmostEqual(
            adaptation_stats['mean_adaptation_time'], 
            expected_mean, 
            places=3
        )
        
        # Efficiency should be positive (better performance change per time unit)
        self.assertGreater(adaptation_stats['adaptation_efficiency'], 0)


class TestHyperPathEvaluator(unittest.TestCase):
    """Test cases for the main HyperPath evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = HyperPathEvaluator()
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict = Mock(return_value=np.array([
            [0, 1, 3], [0, 2, 4], [1, 3, 5], [2, 4, 5], [0, 1, 4]
        ]))
        
        # Create test data
        self.X_test = np.random.randn(5, 10)
        self.y_test = np.array([
            [0, 1, 3], [0, 1, 4], [1, 3, 5], [2, 3, 5], [0, 2, 4]
        ])
        
        # Create sample topology
        self.topology = np.random.rand(6, 6)
        self.topology = (self.topology + self.topology.T) / 2
        np.fill_diagonal(self.topology, 0)
    
    def test_model_evaluation(self):
        """Test comprehensive model evaluation."""
        evaluation_result = self.evaluator.evaluate_model(
            self.mock_model,
            self.X_test,
            self.y_test,
            topology=self.topology,
            compute_detailed_metrics=True
        )
        
        # Check required metrics are present
        required_metrics = [
            'routing_accuracy', 'path_optimality', 'inference_time_ms',
            'memory_usage_mb', 'convergence_rate'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, evaluation_result)
        
        # Check value ranges
        self.assertGreaterEqual(evaluation_result['routing_accuracy'], 0.0)
        self.assertLessEqual(evaluation_result['routing_accuracy'], 1.0)
        
        self.assertGreater(evaluation_result['inference_time_ms'], 0.0)
        self.assertGreater(evaluation_result['memory_usage_mb'], 0.0)
        
        # Verify mock was called correctly
        self.mock_model.predict.assert_called_once_with(self.X_test)
    
    def test_batch_evaluation(self):
        """Test batch evaluation of multiple models."""
        # Create multiple mock models
        models = {
            'model_1': Mock(),
            'model_2': Mock(),
            'model_3': Mock()
        }
        
        # Set up different prediction behaviors
        models['model_1'].predict = Mock(return_value=np.array([
            [0, 1], [0, 2], [1, 3], [2, 4], [0, 1]
        ]))
        models['model_2'].predict = Mock(return_value=np.array([
            [0, 1, 3], [0, 2, 4], [1, 3, 5], [2, 4, 5], [0, 1, 4]
        ]))
        models['model_3'].predict = Mock(return_value=np.array([
            [0, 2], [0, 1], [1, 4], [2, 5], [0, 2]
        ]))
        
        # Perform batch evaluation
        batch_results = self.evaluator.evaluate_multiple_models(
            models,
            self.X_test,
            self.y_test,
            topology=self.topology
        )
        
        # Check that all models were evaluated
        self.assertEqual(len(batch_results), 3)
        
        for model_name in models.keys():
            self.assertIn(model_name, batch_results)
            
            result = batch_results[model_name]
            self.assertIn('routing_accuracy', result)
            self.assertIn('inference_time_ms', result)
            self.assertIn('evaluation_timestamp', result)
    
    def test_performance_comparison(self):
        """Test performance comparison functionality."""
        # Create evaluation results for comparison
        results = {
            'hyperpath_svm': {
                'routing_accuracy': 0.95,
                'inference_time_ms': 1.5,
                'memory_usage_mb': 85.0,
                'path_optimality': 0.92
            },
            'baseline_gnn': {
                'routing_accuracy': 0.88,
                'inference_time_ms': 3.2,
                'memory_usage_mb': 120.0,
                'path_optimality': 0.85
            },
            'static_svm': {
                'routing_accuracy': 0.82,
                'inference_time_ms': 0.8,
                'memory_usage_mb': 45.0,
                'path_optimality': 0.78
            }
        }
        
        comparison = self.evaluator.compare_model_performance(results)
        
        # Check comparison structure
        self.assertIn('rankings', comparison)
        self.assertIn('improvements', comparison)
        self.assertIn('summary', comparison)
        
        # Check rankings
        rankings = comparison['rankings']
        self.assertIn('routing_accuracy', rankings)
        
        # HyperPath-SVM should rank first in accuracy
        accuracy_ranking = rankings['routing_accuracy']
        self.assertEqual(accuracy_ranking[0][0], 'hyperpath_svm')
        
        # Check improvements (relative to best baseline)
        improvements = comparison['improvements']
        self.assertIn('routing_accuracy_improvement', improvements)
        
        # Should show improvement over baselines
        accuracy_improvement = improvements['routing_accuracy_improvement']
        self.assertGreater(accuracy_improvement, 0.0)
    
    def test_statistical_significance_testing(self):
        """Test statistical significance testing."""
        # Create sample performance data for statistical testing
        hyperpath_scores = np.random.normal(0.95, 0.02, 100)  # High performance
        baseline_scores = np.random.normal(0.88, 0.03, 100)   # Lower performance
        
        significance_result = self.evaluator.test_statistical_significance(
            hyperpath_scores,
            baseline_scores,
            test_type='t_test',
            alternative='greater'
        )
        
        self.assertIn('statistic', significance_result)
        self.assertIn('p_value', significance_result)
        self.assertIn('significant', significance_result)
        self.assertIn('effect_size', significance_result)
        
        # Should be statistically significant given the setup
        self.assertTrue(significance_result['significant'])
        self.assertLess(significance_result['p_value'], 0.05)
        
        # Effect size should be positive (hyperpath better than baseline)
        self.assertGreater(significance_result['effect_size'], 0.0)
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        # Sample performance data
        performance_data = np.random.normal(0.92, 0.05, 50)
        
        confidence_interval = self.evaluator.compute_confidence_interval(
            performance_data,
            confidence_level=0.95
        )
        
        self.assertIn('mean', confidence_interval)
        self.assertIn('lower_bound', confidence_interval)
        self.assertIn('upper_bound', confidence_interval)
        self.assertIn('margin_of_error', confidence_interval)
        
        # Check logical relationships
        mean = confidence_interval['mean']
        lower = confidence_interval['lower_bound']
        upper = confidence_interval['upper_bound']
        
        self.assertLess(lower, mean)
        self.assertLess(mean, upper)
        self.assertAlmostEqual(mean, np.mean(performance_data), places=6)
    
    def test_performance_over_time(self):
        """Test performance tracking over time."""
        # Create time-series performance data
        timestamps = [datetime.now().timestamp() + i * 3600 for i in range(24)]  # 24 hours
        accuracy_values = 0.9 + 0.05 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.normal(0, 0.01, 24)
        
        time_series_data = []
        for t, acc in zip(timestamps, accuracy_values):
            time_series_data.append({
                'timestamp': t,
                'routing_accuracy': acc,
                'inference_time_ms': np.random.uniform(1.0, 2.0),
                'memory_usage_mb': np.random.uniform(80, 100)
            })
        
        trend_analysis = self.evaluator.analyze_performance_trends(time_series_data)
        
        self.assertIn('trends', trend_analysis)
        self.assertIn('stability_metrics', trend_analysis)
        self.assertIn('performance_degradation', trend_analysis)
        
        # Check trend analysis for accuracy
        trends = trend_analysis['trends']
        self.assertIn('routing_accuracy', trends)
        
        accuracy_trend = trends['routing_accuracy']
        self.assertIn('slope', accuracy_trend)
        self.assertIn('r_squared', accuracy_trend)
    
    def test_target_compliance_checking(self):
        """Test performance target compliance checking."""
        performance_targets = {
            'routing_accuracy': 0.965,
            'inference_time_ms': 1.8,
            'memory_usage_mb': 98.0,
            'path_optimality': 0.90
        }
        
        test_results = {
            'routing_accuracy': 0.970,     # Meets target
            'inference_time_ms': 1.5,     # Meets target
            'memory_usage_mb': 105.0,     # Exceeds target
            'path_optimality': 0.88       # Below target
        }
        
        compliance_check = self.evaluator.check_target_compliance(
            test_results,
            performance_targets
        )
        
        self.assertIn('overall_compliance', compliance_check)
        self.assertIn('individual_compliance', compliance_check)
        self.assertIn('compliance_score', compliance_check)
        
        # Check individual compliance
        individual = compliance_check['individual_compliance']
        
        self.assertTrue(individual['routing_accuracy'])  # Should meet target
        self.assertTrue(individual['inference_time_ms'])  # Should meet target
        self.assertFalse(individual['memory_usage_mb'])   # Should exceed target
        self.assertFalse(individual['path_optimality'])   # Should be below target
        
        # Overall compliance should be False (not all targets met)
        self.assertFalse(compliance_check['overall_compliance'])
        
        # Compliance score should be 0.5 (2 out of 4 targets met)
        self.assertAlmostEqual(compliance_check['compliance_score'], 0.5, places=3)


class TestTemporalCrossValidator(unittest.TestCase):
    """Test cases for temporal cross-validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cv = TemporalCrossValidator()
        
        # Create mock model with fit and predict methods
        self.mock_model = Mock()
        self.mock_model.fit = Mock()
        self.mock_model.predict = Mock()
        
        # Create time-ordered sample data
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 3, 100)
        
        # Add temporal ordering (timestamps)
        self.timestamps = [datetime.now().timestamp() + i * 3600 for i in range(100)]
    
    def test_temporal_split_generation(self):
        """Test temporal split generation."""
        splits = self.cv.generate_temporal_splits(
            self.X, 
            n_splits=5,
            test_size=0.2
        )
        
        # Should generate 5 splits
        self.assertEqual(len(splits), 5)
        
        for train_indices, test_indices in splits:
            # Check that splits are non-overlapping
            self.assertEqual(len(set(train_indices) & set(test_indices)), 0)
            
            # Check temporal ordering (test indices should come after train indices)
            if len(train_indices) > 0 and len(test_indices) > 0:
                max_train_idx = max(train_indices)
                min_test_idx = min(test_indices)
                self.assertLess(max_train_idx, min_test_idx)
            
            # Check test size approximately matches requested proportion
            total_size = len(train_indices) + len(test_indices)
            actual_test_proportion = len(test_indices) / total_size
            self.assertAlmostEqual(actual_test_proportion, 0.2, delta=0.1)
    
    def test_temporal_cross_validation(self):
        """Test complete temporal cross-validation."""
        # Set up mock model behavior
        self.mock_model.predict.return_value = np.random.randint(0, 3, 20)  # Predict for test set
        
        cv_results = self.cv.validate(
            self.mock_model,
            self.X,
            self.y,
            n_splits=3,
            test_size=0.2,
            temporal_order=True
        )
        
        # Check result structure
        self.assertIn('fold_scores', cv_results)
        self.assertIn('mean_score', cv_results)
        self.assertIn('std_score', cv_results)
        self.assertIn('confidence_interval', cv_results)
        
        # Should have 3 fold scores
        self.assertEqual(len(cv_results['fold_scores']), 3)
        
        # Check that model was fit and predict was called for each fold
        self.assertEqual(self.mock_model.fit.call_count, 3)
        self.assertEqual(self.mock_model.predict.call_count, 3)
        
        # Mean and std should be computed from fold scores
        expected_mean = np.mean(cv_results['fold_scores'])
        expected_std = np.std(cv_results['fold_scores'])
        
        self.assertAlmostEqual(cv_results['mean_score'], expected_mean, places=6)
        self.assertAlmostEqual(cv_results['std_score'], expected_std, places=6)
    
    def test_walk_forward_validation(self):
        """Test walk-forward validation strategy."""
        # Set up mock model behavior
        accuracies = [0.85, 0.87, 0.89, 0.91, 0.88]  # Simulated accuracies for each step
        
        def mock_score(y_true, y_pred):
            return accuracies.pop(0) if accuracies else 0.85
        
        wf_results = self.cv.walk_forward_validation(
            self.mock_model,
            self.X,
            self.y,
            initial_train_size=50,
            step_size=10,
            scoring_function=mock_score
        )
        
        self.assertIn('step_scores', wf_results)
        self.assertIn('cumulative_score', wf_results)
        self.assertIn('score_trend', wf_results)
        
        # Should have multiple step scores
        self.assertGreater(len(wf_results['step_scores']), 0)
        
        # Cumulative score should be computed
        self.assertIsInstance(wf_results['cumulative_score'], float)
    
    def test_blocked_time_series_split(self):
        """Test blocked time series cross-validation."""
        blocked_splits = self.cv.blocked_time_series_split(
            self.X,
            n_splits=4,
            block_size=20,
            gap_size=5
        )
        
        # Should generate 4 splits
        self.assertEqual(len(blocked_splits), 4)
        
        for train_indices, test_indices in blocked_splits:
            # Check block sizes
            self.assertGreater(len(train_indices), 0)
            self.assertGreater(len(test_indices), 0)
            
            # Check gap between train and test
            max_train = max(train_indices)
            min_test = min(test_indices)
            gap = min_test - max_train
            self.assertGreaterEqual(gap, 5)  # Should respect gap_size
    
    def test_expanding_window_validation(self):
        """Test expanding window cross-validation."""
        expanding_results = self.cv.expanding_window_validation(
            self.mock_model,
            self.X,
            self.y,
            initial_size=30,
            step_size=15,
            max_train_size=80
        )
        
        self.assertIn('window_scores', expanding_results)
        self.assertIn('score_progression', expanding_results)
        self.assertIn('learning_curve', expanding_results)
        
        # Should have multiple window scores
        window_scores = expanding_results['window_scores']
        self.assertGreater(len(window_scores), 0)
        
        # Learning curve should show how performance changes with training size
        learning_curve = expanding_results['learning_curve']
        self.assertIn('train_sizes', learning_curve)
        self.assertIn('scores', learning_curve)
        self.assertEqual(len(learning_curve['train_sizes']), len(learning_curve['scores']))
    
    def test_validation_with_custom_metrics(self):
        """Test cross-validation with custom scoring metrics."""
        def custom_routing_accuracy(y_true, y_pred):
            """Custom routing accuracy metric."""
            correct = 0
            for true_path, pred_path in zip(y_true, y_pred):
                if len(true_path) > 0 and len(pred_path) > 0:
                    if true_path[0] == pred_path[0]:  # First hop match
                        correct += 1
            return correct / len(y_true) if len(y_true) > 0 else 0.0
        
        # Mock model to return path-like predictions
        def mock_predict(X):
            return [[np.random.randint(0, 5)] for _ in range(len(X))]
        
        self.mock_model.predict = mock_predict
        
        # Create path-like labels
        y_paths = [[i % 5] for i in range(len(self.y))]
        
        cv_results = self.cv.validate(
            self.mock_model,
            self.X,
            y_paths,
            scoring_function=custom_routing_accuracy,
            n_splits=3
        )
        
        self.assertIn('fold_scores', cv_results)
        self.assertIn('mean_score', cv_results)
        
        # All scores should be between 0 and 1 for accuracy metric
        for score in cv_results['fold_scores']:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_cross_validation_with_groups(self):
        """Test cross-validation with group-based splitting."""
        # Create groups (e.g., different network topologies)
        groups = np.random.randint(0, 5, len(self.X))
        
        grouped_cv_results = self.cv.group_based_validation(
            self.mock_model,
            self.X,
            self.y,
            groups=groups,
            n_splits=3
        )
        
        self.assertIn('fold_scores', grouped_cv_results)
        self.assertIn('group_performance', grouped_cv_results)
        
        # Should track performance per group
        group_performance = grouped_cv_results['group_performance']
        self.assertIsInstance(group_performance, dict)
        
        # Each group should have performance metrics
        for group_id in set(groups):
            if group_id in group_performance:
                self.assertIn('mean_score', group_performance[group_id])
                self.assertIn('sample_count', group_performance[group_id])
    
    def test_nested_cross_validation(self):
        """Test nested cross-validation for hyperparameter tuning."""
        # Mock hyperparameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['auto', 0.001, 0.01]
        }
        
        nested_cv_results = self.cv.nested_cross_validation(
            self.mock_model,
            self.X,
            self.y,
            param_grid=param_grid,
            outer_cv=3,
            inner_cv=3
        )
        
        self.assertIn('outer_fold_scores', nested_cv_results)
        self.assertIn('best_params_per_fold', nested_cv_results)
        self.assertIn('nested_score_mean', nested_cv_results)
        self.assertIn('nested_score_std', nested_cv_results)
        
        # Should have scores for each outer fold
        self.assertEqual(len(nested_cv_results['outer_fold_scores']), 3)
        
        # Should have best parameters for each outer fold
        self.assertEqual(len(nested_cv_results['best_params_per_fold']), 3)


if __name__ == '__main__':
    # Set up test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestRoutingMetrics,
        TestHyperPathEvaluator,
        TestTemporalCrossValidator
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION FRAMEWORK TEST SUMMARY")
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
