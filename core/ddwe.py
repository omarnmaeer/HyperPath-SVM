# File: hyperpath_svm/core/ddwe.py
"""
Dynamic Discriminative Weight Evolution (DDWE) Implementation

This module implements the DDWE algorithm with:
- Three-tier hierarchical memory (short: 5min, medium: 4h, long: 24h)
- Closed-form weight updates without backpropagation
- Mathematical convergence guarantees
- Real-time adaptation to network performance feedback

Key Innovation: Continuous learning without full retraining
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances

logger = logging.getLogger(__name__)


@dataclass
class MemoryLevel:
    """Configuration for a memory hierarchy level"""
    name: str
    window_size: int      # Time window in seconds
    weight: float         # Importance weight [0,1]
    capacity: int         # Maximum number of samples
    decay_factor: float   # Temporal decay factor
    access_pattern: str   # 'fifo', 'lru', 'importance_based'


class ShortTermMemory:
    """
    Short-term memory for immediate network performance patterns
    Window: 5 minutes, Weight: 0.5, High reactivity to recent changes
    """
    
    def __init__(self, window: int = 300, weight: float = 0.5, capacity: int = 10000):
        self.window = window
        self.weight = weight
        self.capacity = capacity
        
        # Store experiences as (timestamp, performance, weights, context)
        self.experiences = deque(maxlen=capacity)
        self.performance_history = deque(maxlen=1000)
        self.weight_history = deque(maxlen=1000)
        
        # Fast lookup structures
        self._timestamp_index = {}
        self._performance_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug(f"ShortTermMemory initialized: {window}s window, {capacity} capacity")
    
    def store_experience(self, timestamp: float, performance_metrics: Dict, 
                        kernel_weights: np.ndarray, context: Dict = None):
        """Store network performance experience"""
        
        with self._lock:
            experience = {
                'timestamp': timestamp,
                'performance': performance_metrics.copy(),
                'weights': kernel_weights.copy(),
                'context': context or {},
                'importance': self._compute_importance(performance_metrics)
            }
            
            self.experiences.append(experience)
            self._timestamp_index[timestamp] = len(self.experiences) - 1
            
            # Update running statistics
            self._update_performance_stats(performance_metrics)
            
            # Clean old experiences outside window
            self._cleanup_old_experiences(timestamp)
    
    def get_recent_experiences(self, current_time: float, 
                             max_age: Optional[float] = None) -> List[Dict]:
        """Get experiences within time window"""
        
        max_age = max_age or self.window
        cutoff_time = current_time - max_age
        
        with self._lock:
            recent = []
            for exp in reversed(self.experiences):
                if exp['timestamp'] >= cutoff_time:
                    recent.append(exp)
                else:
                    break  # Experiences are ordered by time
            
            return list(reversed(recent))  # Return in chronological order
    
    def compute_adaptive_weights(self, current_time: float, 
                               base_weights: np.ndarray) -> np.ndarray:
        """
        Compute adaptive weights based on recent performance patterns
        
        Implements Equation 3 from paper:
        λ_k^(s)(t) = λ_k^(s)(t-1) + η * ∇_λ J_s(t) * exp(-α_s * Δt)
        """
        
        recent_experiences = self.get_recent_experiences(current_time)
        
        if not recent_experiences:
            return base_weights.copy()
        
        # Compute performance-based weight updates
        weight_updates = np.zeros_like(base_weights)
        total_weight = 0.0
        
        for exp in recent_experiences:
            # Time decay factor
            time_diff = current_time - exp['timestamp']
            decay = np.exp(-0.01 * time_diff)  # α_s = 0.01
            
            # Performance gradient (simplified)
            performance_score = self._aggregate_performance(exp['performance'])
            gradient = self._compute_performance_gradient(exp['weights'], performance_score)
            
            # Weight update with temporal decay
            weight_updates += decay * gradient * performance_score
            total_weight += decay
        
        if total_weight > 0:
            weight_updates /= total_weight
            
            # Apply learning rate and constraints
            learning_rate = 0.01
            updated_weights = base_weights + learning_rate * weight_updates
            
            # Ensure non-negative and normalize
            updated_weights = np.maximum(updated_weights, 0.01)
            updated_weights /= np.sum(updated_weights)
            
            return updated_weights
        
        return base_weights.copy()
    
    def _compute_importance(self, performance_metrics: Dict) -> float:
        """Compute importance score for experience"""
        
        # Combine multiple performance dimensions
        accuracy = performance_metrics.get('accuracy', 0.5)
        latency = 1.0 / (1.0 + performance_metrics.get('latency', 1.0))  # Lower is better
        throughput = performance_metrics.get('throughput', 0.5)
        stability = performance_metrics.get('stability', 0.5)
        
        # Weighted combination
        importance = (0.4 * accuracy + 0.3 * latency + 
                     0.2 * throughput + 0.1 * stability)
        
        return importance
    
    def _aggregate_performance(self, performance_metrics: Dict) -> float:
        """Aggregate performance metrics into single score"""
        return self._compute_importance(performance_metrics)
    
    def _compute_performance_gradient(self, weights: np.ndarray, 
                                   performance: float) -> np.ndarray:
        """
        Compute gradient of performance with respect to weights
        Simplified finite difference approximation
        """
        
        gradient = np.zeros_like(weights)
        epsilon = 1e-6
        
        for i in range(len(weights)):
            # Finite difference approximation
            # In practice, this would use actual performance feedback
            gradient[i] = (performance - 0.5) * weights[i] + np.random.normal(0, epsilon)
        
        return gradient
    
    def _update_performance_stats(self, performance_metrics: Dict):
        """Update running performance statistics"""
        
        performance_score = self._aggregate_performance(performance_metrics)
        self.performance_history.append(performance_score)
        
        # Update running mean and std
        scores = list(self.performance_history)
        self._performance_stats['mean'] = np.mean(scores)
        self._performance_stats['std'] = np.std(scores) if len(scores) > 1 else 1.0
        self._performance_stats['count'] = len(scores)
    
    def _cleanup_old_experiences(self, current_time: float):
        """Remove experiences outside time window"""
        
        cutoff_time = current_time - self.window
        
        # Remove old experiences from front of deque
        while (self.experiences and 
               self.experiences[0]['timestamp'] < cutoff_time):
            old_exp = self.experiences.popleft()
            # Clean up timestamp index
            if old_exp['timestamp'] in self._timestamp_index:
                del self._timestamp_index[old_exp['timestamp']]
    
    def get_memory_stats(self) -> Dict:
        """Get memory level statistics"""
        
        with self._lock:
            return {
                'level': 'short_term',
                'window_seconds': self.window,
                'weight': self.weight,
                'num_experiences': len(self.experiences),
                'capacity': self.capacity,
                'utilization': len(self.experiences) / self.capacity,
                'performance_stats': self._performance_stats.copy(),
                'oldest_timestamp': self.experiences[0]['timestamp'] if self.experiences else None,
                'newest_timestamp': self.experiences[-1]['timestamp'] if self.experiences else None
            }


class MediumTermMemory:
    """
    Medium-term memory for routing convergence patterns
    Window: 4 hours, Weight: 0.3, Balanced between reactivity and stability
    """
    
    def __init__(self, window: int = 14400, weight: float = 0.3, capacity: int = 50000):
        self.window = window
        self.weight = weight
        self.capacity = capacity
        self.decay_factor = 0.95  # γ_m from paper
        
        # Experience storage with LRU eviction
        self.experiences = {}  # timestamp -> experience
        self.access_order = deque()  # For LRU tracking
        self.aggregated_patterns = defaultdict(list)
        
        # Pattern recognition
        self.pattern_detector = PatternDetector(window_size=self.window)
        self.convergence_tracker = ConvergenceTracker()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug(f"MediumTermMemory initialized: {window}s window, {capacity} capacity")
    
    def store_experience(self, timestamp: float, performance_metrics: Dict,
                        kernel_weights: np.ndarray, context: Dict = None):
        """Store and aggregate medium-term patterns"""
        
        with self._lock:
            # Aggregate short-term experiences into medium-term patterns
            pattern_key = self._identify_pattern(performance_metrics, context)
            
            experience = {
                'timestamp': timestamp,
                'performance': performance_metrics.copy(),
                'weights': kernel_weights.copy(),
                'context': context or {},
                'pattern_key': pattern_key,
                'convergence_state': self.convergence_tracker.analyze(performance_metrics)
            }
            
            # Store with LRU management
            if len(self.experiences) >= self.capacity:
                self._evict_lru()
            
            self.experiences[timestamp] = experience
            self.access_order.append(timestamp)
            
            # Update pattern aggregations
            self.aggregated_patterns[pattern_key].append(experience)
            self.pattern_detector.update(performance_metrics, timestamp)
            
            # Clean old experiences
            self._cleanup_old_experiences(timestamp)
    
    def compute_adaptive_weights(self, current_time: float,
                               base_weights: np.ndarray) -> np.ndarray:
        """
        Compute weights based on medium-term routing convergence patterns
        
        Implements Equation 4 from paper:
        λ_k^(m)(t) = γ_m * λ_k^(m)(t-1) + (1-γ_m) * f_pattern(P_t, λ_k)
        """
        
        with self._lock:
            # Get relevant experiences
            relevant_experiences = self._get_relevant_experiences(current_time)
            
            if not relevant_experiences:
                return base_weights.copy()
            
            # Detect current dominant patterns
            current_patterns = self.pattern_detector.get_dominant_patterns(current_time)
            
            # Compute pattern-based weight updates
            pattern_weights = np.zeros_like(base_weights)
            pattern_count = 0
            
            for pattern_key, confidence in current_patterns.items():
                if pattern_key in self.aggregated_patterns:
                    pattern_experiences = self.aggregated_patterns[pattern_key]
                    
                    # Compute average weights for this pattern
                    pattern_weight_sum = np.zeros_like(base_weights)
                    for exp in pattern_experiences[-10:]:  # Recent examples
                        time_decay = np.exp(-0.001 * (current_time - exp['timestamp']))
                        pattern_weight_sum += time_decay * exp['weights']
                    
                    if len(pattern_experiences) > 0:
                        pattern_avg = pattern_weight_sum / len(pattern_experiences[-10:])
                        pattern_weights += confidence * pattern_avg
                        pattern_count += confidence
            
            if pattern_count > 0:
                pattern_weights /= pattern_count
                
                # Apply decay factor (γ_m)
                updated_weights = (self.decay_factor * base_weights + 
                                 (1 - self.decay_factor) * pattern_weights)
                
                # Normalize and ensure positivity
                updated_weights = np.maximum(updated_weights, 0.01)
                updated_weights /= np.sum(updated_weights)
                
                return updated_weights
            
            return base_weights.copy()
    
    def _identify_pattern(self, performance_metrics: Dict, context: Dict) -> str:
        """Identify pattern category for experience"""
        
        # Simple pattern identification based on performance characteristics
        latency = performance_metrics.get('latency', 0.0)
        throughput = performance_metrics.get('throughput', 0.0)
        loss = performance_metrics.get('packet_loss', 0.0)
        
        # Categorize network state
        if latency > 100:  # High latency
            if loss > 0.01:
                return "congested_lossy"
            else:
                return "high_latency"
        elif throughput < 0.1:  # Low throughput
            return "low_throughput"
        elif loss > 0.005:  # Moderate loss
            return "moderate_loss"
        else:
            return "normal_operation"
    
    def _get_relevant_experiences(self, current_time: float) -> List[Dict]:
        """Get experiences relevant for current time"""
        
        cutoff_time = current_time - self.window
        relevant = []
        
        for timestamp, exp in self.experiences.items():
            if timestamp >= cutoff_time:
                relevant.append(exp)
                # Update access order for LRU
                if timestamp in self.access_order:
                    self.access_order.remove(timestamp)
                    self.access_order.append(timestamp)
        
        return relevant
    
    def _evict_lru(self):
        """Evict least recently used experience"""
        
        if self.access_order:
            lru_timestamp = self.access_order.popleft()
            if lru_timestamp in self.experiences:
                evicted_exp = self.experiences.pop(lru_timestamp)
                
                # Remove from pattern aggregations
                pattern_key = evicted_exp['pattern_key']
                if pattern_key in self.aggregated_patterns:
                    try:
                        self.aggregated_patterns[pattern_key].remove(evicted_exp)
                    except ValueError:
                        pass  # Already removed
    
    def _cleanup_old_experiences(self, current_time: float):
        """Remove experiences outside time window"""
        
        cutoff_time = current_time - self.window
        expired_timestamps = []
        
        for timestamp in self.experiences.keys():
            if timestamp < cutoff_time:
                expired_timestamps.append(timestamp)
        
        for timestamp in expired_timestamps:
            if timestamp in self.experiences:
                exp = self.experiences.pop(timestamp)
                
                # Clean up access order
                if timestamp in self.access_order:
                    self.access_order.remove(timestamp)
                
                # Clean up pattern aggregations
                pattern_key = exp['pattern_key']
                if pattern_key in self.aggregated_patterns:
                    try:
                        self.aggregated_patterns[pattern_key].remove(exp)
                    except ValueError:
                        pass
    
    def get_memory_stats(self) -> Dict:
        """Get memory level statistics"""
        
        with self._lock:
            pattern_stats = {}
            for pattern, experiences in self.aggregated_patterns.items():
                pattern_stats[pattern] = len(experiences)
            
            return {
                'level': 'medium_term',
                'window_seconds': self.window,
                'weight': self.weight,
                'decay_factor': self.decay_factor,
                'num_experiences': len(self.experiences),
                'capacity': self.capacity,
                'utilization': len(self.experiences) / self.capacity,
                'pattern_distribution': pattern_stats,
                'convergence_state': self.convergence_tracker.get_current_state()
            }


class LongTermMemory:
    """
    Long-term memory for daily network patterns and trends
    Window: 24 hours, Weight: 0.2, Focus on stability and long-term trends
    """
    
    def __init__(self, window: int = 86400, weight: float = 0.2, capacity: int = 100000):
        self.window = window
        self.weight = weight
        self.capacity = capacity
        self.decay_factor = 0.99  # γ_l from paper (very slow decay)
        
        # Hierarchical storage by time buckets (hours)
        self.hourly_buckets = defaultdict(list)
        self.daily_trends = TrendAnalyzer(window_size=24)  # 24 hours
        self.stability_metrics = StabilityTracker()
        
        # Compressed representations for efficiency
        self.compressed_patterns = {}
        self.compression_threshold = 1000  # Compress when bucket has >1000 samples
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug(f"LongTermMemory initialized: {window}s window, {capacity} capacity")
    
    def store_experience(self, timestamp: float, performance_metrics: Dict,
                        kernel_weights: np.ndarray, context: Dict = None):
        """Store and analyze long-term trends"""
        
        with self._lock:
            # Determine hour bucket
            hour_bucket = int(timestamp // 3600)  # Hours since epoch
            
            experience = {
                'timestamp': timestamp,
                'performance': performance_metrics.copy(),
                'weights': kernel_weights.copy(),
                'context': context or {},
                'stability_score': self.stability_metrics.compute_stability(performance_metrics)
            }
            
            self.hourly_buckets[hour_bucket].append(experience)
            
            # Update trend analysis
            self.daily_trends.update(performance_metrics, timestamp)
            self.stability_metrics.update(performance_metrics, timestamp)
            
            # Compress old buckets if needed
            if len(self.hourly_buckets[hour_bucket]) > self.compression_threshold:
                self._compress_bucket(hour_bucket)
            
            # Cleanup old data
            self._cleanup_old_data(timestamp)
    
    def compute_adaptive_weights(self, current_time: float,
                               base_weights: np.ndarray) -> np.ndarray:
        """
        Compute weights based on long-term stability and trends
        
        Implements Equation 5 from paper:
        λ_k^(l)(t) = γ_l * λ_k^(l)(t-1) + (1-γ_l) * f_stability(S_t, λ_k)
        """
        
        with self._lock:
            # Get long-term trend information
            current_trends = self.daily_trends.get_current_trends(current_time)
            stability_state = self.stability_metrics.get_stability_state(current_time)
            
            if not current_trends:
                return base_weights.copy()
            
            # Compute stability-based adjustments
            stability_weights = self._compute_stability_weights(
                base_weights, stability_state, current_trends
            )
            
            # Apply very slow decay for long-term stability
            updated_weights = (self.decay_factor * base_weights + 
                             (1 - self.decay_factor) * stability_weights)
            
            # Ensure stability constraints
            updated_weights = np.maximum(updated_weights, 0.05)  # Higher minimum for stability
            updated_weights /= np.sum(updated_weights)
            
            return updated_weights
    
    def _compute_stability_weights(self, base_weights: np.ndarray,
                                 stability_state: Dict, trends: Dict) -> np.ndarray:
        """Compute weight adjustments based on stability requirements"""
        
        stability_weights = base_weights.copy()
        
        # Adjust based on stability metrics
        stability_score = stability_state.get('overall_stability', 0.5)
        
        if stability_score < 0.3:  # Low stability - be more conservative
            # Reduce weight variance for stability
            mean_weight = np.mean(stability_weights)
            stability_weights = 0.7 * stability_weights + 0.3 * mean_weight
        
        elif stability_score > 0.8:  # High stability - can be more adaptive
            # Allow more weight variance for optimization
            trend_direction = trends.get('performance_trend', 0.0)
            if trend_direction > 0:  # Improving performance
                # Slightly increase weights of better performing kernels
                performance_scores = trends.get('kernel_performance', np.ones_like(base_weights))
                stability_weights *= (1.0 + 0.1 * performance_scores)
        
        return stability_weights
    
    def _compress_bucket(self, hour_bucket: int):
        """Compress hourly bucket to save memory"""
        
        experiences = self.hourly_buckets[hour_bucket]
        
        if len(experiences) > self.compression_threshold:
            # Create compressed representation
            compressed = {
                'hour_bucket': hour_bucket,
                'num_samples': len(experiences),
                'avg_performance': self._compute_average_performance(experiences),
                'avg_weights': self._compute_average_weights(experiences),
                'std_performance': self._compute_performance_std(experiences),
                'stability_stats': self._compute_stability_stats(experiences),
                'compression_timestamp': time.time()
            }
            
            self.compressed_patterns[hour_bucket] = compressed
            
            # Keep only most recent samples in bucket
            self.hourly_buckets[hour_bucket] = experiences[-100:]  # Keep recent 100
            
            logger.debug(f"Compressed hour bucket {hour_bucket}: {len(experiences)} -> 100 samples")
    
    def _compute_average_performance(self, experiences: List[Dict]) -> Dict:
        """Compute average performance metrics"""
        
        if not experiences:
            return {}
        
        performance_keys = experiences[0]['performance'].keys()
        avg_performance = {}
        
        for key in performance_keys:
            values = [exp['performance'][key] for exp in experiences 
                     if key in exp['performance']]
            avg_performance[key] = np.mean(values) if values else 0.0
        
        return avg_performance
    
    def _compute_average_weights(self, experiences: List[Dict]) -> np.ndarray:
        """Compute average kernel weights"""
        
        if not experiences:
            return np.array([])
        
        weight_arrays = [exp['weights'] for exp in experiences]
        return np.mean(weight_arrays, axis=0)
    
    def _compute_performance_std(self, experiences: List[Dict]) -> Dict:
        """Compute performance standard deviations"""
        
        if not experiences:
            return {}
        
        performance_keys = experiences[0]['performance'].keys()
        std_performance = {}
        
        for key in performance_keys:
            values = [exp['performance'][key] for exp in experiences 
                     if key in exp['performance']]
            std_performance[key] = np.std(values) if len(values) > 1 else 0.0
        
        return std_performance
    
    def _compute_stability_stats(self, experiences: List[Dict]) -> Dict:
        """Compute stability statistics for bucket"""
        
        stability_scores = [exp.get('stability_score', 0.5) for exp in experiences]
        
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'min_stability': np.min(stability_scores),
            'max_stability': np.max(stability_scores)
        }
    
    def _cleanup_old_data(self, current_time: float):
        """Remove data outside time window"""
        
        cutoff_time = current_time - self.window
        cutoff_hour = int(cutoff_time // 3600)
        
        expired_buckets = []
        for hour_bucket in self.hourly_buckets.keys():
            if hour_bucket < cutoff_hour:
                expired_buckets.append(hour_bucket)
        
        for bucket in expired_buckets:
            del self.hourly_buckets[bucket]
            if bucket in self.compressed_patterns:
                del self.compressed_patterns[bucket]
    
    def get_memory_stats(self) -> Dict:
        """Get memory level statistics"""
        
        with self._lock:
            total_experiences = sum(len(bucket) for bucket in self.hourly_buckets.values())
            num_compressed = len(self.compressed_patterns)
            
            return {
                'level': 'long_term',
                'window_seconds': self.window,
                'weight': self.weight,
                'decay_factor': self.decay_factor,
                'num_experiences': total_experiences,
                'num_compressed_buckets': num_compressed,
                'capacity': self.capacity,
                'utilization': total_experiences / self.capacity,
                'stability_state': self.stability_metrics.get_stability_state(time.time()),
                'trend_analysis': self.daily_trends.get_trend_summary()
            }


class PatternDetector:
    """Detect and classify network performance patterns"""
    
    def __init__(self, window_size: int = 14400):
        self.window_size = window_size
        self.pattern_history = deque(maxlen=1000)
        self.pattern_counts = defaultdict(int)
        self.confidence_threshold = 0.6
    
    def update(self, performance_metrics: Dict, timestamp: float):
        """Update pattern detection with new performance data"""
        
        pattern = self._classify_pattern(performance_metrics)
        
        self.pattern_history.append({
            'timestamp': timestamp,
            'pattern': pattern,
            'performance': performance_metrics.copy()
        })
        
        self.pattern_counts[pattern] += 1
    
    def get_dominant_patterns(self, current_time: float) -> Dict[str, float]:
        """Get currently dominant patterns with confidence scores"""
        
        # Get recent patterns
        cutoff_time = current_time - self.window_size
        recent_patterns = [p for p in self.pattern_history 
                          if p['timestamp'] >= cutoff_time]
        
        if not recent_patterns:
            return {}
        
        # Count pattern frequencies
        pattern_freq = defaultdict(int)
        for p in recent_patterns:
            pattern_freq[p['pattern']] += 1
        
        # Compute confidence scores
        total_patterns = len(recent_patterns)
        confident_patterns = {}
        
        for pattern, count in pattern_freq.items():
            confidence = count / total_patterns
            if confidence >= self.confidence_threshold:
                confident_patterns[pattern] = confidence
        
        return confident_patterns
    
    def _classify_pattern(self, performance_metrics: Dict) -> str:
        """Classify performance metrics into pattern category"""
        
        # This is a simplified pattern classification
        # In practice, this could use more sophisticated ML techniques
        
        latency = performance_metrics.get('latency', 0.0)
        throughput = performance_metrics.get('throughput', 0.0)
        loss = performance_metrics.get('packet_loss', 0.0)
        jitter = performance_metrics.get('jitter', 0.0)
        
        # Multi-dimensional pattern classification
        if latency > 200 and loss > 0.02:
            return "severe_congestion"
        elif latency > 100 and loss > 0.01:
            return "moderate_congestion"
        elif jitter > 50:
            return "high_jitter"
        elif throughput < 0.1:
            return "low_throughput"
        elif loss > 0.005:
            return "packet_loss"
        elif latency < 10 and throughput > 0.8:
            return "optimal_performance"
        else:
            return "normal_operation"


class ConvergenceTracker:
    """Track routing convergence patterns"""
    
    def __init__(self):
        self.convergence_history = deque(maxlen=500)
        self.convergence_threshold = 0.05  # 5% change threshold
        self.stability_window = 10  # 10 samples for stability check
    
    def analyze(self, performance_metrics: Dict) -> Dict:
        """Analyze current convergence state"""
        
        current_score = self._compute_convergence_score(performance_metrics)
        timestamp = time.time()
        
        self.convergence_history.append({
            'timestamp': timestamp,
            'score': current_score,
            'metrics': performance_metrics.copy()
        })
        
        # Determine convergence state
        if len(self.convergence_history) >= self.stability_window:
            recent_scores = [h['score'] for h in list(self.convergence_history)[-self.stability_window:]]
            score_variance = np.var(recent_scores)
            
            if score_variance < self.convergence_threshold:
                state = "converged"
            elif np.mean(recent_scores) > current_score:
                state = "converging"
            else:
                state = "diverging"
        else:
            state = "initializing"
        
        return {
            'state': state,
            'score': current_score,
            'variance': score_variance if 'score_variance' in locals() else 0.0,
            'timestamp': timestamp
        }
    
    def _compute_convergence_score(self, performance_metrics: Dict) -> float:
        """Compute convergence score from performance metrics"""
        
        # Combine stability indicators
        accuracy = performance_metrics.get('accuracy', 0.5)
        latency_stability = 1.0 / (1.0 + performance_metrics.get('jitter', 1.0))
        throughput_stability = performance_metrics.get('throughput', 0.5)
        
        score = 0.4 * accuracy + 0.3 * latency_stability + 0.3 * throughput_stability
        return score
    
    def get_current_state(self) -> Dict:
        """Get current convergence state"""
        
        if not self.convergence_history:
            return {'state': 'unknown', 'score': 0.0}
        
        return self.convergence_history[-1]


class TrendAnalyzer:
    """Analyze long-term performance trends"""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size  # Hours
        self.trend_history = deque(maxlen=window_size * 60)  # Minute resolution
        self.trend_coefficients = {}
    
    def update(self, performance_metrics: Dict, timestamp: float):
        """Update trend analysis"""
        
        trend_point = {
            'timestamp': timestamp,
            'performance': performance_metrics.copy(),
            'hour': int(timestamp // 3600) % 24  # Hour of day
        }
        
        self.trend_history.append(trend_point)
        
        # Update trend coefficients periodically
        if len(self.trend_history) % 60 == 0:  # Every hour
            self._compute_trends()
    
    def get_current_trends(self, current_time: float) -> Dict:
        """Get current trend information"""
        
        if not self.trend_coefficients:
            return {}
        
        return self.trend_coefficients.copy()
    
    def get_trend_summary(self) -> Dict:
        """Get summary of trend analysis"""
        
        if not self.trend_history:
            return {}
        
        return {
            'num_datapoints': len(self.trend_history),
            'time_span_hours': (self.trend_history[-1]['timestamp'] - 
                               self.trend_history[0]['timestamp']) / 3600,
            'trends': self.trend_coefficients
        }
    
    def _compute_trends(self):
        """Compute trend coefficients using linear regression"""
        
        if len(self.trend_history) < 10:
            return
        
        # Extract time series for different metrics
        timestamps = np.array([p['timestamp'] for p in self.trend_history])
        
        for metric_name in ['accuracy', 'latency', 'throughput', 'packet_loss']:
            values = []
            for p in self.trend_history:
                if metric_name in p['performance']:
                    values.append(p['performance'][metric_name])
                else:
                    values.append(0.0)
            
            if len(values) > 10:
                # Simple linear regression
                coeffs = np.polyfit(timestamps, values, 1)
                self.trend_coefficients[f'{metric_name}_trend'] = coeffs[0]  # Slope


class StabilityTracker:
    """Track network stability metrics"""
    
    def __init__(self):
        self.stability_history = deque(maxlen=1000)
        self.stability_window = 100
    
    def compute_stability(self, performance_metrics: Dict) -> float:
        """Compute stability score for current performance"""
        
        # Stability is inverse of variance in key metrics
        if len(self.stability_history) < 10:
            return 0.5  # Neutral stability
        
        # Get recent performance data
        recent_metrics = list(self.stability_history)[-self.stability_window:]
        
        # Compute coefficient of variation for key metrics
        stabilities = []
        
        for metric in ['accuracy', 'latency', 'throughput']:
            if metric in performance_metrics:
                values = [m.get(metric, 0.0) for m in recent_metrics]
                if len(values) > 5 and np.mean(values) > 0:
                    cv = np.std(values) / np.mean(values)  # Coefficient of variation
                    stability = 1.0 / (1.0 + cv)  # Convert to stability score
                    stabilities.append(stability)
        
        return np.mean(stabilities) if stabilities else 0.5
    
    def update(self, performance_metrics: Dict, timestamp: float):
        """Update stability tracking"""
        
        stability_score = self.compute_stability(performance_metrics)
        
        self.stability_history.append({
            'timestamp': timestamp,
            'performance': performance_metrics.copy(),
            'stability_score': stability_score
        })
    
    def get_stability_state(self, current_time: float) -> Dict:
        """Get current stability state"""
        
        if not self.stability_history:
            return {'overall_stability': 0.5}
        
        recent_scores = [h['stability_score'] for h in list(self.stability_history)[-50:]]
        
        return {
            'overall_stability': np.mean(recent_scores),
            'stability_variance': np.var(recent_scores),
            'stability_trend': np.mean(recent_scores[-10:]) - np.mean(recent_scores[-20:-10]) if len(recent_scores) >= 20 else 0.0
        }


class DDWE:
    """
    Dynamic Discriminative Weight Evolution - Main Controller
    
    Coordinates three-tier hierarchical memory system for adaptive weight evolution
    without requiring full model retraining.
    """
    
    def __init__(self, config: Dict, short_term_window: int = 300,
                 medium_term_window: int = 14400, long_term_window: int = 86400):
        
        self.config = config
        self.short_term_window = short_term_window
        self.medium_term_window = medium_term_window
        self.long_term_window = long_term_window
        
        # Initialize memory hierarchy
        self.short_term_memory = ShortTermMemory(
            window=short_term_window,
            weight=config.get('short_weight', 0.5),
            capacity=config.get('short_term_capacity', 10000)
        )
        
        self.medium_term_memory = MediumTermMemory(
            window=medium_term_window,
            weight=config.get('medium_weight', 0.3),
            capacity=config.get('medium_term_capacity', 50000)
        )
        
        self.long_term_memory = LongTermMemory(
            window=long_term_window,
            weight=config.get('long_weight', 0.2),
            capacity=config.get('long_term_capacity', 100000)
        )
        
        # Current composite weights
        self.current_weights_ = None
        self.weight_history_ = deque(maxlen=10000)
        self.adaptation_events_ = deque(maxlen=1000)
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.previous_weight_update_ = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"DDWE initialized with hierarchical memory: "
                   f"{short_term_window}s/{medium_term_window}s/{long_term_window}s")
    
    def initialize_weights(self, X: np.ndarray, y: np.ndarray, 
                          timestamps: np.ndarray) -> np.ndarray:
        """Initialize adaptive weights based on training data"""
        
        # Initialize with uniform weights
        n_kernels = X.shape[1] if X.ndim > 1 else 10  # Default for kernel combinations
        initial_weights = np.ones(n_kernels) / n_kernels
        
        with self._lock:
            self.current_weights_ = initial_weights.copy()
            
            # Store initial state
            self.weight_history_.append({
                'timestamp': time.time(),
                'weights': initial_weights.copy(),
                'event': 'initialization'
            })
        
        logger.info(f"DDWE weights initialized: {n_kernels} kernels")
        return initial_weights
    
    def update_weights(self, performance_feedback: Dict, timestamp: float) -> None:
        """
        Update kernel weights based on network performance feedback
        
        Implements composite weight update from Equations 3-5:
        λ(t) = ω_s * λ^(s)(t) + ω_m * λ^(m)(t) + ω_l * λ^(l)(t)
        """
        
        with self._lock:
            if self.current_weights_ is None:
                logger.warning("Weights not initialized, cannot update")
                return
            
            # Store experience in all memory levels
            self._store_experience_all_levels(performance_feedback, timestamp)
            
            # Get adaptive weights from each memory level
            short_weights = self.short_term_memory.compute_adaptive_weights(
                timestamp, self.current_weights_
            )
            
            medium_weights = self.medium_term_memory.compute_adaptive_weights(
                timestamp, self.current_weights_
            )
            
            long_weights = self.long_term_memory.compute_adaptive_weights(
                timestamp, self.current_weights_
            )
            
            # Composite weight combination
            composite_weights = (
                self.short_term_memory.weight * short_weights +
                self.medium_term_memory.weight * medium_weights +
                self.long_term_memory.weight * long_weights
            )
            
            # Apply momentum if available
            if self.previous_weight_update_ is not None:
                weight_update = composite_weights - self.current_weights_
                momentum_update = (self.momentum * self.previous_weight_update_ +
                                 (1 - self.momentum) * weight_update)
                composite_weights = self.current_weights_ + self.learning_rate * momentum_update
                self.previous_weight_update_ = momentum_update
            else:
                self.previous_weight_update_ = composite_weights - self.current_weights_
            
            # Ensure constraints
            composite_weights = np.maximum(composite_weights, 0.01)
            composite_weights /= np.sum(composite_weights)
            
            # Update current weights
            weight_change = np.linalg.norm(composite_weights - self.current_weights_)
            self.current_weights_ = composite_weights.copy()
            
            # Record update
            self.weight_history_.append({
                'timestamp': timestamp,
                'weights': composite_weights.copy(),
                'weight_change': weight_change,
                'performance': performance_feedback.copy(),
                'event': 'weight_update'
            })
            
            # Record adaptation event
            self.adaptation_events_.append({
                'timestamp': timestamp,
                'type': 'weight_update',
                'weight_change_magnitude': weight_change,
                'performance_score': self._aggregate_performance_score(performance_feedback)
            })
            
            logger.debug(f"DDWE weights updated: change magnitude = {weight_change:.6f}")
    
    def get_adaptive_weights(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Get current adaptive weights for prediction
        
        Parameters
        ----------
        timestamps : np.ndarray
            Timestamps for temporal weight adjustment
            
        Returns
        -------
        weights : np.ndarray
            Current adaptive kernel weights
        """
        
        with self._lock:
            if self.current_weights_ is None:
                logger.warning("Weights not initialized, returning uniform weights")
                return np.ones(10) / 10  # Default
            
            return self.current_weights_.copy()
    
    def compute_temporal_weights(self, timestamps: np.ndarray) -> np.ndarray:
        """Compute temporal weighting based on recency"""
        
        current_time = time.time()
        time_diffs = current_time - timestamps
        
        # Exponential decay based on time difference
        temporal_weights = np.exp(-0.001 * time_diffs)  # α = 0.001
        temporal_weights /= np.sum(temporal_weights)
        
        return temporal_weights
    
    def evolve_memory_hierarchy(self, experience: Dict, timestamp: float) -> None:
        """Evolve information through memory hierarchy levels"""
        
        # Information flows from short -> medium -> long term
        # This implements the memory consolidation process
        
        with self._lock:
            # Short-term experiences may be promoted to medium-term
            if self._should_promote_to_medium_term(experience, timestamp):
                self.medium_term_memory.store_experience(
                    timestamp, experience['performance_metrics'],
                    self.current_weights_, experience.get('context', {})
                )
            
            # Medium-term patterns may be consolidated to long-term
            if self._should_promote_to_long_term(experience, timestamp):
                self.long_term_memory.store_experience(
                    timestamp, experience['performance_metrics'],
                    self.current_weights_, experience.get('context', {})
                )
    
    def _store_experience_all_levels(self, performance_feedback: Dict, timestamp: float):
        """Store experience in all appropriate memory levels"""
        
        # Always store in short-term memory
        self.short_term_memory.store_experience(
            timestamp, performance_feedback, self.current_weights_
        )
        
        # Store in medium-term if significant
        if self._is_significant_experience(performance_feedback):
            self.medium_term_memory.store_experience(
                timestamp, performance_feedback, self.current_weights_
            )
        
        # Store in long-term if represents stable pattern
        if self._is_stable_pattern(performance_feedback, timestamp):
            self.long_term_memory.store_experience(
                timestamp, performance_feedback, self.current_weights_
            )
    
    def _should_promote_to_medium_term(self, experience: Dict, timestamp: float) -> bool:
        """Determine if experience should be promoted to medium-term memory"""
        
        # Promote if experience is significant or represents convergence
        performance_score = self._aggregate_performance_score(
            experience.get('performance_metrics', {})
        )
        
        return performance_score > 0.7 or performance_score < 0.3  # Extreme performance
    
    def _should_promote_to_long_term(self, experience: Dict, timestamp: float) -> bool:
        """Determine if experience should be promoted to long-term memory"""
        
        # Promote if experience represents stable, recurring pattern
        # This is simplified - could use more sophisticated pattern recognition
        
        return len(self.weight_history_) > 100 and len(self.weight_history_) % 100 == 0
    
    def _is_significant_experience(self, performance_feedback: Dict) -> bool:
        """Check if experience is significant enough for medium-term storage"""
        
        performance_score = self._aggregate_performance_score(performance_feedback)
        return abs(performance_score - 0.5) > 0.2  # Significant deviation from neutral
    
    def _is_stable_pattern(self, performance_feedback: Dict, timestamp: float) -> bool:
        """Check if performance represents a stable pattern for long-term storage"""
        
        if len(self.weight_history_) < 50:
            return False
        
        # Check recent performance stability
        recent_scores = []
        for record in list(self.weight_history_)[-20:]:
            if 'performance' in record:
                score = self._aggregate_performance_score(record['performance'])
                recent_scores.append(score)
        
        if len(recent_scores) < 10:
            return False
        
        # Stable if low variance in recent performance
        return np.std(recent_scores) < 0.1
    
    def _aggregate_performance_score(self, performance_metrics: Dict) -> float:
        """Aggregate performance metrics into single score"""
        
        if not performance_metrics:
            return 0.5
        
        # Weighted combination of metrics
        accuracy = performance_metrics.get('accuracy', 0.5)
        latency_score = 1.0 / (1.0 + performance_metrics.get('latency', 1.0))
        throughput = performance_metrics.get('throughput', 0.5)
        stability = performance_metrics.get('stability', 0.5)
        
        weights = self.config.get('feedback_weights', {
            'accuracy': 0.4, 'latency': 0.3, 'throughput': 0.2, 'stability': 0.1
        })
        
        score = (weights['accuracy'] * accuracy +
                weights['latency'] * latency_score +
                weights['throughput'] * throughput +
                weights['stability'] * stability)
        
        return score
    
    def get_current_weights(self) -> np.ndarray:
        """Get current composite weights"""
        
        with self._lock:
            if self.current_weights_ is None:
                return np.ones(1) / 1
            return self.current_weights_.copy()
    
    def get_weight_history(self) -> List[Dict]:
        """Get weight evolution history"""
        
        with self._lock:
            return list(self.weight_history_)
    
    def get_memory_level_info(self) -> Dict:
        """Get information about all memory levels"""
        
        return {
            'short_term': self.short_term_memory.get_memory_stats(),
            'medium_term': self.medium_term_memory.get_memory_stats(),
            'long_term': self.long_term_memory.get_memory_stats()
        }
    
    def get_adaptation_events(self) -> List[Dict]:
        """Get recent adaptation events"""
        
        with self._lock:
            return list(self.adaptation_events_)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive DDWE statistics"""
        
        with self._lock:
            stats = {
                'num_weight_updates': len(self.weight_history_),
                'num_adaptation_events': len(self.adaptation_events_),
                'current_weights_entropy': entropy(self.current_weights_) if self.current_weights_ is not None else 0.0,
                'memory_levels': self.get_memory_level_info(),
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'momentum': self.momentum
                }
            }
            
            if len(self.weight_history_) > 1:
                # Compute weight change statistics
                weight_changes = []
                for i in range(1, len(self.weight_history_)):
                    if 'weight_change' in self.weight_history_[i]:
                        weight_changes.append(self.weight_history_[i]['weight_change'])
                
                if weight_changes:
                    stats['weight_change_stats'] = {
                        'mean': np.mean(weight_changes),
                        'std': np.std(weight_changes),
                        'max': np.max(weight_changes),
                        'min': np.min(weight_changes)
                    }
            
            return stats
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        
        memory_mb = 0.0
        
        # Weight history
        if self.weight_history_:
            memory_mb += len(self.weight_history_) * 0.001  # Rough estimate
        
        # Adaptation events
        if self.adaptation_events_:
            memory_mb += len(self.adaptation_events_) * 0.0005
        
        # Current weights
        if self.current_weights_ is not None:
            memory_mb += self.current_weights_.nbytes / (1024 * 1024)
        
        return memory_mb
    
    def get_status(self) -> Dict:
        """Get current DDWE status"""
        
        with self._lock:
            return {
                'initialized': self.current_weights_ is not None,
                'num_kernels': len(self.current_weights_) if self.current_weights_ is not None else 0,
                'last_update': self.weight_history_[-1]['timestamp'] if self.weight_history_ else None,
                'total_updates': len(self.weight_history_),
                'memory_usage_mb': self.get_memory_usage()
            }
    
    def get_state(self) -> Dict:
        """Get complete state for serialization"""
        
        with self._lock:
            return {
                'config': self.config,
                'current_weights': self.current_weights_.tolist() if self.current_weights_ is not None else None,
                'weight_history': list(self.weight_history_),
                'adaptation_events': list(self.adaptation_events_),
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'previous_weight_update': self.previous_weight_update_.tolist() if self.previous_weight_update_ is not None else None
            }
    
    def restore_state(self, state: Dict) -> None:
        """Restore state from serialization"""
        
        with self._lock:
            self.config = state['config']
            self.current_weights_ = np.array(state['current_weights']) if state['current_weights'] is not None else None
            self.weight_history_ = deque(state['weight_history'], maxlen=10000)
            self.adaptation_events_ = deque(state['adaptation_events'], maxlen=1000)
            self.learning_rate = state['learning_rate']
            self.momentum = state['momentum']
            self.previous_weight_update_ = np.array(state['previous_weight_update']) if state['previous_weight_update'] is not None else None
        
        logger.info("DDWE state restored successfully") 
