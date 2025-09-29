# File: hyperpath_svm/core/support_vector_manager.py
"""
Support Vector Manager for Continuous Learning

This module implements intelligent support vector management with:
- Dynamic support vector selection and removal
- Importance-based ranking with recency weighting
- Continuous learning without full retraining
- Memory-efficient support vector storage
- Novelty detection for new support vectors

Key Innovation: Maintains optimal support vector set while learning continuously
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.cluster import MiniBatchKMeans
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class SupportVector:
    """Represents a single support vector with metadata"""
    
    # Core support vector data
    feature_vector: np.ndarray
    label: int
    dual_coefficient: float  # α_i from SVM dual formulation
    
    # Temporal information
    timestamp: float
    age: float = field(init=False)
    
    # Importance metrics
    importance_score: float = field(default=0.0)
    usage_count: int = field(default=0)
    last_used: float = field(default_factory=time.time)
    
    # Learning metadata
    prediction_accuracy: float = field(default=0.5)
    contribution_to_margin: float = field(default=0.0)
    novelty_score: float = field(default=0.0)
    
    # Geometric properties
    distance_to_hyperplane: float = field(default=0.0)
    local_density: float = field(default=0.0)
    cluster_id: int = field(default=-1)
    
    # Relationships
    similar_vectors: List[int] = field(default_factory=list)  # Indices of similar SVs
    conflicting_vectors: List[int] = field(default_factory=list)  # Indices of conflicting SVs
    
    def __post_init__(self):
        """Initialize computed fields"""
        current_time = time.time()
        self.age = current_time - self.timestamp
        if not hasattr(self, 'last_used'):
            self.last_used = current_time
    
    def update_age(self, current_time: float = None):
        """Update age based on current time"""
        if current_time is None:
            current_time = time.time()
        self.age = current_time - self.timestamp
    
    def record_usage(self, current_time: float = None):
        """Record usage of this support vector"""
        if current_time is None:
            current_time = time.time()
        self.usage_count += 1
        self.last_used = current_time
    
    def compute_importance(self, alpha_weight: float = 0.4, 
                          recency_weight: float = 0.3,
                          diversity_weight: float = 0.3) -> float:
        """
        Compute overall importance score
        
        Combines three factors:
        - |α|: Magnitude of dual coefficient (SVM importance)
        - Recency: How recently the SV was used/created
        - Diversity: How unique the SV is in feature space
        """
        
        # Alpha magnitude component (higher |α| = more important)
        alpha_component = abs(self.dual_coefficient)
        
        # Recency component (more recent = more important)
        time_since_used = time.time() - self.last_used
        recency_component = np.exp(-0.0001 * time_since_used)  # Slow decay
        
        # Diversity component (based on novelty and local density)
        diversity_component = self.novelty_score * (1.0 / (1.0 + self.local_density))
        
        # Weighted combination
        self.importance_score = (alpha_weight * alpha_component + 
                               recency_weight * recency_component +
                               diversity_weight * diversity_component)
        
        return self.importance_score


class SupportVectorClusterer:
    """
    Clusters support vectors for efficient organization and retrieval
    Uses online clustering to handle streaming support vectors
    """
    
    def __init__(self, n_clusters: int = 50, batch_size: int = 100):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        
        # Online clustering model
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42,
            max_iter=100
        )
        
        # Cluster statistics
        self.cluster_centers_ = None
        self.cluster_sizes_ = np.zeros(n_clusters)
        self.cluster_importance_ = np.zeros(n_clusters)
        self.is_fitted_ = False
        
        # Support vector assignments
        self.sv_cluster_assignments_ = {}  # sv_index -> cluster_id
        
        logger.debug(f"SupportVectorClusterer initialized: {n_clusters} clusters")
    
    def fit_partial(self, support_vectors: List[SupportVector], 
                   sv_indices: List[int]) -> None:
        """Incrementally fit clustering model"""
        
        if not support_vectors:
            return
        
        # Extract feature vectors
        X = np.array([sv.feature_vector for sv in support_vectors])
        
        # Fit or update clustering
        if not self.is_fitted_:
            self.clusterer.fit(X)
            self.is_fitted_ = True
        else:
            self.clusterer.partial_fit(X)
        
        # Update cluster assignments
        cluster_labels = self.clusterer.predict(X)
        for sv_idx, cluster_id in zip(sv_indices, cluster_labels):
            self.sv_cluster_assignments_[sv_idx] = int(cluster_id)
        
        # Update cluster statistics
        self._update_cluster_statistics(support_vectors, cluster_labels)
    
    def _update_cluster_statistics(self, support_vectors: List[SupportVector],
                                 cluster_labels: np.ndarray) -> None:
        """Update cluster size and importance statistics"""
        
        # Reset cluster sizes
        self.cluster_sizes_.fill(0)
        self.cluster_importance_.fill(0)
        
        # Count cluster members and compute importance
        for sv, cluster_id in zip(support_vectors, cluster_labels):
            self.cluster_sizes_[cluster_id] += 1
            self.cluster_importance_[cluster_id] += sv.importance_score
        
        # Average importance per cluster
        for i in range(self.n_clusters):
            if self.cluster_sizes_[i] > 0:
                self.cluster_importance_[i] /= self.cluster_sizes_[i]
    
    def predict_cluster(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new feature vectors"""
        
        if not self.is_fitted_:
            return np.zeros(len(feature_vectors), dtype=int)
        
        return self.clusterer.predict(feature_vectors)
    
    def get_cluster_representatives(self, support_vectors: List[SupportVector],
                                 num_representatives: int = 5) -> List[int]:
        """
        Get representative support vectors from each cluster
        
        Returns indices of support vectors that best represent cluster diversity
        """
        
        if not self.is_fitted_ or not support_vectors:
            return []
        
        representatives = []
        
        for cluster_id in range(self.n_clusters):
            # Find support vectors in this cluster
            cluster_svs = [i for i, sv in enumerate(support_vectors)
                          if self.sv_cluster_assignments_.get(i, -1) == cluster_id]
            
            if cluster_svs:
                # Select representatives based on importance and diversity
                cluster_representatives = self._select_cluster_representatives(
                    support_vectors, cluster_svs, num_representatives
                )
                representatives.extend(cluster_representatives)
        
        return representatives
    
    def _select_cluster_representatives(self, support_vectors: List[SupportVector],
                                     cluster_svs: List[int],
                                     num_representatives: int) -> List[int]:
        """Select representative SVs from a single cluster"""
        
        if len(cluster_svs) <= num_representatives:
            return cluster_svs
        
        # Compute importance scores for cluster members
        importance_scores = [support_vectors[i].importance_score for i in cluster_svs]
        
        # Select top representatives by importance
        top_indices = np.argsort(importance_scores)[-num_representatives:]
        representatives = [cluster_svs[i] for i in top_indices]
        
        return representatives
    
    def get_cluster_statistics(self) -> Dict:
        """Get comprehensive cluster statistics"""
        
        return {
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted_,
            'cluster_sizes': self.cluster_sizes_.tolist(),
            'cluster_importance': self.cluster_importance_.tolist(),
            'total_assignments': len(self.sv_cluster_assignments_),
            'average_cluster_size': np.mean(self.cluster_sizes_),
            'cluster_balance': np.std(self.cluster_sizes_) / (np.mean(self.cluster_sizes_) + 1e-8)
        }


class NoveltyDetector:
    """
    Detects novel support vectors that provide new information
    Uses density-based and distance-based novelty measures
    """
    
    def __init__(self, novelty_threshold: float = 0.1, 
                 density_bandwidth: float = 0.1):
        self.novelty_threshold = novelty_threshold
        self.density_bandwidth = density_bandwidth
        
        # Existing support vector summaries for novelty detection
        self.sv_prototypes = []  # Representative vectors
        self.prototype_weights = []  # Importance weights
        
        logger.debug(f"NoveltyDetector initialized: threshold={novelty_threshold}")
    
    def update_prototypes(self, support_vectors: List[SupportVector]) -> None:
        """Update prototype vectors for novelty detection"""
        
        if not support_vectors:
            return
        
        # Extract feature vectors and importance weights
        feature_vectors = np.array([sv.feature_vector for sv in support_vectors])
        importance_weights = np.array([sv.importance_score for sv in support_vectors])
        
        # Use clustering to create prototypes (simplified approach)
        if len(support_vectors) > 20:
            # Create prototypes using weighted k-means
            n_prototypes = min(20, len(support_vectors) // 2)
            
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_prototypes, random_state=42)
                kmeans.fit(feature_vectors, sample_weight=importance_weights)
                
                self.sv_prototypes = kmeans.cluster_centers_
                
                # Compute prototype weights (average importance in each cluster)
                labels = kmeans.labels_
                prototype_weights = []
                for i in range(n_prototypes):
                    cluster_mask = labels == i
                    if np.any(cluster_mask):
                        avg_importance = np.mean(importance_weights[cluster_mask])
                    else:
                        avg_importance = 0.0
                    prototype_weights.append(avg_importance)
                
                self.prototype_weights = np.array(prototype_weights)
                
            except Exception as e:
                logger.warning(f"Prototype clustering failed: {e}")
                # Fallback: use top important vectors as prototypes
                n_prototypes = min(10, len(support_vectors))
                top_indices = np.argsort(importance_weights)[-n_prototypes:]
                self.sv_prototypes = feature_vectors[top_indices]
                self.prototype_weights = importance_weights[top_indices]
        
        else:
            # For small sets, use all vectors as prototypes
            self.sv_prototypes = feature_vectors
            self.prototype_weights = importance_weights
    
    def compute_novelty(self, candidate_vectors: np.ndarray) -> np.ndarray:
        """
        Compute novelty scores for candidate support vectors
        
        Higher scores indicate more novel (less similar to existing SVs)
        
        Parameters
        ----------
        candidate_vectors : np.ndarray, shape (n_candidates, n_features)
            Candidate feature vectors
            
        Returns  
        -------
        novelty_scores : np.ndarray, shape (n_candidates,)
            Novelty scores [0, 1] where 1 = completely novel
        """
        
        if len(self.sv_prototypes) == 0:
            # No existing prototypes - all candidates are novel
            return np.ones(len(candidate_vectors))
        
        candidate_vectors = np.atleast_2d(candidate_vectors)
        novelty_scores = np.zeros(len(candidate_vectors))
        
        for i, candidate in enumerate(candidate_vectors):
            novelty_scores[i] = self._compute_single_novelty(candidate)
        
        return novelty_scores
    
    def _compute_single_novelty(self, candidate_vector: np.ndarray) -> float:
        """Compute novelty score for a single candidate vector"""
        
        # Compute distances to all prototypes
        distances = np.linalg.norm(
            np.array(self.sv_prototypes) - candidate_vector[np.newaxis, :], 
            axis=1
        )
        
        # Weight distances by prototype importance
        weighted_distances = distances * self.prototype_weights
        
        # Novelty is based on minimum weighted distance
        min_weighted_distance = np.min(weighted_distances)
        
        # Convert distance to novelty score using sigmoid
        novelty_score = expit(10.0 * (min_weighted_distance - self.novelty_threshold))
        
        return float(novelty_score)
    
    def is_novel(self, candidate_vector: np.ndarray) -> bool:
        """Check if candidate vector is sufficiently novel"""
        
        novelty_score = self.compute_novelty(candidate_vector.reshape(1, -1))[0]
        return novelty_score >= self.novelty_threshold
    
    def get_statistics(self) -> Dict:
        """Get novelty detection statistics"""
        
        return {
            'num_prototypes': len(self.sv_prototypes),
            'novelty_threshold': self.novelty_threshold,
            'density_bandwidth': self.density_bandwidth,
            'prototype_importance_mean': np.mean(self.prototype_weights) if len(self.prototype_weights) > 0 else 0.0,
            'prototype_importance_std': np.std(self.prototype_weights) if len(self.prototype_weights) > 0 else 0.0
        }


def expit(x):
    """Sigmoid function (logistic function)"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))


class SupportVectorManager:
    """
    Main Support Vector Manager
    
    Manages the dynamic support vector set for continuous learning:
    - Adds novel support vectors
    - Removes redundant/outdated support vectors  
    - Maintains optimal set size and diversity
    - Provides importance-based ranking
    """
    
    def __init__(self, config: Dict, max_support_vectors: int = 5000):
        
        self.config = config
        self.max_support_vectors = max_support_vectors
        
        # Core support vector storage
        self.support_vectors: List[SupportVector] = []
        self.sv_index_map: Dict[int, int] = {}  # original_index -> current_index
        
        # Configuration parameters
        self.novelty_threshold = config.get('novelty_threshold', 0.1)
        self.redundancy_threshold = config.get('redundancy_threshold', 0.95)
        self.decay_rate = config.get('decay_rate', 0.001)
        
        # Importance weights for ranking
        self.importance_weights = config.get('importance_weights', {
            'alpha_magnitude': 0.4,
            'recency': 0.3,
            'diversity': 0.3
        })
        
        # Management components
        self.clusterer = SupportVectorClusterer(
            n_clusters=min(50, max_support_vectors // 10),
            batch_size=100
        )
        
        self.novelty_detector = NoveltyDetector(
            novelty_threshold=self.novelty_threshold,
            density_bandwidth=0.1
        )
        
        # Update and maintenance scheduling
        self.update_frequency = config.get('update_frequency', 100)
        self.last_update_count = 0
        self.last_maintenance_time = time.time()
        self.maintenance_interval = 3600  # 1 hour
        
        # Statistics tracking
        self.stats = {
            'total_additions': 0,
            'total_removals': 0,
            'novel_additions': 0,
            'redundant_removals': 0,
            'maintenance_operations': 0,
            'average_importance': 0.0,
            'diversity_score': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Recent modifications tracking
        self.recent_modifications = deque(maxlen=1000)
        
        logger.info(f"SupportVectorManager initialized: max_svs={max_support_vectors}, "
                   f"novelty_threshold={self.novelty_threshold}")
    
    def initialize_support_vectors(self, feature_vectors: np.ndarray,
                                 original_indices: np.ndarray,
                                 dual_coefficients: np.ndarray,
                                 timestamps: np.ndarray) -> None:
        """
        Initialize support vector set from SVM training
        
        Parameters
        ----------
        feature_vectors : np.ndarray
            Support vector feature matrices
        original_indices : np.ndarray  
            Original indices in training set
        dual_coefficients : np.ndarray
            SVM dual coefficients (alphas)
        timestamps : np.ndarray
            Timestamps for each support vector
        """
        
        logger.info(f"Initializing {len(feature_vectors)} support vectors")
        
        with self._lock:
            self.support_vectors.clear()
            self.sv_index_map.clear()
            
            current_time = time.time()
            
            for i, (features, orig_idx, alpha, timestamp) in enumerate(
                zip(feature_vectors, original_indices, dual_coefficients, timestamps)):
                
                # Create support vector object
                sv = SupportVector(
                    feature_vector=features.copy(),
                    label=1 if alpha > 0 else -1,  # Infer label from alpha sign
                    dual_coefficient=float(alpha),
                    timestamp=float(timestamp)
                )
                
                # Compute initial importance
                sv.compute_importance(**self.importance_weights)
                
                # Add to collection
                self.support_vectors.append(sv)
                self.sv_index_map[int(orig_idx)] = i
                
                # Update statistics
                self.stats['total_additions'] += 1
            
            # Update components with initial set
            self._update_components()
            
            # Compute initial statistics
            self._compute_statistics()
            
            logger.info(f"Support vector initialization completed: "
                       f"{len(self.support_vectors)} vectors, "
                       f"avg importance: {self.stats['average_importance']:.3f}")
    
    def add_support_vectors(self, feature_vectors: np.ndarray, 
                           labels: np.ndarray, timestamps: np.ndarray,
                           dual_coefficients: Optional[np.ndarray] = None) -> List[int]:
        """
        Add new support vectors with novelty filtering
        
        Parameters
        ----------
        feature_vectors : np.ndarray
            New candidate feature vectors
        labels : np.ndarray
            Labels for new vectors
        timestamps : np.ndarray
            Timestamps for new vectors
        dual_coefficients : np.ndarray, optional
            Dual coefficients if available
            
        Returns
        -------
        added_indices : List[int]
            Indices of vectors that were actually added
        """
        
        if len(feature_vectors) == 0:
            return []
        
        with self._lock:
            added_indices = []
            current_time = time.time()
            
            # Compute novelty scores for candidates
            novelty_scores = self.novelty_detector.compute_novelty(feature_vectors)
            
            for i, (features, label, timestamp, novelty) in enumerate(
                zip(feature_vectors, labels, timestamps, novelty_scores)):
                
                # Check if novel enough to add
                if novelty >= self.novelty_threshold:
                    
                    # Create support vector
                    alpha = dual_coefficients[i] if dual_coefficients is not None else 1.0
                    
                    sv = SupportVector(
                        feature_vector=features.copy(),
                        label=int(label),
                        dual_coefficient=float(alpha),
                        timestamp=float(timestamp),
                        novelty_score=float(novelty)
                    )
                    
                    # Compute importance
                    sv.compute_importance(**self.importance_weights)
                    
                    # Check capacity and make room if needed
                    if len(self.support_vectors) >= self.max_support_vectors:
                        self._make_room_for_new_sv()
                    
                    # Add support vector
                    new_index = len(self.support_vectors)
                    self.support_vectors.append(sv)
                    added_indices.append(new_index)
                    
                    # Update statistics
                    self.stats['total_additions'] += 1
                    self.stats['novel_additions'] += 1
                    
                    # Record modification
                    self.recent_modifications.append({
                        'type': 'addition',
                        'timestamp': current_time,
                        'index': new_index,
                        'novelty_score': novelty,
                        'importance': sv.importance_score
                    })
            
            # Trigger component updates if enough changes
            self.last_update_count += len(added_indices)
            if self.last_update_count >= self.update_frequency:
                self._update_components()
                self.last_update_count = 0
            
            # Periodic maintenance
            if current_time - self.last_maintenance_time > self.maintenance_interval:
                self._perform_maintenance()
                self.last_maintenance_time = current_time
            
            logger.debug(f"Added {len(added_indices)} novel support vectors "
                        f"out of {len(feature_vectors)} candidates")
            
            return added_indices
    
    def remove_support_vectors(self, indices: List[int]) -> int:
        """
        Remove support vectors by indices
        
        Parameters
        ---------- 
        indices : List[int]
            Indices of support vectors to remove
            
        Returns
        -------
        removed_count : int
            Number of vectors actually removed
        """
        
        with self._lock:
            # Sort indices in reverse order for safe removal
            indices = sorted(set(indices), reverse=True)
            removed_count = 0
            current_time = time.time()
            
            for idx in indices:
                if 0 <= idx < len(self.support_vectors):
                    removed_sv = self.support_vectors.pop(idx)
                    removed_count += 1
                    
                    # Update statistics
                    self.stats['total_removals'] += 1
                    
                    # Record modification
                    self.recent_modifications.append({
                        'type': 'removal',
                        'timestamp': current_time,
                        'index': idx,
                        'importance': removed_sv.importance_score,
                        'age': removed_sv.age
                    })
            
            # Update index mappings after removal
            self._rebuild_index_mappings()
            
            logger.debug(f"Removed {removed_count} support vectors")
            
            return removed_count
    
    def update_support_vector_weights(self, performance_feedback: Dict,
                                    timestamp: float) -> None:
        """
        Update support vector importance based on performance feedback
        
        Parameters
        ----------
        performance_feedback : Dict
            Network performance metrics
        timestamp : float
            Current timestamp
        """
        
        with self._lock:
            current_time = time.time()
            
            # Update ages for all support vectors
            for sv in self.support_vectors:
                sv.update_age(current_time)
            
            # Apply temporal decay to importance scores
            decay_factor = np.exp(-self.decay_rate * (current_time - self.last_maintenance_time))
            
            for sv in self.support_vectors:
                # Apply decay
                sv.importance_score *= decay_factor
                
                # Recompute importance with new performance feedback
                # This is simplified - could use more sophisticated feedback integration
                performance_score = self._aggregate_performance_score(performance_feedback)
                
                # Boost importance for SVs that contributed to good performance
                if performance_score > 0.7:
                    sv.importance_score *= 1.1  # Small boost
                elif performance_score < 0.3:
                    sv.importance_score *= 0.9  # Small penalty
            
            # Record weight update
            self.recent_modifications.append({
                'type': 'weight_update',
                'timestamp': current_time,
                'performance_score': performance_score,
                'decay_factor': decay_factor,
                'num_svs': len(self.support_vectors)
            })
    
    def _make_room_for_new_sv(self) -> None:
        """Remove least important support vectors to make room"""
        
        if len(self.support_vectors) < self.max_support_vectors:
            return
        
        # Find least important support vectors
        importance_scores = [sv.importance_score for sv in self.support_vectors]
        
        # Remove bottom 10% or at least 1
        num_to_remove = max(1, len(self.support_vectors) // 10)
        removal_indices = np.argsort(importance_scores)[:num_to_remove]
        
        # Remove redundant support vectors
        self.remove_support_vectors(removal_indices.tolist())
        self.stats['redundant_removals'] += len(removal_indices)
        
        logger.debug(f"Made room by removing {len(removal_indices)} least important SVs")
    
    def _update_components(self) -> None:
        """Update clustering and novelty detection components"""
        
        if not self.support_vectors:
            return
        
        # Update clustering
        self.clusterer.fit_partial(
            self.support_vectors,
            list(range(len(self.support_vectors)))
        )
        
        # Update novelty detector prototypes
        self.novelty_detector.update_prototypes(self.support_vectors)
        
        # Compute cluster information for support vectors
        self._update_sv_cluster_info()
    
    def _update_sv_cluster_info(self) -> None:
        """Update cluster assignment and local density for support vectors"""
        
        if not self.clusterer.is_fitted_:
            return
        
        # Predict cluster assignments
        feature_vectors = np.array([sv.feature_vector for sv in self.support_vectors])
        cluster_assignments = self.clusterer.predict_cluster(feature_vectors)
        
        # Compute local densities
        distances = pairwise_distances(feature_vectors)
        
        for i, sv in enumerate(self.support_vectors):
            sv.cluster_id = int(cluster_assignments[i])
            
            # Local density: number of SVs within threshold distance
            neighbor_distances = distances[i]
            density_threshold = np.percentile(neighbor_distances, 20)  # 20th percentile
            sv.local_density = float(np.sum(neighbor_distances <= density_threshold))
            
            # Find similar and conflicting vectors
            similar_threshold = np.percentile(neighbor_distances, 10)  # Very similar
            similar_mask = neighbor_distances <= similar_threshold
            sv.similar_vectors = [j for j, sim in enumerate(similar_mask) 
                                if sim and j != i]
            
            # Conflicting vectors: same cluster but different labels
            same_cluster_mask = cluster_assignments == sv.cluster_id
            different_label_mask = np.array([other_sv.label != sv.label 
                                           for other_sv in self.support_vectors])
            conflict_mask = same_cluster_mask & different_label_mask
            sv.conflicting_vectors = [j for j, conf in enumerate(conflict_mask) if conf]
    
    def _perform_maintenance(self) -> None:
        """Perform periodic maintenance operations"""
        
        logger.debug("Performing support vector maintenance")
        
        with self._lock:
            # Remove highly redundant support vectors
            redundant_pairs = self._find_redundant_pairs()
            if redundant_pairs:
                removal_indices = self._select_redundant_removals(redundant_pairs)
                self.remove_support_vectors(removal_indices)
                self.stats['redundant_removals'] += len(removal_indices)
            
            # Update all importance scores
            for sv in self.support_vectors:
                sv.compute_importance(**self.importance_weights)
            
            # Recompute statistics
            self._compute_statistics()
            
            # Update components
            self._update_components()
            
            self.stats['maintenance_operations'] += 1
            
            logger.debug(f"Maintenance completed: {len(self.support_vectors)} SVs remaining")
    
    def _find_redundant_pairs(self) -> List[Tuple[int, int, float]]:
        """Find pairs of highly similar support vectors"""
        
        if len(self.support_vectors) < 2:
            return []
        
        redundant_pairs = []
        feature_vectors = np.array([sv.feature_vector for sv in self.support_vectors])
        
        # Compute pairwise similarities (using RBF kernel for efficiency)
        similarities = rbf_kernel(feature_vectors, gamma=0.1)
        
        # Find pairs above redundancy threshold
        n = len(self.support_vectors)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarities[i, j]
                
                if similarity >= self.redundancy_threshold:
                    # Additional check: same label
                    if self.support_vectors[i].label == self.support_vectors[j].label:
                        redundant_pairs.append((i, j, similarity))
        
        return redundant_pairs
    
    def _select_redundant_removals(self, redundant_pairs: List[Tuple[int, int, float]]) -> List[int]:
        """Select which vectors to remove from redundant pairs"""
        
        # For each redundant pair, remove the less important vector
        removal_set = set()
        
        for i, j, similarity in redundant_pairs:
            sv_i = self.support_vectors[i]
            sv_j = self.support_vectors[j]
            
            # Remove the less important one
            if sv_i.importance_score < sv_j.importance_score:
                removal_set.add(i)
            else:
                removal_set.add(j)
        
        return list(removal_set)
    
    def _rebuild_index_mappings(self) -> None:
        """Rebuild index mappings after removals"""
        
        # This is a simplified version - in practice might need more sophisticated mapping
        self.sv_index_map.clear()
        for i, sv in enumerate(self.support_vectors):
            # Use timestamp as a pseudo-original-index for mapping
            # In real implementation, would maintain proper original indices
            self.sv_index_map[int(sv.timestamp * 1000) % 100000] = i
    
    def _aggregate_performance_score(self, performance_feedback: Dict) -> float:
        """Aggregate performance feedback into single score"""
        
        if not performance_feedback:
            return 0.5
        
        # Weighted combination of performance metrics
        accuracy = performance_feedback.get('accuracy', 0.5)
        latency_score = 1.0 / (1.0 + performance_feedback.get('latency', 1.0))
        throughput = performance_feedback.get('throughput', 0.5)
        stability = performance_feedback.get('stability', 0.5)
        
        # Combine with equal weights (could be made configurable)
        score = 0.25 * accuracy + 0.25 * latency_score + 0.25 * throughput + 0.25 * stability
        
        return score
    
    def _compute_statistics(self) -> None:
        """Compute comprehensive statistics"""
        
        if not self.support_vectors:
            self.stats['average_importance'] = 0.0
            self.stats['diversity_score'] = 0.0
            return
        
        # Average importance
        importance_scores = [sv.importance_score for sv in self.support_vectors]
        self.stats['average_importance'] = float(np.mean(importance_scores))
        
        # Diversity score based on cluster distribution
        if self.clusterer.is_fitted_:
            cluster_counts = defaultdict(int)
            for sv in self.support_vectors:
                cluster_counts[sv.cluster_id] += 1
            
            cluster_sizes = np.array(list(cluster_counts.values()))
            if len(cluster_sizes) > 1:
                # Diversity = entropy of cluster distribution
                probabilities = cluster_sizes / np.sum(cluster_sizes)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                max_entropy = np.log(len(cluster_sizes))
                self.stats['diversity_score'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            else:
                self.stats['diversity_score'] = 0.0
        else:
            self.stats['diversity_score'] = 0.0
    
    def get_support_vector_importance(self) -> Dict[int, float]:
        """Get importance scores for all support vectors"""
        
        with self._lock:
            return {i: sv.importance_score for i, sv in enumerate(self.support_vectors)}
    
    def get_most_important_vectors(self, n: int = 10) -> List[Tuple[int, SupportVector]]:
        """Get n most important support vectors"""
        
        with self._lock:
            if not self.support_vectors:
                return []
            
            # Sort by importance
            indexed_svs = [(i, sv) for i, sv in enumerate(self.support_vectors)]
            indexed_svs.sort(key=lambda x: x[1].importance_score, reverse=True)
            
            return indexed_svs[:n]
    
    def get_cluster_representatives(self, max_per_cluster: int = 5) -> List[int]:
        """Get representative support vectors from each cluster"""
        
        with self._lock:
            return self.clusterer.get_cluster_representatives(
                self.support_vectors, max_per_cluster
            )
    
    def compute_novelty(self, candidate_vectors: np.ndarray) -> np.ndarray:
        """Compute novelty scores for candidate vectors"""
        
        return self.novelty_detector.compute_novelty(candidate_vectors)
    
    def get_recent_modifications(self) -> List[Dict]:
        """Get recent modification history"""
        
        with self._lock:
            return list(self.recent_modifications)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive management statistics"""
        
        with self._lock:
            base_stats = self.stats.copy()
            
            # Add current state information
            current_stats = {
                'current_num_svs': len(self.support_vectors),
                'capacity_utilization': len(self.support_vectors) / self.max_support_vectors,
                'avg_age_seconds': np.mean([sv.age for sv in self.support_vectors]) if self.support_vectors else 0.0,
                'avg_usage_count': np.mean([sv.usage_count for sv in self.support_vectors]) if self.support_vectors else 0.0,
                'label_distribution': self._get_label_distribution(),
                'cluster_stats': self.clusterer.get_cluster_statistics(),
                'novelty_stats': self.novelty_detector.get_statistics()
            }
            
            return {**base_stats, **current_stats}
    
    def _get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in support vector set"""
        
        label_counts = defaultdict(int)
        for sv in self.support_vectors:
            label_counts[sv.label] += 1
        
        return dict(label_counts)
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        
        memory_mb = 0.0
        
        # Support vectors
        if self.support_vectors:
            # Approximate: each SV has feature vector + metadata
            bytes_per_sv = self.support_vectors[0].feature_vector.nbytes + 200  # ~200 bytes metadata
            memory_mb += len(self.support_vectors) * bytes_per_sv / (1024 * 1024)
        
        # Recent modifications
        memory_mb += len(self.recent_modifications) * 0.001  # Rough estimate
        
        # Clusterer memory (cluster centers, etc.)
        if self.clusterer.is_fitted_:
            memory_mb += self.clusterer.n_clusters * 0.001  # Rough estimate
        
        return memory_mb
    
    def get_status(self) -> Dict:
        """Get current manager status"""
        
        with self._lock:
            return {
                'initialized': len(self.support_vectors) > 0,
                'num_support_vectors': len(self.support_vectors),
                'capacity_utilization': len(self.support_vectors) / self.max_support_vectors,
                'total_additions': self.stats['total_additions'],
                'total_removals': self.stats['total_removals'],
                'average_importance': self.stats['average_importance'],
                'diversity_score': self.stats['diversity_score'],
                'recent_modifications': len(self.recent_modifications),
                'memory_usage_mb': self.get_memory_usage(),
                'clusterer_fitted': self.clusterer.is_fitted_,
                'num_clusters': self.clusterer.n_clusters,
                'novelty_threshold': self.novelty_threshold
            }
    
    def get_state(self) -> Dict:
        """Get complete state for serialization"""
        
        with self._lock:
            # Convert support vectors to serializable format
            sv_data = []
            for sv in self.support_vectors:
                sv_dict = {
                    'feature_vector': sv.feature_vector.tolist(),
                    'label': sv.label,
                    'dual_coefficient': sv.dual_coefficient,
                    'timestamp': sv.timestamp,
                    'importance_score': sv.importance_score,
                    'usage_count': sv.usage_count,
                    'last_used': sv.last_used,
                    'prediction_accuracy': sv.prediction_accuracy,
                    'contribution_to_margin': sv.contribution_to_margin,
                    'novelty_score': sv.novelty_score,
                    'distance_to_hyperplane': sv.distance_to_hyperplane,
                    'local_density': sv.local_density,
                    'cluster_id': sv.cluster_id
                }
                sv_data.append(sv_dict)
            
            state = {
                'config': self.config,
                'max_support_vectors': self.max_support_vectors,
                'support_vectors': sv_data,
                'sv_index_map': self.sv_index_map,
                'novelty_threshold': self.novelty_threshold,
                'redundancy_threshold': self.redundancy_threshold,
                'decay_rate': self.decay_rate,
                'importance_weights': self.importance_weights,
                'stats': self.stats,
                'recent_modifications': list(self.recent_modifications),
                'last_maintenance_time': self.last_maintenance_time
            }
            
            return state
    
    def restore_state(self, state: Dict) -> None:
        """Restore state from serialization"""
        
        with self._lock:
            self.config = state['config']
            self.max_support_vectors = state['max_support_vectors']
            self.sv_index_map = state['sv_index_map']
            self.novelty_threshold = state['novelty_threshold']
            self.redundancy_threshold = state['redundancy_threshold']
            self.decay_rate = state['decay_rate']
            self.importance_weights = state['importance_weights']
            self.stats = state['stats']
            self.recent_modifications = deque(state['recent_modifications'], maxlen=1000)
            self.last_maintenance_time = state['last_maintenance_time']
            
            # Restore support vectors
            self.support_vectors = []
            for sv_dict in state['support_vectors']:
                sv = SupportVector(
                    feature_vector=np.array(sv_dict['feature_vector']),
                    label=sv_dict['label'],
                    dual_coefficient=sv_dict['dual_coefficient'],
                    timestamp=sv_dict['timestamp']
                )
                
                # Restore additional fields
                sv.importance_score = sv_dict['importance_score']
                sv.usage_count = sv_dict['usage_count']
                sv.last_used = sv_dict['last_used']
                sv.prediction_accuracy = sv_dict['prediction_accuracy']
                sv.contribution_to_margin = sv_dict['contribution_to_margin']
                sv.novelty_score = sv_dict['novelty_score']
                sv.distance_to_hyperplane = sv_dict['distance_to_hyperplane']
                sv.local_density = sv_dict['local_density']
                sv.cluster_id = sv_dict['cluster_id']
                
                self.support_vectors.append(sv)
            
            # Update components
            self._update_components()
        
        logger.info("SupportVectorManager state restored successfully")
    
    def clear(self) -> None:
        """Clear all support vectors and reset state"""
        
        with self._lock:
            self.support_vectors.clear()
            self.sv_index_map.clear()
            self.recent_modifications.clear()
            
            # Reset statistics
            self.stats = {
                'total_additions': 0,
                'total_removals': 0, 
                'novel_additions': 0,
                'redundant_removals': 0,
                'maintenance_operations': 0,
                'average_importance': 0.0,
                'diversity_score': 0.0
            }
        
        logger.info("SupportVectorManager cleared")
    
    def __len__(self) -> int:
        """Get number of support vectors"""
        return len(self.support_vectors)
    
    def __getitem__(self, index: int) -> SupportVector:
        """Get support vector by index"""
        return self.support_vectors[index]
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"SupportVectorManager(num_svs={len(self.support_vectors)}, "
               f"capacity={self.max_support_vectors}, "
               f"avg_importance={self.stats['average_importance']:.3f})") 
