# File: hyperpath_svm/core/memory_hierarchy.py
"""
Hierarchical Memory System Implementation

This module implements a multi-level memory hierarchy for efficient data management:
- Level-specific retention policies and access patterns
- Automatic data promotion and demotion between levels
- Compression and garbage collection for memory efficiency  
- Intelligent caching with performance-based replacement
- Memory usage optimization and monitoring

Key Innovation: Hierarchical memory management for large-scale continuous learning
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict
from enum import Enum
import threading
import pickle
import gzip
import hashlib
from abc import ABC, abstractmethod
import psutil
import gc

logger = logging.getLogger(__name__)


class AccessPattern(Enum):
    """Memory access patterns for different hierarchy levels"""
    FIFO = "fifo"              # First In, First Out
    LRU = "lru"                # Least Recently Used
    LFU = "lfu"                # Least Frequently Used
    IMPORTANCE_BASED = "importance_based"  # Based on importance scores
    TEMPORAL_DECAY = "temporal_decay"      # Time-based decay
    HYBRID = "hybrid"          # Combination of multiple patterns


@dataclass
class MemoryItem:
    """Represents a single item in memory hierarchy"""
    
    # Core data
    data: Any
    key: str
    
    # Temporal information
    created_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    
    # Access statistics
    access_count: int = field(default=0)
    modification_count: int = field(default=0)
    
    # Importance and priority
    importance_score: float = field(default=0.5)
    priority: int = field(default=0)
    
    # Memory management
    size_bytes: int = field(default=0)
    is_compressed: bool = field(default=False)
    compression_ratio: float = field(default=1.0)
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed fields"""
        if self.size_bytes == 0:
            self.size_bytes = self._estimate_size()
    
    def _estimate_size(self) -> int:
        """Estimate memory size of stored data"""
        try:
            import sys
            return sys.getsizeof(pickle.dumps(self.data))
        except Exception:
            # Fallback estimation
            if isinstance(self.data, np.ndarray):
                return self.data.nbytes
            elif isinstance(self.data, (list, tuple)):
                return len(self.data) * 64  # Rough estimate
            elif isinstance(self.data, dict):
                return len(self.data) * 128  # Rough estimate
            else:
                return 1024  # Default estimate
    
    def update_access(self, current_time: float = None):
        """Update access statistics"""
        if current_time is None:
            current_time = time.time()
        
        self.last_accessed = current_time
        self.access_count += 1
    
    def update_modification(self, current_time: float = None):
        """Update modification statistics"""
        if current_time is None:
            current_time = time.time()
        
        self.last_modified = current_time
        self.modification_count += 1
    
    def compute_recency_score(self, current_time: float = None) -> float:
        """Compute recency-based score"""
        if current_time is None:
            current_time = time.time()
        
        time_since_access = current_time - self.last_accessed
        recency_score = np.exp(-0.0001 * time_since_access)  # Exponential decay
        return float(recency_score)
    
    def compute_frequency_score(self) -> float:
        """Compute frequency-based score"""
        # Normalize by age to get frequency rate
        age = time.time() - self.created_time
        if age > 0:
            frequency_rate = self.access_count / age
            return min(frequency_rate * 3600, 1.0)  # Scale to hourly rate, cap at 1.0
        return 0.0
    
    def compute_composite_score(self, recency_weight: float = 0.4,
                               frequency_weight: float = 0.3,
                               importance_weight: float = 0.3) -> float:
        """Compute composite score for replacement decisions"""
        
        recency = self.compute_recency_score()
        frequency = self.compute_frequency_score()
        
        composite = (recency_weight * recency +
                    frequency_weight * frequency +
                    importance_weight * self.importance_score)
        
        return composite


class MemoryLevel(ABC):
    """Abstract base class for memory hierarchy levels"""
    
    def __init__(self, name: str, capacity: int, retention_time: int,
                 access_pattern: AccessPattern = AccessPattern.LRU):
        
        self.name = name
        self.capacity = capacity
        self.retention_time = retention_time
        self.access_pattern = access_pattern
        
        # Storage
        self.items: Dict[str, MemoryItem] = {}
        self.access_order = OrderedDict()  # For LRU tracking
        self.access_frequency = defaultdict(int)  # For LFU tracking
        
        # Statistics
        self.stats = {
            'total_items': 0,
            'total_accesses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'promotions': 0,
            'demotions': 0,
            'evictions': 0,
            'memory_usage_bytes': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug(f"MemoryLevel '{name}' initialized: capacity={capacity}, "
                    f"retention={retention_time}s, pattern={access_pattern.value}")
    
    @abstractmethod
    def should_promote(self, item: MemoryItem) -> bool:
        """Determine if item should be promoted to higher level"""
        pass
    
    @abstractmethod
    def should_demote(self, item: MemoryItem) -> bool:
        """Determine if item should be demoted to lower level"""
        pass
    
    def store(self, key: str, data: Any, importance: float = 0.5,
              metadata: Dict = None) -> bool:
        """Store item in memory level"""
        
        with self._lock:
            # Check if already exists
            if key in self.items:
                self._update_existing_item(key, data, importance, metadata)
                return True
            
            # Check capacity and make room if needed
            if len(self.items) >= self.capacity:
                if not self._make_room():
                    return False  # Could not make room
            
            # Create new memory item
            item = MemoryItem(
                data=data,
                key=key,
                importance_score=importance,
                metadata=metadata or {}
            )
            
            # Store item
            self.items[key] = item
            self._update_access_tracking(key, item)
            
            # Update statistics
            self.stats['total_items'] += 1
            self.stats['memory_usage_bytes'] += item.size_bytes
            
            logger.debug(f"Stored item '{key}' in level '{self.name}'")
            return True
    
    def retrieve(self, key: str) -> Optional[MemoryItem]:
        """Retrieve item from memory level"""
        
        with self._lock:
            if key in self.items:
                item = self.items[key]
                item.update_access()
                self._update_access_tracking(key, item)
                
                # Update statistics
                self.stats['total_accesses'] += 1
                self.stats['cache_hits'] += 1
                
                return item
            else:
                self.stats['cache_misses'] += 1
                return None
    
    def remove(self, key: str) -> bool:
        """Remove item from memory level"""
        
        with self._lock:
            if key in self.items:
                item = self.items.pop(key)
                self._remove_from_access_tracking(key)
                
                # Update statistics
                self.stats['memory_usage_bytes'] -= item.size_bytes
                self.stats['evictions'] += 1
                
                logger.debug(f"Removed item '{key}' from level '{self.name}'")
                return True
            return False
    
    def _update_existing_item(self, key: str, data: Any, importance: float,
                            metadata: Dict):
        """Update existing item"""
        
        item = self.items[key]
        old_size = item.size_bytes
        
        item.data = data
        item.importance_score = importance
        if metadata:
            item.metadata.update(metadata)
        item.update_modification()
        item.size_bytes = item._estimate_size()
        
        # Update memory usage
        self.stats['memory_usage_bytes'] += (item.size_bytes - old_size)
        
        # Update access tracking
        self._update_access_tracking(key, item)
    
    def _update_access_tracking(self, key: str, item: MemoryItem):
        """Update access pattern tracking"""
        
        if self.access_pattern in [AccessPattern.LRU, AccessPattern.HYBRID]:
            # Move to end (most recently used)
            if key in self.access_order:
                del self.access_order[key]
            self.access_order[key] = item
        
        if self.access_pattern in [AccessPattern.LFU, AccessPattern.HYBRID]:
            self.access_frequency[key] += 1
    
    def _remove_from_access_tracking(self, key: str):
        """Remove from access pattern tracking"""
        
        if key in self.access_order:
            del self.access_order[key]
        
        if key in self.access_frequency:
            del self.access_frequency[key]
    
    def _make_room(self) -> bool:
        """Make room by evicting items according to access pattern"""
        
        if not self.items:
            return True
        
        # Determine eviction candidates based on access pattern
        if self.access_pattern == AccessPattern.FIFO:
            candidates = list(self.items.keys())[:1]  # First inserted
        
        elif self.access_pattern == AccessPattern.LRU:
            # Least recently used (first in access_order)
            candidates = [next(iter(self.access_order))] if self.access_order else []
        
        elif self.access_pattern == AccessPattern.LFU:
            # Least frequently used
            if self.access_frequency:
                min_freq_key = min(self.access_frequency.keys(), 
                                 key=lambda k: self.access_frequency[k])
                candidates = [min_freq_key]
            else:
                candidates = []
        
        elif self.access_pattern == AccessPattern.IMPORTANCE_BASED:
            # Lowest importance score
            candidates = [min(self.items.keys(), 
                            key=lambda k: self.items[k].importance_score)]
        
        elif self.access_pattern == AccessPattern.TEMPORAL_DECAY:
            # Oldest items
            candidates = [min(self.items.keys(),
                            key=lambda k: self.items[k].last_accessed)]
        
        elif self.access_pattern == AccessPattern.HYBRID:
            # Combination of factors
            candidates = self._hybrid_eviction_candidates()
        
        else:
            candidates = [next(iter(self.items))]  # Fallback
        
        # Evict candidates
        evicted = False
        for key in candidates:
            if self.remove(key):
                evicted = True
                break
        
        return evicted
    
    def _hybrid_eviction_candidates(self) -> List[str]:
        """Select eviction candidates using hybrid approach"""
        
        if not self.items:
            return []
        
        # Compute composite scores for all items
        scored_items = []
        current_time = time.time()
        
        for key, item in self.items.items():
            score = item.compute_composite_score()
            scored_items.append((key, score))
        
        # Sort by score (lowest first = best eviction candidates)
        scored_items.sort(key=lambda x: x[1])
        
        # Return lowest scoring item(s)
        return [scored_items[0][0]]
    
    def cleanup_expired(self, current_time: float = None) -> int:
        """Remove expired items based on retention time"""
        
        if current_time is None:
            current_time = time.time()
        
        cutoff_time = current_time - self.retention_time
        expired_keys = []
        
        with self._lock:
            for key, item in self.items.items():
                if item.last_accessed < cutoff_time:
                    expired_keys.append(key)
            
            # Remove expired items
            removed_count = 0
            for key in expired_keys:
                if self.remove(key):
                    removed_count += 1
        
        logger.debug(f"Cleaned up {removed_count} expired items from level '{self.name}'")
        return removed_count
    
    def get_promotion_candidates(self) -> List[str]:
        """Get items that should be promoted to higher level"""
        
        candidates = []
        with self._lock:
            for key, item in self.items.items():
                if self.should_promote(item):
                    candidates.append(key)
        
        return candidates
    
    def get_demotion_candidates(self) -> List[str]:
        """Get items that should be demoted to lower level"""
        
        candidates = []
        with self._lock:
            for key, item in self.items.items():
                if self.should_demote(item):
                    candidates.append(key)
        
        return candidates
    
    def get_statistics(self) -> Dict:
        """Get comprehensive level statistics"""
        
        with self._lock:
            stats = self.stats.copy()
            
            # Add current state information
            current_stats = {
                'current_items': len(self.items),
                'capacity_utilization': len(self.items) / self.capacity if self.capacity > 0 else 0.0,
                'hit_rate': self.stats['cache_hits'] / max(self.stats['total_accesses'], 1),
                'average_item_size': (self.stats['memory_usage_bytes'] / 
                                    max(len(self.items), 1)) if self.items else 0,
                'memory_usage_mb': self.stats['memory_usage_bytes'] / (1024 * 1024)
            }
            
            # Add item age statistics if items exist
            if self.items:
                current_time = time.time()
                ages = [current_time - item.created_time for item in self.items.values()]
                access_ages = [current_time - item.last_accessed for item in self.items.values()]
                
                current_stats.update({
                    'avg_item_age': np.mean(ages),
                    'avg_access_age': np.mean(access_ages),
                    'oldest_item_age': np.max(ages),
                    'newest_item_age': np.min(ages)
                })
            
            return {**stats, **current_stats}
    
    def __len__(self) -> int:
        """Get number of items in level"""
        return len(self.items)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in level"""
        return key in self.items


class ShortTermLevel(MemoryLevel):
    """Short-term memory level for immediate access"""
    
    def __init__(self, capacity: int = 10000, retention_time: int = 300):
        super().__init__("short_term", capacity, retention_time, AccessPattern.LRU)
        self.promotion_threshold = 3  # Access count for promotion
        self.demotion_age = 60  # Seconds before considering demotion
    
    def should_promote(self, item: MemoryItem) -> bool:
        """Promote if frequently accessed"""
        return item.access_count >= self.promotion_threshold
    
    def should_demote(self, item: MemoryItem) -> bool:
        """Demote if old and unused"""
        time_since_access = time.time() - item.last_accessed
        return time_since_access > self.demotion_age and item.access_count == 1


class MediumTermLevel(MemoryLevel):
    """Medium-term memory level for working set"""
    
    def __init__(self, capacity: int = 50000, retention_time: int = 14400):
        super().__init__("medium_term", capacity, retention_time, AccessPattern.HYBRID)
        self.promotion_threshold = 0.7  # Importance score for promotion
        self.demotion_threshold = 0.2   # Importance score for demotion
    
    def should_promote(self, item: MemoryItem) -> bool:
        """Promote if important and frequently used"""
        return (item.importance_score >= self.promotion_threshold and
                item.access_count >= 2)
    
    def should_demote(self, item: MemoryItem) -> bool:
        """Demote if unimportant"""
        return item.importance_score <= self.demotion_threshold


class LongTermLevel(MemoryLevel):
    """Long-term memory level for persistent storage"""
    
    def __init__(self, capacity: int = 100000, retention_time: int = 86400):
        super().__init__("long_term", capacity, retention_time, AccessPattern.IMPORTANCE_BASED)
        self.compression_threshold = 3600  # Compress after 1 hour
        self.enable_compression = True
    
    def should_promote(self, item: MemoryItem) -> bool:
        """Long-term is highest level - no promotion"""
        return False
    
    def should_demote(self, item: MemoryItem) -> bool:
        """Demote if very old and unused"""
        age = time.time() - item.last_accessed
        return age > self.retention_time * 0.8 and item.access_count <= 1
    
    def store(self, key: str, data: Any, importance: float = 0.5,
              metadata: Dict = None) -> bool:
        """Store with compression for long-term items"""
        
        # Store normally first
        success = super().store(key, data, importance, metadata)
        
        if success and self.enable_compression:
            # Compress old items periodically
            self._compress_eligible_items()
        
        return success
    
    def _compress_eligible_items(self):
        """Compress items that haven't been accessed recently"""
        
        current_time = time.time()
        compression_candidates = []
        
        for key, item in self.items.items():
            if (not item.is_compressed and 
                current_time - item.last_accessed > self.compression_threshold):
                compression_candidates.append(key)
        
        # Compress candidates
        for key in compression_candidates[:10]:  # Limit to avoid blocking
            self._compress_item(key)
    
    def _compress_item(self, key: str) -> bool:
        """Compress a single item"""
        
        try:
            item = self.items[key]
            if item.is_compressed:
                return True
            
            # Serialize and compress data
            original_data = pickle.dumps(item.data)
            compressed_data = gzip.compress(original_data)
            
            # Update item
            old_size = item.size_bytes
            item.data = compressed_data
            item.is_compressed = True
            item.size_bytes = len(compressed_data)
            item.compression_ratio = len(compressed_data) / len(original_data)
            
            # Update memory usage
            self.stats['memory_usage_bytes'] -= (old_size - item.size_bytes)
            
            logger.debug(f"Compressed item '{key}': {old_size} -> {item.size_bytes} bytes "
                        f"({item.compression_ratio:.2f} ratio)")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to compress item '{key}': {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[MemoryItem]:
        """Retrieve with decompression if needed"""
        
        item = super().retrieve(key)
        
        if item and item.is_compressed:
            self._decompress_item(key)
        
        return item
    
    def _decompress_item(self, key: str) -> bool:
        """Decompress a compressed item"""
        
        try:
            item = self.items[key]
            if not item.is_compressed:
                return True
            
            # Decompress data
            decompressed_data = gzip.decompress(item.data)
            original_data = pickle.loads(decompressed_data)
            
            # Update item
            old_size = item.size_bytes
            item.data = original_data
            item.is_compressed = False
            item.size_bytes = item._estimate_size()
            
            # Update memory usage
            self.stats['memory_usage_bytes'] += (item.size_bytes - old_size)
            
            logger.debug(f"Decompressed item '{key}': {old_size} -> {item.size_bytes} bytes")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to decompress item '{key}': {e}")
            return False


class MemoryHierarchy:
    """
    Main Memory Hierarchy Manager
    
    Coordinates multiple memory levels with automatic promotion/demotion
    and intelligent memory management policies.
    """
    
    def __init__(self, config: Dict, levels: int = 3):
        
        self.config = config
        self.levels_count = levels
        
        # Initialize memory levels
        self.levels = self._initialize_levels(config)
        
        # Global statistics
        self.global_stats = {
            'total_stores': 0,
            'total_retrievals': 0,
            'total_promotions': 0,
            'total_demotions': 0,
            'total_compressions': 0,
            'total_decompressions': 0,
            'garbage_collections': 0
        }
        
        # Management configuration
        self.auto_promotion = config.get('auto_promotion', True)
        self.auto_demotion = config.get('auto_demotion', True)
        self.garbage_collection_interval = config.get('garbage_collection_interval', 1000)
        self.compression_enabled = config.get('compression_enabled', True)
        
        # Operation counters for triggering maintenance
        self.operation_count = 0
        self.last_gc_operation = 0
        self.last_maintenance_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"MemoryHierarchy initialized: {levels} levels, "
                   f"auto_promotion={self.auto_promotion}, "
                   f"auto_demotion={self.auto_demotion}")
    
    def _initialize_levels(self, config: Dict) -> Dict[str, MemoryLevel]:
        """Initialize memory hierarchy levels"""
        
        levels = {}
        
        # Get level configurations
        level_config = config.get('level_config', {})
        
        # Short-term level
        short_config = level_config.get('short_term', {})
        levels['short_term'] = ShortTermLevel(
            capacity=short_config.get('capacity', 10000),
            retention_time=short_config.get('retention_time', 300)
        )
        
        # Medium-term level
        medium_config = level_config.get('medium_term', {})
        levels['medium_term'] = MediumTermLevel(
            capacity=medium_config.get('capacity', 50000),
            retention_time=medium_config.get('retention_time', 14400)
        )
        
        # Long-term level
        long_config = level_config.get('long_term', {})
        levels['long_term'] = LongTermLevel(
            capacity=long_config.get('capacity', 100000),
            retention_time=long_config.get('retention_time', 86400)
        )
        
        return levels
    
    def store_sample(self, sample: Dict, level: str = 'short_term') -> bool:
        """
        Store sample in specified memory level
        
        Parameters
        ----------
        sample : Dict
            Sample data to store
        level : str
            Target memory level ('short_term', 'medium_term', 'long_term')
            
        Returns
        -------
        success : bool
            Whether storage was successful
        """
        
        with self._lock:
            # Generate key from sample
            key = self._generate_key(sample)
            
            # Extract importance score
            importance = sample.get('importance', 0.5)
            
            # Store in specified level
            if level in self.levels:
                success = self.levels[level].store(
                    key=key,
                    data=sample,
                    importance=importance,
                    metadata={'storage_level': level, 'timestamp': time.time()}
                )
                
                if success:
                    self.global_stats['total_stores'] += 1
                    self.operation_count += 1
                    
                    # Trigger maintenance if needed
                    self._check_maintenance_triggers()
                    
                return success
            else:
                logger.warning(f"Invalid memory level: {level}")
                return False
    
    def retrieve_sample(self, key: str, level: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieve sample from memory hierarchy
        
        Parameters
        ----------
        key : str
            Sample key to retrieve
        level : str, optional
            Specific level to search (if None, searches all levels)
            
        Returns
        -------
        sample : Dict or None
            Retrieved sample data
        """
        
        with self._lock:
            self.global_stats['total_retrievals'] += 1
            self.operation_count += 1
            
            if level:
                # Search specific level
                if level in self.levels:
                    item = self.levels[level].retrieve(key)
                    return item.data if item else None
                else:
                    return None
            else:
                # Search all levels (short to long term)
                for level_name in ['short_term', 'medium_term', 'long_term']:
                    if level_name in self.levels:
                        item = self.levels[level_name].retrieve(key)
                        if item:
                            return item.data
                
                return None
    
    def promote_item(self, key: str, from_level: str, to_level: str) -> bool:
        """Promote item between memory levels"""
        
        with self._lock:
            if from_level not in self.levels or to_level not in self.levels:
                return False
            
            # Retrieve from source level
            item = self.levels[from_level].retrieve(key)
            if not item:
                return False
            
            # Store in target level
            success = self.levels[to_level].store(
                key=key,
                data=item.data,
                importance=item.importance_score,
                metadata=item.metadata
            )
            
            if success:
                # Remove from source level
                self.levels[from_level].remove(key)
                
                # Update statistics
                self.global_stats['total_promotions'] += 1
                self.levels[from_level].stats['promotions'] += 1
                
                logger.debug(f"Promoted item '{key}' from {from_level} to {to_level}")
                
                return True
            
            return False
    
    def demote_item(self, key: str, from_level: str, to_level: str) -> bool:
        """Demote item between memory levels"""
        
        with self._lock:
            if from_level not in self.levels or to_level not in self.levels:
                return False
            
            # Retrieve from source level
            item = self.levels[from_level].retrieve(key)
            if not item:
                return False
            
            # Store in target level
            success = self.levels[to_level].store(
                key=key,
                data=item.data,
                importance=item.importance_score,
                metadata=item.metadata
            )
            
            if success:
                # Remove from source level
                self.levels[from_level].remove(key)
                
                # Update statistics
                self.global_stats['total_demotions'] += 1
                self.levels[from_level].stats['demotions'] += 1
                
                logger.debug(f"Demoted item '{key}' from {from_level} to {to_level}")
                
                return True
            
            return False
    
    def auto_manage_levels(self) -> Dict[str, int]:
        """Automatically promote and demote items between levels"""
        
        management_stats = {
            'promotions': 0,
            'demotions': 0,
            'errors': 0
        }
        
        with self._lock:
            
            # Short-term to medium-term promotions
            if self.auto_promotion:
                short_promotions = self.levels['short_term'].get_promotion_candidates()
                for key in short_promotions:
                    try:
                        if self.promote_item(key, 'short_term', 'medium_term'):
                            management_stats['promotions'] += 1
                    except Exception as e:
                        logger.warning(f"Promotion failed for {key}: {e}")
                        management_stats['errors'] += 1
            
            # Medium-term to long-term promotions
            if self.auto_promotion:
                medium_promotions = self.levels['medium_term'].get_promotion_candidates()
                for key in medium_promotions:
                    try:
                        if self.promote_item(key, 'medium_term', 'long_term'):
                            management_stats['promotions'] += 1
                    except Exception as e:
                        logger.warning(f"Promotion failed for {key}: {e}")
                        management_stats['errors'] += 1
            
            # Long-term to medium-term demotions
            if self.auto_demotion:
                long_demotions = self.levels['long_term'].get_demotion_candidates()
                for key in long_demotions:
                    try:
                        if self.demote_item(key, 'long_term', 'medium_term'):
                            management_stats['demotions'] += 1
                    except Exception as e:
                        logger.warning(f"Demotion failed for {key}: {e}")
                        management_stats['errors'] += 1
            
            # Medium-term to short-term demotions
            if self.auto_demotion:
                medium_demotions = self.levels['medium_term'].get_demotion_candidates()
                for key in medium_demotions:
                    try:
                        if self.demote_item(key, 'medium_term', 'short_term'):
                            management_stats['demotions'] += 1
                    except Exception as e:
                        logger.warning(f"Demotion failed for {key}: {e}")
                        management_stats['errors'] += 1
        
        if management_stats['promotions'] > 0 or management_stats['demotions'] > 0:
            logger.debug(f"Auto-management completed: {management_stats}")
        
        return management_stats
    
    def garbage_collect(self) -> Dict[str, int]:
        """Perform garbage collection across all levels"""
        
        gc_stats = {
            'expired_items_removed': 0,
            'memory_freed_bytes': 0,
            'compression_operations': 0
        }
        
        with self._lock:
            current_time = time.time()
            
            # Clean up expired items in all levels
            for level_name, level in self.levels.items():
                before_memory = level.stats['memory_usage_bytes']
                expired_count = level.cleanup_expired(current_time)
                after_memory = level.stats['memory_usage_bytes']
                
                gc_stats['expired_items_removed'] += expired_count
                gc_stats['memory_freed_bytes'] += (before_memory - after_memory)
            
            # Trigger Python garbage collection
            collected = gc.collect()
            
            # Update statistics
            self.global_stats['garbage_collections'] += 1
            self.last_gc_operation = self.operation_count
            
            logger.debug(f"Garbage collection completed: {gc_stats}, "
                        f"Python GC collected {collected} objects")
        
        return gc_stats
    
    def _generate_key(self, sample: Dict) -> str:
        """Generate unique key for sample"""
        
        # Use timestamp and sample hash for uniqueness
        timestamp = sample.get('timestamp', time.time())
        
        # Create hash from sample content
        sample_str = str(sorted(sample.items()))
        sample_hash = hashlib.md5(sample_str.encode()).hexdigest()[:16]
        
        key = f"{timestamp:.3f}_{sample_hash}"
        return key
    
    def _check_maintenance_triggers(self):
        """Check if maintenance operations should be triggered"""
        
        # Garbage collection trigger
        if (self.operation_count - self.last_gc_operation) >= self.garbage_collection_interval:
            self.garbage_collect()
        
        # Auto-management trigger
        current_time = time.time()
        if current_time - self.last_maintenance_time > 300:  # Every 5 minutes
            self.auto_manage_levels()
            self.last_maintenance_time = current_time
    
    def get_memory_usage(self) -> float:
        """Get total memory usage across all levels in MB"""
        
        total_bytes = 0
        with self._lock:
            for level in self.levels.values():
                total_bytes += level.stats['memory_usage_bytes']
        
        return total_bytes / (1024 * 1024)
    
    def get_system_memory_info(self) -> Dict:
        """Get system memory information"""
        
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percentage': memory.percent,
                'hierarchy_usage_mb': self.get_memory_usage()
            }
        except Exception:
            return {'error': 'Unable to get system memory info'}
    
    def get_comprehensive_statistics(self) -> Dict:
        """Get comprehensive statistics for entire hierarchy"""
        
        with self._lock:
            stats = {
                'global_stats': self.global_stats.copy(),
                'level_stats': {},
                'hierarchy_info': {
                    'num_levels': len(self.levels),
                    'auto_promotion': self.auto_promotion,
                    'auto_demotion': self.auto_demotion,
                    'compression_enabled': self.compression_enabled,
                    'total_memory_mb': self.get_memory_usage(),
                    'operation_count': self.operation_count
                },
                'system_memory': self.get_system_memory_info()
            }
            
            # Add statistics from each level
            for level_name, level in self.levels.items():
                stats['level_stats'][level_name] = level.get_statistics()
            
            return stats
    
    def optimize_memory_usage(self) -> Dict:
        """Optimize memory usage across hierarchy"""
        
        optimization_stats = {
            'compressions': 0,
            'evictions': 0,
            'promotions': 0,
            'demotions': 0,
            'memory_saved_mb': 0
        }
        
        with self._lock:
            initial_memory = self.get_memory_usage()
            
            # Force garbage collection
            gc_stats = self.garbage_collect()
            optimization_stats['evictions'] = gc_stats['expired_items_removed']
            
            # Auto-manage levels
            management_stats = self.auto_manage_levels()
            optimization_stats['promotions'] = management_stats['promotions']
            optimization_stats['demotions'] = management_stats['demotions']
            
            # Force compression in long-term storage
            if 'long_term' in self.levels and hasattr(self.levels['long_term'], '_compress_eligible_items'):
                self.levels['long_term']._compress_eligible_items()
            
            final_memory = self.get_memory_usage()
            optimization_stats['memory_saved_mb'] = initial_memory - final_memory
            
            logger.info(f"Memory optimization completed: {optimization_stats}")
        
        return optimization_stats
    
    def clear_level(self, level_name: str) -> bool:
        """Clear all items from specified level"""
        
        with self._lock:
            if level_name in self.levels:
                level = self.levels[level_name]
                keys_to_remove = list(level.items.keys())
                
                for key in keys_to_remove:
                    level.remove(key)
                
                logger.info(f"Cleared {len(keys_to_remove)} items from level '{level_name}'")
                return True
            
            return False
    
    def clear_all_levels(self) -> Dict[str, int]:
        """Clear all items from all levels"""
        
        clear_stats = {}
        
        with self._lock:
            for level_name in self.levels.keys():
                initial_count = len(self.levels[level_name])
                self.clear_level(level_name)
                clear_stats[level_name] = initial_count
        
        logger.info(f"Cleared all levels: {clear_stats}")
        return clear_stats
    
    def get_state(self) -> Dict:
        """Get complete hierarchy state for serialization"""
        
        with self._lock:
            # Note: We don't serialize the actual stored data for efficiency
            # Only the metadata and configuration
            
            state = {
                'config': self.config,
                'levels_count': self.levels_count,
                'global_stats': self.global_stats,
                'auto_promotion': self.auto_promotion,
                'auto_demotion': self.auto_demotion,
                'compression_enabled': self.compression_enabled,
                'operation_count': self.operation_count,
                'last_maintenance_time': self.last_maintenance_time,
                'level_stats': {}
            }
            
            # Add level statistics (not actual data)
            for level_name, level in self.levels.items():
                state['level_stats'][level_name] = level.get_statistics()
            
            return state
    
    def restore_state(self, state: Dict) -> None:
        """Restore hierarchy state from serialization"""
        
        with self._lock:
            self.config = state['config']
            self.levels_count = state['levels_count']
            self.global_stats = state['global_stats']
            self.auto_promotion = state['auto_promotion']
            self.auto_demotion = state['auto_demotion']
            self.compression_enabled = state['compression_enabled']
            self.operation_count = state['operation_count']
            self.last_maintenance_time = state['last_maintenance_time']
            
            # Reinitialize levels (data is not restored, only structure)
            self.levels = self._initialize_levels(self.config)
        
        logger.info("MemoryHierarchy state restored successfully")
    
    def get_level_names(self) -> List[str]:
        """Get names of all memory levels"""
        return list(self.levels.keys())
    
    def get_level(self, level_name: str) -> Optional[MemoryLevel]:
        """Get specific memory level by name"""
        return self.levels.get(level_name)
    
    def __len__(self) -> int:
        """Get total number of items across all levels"""
        total_items = 0
        with self._lock:
            for level in self.levels.values():
                total_items += len(level)
        return total_items
    
    def __repr__(self) -> str:
        """String representation of hierarchy"""
        level_info = []
        with self._lock:
            for name, level in self.levels.items():
                level_info.append(f"{name}({len(level)})")
        
        return f"MemoryHierarchy({', '.join(level_info)}, {self.get_memory_usage():.1f}MB)" 
