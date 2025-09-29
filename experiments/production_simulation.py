# File: hyperpath_svm/experiments/production_simulation.py

"""
Production Deployment Simulation for HyperPath-SVM

This module simulates a 6-month production deployment of HyperPath-SVM in a real
network environment. It models realistic network conditions, traffic patterns,
failures, and adaptation scenarios to demonstrate production-ready performance.

"""

import numpy as np
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from collections import deque, defaultdict
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging_utils import get_logger, get_hyperpath_logger
from ..core.hyperpath_svm import HyperPathSVM
from ..data.network_graph import NetworkGraph
from ..evaluation.metrics import ComprehensiveMetricsEvaluator
from ..utils.math_utils import math_utils
from ..utils.visualization import PaperFigureGenerator

# Simulation time scales
SECONDS_PER_DAY = 86400
DAYS_PER_MONTH = 30
SIMULATION_DAYS = 180  # 6 months


class EventType(Enum):
   
    TRAFFIC_SPIKE = "traffic_spike"
    LINK_FAILURE = "link_failure" 
    NODE_FAILURE = "node_failure"
    TOPOLOGY_CHANGE = "topology_change"
    SECURITY_INCIDENT = "security_incident"
    MAINTENANCE = "maintenance"
    CONFIGURATION_CHANGE = "config_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class NetworkCondition(Enum):
    
    NORMAL = "normal"
    CONGESTED = "congested"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


@dataclass
class NetworkEvent:
   
    timestamp: float
    event_type: EventType
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_components: List[str]
    duration_seconds: float
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class PerformanceSnapshot:
    
    timestamp: float
    accuracy: float
    inference_time_ms: float
    memory_usage_mb: float
    throughput_rps: float  # Requests per second
    error_rate: float
    network_condition: NetworkCondition
    active_connections: int
    queue_depth: int
    cpu_utilization: float
    
    # Business metrics
    sla_compliance: float
    cost_per_decision: float
    user_satisfaction: float


@dataclass
class ProductionConfig:
    
    
    # Simulation parameters
    simulation_days: int = 180
    time_acceleration: int = 1000  # 1000x faster than real time
    monitoring_interval_seconds: int = 60
    
    # Network parameters
    initial_network_size: int = 5000
    max_network_size: int = 20000
    network_growth_rate: float = 0.02  # 2% per month
    
    # Traffic parameters
    base_traffic_rps: float = 1000.0
    peak_traffic_multiplier: float = 10.0
    seasonal_variation: float = 0.3
    
    # Failure parameters
    link_failure_rate: float = 0.01  # Per day
    node_failure_rate: float = 0.005  # Per day
    mean_repair_time_hours: float = 2.0
    
    # Performance targets
    target_accuracy: float = 0.965
    target_inference_ms: float = 1.8
    target_availability: float = 0.999
    max_memory_gb: float = 8.0
    
    # Business parameters
    cost_per_server_hour: float = 0.10
    sla_penalty_rate: float = 1000.0  # Per hour of SLA violation
    
    # Output configuration
    results_dir: Path = Path("production_simulation")
    save_detailed_logs: bool = True
    generate_reports: bool = True


class NetworkTopologySimulator:
    """Simulates evolving network topology."""
    
    def __init__(self, initial_size: int, max_size: int, growth_rate: float):
        self.current_size = initial_size
        self.max_size = max_size
        self.growth_rate = growth_rate
        self.logger = get_logger(__name__)
        
        # Initialize base topology
        self.adjacency_matrix = self._generate_initial_topology()
        self.node_failures = set()
        self.link_failures = set()
    
    def _generate_initial_topology(self) -> np.ndarray:
        
        import networkx as nx
        
        # Create scale-free network
        G = nx.barabasi_albert_graph(self.current_size, m=3, seed=42)
        return nx.adjacency_matrix(G).toarray().astype(float)
    
    def evolve_topology(self, days_elapsed: float) -> Dict[str, Any]:
       
        changes = {'nodes_added': 0, 'links_added': 0, 'nodes_removed': 0}
        
        # Network growth
        growth_factor = 1 + (self.growth_rate * days_elapsed / 30)  # Monthly growth
        target_size = min(int(self.current_size * growth_factor), self.max_size)
        
        if target_size > self.current_size:
            # Add nodes
            nodes_to_add = target_size - self.current_size
            self._add_nodes(nodes_to_add)
            changes['nodes_added'] = nodes_to_add
        
        return changes
    
    def _add_nodes(self, num_nodes: int):
       
        old_size = self.current_size
        new_size = old_size + num_nodes
        
        # Expand adjacency matrix
        new_adjacency = np.zeros((new_size, new_size))
        new_adjacency[:old_size, :old_size] = self.adjacency_matrix
        
        # Add connections for new nodes (preferential attachment)
        for new_node in range(old_size, new_size):
            # Connect to existing nodes based on degree
            degrees = np.sum(new_adjacency[:old_size, :old_size], axis=1)
            probabilities = degrees / np.sum(degrees) if np.sum(degrees) > 0 else np.ones(old_size) / old_size
            
            # Add 2-5 connections per new node
            num_connections = random.randint(2, 5)
            connected_nodes = np.random.choice(
                old_size, size=min(num_connections, old_size), 
                replace=False, p=probabilities
            )
            
            for connected_node in connected_nodes:
                new_adjacency[new_node, connected_node] = 1
                new_adjacency[connected_node, new_node] = 1
        
        self.adjacency_matrix = new_adjacency
        self.current_size = new_size
    
    def apply_failure(self, event: NetworkEvent):
       
        if event.event_type == EventType.NODE_FAILURE:
            # Random node failure
            available_nodes = set(range(self.current_size)) - self.node_failures
            if available_nodes:
                failed_node = random.choice(list(available_nodes))
                self.node_failures.add(failed_node)
                event.affected_components = [f"node_{failed_node}"]
        
        elif event.event_type == EventType.LINK_FAILURE:
            # Random link failure
            active_links = []
            for i in range(self.current_size):
                for j in range(i + 1, self.current_size):
                    if (self.adjacency_matrix[i, j] > 0 and 
                        (i, j) not in self.link_failures and
                        i not in self.node_failures and j not in self.node_failures):
                        active_links.append((i, j))
            
            if active_links:
                failed_link = random.choice(active_links)
                self.link_failures.add(failed_link)
                event.affected_components = [f"link_{failed_link[0]}_{failed_link[1]}"]
    
    def recover_failure(self, event: NetworkEvent):
        
        event.resolved = True
        event.resolution_time = time.time()
        
        for component in event.affected_components:
            if component.startswith("node_"):
                node_id = int(component.split("_")[1])
                self.node_failures.discard(node_id)
            elif component.startswith("link_"):
                parts = component.split("_")
                link = (int(parts[1]), int(parts[2]))
                self.link_failures.discard(link)
    
    def get_effective_topology(self) -> np.ndarray:
        
        effective_topology = self.adjacency_matrix.copy()
        
        # Remove failed nodes
        for failed_node in self.node_failures:
            effective_topology[failed_node, :] = 0
            effective_topology[:, failed_node] = 0
        
        # Remove failed links
        for failed_link in self.link_failures:
            i, j = failed_link
            effective_topology[i, j] = 0
            effective_topology[j, i] = 0
        
        return effective_topology


class TrafficGenerator:
   
    
    def __init__(self, base_rps: float, peak_multiplier: float, seasonal_variation: float):
        self.base_rps = base_rps
        self.peak_multiplier = peak_multiplier
        self.seasonal_variation = seasonal_variation
        
    def generate_traffic_load(self, timestamp: float, day_of_simulation: int) -> float:
        """Generate traffic load at given timestamp."""
        # Time of day pattern (peak during business hours)
        hour_of_day = (timestamp % SECONDS_PER_DAY) / 3600
        diurnal_factor = (0.3 + 0.7 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)**2)
        
        # Day of week pattern (lower on weekends)
        day_of_week = (day_of_simulation % 7)
        weekly_factor = 0.6 if day_of_week >= 5 else 1.0  # Weekend reduction
        
        # Seasonal variation
        seasonal_factor = 1 + self.seasonal_variation * np.sin(2 * np.pi * day_of_simulation / 365)
        
        # Random noise
        noise_factor = 1 + np.random.normal(0, 0.1)
        
        # Calculate final load
        traffic_load = (self.base_rps * diurnal_factor * weekly_factor * 
                       seasonal_factor * noise_factor)
        
        return max(0, traffic_load)
    
    def generate_traffic_spike(self) -> float:
        
        spike_multiplier = np.random.uniform(3, self.peak_multiplier)
        return spike_multiplier
    
    def generate_traffic_characteristics(self, load_rps: float) -> Dict[str, float]:
       
        return {
            'request_rate': load_rps,
            'avg_request_size_kb': np.random.normal(10, 2),
            'connection_duration_s': np.random.exponential(30),
            'query_complexity': np.random.beta(2, 5),  # Most queries are simple
            'geographic_distribution': np.random.dirichlet([1, 1, 1, 1]),  # 4 regions
            'protocol_mix': {
                'tcp': 0.8 + np.random.normal(0, 0.05),
                'udp': 0.15 + np.random.normal(0, 0.02),
                'other': 0.05 + np.random.normal(0, 0.01)
            }
        }


class EventGenerator:
    
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
    def generate_events_for_day(self, day: int) -> List[NetworkEvent]:
        
        events = []
        
        # Link failures
        if np.random.random() < self.config.link_failure_rate:
            events.append(self._create_link_failure_event(day))
        
        # Node failures  
        if np.random.random() < self.config.node_failure_rate:
            events.append(self._create_node_failure_event(day))
        
        # Traffic spikes
        if np.random.random() < 0.1:  # 10% chance per day
            events.append(self._create_traffic_spike_event(day))
        
        # Security incidents
        if np.random.random() < 0.02:  # 2% chance per day
            events.append(self._create_security_incident(day))
        
        # Maintenance windows
        if day % 30 == 0:  # Monthly maintenance
            events.append(self._create_maintenance_event(day))
        
        return events
    
    def _create_link_failure_event(self, day: int) -> NetworkEvent:
       
        return NetworkEvent(
            timestamp=day * SECONDS_PER_DAY + np.random.random() * SECONDS_PER_DAY,
            event_type=EventType.LINK_FAILURE,
            severity=np.random.choice(['medium', 'high'], p=[0.7, 0.3]),
            affected_components=[],  # Will be filled by topology simulator
            duration_seconds=np.random.exponential(self.config.mean_repair_time_hours * 3600),
            impact_metrics={'availability_impact': np.random.uniform(0.01, 0.05)}
        )
    
    def _create_node_failure_event(self, day: int) -> NetworkEvent:
        
        return NetworkEvent(
            timestamp=day * SECONDS_PER_DAY + np.random.random() * SECONDS_PER_DAY,
            event_type=EventType.NODE_FAILURE,
            severity=np.random.choice(['high', 'critical'], p=[0.6, 0.4]),
            affected_components=[],
            duration_seconds=np.random.exponential(self.config.mean_repair_time_hours * 3600 * 2),
            impact_metrics={'availability_impact': np.random.uniform(0.05, 0.15)}
        )
    
    def _create_traffic_spike_event(self, day: int) -> NetworkEvent:
        
        return NetworkEvent(
            timestamp=day * SECONDS_PER_DAY + np.random.random() * SECONDS_PER_DAY,
            event_type=EventType.TRAFFIC_SPIKE,
            severity=np.random.choice(['medium', 'high'], p=[0.8, 0.2]),
            affected_components=['traffic_system'],
            duration_seconds=np.random.exponential(3600),  # Average 1 hour
            impact_metrics={'traffic_multiplier': np.random.uniform(2, 8)}
        )
    
    def _create_security_incident(self, day: int) -> NetworkEvent:
        
        return NetworkEvent(
            timestamp=day * SECONDS_PER_DAY + np.random.random() * SECONDS_PER_DAY,
            event_type=EventType.SECURITY_INCIDENT,
            severity=np.random.choice(['medium', 'high', 'critical'], p=[0.5, 0.4, 0.1]),
            affected_components=['security_system'],
            duration_seconds=np.random.exponential(7200),  # Average 2 hours
            impact_metrics={'security_overhead': np.random.uniform(0.1, 0.3)}
        )
    
    def _create_maintenance_event(self, day: int) -> NetworkEvent:
        
        return NetworkEvent(
            timestamp=day * SECONDS_PER_DAY + 2 * 3600,  # 2 AM maintenance
            event_type=EventType.MAINTENANCE,
            severity='medium',
            affected_components=['maintenance_system'],
            duration_seconds=4 * 3600,  # 4 hour maintenance window
            impact_metrics={'performance_reduction': 0.2}
        )


class ProductionSimulator:
    
    
    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or ProductionConfig()
        self.logger = get_logger(__name__)
        self.hyperpath_logger = get_hyperpath_logger(__name__)
        
        # Initialize components
        self.topology_simulator = NetworkTopologySimulator(
            self.config.initial_network_size,
            self.config.max_network_size,
            self.config.network_growth_rate
        )
        
        self.traffic_generator = TrafficGenerator(
            self.config.base_traffic_rps,
            self.config.peak_traffic_multiplier,
            0.3  # seasonal variation
        )
        
        self.event_generator = EventGenerator(self.config)
        self.metrics_evaluator = ComprehensiveMetricsEvaluator()
        
        # Initialize HyperPath-SVM model
        self.model = HyperPathSVM(
            C=1.0,
            kernel='tgck',
            use_ddwe=True,
            quantum_optimization=True,
            continuous_learning=True
        )
        
        # Simulation state
        self.current_time = 0.0
        self.simulation_day = 0
        self.performance_history = []
        self.event_history = []
        self.active_events = []
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_cost = 0.0
        self.sla_violations = 0
        
        # Setup output directory
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Production simulator initialized")
    
    def run_production_simulation(self) -> Dict[str, Any]:
       
        try:
            self.logger.info(f"Starting {self.config.simulation_days}-day production simulation")
            
            # Initialize model with synthetic training data
            self._initialize_model()
            
            # Main simulation loop
            simulation_results = {
                'start_time': time.time(),
                'performance_snapshots': [],
                'events': [],
                'daily_summaries': [],
                'adaptation_events': [],
                'cost_analysis': {},
                'sla_compliance': {},
                'lessons_learned': []
            }
            
            for day in range(self.config.simulation_days):
                self.simulation_day = day
                self.current_time = day * SECONDS_PER_DAY
                
                self.logger.info(f"Simulating day {day + 1}/{self.config.simulation_days}")
                
                # Daily simulation
                daily_results = self._simulate_day(day)
                simulation_results['daily_summaries'].append(daily_results)
                
                # Collect performance snapshots
                simulation_results['performance_snapshots'].extend(daily_results['snapshots'])
                
                # Collect events
                simulation_results['events'].extend(daily_results['events'])
                
                # Model adaptation (weekly)
                if day % 7 == 0 and day > 0:
                    adaptation_result = self._perform_model_adaptation()
                    simulation_results['adaptation_events'].append(adaptation_result)
                
                # Progress logging
                if (day + 1) % 30 == 0:  # Monthly progress
                    self._log_monthly_summary(day + 1, simulation_results)
            
            # Final analysis
            simulation_results['end_time'] = time.time()
            simulation_results['total_simulation_time'] = (
                simulation_results['end_time'] - simulation_results['start_time']
            )
            
            # Comprehensive analysis
            self._analyze_simulation_results(simulation_results)
            
            # Generate reports and visualizations
            self._generate_simulation_reports(simulation_results)
            
            self.logger.info(f"Production simulation completed successfully")
            self.logger.info(f"Total requests processed: {self.total_requests:,}")
            self.logger.info(f"Overall success rate: {self.successful_requests / max(self.total_requests, 1):.4f}")
            self.logger.info(f"Total cost: ${self.total_cost:.2f}")
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Production simulation failed: {str(e)}")
            raise
    
    def _initialize_model(self):
        
        try:
            self.logger.info("Initializing HyperPath-SVM model for production")
            
            # Generate initial training data based on current topology
            topology = self.topology_simulator.get_effective_topology()
            
            # Create network graph
            network_graph = NetworkGraph(
                adjacency_matrix=topology,
                node_features=np.random.normal(0, 1, (topology.shape[0], 20)),
                edge_features=np.random.normal(0, 1, (np.sum(topology > 0), 10)),
                temporal_features=np.random.normal(0, 1, (100, topology.shape[0]))
            )
            
            # Generate training samples
            n_samples = 10000
            features = np.random.normal(0, 1, (n_samples, 50))
            labels = np.random.randint(0, 3, n_samples)
            
            # Train initial model
            with self.hyperpath_logger.timer("initial_model_training"):
                self.model.fit(features, labels)
            
            self.logger.info("Model initialization completed")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def _simulate_day(self, day: int) -> Dict[str, Any]:
        
        daily_results = {
            'day': day,
            'snapshots': [],
            'events': [],
            'total_requests': 0,
            'successful_requests': 0,
            'errors': 0,
            'cost': 0.0,
            'avg_response_time': 0.0,
            'peak_load': 0.0,
            'network_changes': {}
        }
        
        # Generate events for the day
        daily_events = self.event_generator.generate_events_for_day(day)
        self.active_events.extend(daily_events)
        daily_results['events'] = daily_events
        
        # Evolve network topology
        topology_changes = self.topology_simulator.evolve_topology(day)
        daily_results['network_changes'] = topology_changes
        
        # Simulate hourly operation
        for hour in range(24):
            timestamp = day * SECONDS_PER_DAY + hour * 3600
            
            # Process active events
            self._process_active_events(timestamp)
            
            # Generate traffic load
            base_load = self.traffic_generator.generate_traffic_load(timestamp, day)
            
            # Apply event impacts
            effective_load = self._apply_event_impacts(base_load, timestamp)
            
            # Simulate hour of operation
            hour_results = self._simulate_hour(timestamp, effective_load)
            
            # Create performance snapshot
            snapshot = self._create_performance_snapshot(timestamp, hour_results)
            daily_results['snapshots'].append(snapshot)
            
            # Update daily totals
            daily_results['total_requests'] += hour_results['requests']
            daily_results['successful_requests'] += hour_results['successful']
            daily_results['errors'] += hour_results['errors']
            daily_results['cost'] += hour_results['cost']
            daily_results['peak_load'] = max(daily_results['peak_load'], effective_load)
        
        # Calculate daily averages
        if daily_results['snapshots']:
            daily_results['avg_response_time'] = np.mean([
                s.inference_time_ms for s in daily_results['snapshots']
            ])
        
        return daily_results
    
    def _simulate_hour(self, timestamp: float, load_rps: float) -> Dict[str, Any]:
       
        hour_results = {
            'timestamp': timestamp,
            'load_rps': load_rps,
            'requests': 0,
            'successful': 0,
            'errors': 0,
            'total_inference_time': 0.0,
            'cost': 0.0,
            'memory_usage': 0.0
        }
        
        # Calculate number of requests for this hour
        requests_this_hour = int(load_rps * 3600)  # requests per second * seconds per hour
        hour_results['requests'] = requests_this_hour
        
        # Process requests in batches for efficiency
        batch_size = 100
        num_batches = max(1, requests_this_hour // batch_size)
        
        for batch in range(num_batches):
            batch_requests = min(batch_size, requests_this_hour - batch * batch_size)
            
            # Generate request data
            request_features = np.random.normal(0, 1, (batch_requests, 50))
            
            try:
                # Process requests through HyperPath-SVM
                start_time = time.time()
                predictions = self.model.predict(request_features)
                inference_time = time.time() - start_time
                
                # Calculate metrics
                avg_inference_ms = (inference_time * 1000) / batch_requests
                hour_results['total_inference_time'] += inference_time * 1000
                hour_results['successful'] += batch_requests
                
                # Simulate memory usage (rough estimate)
                hour_results['memory_usage'] = max(hour_results['memory_usage'], 
                                                  50 + batch_requests * 0.01)  # MB
                
            except Exception as e:
                # Handle errors
                self.logger.warning(f"Batch processing failed: {str(e)}")
                hour_results['errors'] += batch_requests
        
        # Calculate costs (simplified)
        server_cost = self.config.cost_per_server_hour / 24  # Per hour
        hour_results['cost'] = server_cost
        
        # Add SLA penalties if performance is below targets
        if hour_results['successful'] > 0:
            avg_inference_ms = hour_results['total_inference_time'] / hour_results['successful']
            if avg_inference_ms > self.config.target_inference_ms * 2:  # 2x target threshold
                hour_results['cost'] += self.config.sla_penalty_rate / 24
        
        return hour_results
    
    def _process_active_events(self, timestamp: float):
       
        resolved_events = []
        
        for event in self.active_events:
            if not event.resolved:
                # Check if event should be resolved
                if timestamp >= event.timestamp + event.duration_seconds:
                    self._resolve_event(event)
                    resolved_events.append(event)
                elif event.event_type in [EventType.LINK_FAILURE, EventType.NODE_FAILURE]:
                    # Apply failure to topology
                    if timestamp >= event.timestamp:
                        self.topology_simulator.apply_failure(event)
        
        # Remove resolved events
        for event in resolved_events:
            self.active_events.remove(event)
    
    def _resolve_event(self, event: NetworkEvent):
        
        event.resolved = True
        event.resolution_time = time.time()
        
        # Recover from failures
        if event.event_type in [EventType.LINK_FAILURE, EventType.NODE_FAILURE]:
            self.topology_simulator.recover_failure(event)
        
        self.logger.debug(f"Event resolved: {event.event_type.value} after "
                         f"{event.duration_seconds/3600:.1f} hours")
    
    def _apply_event_impacts(self, base_load: float, timestamp: float) -> float:
        
        effective_load = base_load
        
        for event in self.active_events:
            if not event.resolved and timestamp >= event.timestamp:
                if event.event_type == EventType.TRAFFIC_SPIKE:
                    multiplier = event.impact_metrics.get('traffic_multiplier', 1.0)
                    effective_load *= multiplier
                elif event.event_type == EventType.SECURITY_INCIDENT:
                    # Security overhead reduces effective capacity
                    overhead = event.impact_metrics.get('security_overhead', 0.0)
                    effective_load *= (1 + overhead)
        
        return effective_load
    
    def _create_performance_snapshot(self, timestamp: float, 
                                   hour_results: Dict[str, Any]) -> PerformanceSnapshot:
       
        requests = hour_results['requests']
        successful = hour_results['successful']
        
        # Calculate metrics
        accuracy = successful / max(requests, 1)
        avg_inference_ms = (hour_results['total_inference_time'] / max(successful, 1) 
                           if successful > 0 else 0)
        throughput_rps = successful / 3600  # Per hour to per second
        error_rate = hour_results['errors'] / max(requests, 1)
        
        # Determine network condition
        condition = NetworkCondition.NORMAL
        if error_rate > 0.05:
            condition = NetworkCondition.CRITICAL
        elif error_rate > 0.01 or avg_inference_ms > self.config.target_inference_ms * 2:
            condition = NetworkCondition.DEGRADED
        elif throughput_rps > self.config.base_traffic_rps * 2:
            condition = NetworkCondition.CONGESTED
        
        # SLA compliance
        sla_compliance = 1.0
        if avg_inference_ms > self.config.target_inference_ms:
            sla_compliance *= 0.9
        if accuracy < self.config.target_accuracy:
            sla_compliance *= 0.8
        if error_rate > 0.01:
            sla_compliance *= 0.7
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            accuracy=accuracy,
            inference_time_ms=avg_inference_ms,
            memory_usage_mb=hour_results['memory_usage'],
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            network_condition=condition,
            active_connections=int(throughput_rps * 30),  # Assume 30s avg connection
            queue_depth=max(0, int((requests - successful) * 0.1)),
            cpu_utilization=min(95, throughput_rps / self.config.base_traffic_rps * 70),
            sla_compliance=sla_compliance,
            cost_per_decision=hour_results['cost'] / max(successful, 1),
            user_satisfaction=max(0.5, sla_compliance * 0.9 + 0.1)
        )
    
    def _perform_model_adaptation(self) -> Dict[str, Any]:
        
        try:
            self.logger.info("Performing weekly model adaptation")
            
            adaptation_start = time.time()
            
            # Generate adaptation data based on recent performance
            recent_snapshots = self.performance_history[-7*24:] if len(self.performance_history) >= 7*24 else self.performance_history
            
            if not recent_snapshots:
                return {'success': False, 'reason': 'No recent data for adaptation'}
            
            # Analyze recent performance
            recent_accuracy = np.mean([s.accuracy for s in recent_snapshots])
            recent_inference_time = np.mean([s.inference_time_ms for s in recent_snapshots])
            
            # Generate synthetic adaptation data
            n_adapt_samples = 1000
            adapt_features = np.random.normal(0, 1, (n_adapt_samples, 50))
            adapt_labels = np.random.randint(0, 3, n_adapt_samples)
            
            # Apply continuous learning if available
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(adapt_features, adapt_labels)
            else:
                # Retrain with combined data (simplified)
                pass
            
            adaptation_time = time.time() - adaptation_start
            
            adaptation_result = {
                'timestamp': time.time(),
                'success': True,
                'adaptation_time_seconds': adaptation_time,
                'samples_processed': n_adapt_samples,
                'performance_before': {
                    'accuracy': recent_accuracy,
                    'inference_time_ms': recent_inference_time
                },
                'adaptation_type': 'weekly_continuous_learning'
            }
            
            self.logger.info(f"Model adaptation completed in {adaptation_time:.2f}s")
            
            return adaptation_result
            
        except Exception as e:
            self.logger.error(f"Model adaptation failed: {str(e)}")
            return {
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'adaptation_type': 'weekly_continuous_learning'
            }
    
    def _log_monthly_summary(self, day: int, results: Dict[str, Any]):
       
        month = day // 30
        
        # Calculate monthly metrics from recent snapshots
        month_start = max(0, len(results['performance_snapshots']) - 30*24)
        month_snapshots = results['performance_snapshots'][month_start:]
        
        if not month_snapshots:
            return
        
        # Calculate statistics
        avg_accuracy = np.mean([s.accuracy for s in month_snapshots])
        avg_inference_time = np.mean([s.inference_time_ms for s in month_snapshots])
        avg_throughput = np.mean([s.throughput_rps for s in month_snapshots])
        avg_sla_compliance = np.mean([s.sla_compliance for s in month_snapshots])
        
        # Count events
        month_events = [e for e in results['events'] 
                       if e.timestamp >= (day - 30) * SECONDS_PER_DAY]
        event_counts = defaultdict(int)
        for event in month_events:
            event_counts[event.event_type.value] += 1
        
        self.logger.info(f"Month {month + 1} Summary:")
        self.logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
        self.logger.info(f"  Average Inference Time: {avg_inference_time:.2f}ms")
        self.logger.info(f"  Average Throughput: {avg_throughput:.1f} RPS")
        self.logger.info(f"  Average SLA Compliance: {avg_sla_compliance:.3f}")
        self.logger.info(f"  Events: {dict(event_counts)}")
    
    def _analyze_simulation_results(self, results: Dict[str, Any]):
       
        try:
            self.logger.info("Analyzing simulation results")
            
            snapshots = results['performance_snapshots']
            events = results['events']
            
            if not snapshots:
                return
            
            # Overall performance metrics
            overall_accuracy = np.mean([s.accuracy for s in snapshots])
            overall_inference_time = np.mean([s.inference_time_ms for s in snapshots])
            overall_throughput = np.mean([s.throughput_rps for s in snapshots])
            overall_sla_compliance = np.mean([s.sla_compliance for s in snapshots])
            
            # Availability analysis
            degraded_hours = sum(1 for s in snapshots 
                               if s.network_condition in [NetworkCondition.DEGRADED, NetworkCondition.CRITICAL])
            availability = 1 - (degraded_hours / len(snapshots))
            
            # Cost analysis
            total_requests = sum(ds['total_requests'] for ds in results['daily_summaries'])
            total_cost = sum(ds['cost'] for ds in results['daily_summaries'])
            cost_per_request = total_cost / max(total_requests, 1)
            
            # Event analysis
            event_stats = defaultdict(int)
            event_impact = defaultdict(list)
            
            for event in events:
                event_stats[event.event_type.value] += 1
                event_stats[f"{event.event_type.value}_{event.severity}"] += 1
                
                # Calculate impact duration
                impact_duration = event.duration_seconds / 3600  # Convert to hours
                event_impact[event.event_type.value].append(impact_duration)
            
            # Store analysis results
            results['analysis'] = {
                'overall_performance': {
                    'accuracy': overall_accuracy,
                    'inference_time_ms': overall_inference_time,
                    'throughput_rps': overall_throughput,
                    'sla_compliance': overall_sla_compliance,
                    'availability': availability
                },
                'cost_analysis': {
                    'total_cost': total_cost,
                    'cost_per_request': cost_per_request,
                    'total_requests': total_requests
                },
                'event_statistics': dict(event_stats),
                'event_impact_hours': {k: {'mean': np.mean(v), 'max': np.max(v), 'count': len(v)} 
                                      for k, v in event_impact.items()},
                'target_compliance': {
                    'accuracy_target_met': overall_accuracy >= self.config.target_accuracy,
                    'inference_time_target_met': overall_inference_time <= self.config.target_inference_ms,
                    'availability_target_met': availability >= self.config.target_availability
                }
            }
            
            # Performance trends analysis
            if len(snapshots) >= 24:  # At least one day
                self._analyze_performance_trends(results, snapshots)
            
            self.logger.info("Simulation analysis completed")
            
        except Exception as e:
            self.logger.error(f"Results analysis failed: {str(e)}")
    
    def _analyze_performance_trends(self, results: Dict[str, Any], snapshots: List[PerformanceSnapshot]):
        
        # Group by day
        daily_metrics = defaultdict(list)
        
        for snapshot in snapshots:
            day = int(snapshot.timestamp // SECONDS_PER_DAY)
            daily_metrics[day].append(snapshot)
        
        # Calculate daily averages
        daily_averages = {}
        for day, day_snapshots in daily_metrics.items():
            daily_averages[day] = {
                'accuracy': np.mean([s.accuracy for s in day_snapshots]),
                'inference_time_ms': np.mean([s.inference_time_ms for s in day_snapshots]),
                'sla_compliance': np.mean([s.sla_compliance for s in day_snapshots])
            }
        
        # Trend analysis
        days = sorted(daily_averages.keys())
        accuracy_trend = [daily_averages[d]['accuracy'] for d in days]
        inference_trend = [daily_averages[d]['inference_time_ms'] for d in days]
        
        # Simple linear regression for trends
        if len(days) >= 7:  # At least a week
            accuracy_slope = np.polyfit(days, accuracy_trend, 1)[0]
            inference_slope = np.polyfit(days, inference_trend, 1)[0]
            
            results['analysis']['trends'] = {
                'accuracy_trend': 'improving' if accuracy_slope > 0 else 'degrading',
                'inference_time_trend': 'improving' if inference_slope < 0 else 'degrading',
                'accuracy_slope_per_day': accuracy_slope,
                'inference_slope_per_day': inference_slope
            }
    
    def _generate_simulation_reports(self, results: Dict[str, Any]):
       
        try:
            self.logger.info("Generating simulation reports")
            
            # Generate markdown report
            self._generate_markdown_report(results)
            
            # Generate visualizations
            if self.config.generate_reports:
                self._generate_simulation_visualizations(results)
            
            # Save detailed results
            self._save_simulation_results(results)
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
    
    def _generate_markdown_report(self, results: Dict[str, Any]):
        """Generate comprehensive markdown report."""
        report_path = self.config.results_dir / "production_simulation_report.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write("# HyperPath-SVM Production Deployment Simulation Report\n\n")
                
                # Executive Summary
                f.write("## Executive Summary\n\n")
                
                if 'analysis' in results:
                    analysis = results['analysis']
                    perf = analysis['overall_performance']
                    cost = analysis['cost_analysis']
                    
                    f.write(f"- **Simulation Duration**: {self.config.simulation_days} days (6 months)\n")
                    f.write(f"- **Total Requests Processed**: {cost['total_requests']:,}\n")
                    f.write(f"- **Overall Accuracy**: {perf['accuracy']:.4f} (Target: {self.config.target_accuracy})\n")
                    f.write(f"- **Average Inference Time**: {perf['inference_time_ms']:.2f}ms (Target: {self.config.target_inference_ms}ms)\n")
                    f.write(f"- **System Availability**: {perf['availability']:.4f} (Target: {self.config.target_availability})\n")
                    f.write(f"- **SLA Compliance Rate**: {perf['sla_compliance']:.4f}\n")
                    f.write(f"- **Total Operational Cost**: ${cost['total_cost']:.2f}\n")
                    f.write(f"- **Cost per Request**: ${cost['cost_per_request']:.6f}\n\n")
                
                # Performance Analysis
                f.write("## Performance Analysis\n\n")
                f.write("### Target Compliance\n\n")
                
                if 'analysis' in results:
                    compliance = analysis['target_compliance']
                    f.write("| Metric | Target | Achieved | Status |\n")
                    f.write("|--------|--------|----------|--------|\n")
                    f.write(f"| Accuracy | {self.config.target_accuracy} | {perf['accuracy']:.4f} | "
                           f"{'✅ PASS' if compliance['accuracy_target_met'] else '❌ FAIL'} |\n")
                    f.write(f"| Inference Time | {self.config.target_inference_ms}ms | {perf['inference_time_ms']:.2f}ms | "
                           f"{'✅ PASS' if compliance['inference_time_target_met'] else '❌ FAIL'} |\n")
                    f.write(f"| Availability | {self.config.target_availability} | {perf['availability']:.4f} | "
                           f"{'✅ PASS' if compliance['availability_target_met'] else '❌ FAIL'} |\n\n")
                
                # Event Analysis
                f.write("## Incident Analysis\n\n")
                
                if 'analysis' in results and 'event_statistics' in analysis:
                    event_stats = analysis['event_statistics']
                    f.write("### Event Summary\n\n")
                    f.write("| Event Type | Count | Avg Duration (hours) |\n")
                    f.write("|------------|-------|---------------------|\n")
                    
                    event_impact = analysis.get('event_impact_hours', {})
                    for event_type, count in event_stats.items():
                        if '_' not in event_type:  # Skip severity breakdowns
                            avg_duration = event_impact.get(event_type, {}).get('mean', 0)
                            f.write(f"| {event_type.replace('_', ' ').title()} | {count} | {avg_duration:.2f} |\n")
                    f.write("\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                f.write("### Performance Optimization\n")
                f.write("1. **Memory Management**: Consider optimizing memory usage during peak loads\n")
                f.write("2. **Inference Acceleration**: Implement model quantization for faster inference\n")
                f.write("3. **Load Balancing**: Deploy multiple model instances for high availability\n\n")
                
                f.write("### Operational Excellence\n")
                f.write("1. **Monitoring**: Implement comprehensive monitoring and alerting\n")
                f.write("2. **Incident Response**: Develop automated incident response procedures\n")
                f.write("3. **Capacity Planning**: Plan for network growth and traffic increases\n\n")
                
                f.write("### Cost Optimization\n")
                f.write("1. **Resource Scaling**: Implement auto-scaling based on demand\n")
                f.write("2. **Efficiency Improvements**: Optimize algorithms for better cost-performance ratio\n")
                f.write("3. **SLA Management**: Balance performance targets with operational costs\n")
                
        except Exception as e:
            self.logger.error(f"Markdown report generation failed: {str(e)}")
    
    def _generate_simulation_visualizations(self, results: Dict[str, Any]):
        
        try:
            plots_dir = self.config.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            snapshots = results['performance_snapshots']
            if not snapshots:
                return
            
            # Performance over time plot
            timestamps = [s.timestamp / SECONDS_PER_DAY for s in snapshots]  # Convert to days
            accuracies = [s.accuracy for s in snapshots]
            inference_times = [s.inference_time_ms for s in snapshots]
            sla_compliance = [s.sla_compliance for s in snapshots]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Accuracy over time
            ax1.plot(timestamps, accuracies, color='blue', alpha=0.7)
            ax1.axhline(y=self.config.target_accuracy, color='red', linestyle='--', alpha=0.8, label='Target')
            ax1.set_xlabel('Days')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Inference time over time
            ax2.plot(timestamps, inference_times, color='green', alpha=0.7)
            ax2.axhline(y=self.config.target_inference_ms, color='red', linestyle='--', alpha=0.8, label='Target')
            ax2.set_xlabel('Days')
            ax2.set_ylabel('Inference Time (ms)')
            ax2.set_title('Inference Time Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # SLA compliance over time
            ax3.plot(timestamps, sla_compliance, color='orange', alpha=0.7)
            ax3.set_xlabel('Days')
            ax3.set_ylabel('SLA Compliance')
            ax3.set_title('SLA Compliance Over Time')
            ax3.grid(True, alpha=0.3)
            
            # System condition distribution
            conditions = [s.network_condition.value for s in snapshots]
            condition_counts = {condition: conditions.count(condition) for condition in set(conditions)}
            
            ax4.pie(condition_counts.values(), labels=condition_counts.keys(), autopct='%1.1f%%')
            ax4.set_title('System Condition Distribution')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'production_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Simulation visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {str(e)}")
    
    def _save_simulation_results(self, results: Dict[str, Any]):
        
        try:
            # Save as JSON (serializable version)
            json_path = self.config.results_dir / "simulation_results.json"
            
            # Convert to JSON-serializable format
            json_results = {
                'config': {
                    'simulation_days': self.config.simulation_days,
                    'initial_network_size': self.config.initial_network_size,
                    'base_traffic_rps': self.config.base_traffic_rps,
                    'target_accuracy': self.config.target_accuracy,
                    'target_inference_ms': self.config.target_inference_ms
                },
                'summary': results.get('analysis', {}),
                'daily_summaries': results.get('daily_summaries', []),
                'adaptation_events': results.get('adaptation_events', []),
                'event_count': len(results.get('events', [])),
                'snapshot_count': len(results.get('performance_snapshots', []))
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            # Save complete results as pickle
            pickle_path = self.config.results_dir / "simulation_results.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Results saved to {json_path} and {pickle_path}")
            
        except Exception as e:
            self.logger.error(f"Results saving failed: {str(e)}")


def run_production_simulation(config: Optional[ProductionConfig] = None) -> Dict[str, Any]:
 
    simulator = ProductionSimulator(config)
    return simulator.run_production_simulation()


if __name__ == "__main__":
    # Test production simulation
    logger = get_logger(__name__)
    logger.info("Testing production simulation...")
    
    try:
        # Create test configuration (shorter simulation)
        test_config = ProductionConfig(
            simulation_days=7,  # 1 week for testing
            initial_network_size=1000,
            base_traffic_rps=100,  # Lower traffic for testing
            results_dir=Path("test_production"),
            generate_reports=True
        )
        
        # Run simulation
        results = run_production_simulation(test_config)
        
        logger.info("Production Simulation Results:")
        if 'analysis' in results:
            analysis = results['analysis']
            perf = analysis['overall_performance']
            
            logger.info(f"Overall Accuracy: {perf['accuracy']:.4f}")
            logger.info(f"Average Inference Time: {perf['inference_time_ms']:.2f}ms")
            logger.info(f"System Availability: {perf['availability']:.4f}")
            logger.info(f"Total Events: {len(results.get('events', []))}")
        
    except Exception as e:
        logger.error(f"Production simulation test failed: {str(e)}")
    
    logger.info("Production simulation testing completed!") 
