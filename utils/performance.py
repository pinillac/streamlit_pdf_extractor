"""
Performance Monitor Utility
Tracks and optimizes performance metrics for PDF processing
"""

import time
import psutil
import streamlit as st
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import threading
from collections import deque, defaultdict
import logging
import json
from pathlib import Path

class PerformanceMonitor:
    """Monitors and tracks performance metrics"""
    
    def __init__(self, history_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.current_operations = {}
        self.completed_operations = defaultdict(list)
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_operation(self, operation_id: str, operation_type: str, 
                       metadata: Dict[str, Any] = None) -> None:
        """
        Start tracking an operation
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation (e.g., 'file_processing', 'extraction')
            metadata: Additional operation metadata
        """
        self.current_operations[operation_id] = {
            'type': operation_type,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'metadata': metadata or {},
            'checkpoints': []
        }
        
        self.logger.debug(f"Started operation {operation_id} ({operation_type})")
    
    def add_checkpoint(self, operation_id: str, checkpoint_name: str, 
                      data: Dict[str, Any] = None) -> None:
        """
        Add a checkpoint to an ongoing operation
        
        Args:
            operation_id: Operation identifier
            checkpoint_name: Name of checkpoint
            data: Checkpoint data
        """
        if operation_id in self.current_operations:
            checkpoint = {
                'name': checkpoint_name,
                'time': time.time(),
                'memory': psutil.Process().memory_info().rss / 1024 / 1024,
                'data': data or {}
            }
            self.current_operations[operation_id]['checkpoints'].append(checkpoint)
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     result_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        End tracking an operation
        
        Args:
            operation_id: Operation identifier
            success: Whether operation was successful
            result_data: Operation results
            
        Returns:
            Operation metrics
        """
        if operation_id not in self.current_operations:
            self.logger.warning(f"Operation {operation_id} not found")
            return {}
        
        operation = self.current_operations.pop(operation_id)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        duration = end_time - operation['start_time']
        memory_used = end_memory - operation['start_memory']
        
        # Create operation record
        operation_record = {
            'operation_id': operation_id,
            'type': operation['type'],
            'start_time': datetime.fromtimestamp(operation['start_time']),
            'end_time': datetime.fromtimestamp(end_time),
            'duration_seconds': duration,
            'memory_used_mb': memory_used,
            'success': success,
            'metadata': operation['metadata'],
            'result_data': result_data or {},
            'checkpoints': operation['checkpoints']
        }
        
        # Store in history
        self.completed_operations[operation['type']].append(operation_record)
        self.metrics_history.append(operation_record)
        
        self.logger.info(
            f"Completed operation {operation_id} ({operation['type']}) "
            f"in {duration:.2f}s using {memory_used:.1f}MB memory"
        )
        
        return operation_record
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # Process specific
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'timestamp': datetime.now(),
            'system': {
                'cpu_percent': cpu_percent,
                'cpu_count': psutil.cpu_count(),
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            },
            'process': {
                'memory_rss_mb': process_memory.rss / (1024**2),
                'memory_vms_mb': process_memory.vms / (1024**2),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'open_files': len(process.open_files())
            },
            'network': {
                'bytes_sent_mb': net_io.bytes_sent / (1024**2),
                'bytes_recv_mb': net_io.bytes_recv / (1024**2)
            },
            'operations': {
                'active': len(self.current_operations),
                'completed': sum(len(ops) for ops in self.completed_operations.values())
            }
        }
    
    def get_operation_statistics(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations
        
        Args:
            operation_type: Filter by operation type (None for all)
            
        Returns:
            Operation statistics
        """
        if operation_type:
            operations = self.completed_operations.get(operation_type, [])
        else:
            operations = list(self.metrics_history)
        
        if not operations:
            return {
                'count': 0,
                'success_rate': 0,
                'avg_duration': 0,
                'avg_memory': 0
            }
        
        successful = [op for op in operations if op['success']]
        failed = [op for op in operations if not op['success']]
        
        durations = [op['duration_seconds'] for op in operations]
        memory_usage = [op['memory_used_mb'] for op in operations]
        
        return {
            'count': len(operations),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(operations) * 100,
            'duration': {
                'total': sum(durations),
                'average': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations)
            },
            'memory': {
                'average': sum(memory_usage) / len(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage),
                'total': sum(memory_usage)
            },
            'throughput': {
                'operations_per_minute': len(operations) / (sum(durations) / 60) if sum(durations) > 0 else 0
            }
        }
    
    def get_performance_trends(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance trends over time window
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Performance trends
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_operations = [
            op for op in self.metrics_history 
            if op['start_time'] >= cutoff_time
        ]
        
        if not recent_operations:
            return {'message': 'No operations in time window'}
        
        # Group by time buckets (5-minute intervals)
        bucket_size = 5  # minutes
        buckets = defaultdict(list)
        
        for op in recent_operations:
            bucket = op['start_time'].replace(
                minute=(op['start_time'].minute // bucket_size) * bucket_size,
                second=0,
                microsecond=0
            )
            buckets[bucket].append(op)
        
        # Calculate trends
        trends = []
        for bucket_time in sorted(buckets.keys()):
            bucket_ops = buckets[bucket_time]
            
            trends.append({
                'time': bucket_time,
                'operations': len(bucket_ops),
                'avg_duration': sum(op['duration_seconds'] for op in bucket_ops) / len(bucket_ops),
                'avg_memory': sum(op['memory_used_mb'] for op in bucket_ops) / len(bucket_ops),
                'success_rate': sum(1 for op in bucket_ops if op['success']) / len(bucket_ops) * 100
            })
        
        return {
            'window_minutes': window_minutes,
            'bucket_size_minutes': bucket_size,
            'trends': trends,
            'summary': {
                'total_operations': len(recent_operations),
                'avg_operations_per_bucket': len(recent_operations) / len(buckets),
                'performance_improving': trends[-1]['avg_duration'] < trends[0]['avg_duration'] if len(trends) > 1 else None
            }
        }
    
    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze checkpoint data
        checkpoint_durations = defaultdict(list)
        
        for op in self.metrics_history:
            if 'checkpoints' in op and len(op['checkpoints']) > 1:
                for i in range(1, len(op['checkpoints'])):
                    prev_checkpoint = op['checkpoints'][i-1]
                    curr_checkpoint = op['checkpoints'][i]
                    
                    duration = curr_checkpoint['time'] - prev_checkpoint['time']
                    checkpoint_name = f"{prev_checkpoint['name']} -> {curr_checkpoint['name']}"
                    
                    checkpoint_durations[checkpoint_name].append(duration)
        
        # Find slow checkpoints
        for checkpoint_name, durations in checkpoint_durations.items():
            avg_duration = sum(durations) / len(durations)
            
            if avg_duration > 5.0:  # More than 5 seconds
                bottlenecks.append({
                    'type': 'slow_checkpoint',
                    'name': checkpoint_name,
                    'avg_duration': avg_duration,
                    'occurrences': len(durations),
                    'severity': 'high' if avg_duration > 10 else 'medium'
                })
        
        # Check for memory leaks
        memory_trend = []
        for op in sorted(self.metrics_history, key=lambda x: x['start_time'])[-20:]:
            memory_trend.append(op['memory_used_mb'])
        
        if len(memory_trend) > 10:
            # Simple linear regression to check trend
            x = list(range(len(memory_trend)))
            y = memory_trend
            
            # Calculate slope
            n = len(x)
            slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                   (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
            
            if slope > 10:  # Memory increasing by >10MB per operation
                bottlenecks.append({
                    'type': 'memory_leak',
                    'name': 'Increasing memory usage',
                    'slope_mb_per_operation': slope,
                    'severity': 'high' if slope > 50 else 'medium'
                })
        
        # Check for high failure rates
        stats = self.get_operation_statistics()
        if stats['count'] > 10 and stats['success_rate'] < 90:
            bottlenecks.append({
                'type': 'high_failure_rate',
                'name': 'Operations failing frequently',
                'failure_rate': 100 - stats['success_rate'],
                'failed_count': stats['failed'],
                'severity': 'high' if stats['success_rate'] < 50 else 'medium'
            })
        
        return bottlenecks
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'summary': {
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'total_operations': sum(len(ops) for ops in self.completed_operations.values()),
                'operation_types': list(self.completed_operations.keys()),
                'current_metrics': self.get_current_metrics()
            },
            'statistics': {
                type_name: self.get_operation_statistics(type_name)
                for type_name in self.completed_operations.keys()
            },
            'trends': self.get_performance_trends(),
            'bottlenecks': self.get_bottlenecks(),
            'recommendations': self.generate_recommendations()
        }
    
    def generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check current metrics
        metrics = self.get_current_metrics()
        
        # CPU recommendations
        if metrics['system']['cpu_percent'] > 80:
            recommendations.append({
                'category': 'CPU',
                'issue': 'High CPU usage',
                'recommendation': 'Reduce worker threads or chunk size',
                'priority': 'high'
            })
        
        # Memory recommendations
        if metrics['system']['memory_percent'] > 80:
            recommendations.append({
                'category': 'Memory',
                'issue': 'High memory usage',
                'recommendation': 'Reduce memory limit or enable more aggressive garbage collection',
                'priority': 'high'
            })
        
        # Operation-specific recommendations
        for op_type, stats in self.get_operation_statistics().items():
            if isinstance(stats, dict) and stats.get('count', 0) > 0:
                if stats.get('duration', {}).get('average', 0) > 60:
                    recommendations.append({
                        'category': 'Performance',
                        'issue': f'Slow {op_type} operations',
                        'recommendation': 'Consider optimizing patterns or increasing chunk size',
                        'priority': 'medium'
                    })
        
        # Bottleneck-based recommendations
        bottlenecks = self.get_bottlenecks()
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'memory_leak':
                recommendations.append({
                    'category': 'Memory',
                    'issue': 'Potential memory leak detected',
                    'recommendation': 'Enable periodic garbage collection and monitor memory usage',
                    'priority': 'high'
                })
        
        return recommendations
    
    def export_metrics(self, filepath: Path) -> None:
        """Export metrics to file"""
        export_data = {
            'export_time': datetime.now().isoformat(),
            'metrics_history': [
                {
                    **record,
                    'start_time': record['start_time'].isoformat(),
                    'end_time': record['end_time'].isoformat()
                }
                for record in self.metrics_history
            ],
            'statistics': {
                op_type: self.get_operation_statistics(op_type)
                for op_type in self.completed_operations.keys()
            },
            'performance_report': self.generate_performance_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported metrics to {filepath}")
    
    def reset_metrics(self, keep_history: bool = False) -> None:
        """Reset performance metrics"""
        if not keep_history:
            self.metrics_history.clear()
        
        self.current_operations.clear()
        self.completed_operations.clear()
        self.start_time = time.time()
        
        self.logger.info("Performance metrics reset")
    
    def create_monitoring_callback(self, update_interval: float = 1.0) -> Callable:
        """
        Create a callback function for real-time monitoring
        
        Args:
            update_interval: Update interval in seconds
            
        Returns:
            Monitoring callback function
        """
        def monitor_callback():
            while self.monitoring:
                metrics = self.get_current_metrics()
                
                # Store in session state for UI access
                if hasattr(st, 'session_state'):
                    st.session_state['performance_metrics'] = metrics
                
                time.sleep(update_interval)
        
        return monitor_callback
    
    def start_monitoring(self, callback: Optional[Callable] = None) -> None:
        """Start continuous monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        if callback is None:
            callback = self.create_monitoring_callback()
        
        self.monitor_thread = threading.Thread(target=callback, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")