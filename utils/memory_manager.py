"""
Memory Manager Utility
Optimizes memory usage during PDF processing
"""

import gc
import psutil
import streamlit as st
from typing import Dict, Any, Optional, Callable
import threading
import time
import logging
from contextlib import contextmanager
import os
import sys

class MemoryOptimizer:
    """Manages memory optimization for PDF processing"""
    
    def __init__(self, max_memory_mb: int = 1024, gc_threshold_percent: float = 80):
        self.max_memory_mb = max_memory_mb
        self.gc_threshold_percent = gc_threshold_percent
        self.process = psutil.Process()
        self.initial_memory = self.get_current_memory_mb()
        self.peak_memory = self.initial_memory
        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def get_memory_percent(self) -> float:
        """Get current memory usage percentage"""
        return self.process.memory_percent()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information"""
        vm = psutil.virtual_memory()
        current_mb = self.get_current_memory_mb()
        
        return {
            'current_mb': round(current_mb, 2),
            'peak_mb': round(self.peak_memory, 2),
            'available_mb': round(vm.available / 1024 / 1024, 2),
            'total_mb': round(vm.total / 1024 / 1024, 2),
            'percent_used': round(vm.percent, 1),
            'process_percent': round(self.get_memory_percent(), 1),
            'threshold_mb': self.max_memory_mb,
            'gc_threshold_percent': self.gc_threshold_percent
        }
    
    def check_memory_pressure(self) -> bool:
        """
        Check if memory usage is approaching limits
        
        Returns:
            True if memory pressure is high
        """
        current_mb = self.get_current_memory_mb()
        self.peak_memory = max(self.peak_memory, current_mb)
        
        # Check process memory limit
        if current_mb > self.max_memory_mb:
            self.logger.warning(f"Memory usage ({current_mb:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
            return True
        
        # Check system memory
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.gc_threshold_percent:
            self.logger.warning(f"System memory usage high: {memory_percent:.1f}%")
            return True
        
        return False
    
    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory optimization
        
        Args:
            force: Force garbage collection regardless of thresholds
            
        Returns:
            Optimization results
        """
        before_mb = self.get_current_memory_mb()
        
        # Check if optimization is needed
        if not force and not self.check_memory_pressure():
            return {
                'optimized': False,
                'reason': 'Memory usage within limits',
                'current_mb': before_mb
            }
        
        # Perform garbage collection
        collected = gc.collect()
        
        # Force collection of higher generations
        for generation in range(gc.get_count().__len__()):
            gc.collect(generation)
        
        # Clear Streamlit cache if available
        try:
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
        except Exception as e:
            self.logger.debug(f"Could not clear Streamlit cache: {e}")
        
        after_mb = self.get_current_memory_mb()
        freed_mb = before_mb - after_mb
        
        return {
            'optimized': True,
            'before_mb': round(before_mb, 2),
            'after_mb': round(after_mb, 2),
            'freed_mb': round(freed_mb, 2),
            'objects_collected': collected,
            'freed_percent': round((freed_mb / before_mb) * 100, 1) if before_mb > 0 else 0
        }
    
    def start_monitoring(self, callback: Optional[Callable] = None, interval: float = 5.0):
        """
        Start continuous memory monitoring
        
        Args:
            callback: Optional callback function for memory updates
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                memory_info = self.get_memory_info()
                
                # Check for memory pressure
                if self.check_memory_pressure():
                    self.optimize_memory()
                
                # Call callback if provided
                if callback:
                    callback(memory_info)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Memory monitoring stopped")
    
    @contextmanager
    def memory_limit_context(self, limit_mb: Optional[int] = None):
        """
        Context manager for temporary memory limit
        
        Args:
            limit_mb: Temporary memory limit in MB
        """
        original_limit = self.max_memory_mb
        
        if limit_mb:
            self.max_memory_mb = limit_mb
        
        try:
            yield self
        finally:
            self.max_memory_mb = original_limit
            self.optimize_memory()  # Clean up after context
    
    def estimate_memory_requirement(self, file_size_mb: float, 
                                  page_count: int = None) -> Dict[str, Any]:
        """
        Estimate memory requirement for processing
        
        Args:
            file_size_mb: Size of PDF file in MB
            page_count: Number of pages (optional)
            
        Returns:
            Memory requirement estimation
        """
        # Base memory overhead
        base_memory = 100  # MB
        
        # File size factor (memory typically 2-5x file size)
        file_memory = file_size_mb * 3
        
        # Page processing memory (if known)
        page_memory = 0
        if page_count:
            # Approximately 1-2 MB per page for processing
            page_memory = page_count * 1.5
        
        # Total estimation
        estimated_memory = base_memory + file_memory + page_memory
        
        # Add safety margin
        recommended_memory = estimated_memory * 1.5
        
        return {
            'estimated_mb': round(estimated_memory, 0),
            'recommended_mb': round(recommended_memory, 0),
            'base_overhead_mb': base_memory,
            'file_factor': 3,
            'per_page_mb': 1.5 if page_count else None,
            'safety_margin': 1.5,
            'available_mb': round(self.get_available_memory_mb(), 0),
            'sufficient': self.get_available_memory_mb() > recommended_memory
        }
    
    def get_optimal_chunk_size(self, file_size_mb: float, 
                             available_memory_mb: Optional[float] = None) -> int:
        """
        Calculate optimal chunk size based on file size and memory
        
        Args:
            file_size_mb: Size of PDF file
            available_memory_mb: Available memory (auto-detected if None)
            
        Returns:
            Optimal chunk size in pages
        """
        if available_memory_mb is None:
            available_memory_mb = self.get_available_memory_mb()
        
        # Base calculations
        if file_size_mb < 10:
            # Small files can be processed in larger chunks
            base_chunk = 50
        elif file_size_mb < 50:
            # Medium files need moderate chunks
            base_chunk = 20
        elif file_size_mb < 100:
            # Large files need smaller chunks
            base_chunk = 10
        else:
            # Very large files need minimal chunks
            base_chunk = 5
        
        # Adjust based on available memory
        memory_factor = min(available_memory_mb / 1024, 2.0)  # Cap at 2x
        
        optimal_chunk = int(base_chunk * memory_factor)
        
        # Ensure reasonable bounds
        return max(1, min(optimal_chunk, 100))
    
    def monitor_memory_usage(self, func: Callable) -> Callable:
        """
        Decorator to monitor memory usage of a function
        
        Args:
            func: Function to monitor
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            # Record initial state
            start_memory = self.get_current_memory_mb()
            start_time = time.time()
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Record final state
                end_memory = self.get_current_memory_mb()
                end_time = time.time()
                
                # Log memory usage
                memory_used = end_memory - start_memory
                duration = end_time - start_time
                
                self.logger.info(
                    f"{func.__name__} completed: "
                    f"Memory: {memory_used:.1f}MB, "
                    f"Time: {duration:.2f}s, "
                    f"Peak: {self.peak_memory:.1f}MB"
                )
                
                return result
                
            except MemoryError as e:
                self.logger.error(f"{func.__name__} failed with MemoryError")
                self.optimize_memory(force=True)
                raise
            
        return wrapper
    
    def create_memory_profile(self) -> Dict[str, Any]:
        """
        Create detailed memory profile
        
        Returns:
            Memory profile data
        """
        import tracemalloc
        
        # Start tracing
        tracemalloc.start()
        
        # Get current snapshot
        snapshot = tracemalloc.take_snapshot()
        
        # Get top memory consumers
        top_stats = snapshot.statistics('lineno')[:10]
        
        # Format statistics
        top_consumers = []
        for stat in top_stats:
            top_consumers.append({
                'file': stat.traceback.format()[0],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        # Get garbage collection stats
        gc_stats = {
            f'generation_{i}': {
                'count': gc.get_count()[i] if i < len(gc.get_count()) else 0,
                'threshold': gc.get_threshold()[i] if i < len(gc.get_threshold()) else 0
            }
            for i in range(3)
        }
        
        # Stop tracing
        tracemalloc.stop()
        
        return {
            'current_memory_mb': round(self.get_current_memory_mb(), 2),
            'peak_memory_mb': round(self.peak_memory, 2),
            'available_system_mb': round(self.get_available_memory_mb(), 2),
            'top_consumers': top_consumers,
            'gc_stats': gc_stats,
            'python_version': sys.version,
            'process_id': os.getpid()
        }
    
    def suggest_configuration(self, files_info: list) -> Dict[str, Any]:
        """
        Suggest optimal configuration based on files
        
        Args:
            files_info: List of file information dictionaries
            
        Returns:
            Configuration suggestions
        """
        total_size_mb = sum(f.get('size_mb', 0) for f in files_info)
        max_size_mb = max(f.get('size_mb', 0) for f in files_info)
        file_count = len(files_info)
        
        available_memory = self.get_available_memory_mb()
        
        # Memory limit suggestion
        # Use 50% of available memory, but at least 512MB and at most 4GB
        suggested_memory_limit = max(
            512,
            min(4096, int(available_memory * 0.5))
        )
        
        # Chunk size suggestion
        suggested_chunk_size = self.get_optimal_chunk_size(max_size_mb)
        
        # Worker threads suggestion
        cpu_count = psutil.cpu_count()
        if file_count > 10 and total_size_mb < 500:
            suggested_workers = min(cpu_count, 8)
        elif file_count > 5:
            suggested_workers = min(cpu_count, 4)
        else:
            suggested_workers = 2
        
        # Timeout suggestion
        if max_size_mb > 100:
            suggested_timeout = 900  # 15 minutes
        elif max_size_mb > 50:
            suggested_timeout = 600  # 10 minutes
        else:
            suggested_timeout = 300  # 5 minutes
        
        return {
            'memory_limit_mb': suggested_memory_limit,
            'chunk_size': suggested_chunk_size,
            'max_workers': suggested_workers,
            'processing_timeout': suggested_timeout,
            'rationale': {
                'memory': f"50% of available {available_memory:.0f}MB",
                'chunk': f"Optimized for {max_size_mb:.1f}MB max file size",
                'workers': f"{file_count} files on {cpu_count} CPU cores",
                'timeout': f"Based on largest file {max_size_mb:.1f}MB"
            }
        }