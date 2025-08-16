"""
Memory optimization utilities for GPU memory management during model training.

This module provides tools for monitoring GPU memory usage, automatic cleanup,
and memory safety validation to ensure training stays within memory limits.
"""

import torch
import gc
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from .memory_exceptions import (
    OutOfMemoryError, 
    InsufficientMemoryError, 
    MemoryLeakError,
    MemoryErrorHandler
)


@dataclass
class MemoryStatus:
    """Data class to hold GPU memory status information."""
    allocated_gb: float
    cached_gb: float
    total_gb: float
    available_gb: float
    is_safe: bool
    timestamp: datetime


class MemoryOptimizer:
    """
    GPU memory optimizer for managing memory usage during model training.
    
    This class provides functionality to monitor GPU memory allocation,
    perform automatic cleanup, and validate memory safety before operations.
    """
    
    def __init__(self, max_memory_gb: float = 13.0, safety_threshold: float = 0.85):
        """
        Initialize the MemoryOptimizer.
        
        Args:
            max_memory_gb: Maximum allowed GPU memory usage in GB
            safety_threshold: Threshold (0-1) for memory safety warnings
        """
        self.max_memory_gb = max_memory_gb
        self.safety_threshold = safety_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize error handler
        self.error_handler = MemoryErrorHandler(memory_optimizer=self)
        
        # Verify CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU memory optimization requires CUDA.")
        
        self.device = torch.cuda.current_device()
        self.logger.info(f"MemoryOptimizer initialized for device {self.device}")
        self.logger.info(f"Max memory limit: {max_memory_gb}GB, Safety threshold: {safety_threshold}")
    
    def monitor_gpu_memory(self) -> Tuple[float, float, float]:
        """
        Monitor current GPU memory allocation, cache, and total memory.
        
        Returns:
            Tuple of (allocated_gb, cached_gb, total_gb)
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        
        # Get memory statistics in bytes and convert to GB
        allocated_bytes = torch.cuda.memory_allocated(self.device)
        cached_bytes = torch.cuda.memory_reserved(self.device)
        total_bytes = torch.cuda.get_device_properties(self.device).total_memory
        
        allocated_gb = allocated_bytes / (1024**3)
        cached_gb = cached_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        
        return allocated_gb, cached_gb, total_gb
    
    def get_memory_status(self) -> MemoryStatus:
        """
        Get comprehensive memory status information.
        
        Returns:
            MemoryStatus object with current memory information
        """
        allocated_gb, cached_gb, total_gb = self.monitor_gpu_memory()
        available_gb = total_gb - allocated_gb
        is_safe = allocated_gb < (self.max_memory_gb * self.safety_threshold)
        
        return MemoryStatus(
            allocated_gb=allocated_gb,
            cached_gb=cached_gb,
            total_gb=total_gb,
            available_gb=available_gb,
            is_safe=is_safe,
            timestamp=datetime.now()
        )
    
    def cleanup_gpu_memory(self) -> None:
        """
        Perform automatic GPU memory cleanup using torch.cuda.empty_cache().
        
        This function clears the GPU memory cache and runs garbage collection
        to free up unused memory.
        """
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, skipping GPU memory cleanup")
            return
        
        # Get memory before cleanup
        before_allocated, before_cached, _ = self.monitor_gpu_memory()
        
        # Perform cleanup
        gc.collect()  # Python garbage collection
        torch.cuda.empty_cache()  # Clear GPU cache
        
        # Get memory after cleanup
        after_allocated, after_cached, _ = self.monitor_gpu_memory()
        
        # Log cleanup results
        freed_allocated = before_allocated - after_allocated
        freed_cached = before_cached - after_cached
        
        self.logger.info(f"GPU memory cleanup completed:")
        self.logger.info(f"  Freed allocated memory: {freed_allocated:.2f}GB")
        self.logger.info(f"  Freed cached memory: {freed_cached:.2f}GB")
        self.logger.info(f"  Current allocated: {after_allocated:.2f}GB")
        self.logger.info(f"  Current cached: {after_cached:.2f}GB")
    
    def check_memory_safety(self, required_gb: Optional[float] = None) -> bool:
        """
        Validate available memory before operations to ensure memory safety.
        
        Args:
            required_gb: Optional specific memory requirement to check
            
        Returns:
            True if memory usage is safe, False otherwise
        """
        status = self.get_memory_status()
        
        # Track memory usage for leak detection
        self.error_handler.track_memory_usage(status.allocated_gb)
        
        # Check for proactive warnings
        self.error_handler.check_memory_warnings(
            status.allocated_gb, 
            self.max_memory_gb, 
            self.safety_threshold
        )
        
        # Check against general safety threshold
        if not status.is_safe:
            self.logger.warning(f"Memory usage unsafe: {status.allocated_gb:.2f}GB allocated "
                              f"(threshold: {self.max_memory_gb * self.safety_threshold:.2f}GB)")
            return False
        
        # Check against specific requirement if provided
        if required_gb is not None:
            if status.available_gb < required_gb:
                self.logger.warning(f"Insufficient memory: {status.available_gb:.2f}GB available, "
                                  f"{required_gb:.2f}GB required")
                
                # Raise InsufficientMemoryError
                error = InsufficientMemoryError(
                    "Insufficient GPU memory for operation",
                    current_usage_gb=status.allocated_gb,
                    limit_gb=self.max_memory_gb
                )
                raise error
        
        # Check against absolute limit
        if status.allocated_gb > self.max_memory_gb:
            self.logger.error(f"Memory limit exceeded: {status.allocated_gb:.2f}GB > {self.max_memory_gb}GB")
            
            # Raise OutOfMemoryError
            error = OutOfMemoryError(
                "GPU memory limit exceeded",
                current_usage_gb=status.allocated_gb,
                required_gb=required_gb or 0.0,
                available_gb=status.available_gb
            )
            raise error
        
        self.logger.debug(f"Memory safety check passed: {status.allocated_gb:.2f}GB allocated, "
                         f"{status.available_gb:.2f}GB available")
        return True
    
    def optimize_for_training(self) -> None:
        """
        Optimize GPU memory settings for training operations.
        
        This function performs comprehensive memory optimization including
        cleanup and configuration adjustments for training.
        """
        self.logger.info("Optimizing GPU memory for training...")
        
        # Perform initial cleanup
        self.cleanup_gpu_memory()
        
        # Set memory fraction if needed
        if torch.cuda.is_available():
            # Calculate memory fraction based on max_memory_gb
            total_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            memory_fraction = min(self.max_memory_gb / total_memory_gb, 0.95)
            
            # Set memory fraction (this affects future allocations)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)
            
            self.logger.info(f"Set memory fraction to {memory_fraction:.2f} "
                           f"({self.max_memory_gb:.1f}GB of {total_memory_gb:.1f}GB)")
        
        # Final memory status
        status = self.get_memory_status()
        self.logger.info(f"Memory optimization complete. Current usage: {status.allocated_gb:.2f}GB")
    
    def handle_memory_error(self, error: Exception, auto_recover: bool = True) -> bool:
        """
        Handle memory-related errors with automatic recovery attempts.
        
        Args:
            error: The memory error to handle
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if isinstance(error, OutOfMemoryError):
            return self.error_handler.handle_out_of_memory(error, auto_recover)
        elif isinstance(error, InsufficientMemoryError):
            return self.error_handler.handle_insufficient_memory(error, auto_recover)
        elif isinstance(error, MemoryLeakError):
            return self.error_handler.handle_memory_leak(error, auto_recover)
        else:
            self.logger.error(f"Unhandled memory error: {error}")
            return False
    
    def safe_operation(self, operation_func, required_gb: Optional[float] = None, 
                      max_retries: int = 3):
        """
        Execute an operation with memory safety checks and error recovery.
        
        Args:
            operation_func: Function to execute safely
            required_gb: Optional memory requirement for the operation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Result of the operation function
            
        Raises:
            Exception: If operation fails after all retry attempts
        """
        for attempt in range(max_retries + 1):
            try:
                # Check memory safety before operation
                if not self.check_memory_safety(required_gb):
                    if attempt == max_retries:
                        raise InsufficientMemoryError(
                            f"Memory safety check failed after {max_retries} attempts",
                            current_usage_gb=self.get_memory_status().allocated_gb,
                            limit_gb=self.max_memory_gb
                        )
                    continue
                
                # Execute the operation
                result = operation_func()
                return result
                
            except (OutOfMemoryError, InsufficientMemoryError, MemoryLeakError) as e:
                self.logger.warning(f"Memory error on attempt {attempt + 1}: {e}")
                
                if attempt == max_retries:
                    self.logger.error(f"Operation failed after {max_retries} attempts")
                    raise
                
                # Attempt recovery
                recovery_success = self.handle_memory_error(e, auto_recover=True)
                if not recovery_success and attempt == max_retries:
                    raise
                
                self.logger.info(f"Retrying operation (attempt {attempt + 2}/{max_retries + 1})")
            
            except Exception as e:
                self.logger.error(f"Non-memory error during operation: {e}")
                raise
    
    def log_memory_status(self, prefix: str = "") -> None:
        """
        Log current memory status with optional prefix.
        
        Args:
            prefix: Optional prefix for log messages
        """
        status = self.get_memory_status()
        prefix_str = f"{prefix}: " if prefix else ""
        
        self.logger.info(f"{prefix_str}GPU Memory Status:")
        self.logger.info(f"  Allocated: {status.allocated_gb:.2f}GB")
        self.logger.info(f"  Cached: {status.cached_gb:.2f}GB") 
        self.logger.info(f"  Available: {status.available_gb:.2f}GB")
        self.logger.info(f"  Total: {status.total_gb:.2f}GB")
        self.logger.info(f"  Safe: {status.is_safe}")