"""
Memory-aware error handling for GPU memory management.

This module provides custom exception classes and error recovery functions
for handling memory-related errors during model training and inference.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class OptimizationSuggestion:
    """Data class for memory optimization suggestions."""
    action: str
    description: str
    priority: int  # 1=high, 2=medium, 3=low


class MemoryError(Exception):
    """Base class for memory-related errors."""
    
    def __init__(self, message: str, current_usage_gb: float = 0.0, 
                 suggestions: Optional[List[OptimizationSuggestion]] = None):
        super().__init__(message)
        self.current_usage_gb = current_usage_gb
        self.suggestions = suggestions or []


class OutOfMemoryError(MemoryError):
    """Raised when GPU runs out of memory during operations."""
    
    def __init__(self, message: str, current_usage_gb: float = 0.0, 
                 required_gb: float = 0.0, available_gb: float = 0.0):
        suggestions = [
            OptimizationSuggestion(
                action="reduce_batch_size",
                description="Reduce batch size to decrease memory usage",
                priority=1
            ),
            OptimizationSuggestion(
                action="enable_gradient_checkpointing",
                description="Enable gradient checkpointing to trade compute for memory",
                priority=1
            ),
            OptimizationSuggestion(
                action="reduce_sequence_length",
                description="Reduce maximum sequence length",
                priority=2
            ),
            OptimizationSuggestion(
                action="use_gradient_accumulation",
                description="Use gradient accumulation with smaller batch sizes",
                priority=2
            )
        ]
        
        detailed_message = (f"{message} "
                          f"Required: {required_gb:.2f}GB, "
                          f"Available: {available_gb:.2f}GB, "
                          f"Current usage: {current_usage_gb:.2f}GB")
        
        super().__init__(detailed_message, current_usage_gb, suggestions)
        self.required_gb = required_gb
        self.available_gb = available_gb


class InsufficientMemoryError(MemoryError):
    """Raised when available memory is insufficient for planned operations."""
    
    def __init__(self, message: str, current_usage_gb: float = 0.0, 
                 limit_gb: float = 0.0):
        suggestions = [
            OptimizationSuggestion(
                action="cleanup_memory",
                description="Run memory cleanup to free cached memory",
                priority=1
            ),
            OptimizationSuggestion(
                action="reduce_model_precision",
                description="Use lower precision (fp16/bf16) or quantization",
                priority=1
            ),
            OptimizationSuggestion(
                action="optimize_model_loading",
                description="Use model sharding or offloading techniques",
                priority=2
            )
        ]
        
        detailed_message = (f"{message} "
                          f"Current usage: {current_usage_gb:.2f}GB, "
                          f"Memory limit: {limit_gb:.2f}GB")
        
        super().__init__(detailed_message, current_usage_gb, suggestions)
        self.limit_gb = limit_gb


class MemoryLeakError(MemoryError):
    """Raised when memory usage continues to grow unexpectedly."""
    
    def __init__(self, message: str, current_usage_gb: float = 0.0, 
                 previous_usage_gb: float = 0.0):
        suggestions = [
            OptimizationSuggestion(
                action="force_cleanup",
                description="Force garbage collection and cache clearing",
                priority=1
            ),
            OptimizationSuggestion(
                action="check_references",
                description="Check for circular references or unclosed resources",
                priority=1
            ),
            OptimizationSuggestion(
                action="restart_process",
                description="Restart the training process to clear memory leaks",
                priority=3
            )
        ]
        
        growth = current_usage_gb - previous_usage_gb
        detailed_message = (f"{message} "
                          f"Memory grew by {growth:.2f}GB "
                          f"(from {previous_usage_gb:.2f}GB to {current_usage_gb:.2f}GB)")
        
        super().__init__(detailed_message, current_usage_gb, suggestions)
        self.previous_usage_gb = previous_usage_gb
        self.growth_gb = growth


class MemoryErrorHandler:
    """
    Handler for memory-related errors with automatic recovery suggestions.
    
    This class provides methods to handle different types of memory errors
    and suggest appropriate recovery actions.
    """
    
    def __init__(self, memory_optimizer=None):
        """
        Initialize the error handler.
        
        Args:
            memory_optimizer: Optional MemoryOptimizer instance for cleanup operations
        """
        self.memory_optimizer = memory_optimizer
        self.logger = logging.getLogger(__name__)
        self._memory_history = []  # Track memory usage over time
    
    def handle_out_of_memory(self, error: OutOfMemoryError, 
                           auto_recover: bool = True) -> bool:
        """
        Handle OutOfMemoryError with automatic recovery attempts.
        
        Args:
            error: The OutOfMemoryError to handle
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.logger.error(f"Out of memory error: {error}")
        
        if not auto_recover:
            self._log_suggestions(error.suggestions)
            return False
        
        # Attempt automatic recovery
        recovery_success = False
        
        try:
            # Step 1: Immediate cleanup
            if self.memory_optimizer:
                self.logger.info("Attempting memory cleanup...")
                self.memory_optimizer.cleanup_gpu_memory()
                
                # Check if cleanup was sufficient
                if self.memory_optimizer.check_memory_safety():
                    self.logger.info("Memory cleanup successful")
                    recovery_success = True
                else:
                    self.logger.warning("Memory cleanup insufficient")
            
            # Step 2: Log suggestions for manual intervention
            if not recovery_success:
                self._log_suggestions(error.suggestions)
                
        except Exception as cleanup_error:
            self.logger.error(f"Error during automatic recovery: {cleanup_error}")
        
        return recovery_success
    
    def handle_insufficient_memory(self, error: InsufficientMemoryError,
                                 auto_recover: bool = True) -> bool:
        """
        Handle InsufficientMemoryError with recovery suggestions.
        
        Args:
            error: The InsufficientMemoryError to handle
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.logger.error(f"Insufficient memory error: {error}")
        
        if not auto_recover:
            self._log_suggestions(error.suggestions)
            return False
        
        # Attempt automatic recovery
        recovery_success = False
        
        try:
            if self.memory_optimizer:
                self.logger.info("Attempting memory optimization...")
                self.memory_optimizer.cleanup_gpu_memory()
                
                # Check if optimization was sufficient
                if self.memory_optimizer.check_memory_safety():
                    self.logger.info("Memory optimization successful")
                    recovery_success = True
                else:
                    self.logger.warning("Memory optimization insufficient")
            
            if not recovery_success:
                self._log_suggestions(error.suggestions)
                
        except Exception as cleanup_error:
            self.logger.error(f"Error during memory optimization: {cleanup_error}")
        
        return recovery_success
    
    def handle_memory_leak(self, error: MemoryLeakError,
                          auto_recover: bool = True) -> bool:
        """
        Handle MemoryLeakError with aggressive cleanup.
        
        Args:
            error: The MemoryLeakError to handle
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.logger.error(f"Memory leak detected: {error}")
        
        if not auto_recover:
            self._log_suggestions(error.suggestions)
            return False
        
        # Attempt aggressive cleanup
        recovery_success = False
        
        try:
            if self.memory_optimizer:
                self.logger.info("Attempting aggressive memory cleanup...")
                
                # Multiple cleanup attempts
                for i in range(3):
                    self.memory_optimizer.cleanup_gpu_memory()
                    
                    # Check if leak was stopped
                    current_status = self.memory_optimizer.get_memory_status()
                    if current_status.allocated_gb < error.current_usage_gb:
                        self.logger.info(f"Memory leak cleanup successful after {i+1} attempts")
                        recovery_success = True
                        break
                
                if not recovery_success:
                    self.logger.warning("Aggressive cleanup failed to resolve memory leak")
            
            if not recovery_success:
                self._log_suggestions(error.suggestions)
                
        except Exception as cleanup_error:
            self.logger.error(f"Error during memory leak cleanup: {cleanup_error}")
        
        return recovery_success
    
    def check_memory_warnings(self, current_usage_gb: float, 
                            limit_gb: float, warning_threshold: float = 0.8) -> None:
        """
        Check for proactive memory warnings when usage approaches limits.
        
        Args:
            current_usage_gb: Current memory usage in GB
            limit_gb: Memory limit in GB
            warning_threshold: Threshold (0-1) for warnings
        """
        usage_ratio = current_usage_gb / limit_gb
        
        if usage_ratio >= warning_threshold:
            warning_level = "CRITICAL" if usage_ratio >= 0.95 else "WARNING"
            
            self.logger.warning(f"{warning_level}: High memory usage detected")
            self.logger.warning(f"Current usage: {current_usage_gb:.2f}GB "
                              f"({usage_ratio*100:.1f}% of {limit_gb:.2f}GB limit)")
            
            # Suggest proactive measures
            suggestions = [
                OptimizationSuggestion(
                    action="proactive_cleanup",
                    description="Run memory cleanup before continuing",
                    priority=1
                ),
                OptimizationSuggestion(
                    action="reduce_batch_size",
                    description="Consider reducing batch size for next operations",
                    priority=2
                )
            ]
            
            self._log_suggestions(suggestions)
    
    def track_memory_usage(self, current_usage_gb: float) -> None:
        """
        Track memory usage over time to detect potential leaks.
        
        Args:
            current_usage_gb: Current memory usage in GB
        """
        self._memory_history.append(current_usage_gb)
        
        # Keep only recent history (last 10 measurements)
        if len(self._memory_history) > 10:
            self._memory_history.pop(0)
        
        # Check for consistent growth (potential leak)
        if len(self._memory_history) >= 5:
            recent_growth = self._memory_history[-1] - self._memory_history[-5]
            if recent_growth > 1.0:  # More than 1GB growth in recent measurements
                self.logger.warning(f"Potential memory leak detected: "
                                  f"{recent_growth:.2f}GB growth in recent operations")
    
    def _log_suggestions(self, suggestions: List[OptimizationSuggestion]) -> None:
        """
        Log optimization suggestions in order of priority.
        
        Args:
            suggestions: List of optimization suggestions
        """
        if not suggestions:
            return
        
        # Sort by priority (1=high, 2=medium, 3=low)
        sorted_suggestions = sorted(suggestions, key=lambda x: x.priority)
        
        self.logger.info("Memory optimization suggestions:")
        for i, suggestion in enumerate(sorted_suggestions, 1):
            priority_str = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}[suggestion.priority]
            self.logger.info(f"  {i}. [{priority_str}] {suggestion.action}: {suggestion.description}")
    
    def get_recovery_config(self, error_type: str) -> Dict[str, Any]:
        """
        Get recommended configuration changes for recovery.
        
        Args:
            error_type: Type of error ("oom", "insufficient", "leak")
            
        Returns:
            Dictionary of recommended configuration changes
        """
        configs = {
            "oom": {
                "batch_size_multiplier": 0.5,
                "gradient_accumulation_steps_multiplier": 2.0,
                "max_sequence_length_multiplier": 0.8,
                "enable_gradient_checkpointing": True,
                "use_fp16": True
            },
            "insufficient": {
                "batch_size_multiplier": 0.7,
                "enable_gradient_checkpointing": True,
                "use_quantization": True,
                "optimize_model_loading": True
            },
            "leak": {
                "force_cleanup_frequency": 10,  # Every 10 steps
                "restart_recommendation": True,
                "enable_memory_profiling": True
            }
        }
        
        return configs.get(error_type, {})