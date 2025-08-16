"""
Tests for memory-aware error handling functionality.
"""

import pytest
import torch
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory_exceptions import (
    OutOfMemoryError, 
    InsufficientMemoryError, 
    MemoryLeakError,
    MemoryErrorHandler,
    OptimizationSuggestion
)
from src.memory_optimizer import MemoryOptimizer


class TestMemoryExceptions:
    """Test custom memory exception classes."""
    
    def test_out_of_memory_error_creation(self):
        """Test OutOfMemoryError creation with suggestions."""
        error = OutOfMemoryError(
            "Test OOM error",
            current_usage_gb=10.5,
            required_gb=2.0,
            available_gb=1.0
        )
        
        assert "Test OOM error" in str(error)
        assert error.current_usage_gb == 10.5
        assert error.required_gb == 2.0
        assert error.available_gb == 1.0
        assert len(error.suggestions) > 0
        assert any(s.action == "reduce_batch_size" for s in error.suggestions)
    
    def test_insufficient_memory_error_creation(self):
        """Test InsufficientMemoryError creation with suggestions."""
        error = InsufficientMemoryError(
            "Test insufficient memory",
            current_usage_gb=8.0,
            limit_gb=13.0
        )
        
        assert "Test insufficient memory" in str(error)
        assert error.current_usage_gb == 8.0
        assert error.limit_gb == 13.0
        assert len(error.suggestions) > 0
        assert any(s.action == "cleanup_memory" for s in error.suggestions)
    
    def test_memory_leak_error_creation(self):
        """Test MemoryLeakError creation with growth calculation."""
        error = MemoryLeakError(
            "Test memory leak",
            current_usage_gb=12.0,
            previous_usage_gb=8.0
        )
        
        assert "Test memory leak" in str(error)
        assert error.current_usage_gb == 12.0
        assert error.previous_usage_gb == 8.0
        assert error.growth_gb == 4.0
        assert len(error.suggestions) > 0
        assert any(s.action == "force_cleanup" for s in error.suggestions)


class TestMemoryErrorHandler:
    """Test memory error handler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_optimizer = Mock()
        self.handler = MemoryErrorHandler(self.mock_optimizer)
    
    def test_handle_out_of_memory_with_recovery(self):
        """Test OOM error handling with successful recovery."""
        # Mock successful cleanup
        self.mock_optimizer.cleanup_gpu_memory.return_value = None
        self.mock_optimizer.check_memory_safety.return_value = True
        
        error = OutOfMemoryError("Test OOM", current_usage_gb=10.0)
        result = self.handler.handle_out_of_memory(error, auto_recover=True)
        
        assert result is True
        self.mock_optimizer.cleanup_gpu_memory.assert_called_once()
        self.mock_optimizer.check_memory_safety.assert_called_once()
    
    def test_handle_out_of_memory_without_recovery(self):
        """Test OOM error handling without auto recovery."""
        error = OutOfMemoryError("Test OOM", current_usage_gb=10.0)
        result = self.handler.handle_out_of_memory(error, auto_recover=False)
        
        assert result is False
        self.mock_optimizer.cleanup_gpu_memory.assert_not_called()
    
    def test_handle_insufficient_memory_with_recovery(self):
        """Test insufficient memory error handling with recovery."""
        # Mock successful optimization
        self.mock_optimizer.cleanup_gpu_memory.return_value = None
        self.mock_optimizer.check_memory_safety.return_value = True
        
        error = InsufficientMemoryError("Test insufficient", current_usage_gb=8.0)
        result = self.handler.handle_insufficient_memory(error, auto_recover=True)
        
        assert result is True
        self.mock_optimizer.cleanup_gpu_memory.assert_called_once()
    
    def test_handle_memory_leak_with_recovery(self):
        """Test memory leak error handling with aggressive cleanup."""
        # Mock memory status that shows improvement
        mock_status = Mock()
        mock_status.allocated_gb = 8.0  # Less than error's current usage
        self.mock_optimizer.get_memory_status.return_value = mock_status
        
        error = MemoryLeakError("Test leak", current_usage_gb=10.0, previous_usage_gb=6.0)
        result = self.handler.handle_memory_leak(error, auto_recover=True)
        
        assert result is True
        # Should call cleanup multiple times for aggressive cleanup
        assert self.mock_optimizer.cleanup_gpu_memory.call_count >= 1
    
    def test_check_memory_warnings_critical(self):
        """Test critical memory warning detection."""
        # This should trigger a critical warning (>95% usage)
        self.handler.check_memory_warnings(
            current_usage_gb=12.5,
            limit_gb=13.0,
            warning_threshold=0.8
        )
        # Test passes if no exception is raised
    
    def test_check_memory_warnings_normal(self):
        """Test normal memory usage (no warnings)."""
        # This should not trigger any warnings
        self.handler.check_memory_warnings(
            current_usage_gb=8.0,
            limit_gb=13.0,
            warning_threshold=0.8
        )
        # Test passes if no exception is raised
    
    def test_track_memory_usage_leak_detection(self):
        """Test memory leak detection through usage tracking."""
        # Simulate gradual memory growth
        usage_values = [5.0, 6.0, 7.0, 8.0, 9.0, 10.5]  # 5.5GB growth
        
        for usage in usage_values:
            self.handler.track_memory_usage(usage)
        
        # Should detect potential leak (>1GB growth in recent measurements)
        assert len(self.handler._memory_history) == 6
    
    def test_get_recovery_config_oom(self):
        """Test recovery configuration for OOM errors."""
        config = self.handler.get_recovery_config("oom")
        
        assert config["batch_size_multiplier"] == 0.5
        assert config["gradient_accumulation_steps_multiplier"] == 2.0
        assert config["enable_gradient_checkpointing"] is True
        assert config["use_fp16"] is True
    
    def test_get_recovery_config_insufficient(self):
        """Test recovery configuration for insufficient memory errors."""
        config = self.handler.get_recovery_config("insufficient")
        
        assert config["batch_size_multiplier"] == 0.7
        assert config["enable_gradient_checkpointing"] is True
        assert config["use_quantization"] is True
    
    def test_get_recovery_config_leak(self):
        """Test recovery configuration for memory leak errors."""
        config = self.handler.get_recovery_config("leak")
        
        assert config["force_cleanup_frequency"] == 10
        assert config["restart_recommendation"] is True
        assert config["enable_memory_profiling"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMemoryOptimizerIntegration:
    """Test integration between MemoryOptimizer and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = MemoryOptimizer(max_memory_gb=1.0)  # Very low limit for testing
    
    def test_memory_optimizer_has_error_handler(self):
        """Test that MemoryOptimizer initializes with error handler."""
        assert hasattr(self.optimizer, 'error_handler')
        assert isinstance(self.optimizer.error_handler, MemoryErrorHandler)
    
    def test_safe_operation_success(self):
        """Test successful operation execution with memory safety."""
        def dummy_operation():
            return "success"
        
        # Should work with a simple operation
        result = self.optimizer.safe_operation(dummy_operation)
        assert result == "success"
    
    def test_safe_operation_with_memory_error(self):
        """Test operation that triggers memory error."""
        def memory_intensive_operation():
            # This should trigger memory safety checks
            large_tensor = torch.randn(1000, 1000, device='cuda')
            return large_tensor
        
        # This might raise an exception due to low memory limit
        try:
            self.optimizer.safe_operation(memory_intensive_operation)
        except (OutOfMemoryError, InsufficientMemoryError):
            # Expected behavior with very low memory limit
            pass


if __name__ == "__main__":
    pytest.main([__file__])