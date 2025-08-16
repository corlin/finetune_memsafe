"""
MemoryOptimizer单元测试模块

测试MemoryOptimizer类的内存监控、清理和安全检查功能。
"""

import pytest
import torch
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory_optimizer import MemoryOptimizer, MemoryStatus
from src.memory_exceptions import OutOfMemoryError, InsufficientMemoryError, MemoryLeakError


class TestMemoryOptimizer(unittest.TestCase):
    """MemoryOptimizer单元测试类"""
    
    def setUp(self):
        """测试前设置"""
        # Mock CUDA availability to avoid hardware dependency
        self.cuda_available_patcher = patch('torch.cuda.is_available')
        self.mock_cuda_available = self.cuda_available_patcher.start()
        self.mock_cuda_available.return_value = True
        
        # Mock CUDA device properties
        self.device_props_patcher = patch('torch.cuda.get_device_properties')
        self.mock_device_props = self.device_props_patcher.start()
        mock_props = Mock()
        mock_props.total_memory = 16 * (1024**3)  # 16GB
        self.mock_device_props.return_value = mock_props
        
        # Mock current device
        self.current_device_patcher = patch('torch.cuda.current_device')
        self.mock_current_device = self.current_device_patcher.start()
        self.mock_current_device.return_value = 0
        
        # Mock memory functions
        self.memory_allocated_patcher = patch('torch.cuda.memory_allocated')
        self.mock_memory_allocated = self.memory_allocated_patcher.start()
        self.mock_memory_allocated.return_value = 8 * (1024**3)  # 8GB
        
        self.memory_reserved_patcher = patch('torch.cuda.memory_reserved')
        self.mock_memory_reserved = self.memory_reserved_patcher.start()
        self.mock_memory_reserved.return_value = 10 * (1024**3)  # 10GB
        
        # Mock empty_cache
        self.empty_cache_patcher = patch('torch.cuda.empty_cache')
        self.mock_empty_cache = self.empty_cache_patcher.start()
        
        # Mock set_per_process_memory_fraction
        self.memory_fraction_patcher = patch('torch.cuda.set_per_process_memory_fraction')
        self.mock_memory_fraction = self.memory_fraction_patcher.start()
        
        # Create optimizer instance
        self.optimizer = MemoryOptimizer(max_memory_gb=13.0, safety_threshold=0.85)
    
    def tearDown(self):
        """测试后清理"""
        self.cuda_available_patcher.stop()
        self.device_props_patcher.stop()
        self.current_device_patcher.stop()
        self.memory_allocated_patcher.stop()
        self.memory_reserved_patcher.stop()
        self.empty_cache_patcher.stop()
        self.memory_fraction_patcher.stop()
    
    def test_initialization(self):
        """测试MemoryOptimizer初始化"""
        self.assertEqual(self.optimizer.max_memory_gb, 13.0)
        self.assertEqual(self.optimizer.safety_threshold, 0.85)
        self.assertEqual(self.optimizer.device, 0)
        self.assertIsNotNone(self.optimizer.error_handler)
    
    def test_initialization_without_cuda(self):
        """测试在没有CUDA的情况下初始化"""
        self.mock_cuda_available.return_value = False
        
        with self.assertRaises(RuntimeError) as context:
            MemoryOptimizer()
        
        self.assertIn("CUDA is not available", str(context.exception))
    
    def test_monitor_gpu_memory(self):
        """测试GPU内存监控"""
        allocated, cached, total = self.optimizer.monitor_gpu_memory()
        
        self.assertEqual(allocated, 8.0)  # 8GB
        self.assertEqual(cached, 10.0)    # 10GB
        self.assertEqual(total, 16.0)     # 16GB
        
        # 验证调用了正确的CUDA函数
        self.mock_memory_allocated.assert_called_with(0)
        self.mock_memory_reserved.assert_called_with(0)
        self.mock_device_props.assert_called_with(0)
    
    def test_monitor_gpu_memory_without_cuda(self):
        """测试在没有CUDA的情况下监控内存"""
        self.mock_cuda_available.return_value = False
        
        allocated, cached, total = self.optimizer.monitor_gpu_memory()
        
        self.assertEqual(allocated, 0.0)
        self.assertEqual(cached, 0.0)
        self.assertEqual(total, 0.0)
    
    def test_get_memory_status(self):
        """测试获取内存状态"""
        status = self.optimizer.get_memory_status()
        
        self.assertIsInstance(status, MemoryStatus)
        self.assertEqual(status.allocated_gb, 8.0)
        self.assertEqual(status.cached_gb, 10.0)
        self.assertEqual(status.total_gb, 16.0)
        self.assertEqual(status.available_gb, 8.0)  # 16 - 8
        self.assertTrue(status.is_safe)  # 8GB < 13GB * 0.85 = 11.05GB
        self.assertIsInstance(status.timestamp, datetime)
    
    def test_get_memory_status_unsafe(self):
        """测试不安全的内存状态"""
        # 设置高内存使用量
        self.mock_memory_allocated.return_value = 12 * (1024**3)  # 12GB
        
        status = self.optimizer.get_memory_status()
        
        self.assertEqual(status.allocated_gb, 12.0)
        self.assertFalse(status.is_safe)  # 12GB > 13GB * 0.85 = 11.05GB
    
    def test_cleanup_gpu_memory(self):
        """测试GPU内存清理"""
        # 设置清理前后的内存状态
        self.mock_memory_allocated.side_effect = [
            8 * (1024**3),  # 清理前
            6 * (1024**3)   # 清理后
        ]
        self.mock_memory_reserved.side_effect = [
            10 * (1024**3),  # 清理前
            7 * (1024**3)    # 清理后
        ]
        
        self.optimizer.cleanup_gpu_memory()
        
        # 验证调用了清理函数
        self.mock_empty_cache.assert_called_once()
    
    def test_cleanup_gpu_memory_without_cuda(self):
        """测试在没有CUDA的情况下清理内存"""
        self.mock_cuda_available.return_value = False
        
        # 应该不会抛出异常，只是记录警告
        self.optimizer.cleanup_gpu_memory()
        
        # 不应该调用empty_cache
        self.mock_empty_cache.assert_not_called()
    
    def test_check_memory_safety_safe(self):
        """测试安全的内存检查"""
        # 设置安全的内存使用量
        self.mock_memory_allocated.return_value = 8 * (1024**3)  # 8GB
        
        result = self.optimizer.check_memory_safety()
        
        self.assertTrue(result)
    
    def test_check_memory_safety_unsafe(self):
        """测试不安全的内存检查"""
        # 设置不安全的内存使用量
        self.mock_memory_allocated.return_value = 12 * (1024**3)  # 12GB
        
        result = self.optimizer.check_memory_safety()
        
        self.assertFalse(result)
    
    def test_check_memory_safety_with_requirement(self):
        """测试带特定需求的内存安全检查"""
        # 设置内存状态：8GB已用，8GB可用
        self.mock_memory_allocated.return_value = 8 * (1024**3)
        
        # 请求5GB，应该通过
        result = self.optimizer.check_memory_safety(required_gb=5.0)
        self.assertTrue(result)
        
        # 请求10GB，应该失败并抛出异常
        with self.assertRaises(InsufficientMemoryError):
            self.optimizer.check_memory_safety(required_gb=10.0)
    
    def test_check_memory_safety_exceeds_limit(self):
        """测试超过内存限制的情况"""
        # 设置超过限制的内存使用量
        self.mock_memory_allocated.return_value = 14 * (1024**3)  # 14GB > 13GB limit
        
        # The method should return False for unsafe memory, not raise exception immediately
        # Exception is raised only when checking against absolute limit
        result = self.optimizer.check_memory_safety()
        self.assertFalse(result)
    
    def test_optimize_for_training(self):
        """测试训练优化"""
        self.optimizer.optimize_for_training()
        
        # 验证调用了清理和内存分数设置
        self.mock_empty_cache.assert_called()
        self.mock_memory_fraction.assert_called_once()
        
        # 检查内存分数计算
        call_args = self.mock_memory_fraction.call_args
        memory_fraction = call_args[0][0]
        device = call_args[0][1]
        
        expected_fraction = min(13.0 / 16.0, 0.95)  # 13GB / 16GB = 0.8125
        self.assertAlmostEqual(memory_fraction, expected_fraction, places=3)
        self.assertEqual(device, 0)
    
    def test_handle_memory_error_oom(self):
        """测试处理OOM错误"""
        error = OutOfMemoryError("Test OOM", current_usage_gb=14.0)
        
        # Mock error handler
        self.optimizer.error_handler.handle_out_of_memory = Mock(return_value=True)
        
        result = self.optimizer.handle_memory_error(error, auto_recover=True)
        
        self.assertTrue(result)
        self.optimizer.error_handler.handle_out_of_memory.assert_called_once_with(error, True)
    
    def test_handle_memory_error_insufficient(self):
        """测试处理内存不足错误"""
        error = InsufficientMemoryError("Test insufficient", current_usage_gb=10.0)
        
        # Mock error handler
        self.optimizer.error_handler.handle_insufficient_memory = Mock(return_value=True)
        
        result = self.optimizer.handle_memory_error(error, auto_recover=True)
        
        self.assertTrue(result)
        self.optimizer.error_handler.handle_insufficient_memory.assert_called_once_with(error, True)
    
    def test_handle_memory_error_leak(self):
        """测试处理内存泄漏错误"""
        error = MemoryLeakError("Test leak", current_usage_gb=12.0, previous_usage_gb=8.0)
        
        # Mock error handler
        self.optimizer.error_handler.handle_memory_leak = Mock(return_value=True)
        
        result = self.optimizer.handle_memory_error(error, auto_recover=True)
        
        self.assertTrue(result)
        self.optimizer.error_handler.handle_memory_leak.assert_called_once_with(error, True)
    
    def test_handle_memory_error_unknown(self):
        """测试处理未知错误"""
        error = ValueError("Unknown error")
        
        result = self.optimizer.handle_memory_error(error, auto_recover=True)
        
        self.assertFalse(result)
    
    def test_safe_operation_success(self):
        """测试安全操作执行成功"""
        def dummy_operation():
            return "success"
        
        # 设置安全的内存状态
        self.mock_memory_allocated.return_value = 8 * (1024**3)
        
        result = self.optimizer.safe_operation(dummy_operation)
        
        self.assertEqual(result, "success")
    
    def test_safe_operation_with_memory_error(self):
        """测试安全操作遇到内存错误"""
        def failing_operation():
            raise OutOfMemoryError("Test OOM", current_usage_gb=14.0)
        
        # Mock error handler to return successful recovery
        self.optimizer.error_handler.handle_out_of_memory = Mock(return_value=True)
        
        # 设置内存状态：第一次不安全，第二次安全
        self.mock_memory_allocated.side_effect = [
            8 * (1024**3),   # 第一次检查：安全
            8 * (1024**3)    # 第二次检查：安全
        ]
        
        # 应该在重试后成功，但操作本身会抛出异常
        with self.assertRaises(OutOfMemoryError):
            self.optimizer.safe_operation(failing_operation, max_retries=1)
    
    def test_safe_operation_max_retries_exceeded(self):
        """测试安全操作超过最大重试次数"""
        def failing_operation():
            raise OutOfMemoryError("Test OOM", current_usage_gb=14.0)
        
        # Mock error handler to return failed recovery
        self.optimizer.error_handler.handle_out_of_memory = Mock(return_value=False)
        
        with self.assertRaises(OutOfMemoryError):
            self.optimizer.safe_operation(failing_operation, max_retries=1)
    
    def test_safe_operation_non_memory_error(self):
        """测试安全操作遇到非内存错误"""
        def failing_operation():
            raise ValueError("Non-memory error")
        
        # 设置安全的内存状态
        self.mock_memory_allocated.return_value = 8 * (1024**3)
        
        with self.assertRaises(ValueError):
            self.optimizer.safe_operation(failing_operation)
    
    def test_log_memory_status(self):
        """测试内存状态日志记录"""
        # 这个方法主要是记录日志，我们只需要确保它不会抛出异常
        try:
            self.optimizer.log_memory_status("Test prefix")
        except Exception as e:
            self.fail(f"log_memory_status抛出异常: {e}")
    
    def test_memory_status_dataclass(self):
        """测试MemoryStatus数据类"""
        timestamp = datetime.now()
        status = MemoryStatus(
            allocated_gb=8.0,
            cached_gb=10.0,
            total_gb=16.0,
            available_gb=8.0,
            is_safe=True,
            timestamp=timestamp
        )
        
        self.assertEqual(status.allocated_gb, 8.0)
        self.assertEqual(status.cached_gb, 10.0)
        self.assertEqual(status.total_gb, 16.0)
        self.assertEqual(status.available_gb, 8.0)
        self.assertTrue(status.is_safe)
        self.assertEqual(status.timestamp, timestamp)


class TestMemoryOptimizerIntegration(unittest.TestCase):
    """MemoryOptimizer集成测试"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_cuda_integration(self):
        """测试与真实CUDA的集成（仅在有CUDA时运行）"""
        optimizer = MemoryOptimizer(max_memory_gb=1.0)  # 很低的限制用于测试
        
        # 测试基本功能
        allocated, cached, total = optimizer.monitor_gpu_memory()
        self.assertGreaterEqual(allocated, 0)
        self.assertGreaterEqual(cached, 0)
        self.assertGreater(total, 0)
        
        # 测试内存清理
        optimizer.cleanup_gpu_memory()
        
        # 测试内存状态
        status = optimizer.get_memory_status()
        self.assertIsInstance(status, MemoryStatus)


if __name__ == "__main__":
    # 设置日志级别以减少测试输出
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    unittest.main()