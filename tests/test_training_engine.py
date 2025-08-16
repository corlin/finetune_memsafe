"""
训练引擎测试模块

测试TrainingEngine类的各种功能，包括训练参数创建、训练器配置、
内存监控和错误处理等。
"""

import unittest
import tempfile
import shutil
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.training_engine import (
    TrainingEngine, 
    TrainingConfig, 
    MemoryMonitoringCallback,
    create_training_config
)
from src.memory_optimizer import MemoryOptimizer


class TestTrainingConfig(unittest.TestCase):
    """测试TrainingConfig数据类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        self.assertEqual(config.output_dir, "./qwen3-finetuned")
        self.assertEqual(config.max_memory_gb, 13.0)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.gradient_accumulation_steps, 16)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.num_epochs, 10)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            output_dir="./custom_output",
            batch_size=8,
            learning_rate=1e-4
        )
        
        self.assertEqual(config.output_dir, "./custom_output")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.learning_rate, 1e-4)
        # 其他参数应该保持默认值
        self.assertEqual(config.max_memory_gb, 13.0)


class TestTrainingEngine(unittest.TestCase):
    """测试TrainingEngine类"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            output_dir=self.temp_dir,
            batch_size=2,
            num_epochs=1,
            max_memory_gb=8.0
        )
        
        # Mock memory optimizer to avoid CUDA requirements
        with patch('src.training_engine.MemoryOptimizer') as mock_memory_optimizer:
            mock_instance = Mock()
            mock_instance.get_memory_status.return_value = Mock(
                allocated_gb=2.0,
                available_gb=6.0,
                is_safe=True
            )
            mock_memory_optimizer.return_value = mock_instance
            
            self.engine = TrainingEngine(self.config)
            self.engine.memory_optimizer = mock_instance
    
    def tearDown(self):
        """清理测试环境"""
        # 清理日志处理器以释放文件句柄
        import logging
        logger = logging.getLogger('qwen3_training')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # 等待一下让文件句柄释放
        import time
        time.sleep(0.1)
        
        if Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # Windows上可能需要多次尝试
                time.sleep(0.5)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    pass  # 忽略清理失败
    
    def test_initialization(self):
        """测试训练引擎初始化"""
        self.assertEqual(self.engine.config.output_dir, self.temp_dir)
        self.assertIsNotNone(self.engine.memory_optimizer)
        
        # 检查输出目录是否创建
        self.assertTrue(Path(self.temp_dir).exists())
    
    def test_create_training_args(self):
        """测试训练参数创建"""
        training_args = self.engine.create_training_args()
        
        self.assertEqual(training_args.output_dir, self.temp_dir)
        self.assertEqual(training_args.per_device_train_batch_size, 2)
        self.assertEqual(training_args.gradient_accumulation_steps, 16)
        self.assertEqual(training_args.num_train_epochs, 1)
        self.assertEqual(training_args.optim, "paged_adamw_8bit")
        self.assertTrue(training_args.gradient_checkpointing)
        self.assertEqual(training_args.save_total_limit, 3)
    
    @patch('src.training_engine.Trainer')
    def test_create_trainer(self, mock_trainer_class):
        """测试训练器创建"""
        # Mock dependencies
        mock_model = Mock()
        mock_dataset = Mock()
        mock_tokenizer = Mock()
        mock_data_collator = Mock()
        
        # Mock trainer instance
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        trainer = self.engine.create_trainer(
            model=mock_model,
            train_dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            data_collator=mock_data_collator
        )
        
        # 验证Trainer被正确调用
        mock_trainer_class.assert_called_once()
        call_args = mock_trainer_class.call_args
        
        self.assertEqual(call_args[1]['model'], mock_model)
        self.assertEqual(call_args[1]['train_dataset'], mock_dataset)
        self.assertEqual(call_args[1]['processing_class'], mock_tokenizer)  # 使用 processing_class 替代 tokenizer
        self.assertEqual(call_args[1]['data_collator'], mock_data_collator)
        
        # 检查回调函数
        callbacks = call_args[1]['callbacks']
        # 检查是否包含内存监控回调（可能不是第一个）
        memory_callback_found = any(isinstance(cb, MemoryMonitoringCallback) for cb in callbacks)
        self.assertTrue(memory_callback_found, "应该包含内存监控回调")
    
    def test_disk_space_check(self):
        """测试磁盘空间检查"""
        # 这个测试应该通过，因为临时目录通常有足够空间
        try:
            self.engine._check_disk_space(self.temp_dir, required_gb=0.001)  # 很小的需求
        except RuntimeError:
            self.fail("磁盘空间检查失败")
    
    def test_cleanup_old_checkpoints(self):
        """测试检查点清理"""
        # 创建一些模拟检查点目录
        checkpoint_dirs = []
        for i in range(5):
            checkpoint_dir = Path(self.temp_dir) / f"checkpoint-{i*100}"
            checkpoint_dir.mkdir()
            
            # 创建一个小文件
            (checkpoint_dir / "test_file.txt").write_text("test content")
            checkpoint_dirs.append(checkpoint_dir)
        
        # 执行清理
        freed_space = self.engine._cleanup_old_checkpoints(self.temp_dir)
        
        # 验证只保留了指定数量的检查点
        remaining_checkpoints = [d for d in Path(self.temp_dir).iterdir() 
                               if d.is_dir() and d.name.startswith("checkpoint-")]
        
        self.assertEqual(len(remaining_checkpoints), self.config.save_total_limit)
        self.assertGreaterEqual(freed_space, 0)
    
    def test_get_disk_usage_report(self):
        """测试磁盘使用报告"""
        # 创建一些检查点
        checkpoint_dir = Path(self.temp_dir) / "checkpoint-100"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.bin").write_text("model data")
        
        report = self.engine.get_disk_usage_report(self.temp_dir)
        
        self.assertIn("total_gb", report)
        self.assertIn("free_gb", report)
        self.assertIn("checkpoint_count", report)
        self.assertEqual(report["checkpoint_count"], 1)
    
    @patch('src.training_engine.torch.cuda.is_available')
    def test_training_args_without_cuda(self, mock_cuda_available):
        """测试在没有CUDA的情况下创建训练参数"""
        mock_cuda_available.return_value = False
        
        training_args = self.engine.create_training_args()
        
        # 在没有CUDA的情况下，应该禁用BF16
        self.assertFalse(training_args.bf16)
    
    def test_error_guidance(self):
        """测试错误指导功能"""
        # 测试内存错误
        memory_error = Exception("CUDA out of memory")
        
        # 这个方法主要是记录日志，我们只需要确保它不会抛出异常
        try:
            self.engine._provide_error_guidance(memory_error)
        except Exception as e:
            self.fail(f"错误指导功能抛出异常: {e}")


class TestMemoryMonitoringCallback(unittest.TestCase):
    """测试内存监控回调"""
    
    def setUp(self):
        """设置测试环境"""
        # Mock memory optimizer
        self.mock_memory_optimizer = Mock()
        self.mock_memory_optimizer.get_memory_status.return_value = Mock(
            allocated_gb=2.0,
            cached_gb=1.0,
            available_gb=6.0,
            is_safe=True
        )
        
        self.callback = MemoryMonitoringCallback(self.mock_memory_optimizer)
    
    def test_initialization(self):
        """测试回调初始化"""
        self.assertEqual(self.callback.step_count, 0)
        self.assertEqual(len(self.callback.memory_history), 0)
        self.assertEqual(self.callback.last_cleanup_step, 0)
    
    def test_on_step_begin(self):
        """测试步骤开始回调"""
        # 模拟多个步骤
        for i in range(15):
            self.callback.on_step_begin(None, None, None)
        
        self.assertEqual(self.callback.step_count, 15)
        
        # 应该有内存历史记录（每10步记录一次）
        self.assertGreater(len(self.callback.memory_history), 0)
    
    def test_on_log(self):
        """测试日志回调"""
        # 设置步骤计数为50的倍数以触发日志记录
        self.callback.step_count = 50
        
        logs = {}
        self.callback.on_log(None, None, None, logs=logs)
        
        # 检查是否添加了内存信息
        self.assertIn("memory_allocated_gb", logs)
        self.assertIn("memory_is_safe", logs)
    
    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        # 模拟内存持续增长，需要模拟更多步骤因为只有每10步才记录一次
        for i in range(25):  # 增加到25步以确保有足够的记录
            self.mock_memory_optimizer.get_memory_status.return_value = Mock(
                allocated_gb=2.0 + i * 0.2,  # 内存持续增长
                cached_gb=1.0,
                available_gb=6.0 - i * 0.2,
                is_safe=True
            )
            self.callback.on_step_begin(None, None, None)
        
        # 检查是否有内存历史记录（每10步记录一次，25步应该有2-3个记录）
        self.assertGreater(len(self.callback.memory_history), 1)


class TestCreateTrainingConfig(unittest.TestCase):
    """测试训练配置创建函数"""
    
    def test_default_parameters(self):
        """测试默认参数"""
        config = create_training_config()
        
        self.assertEqual(config.output_dir, "./qwen3-finetuned")
        self.assertEqual(config.max_memory_gb, 13.0)
        self.assertEqual(config.batch_size, 4)
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        config = create_training_config(
            output_dir="./test_output",
            batch_size=8,
            learning_rate=1e-4,
            warmup_ratio=0.2
        )
        
        self.assertEqual(config.output_dir, "./test_output")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.warmup_ratio, 0.2)


if __name__ == "__main__":
    # 设置日志级别以减少测试输出
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    unittest.main()