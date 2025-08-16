"""
集成测试模块

测试系统各组件之间的集成，包括端到端训练流程、内存约束测试和错误恢复测试。
"""

import pytest
import torch
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory_optimizer import MemoryOptimizer
from src.model_manager import ModelManager
from src.data_pipeline import DataPipeline, QAData
from src.training_engine import TrainingEngine, TrainingConfig
from src.memory_exceptions import OutOfMemoryError, InsufficientMemoryError


class TestEndToEndIntegration(unittest.TestCase):
    """端到端集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试数据文件
        self._create_test_data_files()
        
        # 配置
        self.config = TrainingConfig(
            output_dir=str(Path(self.temp_dir) / "output"),
            batch_size=1,  # 小批次用于测试
            num_epochs=1,
            max_memory_gb=8.0,
            gradient_accumulation_steps=2
        )
    
    def tearDown(self):
        """测试后清理"""
        if Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # 在Windows上，有时文件被占用，稍后再试
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    # 如果仍然失败，跳过清理
                    pass
    
    def _create_test_data_files(self):
        """创建测试数据文件"""
        test_content = """# 测试QA数据

### Q1: 什么是机器学习？

A1: 机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。

### Q2: 什么是深度学习？

A2: 深度学习是机器学习的一个子集，使用多层神经网络来学习复杂的数据表示。

### Q3: 什么是自然语言处理？

A3: 自然语言处理是计算机科学和人工智能的一个分支，专注于计算机与人类语言的交互。
"""
        test_file = self.data_dir / "test_qa.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
    
    @patch('src.memory_optimizer.torch.cuda.is_available')
    @patch('src.model_manager.AutoModelForCausalLM')
    @patch('src.model_manager.AutoTokenizer')
    def test_small_dataset_end_to_end_training(self, mock_tokenizer_class, mock_model_class, mock_cuda):
        """测试使用小数据集的端到端训练流程"""
        # Mock CUDA availability
        mock_cuda.return_value = True
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.device = "cuda:0"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000, requires_grad=True)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.__len__ = Mock(return_value=50000)
        
        # Mock tokenizer call
        def mock_tokenizer_call(texts, **kwargs):
            return {
                "input_ids": [[1, 2, 3, 4, 5] for _ in texts],
                "attention_mask": [[1, 1, 1, 1, 1] for _ in texts]
            }
        mock_tokenizer.side_effect = mock_tokenizer_call
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock memory functions
        with patch('src.memory_optimizer.torch.cuda.memory_allocated', return_value=4 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.memory_reserved', return_value=5 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.get_device_properties') as mock_props, \
             patch('src.memory_optimizer.torch.cuda.current_device', return_value=0), \
             patch('src.memory_optimizer.torch.cuda.empty_cache'), \
             patch('src.memory_optimizer.torch.cuda.set_per_process_memory_fraction'):
            
            # Mock device properties
            mock_device_props = Mock()
            mock_device_props.total_memory = 16 * (1024**3)
            mock_props.return_value = mock_device_props
            
            # 1. 初始化组件
            memory_optimizer = MemoryOptimizer(max_memory_gb=8.0)
            model_manager = ModelManager(max_memory_gb=8.0)
            data_pipeline = DataPipeline(str(self.data_dir), max_sequence_length=128)
            
            # 2. 加载和处理数据
            qa_data = data_pipeline.load_qa_data_from_files()
            self.assertGreater(len(qa_data), 0)
            
            formatted_dataset = data_pipeline.format_for_qwen(qa_data)
            self.assertGreater(len(formatted_dataset), 0)
            
            # Mock dataset tokenization
            with patch.object(formatted_dataset, 'map') as mock_map:
                mock_tokenized_dataset = Mock()
                mock_tokenized_dataset.filter.return_value = Dataset.from_dict({
                    "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                    "labels": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
                })
                mock_map.return_value = mock_tokenized_dataset
                
                tokenized_dataset = data_pipeline.tokenize_dataset(formatted_dataset, mock_tokenizer)
                self.assertGreater(len(tokenized_dataset), 0)
            
            # 3. 加载模型
            model, tokenizer = model_manager.load_model_with_quantization("test-model")
            prepared_model = model_manager.prepare_for_training(model)
            
            # 4. 创建数据整理器
            data_collator = data_pipeline.create_data_collator(tokenizer)
            
            # 5. 验证内存状态
            memory_status = memory_optimizer.get_memory_status()
            self.assertTrue(memory_status.is_safe)
            
            # 验证所有组件都正确初始化
            self.assertIsNotNone(memory_optimizer)
            self.assertIsNotNone(model_manager)
            self.assertIsNotNone(data_pipeline)
            self.assertIsNotNone(prepared_model)
            self.assertIsNotNone(tokenizer)
            self.assertIsNotNone(data_collator)
    
    @patch('src.memory_optimizer.torch.cuda.is_available')
    def test_memory_constraint_simulation(self, mock_cuda):
        """测试模拟低内存条件的内存约束测试"""
        mock_cuda.return_value = True
        
        # 模拟低内存条件
        with patch('src.memory_optimizer.torch.cuda.memory_allocated', return_value=7 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.memory_reserved', return_value=8 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.get_device_properties') as mock_props, \
             patch('src.memory_optimizer.torch.cuda.current_device', return_value=0), \
             patch('src.memory_optimizer.torch.cuda.empty_cache'), \
             patch('src.memory_optimizer.torch.cuda.set_per_process_memory_fraction'):
            
            # Mock device properties - 8GB total memory
            mock_device_props = Mock()
            mock_device_props.total_memory = 8 * (1024**3)
            mock_props.return_value = mock_device_props
            
            # 创建内存优化器，限制为6GB
            memory_optimizer = MemoryOptimizer(max_memory_gb=6.0, safety_threshold=0.8)
            
            # 检查内存状态 - 应该不安全
            status = memory_optimizer.get_memory_status()
            self.assertFalse(status.is_safe)  # 7GB > 6GB * 0.8 = 4.8GB
            
            # 测试内存清理
            memory_optimizer.cleanup_gpu_memory()
            
            # 测试内存安全检查失败
            result = memory_optimizer.check_memory_safety()
            self.assertFalse(result)
    
    @patch('src.memory_optimizer.torch.cuda.is_available')
    def test_oom_error_recovery_scenario(self, mock_cuda):
        """测试OOM场景的错误恢复测试"""
        mock_cuda.return_value = True
        
        # 模拟OOM条件
        with patch('src.memory_optimizer.torch.cuda.memory_allocated', return_value=15 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.memory_reserved', return_value=16 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.get_device_properties') as mock_props, \
             patch('src.memory_optimizer.torch.cuda.current_device', return_value=0), \
             patch('src.memory_optimizer.torch.cuda.empty_cache'), \
             patch('src.memory_optimizer.torch.cuda.set_per_process_memory_fraction'):
            
            # Mock device properties
            mock_device_props = Mock()
            mock_device_props.total_memory = 16 * (1024**3)
            mock_props.return_value = mock_device_props
            
            memory_optimizer = MemoryOptimizer(max_memory_gb=13.0)
            
            # 应该返回False表示不安全，而不是直接抛出异常
            result = memory_optimizer.check_memory_safety()
            self.assertFalse(result)
            
            # 测试错误处理
            error = OutOfMemoryError("Test OOM", current_usage_gb=15.0)
            
            # Mock successful recovery
            with patch.object(memory_optimizer.error_handler, 'handle_out_of_memory', return_value=True):
                result = memory_optimizer.handle_memory_error(error, auto_recover=True)
                self.assertTrue(result)
    
    def test_data_pipeline_integration_with_various_formats(self):
        """测试数据管道与各种格式的集成"""
        # 创建多种格式的测试文件
        
        # 格式1文件
        format1_content = """Q1: 什么是Python？
A1: Python是一种高级编程语言。

Q2: 什么是机器学习？
A2: 机器学习是人工智能的一个分支。
"""
        format1_file = self.data_dir / "format1.md"
        with open(format1_file, 'w', encoding='utf-8') as f:
            f.write(format1_content)
        
        # 格式2文件
        format2_content = """### Q1: 什么是深度学习？

A1: 深度学习使用多层神经网络。

### Q2: 什么是NLP？

A2: NLP是自然语言处理的缩写。
"""
        format2_file = self.data_dir / "format2.md"
        with open(format2_file, 'w', encoding='utf-8') as f:
            f.write(format2_content)
        
        # 测试数据管道
        data_pipeline = DataPipeline(str(self.data_dir))
        qa_data = data_pipeline.load_qa_data_from_files()
        
        # 应该加载所有格式的数据
        self.assertGreaterEqual(len(qa_data), 4)  # 至少4个QA对
        
        # 验证数据来源
        sources = {qa.source for qa in qa_data}
        self.assertIn("format1.md", sources)
        self.assertIn("format2.md", sources)
        
        # 测试格式化
        formatted_dataset = data_pipeline.format_for_qwen(qa_data)
        self.assertEqual(len(formatted_dataset), len(qa_data))
        
        # 验证格式化结果
        for i, item in enumerate(formatted_dataset):
            self.assertIn("<|im_start|>user", item["text"])
            self.assertIn("<|im_start|>assistant", item["text"])
            self.assertIn("<|im_end|>", item["text"])
    
    @patch('src.training_engine.Trainer')
    def test_training_engine_integration(self, mock_trainer_class):
        """测试训练引擎集成"""
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "labels": [[1, 2, 3], [4, 5, 6]]
        })
        mock_data_collator = Mock()
        
        # Mock memory optimizer
        with patch('src.training_engine.MemoryOptimizer') as mock_memory_optimizer_class:
            mock_memory_optimizer = Mock()
            mock_memory_optimizer.get_memory_status.return_value = Mock(
                allocated_gb=4.0,
                available_gb=4.0,
                is_safe=True
            )
            mock_memory_optimizer_class.return_value = mock_memory_optimizer
            
            # 创建训练引擎
            training_engine = TrainingEngine(self.config)
            
            # 创建训练器
            trainer = training_engine.create_trainer(
                model=mock_model,
                train_dataset=mock_dataset,
                tokenizer=mock_tokenizer,
                data_collator=mock_data_collator
            )
            
            # 验证训练器创建
            mock_trainer_class.assert_called_once()
            
            # 验证训练器创建
            call_args = mock_trainer_class.call_args
            # 检查关键参数是否传递
            self.assertIsNotNone(call_args)
            
            # 验证传递的参数包含必要的组件
            kwargs = call_args[1] if len(call_args) > 1 else call_args[0]
            self.assertEqual(kwargs['model'], mock_model)
            self.assertEqual(kwargs['train_dataset'], mock_dataset)
            self.assertEqual(kwargs['data_collator'], mock_data_collator)
    
    def test_error_propagation_between_components(self):
        """测试组件间错误传播"""
        # 测试数据管道错误处理
        data_pipeline = DataPipeline("nonexistent_directory")
        
        # 应该回退到示例数据而不是抛出异常
        qa_data = data_pipeline.load_qa_data_from_files()
        self.assertGreater(len(qa_data), 0)
        self.assertTrue(all(qa.source == "example_data" for qa in qa_data))
        
        # 测试空数据集处理
        empty_dataset = Dataset.from_dict({"text": []})
        formatted_empty = data_pipeline.format_for_qwen([])
        self.assertEqual(len(formatted_empty), 0)


class TestMemoryConstraintIntegration(unittest.TestCase):
    """内存约束集成测试"""
    
    @patch('src.memory_optimizer.torch.cuda.is_available')
    def test_memory_monitoring_during_operations(self, mock_cuda):
        """测试操作期间的内存监控"""
        mock_cuda.return_value = True
        
        # 模拟内存使用逐渐增加
        memory_values = [4, 6, 8, 10, 12]  # GB
        memory_iter = iter(memory_values)
        
        def mock_memory_allocated(*args):
            try:
                return next(memory_iter) * (1024**3)
            except StopIteration:
                return 12 * (1024**3)
        
        with patch('src.memory_optimizer.torch.cuda.memory_allocated', side_effect=mock_memory_allocated), \
             patch('src.memory_optimizer.torch.cuda.memory_reserved', return_value=8 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.get_device_properties') as mock_props, \
             patch('src.memory_optimizer.torch.cuda.current_device', return_value=0), \
             patch('src.memory_optimizer.torch.cuda.empty_cache'), \
             patch('src.memory_optimizer.torch.cuda.set_per_process_memory_fraction'):
            
            # Mock device properties
            mock_device_props = Mock()
            mock_device_props.total_memory = 16 * (1024**3)
            mock_props.return_value = mock_device_props
            
            memory_optimizer = MemoryOptimizer(max_memory_gb=10.0, safety_threshold=0.8)
            
            # 模拟多次操作，内存逐渐增加
            for i in range(5):
                status = memory_optimizer.get_memory_status()
                
                if i < 2:  # 4GB和6GB应该安全 (< 10GB * 0.8 = 8GB)
                    self.assertTrue(status.is_safe)
                else:  # 8GB, 10GB, 12GB应该不安全 (>= 8GB)
                    self.assertFalse(status.is_safe)
    
    @patch('src.memory_optimizer.torch.cuda.is_available')
    def test_automatic_memory_cleanup_integration(self, mock_cuda):
        """测试自动内存清理集成"""
        mock_cuda.return_value = True
        
        cleanup_called = False
        
        def mock_empty_cache():
            nonlocal cleanup_called
            cleanup_called = True
        
        with patch('src.memory_optimizer.torch.cuda.memory_allocated', return_value=9 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.memory_reserved', return_value=10 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.get_device_properties') as mock_props, \
             patch('src.memory_optimizer.torch.cuda.current_device', return_value=0), \
             patch('src.memory_optimizer.torch.cuda.empty_cache', side_effect=mock_empty_cache), \
             patch('src.memory_optimizer.torch.cuda.set_per_process_memory_fraction'):
            
            # Mock device properties
            mock_device_props = Mock()
            mock_device_props.total_memory = 16 * (1024**3)
            mock_props.return_value = mock_device_props
            
            memory_optimizer = MemoryOptimizer(max_memory_gb=10.0)
            
            # 执行清理
            memory_optimizer.cleanup_gpu_memory()
            
            # 验证清理被调用
            self.assertTrue(cleanup_called)
    
    def test_memory_error_recovery_chain(self):
        """测试内存错误恢复链"""
        # 创建模拟的内存优化器
        mock_memory_optimizer = Mock()
        
        # 模拟错误恢复链
        from src.memory_exceptions import MemoryErrorHandler
        
        error_handler = MemoryErrorHandler(mock_memory_optimizer)
        
        # 测试OOM错误恢复
        oom_error = OutOfMemoryError("Test OOM", current_usage_gb=15.0)
        
        # Mock successful cleanup
        mock_memory_optimizer.cleanup_gpu_memory.return_value = None
        mock_memory_optimizer.check_memory_safety.return_value = True
        
        result = error_handler.handle_out_of_memory(oom_error, auto_recover=True)
        self.assertTrue(result)
        
        # 验证清理被调用
        mock_memory_optimizer.cleanup_gpu_memory.assert_called()
        mock_memory_optimizer.check_memory_safety.assert_called()


class TestComponentInteractionIntegration(unittest.TestCase):
    """组件交互集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.memory_optimizer.torch.cuda.is_available')
    @patch('src.model_manager.AutoModelForCausalLM')
    @patch('src.model_manager.AutoTokenizer')
    def test_model_manager_memory_optimizer_interaction(self, mock_tokenizer_class, mock_model_class, mock_cuda):
        """测试模型管理器与内存优化器的交互"""
        mock_cuda.return_value = True
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.__len__ = Mock(return_value=50000)
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        with patch('src.memory_optimizer.torch.cuda.memory_allocated', return_value=4 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.memory_reserved', return_value=5 * (1024**3)), \
             patch('src.memory_optimizer.torch.cuda.get_device_properties') as mock_props, \
             patch('src.memory_optimizer.torch.cuda.current_device', return_value=0), \
             patch('src.memory_optimizer.torch.cuda.empty_cache'), \
             patch('src.memory_optimizer.torch.cuda.set_per_process_memory_fraction'):
            
            # Mock device properties
            mock_device_props = Mock()
            mock_device_props.total_memory = 16 * (1024**3)
            mock_props.return_value = mock_device_props
            
            # 创建组件
            memory_optimizer = MemoryOptimizer(max_memory_gb=13.0)
            model_manager = ModelManager(max_memory_gb=13.0)
            
            # 检查初始内存状态
            initial_status = memory_optimizer.get_memory_status()
            self.assertTrue(initial_status.is_safe)
            
            # 加载模型（模拟内存使用增加）
            model, tokenizer = model_manager.load_model_with_quantization("test-model")
            
            # 验证模型加载成功
            self.assertIsNotNone(model)
            self.assertIsNotNone(tokenizer)
            
            # 准备训练
            prepared_model = model_manager.prepare_for_training(model)
            
            # 验证梯度检查点启用
            mock_model.gradient_checkpointing_enable.assert_called_once()
    
    def test_data_pipeline_training_engine_interaction(self):
        """测试数据管道与训练引擎的交互"""
        # 创建测试数据
        qa_data = [
            QAData("问题1", "答案1", "test"),
            QAData("问题2", "答案2", "test")
        ]
        
        data_pipeline = DataPipeline(str(self.temp_dir))
        
        # 格式化数据
        formatted_dataset = data_pipeline.format_for_qwen(qa_data)
        self.assertEqual(len(formatted_dataset), 2)
        
        # 验证格式化结果
        for item in formatted_dataset:
            self.assertIn("<|im_start|>user", item["text"])
            self.assertIn("<|im_start|>assistant", item["text"])
        
        # 创建模拟分词器
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        
        # 创建数据整理器
        data_collator = data_pipeline.create_data_collator(mock_tokenizer)
        self.assertIsNotNone(data_collator)
        
        # 测试数据整理器
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "labels": [4, 5]}
        ]
        
        batch = data_collator(features)
        
        # 验证批次结构
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertIn("attention_mask", batch)


if __name__ == "__main__":
    # 设置日志级别以减少测试输出
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    unittest.main()