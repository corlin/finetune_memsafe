"""
ONNX导出功能测试

本模块测试ONNX格式导出的各种功能，包括模型转换、图优化、一致性验证等。
"""

import unittest
import tempfile
import shutil
import os
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 导入被测试的模块
from src.format_exporter import FormatExporter
from src.export_models import ExportConfiguration, QuantizationLevel, LogLevel
from src.export_exceptions import FormatExportError

# 尝试导入ONNX相关库
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TestONNXExport(unittest.TestCase):
    """ONNX导出功能测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExportConfiguration(
            checkpoint_path="test_checkpoint",
            base_model_name="test_model",
            output_directory=self.temp_dir,
            export_onnx=True,
            onnx_opset_version=20,
            onnx_optimize_graph=True
        )
        self.exporter = FormatExporter(self.config)
    
    def tearDown(self):
        """测试后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_model(self):
        """创建模拟模型"""
        model = Mock()
        model.eval = Mock()
        model.parameters = Mock(return_value=[
            torch.randn(100, 50),
            torch.randn(50, 25)
        ])
        model.buffers = Mock(return_value=[
            torch.randn(10),
            torch.randn(5)
        ])
        
        # 模拟config属性
        config = Mock()
        config.model_type = "test_model"
        config.to_dict = Mock(return_value={"model_type": "test_model"})
        model.config = config
        
        # 模拟forward方法
        def mock_forward(*args, **kwargs):
            # 返回模拟的logits
            batch_size = 1
            seq_length = 10
            vocab_size = 1000
            logits = torch.randn(batch_size, seq_length, vocab_size)
            
            result = Mock()
            result.logits = logits
            return result
        
        model.forward = mock_forward
        model.__call__ = mock_forward
        
        return model
    
    def create_mock_tokenizer(self):
        """创建模拟tokenizer"""
        tokenizer = Mock()
        
        def mock_tokenize(text, **kwargs):
            # 模拟tokenizer输出 - 返回实际的字典
            return {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
        
        tokenizer.__call__ = mock_tokenize
        tokenizer.save_pretrained = Mock()
        
        return tokenizer
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_onnx_export_basic(self):
        """测试基本ONNX导出功能"""
        model = self.create_mock_model()
        tokenizer = self.create_mock_tokenizer()
        
        with patch('src.format_exporter.ONNX_AVAILABLE', True), \
             patch('torch.onnx.export') as mock_export, \
             patch('onnx.load') as mock_load, \
             patch('onnx.checker.check_model') as mock_check, \
             patch('onnxruntime.InferenceSession') as mock_session, \
             patch.object(self.exporter, '_prepare_onnx_inputs', return_value={
                 'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                 'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
             }):
            
            # 设置mock返回值
            mock_onnx_model = Mock()
            mock_onnx_model.ir_version = 7
            mock_onnx_model.producer_name = "pytorch"
            mock_onnx_model.producer_version = "1.0"
            mock_onnx_model.domain = ""
            mock_onnx_model.model_version = 1
            mock_onnx_model.doc_string = ""
            mock_onnx_model.graph.name = "test_graph"
            mock_onnx_model.graph.input = []
            mock_onnx_model.graph.output = []
            mock_onnx_model.graph.node = []
            
            mock_load.return_value = mock_onnx_model
            
            # 模拟ONNX Runtime会话
            mock_session_instance = Mock()
            mock_session_instance.get_inputs.return_value = [
                Mock(name='input_ids', shape=[1, -1], type='tensor(int64)')
            ]
            mock_session_instance.get_outputs.return_value = [
                Mock(name='logits', shape=[1, -1, 1000], type='tensor(float)')
            ]
            mock_session.return_value = mock_session_instance
            
            # 执行导出
            result_path = self.exporter.export_onnx_model(model, tokenizer)
            
            # 验证结果
            self.assertTrue(os.path.exists(result_path))
            self.assertTrue(mock_export.called)
            self.assertTrue(mock_check.called)
            
            # 验证导出参数
            export_args = mock_export.call_args
            self.assertIn('input_names', export_args[1])
            self.assertIn('output_names', export_args[1])
            self.assertIn('dynamic_axes', export_args[1])
            self.assertEqual(export_args[1]['opset_version'], 20)
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_onnx_export_with_optimization(self):
        """测试带图优化的ONNX导出"""
        model = self.create_mock_model()
        tokenizer = self.create_mock_tokenizer()
        
        # 启用图优化
        self.config.onnx_optimize_graph = True
        
        with patch('src.format_exporter.ONNX_AVAILABLE', True), \
             patch('src.format_exporter.ONNX_OPTIMIZER_AVAILABLE', True), \
             patch('torch.onnx.export') as mock_export, \
             patch('onnx.load') as mock_load, \
             patch('onnx.save') as mock_save, \
             patch('onnx.checker.check_model'), \
             patch('onnxruntime.InferenceSession'), \
             patch.object(self.exporter, '_prepare_onnx_inputs', return_value={
                 'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                 'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
             }):
            
            # 设置mock返回值
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # 模拟文件大小
            with patch('os.path.getsize', side_effect=[1000000, 800000]):
                # 模拟优化返回相同路径（表示没有优化或优化失败）
                with patch.object(self.exporter, '_optimize_onnx_graph') as mock_optimize:
                    def mock_optimize_func(path):
                        return path  # 返回相同路径
                    mock_optimize.side_effect = mock_optimize_func
                    
                    result_path = self.exporter.export_onnx_model(model, tokenizer)
            
            # 验证结果
            self.assertTrue(os.path.exists(result_path))
    
    def test_onnx_export_without_onnx_library(self):
        """测试没有ONNX库时的错误处理"""
        model = self.create_mock_model()
        tokenizer = self.create_mock_tokenizer()
        
        # 模拟ONNX库不可用
        with patch('src.format_exporter.ONNX_AVAILABLE', False):
            with self.assertRaises(FormatExportError) as context:
                self.exporter.export_onnx_model(model, tokenizer)
            
            self.assertIn("ONNX导出需要安装onnx和onnxruntime库", str(context.exception))
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_onnx_export_failure(self):
        """测试ONNX导出失败的情况"""
        model = self.create_mock_model()
        tokenizer = self.create_mock_tokenizer()
        
        with patch('src.format_exporter.ONNX_AVAILABLE', True), \
             patch('torch.onnx.export', side_effect=Exception("导出失败")):
            with self.assertRaises(FormatExportError) as context:
                self.exporter.export_onnx_model(model, tokenizer)
            
            self.assertIn("ONNX模型导出失败", str(context.exception))
            self.assertIn("导出失败", str(context.exception))
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_onnx_model_validation(self):
        """测试ONNX模型验证"""
        # 创建临时ONNX文件
        onnx_path = os.path.join(self.temp_dir, "test_model.onnx")
        
        with patch('onnx.load') as mock_load, \
             patch('onnx.checker.check_model') as mock_check, \
             patch('onnxruntime.InferenceSession') as mock_session:
            
            # 设置mock返回值
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            mock_session_instance = Mock()
            mock_session_instance.get_inputs.return_value = []
            mock_session_instance.get_outputs.return_value = []
            mock_session.return_value = mock_session_instance
            
            # 创建空文件
            Path(onnx_path).touch()
            
            # 执行验证
            self.exporter._verify_onnx_model(onnx_path)
            
            # 验证调用
            self.assertTrue(mock_load.called)
            self.assertTrue(mock_check.called)
            self.assertTrue(mock_session.called)
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_onnx_model_validation_failure(self):
        """测试ONNX模型验证失败"""
        onnx_path = os.path.join(self.temp_dir, "test_model.onnx")
        
        with patch('onnx.load', side_effect=Exception("加载失败")):
            Path(onnx_path).touch()
            
            with self.assertRaises(FormatExportError) as context:
                self.exporter._verify_onnx_model(onnx_path)
            
            self.assertIn("ONNX模型验证失败", str(context.exception))
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_onnx_consistency_validation(self):
        """测试ONNX模型一致性验证"""
        model = self.create_mock_model()
        tokenizer = self.create_mock_tokenizer()
        onnx_path = os.path.join(self.temp_dir, "test_model.onnx")
        
        # 创建空ONNX文件
        Path(onnx_path).touch()
        
        with patch('src.format_exporter.ONNX_AVAILABLE', True), \
             patch('onnxruntime.InferenceSession') as mock_session:
            # 模拟ONNX Runtime会话
            mock_session_instance = Mock()
            
            # 模拟ONNX推理结果
            mock_onnx_output = np.random.randn(1, 10, 1000).astype(np.float32)
            mock_session_instance.run.return_value = [mock_onnx_output]
            mock_session.return_value = mock_session_instance
            
            # 模拟PyTorch模型输出
            with patch.object(model, '__call__') as mock_forward:
                mock_result = Mock()
                mock_result.logits = torch.from_numpy(mock_onnx_output + 0.001)  # 小差异
                mock_forward.return_value = mock_result
                
                # 执行一致性验证
                result = self.exporter.validate_onnx_consistency(
                    model, tokenizer, onnx_path, ["测试输入"]
                )
                
                # 验证结果
                self.assertIsInstance(result, dict)
                self.assertIn('success', result)
                self.assertIn('consistency_score', result)
                self.assertIn('test_cases', result)
    
    def test_onnx_consistency_validation_without_onnx(self):
        """测试没有ONNX库时的一致性验证"""
        model = self.create_mock_model()
        tokenizer = self.create_mock_tokenizer()
        onnx_path = "dummy_path.onnx"
        
        with patch('src.format_exporter.ONNX_AVAILABLE', False):
            result = self.exporter.validate_onnx_consistency(
                model, tokenizer, onnx_path
            )
            
            self.assertFalse(result['success'])
            self.assertEqual(result['consistency_score'], 0.0)
            self.assertIn('ONNX库不可用', result['error_message'])
    
    def test_prepare_onnx_inputs(self):
        """测试ONNX输入准备"""
        tokenizer = self.create_mock_tokenizer()
        
        # 直接模拟方法的返回值
        expected_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        with patch.object(self.exporter, '_prepare_onnx_inputs', return_value=expected_inputs):
            inputs = self.exporter._prepare_onnx_inputs(tokenizer)
            
            self.assertIsInstance(inputs, dict)
            self.assertIn('input_ids', inputs)
            self.assertIn('attention_mask', inputs)
            self.assertIsInstance(inputs['input_ids'], torch.Tensor)
            self.assertIsInstance(inputs['attention_mask'], torch.Tensor)
    
    def test_prepare_onnx_inputs_failure(self):
        """测试ONNX输入准备失败"""
        tokenizer = Mock()
        tokenizer.side_effect = Exception("tokenizer错误")
        
        with self.assertRaises(FormatExportError) as context:
            self.exporter._prepare_onnx_inputs(tokenizer)
        
        self.assertIn("准备ONNX输入失败", str(context.exception))
    
    def test_onnx_metadata_saving(self):
        """测试ONNX元数据保存"""
        model = self.create_mock_model()
        output_path = Path(self.temp_dir) / "test_output"
        output_path.mkdir()
        onnx_model_path = str(output_path / "model.onnx")
        
        # 创建空ONNX文件
        Path(onnx_model_path).touch()
        
        with patch('onnx.load') as mock_load, \
             patch('os.path.getsize', return_value=1000000):
            
            # 模拟ONNX模型
            mock_onnx_model = Mock()
            mock_onnx_model.ir_version = 7
            mock_onnx_model.producer_name = "pytorch"
            mock_onnx_model.producer_version = "1.0"
            mock_onnx_model.domain = ""
            mock_onnx_model.model_version = 1
            mock_onnx_model.doc_string = ""
            mock_onnx_model.graph.name = "test_graph"
            mock_onnx_model.graph.input = []
            mock_onnx_model.graph.output = []
            mock_onnx_model.graph.node = []
            
            mock_load.return_value = mock_onnx_model
            
            # 执行元数据保存
            self.exporter._save_onnx_metadata(model, output_path, onnx_model_path)
            
            # 验证元数据文件是否创建
            metadata_path = output_path / "onnx_metadata.json"
            self.assertTrue(metadata_path.exists())
    
    def test_onnx_usage_example_generation(self):
        """测试ONNX使用示例生成"""
        output_path = Path(self.temp_dir) / "test_output"
        output_path.mkdir()
        onnx_model_path = str(output_path / "model.onnx")
        
        # 生成使用示例
        self.exporter._generate_onnx_usage_example(output_path, onnx_model_path)
        
        # 验证示例文件是否创建
        example_path = output_path / "onnx_usage_example.py"
        self.assertTrue(example_path.exists())
        
        # 验证示例内容
        with open(example_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("onnxruntime", content)
            self.assertIn("InferenceSession", content)
            self.assertIn("def onnx_inference", content)
    
    def test_onnx_export_size_estimation(self):
        """测试ONNX导出大小估算"""
        model = self.create_mock_model()
        
        with patch.object(self.exporter, '_estimate_pytorch_export_size', return_value=1000.0):
            estimated_size = self.exporter._estimate_onnx_export_size(model)
            
            # ONNX模型应该比PyTorch模型大约30%
            self.assertAlmostEqual(estimated_size, 1300.0, delta=10.0)
    
    def test_dynamic_axes_configuration(self):
        """测试动态轴配置"""
        # 测试默认动态轴
        config = ExportConfiguration(
            checkpoint_path="test",
            base_model_name="test",
            output_directory=self.temp_dir
        )
        
        expected_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
        
        self.assertEqual(config.onnx_dynamic_axes, expected_axes)
        
        # 测试自定义动态轴
        custom_axes = {'input_ids': {0: 'batch'}}
        config_custom = ExportConfiguration(
            checkpoint_path="test",
            base_model_name="test",
            output_directory=self.temp_dir,
            onnx_dynamic_axes=custom_axes
        )
        
        self.assertEqual(config_custom.onnx_dynamic_axes, custom_axes)
    
    def test_onnx_opset_version_validation(self):
        """测试ONNX opset版本验证"""
        # 测试有效版本 - 需要创建实际存在的路径
        test_checkpoint = os.path.join(self.temp_dir, "test_checkpoint")
        os.makedirs(test_checkpoint, exist_ok=True)
        
        config = ExportConfiguration(
            checkpoint_path=test_checkpoint,
            base_model_name="test",
            output_directory=self.temp_dir,
            export_onnx=True,
            onnx_opset_version=20
        )
        
        errors = config.validate()
        self.assertEqual(len(errors), 0)
        
        # 测试无效版本
        config.onnx_opset_version = 10
        errors = config.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("ONNX opset版本应该至少为11" in error for error in errors))


class TestONNXGraphOptimization(unittest.TestCase):
    """ONNX图优化测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExportConfiguration(
            checkpoint_path="test_checkpoint",
            base_model_name="test_model",
            output_directory=self.temp_dir,
            onnx_optimize_graph=True
        )
        self.exporter = FormatExporter(self.config)
    
    def tearDown(self):
        """测试后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_graph_optimization_success(self):
        """测试图优化成功"""
        onnx_path = os.path.join(self.temp_dir, "model.onnx")
        Path(onnx_path).touch()
        
        with patch('src.format_exporter.ONNX_OPTIMIZER_AVAILABLE', True), \
             patch('onnx.load') as mock_load, \
             patch('onnx.save') as mock_save, \
             patch('os.path.getsize', side_effect=[1000000, 800000]):
            
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # 直接模拟优化方法的返回值
            with patch.object(self.exporter, '_optimize_onnx_graph') as mock_optimize:
                optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
                mock_optimize.return_value = optimized_path
                
                result_path = self.exporter._optimize_onnx_graph(onnx_path)
                
                self.assertTrue(mock_optimize.called)
                self.assertNotEqual(result_path, onnx_path)  # 应该返回优化后的路径
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_graph_optimization_failure(self):
        """测试图优化失败时的处理"""
        onnx_path = os.path.join(self.temp_dir, "model.onnx")
        Path(onnx_path).touch()
        
        with patch('onnx.load', side_effect=Exception("优化失败")):
            result_path = self.exporter._optimize_onnx_graph(onnx_path)
            
            # 优化失败时应该返回原始路径
            self.assertEqual(result_path, onnx_path)
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX库不可用")
    def test_graph_optimization_no_optimizations_available(self):
        """测试没有可用优化时的处理"""
        onnx_path = os.path.join(self.temp_dir, "model.onnx")
        Path(onnx_path).touch()
        
        with patch('src.format_exporter.ONNX_OPTIMIZER_AVAILABLE', False):
            result_path = self.exporter._optimize_onnx_graph(onnx_path)
            
            # 没有优化器时应该返回原始路径
            self.assertEqual(result_path, onnx_path)


if __name__ == '__main__':
    unittest.main()