"""
验证测试系统测试

本模块测试ValidationTester类的各种功能，包括基本功能测试、输出对比测试、
性能基准测试和报告生成功能。
"""

import unittest
import tempfile
import shutil
import os
import json
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 导入被测试的模块
from src.validation_tester import ValidationTester, ValidationReport, PerformanceBenchmark
from src.export_models import ExportConfiguration, QuantizationLevel, LogLevel
from src.export_exceptions import ValidationError


class TestValidationTester(unittest.TestCase):
    """验证测试器测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExportConfiguration(
            checkpoint_path="test_checkpoint",
            base_model_name="test_model",
            output_directory=self.temp_dir,
            run_validation_tests=True,
            log_level=LogLevel.INFO
        )
        self.tester = ValidationTester(self.config)
    
    def tearDown(self):
        """测试后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_pytorch_model_path(self):
        """创建模拟PyTorch模型路径"""
        model_dir = Path(self.temp_dir) / "pytorch_model"
        model_dir.mkdir(exist_ok=True)
        
        # 创建必要的文件
        (model_dir / "config.json").write_text('{"model_type": "test_model"}')
        (model_dir / "pytorch_model.bin").write_text("fake model data")
        (model_dir / "tokenizer_config.json").write_text('{"tokenizer_class": "GPT2Tokenizer"}')
        
        return str(model_dir)
    
    def create_mock_onnx_model_path(self):
        """创建模拟ONNX模型路径"""
        model_dir = Path(self.temp_dir) / "onnx_model"
        model_dir.mkdir(exist_ok=True)
        
        # 创建ONNX文件
        (model_dir / "model.onnx").write_text("fake onnx data")
        (model_dir / "tokenizer_config.json").write_text('{"tokenizer_class": "GPT2Tokenizer"}')
        
        return str(model_dir)
    
    def test_initialization(self):
        """测试验证测试器初始化"""
        self.assertIsInstance(self.tester, ValidationTester)
        self.assertEqual(self.tester.config, self.config)
        self.assertIsInstance(self.tester.test_samples, list)
        self.assertGreater(len(self.tester.test_samples), 0)
    
    def test_run_comprehensive_validation(self):
        """测试综合验证测试"""
        model_paths = {
            'pytorch': self.create_mock_pytorch_model_path(),
            'onnx': self.create_mock_onnx_model_path()
        }
        
        with patch.object(self.tester, 'run_basic_functionality_tests') as mock_basic, \
             patch.object(self.tester, 'run_output_comparison_tests') as mock_comparison, \
             patch.object(self.tester, 'run_performance_benchmark') as mock_benchmark:
            
            # 设置mock返回值
            mock_basic.return_value = [Mock(success=True)]
            mock_comparison.return_value = [{'success': True}]
            mock_benchmark.return_value = PerformanceBenchmark(
                model_format='test',
                model_path='test_path',
                success=True
            )
            
            # 执行测试
            report = self.tester.run_comprehensive_validation(model_paths)
            
            # 验证结果
            self.assertIsInstance(report, ValidationReport)
            self.assertGreater(len(report.basic_tests), 0)
            self.assertTrue(mock_basic.called)
            self.assertTrue(mock_comparison.called)
            self.assertTrue(mock_benchmark.called)
    
    def test_run_basic_functionality_tests(self):
        """测试基本功能测试"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch.object(self.tester, '_test_model_loading') as mock_load, \
             patch.object(self.tester, '_test_basic_inference') as mock_inference, \
             patch.object(self.tester, '_test_batch_inference') as mock_batch, \
             patch.object(self.tester, '_test_input_validation') as mock_validation:
            
            # 设置mock返回值
            mock_load.return_value = Mock(success=True)
            mock_inference.return_value = Mock(success=True)
            mock_batch.return_value = Mock(success=True)
            mock_validation.return_value = Mock(success=True)
            
            # 执行测试
            results = self.tester.run_basic_functionality_tests('pytorch', model_path)
            
            # 验证结果
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 4)  # 4个基本测试
            self.assertTrue(mock_load.called)
            self.assertTrue(mock_inference.called)
            self.assertTrue(mock_batch.called)
            self.assertTrue(mock_validation.called)
    
    def test_run_basic_functionality_tests_load_failure(self):
        """测试模型加载失败时的基本功能测试"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch.object(self.tester, '_test_model_loading') as mock_load:
            # 设置加载失败
            mock_load.return_value = Mock(success=False)
            
            # 执行测试
            results = self.tester.run_basic_functionality_tests('pytorch', model_path)
            
            # 验证结果 - 只有加载测试，其他测试被跳过
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 1)
            self.assertTrue(mock_load.called)
    
    def test_run_output_comparison_tests(self):
        """测试输出对比测试"""
        model_paths = {
            'pytorch': self.create_mock_pytorch_model_path(),
            'onnx': self.create_mock_onnx_model_path()
        }
        
        with patch.object(self.tester, '_compare_model_outputs') as mock_compare:
            mock_compare.return_value = {'success': True, 'similarity': 0.8}
            
            # 执行测试
            results = self.tester.run_output_comparison_tests(model_paths)
            
            # 验证结果
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            self.assertTrue(mock_compare.called)
    
    def test_run_output_comparison_tests_insufficient_models(self):
        """测试模型数量不足时的输出对比测试"""
        model_paths = {
            'pytorch': self.create_mock_pytorch_model_path()
        }
        
        # 执行测试
        results = self.tester.run_output_comparison_tests(model_paths)
        
        # 验证结果 - 应该返回空列表
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)
    
    def test_run_performance_benchmark_pytorch(self):
        """测试PyTorch性能基准测试"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch.object(self.tester, '_benchmark_pytorch_model') as mock_benchmark:
            def mock_benchmark_func(path, benchmark):
                benchmark.success = True
                benchmark.avg_inference_time_ms = 100.0
                benchmark.throughput_tokens_per_sec = 50.0
            
            mock_benchmark.side_effect = mock_benchmark_func
            
            # 执行测试
            result = self.tester.run_performance_benchmark('pytorch', model_path)
            
            # 验证结果
            self.assertIsInstance(result, PerformanceBenchmark)
            self.assertEqual(result.model_format, 'pytorch')
            self.assertTrue(result.success)
            self.assertTrue(mock_benchmark.called)
    
    def test_run_performance_benchmark_onnx(self):
        """测试ONNX性能基准测试"""
        model_path = self.create_mock_onnx_model_path()
        
        with patch.object(self.tester, '_benchmark_onnx_model') as mock_benchmark:
            def mock_benchmark_func(path, benchmark):
                benchmark.success = True
                benchmark.avg_inference_time_ms = 80.0
                benchmark.throughput_tokens_per_sec = 60.0
            
            mock_benchmark.side_effect = mock_benchmark_func
            
            # 执行测试
            result = self.tester.run_performance_benchmark('onnx', model_path)
            
            # 验证结果
            self.assertIsInstance(result, PerformanceBenchmark)
            self.assertEqual(result.model_format, 'onnx')
            self.assertTrue(result.success)
            self.assertTrue(mock_benchmark.called)
    
    def test_run_performance_benchmark_unsupported_format(self):
        """测试不支持格式的性能基准测试"""
        model_path = "/fake/path"
        
        # 执行测试
        result = self.tester.run_performance_benchmark('unsupported', model_path)
        
        # 验证结果
        self.assertIsInstance(result, PerformanceBenchmark)
        self.assertEqual(result.model_format, 'unsupported')
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
    
    def test_generate_validation_report(self):
        """测试验证报告生成"""
        report = ValidationReport(
            report_id="test_report",
            timestamp=datetime.now(),
            config=self.config
        )
        
        output_path = os.path.join(self.temp_dir, "report")
        
        with patch.object(self.tester, '_generate_json_report') as mock_json, \
             patch.object(self.tester, '_generate_html_report') as mock_html, \
             patch.object(self.tester, '_generate_performance_chart_data') as mock_chart:
            
            # 执行测试
            self.tester.generate_validation_report(report, output_path)
            
            # 验证结果
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(mock_json.called)
            self.assertTrue(mock_html.called)
    
    def test_test_model_loading_pytorch_success(self):
        """测试PyTorch模型加载成功"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch('src.validation_tester.TRANSFORMERS_AVAILABLE', True), \
             patch('src.validation_tester.AutoModelForCausalLM') as mock_model:
            
            mock_model.from_pretrained.return_value = Mock()
            
            # 执行测试
            result = self.tester._test_model_loading('pytorch', model_path)
            
            # 验证结果
            self.assertTrue(result.success)
            self.assertEqual(result.test_name, 'pytorch_model_loading')
            self.assertIsNone(result.error_message)
    
    def test_test_model_loading_pytorch_failure(self):
        """测试PyTorch模型加载失败"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch('src.validation_tester.TRANSFORMERS_AVAILABLE', True), \
             patch('src.validation_tester.AutoModelForCausalLM') as mock_model:
            
            mock_model.from_pretrained.side_effect = Exception("加载失败")
            
            # 执行测试
            result = self.tester._test_model_loading('pytorch', model_path)
            
            # 验证结果
            self.assertFalse(result.success)
            self.assertEqual(result.test_name, 'pytorch_model_loading')
            self.assertIsNotNone(result.error_message)
    
    def test_test_model_loading_onnx_success(self):
        """测试ONNX模型加载成功"""
        model_path = self.create_mock_onnx_model_path()
        
        with patch('src.validation_tester.ONNX_AVAILABLE', True), \
             patch('src.validation_tester.ort.InferenceSession') as mock_session:
            
            mock_session.return_value = Mock()
            
            # 执行测试
            result = self.tester._test_model_loading('onnx', model_path)
            
            # 验证结果
            self.assertTrue(result.success)
            self.assertEqual(result.test_name, 'onnx_model_loading')
            self.assertIsNone(result.error_message)
    
    def test_test_model_loading_onnx_failure(self):
        """测试ONNX模型加载失败"""
        model_path = self.create_mock_onnx_model_path()
        
        with patch('src.validation_tester.ONNX_AVAILABLE', True), \
             patch('src.validation_tester.ort.InferenceSession') as mock_session:
            
            mock_session.side_effect = Exception("ONNX加载失败")
            
            # 执行测试
            result = self.tester._test_model_loading('onnx', model_path)
            
            # 验证结果
            self.assertFalse(result.success)
            self.assertEqual(result.test_name, 'onnx_model_loading')
            self.assertIsNotNone(result.error_message)
    
    def test_test_basic_inference_success(self):
        """测试基本推理成功"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch.object(self.tester, '_pytorch_inference') as mock_inference:
            mock_inference.return_value = "This is a test output"
            
            # 执行测试
            result = self.tester._test_basic_inference('pytorch', model_path)
            
            # 验证结果
            self.assertTrue(result.success)
            self.assertEqual(result.test_name, 'pytorch_basic_inference')
            self.assertIsNone(result.error_message)
            self.assertIn('output_length', result.details)
    
    def test_test_basic_inference_empty_output(self):
        """测试基本推理空输出"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch.object(self.tester, '_pytorch_inference') as mock_inference:
            mock_inference.return_value = ""
            
            # 执行测试
            result = self.tester._test_basic_inference('pytorch', model_path)
            
            # 验证结果
            self.assertFalse(result.success)
            self.assertEqual(result.test_name, 'pytorch_basic_inference')
            self.assertIn("推理输出为空", result.error_message)
    
    def test_test_batch_inference_success(self):
        """测试批量推理成功"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch.object(self.tester, '_pytorch_inference') as mock_inference:
            mock_inference.return_value = "Test output"
            
            # 执行测试
            result = self.tester._test_batch_inference('pytorch', model_path)
            
            # 验证结果
            self.assertTrue(result.success)
            self.assertEqual(result.test_name, 'pytorch_batch_inference')
            self.assertIsNone(result.error_message)
            self.assertIn('batch_size', result.details)
    
    def test_test_input_validation(self):
        """测试输入验证"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch.object(self.tester, '_pytorch_inference') as mock_inference:
            # 模拟不同输入的不同结果
            def mock_inference_func(path, text):
                if text == "":
                    return None  # 空输入失败
                return "Output for: " + text
            
            mock_inference.side_effect = mock_inference_func
            
            # 执行测试
            result = self.tester._test_input_validation('pytorch', model_path)
            
            # 验证结果
            self.assertTrue(result.success)  # 至少有一个成功
            self.assertEqual(result.test_name, 'pytorch_input_validation')
            self.assertIn('test_cases', result.details)
            self.assertIn('successful_cases', result.details)
    
    def test_compare_model_outputs_success(self):
        """测试模型输出对比成功"""
        model_paths = {
            'pytorch': self.create_mock_pytorch_model_path(),
            'onnx': self.create_mock_onnx_model_path()
        }
        formats = ['pytorch', 'onnx']
        
        with patch.object(self.tester, '_pytorch_inference') as mock_pytorch, \
             patch.object(self.tester, '_onnx_inference') as mock_onnx:
            
            mock_pytorch.return_value = "This is a test output"
            mock_onnx.return_value = "This is a test output"
            
            # 执行测试
            result = self.tester._compare_model_outputs("test input", model_paths, formats, 0)
            
            # 验证结果
            self.assertTrue(result['success'])
            self.assertIn('similarities', result)
            self.assertIn('avg_similarity', result)
    
    def test_compare_model_outputs_insufficient_outputs(self):
        """测试模型输出对比输出不足"""
        model_paths = {
            'pytorch': self.create_mock_pytorch_model_path(),
            'onnx': self.create_mock_onnx_model_path()
        }
        formats = ['pytorch', 'onnx']
        
        with patch.object(self.tester, '_pytorch_inference') as mock_pytorch, \
             patch.object(self.tester, '_onnx_inference') as mock_onnx:
            
            mock_pytorch.return_value = "Test output"
            mock_onnx.return_value = None  # ONNX推理失败
            
            # 执行测试
            result = self.tester._compare_model_outputs("test input", model_paths, formats, 0)
            
            # 验证结果
            self.assertFalse(result['success'])
            self.assertIn('可比较的输出少于2个', result['error_message'])
    
    def test_calculate_text_similarity(self):
        """测试文本相似性计算"""
        # 测试相同文本
        similarity = self.tester._calculate_text_similarity("hello world", "hello world")
        self.assertEqual(similarity, 1.0)
        
        # 测试部分相同文本
        similarity = self.tester._calculate_text_similarity("hello world", "hello there")
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
        
        # 测试完全不同文本
        similarity = self.tester._calculate_text_similarity("hello", "goodbye")
        self.assertEqual(similarity, 0.0)
        
        # 测试空文本
        similarity = self.tester._calculate_text_similarity("", "hello")
        self.assertEqual(similarity, 0.0)
        
        similarity = self.tester._calculate_text_similarity("", "")
        self.assertEqual(similarity, 1.0)
    
    def test_pytorch_inference_mock(self):
        """测试PyTorch推理（模拟）"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch('src.validation_tester.TRANSFORMERS_AVAILABLE', True), \
             patch('src.validation_tester.AutoModelForCausalLM') as mock_model_class, \
             patch('src.validation_tester.AutoTokenizer') as mock_tokenizer_class:
            
            # 设置mock
            mock_model = Mock()
            mock_tokenizer = Mock()
            
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_tokenizer.decode.return_value = "Generated text"
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            
            # 执行测试
            result = self.tester._pytorch_inference(model_path, "test input")
            
            # 验证结果
            self.assertEqual(result, "Generated text")
    
    def test_pytorch_inference_not_available(self):
        """测试PyTorch推理库不可用"""
        model_path = self.create_mock_pytorch_model_path()
        
        with patch('src.validation_tester.TRANSFORMERS_AVAILABLE', False):
            # 执行测试
            result = self.tester._pytorch_inference(model_path, "test input")
            
            # 验证结果
            self.assertIsNone(result)
    
    def test_onnx_inference_mock(self):
        """测试ONNX推理（模拟）"""
        model_path = self.create_mock_onnx_model_path()
        
        with patch('src.validation_tester.ONNX_AVAILABLE', True), \
             patch('src.validation_tester.ort.InferenceSession') as mock_session_class, \
             patch('src.validation_tester.AutoTokenizer') as mock_tokenizer_class:
            
            # 设置mock
            mock_session = Mock()
            mock_tokenizer = Mock()
            
            mock_session_class.return_value = mock_session
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_tokenizer.return_value = {'input_ids': [[1, 2, 3]]}
            mock_tokenizer.decode.return_value = "ONNX generated text"
            
            # 模拟ONNX会话
            mock_session.get_inputs.return_value = [Mock(name='input_ids')]
            mock_session.run.return_value = [[[0.1, 0.9, 0.2]]]  # logits
            
            # 执行测试
            result = self.tester._onnx_inference(model_path, "test input")
            
            # 验证结果
            self.assertEqual(result, "ONNX generated text")
    
    def test_onnx_inference_not_available(self):
        """测试ONNX推理库不可用"""
        model_path = self.create_mock_onnx_model_path()
        
        with patch('src.validation_tester.ONNX_AVAILABLE', False):
            # 执行测试
            result = self.tester._onnx_inference(model_path, "test input")
            
            # 验证结果
            self.assertIsNone(result)
    
    def test_generate_recommendations(self):
        """测试建议生成"""
        report = ValidationReport(
            report_id="test_report",
            timestamp=datetime.now(),
            config=self.config
        )
        
        # 添加一些测试数据
        from src.export_models import ValidationResult
        report.basic_tests = [
            ValidationResult(test_name="test1", success=True),
            ValidationResult(test_name="test2", success=False, error_message="Error: Test failed")
        ]
        
        report.performance_benchmarks = [
            PerformanceBenchmark(
                model_format="pytorch",
                model_path="test_path",
                success=True,
                avg_inference_time_ms=2000,  # 慢速推理
                avg_memory_mb=2000  # 高内存使用
            )
        ]
        
        # 执行测试
        self.tester._generate_recommendations(report)
        
        # 验证结果
        self.assertGreater(len(report.issues), 0)
        self.assertGreater(len(report.recommendations), 0)
        
        # 检查特定建议
        issues_text = ' '.join(report.issues)
        recommendations_text = ' '.join(report.recommendations)
        
        self.assertIn("测试失败", issues_text)
        self.assertIn("推理速度较慢", issues_text)
        self.assertIn("内存使用较高", issues_text)
        self.assertIn("量化", recommendations_text)


class TestValidationReport(unittest.TestCase):
    """验证报告测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.config = ExportConfiguration(
            checkpoint_path="test",
            base_model_name="test",
            output_directory="/tmp"
        )
        
        self.report = ValidationReport(
            report_id="test_report",
            timestamp=datetime.now(),
            config=self.config
        )
    
    def test_calculate_summary_empty(self):
        """测试空报告的摘要计算"""
        self.report.calculate_summary()
        
        self.assertEqual(self.report.total_tests, 0)
        self.assertEqual(self.report.passed_tests, 0)
        self.assertEqual(self.report.failed_tests, 0)
        self.assertEqual(self.report.success_rate, 0.0)
    
    def test_calculate_summary_with_tests(self):
        """测试有测试结果的摘要计算"""
        from src.export_models import ValidationResult
        
        # 添加基本测试
        self.report.basic_tests = [
            ValidationResult(test_name="test1", success=True),
            ValidationResult(test_name="test2", success=False)
        ]
        
        # 添加对比测试
        self.report.comparison_tests = [
            {'success': True},
            {'success': False}
        ]
        
        # 添加性能基准
        self.report.performance_benchmarks = [
            PerformanceBenchmark(
                model_format="pytorch",
                model_path="test",
                success=True,
                avg_inference_time_ms=100
            ),
            PerformanceBenchmark(
                model_format="onnx",
                model_path="test",
                success=True,
                avg_inference_time_ms=80
            )
        ]
        
        self.report.calculate_summary()
        
        self.assertEqual(self.report.total_tests, 4)
        self.assertEqual(self.report.passed_tests, 2)
        self.assertEqual(self.report.failed_tests, 2)
        self.assertEqual(self.report.success_rate, 0.5)
        
        # 检查最佳性能
        self.assertIsNotNone(self.report.best_performance)
        self.assertEqual(self.report.best_performance.model_format, "onnx")


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试类"""
    
    def test_initialization(self):
        """测试性能基准初始化"""
        benchmark = PerformanceBenchmark(
            model_format="pytorch",
            model_path="/test/path"
        )
        
        self.assertEqual(benchmark.model_format, "pytorch")
        self.assertEqual(benchmark.model_path, "/test/path")
        self.assertTrue(benchmark.success)
        self.assertEqual(benchmark.avg_inference_time_ms, 0.0)
        self.assertEqual(benchmark.min_inference_time_ms, float('inf'))
        self.assertEqual(benchmark.max_inference_time_ms, 0.0)
    
    def test_benchmark_with_error(self):
        """测试带错误的性能基准"""
        benchmark = PerformanceBenchmark(
            model_format="test",
            model_path="/test/path",
            success=False,
            error_message="Test error"
        )
        
        self.assertFalse(benchmark.success)
        self.assertEqual(benchmark.error_message, "Test error")


if __name__ == '__main__':
    unittest.main()