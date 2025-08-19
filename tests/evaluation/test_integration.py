"""
评估系统集成测试

测试完整的数据处理流程，包括不同任务类型、错误恢复机制和性能测试。
"""

import unittest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from datasets import Dataset

from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.data_models import EvaluationConfig
from src.evaluation.data_preprocessor import DataPreprocessor
from src.evaluation.config_loader import ConfigLoader


class TestEvaluationIntegration(unittest.TestCase):
    """评估系统集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = EvaluationConfig(
            tasks=["text_generation", "question_answering"],
            batch_size=4,
            num_samples=20,
            enable_efficiency_metrics=False,  # 禁用以加快测试
            enable_quality_analysis=False
        )
        
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_text_generation_processing(self):
        """测试端到端文本生成处理流程"""
        # 创建测试数据集
        dataset = Dataset.from_dict({
            "text": [
                "Hello world",
                "Good morning",
                "How are you?",
                "Nice to meet you",
                "What's your name?",
                "",  # 空值测试
                "Short",
                "This is a longer sentence for testing purposes"
            ],
            "target": [
                "Bonjour monde",
                "Bonjour",
                "Comment allez-vous?",
                "Enchanté",
                "Comment vous appelez-vous?",
                "Vide",
                "Court",
                "Ceci est une phrase plus longue pour les tests"
            ]
        })
        
        datasets = {"text_generation": dataset}
        
        # 创建模拟的模型和分词器
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(device="cpu")]
        mock_model.config.is_encoder_decoder = False
        mock_model.generate.return_value = Mock()
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {
            "input_ids": Mock(to=Mock(return_value=Mock())),
            "attention_mask": Mock(to=Mock(return_value=Mock()))
        }
        mock_tokenizer.batch_decode.return_value = ["Generated response"] * 4
        
        # 创建评估引擎
        engine = EvaluationEngine(self.config)
        
        # 执行评估
        result = engine.evaluate_model(
            mock_model, 
            mock_tokenizer, 
            datasets, 
            "test_model"
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.model_name, "test_model")
        self.assertIn("text_generation", result.task_results)
        
        # 验证任务结果
        task_result = result.task_results["text_generation"]
        self.assertGreater(len(task_result.predictions), 0)
        self.assertGreater(len(task_result.references), 0)
        self.assertIsInstance(task_result.metrics, dict)
    
    def test_question_answering_processing(self):
        """测试问答任务处理"""
        dataset = Dataset.from_dict({
            "question": [
                "What is AI?",
                "How does machine learning work?",
                "What is deep learning?",
                ""  # 空问题测试
            ],
            "context": [
                "AI stands for Artificial Intelligence",
                "Machine learning uses algorithms to learn patterns",
                "Deep learning uses neural networks",
                "No context"
            ],
            "answer": [
                "Artificial Intelligence",
                "Using algorithms",
                "Neural networks",
                "No answer"
            ]
        })
        
        datasets = {"question_answering": dataset}
        
        # 创建数据预处理器
        preprocessor = DataPreprocessor(self.config)
        
        # 测试批次处理
        batch = dataset[:4]
        result = preprocessor.preprocess_batch(batch, "question_answering")
        
        # 验证结果
        self.assertGreater(len(result.inputs), 0)
        self.assertIn("问题:", result.inputs[0])
        self.assertIn("上下文:", result.inputs[0])
        
        # 验证处理统计
        self.assertIn("task_name", result.processing_stats)
        self.assertEqual(result.processing_stats["task_name"], "question_answering")
    
    def test_error_recovery_mechanisms(self):
        """测试错误恢复机制"""
        # 创建包含各种问题的数据集
        problematic_dataset = Dataset.from_dict({
            "input_ids": [1, 2, 3, 4],  # 错误的字段名
            "attention_mask": [1, 1, 1, 1],
            "labels": [0, 1, 0, 1]
        })
        
        datasets = {"text_generation": problematic_dataset}
        
        # 创建数据预处理器
        preprocessor = DataPreprocessor(self.config)
        
        # 测试诊断功能
        diagnosis = preprocessor.diagnose_batch(problematic_dataset[:4], "text_generation")
        
        # 验证诊断结果
        self.assertIn("batch_info", diagnosis)
        self.assertIn("recommendations", diagnosis)
        self.assertGreater(len(diagnosis["recommendations"]), 0)
        
        # 验证错误处理建议
        recommendations_str = " ".join(diagnosis["recommendations"])
        self.assertTrue(
            "字段" in recommendations_str or 
            "映射" in recommendations_str or
            "配置" in recommendations_str
        )
    
    def test_different_data_formats(self):
        """测试不同数据格式的处理"""
        test_cases = [
            # 标准格式
            {
                "name": "standard_format",
                "data": {
                    "text": ["Hello", "World"],
                    "target": ["Hi", "Earth"]
                }
            },
            # 自定义字段名
            {
                "name": "custom_fields",
                "data": {
                    "prompt": ["Hello", "World"],
                    "response": ["Hi", "Earth"]
                }
            },
            # 混合数据类型
            {
                "name": "mixed_types",
                "data": {
                    "content": ["Hello", 123, None, "World"],
                    "label": ["Hi", "Number", "Empty", "Earth"]
                }
            },
            # 长度不一致
            {
                "name": "inconsistent_lengths",
                "data": {
                    "text": ["Hello", "World", "Extra"],
                    "target": ["Hi", "Earth"]
                }
            }
        ]
        
        preprocessor = DataPreprocessor(self.config)
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case["name"]):
                dataset = Dataset.from_dict(test_case["data"])
                batch = dataset[:]
                
                # 执行预处理
                result = preprocessor.preprocess_batch(batch, "text_generation")
                
                # 验证结果结构
                self.assertIsInstance(result.inputs, list)
                self.assertIsInstance(result.valid_indices, list)
                self.assertIsInstance(result.processing_stats, dict)
                
                # 记录处理结果
                print(f"测试用例 {test_case['name']}: "
                      f"输入数={len(result.inputs)}, "
                      f"有效索引数={len(result.valid_indices)}, "
                      f"警告数={len(result.warnings)}")
    
    def test_performance_with_large_batches(self):
        """测试大批次数据的性能"""
        # 创建大数据集
        large_dataset = Dataset.from_dict({
            "text": [f"Sample text {i}" for i in range(1000)],
            "target": [f"Target {i}" for i in range(1000)]
        })
        
        preprocessor = DataPreprocessor(self.config)
        
        # 测试不同批次大小的性能
        batch_sizes = [10, 50, 100, 200]
        performance_results = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # 处理多个批次
            for i in range(0, min(500, len(large_dataset)), batch_size):
                batch = large_dataset[i:i+batch_size]
                result = preprocessor.preprocess_batch(batch, "text_generation")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            performance_results.append({
                "batch_size": batch_size,
                "processing_time": processing_time,
                "samples_per_second": 500 / processing_time
            })
            
            print(f"批次大小 {batch_size}: {processing_time:.2f}s, "
                  f"{500/processing_time:.1f} samples/s")
        
        # 验证性能合理性（不应该太慢）
        fastest_result = min(performance_results, key=lambda x: x["processing_time"])
        self.assertLess(fastest_result["processing_time"], 10.0)  # 不超过10秒
        self.assertGreater(fastest_result["samples_per_second"], 10)  # 至少10 samples/s
    
    def test_config_loading_and_validation(self):
        """测试配置加载和验证"""
        # 创建测试配置文件
        config_data = {
            "data_processing": {
                "field_mapping": {
                    "text_generation": {
                        "input_fields": ["text", "prompt"],
                        "target_fields": ["target", "response"]
                    }
                },
                "validation": {
                    "min_valid_samples_ratio": 0.2,
                    "enable_data_cleaning": True
                },
                "diagnostics": {
                    "enable_detailed_logging": True,
                    "log_batch_statistics": True
                }
            }
        }
        
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # 测试配置加载
        loader = ConfigLoader()
        evaluation_config = loader.create_evaluation_config(config_file=config_file)
        
        # 验证配置加载成功
        self.assertIsInstance(evaluation_config, EvaluationConfig)
        self.assertIn("field_mapping", evaluation_config.data_processing)
        
        # 测试配置验证
        validation_result = loader.validate_config_file(config_file)
        self.assertTrue(validation_result["is_valid"])
        self.assertEqual(len(validation_result["errors"]), 0)
    
    def test_diagnostic_reporting(self):
        """测试诊断报告生成"""
        dataset = Dataset.from_dict({
            "text": ["Hello", "World", "", "Test"],
            "target": ["Hi", "Earth", "Empty", "Testing"]
        })
        
        preprocessor = DataPreprocessor(self.config)
        
        # 处理多个批次
        for i in range(0, len(dataset), 2):
            batch = dataset[i:i+2]
            preprocessor.preprocess_batch(batch, "text_generation")
        
        # 生成诊断报告
        report = preprocessor.generate_processing_report()
        
        # 验证报告结构
        self.assertIn("session_info", report)
        self.assertIn("batch_statistics", report)
        self.assertIn("data_quality", report)
        self.assertIn("performance", report)
        self.assertIn("recommendations", report)
        
        # 验证会话信息
        session_info = report["session_info"]
        self.assertIn("session_id", session_info)
        self.assertIn("duration_seconds", session_info)
        
        # 验证批次统计
        batch_stats = report["batch_statistics"]
        if "summary" in batch_stats:
            summary = batch_stats["summary"]
            self.assertGreater(summary["total_batches"], 0)
            self.assertGreaterEqual(summary["total_samples"], 0)
    
    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建大数据集进行处理
        large_dataset = Dataset.from_dict({
            "text": [f"This is a longer sample text for memory testing {i}" * 10 for i in range(500)],
            "target": [f"Target response {i}" for i in range(500)]
        })
        
        preprocessor = DataPreprocessor(self.config)
        
        # 处理数据
        for i in range(0, len(large_dataset), 50):
            batch = large_dataset[i:i+50]
            result = preprocessor.preprocess_batch(batch, "text_generation")
        
        # 检查内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"内存使用: 初始={initial_memory:.1f}MB, "
              f"最终={final_memory:.1f}MB, "
              f"增加={memory_increase:.1f}MB")
        
        # 验证内存使用合理（不应该增加太多）
        self.assertLess(memory_increase, 500)  # 不超过500MB增长
    
    def test_concurrent_processing(self):
        """测试并发处理能力"""
        import threading
        import queue
        
        dataset = Dataset.from_dict({
            "text": [f"Concurrent test {i}" for i in range(100)],
            "target": [f"Response {i}" for i in range(100)]
        })
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def process_batch(batch_data, task_name, thread_id):
            try:
                preprocessor = DataPreprocessor(self.config)
                result = preprocessor.preprocess_batch(batch_data, task_name)
                results_queue.put((thread_id, result))
            except Exception as e:
                errors_queue.put((thread_id, str(e)))
        
        # 创建多个线程并发处理
        threads = []
        num_threads = 4
        batch_size = len(dataset) // num_threads
        
        for i in range(num_threads):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < num_threads - 1 else len(dataset)
            batch = dataset[start_idx:end_idx]
            
            thread = threading.Thread(
                target=process_batch,
                args=(batch, "text_generation", i)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=30)  # 30秒超时
        
        # 检查结果
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        # 验证并发处理结果
        self.assertEqual(len(results), num_threads)
        self.assertEqual(len(errors), 0)
        
        print(f"并发处理完成: {len(results)} 个线程成功, {len(errors)} 个错误")


if __name__ == '__main__':
    unittest.main(verbosity=2)