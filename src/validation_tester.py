"""
模型验证测试系统

本模块实现了完整的模型验证测试功能，包括基本功能测试、多格式模型输出对比、
推理性能基准测试和详细的验证报告生成。
"""

import os
import time
import json
import torch
import numpy as np
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import tempfile
import gc

from .export_models import ExportConfiguration, ValidationResult
from .export_exceptions import ValidationError, MemoryError
from .export_utils import format_size, ProgressTracker

# 尝试导入ONNX相关库
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# 尝试导入transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class PerformanceBenchmark:
    """性能基准测试结果"""
    
    model_format: str
    model_path: str
    
    # 推理性能
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = float('inf')
    max_inference_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    
    # 内存使用
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_efficiency: float = 0.0  # tokens per MB
    
    # 模型信息
    model_size_mb: float = 0.0
    parameter_count: int = 0
    
    # 测试配置
    test_samples: int = 0
    batch_size: int = 1
    sequence_length: int = 0
    
    # 错误信息
    error_message: Optional[str] = None
    success: bool = True


@dataclass
class ValidationReport:
    """验证报告"""
    
    # 基本信息
    report_id: str
    timestamp: datetime
    config: ExportConfiguration
    
    # 测试结果
    basic_tests: List[ValidationResult] = field(default_factory=list)
    comparison_tests: List[Dict[str, Any]] = field(default_factory=list)
    performance_benchmarks: List[PerformanceBenchmark] = field(default_factory=list)
    
    # 总体结果
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    success_rate: float = 0.0
    
    # 性能摘要
    best_performance: Optional[PerformanceBenchmark] = None
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    
    # 建议和问题
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_summary(self):
        """计算报告摘要"""
        self.total_tests = len(self.basic_tests) + len(self.comparison_tests)
        self.passed_tests = sum(1 for test in self.basic_tests if test.success)
        self.passed_tests += sum(1 for test in self.comparison_tests if test.get('success', False))
        self.failed_tests = self.total_tests - self.passed_tests
        self.success_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0
        
        # 找到最佳性能
        if self.performance_benchmarks:
            successful_benchmarks = [b for b in self.performance_benchmarks if b.success]
            if successful_benchmarks:
                self.best_performance = min(successful_benchmarks, 
                                          key=lambda x: x.avg_inference_time_ms)


class ValidationTester:
    """模型验证测试器"""
    
    def __init__(self, config: ExportConfiguration):
        """
        初始化验证测试器
        
        Args:
            config: 导出配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 测试配置
        self.test_samples = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the process of photosynthesis.",
            "What is the meaning of life?",
            "How to cook a perfect pasta?",
            "Explain the theory of relativity."
        ]
        
        self.performance_test_samples = self.test_samples[:5]  # 使用前5个样本进行性能测试
        
        self.logger.info("验证测试器初始化完成")
    
    def run_comprehensive_validation(self, model_paths: Dict[str, str]) -> ValidationReport:
        """
        运行综合验证测试
        
        Args:
            model_paths: 模型路径字典，格式为 {'format': 'path'}
            
        Returns:
            ValidationReport: 验证报告
        """
        report = ValidationReport(
            report_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            config=self.config
        )
        
        try:
            self.logger.info("开始综合验证测试")
            
            # 1. 基本功能测试
            self.logger.info("执行基本功能测试...")
            for format_name, model_path in model_paths.items():
                if os.path.exists(model_path):
                    basic_results = self.run_basic_functionality_tests(format_name, model_path)
                    report.basic_tests.extend(basic_results)
            
            # 2. 多格式输出对比测试
            if len(model_paths) > 1:
                self.logger.info("执行多格式输出对比测试...")
                comparison_results = self.run_output_comparison_tests(model_paths)
                report.comparison_tests.extend(comparison_results)
            
            # 3. 性能基准测试
            self.logger.info("执行性能基准测试...")
            for format_name, model_path in model_paths.items():
                if os.path.exists(model_path):
                    benchmark = self.run_performance_benchmark(format_name, model_path)
                    if benchmark:
                        report.performance_benchmarks.append(benchmark)
            
            # 4. 计算报告摘要
            report.calculate_summary()
            
            # 5. 生成建议和问题
            self._generate_recommendations(report)
            
            self.logger.info(f"综合验证测试完成，成功率: {report.success_rate:.2%}")
            
        except Exception as e:
            self.logger.error(f"综合验证测试失败: {str(e)}")
            report.issues.append(f"验证测试过程中发生错误: {str(e)}")
        
        return report
    
    def run_basic_functionality_tests(self, format_name: str, model_path: str) -> List[ValidationResult]:
        """
        运行基本功能测试
        
        Args:
            format_name: 模型格式名称
            model_path: 模型路径
            
        Returns:
            List[ValidationResult]: 测试结果列表
        """
        results = []
        
        try:
            self.logger.info(f"开始{format_name}格式基本功能测试")
            
            # 测试1: 模型加载测试
            load_result = self._test_model_loading(format_name, model_path)
            results.append(load_result)
            
            if load_result.success:
                # 测试2: 基本推理测试
                inference_result = self._test_basic_inference(format_name, model_path)
                results.append(inference_result)
                
                # 测试3: 批量推理测试
                batch_result = self._test_batch_inference(format_name, model_path)
                results.append(batch_result)
                
                # 测试4: 输入验证测试
                input_validation_result = self._test_input_validation(format_name, model_path)
                results.append(input_validation_result)
            
        except Exception as e:
            error_result = ValidationResult(
                test_name=f"{format_name}_basic_functionality",
                success=False,
                error_message=f"基本功能测试失败: {str(e)}"
            )
            results.append(error_result)
        
        return results
    
    def run_output_comparison_tests(self, model_paths: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        运行多格式模型输出对比测试
        
        Args:
            model_paths: 模型路径字典
            
        Returns:
            List[Dict[str, Any]]: 对比测试结果
        """
        results = []
        
        try:
            self.logger.info("开始多格式输出对比测试")
            
            # 获取所有可用的模型格式
            available_formats = [fmt for fmt, path in model_paths.items() if os.path.exists(path)]
            
            if len(available_formats) < 2:
                self.logger.warning("可用模型格式少于2个，跳过对比测试")
                return results
            
            # 对每个测试样本进行对比
            for i, test_input in enumerate(self.test_samples[:3]):  # 使用前3个样本
                comparison_result = self._compare_model_outputs(
                    test_input, model_paths, available_formats, i
                )
                results.append(comparison_result)
            
        except Exception as e:
            self.logger.error(f"输出对比测试失败: {str(e)}")
            results.append({
                'test_name': 'output_comparison',
                'success': False,
                'error_message': str(e)
            })
        
        return results
    
    def run_performance_benchmark(self, format_name: str, model_path: str) -> Optional[PerformanceBenchmark]:
        """
        运行性能基准测试
        
        Args:
            format_name: 模型格式名称
            model_path: 模型路径
            
        Returns:
            Optional[PerformanceBenchmark]: 性能基准测试结果
        """
        try:
            self.logger.info(f"开始{format_name}格式性能基准测试")
            
            benchmark = PerformanceBenchmark(
                model_format=format_name,
                model_path=model_path
            )
            
            # 获取模型大小
            if os.path.isfile(model_path):
                benchmark.model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            elif os.path.isdir(model_path):
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
                benchmark.model_size_mb = total_size / (1024 * 1024)
            
            # 执行性能测试
            if format_name.lower() == 'pytorch':
                self._benchmark_pytorch_model(model_path, benchmark)
            elif format_name.lower() == 'onnx' and ONNX_AVAILABLE:
                self._benchmark_onnx_model(model_path, benchmark)
            else:
                benchmark.success = False
                benchmark.error_message = f"不支持的模型格式: {format_name}"
            
            return benchmark
            
        except Exception as e:
            self.logger.error(f"{format_name}性能基准测试失败: {str(e)}")
            return PerformanceBenchmark(
                model_format=format_name,
                model_path=model_path,
                success=False,
                error_message=str(e)
            )
    
    def generate_validation_report(self, report: ValidationReport, output_path: str):
        """
        生成详细的验证报告
        
        Args:
            report: 验证报告数据
            output_path: 输出路径
        """
        try:
            self.logger.info(f"生成验证报告: {output_path}")
            
            # 创建报告目录
            report_dir = Path(output_path)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成JSON报告
            json_report_path = report_dir / "validation_report.json"
            self._generate_json_report(report, json_report_path)
            
            # 生成HTML报告
            html_report_path = report_dir / "validation_report.html"
            self._generate_html_report(report, html_report_path)
            
            # 生成性能对比图表（如果有多个格式）
            if len(report.performance_benchmarks) > 1:
                chart_path = report_dir / "performance_comparison.json"
                self._generate_performance_chart_data(report, chart_path)
            
            self.logger.info("验证报告生成完成")
            
        except Exception as e:
            self.logger.error(f"生成验证报告失败: {str(e)}")
            raise ValidationError(f"生成验证报告失败: {str(e)}")
    
    def _test_model_loading(self, format_name: str, model_path: str) -> ValidationResult:
        """测试模型加载"""
        start_time = time.time()
        
        try:
            if format_name.lower() == 'pytorch':
                if TRANSFORMERS_AVAILABLE:
                    # 尝试加载PyTorch模型
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                else:
                    raise ImportError("transformers库不可用")
                    
            elif format_name.lower() == 'onnx':
                if ONNX_AVAILABLE:
                    # 尝试加载ONNX模型
                    onnx_file = model_path if model_path.endswith('.onnx') else os.path.join(model_path, 'model.onnx')
                    session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
                    del session
                else:
                    raise ImportError("onnxruntime库不可用")
            else:
                raise ValueError(f"不支持的模型格式: {format_name}")
            
            duration = time.time() - start_time
            
            return ValidationResult(
                test_name=f"{format_name}_model_loading",
                success=True,
                score=1.0,
                duration_seconds=duration,
                details={'load_time_seconds': duration}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=f"{format_name}_model_loading",
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def _test_basic_inference(self, format_name: str, model_path: str) -> ValidationResult:
        """测试基本推理功能"""
        start_time = time.time()
        
        try:
            test_input = "Hello, how are you?"
            
            if format_name.lower() == 'pytorch':
                result = self._pytorch_inference(model_path, test_input)
            elif format_name.lower() == 'onnx':
                result = self._onnx_inference(model_path, test_input)
            else:
                raise ValueError(f"不支持的模型格式: {format_name}")
            
            duration = time.time() - start_time
            
            # 验证输出
            if result and len(result) > 0:
                return ValidationResult(
                    test_name=f"{format_name}_basic_inference",
                    success=True,
                    score=1.0,
                    duration_seconds=duration,
                    details={
                        'input': test_input,
                        'output_length': len(result),
                        'inference_time_seconds': duration
                    }
                )
            else:
                return ValidationResult(
                    test_name=f"{format_name}_basic_inference",
                    success=False,
                    error_message="推理输出为空",
                    duration_seconds=duration
                )
                
        except Exception as e:
            return ValidationResult(
                test_name=f"{format_name}_basic_inference",
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def _test_batch_inference(self, format_name: str, model_path: str) -> ValidationResult:
        """测试批量推理功能"""
        start_time = time.time()
        
        try:
            test_inputs = self.test_samples[:3]  # 使用前3个样本
            
            if format_name.lower() == 'pytorch':
                results = [self._pytorch_inference(model_path, inp) for inp in test_inputs]
            elif format_name.lower() == 'onnx':
                results = [self._onnx_inference(model_path, inp) for inp in test_inputs]
            else:
                raise ValueError(f"不支持的模型格式: {format_name}")
            
            duration = time.time() - start_time
            
            # 验证所有输出
            successful_results = [r for r in results if r and len(r) > 0]
            
            if len(successful_results) == len(test_inputs):
                return ValidationResult(
                    test_name=f"{format_name}_batch_inference",
                    success=True,
                    score=1.0,
                    duration_seconds=duration,
                    details={
                        'batch_size': len(test_inputs),
                        'successful_inferences': len(successful_results),
                        'avg_inference_time_seconds': duration / len(test_inputs)
                    }
                )
            else:
                return ValidationResult(
                    test_name=f"{format_name}_batch_inference",
                    success=False,
                    error_message=f"批量推理部分失败: {len(successful_results)}/{len(test_inputs)}",
                    duration_seconds=duration
                )
                
        except Exception as e:
            return ValidationResult(
                test_name=f"{format_name}_batch_inference",
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def _test_input_validation(self, format_name: str, model_path: str) -> ValidationResult:
        """测试输入验证功能"""
        start_time = time.time()
        
        try:
            # 测试不同类型的输入
            test_cases = [
                ("正常输入", "This is a normal input."),
                ("空输入", ""),
                ("长输入", "This is a very long input. " * 50),
                ("特殊字符", "Hello! @#$%^&*()_+ 你好 🌟"),
            ]
            
            results = []
            for case_name, test_input in test_cases:
                try:
                    if format_name.lower() == 'pytorch':
                        result = self._pytorch_inference(model_path, test_input)
                    elif format_name.lower() == 'onnx':
                        result = self._onnx_inference(model_path, test_input)
                    else:
                        raise ValueError(f"不支持的模型格式: {format_name}")
                    
                    results.append({
                        'case': case_name,
                        'input': test_input[:50] + "..." if len(test_input) > 50 else test_input,
                        'success': result is not None,
                        'output_length': len(result) if result else 0
                    })
                    
                except Exception as e:
                    results.append({
                        'case': case_name,
                        'input': test_input[:50] + "..." if len(test_input) > 50 else test_input,
                        'success': False,
                        'error': str(e)
                    })
            
            duration = time.time() - start_time
            successful_cases = sum(1 for r in results if r['success'])
            
            return ValidationResult(
                test_name=f"{format_name}_input_validation",
                success=successful_cases > 0,
                score=successful_cases / len(test_cases),
                duration_seconds=duration,
                details={
                    'test_cases': results,
                    'successful_cases': successful_cases,
                    'total_cases': len(test_cases)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=f"{format_name}_input_validation",
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def _compare_model_outputs(self, test_input: str, model_paths: Dict[str, str], 
                              formats: List[str], test_index: int) -> Dict[str, Any]:
        """比较不同格式模型的输出"""
        try:
            outputs = {}
            
            # 获取每个格式的输出
            for format_name in formats:
                model_path = model_paths[format_name]
                try:
                    if format_name.lower() == 'pytorch':
                        output = self._pytorch_inference(model_path, test_input)
                    elif format_name.lower() == 'onnx':
                        output = self._onnx_inference(model_path, test_input)
                    else:
                        continue
                    
                    outputs[format_name] = output
                    
                except Exception as e:
                    self.logger.warning(f"{format_name}格式推理失败: {str(e)}")
                    outputs[format_name] = None
            
            # 分析输出一致性
            valid_outputs = {k: v for k, v in outputs.items() if v is not None}
            
            if len(valid_outputs) < 2:
                return {
                    'test_name': f'output_comparison_{test_index}',
                    'test_input': test_input,
                    'success': False,
                    'error_message': '可比较的输出少于2个',
                    'outputs': outputs
                }
            
            # 计算输出相似性（简单的长度和内容比较）
            output_lengths = {k: len(v) for k, v in valid_outputs.items()}
            output_similarities = {}
            
            format_list = list(valid_outputs.keys())
            for i in range(len(format_list)):
                for j in range(i + 1, len(format_list)):
                    fmt1, fmt2 = format_list[i], format_list[j]
                    similarity = self._calculate_text_similarity(
                        valid_outputs[fmt1], valid_outputs[fmt2]
                    )
                    output_similarities[f"{fmt1}_vs_{fmt2}"] = similarity
            
            avg_similarity = sum(output_similarities.values()) / len(output_similarities)
            
            return {
                'test_name': f'output_comparison_{test_index}',
                'test_input': test_input,
                'success': avg_similarity > 0.5,  # 相似度阈值
                'outputs': {k: v[:100] + "..." if v and len(v) > 100 else v 
                           for k, v in outputs.items()},
                'output_lengths': output_lengths,
                'similarities': output_similarities,
                'avg_similarity': avg_similarity,
                'details': {
                    'valid_formats': list(valid_outputs.keys()),
                    'comparison_count': len(output_similarities)
                }
            }
            
        except Exception as e:
            return {
                'test_name': f'output_comparison_{test_index}',
                'test_input': test_input,
                'success': False,
                'error_message': str(e)
            }
    
    def _benchmark_pytorch_model(self, model_path: str, benchmark: PerformanceBenchmark):
        """PyTorch模型性能基准测试"""
        if not TRANSFORMERS_AVAILABLE:
            benchmark.success = False
            benchmark.error_message = "transformers库不可用"
            return
        
        try:
            # 加载模型和tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                # 如果没有tokenizer，使用默认的
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            
            # 获取参数数量
            benchmark.parameter_count = sum(p.numel() for p in model.parameters())
            
            # 性能测试
            inference_times = []
            memory_usage = []
            
            process = psutil.Process()
            
            with torch.no_grad():
                for test_input in self.performance_test_samples:
                    # 记录内存使用
                    memory_before = process.memory_info().rss / (1024 * 1024)
                    
                    # 推理计时
                    start_time = time.time()
                    
                    inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 20,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    end_time = time.time()
                    
                    # 记录内存使用
                    memory_after = process.memory_info().rss / (1024 * 1024)
                    
                    inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                    inference_times.append(inference_time)
                    memory_usage.append(memory_after)
                    
                    # 计算序列长度
                    if benchmark.sequence_length == 0:
                        benchmark.sequence_length = outputs.shape[1]
            
            # 计算统计信息
            benchmark.avg_inference_time_ms = np.mean(inference_times)
            benchmark.min_inference_time_ms = np.min(inference_times)
            benchmark.max_inference_time_ms = np.max(inference_times)
            benchmark.peak_memory_mb = np.max(memory_usage)
            benchmark.avg_memory_mb = np.mean(memory_usage)
            benchmark.test_samples = len(self.performance_test_samples)
            
            # 计算吞吐量
            if benchmark.avg_inference_time_ms > 0:
                benchmark.throughput_tokens_per_sec = (benchmark.sequence_length * 1000) / benchmark.avg_inference_time_ms
                benchmark.memory_efficiency = benchmark.throughput_tokens_per_sec / benchmark.avg_memory_mb if benchmark.avg_memory_mb > 0 else 0
            
            benchmark.success = True
            
            # 清理内存
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            benchmark.success = False
            benchmark.error_message = str(e)
    
    def _benchmark_onnx_model(self, model_path: str, benchmark: PerformanceBenchmark):
        """ONNX模型性能基准测试"""
        try:
            # 确定ONNX文件路径
            onnx_file = model_path if model_path.endswith('.onnx') else os.path.join(model_path, 'model.onnx')
            
            if not os.path.exists(onnx_file):
                benchmark.success = False
                benchmark.error_message = f"ONNX文件不存在: {onnx_file}"
                return
            
            # 创建推理会话
            session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
            
            # 尝试加载tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(onnx_file))
            except:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            
            # 获取输入输出信息
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            
            # 性能测试
            inference_times = []
            memory_usage = []
            
            process = psutil.Process()
            
            for test_input in self.performance_test_samples:
                # 记录内存使用
                memory_before = process.memory_info().rss / (1024 * 1024)
                
                # 准备输入
                inputs = tokenizer(test_input, return_tensors="np", padding=True, truncation=True)
                onnx_inputs = {name: inputs[name] for name in input_names if name in inputs}
                
                # 推理计时
                start_time = time.time()
                outputs = session.run(output_names, onnx_inputs)
                end_time = time.time()
                
                # 记录内存使用
                memory_after = process.memory_info().rss / (1024 * 1024)
                
                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)
                memory_usage.append(memory_after)
                
                # 计算序列长度
                if benchmark.sequence_length == 0 and outputs:
                    benchmark.sequence_length = outputs[0].shape[1] if len(outputs[0].shape) > 1 else 1
            
            # 计算统计信息
            benchmark.avg_inference_time_ms = np.mean(inference_times)
            benchmark.min_inference_time_ms = np.min(inference_times)
            benchmark.max_inference_time_ms = np.max(inference_times)
            benchmark.peak_memory_mb = np.max(memory_usage)
            benchmark.avg_memory_mb = np.mean(memory_usage)
            benchmark.test_samples = len(self.performance_test_samples)
            
            # 计算吞吐量
            if benchmark.avg_inference_time_ms > 0:
                benchmark.throughput_tokens_per_sec = (benchmark.sequence_length * 1000) / benchmark.avg_inference_time_ms
                benchmark.memory_efficiency = benchmark.throughput_tokens_per_sec / benchmark.avg_memory_mb if benchmark.avg_memory_mb > 0 else 0
            
            benchmark.success = True
            
            # 清理资源
            del session
            gc.collect()
            
        except Exception as e:
            benchmark.success = False
            benchmark.error_message = str(e)
    
    def _pytorch_inference(self, model_path: str, text: str) -> Optional[str]:
        """PyTorch模型推理"""
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 清理内存
                del model, tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                return result
                
        except Exception as e:
            self.logger.warning(f"PyTorch推理失败: {str(e)}")
            return None
    
    def _onnx_inference(self, model_path: str, text: str) -> Optional[str]:
        """ONNX模型推理"""
        if not ONNX_AVAILABLE:
            return None
        
        try:
            # 确定ONNX文件路径
            onnx_file = model_path if model_path.endswith('.onnx') else os.path.join(model_path, 'model.onnx')
            
            if not os.path.exists(onnx_file):
                return None
            
            # 创建推理会话
            session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
            
            # 加载tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(onnx_file))
            except:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            
            # 准备输入
            inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
            input_names = [inp.name for inp in session.get_inputs()]
            onnx_inputs = {name: inputs[name] for name in input_names if name in inputs}
            
            # 执行推理
            outputs = session.run(None, onnx_inputs)
            
            # 解码输出（简化处理）
            if outputs and len(outputs) > 0:
                logits = outputs[0]
                
                # 处理不同形状的logits
                if isinstance(logits, (list, tuple)):
                    logits = np.array(logits)
                
                if len(logits.shape) > 2:
                    predicted_ids = np.argmax(logits[0], axis=-1)
                elif len(logits.shape) == 2:
                    predicted_ids = np.argmax(logits, axis=-1)
                else:
                    predicted_ids = logits
                
                result = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                return result
            
            return None
            
        except Exception as e:
            self.logger.warning(f"ONNX推理失败: {str(e)}")
            return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性（简单实现）"""
        if not text1 and not text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        # 简单的字符级相似性计算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_recommendations(self, report: ValidationReport):
        """生成建议和问题"""
        # 分析测试结果
        failed_tests = [test for test in report.basic_tests if not test.success]
        
        if failed_tests:
            report.issues.append(f"有{len(failed_tests)}个基本功能测试失败")
            
            # 分析失败原因
            error_types = {}
            for test in failed_tests:
                if test.error_message:
                    error_type = test.error_message.split(':')[0]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                report.issues.append(f"{error_type}错误出现{count}次")
        
        # 性能分析
        if report.performance_benchmarks:
            successful_benchmarks = [b for b in report.performance_benchmarks if b.success]
            
            if successful_benchmarks:
                avg_inference_time = np.mean([b.avg_inference_time_ms for b in successful_benchmarks])
                if avg_inference_time > 1000:  # 超过1秒
                    report.issues.append("推理速度较慢，可能需要优化")
                    report.recommendations.append("考虑使用量化或模型压缩技术")
                
                avg_memory = np.mean([b.avg_memory_mb for b in successful_benchmarks])
                if avg_memory > 1000:  # 超过1GB
                    report.issues.append("内存使用较高")
                    report.recommendations.append("考虑使用低精度推理或分批处理")
            else:
                report.issues.append("所有性能基准测试都失败了")
                report.recommendations.append("检查模型文件完整性和依赖库安装")
        
        # 对比测试分析
        if report.comparison_tests:
            failed_comparisons = [test for test in report.comparison_tests if not test.get('success', False)]
            if failed_comparisons:
                report.issues.append(f"有{len(failed_comparisons)}个输出对比测试失败")
                report.recommendations.append("检查不同格式模型的一致性")
        
        # 通用建议
        if report.success_rate < 0.8:
            report.recommendations.append("测试成功率较低，建议检查模型导出过程")
        
        if not report.issues:
            report.recommendations.append("所有测试通过，模型导出质量良好")
    
    def _generate_json_report(self, report: ValidationReport, output_path: Path):
        """生成JSON格式报告"""
        report_data = {
            'report_id': report.report_id,
            'timestamp': report.timestamp.isoformat(),
            'summary': {
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'failed_tests': report.failed_tests,
                'success_rate': report.success_rate
            },
            'basic_tests': [
                {
                    'test_name': test.test_name,
                    'success': test.success,
                    'score': test.score,
                    'duration_seconds': test.duration_seconds,
                    'error_message': test.error_message,
                    'details': test.details
                }
                for test in report.basic_tests
            ],
            'comparison_tests': report.comparison_tests,
            'performance_benchmarks': [
                {
                    'model_format': b.model_format,
                    'model_path': b.model_path,
                    'success': b.success,
                    'avg_inference_time_ms': b.avg_inference_time_ms,
                    'throughput_tokens_per_sec': b.throughput_tokens_per_sec,
                    'peak_memory_mb': b.peak_memory_mb,
                    'model_size_mb': b.model_size_mb,
                    'parameter_count': b.parameter_count,
                    'error_message': b.error_message
                }
                for b in report.performance_benchmarks
            ],
            'issues': report.issues,
            'recommendations': report.recommendations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_html_report(self, report: ValidationReport, output_path: Path):
        """生成HTML格式报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型验证报告 - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; }}
        .metric h3 {{ margin: 0; color: #333; }}
        .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .section {{ margin: 20px 0; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .test-success {{ border-left-color: #28a745; }}
        .test-failure {{ border-left-color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .issues {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
        .recommendations {{ background-color: #d1ecf1; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>模型验证报告</h1>
        <p><strong>报告ID:</strong> {report.report_id}</p>
        <p><strong>生成时间:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>总测试数</h3>
            <p>{report.total_tests}</p>
        </div>
        <div class="metric">
            <h3>通过测试</h3>
            <p class="success">{report.passed_tests}</p>
        </div>
        <div class="metric">
            <h3>失败测试</h3>
            <p class="failure">{report.failed_tests}</p>
        </div>
        <div class="metric">
            <h3>成功率</h3>
            <p class="{'success' if report.success_rate >= 0.8 else 'failure'}">{report.success_rate:.1%}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>基本功能测试</h2>
        {''.join([f'''
        <div class="test-result {'test-success' if test.success else 'test-failure'}">
            <h4>{test.test_name}</h4>
            <p><strong>状态:</strong> {'通过' if test.success else '失败'}</p>
            <p><strong>耗时:</strong> {test.duration_seconds:.2f}秒</p>
            {f'<p><strong>错误:</strong> {test.error_message}</p>' if test.error_message else ''}
        </div>
        ''' for test in report.basic_tests])}
    </div>
    
    {f'''
    <div class="section">
        <h2>性能基准测试</h2>
        <table>
            <tr>
                <th>模型格式</th>
                <th>平均推理时间(ms)</th>
                <th>吞吐量(tokens/s)</th>
                <th>峰值内存(MB)</th>
                <th>模型大小(MB)</th>
                <th>状态</th>
            </tr>
            {''.join([f'''
            <tr>
                <td>{b.model_format}</td>
                <td>{b.avg_inference_time_ms:.2f}</td>
                <td>{b.throughput_tokens_per_sec:.2f}</td>
                <td>{b.peak_memory_mb:.2f}</td>
                <td>{b.model_size_mb:.2f}</td>
                <td class="{'success' if b.success else 'failure'}">{'成功' if b.success else '失败'}</td>
            </tr>
            ''' for b in report.performance_benchmarks])}
        </table>
    </div>
    ''' if report.performance_benchmarks else ''}
    
    {f'''
    <div class="section issues">
        <h2>发现的问题</h2>
        <ul>
            {''.join([f'<li>{issue}</li>' for issue in report.issues])}
        </ul>
    </div>
    ''' if report.issues else ''}
    
    {f'''
    <div class="section recommendations">
        <h2>建议</h2>
        <ul>
            {''.join([f'<li>{rec}</li>' for rec in report.recommendations])}
        </ul>
    </div>
    ''' if report.recommendations else ''}
    
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_performance_chart_data(self, report: ValidationReport, output_path: Path):
        """生成性能对比图表数据"""
        chart_data = {
            'labels': [b.model_format for b in report.performance_benchmarks if b.success],
            'datasets': [
                {
                    'label': '平均推理时间(ms)',
                    'data': [b.avg_inference_time_ms for b in report.performance_benchmarks if b.success],
                    'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                    'borderColor': 'rgba(54, 162, 235, 1)'
                },
                {
                    'label': '峰值内存使用(MB)',
                    'data': [b.peak_memory_mb for b in report.performance_benchmarks if b.success],
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                    'borderColor': 'rgba(255, 99, 132, 1)'
                },
                {
                    'label': '吞吐量(tokens/s)',
                    'data': [b.throughput_tokens_per_sec for b in report.performance_benchmarks if b.success],
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'borderColor': 'rgba(75, 192, 192, 1)'
                }
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chart_data, f, indent=2)