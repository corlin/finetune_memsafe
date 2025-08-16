#!/usr/bin/env python3
"""
性能基准测试脚本

用于测试不同配置下的训练性能，包括内存使用、训练速度和模型质量。
"""

import json
import time
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import psutil
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.results = []
        self.base_config = {
            "model_name": "Qwen/Qwen3-4B-Thinking-2507",
            "num_epochs": 5,  # 短期测试
            "data_dir": "examples/datasets",
            "enable_tensorboard": False,
            "enable_inference_test": True,
            "verify_environment": False,
            "auto_install_deps": False
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__
            })
        
        return info
    
    def run_benchmark(self, config_path: str, test_name: str) -> Dict[str, Any]:
        """运行单个基准测试"""
        logger.info(f"开始基准测试: {test_name}")
        
        # 记录开始时间和内存
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        
        try:
            # 运行训练
            cmd = [sys.executable, "main.py", "--config", config_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
            
            # 记录结束时间和内存
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)
            
            training_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            gpu_memory_peak = 0
            if torch.cuda.is_available():
                gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024**3)
            
            # 解析输出获取训练指标
            training_metrics = self.parse_training_output(result.stdout)
            
            benchmark_result = {
                "test_name": test_name,
                "config_path": config_path,
                "success": result.returncode == 0,
                "training_time_seconds": training_time,
                "memory_usage_gb": memory_usage,
                "gpu_memory_peak_gb": gpu_memory_peak,
                "training_metrics": training_metrics,
                "error_message": result.stderr if result.returncode != 0 else None
            }
            
            logger.info(f"基准测试完成: {test_name}, 耗时: {training_time:.1f}s")
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"基准测试超时: {test_name}")
            return {
                "test_name": test_name,
                "config_path": config_path,
                "success": False,
                "error_message": "测试超时"
            }
        except Exception as e:
            logger.error(f"基准测试失败: {test_name}, 错误: {e}")
            return {
                "test_name": test_name,
                "config_path": config_path,
                "success": False,
                "error_message": str(e)
            }
    
    def parse_training_output(self, output: str) -> Dict[str, Any]:
        """解析训练输出获取指标"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            # 解析训练损失
            if "train_loss" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "train_loss" and i + 1 < len(parts):
                            metrics["final_train_loss"] = float(parts[i + 1])
                            break
                except:
                    pass
            
            # 解析推理测试结果
            if "推理测试成功" in line:
                metrics["inference_test_passed"] = True
            elif "推理测试失败" in line:
                metrics["inference_test_passed"] = False
        
        return metrics
    
    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """运行所有基准测试"""
        configs = [
            ("examples/configs/quick_test.json", "快速测试"),
            ("examples/configs/low_memory_8gb.json", "低内存8GB"),
            ("examples/configs/standard_12gb.json", "标准12GB"),
            ("examples/configs/high_performance_16gb.json", "高性能16GB")
        ]
        
        results = []
        system_info = self.get_system_info()
        
        logger.info("系统信息:")
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")
        
        for config_path, test_name in configs:
            if Path(config_path).exists():
                result = self.run_benchmark(config_path, test_name)
                result["system_info"] = system_info
                results.append(result)
            else:
                logger.warning(f"配置文件不存在: {config_path}")
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """生成基准测试报告"""
        report = ["# 性能基准测试报告\n"]
        
        # 系统信息
        if results:
            system_info = results[0].get("system_info", {})
            report.append("## 系统信息\n")
            for key, value in system_info.items():
                report.append(f"- **{key}**: {value}")
            report.append("\n")
        
        # 测试结果汇总
        report.append("## 测试结果汇总\n")
        report.append("| 测试名称 | 状态 | 训练时间(s) | 内存使用(GB) | GPU峰值内存(GB) | 最终损失 | 推理测试 |")
        report.append("|---------|------|------------|-------------|----------------|----------|----------|")
        
        for result in results:
            status = "✅ 成功" if result["success"] else "❌ 失败"
            training_time = f"{result.get('training_time_seconds', 0):.1f}"
            memory_usage = f"{result.get('memory_usage_gb', 0):.2f}"
            gpu_memory = f"{result.get('gpu_memory_peak_gb', 0):.2f}"
            
            metrics = result.get("training_metrics", {})
            final_loss = f"{metrics.get('final_train_loss', 'N/A'):.4f}" if isinstance(metrics.get('final_train_loss'), float) else "N/A"
            inference_test = "✅" if metrics.get('inference_test_passed') else "❌" if 'inference_test_passed' in metrics else "N/A"
            
            report.append(f"| {result['test_name']} | {status} | {training_time} | {memory_usage} | {gpu_memory} | {final_loss} | {inference_test} |")
        
        report.append("\n")
        
        # 详细结果
        report.append("## 详细结果\n")
        for result in results:
            report.append(f"### {result['test_name']}\n")
            report.append(f"- **配置文件**: {result['config_path']}")
            report.append(f"- **状态**: {'成功' if result['success'] else '失败'}")
            
            if result["success"]:
                report.append(f"- **训练时间**: {result.get('training_time_seconds', 0):.1f} 秒")
                report.append(f"- **内存使用**: {result.get('memory_usage_gb', 0):.2f} GB")
                report.append(f"- **GPU峰值内存**: {result.get('gpu_memory_peak_gb', 0):.2f} GB")
                
                metrics = result.get("training_metrics", {})
                if metrics:
                    report.append("- **训练指标**:")
                    for key, value in metrics.items():
                        report.append(f"  - {key}: {value}")
            else:
                report.append(f"- **错误信息**: {result.get('error_message', '未知错误')}")
            
            report.append("\n")
        
        # 性能建议
        report.append("## 性能建议\n")
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            # 找出最快的配置
            fastest = min(successful_results, key=lambda x: x.get("training_time_seconds", float('inf')))
            report.append(f"- **最快配置**: {fastest['test_name']} ({fastest.get('training_time_seconds', 0):.1f}s)")
            
            # 找出内存使用最少的配置
            lowest_memory = min(successful_results, key=lambda x: x.get('gpu_memory_peak_gb', float('inf')))
            report.append(f"- **最低GPU内存**: {lowest_memory['test_name']} ({lowest_memory.get('gpu_memory_peak_gb', 0):.2f}GB)")
            
            # 根据GPU内存给出建议
            if system_info.get("gpu_memory_gb", 0) < 10:
                report.append("- **建议**: 您的GPU内存较少，推荐使用低内存8GB配置")
            elif system_info.get("gpu_memory_gb", 0) < 14:
                report.append("- **建议**: 您的GPU内存适中，推荐使用标准12GB配置")
            else:
                report.append("- **建议**: 您的GPU内存充足，可以使用高性能16GB配置")
        
        return "\n".join(report)
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "benchmark_results.json"):
        """保存基准测试结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"基准测试结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Qwen3优化微调系统性能基准测试")
    parser.add_argument("--output", "-o", default="benchmark_results.json", help="结果输出文件")
    parser.add_argument("--report", "-r", default="benchmark_report.md", help="报告输出文件")
    parser.add_argument("--config", "-c", help="运行单个配置的基准测试")
    parser.add_argument("--name", "-n", help="单个测试的名称")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    if args.config:
        # 运行单个基准测试
        if not args.name:
            args.name = Path(args.config).stem
        
        results = [benchmark.run_benchmark(args.config, args.name)]
    else:
        # 运行所有基准测试
        results = benchmark.run_all_benchmarks()
    
    # 保存结果
    benchmark.save_results(results, args.output)
    
    # 生成报告
    report = benchmark.generate_report(results)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"基准测试报告已保存到: {args.report}")
    
    # 打印简要结果
    print("\n=== 基准测试结果 ===")
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['test_name']}: {result.get('training_time_seconds', 0):.1f}s")

if __name__ == "__main__":
    main()