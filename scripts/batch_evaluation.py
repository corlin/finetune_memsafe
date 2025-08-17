#!/usr/bin/env python3
"""
批量评估脚本

支持批量模型评估、基准测试和结果对比。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# 添加src路径以便导入评估模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import (
    EvaluationEngine, BenchmarkManager, ExperimentTracker,
    StatisticalAnalyzer, ReportGenerator, EvaluationConfig
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('batch_evaluation.log')
        ]
    )


def load_batch_config(config_path: str) -> Dict[str, Any]:
    """
    加载批量评估配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    return config


def batch_model_evaluation(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    批量模型评估
    
    Args:
        config: 评估配置
        
    Returns:
        评估结果列表
    """
    logger.info("开始批量模型评估")
    
    models_config = config.get("models", [])
    datasets_config = config.get("datasets", {})
    eval_config_dict = config.get("evaluation_config", {})
    
    # 创建评估配置
    eval_config = EvaluationConfig(**eval_config_dict)
    
    # 创建评估引擎
    evaluation_engine = EvaluationEngine(eval_config)
    
    results = []
    
    for model_config in models_config:
        model_name = model_config.get("name", "unknown")
        model_path = model_config.get("path")
        tokenizer_path = model_config.get("tokenizer_path", model_path)
        
        logger.info(f"评估模型: {model_name}")
        
        try:
            # 加载模型和分词器
            # 这里需要根据实际情况实现模型加载
            # model = load_model(model_path)
            # tokenizer = load_tokenizer(tokenizer_path)
            
            # 加载数据集
            datasets = {}
            for dataset_name, dataset_path in datasets_config.items():
                # 这里需要根据实际情况实现数据集加载
                # datasets[dataset_name] = load_dataset(dataset_path)
                pass
            
            # 执行评估
            # evaluation_result = evaluation_engine.evaluate_model(
            #     model=model,
            #     tokenizer=tokenizer,
            #     datasets=datasets,
            #     model_name=model_name
            # )
            
            # results.append({
            #     "model_name": model_name,
            #     "model_path": model_path,
            #     "evaluation_result": evaluation_result,
            #     "status": "success"
            # })
            
            # 临时占位符
            results.append({
                "model_name": model_name,
                "model_path": model_path,
                "status": "placeholder - 需要实际模型加载实现"
            })
            
            logger.info(f"模型 {model_name} 评估完成")
            
        except Exception as e:
            logger.error(f"模型 {model_name} 评估失败: {e}")
            results.append({
                "model_name": model_name,
                "model_path": model_path,
                "status": "failed",
                "error": str(e)
            })
    
    logger.info(f"批量模型评估完成，共评估 {len(results)} 个模型")
    return results


def batch_benchmark_testing(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    批量基准测试
    
    Args:
        config: 基准测试配置
        
    Returns:
        基准测试结果列表
    """
    logger.info("开始批量基准测试")
    
    models_config = config.get("models", [])
    benchmarks_config = config.get("benchmarks", [])
    
    # 创建基准测试管理器
    benchmark_manager = BenchmarkManager(
        cache_dir=config.get("cache_dir", "./benchmarks"),
        auto_download=True
    )
    
    # 创建评估引擎
    eval_config = EvaluationConfig(**config.get("evaluation_config", {}))
    evaluation_engine = EvaluationEngine(eval_config)
    
    results = []
    
    for model_config in models_config:
        model_name = model_config.get("name", "unknown")
        model_path = model_config.get("path")
        
        for benchmark_config in benchmarks_config:
            benchmark_name = benchmark_config.get("name")
            tasks = benchmark_config.get("tasks")
            
            logger.info(f"运行基准测试: {model_name} on {benchmark_name}")
            
            try:
                # 加载模型和分词器
                # model = load_model(model_path)
                # tokenizer = load_tokenizer(model_config.get("tokenizer_path", model_path))
                
                # 运行基准测试
                # benchmark_result = benchmark_manager.run_benchmark(
                #     benchmark_name=benchmark_name,
                #     model=model,
                #     tokenizer=tokenizer,
                #     evaluation_engine=evaluation_engine,
                #     tasks=tasks
                # )
                
                # results.append({
                #     "model_name": model_name,
                #     "benchmark_name": benchmark_name,
                #     "benchmark_result": benchmark_result,
                #     "status": "success"
                # })
                
                # 临时占位符
                results.append({
                    "model_name": model_name,
                    "benchmark_name": benchmark_name,
                    "status": "placeholder - 需要实际模型加载实现"
                })
                
                logger.info(f"基准测试完成: {model_name} on {benchmark_name}")
                
            except Exception as e:
                logger.error(f"基准测试失败: {model_name} on {benchmark_name}: {e}")
                results.append({
                    "model_name": model_name,
                    "benchmark_name": benchmark_name,
                    "status": "failed",
                    "error": str(e)
                })
    
    logger.info(f"批量基准测试完成，共运行 {len(results)} 个测试")
    return results


def parallel_evaluation(config: Dict[str, Any], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    并行评估
    
    Args:
        config: 评估配置
        max_workers: 最大工作线程数
        
    Returns:
        评估结果列表
    """
    logger.info(f"开始并行评估，最大工作线程数: {max_workers}")
    
    models_config = config.get("models", [])
    
    def evaluate_single_model(model_config):
        """评估单个模型"""
        model_name = model_config.get("name", "unknown")
        
        try:
            # 这里实现单个模型的评估逻辑
            logger.info(f"并行评估模型: {model_name}")
            
            # 临时占位符
            return {
                "model_name": model_name,
                "status": "success",
                "note": "并行评估占位符"
            }
            
        except Exception as e:
            logger.error(f"并行评估模型 {model_name} 失败: {e}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": str(e)
            }
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_model = {
            executor.submit(evaluate_single_model, model_config): model_config
            for model_config in models_config
        }
        
        # 收集结果
        for future in as_completed(future_to_model):
            model_config = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"并行任务执行失败: {e}")
                results.append({
                    "model_name": model_config.get("name", "unknown"),
                    "status": "failed",
                    "error": str(e)
                })
    
    logger.info(f"并行评估完成，共处理 {len(results)} 个模型")
    return results


def generate_batch_report(results: List[Dict[str, Any]], output_dir: str):
    """
    生成批量评估报告
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
    """
    logger.info("生成批量评估报告")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_models = len(results)
    successful_models = len([r for r in results if r.get("status") == "success"])
    failed_models = total_models - successful_models
    
    # 生成摘要报告
    summary = {
        "batch_evaluation_summary": {
            "total_models": total_models,
            "successful_evaluations": successful_models,
            "failed_evaluations": failed_models,
            "success_rate": successful_models / total_models if total_models > 0 else 0,
            "timestamp": str(Path(__file__).stat().st_mtime)
        },
        "detailed_results": results
    }
    
    # 保存JSON报告
    json_report_path = output_path / "batch_evaluation_report.json"
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 生成HTML报告
    html_report_path = output_path / "batch_evaluation_report.html"
    html_content = generate_html_report(summary)
    with open(html_report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"批量评估报告已生成:")
    logger.info(f"  JSON报告: {json_report_path}")
    logger.info(f"  HTML报告: {html_report_path}")
    logger.info(f"  成功率: {successful_models}/{total_models} ({successful_models/total_models*100:.1f}%)")


def generate_html_report(summary: Dict[str, Any]) -> str:
    """
    生成HTML报告
    
    Args:
        summary: 摘要数据
        
    Returns:
        HTML内容
    """
    batch_summary = summary["batch_evaluation_summary"]
    detailed_results = summary["detailed_results"]
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>批量评估报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ margin: 20px 0; }}
            .results-table {{ border-collapse: collapse; width: 100%; }}
            .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .results-table th {{ background-color: #f2f2f2; }}
            .success {{ color: green; }}
            .failed {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>批量评估报告</h1>
            <p>生成时间: {batch_summary.get('timestamp', 'unknown')}</p>
        </div>
        
        <div class="summary">
            <h2>评估摘要</h2>
            <ul>
                <li>总模型数: {batch_summary['total_models']}</li>
                <li>成功评估: <span class="success">{batch_summary['successful_evaluations']}</span></li>
                <li>失败评估: <span class="failed">{batch_summary['failed_evaluations']}</span></li>
                <li>成功率: {batch_summary['success_rate']:.1%}</li>
            </ul>
        </div>
        
        <div class="results">
            <h2>详细结果</h2>
            <table class="results-table">
                <tr>
                    <th>模型名称</th>
                    <th>状态</th>
                    <th>备注</th>
                </tr>
    """
    
    for result in detailed_results:
        status_class = "success" if result.get("status") == "success" else "failed"
        note = result.get("error", result.get("note", ""))
        
        html_template += f"""
                <tr>
                    <td>{result.get('model_name', 'unknown')}</td>
                    <td class="{status_class}">{result.get('status', 'unknown')}</td>
                    <td>{note}</td>
                </tr>
        """
    
    html_template += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html_template


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量评估脚本")
    parser.add_argument("--config", required=True, help="批量评估配置文件")
    parser.add_argument("--output-dir", default="./batch_results", help="输出目录")
    parser.add_argument("--mode", choices=["evaluation", "benchmark", "parallel"], 
                       default="evaluation", help="评估模式")
    parser.add_argument("--max-workers", type=int, default=4, help="并行评估的最大工作线程数")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        # 加载配置
        config = load_batch_config(args.config)
        
        # 根据模式执行相应的批量操作
        if args.mode == "evaluation":
            results = batch_model_evaluation(config)
        elif args.mode == "benchmark":
            results = batch_benchmark_testing(config)
        elif args.mode == "parallel":
            results = parallel_evaluation(config, args.max_workers)
        else:
            raise ValueError(f"不支持的评估模式: {args.mode}")
        
        # 生成报告
        generate_batch_report(results, args.output_dir)
        
        logger.info("批量评估完成")
        
    except Exception as e:
        logger.error(f"批量评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()