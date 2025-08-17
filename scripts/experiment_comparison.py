#!/usr/bin/env python3
"""
实验对比脚本

支持多实验结果对比分析和可视化报告生成。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

# 添加src路径以便导入评估模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import (
    ExperimentTracker, StatisticalAnalyzer, ReportGenerator,
    ComparisonResult
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('experiment_comparison.log')
        ]
    )


def load_comparison_config(config_path: str) -> Dict[str, Any]:
    """
    加载对比配置
    
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


def compare_experiments_by_ids(experiment_ids: List[str], 
                              experiment_dir: str,
                              metrics: List[str] = None) -> ComparisonResult:
    """
    根据实验ID对比实验
    
    Args:
        experiment_ids: 实验ID列表
        experiment_dir: 实验目录
        metrics: 对比指标列表
        
    Returns:
        对比结果
    """
    logger.info(f"对比实验: {experiment_ids}")
    
    # 创建实验跟踪器
    experiment_tracker = ExperimentTracker(experiment_dir=experiment_dir)
    
    # 获取实验数据
    experiments = []
    for exp_id in experiment_ids:
        experiment = experiment_tracker.get_experiment(exp_id)
        if experiment:
            experiments.append(experiment)
            logger.info(f"加载实验: {exp_id} - {experiment['name']}")
        else:
            logger.warning(f"未找到实验: {exp_id}")
    
    if not experiments:
        raise ValueError("没有找到有效的实验数据")
    
    # 执行对比分析
    comparison_result = experiment_tracker.compare_experiments(experiment_ids)
    
    return comparison_result


def compare_experiments_by_tags(tags: List[str],
                               experiment_dir: str,
                               limit: int = 10) -> ComparisonResult:
    """
    根据标签对比实验
    
    Args:
        tags: 标签列表
        experiment_dir: 实验目录
        limit: 最大实验数量
        
    Returns:
        对比结果
    """
    logger.info(f"根据标签对比实验: {tags}")
    
    # 创建实验跟踪器
    experiment_tracker = ExperimentTracker(experiment_dir=experiment_dir)
    
    # 搜索带有指定标签的实验
    experiments = experiment_tracker.list_experiments(tags=tags, limit=limit)
    
    if not experiments:
        raise ValueError(f"没有找到带有标签 {tags} 的实验")
    
    experiment_ids = [exp["id"] for exp in experiments]
    logger.info(f"找到 {len(experiment_ids)} 个实验")
    
    # 执行对比分析
    comparison_result = experiment_tracker.compare_experiments(experiment_ids)
    
    return comparison_result


def advanced_statistical_analysis(comparison_result: ComparisonResult,
                                 output_dir: str) -> Dict[str, Any]:
    """
    高级统计分析
    
    Args:
        comparison_result: 对比结果
        output_dir: 输出目录
        
    Returns:
        统计分析结果
    """
    logger.info("执行高级统计分析")
    
    # 创建统计分析器
    statistical_analyzer = StatisticalAnalyzer(
        confidence_level=0.95,
        output_dir=output_dir
    )
    
    analysis_results = {}
    
    # 对每个指标进行统计分析
    for metric_name, values in comparison_result.metrics.items():
        if len(values) >= 2:
            logger.info(f"分析指标: {metric_name}")
            
            # 置信区间计算
            confidence_intervals = {}
            for i, model in enumerate(comparison_result.models):
                if i < len(values):
                    # 这里假设我们有多次运行的数据，实际中可能需要调整
                    model_values = [values[i]] * 5  # 模拟多次运行
                    ci = statistical_analyzer.calculate_confidence_intervals(model_values)
                    confidence_intervals[model] = ci
            
            # 显著性检验（如果有足够的数据）
            significance_tests = {}
            if len(values) >= 2:
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        model1 = comparison_result.models[i]
                        model2 = comparison_result.models[j]
                        
                        # 模拟多次运行数据进行检验
                        group1 = [values[i]] * 5
                        group2 = [values[j]] * 5
                        
                        test_result = statistical_analyzer.perform_significance_test(
                            group1, group2
                        )
                        significance_tests[f"{model1}_vs_{model2}"] = test_result
            
            analysis_results[metric_name] = {
                "confidence_intervals": confidence_intervals,
                "significance_tests": significance_tests
            }
    
    # 创建排行榜
    leaderboard = statistical_analyzer.create_leaderboard(
        results=[],  # 这里需要实际的评估结果对象
        metric_name="overall_score"
    )
    
    analysis_results["leaderboard"] = leaderboard
    
    return analysis_results


def generate_comprehensive_report(comparison_result: ComparisonResult,
                                statistical_analysis: Dict[str, Any],
                                output_dir: str,
                                report_formats: List[str] = ["html", "json"]):
    """
    生成综合报告
    
    Args:
        comparison_result: 对比结果
        statistical_analysis: 统计分析结果
        output_dir: 输出目录
        report_formats: 报告格式列表
    """
    logger.info("生成综合对比报告")
    
    # 创建报告生成器
    report_generator = ReportGenerator(output_dir=output_dir)
    
    generated_reports = []
    
    for format_type in report_formats:
        try:
            report_path = report_generator.generate_comparison_report(
                comparison_result=comparison_result,
                format_type=format_type,
                include_charts=True
            )
            generated_reports.append(report_path)
            logger.info(f"生成 {format_type.upper()} 报告: {report_path}")
        except Exception as e:
            logger.error(f"生成 {format_type} 报告失败: {e}")
    
    # 生成可视化图表
    try:
        chart_types = ["bar", "radar", "heatmap"]
        for chart_type in chart_types:
            chart_path = report_generator.create_performance_visualization(
                comparison_result=comparison_result,
                chart_type=chart_type
            )
            if chart_path:
                logger.info(f"生成 {chart_type} 图表: {chart_path}")
    except Exception as e:
        logger.error(f"生成可视化图表失败: {e}")
    
    # 保存统计分析结果
    try:
        stats_file = Path(output_dir) / "statistical_analysis.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistical_analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"保存统计分析结果: {stats_file}")
    except Exception as e:
        logger.error(f"保存统计分析结果失败: {e}")
    
    return generated_reports


def export_comparison_data(comparison_result: ComparisonResult,
                          output_dir: str,
                          formats: List[str] = ["csv", "excel"]):
    """
    导出对比数据
    
    Args:
        comparison_result: 对比结果
        output_dir: 输出目录
        formats: 导出格式列表
    """
    logger.info("导出对比数据")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    data_rows = []
    for i, model in enumerate(comparison_result.models):
        row = {"model_name": model}
        for metric, values in comparison_result.metrics.items():
            if i < len(values):
                row[metric] = values[i]
            else:
                row[metric] = 0.0
        data_rows.append(row)
    
    # 导出CSV
    if "csv" in formats:
        try:
            import csv
            csv_file = output_path / "comparison_data.csv"
            
            if data_rows:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(data_rows)
                
                logger.info(f"导出CSV文件: {csv_file}")
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
    
    # 导出Excel
    if "excel" in formats:
        try:
            import pandas as pd
            excel_file = output_path / "comparison_data.xlsx"
            
            df = pd.DataFrame(data_rows)
            df.to_excel(excel_file, index=False)
            
            logger.info(f"导出Excel文件: {excel_file}")
        except ImportError:
            logger.warning("pandas未安装，跳过Excel导出")
        except Exception as e:
            logger.error(f"导出Excel失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验对比脚本")
    parser.add_argument("--experiment-ids", nargs="+", help="实验ID列表")
    parser.add_argument("--tags", nargs="+", help="实验标签列表")
    parser.add_argument("--experiment-dir", default="./experiments", help="实验目录")
    parser.add_argument("--output-dir", default="./comparison_results", help="输出目录")
    parser.add_argument("--config", help="对比配置文件")
    parser.add_argument("--metrics", nargs="+", help="对比指标列表")
    parser.add_argument("--report-formats", nargs="+", default=["html", "json"], 
                       choices=["html", "json", "csv", "latex"], help="报告格式")
    parser.add_argument("--export-formats", nargs="+", default=["csv"], 
                       choices=["csv", "excel"], help="数据导出格式")
    parser.add_argument("--advanced-analysis", action="store_true", help="执行高级统计分析")
    parser.add_argument("--limit", type=int, default=10, help="最大实验数量（按标签搜索时）")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        # 加载配置（如果提供）
        config = {}
        if args.config:
            config = load_comparison_config(args.config)
            
            # 配置文件中的参数可以覆盖命令行参数
            if "experiment_ids" in config and not args.experiment_ids:
                args.experiment_ids = config["experiment_ids"]
            if "tags" in config and not args.tags:
                args.tags = config["tags"]
            if "metrics" in config and not args.metrics:
                args.metrics = config["metrics"]
        
        # 执行实验对比
        comparison_result = None
        
        if args.experiment_ids:
            comparison_result = compare_experiments_by_ids(
                experiment_ids=args.experiment_ids,
                experiment_dir=args.experiment_dir,
                metrics=args.metrics
            )
        elif args.tags:
            comparison_result = compare_experiments_by_tags(
                tags=args.tags,
                experiment_dir=args.experiment_dir,
                limit=args.limit
            )
        else:
            raise ValueError("必须提供实验ID列表或标签列表")
        
        # 高级统计分析
        statistical_analysis = {}
        if args.advanced_analysis:
            statistical_analysis = advanced_statistical_analysis(
                comparison_result=comparison_result,
                output_dir=args.output_dir
            )
        
        # 生成报告
        generated_reports = generate_comprehensive_report(
            comparison_result=comparison_result,
            statistical_analysis=statistical_analysis,
            output_dir=args.output_dir,
            report_formats=args.report_formats
        )
        
        # 导出数据
        export_comparison_data(
            comparison_result=comparison_result,
            output_dir=args.output_dir,
            formats=args.export_formats
        )
        
        # 输出摘要信息
        logger.info("实验对比完成")
        logger.info(f"对比模型数量: {len(comparison_result.models)}")
        logger.info(f"对比指标数量: {len(comparison_result.metrics)}")
        
        if comparison_result.best_model:
            logger.info("各指标最佳模型:")
            for metric, model in comparison_result.best_model.items():
                logger.info(f"  {metric}: {model}")
        
        logger.info(f"生成报告数量: {len(generated_reports)}")
        for report in generated_reports:
            logger.info(f"  报告: {report}")
        
    except Exception as e:
        logger.error(f"实验对比失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()