"""
命令行接口

提供数据拆分和评估功能的命令行工具。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset
from .data_splitter import DataSplitter
from .quality_analyzer import QualityAnalyzer
from .config_manager import config_manager

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('evaluation.log')
        ]
    )


def split_data_command(args):
    """数据拆分命令"""
    logger.info("开始执行数据拆分命令")
    
    # 加载配置
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        config = config_manager.get_default_config()
    
    # 覆盖命令行参数
    split_config = config["data_split"]
    if args.train_ratio:
        split_config["train_ratio"] = args.train_ratio
    if args.val_ratio:
        split_config["val_ratio"] = args.val_ratio
    if args.test_ratio:
        split_config["test_ratio"] = args.test_ratio
    if args.stratify_by:
        split_config["stratify_by"] = args.stratify_by
    if args.random_seed is not None:
        split_config["random_seed"] = args.random_seed
    
    # 创建数据拆分器
    splitter = DataSplitter(
        train_ratio=split_config["train_ratio"],
        val_ratio=split_config["val_ratio"],
        test_ratio=split_config["test_ratio"],
        stratify_by=split_config.get("stratify_by"),
        random_seed=split_config["random_seed"],
        min_samples_per_split=split_config.get("min_samples_per_split", 10),
        enable_quality_analysis=not args.no_quality_analysis
    )
    
    # 加载数据集
    logger.info(f"从 {args.input} 加载数据集")
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # 检查是否是Hugging Face数据集格式
        if (input_path / "dataset_info.json").exists() or (input_path / "state.json").exists():
            dataset = Dataset.load_from_disk(args.input)
        else:
            # 处理包含多个文件的目录
            from datasets import Dataset as HFDataset
            import glob
            
            # 查找所有markdown文件
            md_files = glob.glob(str(input_path / "*.md"))
            if not md_files:
                raise ValueError(f"在目录 {args.input} 中未找到markdown文件")
            
            # 读取所有markdown文件内容
            texts = []
            filenames = []
            for md_file in md_files:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts.append(content)
                    filenames.append(Path(md_file).name)
            
            # 创建数据集
            dataset = HFDataset.from_dict({
                "text": texts,
                "filename": filenames
            })
            logger.info(f"从 {len(md_files)} 个markdown文件创建数据集，共 {len(dataset)} 条记录")
    else:
        raise ValueError(f"不支持的输入格式: {args.input}")
    
    # 执行拆分
    result = splitter.split_data(dataset, args.output)
    
    logger.info(f"数据拆分完成，结果保存到: {args.output}")
    logger.info(f"分布一致性分数: {result.distribution_analysis.consistency_score:.4f}")


def analyze_quality_command(args):
    """数据质量分析命令"""
    logger.info("开始执行数据质量分析命令")
    
    # 创建质量分析器
    analyzer = QualityAnalyzer(
        min_length=args.min_length,
        max_length=args.max_length,
        length_outlier_threshold=args.outlier_threshold
    )
    
    # 加载数据集
    logger.info(f"从 {args.input} 加载数据集")
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # 检查是否是Hugging Face数据集格式
        if (input_path / "dataset_info.json").exists() or (input_path / "state.json").exists():
            dataset = Dataset.load_from_disk(args.input)
        else:
            # 处理包含多个文件的目录
            from datasets import Dataset as HFDataset
            import glob
            
            # 查找所有markdown文件
            md_files = glob.glob(str(input_path / "*.md"))
            if not md_files:
                raise ValueError(f"在目录 {args.input} 中未找到markdown文件")
            
            # 读取所有markdown文件内容
            texts = []
            filenames = []
            for md_file in md_files:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts.append(content)
                    filenames.append(Path(md_file).name)
            
            # 创建数据集
            dataset = HFDataset.from_dict({
                "text": texts,
                "filename": filenames
            })
            logger.info(f"从 {len(md_files)} 个markdown文件创建数据集，共 {len(dataset)} 条记录")
    else:
        raise ValueError(f"不支持的输入格式: {args.input}")
    
    # 执行质量分析
    report = analyzer.analyze_data_quality(dataset, args.dataset_name or "dataset")
    
    # 生成报告
    report_path = analyzer.generate_quality_report(report, args.output)
    
    logger.info(f"质量分析完成，报告保存到: {report_path}")
    logger.info(f"质量分数: {report.quality_score:.4f}")
    logger.info(f"发现问题: {len(report.issues)} 个")


def create_config_command(args):
    """创建配置文件命令"""
    logger.info("创建配置文件模板")
    
    config_manager.create_config_template(args.output)
    
    logger.info(f"配置文件模板已创建: {args.output}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据拆分和评估工具")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 数据拆分命令
    split_parser = subparsers.add_parser("split", help="拆分数据集")
    split_parser.add_argument("--input", required=True, help="输入数据集路径")
    split_parser.add_argument("--output", required=True, help="输出目录")
    split_parser.add_argument("--config", help="配置文件路径")
    split_parser.add_argument("--train-ratio", type=float, help="训练集比例")
    split_parser.add_argument("--val-ratio", type=float, help="验证集比例")
    split_parser.add_argument("--test-ratio", type=float, help="测试集比例")
    split_parser.add_argument("--stratify-by", help="分层字段")
    split_parser.add_argument("--random-seed", type=int, help="随机种子")
    split_parser.add_argument("--no-quality-analysis", action="store_true", 
                             help="禁用质量分析")
    split_parser.set_defaults(func=split_data_command)
    
    # 质量分析命令
    quality_parser = subparsers.add_parser("quality", help="分析数据质量")
    quality_parser.add_argument("--input", required=True, help="输入数据集路径")
    quality_parser.add_argument("--output", required=True, help="输出目录")
    quality_parser.add_argument("--dataset-name", help="数据集名称")
    quality_parser.add_argument("--min-length", type=int, default=5, help="最小文本长度")
    quality_parser.add_argument("--max-length", type=int, default=2048, help="最大文本长度")
    quality_parser.add_argument("--outlier-threshold", type=float, default=3.0, 
                               help="异常值阈值（标准差倍数）")
    quality_parser.set_defaults(func=analyze_quality_command)
    
    # 创建配置命令
    config_parser = subparsers.add_parser("create-config", help="创建配置文件模板")
    config_parser.add_argument("--output", required=True, help="输出配置文件路径")
    config_parser.set_defaults(func=create_config_command)
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 执行命令
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()