#!/usr/bin/env python3
"""
简化的QA评估运行脚本
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """检查运行环境"""
    logger.info("检查运行环境...")
    
    # 检查Python包
    required_packages = ['torch', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} 未安装")
    
    if missing_packages:
        logger.error(f"缺少必要的包: {missing_packages}")
        logger.info("请运行: pip install " + " ".join(missing_packages))
        return False
    
    # 检查数据目录
    data_dir = Path("data/raw")
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return False
    
    qa_files = list(data_dir.glob("QA*.md"))
    if not qa_files:
        logger.error(f"在 {data_dir} 中未找到QA*.md文件")
        return False
    
    logger.info(f"✓ 找到 {len(qa_files)} 个QA文件")
    
    # 检查checkpoint目录
    checkpoint_dir = Path("exported_models/qwen3_merged_lightweight/Qwen_Qwen3-4B-Thinking-2507_pytorch_20250820_225451")
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint目录不存在: {checkpoint_dir}")
        logger.info("将使用默认路径，如果路径不正确请手动指定")
    else:
        logger.info(f"✓ Checkpoint目录存在: {checkpoint_dir}")
    
    return True

def run_extraction_only():
    """仅运行QA提取"""
    logger.info("运行QA数据提取...")
    
    cmd = [
        sys.executable, "qa_extractor_evaluator.py",
        "--extract-only",
        "--max-samples", "0",  # 提取所有
        "--output-qa", "all_qa_pairs.json"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("QA数据提取完成")
        logger.info("输出文件: all_qa_pairs.json")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"QA数据提取失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def run_quick_evaluation():
    """运行快速评估（少量样本）"""
    logger.info("运行快速评估（10个样本）...")
    
    cmd = [
        sys.executable, "qa_extractor_evaluator.py",
        "--max-samples", "10",
        "--sample-strategy", "balanced",
        "--output-qa", "quick_qa_pairs.json",
        "--output-report", "quick_evaluation_report.json"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("快速评估完成")
        logger.info("输出文件: quick_qa_pairs.json, quick_evaluation_report.json")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"快速评估失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def run_full_evaluation():
    """运行完整评估"""
    logger.info("运行完整评估（50个样本）...")
    
    cmd = [
        sys.executable, "qa_extractor_evaluator.py",
        "--max-samples", "50",
        "--sample-strategy", "balanced",
        "--output-qa", "full_qa_pairs.json",
        "--output-report", "full_evaluation_report.json"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("完整评估完成")
        logger.info("输出文件: full_qa_pairs.json, full_evaluation_report.json")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"完整评估失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def run_category_evaluation():
    """按分类运行评估"""
    logger.info("按分类运行评估...")
    
    categories = ["基础概念类", "密码应用等级类", "技术要求类"]
    
    for category in categories:
        logger.info(f"评估分类: {category}")
        
        safe_category = category.replace(" ", "_").replace("/", "_")
        cmd = [
            sys.executable, "qa_extractor_evaluator.py",
            "--categories", category,
            "--max-samples", "20",
            "--output-qa", f"qa_pairs_{safe_category}.json",
            "--output-report", f"evaluation_report_{safe_category}.json"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"分类 {category} 评估完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"分类 {category} 评估失败: {e}")
            continue

def main():
    """主函数"""
    print("="*60)
    print("QA数据提取和Checkpoint评估工具")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请解决上述问题后重试")
        sys.exit(1)
    
    print("\n请选择运行模式:")
    print("1. 仅提取QA数据（不评估模型）")
    print("2. 快速评估（10个样本）")
    print("3. 完整评估（50个样本）")
    print("4. 按分类评估（每个分类20个样本）")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == "1":
                success = run_extraction_only()
                break
            elif choice == "2":
                success = run_quick_evaluation()
                break
            elif choice == "3":
                success = run_full_evaluation()
                break
            elif choice == "4":
                success = run_category_evaluation()
                break
            elif choice == "5":
                print("退出程序")
                sys.exit(0)
            else:
                print("无效选择，请输入1-5")
                continue
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            sys.exit(0)
        except Exception as e:
            logger.error(f"运行出错: {e}")
            sys.exit(1)
    
    if success:
        print("\n✅ 任务完成！")
        print("请查看生成的JSON文件获取详细结果")
    else:
        print("\n❌ 任务失败！")
        print("请查看上面的错误信息")

if __name__ == "__main__":
    main()