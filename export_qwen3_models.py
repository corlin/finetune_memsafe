#!/usr/bin/env python3
"""
Qwen3模型导出脚本

专门用于导出合并后的PyTorch模型和ONNX模型
基座模型: Qwen/Qwen3-4B-Thinking-2507
Checkpoint: qwen3-finetuned/checkpoint-30

使用uv进行Python环境和依赖管理
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

def check_uv_installation():
    """检查uv是否已安装"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ uv已安装: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv未安装或不在PATH中")
        print("请安装uv: https://docs.astral.sh/uv/getting-started/installation/")
        print("Windows: powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print("macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

def setup_uv_environment():
    """设置uv环境"""
    logger = logging.getLogger(__name__)
    
    # 检查是否存在pyproject.toml
    if not Path("pyproject.toml").exists():
        logger.error("未找到pyproject.toml文件")
        return False
    
    try:
        # 同步依赖
        logger.info("正在同步uv依赖...")
        result = subprocess.run(['uv', 'sync'], check=True, capture_output=True, text=True)
        logger.info("uv依赖同步完成")
        
        # 检查虚拟环境
        venv_result = subprocess.run(['uv', 'venv', '--seed'], capture_output=True, text=True)
        if venv_result.returncode == 0:
            logger.info("uv虚拟环境已准备就绪")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"uv环境设置失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def run_with_uv():
    """使用uv运行导出脚本"""
    try:
        # 构建uv run命令
        cmd = ['uv', 'run', 'python', __file__, '--internal']
        
        # 执行命令
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"uv运行失败: {e}")
        return False

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_export_controller import ModelExportController
from src.export_config import ExportConfiguration

def setup_logging():
    """设置日志配置"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f'qwen3_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
        ]
    )

def check_prerequisites():
    """检查运行前提条件"""
    logger = logging.getLogger(__name__)
    logger.info("检查运行前提条件...")
    
    # 检查checkpoint目录
    checkpoint_path = "./qwen3-finetuned/checkpoint-30"
    if not os.path.exists(checkpoint_path):
        logger.error(f"错误: Checkpoint目录不存在: {checkpoint_path}")
        logger.error("请确保checkpoint-30目录存在")
        return False
    
    # 检查必要文件
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"缺少以下文件: {missing_files}")
        logger.warning("将尝试从主目录获取tokenizer文件")
    
    # 检查主目录的tokenizer文件
    main_dir = "./qwen3-finetuned"
    for file in ["tokenizer.json", "tokenizer_config.json"]:
        if file in missing_files:
            main_file = os.path.join(main_dir, file)
            if os.path.exists(main_file):
                logger.info(f"在主目录找到 {file}")
            else:
                logger.error(f"在主目录也未找到 {file}")
    
    # 创建输出目录
    output_dir = "./exported_models"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录已准备: {output_dir}")
    
    logger.info("前提条件检查完成")
    return True

def create_export_config():
    """创建导出配置"""
    config = ExportConfiguration(
        # 基本配置
        checkpoint_path="./qwen3-finetuned/checkpoint-30",
        base_model_name="Qwen/Qwen3-4B-Thinking-2507",
        output_directory="./exported_models/qwen3_merged",
        
        # 优化配置
        quantization_level="int8",  # 使用int8量化减少模型大小
        remove_training_artifacts=True,
        compress_weights=True,
        
        # 导出格式 - 只导出PyTorch和ONNX
        export_pytorch=True,
        export_onnx=True,
        export_tensorrt=False,  # 不导出TensorRT以节省时间
        
        # ONNX配置
        onnx_dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        },
        onnx_opset_version=20,
        onnx_optimize_graph=True,
        
        # 验证配置
        run_validation_tests=True,
        test_input_samples=[
            "你好，请介绍一下自己。",
            "什么是人工智能？请简单解释一下。",
            "请解释深度学习的基本概念。",
            "如何优化神经网络的性能？"
        ],
        
        # 监控配置
        enable_progress_monitoring=True,
        log_level="INFO",
        
        # 内存配置
        max_memory_usage_gb=12.0,  # 限制内存使用
        enable_memory_optimization=True,
        
        # 错误处理配置
        enable_error_recovery=True,
        max_retry_attempts=3
    )
    
    return config

def export_qwen3_models():
    """导出Qwen3模型"""
    logger = logging.getLogger(__name__)
    logger.info("=== 开始Qwen3模型导出 ===")
    
    try:
        # 创建导出配置
        config = create_export_config()
        
        logger.info("导出配置:")
        logger.info(f"  基座模型: {config.base_model_name}")
        logger.info(f"  Checkpoint: {config.checkpoint_path}")
        logger.info(f"  输出目录: {config.output_directory}")
        logger.info(f"  量化级别: {config.quantization_level}")
        logger.info(f"  导出格式: PyTorch={config.export_pytorch}, ONNX={config.export_onnx}")
        
        # 创建导出控制器
        controller = ModelExportController(config)
        
        # 执行导出
        logger.info("\n开始模型导出...")
        result = controller.export_model()
        
        # 处理结果
        if result.success:
            logger.info("\n=== 导出成功 ===")
            logger.info(f"导出ID: {result.export_id}")
            logger.info(f"导出时间: {result.timestamp}")
            logger.info(f"总耗时: {result.total_duration_seconds:.2f} 秒")
            
            # 显示导出的模型路径
            if result.pytorch_model_path:
                logger.info(f"PyTorch模型: {result.pytorch_model_path}")
                logger.info(f"  模型大小: {_get_file_size(result.pytorch_model_path)}")
            
            if result.onnx_model_path:
                logger.info(f"ONNX模型: {result.onnx_model_path}")
                logger.info(f"  模型大小: {_get_file_size(result.onnx_model_path)}")
            
            # 显示优化统计
            if hasattr(result, 'original_size_mb') and hasattr(result, 'optimized_size_mb'):
                logger.info(f"\n=== 优化统计 ===")
                logger.info(f"原始大小: {result.original_size_mb:.1f} MB")
                logger.info(f"优化后大小: {result.optimized_size_mb:.1f} MB")
                logger.info(f"大小减少: {result.size_reduction_percentage:.1f}%")
            
            # 显示性能指标
            if hasattr(result, 'inference_speed_ms') and result.inference_speed_ms:
                logger.info(f"推理速度: {result.inference_speed_ms:.2f} ms")
            if hasattr(result, 'memory_usage_mb') and result.memory_usage_mb:
                logger.info(f"内存使用: {result.memory_usage_mb:.1f} MB")
            
            # 显示验证结果
            logger.info(f"\n=== 验证结果 ===")
            if hasattr(result, 'validation_passed'):
                logger.info(f"验证通过: {result.validation_passed}")
            if hasattr(result, 'validation_report_path') and result.validation_report_path:
                logger.info(f"验证报告: {result.validation_report_path}")
            
            # 显示警告信息
            if hasattr(result, 'warnings') and result.warnings:
                logger.info(f"\n=== 警告信息 ===")
                for warning in result.warnings:
                    logger.warning(f"- {warning}")
            
            logger.info(f"\n导出完成！模型已保存到: {config.output_directory}")
            
            # 显示使用建议
            logger.info("\n=== 使用建议 ===")
            logger.info("1. PyTorch模型可以直接用于推理和进一步微调")
            logger.info("2. ONNX模型适合部署到不同的推理引擎")
            logger.info("3. 建议在部署前进行充分的测试")
            
            return True
            
        else:
            logger.error("\n=== 导出失败 ===")
            logger.error(f"错误信息: {result.error_message}")
            
            if hasattr(result, 'warnings') and result.warnings:
                logger.error("警告信息:")
                for warning in result.warnings:
                    logger.warning(f"- {warning}")
            
            # 显示恢复建议
            if hasattr(controller, 'get_recovery_suggestions'):
                suggestions = controller.get_recovery_suggestions(Exception(result.error_message))
                if suggestions:
                    logger.error("\n=== 恢复建议 ===")
                    for suggestion in suggestions:
                        logger.error(f"- {suggestion}")
            
            return False
            
    except Exception as e:
        logger.error(f"\n导出过程出错: {e}")
        logger.error("详细错误信息:", exc_info=True)
        return False

def _get_file_size(file_path):
    """获取文件大小的可读格式"""
    if not file_path or not os.path.exists(file_path):
        return "未知"
    
    try:
        size_bytes = os.path.getsize(file_path)
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.1f} MB"
        else:
            return f"{size_bytes / 1024**3:.1f} GB"
    except:
        return "未知"

def main():
    """主函数"""
    # 检查是否是内部调用（通过uv run）
    if '--internal' in sys.argv:
        # 这是通过uv run调用的，直接执行导出逻辑
        return main_internal()
    
    print("Qwen3模型导出工具 (使用uv)")
    print("=" * 50)
    print(f"基座模型: Qwen/Qwen3-4B-Thinking-2507")
    print(f"Checkpoint: qwen3-finetuned/checkpoint-30")
    print(f"导出格式: PyTorch + ONNX")
    print("=" * 50)
    
    # 检查uv安装
    if not check_uv_installation():
        sys.exit(1)
    
    # 设置uv环境
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if not setup_uv_environment():
        logger.error("uv环境设置失败")
        sys.exit(1)
    
    # 使用uv运行实际的导出逻辑
    print("\n🚀 使用uv启动模型导出...")
    success = run_with_uv()
    
    if not success:
        print("\n❌ 模型导出失败")
        sys.exit(1)

def main_internal():
    """内部主函数（通过uv run调用）"""
    print("\n📦 在uv环境中运行模型导出...")
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 检查前提条件
        if not check_prerequisites():
            logger.error("前提条件检查失败，退出程序")
            return False
        
        # 执行导出
        success = export_qwen3_models()
        
        if success:
            print("\n✅ 模型导出成功！")
            print("\n下一步:")
            print("1. 查看导出的模型文件")
            print("2. 运行验证脚本测试模型功能")
            print("3. 部署模型到目标环境")
            return True
        else:
            print("\n❌ 模型导出失败，请查看日志文件获取详细信息")
            return False
            
    except KeyboardInterrupt:
        logger.info("用户中断了导出过程")
        print("\n用户中断了导出过程")
        return False
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"\n程序执行失败: {e}")
        return False

if __name__ == "__main__":
    main()