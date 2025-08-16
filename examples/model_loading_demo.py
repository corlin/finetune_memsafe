"""
模型加载和LoRA配置演示
"""

import sys
import os
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 现在可以正确导入src模块
try:
    from src.model_manager import ModelManager
    from src.lora_adapter import LoRAAdapter
    from src.memory_optimizer import MemoryOptimizer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保从项目根目录运行此脚本，或使用 'uv run examples/model_loading_demo.py'")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_model_loading_and_lora():
    """演示模型加载和LoRA配置的完整流程"""
    
    try:
        # 1. 初始化组件
        logger.info("=== 初始化组件 ===")
        memory_optimizer = MemoryOptimizer()
        model_manager = ModelManager(max_memory_gb=13.0)
        lora_adapter = LoRAAdapter(r=6, alpha=12)
        
        # 2. 检查内存状态
        logger.info("=== 检查内存状态 ===")
        memory_status = memory_optimizer.monitor_gpu_memory()
        logger.info(f"当前GPU内存状态: {memory_status}")
        
        # 3. 清理内存
        logger.info("=== 清理GPU内存 ===")
        memory_optimizer.cleanup_gpu_memory()
        
        # 4. 加载模型（使用较小的模型进行演示）
        logger.info("=== 加载模型 ===")
        model_name = "Qwen/Qwen3-4B-Thinking-2507"  # 使用较小的模型进行演示
        
        try:
            model, tokenizer = model_manager.load_model_with_quantization(model_name)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败，使用备用模型: {e}")
            # 如果主模型加载失败，可以尝试其他模型
            model_name = "microsoft/DialoGPT-small"
            model, tokenizer = model_manager.load_model_with_quantization(model_name)
        
        # 5. 获取模型信息
        logger.info("=== 模型信息 ===")
        model_info = model_manager.get_model_info(model)
        for key, value in model_info.items():
            logger.info(f"{key}: {value}")
        
        # 6. 准备模型用于训练
        logger.info("=== 准备模型用于训练 ===")
        model = model_manager.prepare_for_training(model)
        
        # 7. 应用LoRA
        logger.info("=== 应用LoRA适配器 ===")
        model = lora_adapter.setup_lora_for_model(model)
        
        # 8. 获取LoRA参数信息
        logger.info("=== LoRA参数信息 ===")
        params_info = lora_adapter.get_trainable_params_info(model)
        for key, value in params_info.items():
            if key != 'lora_modules':  # 不打印所有模块名称
                logger.info(f"{key}: {value}")
        
        # 9. 检查最终内存状态
        logger.info("=== 最终内存状态 ===")
        final_memory_status = memory_optimizer.monitor_gpu_memory()
        logger.info(f"最终GPU内存状态: {final_memory_status}")
        
        logger.info("=== 演示完成 ===")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    demo_model_loading_and_lora()