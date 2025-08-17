#!/usr/bin/env python3
"""
使用uv管理依赖，将PyTorch模型转换为GGUF格式
"""

import os
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, cwd=None):
    """运行命令并处理错误"""
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")
        if e.stderr:
            logger.error(f"错误信息: {e.stderr}")
        return False

def setup_llama_cpp():
    """设置llama.cpp环境"""
    llama_cpp_dir = "llama.cpp"
    
    # 检查是否已存在
    if os.path.exists(llama_cpp_dir):
        logger.info("llama.cpp目录已存在")
        return llama_cpp_dir
    
    # 克隆仓库
    logger.info("正在克隆llama.cpp仓库...")
    if not run_command(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"]):
        return None
    
    return llama_cpp_dir

def install_dependencies_with_uv():
    """使用uv安装所有必要的依赖"""
    logger.info("正在使用uv安装依赖包...")
    
    # 必要的依赖包
    dependencies = [
        "torch",
        "transformers", 
        "sentencepiece",
        "protobuf",
        "accelerate",
        "safetensors",
        "numpy",
        "huggingface-hub"
    ]
    
    # 批量安装依赖
    logger.info("批量安装依赖包...")
    cmd = ["uv", "add"] + dependencies
    
    if run_command(cmd):
        logger.info("所有依赖安装成功")
        return True
    else:
        logger.warning("uv批量安装失败，尝试逐个安装...")
        
        # 逐个安装
        success_count = 0
        for dep in dependencies:
            logger.info(f"安装 {dep}...")
            if run_command(["uv", "add", dep]):
                success_count += 1
                logger.info(f"✓ {dep} 安装成功")
            else:
                logger.warning(f"✗ {dep} 安装失败")
        
        logger.info(f"成功安装 {success_count}/{len(dependencies)} 个依赖包")
        return success_count > len(dependencies) // 2  # 至少一半成功

def check_model_files(model_path):
    """检查模型文件完整性"""
    logger.info("检查模型文件...")
    
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    safetensors_files = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"
    ]
    
    # 检查必需文件
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            logger.error(f"缺少必需文件: {file}")
            return False
        logger.info(f"✓ 找到文件: {file}")
    
    # 检查模型权重文件
    found_weights = False
    for file in safetensors_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            logger.info(f"✓ 找到权重文件: {file}")
            found_weights = True
    
    if not found_weights:
        logger.error("未找到模型权重文件")
        return False
    
    logger.info("模型文件检查完成")
    return True

def convert_model():
    """转换模型"""
    # 路径设置
    model_path = "exported_models/qwen3_merged_lightweight/Qwen_Qwen3-4B-Thinking-2507_pytorch_20250816_225620"
    output_dir = "exported_models/qwen3_merged_lightweight/gguf"
    output_file = os.path.join(output_dir, "qwen3-4b-thinking-f16.gguf")
    llama_cpp_dir = "llama.cpp"
    
    # 检查输入路径
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return False
    
    # 检查模型文件
    if not check_model_files(model_path):
        logger.error("模型文件检查失败")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")
    
    # 检查转换脚本
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        logger.error(f"转换脚本不存在: {convert_script}")
        logger.info("请确保llama.cpp已正确克隆")
        return False
    
    # 执行转换
    logger.info("="*60)
    logger.info("开始转换模型...")
    logger.info(f"输入: {model_path}")
    logger.info(f"输出: {output_file}")
    logger.info("="*60)
    
    # 使用uv run来执行转换脚本
    cmd = [
        "uv", "run", "python", convert_script,
        model_path,
        "--outfile", output_file,
        "--outtype", "f16"
    ]
    
    if run_command(cmd):
        logger.info("模型转换成功!")
        
        # 检查输出文件
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024 * 1024)
            logger.info(f"GGUF文件大小: {file_size:.2f} GB")
            logger.info(f"输出文件位置: {os.path.abspath(output_file)}")
            return True
        else:
            logger.error("转换完成但找不到输出文件")
            return False
    else:
        logger.error("模型转换失败")
        return False

def create_requirements_txt():
    """创建requirements.txt文件以备用"""
    requirements_content = """torch>=2.0.0
transformers>=4.30.0
sentencepiece
protobuf
accelerate
safetensors
numpy
huggingface-hub
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    logger.info("已创建requirements.txt文件")

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("开始PyTorch到GGUF的转换过程 (使用uv)")
    logger.info("="*60)
    
    # 0. 创建requirements.txt备用
    create_requirements_txt()
    
    # 1. 设置llama.cpp
    logger.info("步骤 1: 设置llama.cpp环境")
    llama_cpp_dir = setup_llama_cpp()
    if not llama_cpp_dir:
        logger.error("无法设置llama.cpp环境")
        return
    
    # 2. 使用uv安装依赖
    logger.info("步骤 2: 使用uv安装Python依赖")
    if not install_dependencies_with_uv():
        logger.error("依赖安装失败")
        logger.info("尝试手动安装: uv add torch transformers sentencepiece protobuf accelerate safetensors numpy")
        return
    
    # 3. 转换模型
    logger.info("步骤 3: 转换模型")
    if convert_model():
        logger.info("="*60)
        logger.info("🎉 转换完成!")
        logger.info("GGUF模型已保存到: exported_models/qwen3_merged_lightweight/gguf/")
        logger.info("文件名: qwen3-4b-thinking-f16.gguf")
        logger.info("="*60)
    else:
        logger.error("转换失败")
        logger.info("="*60)
        logger.info("手动转换步骤:")
        logger.info("1. 确保已克隆llama.cpp: git clone https://github.com/ggerganov/llama.cpp.git")
        logger.info("2. 安装依赖: uv add torch transformers sentencepiece protobuf accelerate safetensors numpy")
        logger.info("3. 运行转换:")
        logger.info("   uv run python llama.cpp/convert_hf_to_gguf.py \\")
        logger.info("   exported_models/qwen3_merged_lightweight/Qwen_Qwen3-4B-Thinking-2507_pytorch_20250816_225620 \\")
        logger.info("   --outfile exported_models/qwen3_merged_lightweight/gguf/qwen3-4b-thinking-f16.gguf \\")
        logger.info("   --outtype f16")
        logger.info("="*60)

if __name__ == "__main__":
    main()