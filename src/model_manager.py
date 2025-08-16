"""
模型管理器 - 处理带4位量化的模型加载和配置
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """管理模型加载、量化和分词器配置的类"""
    
    def __init__(self, max_memory_gb: float = 13.0):
        """
        初始化ModelManager
        
        Args:
            max_memory_gb: 最大GPU内存限制（GB）
        """
        self.max_memory_gb = max_memory_gb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def configure_quantization(self) -> BitsAndBytesConfig:
        """
        创建优化的4位量化配置
        
        Returns:
            BitsAndBytesConfig: 配置好的4位量化设置
        """
        logger.info("配置4位量化设置")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        logger.info("4位量化配置完成")
        return quantization_config
    
    def _configure_device_map(self) -> dict:
        """
        配置设备映射以优化内存使用
        
        Returns:
            dict: 设备映射配置
        """
        if self.device == "cuda":
            # 自动设备映射，让transformers自动分配
            return "auto"
        else:
            return {"": "cpu"}
    
    def load_model_with_quantization(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载带量化和设备映射的模型
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: 加载的模型和分词器
        """
        logger.info(f"开始加载模型: {model_name}")
        
        try:
            # 配置量化
            quantization_config = self.configure_quantization()
            device_map = self._configure_device_map()
            
            # 加载模型
            logger.info("加载量化模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=None  # 使用默认注意力实现，避免flash_attention_2依赖问题
            )
            
            # 加载分词器
            logger.info("加载分词器...")
            tokenizer = self._setup_tokenizer(model_name)
            
            logger.info(f"模型加载完成，设备: {model.device}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise RuntimeError(f"无法加载模型 {model_name}: {str(e)}")
    
    def _setup_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        设置带适当pad token配置的分词器
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            AutoTokenizer: 配置好的分词器
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 配置pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("使用EOS token作为pad token")
            else:
                # 如果没有EOS token，添加一个特殊的pad token
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                logger.info("添加新的pad token")
        
        logger.info(f"分词器配置完成，词汇表大小: {len(tokenizer)}")
        return tokenizer
    
    def prepare_for_training(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        为训练准备模型
        
        Args:
            model: 要准备的模型
            
        Returns:
            AutoModelForCausalLM: 准备好的模型
        """
        logger.info("为训练准备模型...")
        
        # 启用梯度检查点以节省内存
        model.gradient_checkpointing_enable()
        
        # 确保模型在训练模式
        model.train()
        
        logger.info("模型训练准备完成")
        return model
    
    def get_model_info(self, model: AutoModelForCausalLM) -> dict:
        """
        获取模型信息
        
        Args:
            model: 模型实例
            
        Returns:
            dict: 模型信息
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0,
            "device": str(model.device),
            "dtype": str(model.dtype) if hasattr(model, 'dtype') else "unknown"
        }
        
        logger.info(f"模型信息: {info}")
        return info
