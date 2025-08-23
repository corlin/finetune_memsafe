"""
LoRA适配器 - 处理参数高效微调配置
"""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class LoRAAdapter:
    """管理LoRA配置和应用的类"""
    
    def __init__(self, r: int = 6, alpha: int = 12, dropout: float = 0.1):
        """
        初始化LoRA适配器
        
        Args:
            r: LoRA rank，控制适配器大小
            alpha: LoRA alpha，控制适配器强度
            dropout: dropout率
        """
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        
    def create_lora_config(self, target_modules: list = None) -> LoraConfig:
        """
        创建内存优化的LoRA配置
        
        Args:
            target_modules: 目标模块列表，如果为None则使用默认值
            
        Returns:
            LoraConfig: 配置好的LoRA设置
        """
        logger.info(f"创建LoRA配置 - r={self.r}, alpha={self.alpha}, dropout={self.dropout}")
        
        # 如果没有指定目标模块，使用Qwen3的默认模块
        if target_modules is None:
            target_modules = [
                "q_proj", "v_proj"#, "ffn.w1"
                #"k_proj", 
                #"o_proj",
                #"gate_proj",
                #"up_proj",
                #"down_proj"
            ]
        
        lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
            use_rslora=True,  # 使用RSLoRA以提高稳定性
            use_dora=False    # 不使用DoRA以节省内存
        )
        
        logger.info(f"LoRA配置创建完成，目标模块: {target_modules}")
        return lora_config
    
    def prepare_model_for_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        为LoRA训练准备量化模型
        
        Args:
            model: 量化后的模型
            
        Returns:
            AutoModelForCausalLM: 准备好的模型
        """
        logger.info("为LoRA训练准备模型...")
        
        try:
            # 为k-bit训练准备模型
            model = prepare_model_for_kbit_training(model)
            
            # 启用梯度检查点
            model.gradient_checkpointing_enable()
            
            logger.info("模型LoRA准备完成")
            return model
            
        except Exception as e:
            logger.error(f"模型LoRA准备失败: {str(e)}")
            raise RuntimeError(f"无法为LoRA准备模型: {str(e)}")
    
    def apply_lora(self, model: AutoModelForCausalLM, lora_config: LoraConfig = None) -> AutoModelForCausalLM:
        """
        将LoRA应用到带梯度检查点的量化模型
        
        Args:
            model: 准备好的模型
            lora_config: LoRA配置，如果为None则创建默认配置
            
        Returns:
            AutoModelForCausalLM: 应用LoRA后的模型
        """
        logger.info("应用LoRA适配器...")
        
        try:
            # 如果没有提供配置，创建默认配置
            if lora_config is None:
                lora_config = self.create_lora_config()
            
            # 应用LoRA
            model = get_peft_model(model, lora_config)
            
            # 确保模型在训练模式
            model.train()
            
            logger.info("LoRA适配器应用完成")
            return model
            
        except Exception as e:
            logger.error(f"LoRA应用失败: {str(e)}")
            raise RuntimeError(f"无法应用LoRA: {str(e)}")
    
    def get_trainable_params_info(self, model: AutoModelForCausalLM) -> Dict[str, Any]:
        """
        获取可训练参数计数和报告
        
        Args:
            model: 模型实例
            
        Returns:
            Dict[str, Any]: 参数信息字典
        """
        logger.info("计算可训练参数信息...")
        
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        # 获取LoRA特定信息
        lora_params = 0
        lora_modules = []
        
        if hasattr(model, 'peft_config'):
            for name, module in model.named_modules():
                if 'lora' in name.lower():
                    lora_modules.append(name)
                    for param in module.parameters():
                        if param.requires_grad:
                            lora_params += param.numel()
        
        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_parameters": lora_params,
            "trainable_percentage": round(trainable_percentage, 4),
            "lora_percentage": round((lora_params / total_params) * 100, 4) if total_params > 0 else 0,
            "lora_modules_count": len(lora_modules),
            "lora_modules": lora_modules[:10],  # 只显示前10个模块
            "memory_efficient": trainable_percentage < 5.0  # 如果可训练参数少于5%则认为内存高效
        }
        
        # 记录详细信息
        logger.info(f"参数统计:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  LoRA参数: {lora_params:,}")
        logger.info(f"  可训练比例: {trainable_percentage:.4f}%")
        logger.info(f"  LoRA比例: {info['lora_percentage']:.4f}%")
        logger.info(f"  内存高效: {info['memory_efficient']}")
        
        return info
    
    def setup_lora_for_model(self, model: AutoModelForCausalLM, target_modules: list = None) -> AutoModelForCausalLM:
        """
        为模型完整设置LoRA（组合方法）
        
        Args:
            model: 量化后的模型
            target_modules: 目标模块列表
            
        Returns:
            AutoModelForCausalLM: 完全配置好的LoRA模型
        """
        logger.info("开始完整LoRA设置...")
        
        # 1. 准备模型
        model = self.prepare_model_for_lora(model)
        
        # 2. 创建LoRA配置
        lora_config = self.create_lora_config(target_modules)
        
        # 3. 应用LoRA
        model = self.apply_lora(model, lora_config)
        
        # 4. 获取参数信息
        params_info = self.get_trainable_params_info(model)
        
        logger.info("完整LoRA设置完成")
        logger.info(f"最终模型可训练参数: {params_info['trainable_parameters']:,} ({params_info['trainable_percentage']:.4f}%)")
        
        return model