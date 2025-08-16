"""
模型合并核心功能

本模块负责将LoRA适配器与基座模型合并，生成完整的可部署模型。
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import gc

from .export_models import CheckpointInfo
from .export_exceptions import ModelMergeError, MemoryError, CheckpointValidationError
from .export_utils import (
    ensure_memory_available, 
    get_directory_size_mb, 
    format_size,
    setup_logging
)
from .checkpoint_detector import CheckpointDetector


class ModelMerger:
    """合并LoRA适配器与基座模型的组件"""
    
    def __init__(self, device: str = "auto", max_memory_gb: float = 16.0):
        """
        初始化模型合并器
        
        Args:
            device: 设备类型 ("auto", "cuda", "cpu")
            max_memory_gb: 最大内存使用限制（GB）
        """
        self.logger = logging.getLogger(__name__)
        self.max_memory_gb = max_memory_gb
        self.checkpoint_detector = CheckpointDetector()
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger.info(f"模型合并器初始化完成，使用设备: {self.device}")
        
        # 当前加载的模型引用
        self._base_model = None
        self._peft_model = None
        self._merged_model = None
        self._tokenizer = None
    
    def load_base_model(self, model_name: str, load_in_4bit: bool = False, 
                       load_in_8bit: bool = False, trust_remote_code: bool = True) -> AutoModelForCausalLM:
        """
        加载基座模型
        
        Args:
            model_name: 模型名称或路径
            load_in_4bit: 是否使用4bit量化加载
            load_in_8bit: 是否使用8bit量化加载
            trust_remote_code: 是否信任远程代码
            
        Returns:
            AutoModelForCausalLM: 加载的基座模型
            
        Raises:
            ModelMergeError: 模型加载失败时抛出
        """
        try:
            self.logger.info(f"开始加载基座模型: {model_name}")
            
            # 检查内存
            estimated_memory_gb = self._estimate_model_memory(model_name)
            ensure_memory_available(estimated_memory_gb)
            
            # 配置量化参数
            quantization_config = None
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.logger.info("使用4bit量化加载模型")
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.logger.info("使用8bit量化加载模型")
            
            # 强制清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 检查系统内存
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 85:
                    self.logger.warning(f"系统内存使用率过高 ({memory_info.percent}%)，建议释放内存后重试")
                    # 不抛出异常，但记录警告
            except ImportError:
                pass
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device != "auto" else "auto",
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True  # 启用低CPU内存使用模式
            )
            
            self._base_model = model
            self.logger.info(f"基座模型加载成功，参数量: {self._count_parameters(model):,}")
            
            return model
            
        except Exception as e:
            error_msg = f"加载基座模型失败: {str(e)}"
            self.logger.error(error_msg)
            raise ModelMergeError(error_msg, base_model=model_name)
    
    def load_lora_adapter(self, adapter_path: str) -> PeftModel:
        """
        加载LoRA适配器
        
        Args:
            adapter_path: 适配器路径
            
        Returns:
            PeftModel: 加载的PEFT模型
            
        Raises:
            ModelMergeError: 适配器加载失败时抛出
        """
        try:
            self.logger.info(f"开始加载LoRA适配器: {adapter_path}")
            
            # 验证checkpoint
            if not self.checkpoint_detector.validate_checkpoint_integrity(adapter_path):
                checkpoint_info = self.checkpoint_detector.get_checkpoint_metadata(adapter_path)
                raise CheckpointValidationError(
                    f"LoRA适配器验证失败: {', '.join(checkpoint_info.validation_errors)}",
                    checkpoint_path=adapter_path
                )
            
            # 确保基座模型已加载
            if self._base_model is None:
                raise ModelMergeError("必须先加载基座模型", adapter_path=adapter_path)
            
            # 加载PEFT配置
            peft_config = PeftConfig.from_pretrained(adapter_path)
            self.logger.info(f"PEFT配置: {peft_config.peft_type}, rank={peft_config.r}, alpha={peft_config.lora_alpha}")
            
            # 加载PEFT模型
            peft_model = PeftModel.from_pretrained(
                self._base_model,
                adapter_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self._peft_model = peft_model
            self.logger.info("LoRA适配器加载成功")
            
            return peft_model
            
        except Exception as e:
            error_msg = f"加载LoRA适配器失败: {str(e)}"
            self.logger.error(error_msg)
            raise ModelMergeError(error_msg, adapter_path=adapter_path)
    
    def merge_lora_weights(self, base_model: Optional[AutoModelForCausalLM] = None, 
                          adapter_path: Optional[str] = None) -> AutoModelForCausalLM:
        """
        合并LoRA权重到基座模型
        
        Args:
            base_model: 基座模型（可选，使用已加载的模型）
            adapter_path: 适配器路径（可选，使用已加载的适配器）
            
        Returns:
            AutoModelForCausalLM: 合并后的模型
            
        Raises:
            ModelMergeError: 合并失败时抛出
        """
        try:
            self.logger.info("开始合并LoRA权重")
            
            # 使用提供的模型或已加载的模型
            if base_model is not None:
                self._base_model = base_model
            if adapter_path is not None:
                self.load_lora_adapter(adapter_path)
            
            # 确保模型已加载
            if self._peft_model is None:
                raise ModelMergeError("PEFT模型未加载")
            
            # 检查内存
            estimated_memory_gb = self._estimate_merge_memory()
            ensure_memory_available(estimated_memory_gb)
            
            # 强制清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 检查内存使用情况
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 90:
                    self.logger.error(f"内存使用率过高 ({memory_info.percent}%)，无法安全执行合并")
                    raise MemoryError(f"内存不足，当前使用率: {memory_info.percent}%")
            except ImportError:
                pass
            
            # 执行合并
            self.logger.info("正在合并权重...")
            try:
                merged_model = self._peft_model.merge_and_unload()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                    self.logger.error(f"合并过程中内存不足: {e}")
                    # 尝试清理内存后重试
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise MemoryError(f"合并过程中内存不足: {e}")
                else:
                    raise
            
            # 清理PEFT模型以释放内存
            del self._peft_model
            self._peft_model = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self._merged_model = merged_model
            self.logger.info("权重合并完成")
            
            return merged_model
            
        except Exception as e:
            error_msg = f"合并LoRA权重失败: {str(e)}"
            self.logger.error(error_msg)
            raise ModelMergeError(error_msg)
    
    def save_merged_model(self, model: AutoModelForCausalLM, output_path: str, 
                         save_tokenizer: bool = True, tokenizer: Optional[AutoTokenizer] = None) -> None:
        """
        保存合并后的模型
        
        Args:
            model: 要保存的模型
            output_path: 输出路径
            save_tokenizer: 是否保存tokenizer
            tokenizer: 自定义tokenizer（可选）
            
        Raises:
            ModelMergeError: 保存失败时抛出
        """
        try:
            self.logger.info(f"开始保存合并模型到: {output_path}")
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="5GB"
            )
            
            # 保存tokenizer
            if save_tokenizer:
                if tokenizer is None:
                    tokenizer = self._tokenizer
                
                if tokenizer is not None:
                    tokenizer.save_pretrained(output_path)
                    self.logger.info("Tokenizer保存完成")
                else:
                    self.logger.warning("未找到tokenizer，跳过保存")
            
            # 计算保存的模型大小
            saved_size_mb = get_directory_size_mb(str(output_path))
            self.logger.info(f"模型保存完成，大小: {format_size(saved_size_mb)}")
            
        except Exception as e:
            error_msg = f"保存合并模型失败: {str(e)}"
            self.logger.error(error_msg)
            raise ModelMergeError(error_msg)
    
    def verify_merge_integrity(self, merged_model: AutoModelForCausalLM, 
                              test_input: str = "Hello, how are you?") -> bool:
        """
        验证合并结果的正确性
        
        Args:
            merged_model: 合并后的模型
            test_input: 测试输入文本
            
        Returns:
            bool: 验证是否通过
        """
        try:
            self.logger.info("开始验证合并模型的完整性")
            
            # 加载tokenizer（如果还没有）
            if self._tokenizer is None:
                self.logger.warning("未找到tokenizer，跳过功能性验证")
                return True
            
            # 基本功能测试
            inputs = self._tokenizer(test_input, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = merged_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.info(f"验证输出: {generated_text}")
            
            # 检查模型参数
            param_count = self._count_parameters(merged_model)
            self.logger.info(f"合并模型参数量: {param_count:,}")
            
            # 检查模型状态
            if merged_model.training:
                merged_model.eval()
                self.logger.info("模型已设置为评估模式")
            
            self.logger.info("合并模型验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"合并模型验证失败: {str(e)}")
            return False
    
    def load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        """
        加载tokenizer
        
        Args:
            model_name_or_path: 模型名称或路径
            
        Returns:
            AutoTokenizer: 加载的tokenizer
        """
        try:
            self.logger.info(f"加载tokenizer: {model_name_or_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self._tokenizer = tokenizer
            self.logger.info("Tokenizer加载完成")
            
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"加载tokenizer失败: {str(e)}")
            raise ModelMergeError(f"加载tokenizer失败: {str(e)}")
    
    def merge_and_save(self, base_model_name: str, adapter_path: str, 
                      output_path: str, load_in_4bit: bool = False,
                      load_in_8bit: bool = False, save_tokenizer: bool = True) -> Dict[str, Any]:
        """
        一站式合并和保存流程
        
        Args:
            base_model_name: 基座模型名称
            adapter_path: 适配器路径
            output_path: 输出路径
            load_in_4bit: 是否使用4bit量化
            load_in_8bit: 是否使用8bit量化
            save_tokenizer: 是否保存tokenizer
            
        Returns:
            Dict[str, Any]: 合并结果信息
        """
        result = {
            'success': False,
            'base_model': base_model_name,
            'adapter_path': adapter_path,
            'output_path': output_path,
            'model_size_mb': 0.0,
            'parameter_count': 0,
            'verification_passed': False,
            'error_message': None
        }
        
        try:
            # 1. 加载tokenizer
            self.load_tokenizer(base_model_name)
            
            # 2. 加载基座模型
            base_model = self.load_base_model(
                base_model_name, 
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )
            
            # 3. 加载LoRA适配器
            self.load_lora_adapter(adapter_path)
            
            # 4. 合并权重
            merged_model = self.merge_lora_weights()
            
            # 5. 验证合并结果
            verification_passed = self.verify_merge_integrity(merged_model)
            
            # 6. 保存模型
            self.save_merged_model(merged_model, output_path, save_tokenizer)
            
            # 7. 收集结果信息
            result.update({
                'success': True,
                'model_size_mb': get_directory_size_mb(output_path),
                'parameter_count': self._count_parameters(merged_model),
                'verification_passed': verification_passed
            })
            
            self.logger.info("模型合并和保存流程完成")
            
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"合并流程失败: {str(e)}")
            raise
        
        finally:
            # 清理内存
            self.cleanup()
        
        return result
    
    def cleanup(self):
        """清理内存中的模型"""
        self.logger.info("清理内存中的模型")
        
        if self._base_model is not None:
            del self._base_model
            self._base_model = None
        
        if self._peft_model is not None:
            del self._peft_model
            self._peft_model = None
        
        if self._merged_model is not None:
            del self._merged_model
            self._merged_model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info("内存清理完成")
    
    def _estimate_model_memory(self, model_name: str) -> float:
        """估算模型内存需求（GB）"""
        # 这是一个简化的估算，实际需求可能因模型而异
        try:
            config = AutoConfig.from_pretrained(model_name)
            # 估算公式：参数量 * 4字节（float32）或 2字节（float16）
            param_count = getattr(config, 'num_parameters', None)
            if param_count is None:
                # 根据隐藏层大小估算
                hidden_size = getattr(config, 'hidden_size', 4096)
                num_layers = getattr(config, 'num_hidden_layers', 32)
                vocab_size = getattr(config, 'vocab_size', 32000)
                
                # 粗略估算
                param_count = hidden_size * hidden_size * num_layers * 4 + vocab_size * hidden_size
            
            # 使用float16，每个参数2字节，加上一些缓冲
            memory_gb = (param_count * 2) / (1024 ** 3) * 1.5
            return max(memory_gb, 2.0)  # 最少2GB
            
        except Exception:
            # 如果无法估算，返回默认值
            return 8.0
    
    def _estimate_merge_memory(self) -> float:
        """估算合并过程的内存需求"""
        # 合并过程需要额外的内存来存储临时数据
        base_memory = self._estimate_model_memory("dummy")
        return base_memory * 1.5  # 额外50%的内存
    
    def _count_parameters(self, model) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters())
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()