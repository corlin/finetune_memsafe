"""
模型优化处理器

本模块负责对合并后的模型进行各种优化处理，包括量化、压缩和结构优化。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging
import gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# from transformers.utils.quantization_config import QuantizationConfig  # 可能不存在
import copy

from .export_models import QuantizationLevel
from .export_exceptions import OptimizationError, MemoryError
from .export_utils import ensure_memory_available, get_system_info, format_size


class OptimizationProcessor:
    """模型优化处理组件"""
    
    def __init__(self, device: str = "auto", max_memory_gb: float = 16.0):
        """
        初始化优化处理器
        
        Args:
            device: 设备类型
            max_memory_gb: 最大内存使用限制
        """
        self.logger = logging.getLogger(__name__)
        self.max_memory_gb = max_memory_gb
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger.info(f"优化处理器初始化完成，使用设备: {self.device}")
        
        # 优化统计信息
        self.optimization_stats = {
            'original_size_mb': 0.0,
            'optimized_size_mb': 0.0,
            'size_reduction_mb': 0.0,
            'size_reduction_percentage': 0.0,
            'optimization_steps': []
        }
    
    def apply_quantization(self, model: AutoModelForCausalLM, 
                          quant_level) -> AutoModelForCausalLM:
        """
        应用量化到模型
        
        Args:
            model: 输入模型
            quant_level: 量化级别 (QuantizationLevel枚举或字符串)
            
        Returns:
            AutoModelForCausalLM: 量化后的模型
            
        Raises:
            OptimizationError: 量化失败时抛出
        """
        try:
            # 处理字符串输入，转换为枚举
            if isinstance(quant_level, str):
                try:
                    quant_level = QuantizationLevel(quant_level.lower())
                except ValueError:
                    raise OptimizationError(f"不支持的量化级别字符串: {quant_level}")
            
            self.logger.info(f"开始应用 {quant_level.value} 量化")
            
            # 记录原始大小
            original_size = self._calculate_model_size(model)
            self.optimization_stats['original_size_mb'] = original_size
            
            if quant_level == QuantizationLevel.NONE:
                self.logger.info("跳过量化（级别为NONE）")
                return model
            
            # 检查内存
            estimated_memory = self._estimate_quantization_memory(model, quant_level)
            ensure_memory_available(estimated_memory)
            
            # 应用不同级别的量化
            if quant_level == QuantizationLevel.FP16:
                quantized_model = self._apply_fp16_quantization(model)
            elif quant_level == QuantizationLevel.INT8:
                quantized_model = self._apply_int8_quantization(model)
            elif quant_level == QuantizationLevel.INT4:
                quantized_model = self._apply_int4_quantization(model)
            else:
                raise OptimizationError(f"不支持的量化级别: {quant_level.value}")
            
            # 记录优化后大小
            optimized_size = self._calculate_model_size(quantized_model)
            self.optimization_stats['optimized_size_mb'] = optimized_size
            
            # 计算压缩比
            self._update_compression_stats(original_size, optimized_size, f"{quant_level.value}量化")
            
            self.logger.info(f"{quant_level.value}量化完成，大小从 {format_size(original_size)} 减少到 {format_size(optimized_size)}")
            
            return quantized_model
            
        except Exception as e:
            error_msg = f"量化失败: {str(e)}"
            self.logger.error(error_msg)
            # 安全地获取量化级别名称
            quant_name = quant_level.value if hasattr(quant_level, 'value') else str(quant_level)
            raise OptimizationError(error_msg, optimization_type=quant_name)
    
    def _apply_fp16_quantization(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """应用FP16量化"""
        self.logger.info("应用FP16量化")
        
        # 将模型转换为half精度
        if self.device == "cuda":
            model = model.half()
        else:
            # CPU上使用float32，因为CPU通常不支持half精度的高效计算
            self.logger.warning("CPU设备上跳过FP16量化，保持FP32")
        
        return model
    
    def _apply_int8_quantization(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """应用INT8量化"""
        self.logger.info("应用INT8量化")
        
        try:
            # 使用torch的量化功能
            model.eval()  # 确保模型在评估模式
            
            # 准备量化
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # 校准（这里使用简单的前向传播）
            self._calibrate_model(model)
            
            # 转换为量化模型
            quantized_model = torch.quantization.convert(model, inplace=False)
            
            return quantized_model
            
        except Exception as e:
            self.logger.warning(f"标准INT8量化失败: {e}，尝试简化量化")
            # 回退到简单的权重量化
            return self._apply_simple_int8_quantization(model)
    
    def _apply_int4_quantization(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """应用INT4量化"""
        self.logger.info("应用INT4量化")
        
        # INT4量化通常需要专门的库支持，这里实现一个简化版本
        try:
            # 使用BitsAndBytesConfig进行4bit量化（如果可用）
            if hasattr(model, 'quantize'):
                return model.quantize(bits=4)
            else:
                # 简化的INT4量化实现
                return self._apply_simple_int4_quantization(model)
                
        except Exception as e:
            self.logger.warning(f"INT4量化失败: {e}，回退到INT8量化")
            return self._apply_int8_quantization(model)
    
    def _apply_simple_int8_quantization(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """简化的INT8量化实现"""
        quantized_model = copy.deepcopy(model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # 量化权重
                weight = module.weight.data
                scale = weight.abs().max() / 127.0
                quantized_weight = torch.round(weight / scale).clamp(-128, 127)
                
                # 创建量化参数
                module.weight.data = quantized_weight * scale
                
                # 添加量化信息（用于记录）
                setattr(module, '_quantized', True)
                setattr(module, '_quantization_scale', scale)
        
        return quantized_model
    
    def _apply_simple_int4_quantization(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """简化的INT4量化实现"""
        quantized_model = copy.deepcopy(model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # 量化权重到4bit
                weight = module.weight.data
                scale = weight.abs().max() / 7.0  # 4bit: -8 to 7
                quantized_weight = torch.round(weight / scale).clamp(-8, 7)
                
                # 创建量化参数
                module.weight.data = quantized_weight * scale
                
                # 添加量化信息
                setattr(module, '_quantized', True)
                setattr(module, '_quantization_scale', scale)
                setattr(module, '_quantization_bits', 4)
        
        return quantized_model
    
    def _calibrate_model(self, model: AutoModelForCausalLM):
        """校准量化模型"""
        # 简单的校准过程，使用随机数据
        try:
            with torch.no_grad():
                # 创建随机输入进行校准
                batch_size = 1
                seq_length = 128
                vocab_size = getattr(model.config, 'vocab_size', 32000)
                
                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
                if self.device == "cuda":
                    dummy_input = dummy_input.to(self.device)
                
                # 前向传播进行校准
                _ = model(dummy_input)
                
        except Exception as e:
            self.logger.warning(f"模型校准失败: {e}")
    
    def compress_model_weights(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        压缩模型权重
        
        Args:
            model: 输入模型
            
        Returns:
            AutoModelForCausalLM: 压缩后的模型
        """
        try:
            self.logger.info("开始压缩模型权重")
            
            original_size = self._calculate_model_size(model)
            
            # 更严格的内存检查
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                available_gb = memory_info.available / (1024**3)
                
                if memory_percent > 85:  # 降低阈值到85%
                    self.logger.warning(f"内存使用率过高 ({memory_percent}%)，跳过权重压缩以避免内存不足")
                    return model
                
                if available_gb < 3.0:  # 需要至少3GB可用内存
                    self.logger.warning(f"可用内存不足 ({available_gb:.1f}GB)，跳过权重压缩以避免内存不足")
                    return model
                    
            except ImportError:
                self.logger.warning("psutil未安装，无法检查内存使用情况，跳过权重压缩以确保安全")
                return model
            
            # 强制清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            compressed_model = copy.deepcopy(model)
            
            # 应用权重压缩技术 - 更保守的方法
            try:
                compressed_model = self._apply_weight_pruning(compressed_model)
                # 在剪枝后立即清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                self.logger.warning(f"权重剪枝失败: {e}，跳过剪枝步骤")
            
            try:
                compressed_model = self._apply_weight_sharing(compressed_model)
                # 在权重共享后立即清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                self.logger.warning(f"权重共享失败: {e}，跳过权重共享步骤")
            
            optimized_size = self._calculate_model_size(compressed_model)
            self._update_compression_stats(original_size, optimized_size, "权重压缩")
            
            self.logger.info(f"权重压缩完成，大小从 {format_size(original_size)} 减少到 {format_size(optimized_size)}")
            
            return compressed_model
            
        except Exception as e:
            error_msg = f"权重压缩失败: {str(e)}"
            self.logger.error(error_msg)
            # 如果压缩失败，返回原始模型而不是抛出异常
            self.logger.warning("权重压缩失败，返回原始模型")
            return model
    
    def _apply_weight_pruning(self, model: AutoModelForCausalLM, 
                             pruning_ratio: float = 0.1) -> AutoModelForCausalLM:
        """应用权重剪枝"""
        self.logger.info(f"应用权重剪枝，剪枝比例: {pruning_ratio}")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    weight = module.weight.data
                    
                    # 更严格的内存检查 - 降低阈值到500万参数
                    if weight.numel() > 5_000_000:  # 如果超过500万个参数
                        self.logger.warning(f"模块 {name} 权重过大 ({weight.numel()} 参数)，跳过权重剪枝")
                        continue
                    
                    # 检查权重所需内存
                    estimated_memory_mb = (weight.numel() * 4 * 3) / (1024 * 1024)  # 3倍安全系数
                    if estimated_memory_mb > 300:  # 如果估计需要超过300MB内存
                        self.logger.warning(f"模块 {name} 权重剪枝预计需要 {estimated_memory_mb:.1f}MB 内存，跳过以避免内存不足")
                        continue
                    
                    # 计算权重的绝对值
                    weight_abs = weight.abs()
                    
                    # 确保权重是float或double类型，以便使用quantile函数
                    original_dtype = weight.dtype
                    if weight_abs.dtype not in [torch.float32, torch.float64, torch.float16]:
                        weight_abs = weight_abs.float()
                    elif weight_abs.dtype == torch.float16:
                        # 将float16转换为float32以确保quantile函数正常工作
                        weight_abs = weight_abs.float()
                    
                    # 检查张量大小，如果太大则使用采样方法
                    flattened_weight = weight_abs.flatten()
                    if flattened_weight.numel() > 5_000_000:  # 降低阈值到500万个元素
                        self.logger.warning(f"权重张量过大 ({flattened_weight.numel()} 元素)，使用采样方法计算阈值")
                        # 随机采样一部分权重来计算阈值
                        sample_size = min(500_000, flattened_weight.numel())  # 降低采样大小
                        indices = torch.randperm(flattened_weight.numel())[:sample_size]
                        sampled_weights = flattened_weight[indices]
                        # 确保采样的权重也是正确的数据类型
                        if sampled_weights.dtype not in [torch.float32, torch.float64]:
                            sampled_weights = sampled_weights.float()
                        threshold = torch.quantile(sampled_weights, pruning_ratio)
                        
                        # 清理临时变量
                        del sampled_weights, indices
                    else:
                        # 找到阈值（保留最大的(1-pruning_ratio)比例的权重）
                        threshold = torch.quantile(flattened_weight, pruning_ratio)
                    
                    # 清理临时变量
                    del flattened_weight, weight_abs
                    
                    # 应用剪枝掩码，使用原始权重的绝对值进行比较
                    mask = weight.abs() > threshold.to(weight.device)
                    # 保持原始数据类型
                    module.weight.data = weight * mask.to(original_dtype)
                    
                    # 清理临时变量
                    del mask
                    
                    # 强制清理内存
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    self.logger.warning(f"模块 {name} 的权重剪枝失败: {e}，跳过此模块")
                    continue
        
        return model
    
    def _apply_weight_sharing(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """应用权重共享（聚类）"""
        self.logger.info("应用权重共享")
        
        # 检查系统内存使用情况
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                self.logger.warning(f"系统内存使用率过高 ({memory_percent}%)，跳过权重共享以避免内存不足")
                return model
        except ImportError:
            self.logger.warning("psutil未安装，无法检查系统内存使用情况")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    weight = module.weight.data
                    
                    # 更严格的内存检查 - 降低阈值到1000万参数
                    if weight.numel() > 10_000_000:  # 如果超过1000万个参数
                        self.logger.warning(f"模块 {name} 权重过大 ({weight.numel()} 参数)，跳过权重共享")
                        continue
                    
                    # 检查权重所需内存 (参数数量 * 4字节 * 2倍安全系数)
                    estimated_memory_mb = (weight.numel() * 4 * 2) / (1024 * 1024)
                    if estimated_memory_mb > 500:  # 如果估计需要超过500MB内存
                        self.logger.warning(f"模块 {name} 权重共享预计需要 {estimated_memory_mb:.1f}MB 内存，跳过以避免内存不足")
                        continue
                    
                    # 简单的K-means聚类进行权重共享
                    # 这里使用简化版本，将权重量化到固定的几个值
                    num_clusters = min(128, weight.numel() // 8)  # 进一步限制聚类数量
                    
                    if num_clusters > 1:
                        # 使用简单的均匀量化作为权重共享的近似
                        weight_min = weight.min()
                        weight_max = weight.max()
                        
                        # 避免除零错误
                        if weight_max == weight_min:
                            self.logger.debug(f"模块 {name} 权重值相同，跳过权重共享")
                            continue
                        
                        # 创建聚类中心 - 使用更小的张量避免内存问题
                        step = (weight_max - weight_min) / (num_clusters - 1)
                        centers = torch.arange(num_clusters, dtype=weight.dtype, device=weight.device) * step + weight_min
                        
                        # 分块处理权重以避免内存不足
                        chunk_size = min(1000000, weight.numel())  # 每次处理100万个元素
                        weight_flat = weight.flatten()
                        
                        for i in range(0, weight_flat.numel(), chunk_size):
                            end_idx = min(i + chunk_size, weight_flat.numel())
                            chunk = weight_flat[i:end_idx]
                            
                            # 将权重分配到最近的聚类中心
                            distances = torch.abs(chunk.unsqueeze(-1) - centers.unsqueeze(0))
                            cluster_indices = torch.argmin(distances, dim=-1)
                            
                            # 用聚类中心替换原始权重
                            weight_flat[i:end_idx] = centers[cluster_indices]
                        
                        # 重新整形回原始形状
                        module.weight.data = weight_flat.reshape(weight.shape)
                        
                        # 强制清理内存
                        del weight_flat, distances, cluster_indices
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except Exception as e:
                    self.logger.warning(f"模块 {name} 的权重共享失败: {e}，跳过此模块")
                    continue
        
        return model
    
    def remove_training_artifacts(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        移除训练时的冗余参数和元数据
        
        Args:
            model: 输入模型
            
        Returns:
            AutoModelForCausalLM: 清理后的模型
        """
        try:
            self.logger.info("开始移除训练artifacts")
            
            original_size = self._calculate_model_size(model)
            
            # 确保模型在评估模式
            model.eval()
            
            # 移除训练相关的状态
            cleaned_model = self._remove_training_states(model)
            
            # 移除不必要的缓冲区
            cleaned_model = self._remove_unnecessary_buffers(cleaned_model)
            
            # 清理模型配置中的训练相关信息
            cleaned_model = self._clean_model_config(cleaned_model)
            
            optimized_size = self._calculate_model_size(cleaned_model)
            self._update_compression_stats(original_size, optimized_size, "移除训练artifacts")
            
            self.logger.info(f"训练artifacts移除完成，大小从 {format_size(original_size)} 减少到 {format_size(optimized_size)}")
            
            return cleaned_model
            
        except Exception as e:
            error_msg = f"移除训练artifacts失败: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg, optimization_type="移除训练artifacts")
    
    def _remove_training_states(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """移除训练状态"""
        # 创建模型副本
        cleaned_model = copy.deepcopy(model)
        
        # 移除所有模块的训练状态
        for module in cleaned_model.modules():
            # 移除dropout的训练状态
            if hasattr(module, 'training'):
                module.training = False
            
            # 移除batch normalization的运行统计信息（如果有）
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(module, 'num_batches_tracked'):
                    delattr(module, 'num_batches_tracked')
        
        return cleaned_model
    
    def _remove_unnecessary_buffers(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """移除不必要的缓冲区"""
        # 移除一些可能不需要的缓冲区
        buffers_to_remove = [
            'position_ids',  # 位置编码缓冲区
            'attention_mask',  # 注意力掩码缓冲区
        ]
        
        for name, buffer in list(model.named_buffers()):
            buffer_name = name.split('.')[-1]
            if buffer_name in buffers_to_remove:
                # 获取包含该缓冲区的模块
                module_path = name.rsplit('.', 1)[0] if '.' in name else ''
                if module_path:
                    module = model.get_submodule(module_path)
                    if hasattr(module, buffer_name):
                        delattr(module, buffer_name)
                        self.logger.debug(f"移除缓冲区: {name}")
        
        return model
    
    def _clean_model_config(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """清理模型配置"""
        if hasattr(model, 'config'):
            config = model.config
            
            # 移除训练相关的配置
            training_configs = [
                'gradient_checkpointing',
                'use_cache',  # 推理时可能不需要
                'output_attentions',
                'output_hidden_states',
                'return_dict'
            ]
            
            for config_name in training_configs:
                if hasattr(config, config_name):
                    # 设置为推理优化的值
                    if config_name == 'use_cache':
                        setattr(config, config_name, True)  # 推理时启用缓存
                    else:
                        setattr(config, config_name, False)
        
        return model
    
    def optimize_model_structure(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        优化模型结构以减少内存占用
        
        Args:
            model: 输入模型
            
        Returns:
            AutoModelForCausalLM: 结构优化后的模型
        """
        try:
            self.logger.info("开始优化模型结构")
            
            original_size = self._calculate_model_size(model)
            optimized_model = copy.deepcopy(model)
            
            # 应用结构优化
            optimized_model = self._optimize_attention_layers(optimized_model)
            optimized_model = self._optimize_feed_forward_layers(optimized_model)
            
            optimized_size = self._calculate_model_size(optimized_model)
            self._update_compression_stats(original_size, optimized_size, "结构优化")
            
            self.logger.info(f"结构优化完成，大小从 {format_size(original_size)} 减少到 {format_size(optimized_size)}")
            
            return optimized_model
            
        except Exception as e:
            error_msg = f"结构优化失败: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg, optimization_type="结构优化")
    
    def _optimize_attention_layers(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """优化注意力层"""
        # 这里可以实现注意力层的优化，比如：
        # - 合并多头注意力的权重矩阵
        # - 优化注意力计算的内存使用
        
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                # 简单的优化：确保注意力层使用最优的数据类型
                if hasattr(module, 'weight') and module.weight.dtype == torch.float32:
                    if self.device == "cuda":
                        module.weight.data = module.weight.data.half()
        
        return model
    
    def _optimize_feed_forward_layers(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """优化前馈层"""
        # 优化前馈网络层
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name.lower():
                # 简单的优化：确保使用最优的数据类型
                if module.weight.dtype == torch.float32 and self.device == "cuda":
                    module.weight.data = module.weight.data.half()
                    if module.bias is not None:
                        module.bias.data = module.bias.data.half()
        
        return model
    
    def calculate_size_reduction(self, original_size: float, 
                               optimized_size: float) -> Dict[str, float]:
        """
        计算大小减少统计信息
        
        Args:
            original_size: 原始大小（MB）
            optimized_size: 优化后大小（MB）
            
        Returns:
            Dict[str, float]: 大小减少统计信息
        """
        size_reduction = original_size - optimized_size
        reduction_percentage = (size_reduction / original_size * 100) if original_size > 0 else 0
        
        return {
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'size_reduction_mb': size_reduction,
            'size_reduction_percentage': reduction_percentage,
            'compression_ratio': original_size / optimized_size if optimized_size > 0 else 1.0
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        获取优化报告
        
        Returns:
            Dict[str, Any]: 优化报告
        """
        return {
            'optimization_stats': self.optimization_stats.copy(),
            'system_info': get_system_info(),
            'device': self.device,
            'max_memory_gb': self.max_memory_gb
        }
    
    def _calculate_model_size(self, model: AutoModelForCausalLM) -> float:
        """计算模型大小（MB）"""
        total_size = 0
        
        for param in model.parameters():
            param_size = param.numel() * param.element_size()
            total_size += param_size
        
        # 添加缓冲区大小
        for buffer in model.buffers():
            buffer_size = buffer.numel() * buffer.element_size()
            total_size += buffer_size
        
        return total_size / (1024 * 1024)  # 转换为MB
    
    def _estimate_quantization_memory(self, model: AutoModelForCausalLM, 
                                    quant_level: QuantizationLevel) -> float:
        """估算量化过程的内存需求（返回GB）"""
        model_size_mb = self._calculate_model_size(model)
        
        # 量化过程可能需要额外的内存来存储临时数据
        if quant_level == QuantizationLevel.FP16:
            additional_memory_mb = model_size_mb * 0.1  # FP16转换需要较少额外内存
        elif quant_level == QuantizationLevel.INT8:
            additional_memory_mb = model_size_mb * 0.5  # INT8量化需要中等额外内存
        elif quant_level == QuantizationLevel.INT4:
            additional_memory_mb = model_size_mb * 0.8  # INT4量化需要更多额外内存
        else:
            additional_memory_mb = 0.0
        
        # 转换为GB并返回
        return additional_memory_mb / 1024
    
    def _update_compression_stats(self, original_size: float, optimized_size: float, step_name: str):
        """更新压缩统计信息"""
        reduction = original_size - optimized_size
        reduction_percentage = (reduction / original_size * 100) if original_size > 0 else 0
        
        self.optimization_stats['optimization_steps'].append({
            'step': step_name,
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'size_reduction_mb': reduction,
            'size_reduction_percentage': reduction_percentage
        })
        
        # 更新总体统计
        if self.optimization_stats['original_size_mb'] == 0:
            self.optimization_stats['original_size_mb'] = original_size
        
        self.optimization_stats['optimized_size_mb'] = optimized_size
        self.optimization_stats['size_reduction_mb'] = (
            self.optimization_stats['original_size_mb'] - optimized_size
        )
        self.optimization_stats['size_reduction_percentage'] = (
            (self.optimization_stats['size_reduction_mb'] / self.optimization_stats['original_size_mb'] * 100)
            if self.optimization_stats['original_size_mb'] > 0 else 0
        )
    
    def cleanup(self):
        """清理内存"""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info("优化处理器内存清理完成")