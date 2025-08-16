"""
修复后的 training_engine.py 中 total_steps 计算逻辑

这个文件展示了如何修复原始代码中发现的问题。
"""

import math
import logging
from typing import Optional, Dict, Any
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class ProgressMonitoringCallback(TrainerCallback):
    """修复后的进度监控回调类"""
    
    def __init__(self, progress_monitor):
        self.progress_monitor = progress_monitor
        self.training_started = False
    
    def _calculate_total_steps(self, args, state, **kwargs) -> int:
        """
        计算总训练步数
        
        Args:
            args: 训练参数
            state: 训练状态
            **kwargs: 其他参数
            
        Returns:
            int: 总训练步数
        """
        try:
            # 优先使用明确设置的 max_steps
            if hasattr(state, 'max_steps') and state.max_steps > 0:
                logger.info(f"使用预设的最大步数: {state.max_steps}")
                return self._validate_total_steps(state.max_steps, args)
            
            # 尝试从数据加载器计算
            train_dataloader = kwargs.get('train_dataloader')
            if train_dataloader is not None:
                return self._calculate_from_dataloader(args, train_dataloader)
            
            # 从数据集估算
            train_dataset = kwargs.get('train_dataset')
            if train_dataset is not None and hasattr(train_dataset, '__len__'):
                return self._calculate_from_dataset(args, train_dataset)
            
            # 最后的备选方案
            return self._get_fallback_total_steps(args)
            
        except Exception as e:
            logger.error(f"计算总步数时发生错误: {e}")
            return self._get_fallback_total_steps(args)
    
    def _calculate_from_dataloader(self, args, train_dataloader) -> int:
        """从数据加载器计算总步数"""
        try:
            dataloader_length = len(train_dataloader)
            logger.info(f"数据加载器长度: {dataloader_length}")
            
            # 考虑梯度累积步数
            gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
            if gradient_accumulation_steps > 1:
                # 梯度累积会减少实际的优化器更新步数
                steps_per_epoch = math.ceil(dataloader_length / gradient_accumulation_steps)
