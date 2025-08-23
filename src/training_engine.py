"""
训练引擎 - 处理带优化参数的模型训练

提供内存高效的训练配置，包括梯度累积、优化器配置和混合精度训练。
支持带监控的训练循环和错误恢复机制。
"""

import os
import json
import math
import torch
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset
from peft import PeftModel

try:
    from .memory_optimizer import MemoryOptimizer, MemoryStatus
    from .logging_system import LoggingSystem, TrainingMetrics
    from .progress_monitor import ProgressMonitor
except ImportError:
    from memory_optimizer import MemoryOptimizer, MemoryStatus
    from logging_system import LoggingSystem, TrainingMetrics
    from progress_monitor import ProgressMonitor

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    output_dir: str = "./qwen3-finetuned"
    max_memory_gb: float = 15.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 10
    max_sequence_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0  # 设为0以避免多进程问题
    save_total_limit: int = 3  # 限制保存的检查点数量


@dataclass
class EpochMemoryMetrics:
    """轮次内存指标数据类"""
    epoch: int
    start_memory_gb: float
    end_memory_gb: float
    peak_memory_gb: float
    memory_delta_gb: float
    cleanup_count: int
    timestamp: datetime
    
    @property
    def memory_efficiency(self) -> float:
        """计算内存效率（越低越好）"""
        if self.start_memory_gb == 0:
            return 0.0
        return self.memory_delta_gb / self.start_memory_gb


class TrainingEngine:
    """
    训练引擎类
    
    负责创建优化的训练参数、配置训练器和执行训练过程。
    包含内存监控和错误恢复功能。
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None, 
                 memory_optimizer: Optional[MemoryOptimizer] = None,
                 logging_system: Optional[LoggingSystem] = None,
                 progress_monitor: Optional[ProgressMonitor] = None):
        """
        初始化训练引擎
        
        Args:
            config: 训练配置，如果为None则使用默认配置
            memory_optimizer: 内存优化器，如果为None则创建新实例
            logging_system: 日志记录系统，如果为None则创建新实例
            progress_monitor: 进度监控器，如果为None则创建新实例
        """
        self.config = config or TrainingConfig()
        self.memory_optimizer = memory_optimizer or MemoryOptimizer(
            max_memory_gb=self.config.max_memory_gb
        )
        
        # 确保输出目录存在
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化日志记录系统
        if logging_system is None:
            log_dir = os.path.join(self.config.output_dir, "logs")
            self.logging_system = LoggingSystem(log_dir=log_dir)
        else:
            self.logging_system = logging_system
        
        # 初始化进度监控器
        if progress_monitor is None:
            self.progress_monitor = ProgressMonitor(
                memory_optimizer=self.memory_optimizer,
                logging_system=self.logging_system,
                enable_rich_display=True
            )
        else:
            self.progress_monitor = progress_monitor
        
        logger.info(f"训练引擎初始化完成，输出目录: {self.config.output_dir}")
        self.logging_system.info("训练引擎初始化完成", "TRAINING_ENGINE", 
                                {"output_dir": self.config.output_dir})
    
    def create_training_args(self, eval_dataset: Optional[Dataset] = None) -> TrainingArguments:
        """
        创建内存高效设置的TrainingArguments函数
        
        Args:
            eval_dataset: 可选的评估数据集，用于确定是否启用评估
            
        Returns:
            TrainingArguments: 优化的训练参数
        """
        logger.info("创建训练参数...")
        
        # 计算有效批次大小
        effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
        logger.info(f"有效批次大小: {effective_batch_size} "
                   f"(per_device: {self.config.batch_size}, "
                   f"accumulation: {self.config.gradient_accumulation_steps})")
        
        # 只有在有评估数据集时才启用评估
        has_eval_dataset = eval_dataset is not None
        
        training_args = TrainingArguments(
            # 基本设置
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # 优化器设置 - 使用paged_adamw_8bit以提高内存效率
            optim="paged_adamw_8bit",
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            
            # 学习率调度
            lr_scheduler_type="cosine",
            warmup_ratio=self.config.warmup_ratio,
            
            # 混合精度训练 - 使用BF16以提高稳定性
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=False,  # 不使用FP16，BF16更稳定
            
            # 内存优化设置
            gradient_checkpointing=True,  # 启用梯度检查点
            dataloader_pin_memory=False,  # 禁用pin_memory以节省内存
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # 日志和保存设置
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,  # 限制保存的检查点数量
            
            # 评估设置 - 只有在有评估数据集时才启用
            eval_strategy="steps" if has_eval_dataset and self.config.eval_steps > 0 else "no",
            eval_steps=self.config.eval_steps if has_eval_dataset and self.config.eval_steps > 0 else None,
            
            # 其他优化设置
            remove_unused_columns=True,  # 移除未使用的列以节省内存
            load_best_model_at_end=has_eval_dataset,
            metric_for_best_model="eval_loss" if has_eval_dataset else None,
            greater_is_better=False,
            
            # 报告设置
            report_to=["tensorboard"],  # 使用TensorBoard记录
            run_name=f"qwen3-finetuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
            # 禁用一些可能导致内存问题的功能
            push_to_hub=False,
            hub_model_id=None,
            
            # 设置种子以确保可重现性
            seed=42,
            data_seed=42,
        )
        
        logger.info("训练参数创建完成")
        logger.info(f"  优化器: {training_args.optim}")
        logger.info(f"  混合精度: BF16={training_args.bf16}, FP16={training_args.fp16}")
        logger.info(f"  梯度检查点: {training_args.gradient_checkpointing}")
        logger.info(f"  学习率: {training_args.learning_rate}")
        logger.info(f"  权重衰减: {training_args.weight_decay}")
        
        return training_args
    
    def create_trainer(self, 
                      model: PreTrainedModel,
                      train_dataset: Dataset,
                      tokenizer: PreTrainedTokenizer,
                      data_collator: Any,
                      eval_dataset: Optional[Dataset] = None) -> Trainer:
        """
        创建带数据整理器和内存监控的Trainer实例
        
        Args:
            model: 训练模型
            train_dataset: 训练数据集
            tokenizer: 分词器
            data_collator: 数据整理器
            eval_dataset: 可选的评估数据集
            
        Returns:
            Trainer: 配置好的训练器
        """
        logger.info("创建训练器...")
        
        # 创建训练参数
        training_args = self.create_training_args(eval_dataset)
        
        # 创建回调函数列表
        callbacks = []
        
        # 添加早停回调（如果有评估数据集）
        if eval_dataset is not None:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001
            )
            callbacks.append(early_stopping)
        
        # 添加进度监控回调
        progress_callback = ProgressMonitoringCallback(self.progress_monitor)
        callbacks.append(progress_callback)
        
        # 添加内存监控回调
        memory_callback = MemoryMonitoringCallback(self.memory_optimizer, self.logging_system)
        callbacks.append(memory_callback)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # 使用 processing_class 替代已弃用的 tokenizer
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        logger.info("训练器创建完成")
        return trainer
    
    def create_safe_data_collator(self, tokenizer: PreTrainedTokenizer):
        """
        创建安全的数据整理器，具有自动错误恢复功能
        
        Args:
            tokenizer: 分词器
            
        Returns:
            安全的数据整理器
        """
        try:
            # 尝试导入增强的数据处理功能
            try:
                from .data_pipeline import create_safe_data_collator
            except ImportError:
                from data_pipeline import create_safe_data_collator
            
            logger.info("使用增强的安全数据整理器")
            return create_safe_data_collator(
                tokenizer=tokenizer,
                max_length=self.config.max_sequence_length,
                pad_to_multiple_of=8
            )
        except ImportError:
            logger.warning("无法导入增强数据处理，使用标准数据整理器")
            from transformers import DataCollatorForLanguageModeling
            
            # 确保tokenizer有pad_token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            )
    
    def train_model(self, trainer: Trainer, resume_from_checkpoint: Optional[str] = None) -> Trainer:
        """
        执行带错误处理和恢复的训练
        
        Args:
            trainer: 训练器实例
            resume_from_checkpoint: 可选的检查点路径用于恢复训练
            
        Returns:
            Trainer: 训练完成的训练器
        """
        logger.info("开始模型训练...")
        self.logging_system.info("开始模型训练", "TRAINING_ENGINE")
        
        # 训练前内存检查和优化
        self.memory_optimizer.log_memory_status("训练前")
        self.memory_optimizer.optimize_for_training()
        
        # 记录训练前内存状态
        pre_training_memory = self.memory_optimizer.get_memory_status()
        self.logging_system.log_memory_status(pre_training_memory, 0, "PreTraining")
        
        # 记录训练开始时间
        start_time = datetime.now()
        self.logging_system.info("训练开始", "TRAINING_ENGINE", {"start_time": start_time.isoformat()})
        
        try:
            # 使用内存优化器的安全操作包装训练
            def training_operation():
                return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # 执行训练
            train_result = self.memory_optimizer.safe_operation(
                training_operation,
                required_gb=2.0,  # 估计训练需要的额外内存
                max_retries=2
            )
            self.logging_system.info("###################################imhere")
            
            # 计算训练时间
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            logger.info("训练完成")
            logger.info(f"训练时间: {training_duration}")
            logger.info(f"训练损失: {train_result.training_loss:.4f}")
            
            # 记录训练完成信息
            self.logging_system.info("训练完成", "TRAINING_ENGINE", {
                "training_duration": str(training_duration),
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "epoch": getattr(train_result, 'epoch', 'N/A')
            })
            
            # 记录训练统计信息
            self._log_training_stats(train_result, training_duration)
            
            # 训练后内存状态
            self.memory_optimizer.log_memory_status("训练后")
            post_training_memory = self.memory_optimizer.get_memory_status()
            self.logging_system.log_memory_status(post_training_memory, train_result.global_step, "PostTraining")
            
            # 生成并保存最终报告
            try:
                final_report_path = self.save_final_report()
                self.logging_system.info(f"最终训练报告已生成: {final_report_path}", "TRAINING_ENGINE")
            except Exception as e:
                self.logging_system.error(f"生成最终报告失败: {e}", "TRAINING_ENGINE")
            
            return trainer
            
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            # 保存当前状态
            try:
                trainer.save_state()
                logger.info("训练状态已保存")
            except Exception as save_error:
                logger.error(f"保存训练状态失败: {save_error}")
            raise
            
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e.with_traceback}")
            
            # 尝试错误恢复
            recovery_success = self._attempt_training_recovery(e, trainer)
            
            if not recovery_success:
                # 如果恢复失败，提供详细的错误信息和建议
                self._provide_error_guidance(e)
                raise RuntimeError(f"训练失败: {e}")
            
            # 如果恢复成功，重新尝试训练
            logger.info("错误恢复成功，重新开始训练...")
            return self.train_model(trainer, resume_from_checkpoint)
    
    def _attempt_training_recovery(self, error: Exception, trainer: Trainer) -> bool:
        """
        尝试从训练错误中恢复
        
        Args:
            error: 发生的错误
            trainer: 训练器实例
            
        Returns:
            bool: 是否恢复成功
        """
        logger.info("尝试从训练错误中恢复...")
        
        try:
            # 内存清理
            self.memory_optimizer.cleanup_gpu_memory()
            
            # 检查是否是内存相关错误
            error_str = str(error).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logger.info("检测到内存错误，尝试内存优化...")
                
                # 尝试减少批次大小
                if hasattr(trainer.args, 'per_device_train_batch_size'):
                    original_batch_size = trainer.args.per_device_train_batch_size
                    if original_batch_size > 1:
                        new_batch_size = max(1, original_batch_size // 2)
                        trainer.args.per_device_train_batch_size = new_batch_size
                        
                        # 相应增加梯度累积步数以保持有效批次大小
                        trainer.args.gradient_accumulation_steps *= 2
                        
                        logger.info(f"批次大小从 {original_batch_size} 减少到 {new_batch_size}")
                        logger.info(f"梯度累积步数增加到 {trainer.args.gradient_accumulation_steps}")
                        
                        return True
                
                # 如果批次大小已经是1，尝试其他优化
                logger.info("尝试其他内存优化策略...")
                
                # 强制启用梯度检查点
                if hasattr(trainer.model, 'gradient_checkpointing_enable'):
                    trainer.model.gradient_checkpointing_enable()
                
                # 设置更严格的内存限制
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(0.85)
                
                return True
            
            # 检查是否是张量创建错误
            elif "unable to create tensor" in error_str or "padding" in error_str or "truncation" in error_str:
                logger.info("检测到张量创建错误，尝试数据处理优化...")
                
                # 这种错误通常需要重新创建数据整理器
                logger.info("张量创建错误通常需要重新启动训练，建议:")
                logger.info("1. 检查数据预处理是否正确")
                logger.info("2. 确保所有标签都是整数列表")
                logger.info("3. 验证分词器设置")
                logger.info("4. 使用增强的数据整理器")
                
                return False  # 这种错误通常需要重新启动
            
            # 检查是否是数据相关错误
            elif "dataloader" in error_str or "dataset" in error_str:
                logger.info("检测到数据加载错误，尝试数据优化...")
                
                # 减少数据加载器的工作进程数
                if hasattr(trainer.args, 'dataloader_num_workers'):
                    trainer.args.dataloader_num_workers = 0
                    logger.info("设置数据加载器工作进程数为0")
                
                return True
            
            # 其他类型的错误
            else:
                logger.warning(f"未知错误类型，无法自动恢复: {error}")
                return False
                
        except Exception as recovery_error:
            logger.error(f"错误恢复过程中发生异常: {recovery_error}")
            return False
    
    def _provide_error_guidance(self, error: Exception) -> None:
        """
        为训练错误提供详细的指导建议
        
        Args:
            error: 发生的错误
        """
        error_str = str(error).lower()
        
        logger.error("训练失败，错误分析和建议:")
        logger.error(f"错误信息: {error}")
        
        if "out of memory" in error_str or "cuda" in error_str:
            logger.error("内存相关错误建议:")
            logger.error("1. 减少 batch_size (当前: {})".format(
                getattr(self.config, 'batch_size', 'unknown')))
            logger.error("2. 增加 gradient_accumulation_steps (当前: {})".format(
                getattr(self.config, 'gradient_accumulation_steps', 'unknown')))
            logger.error("3. 减少 max_sequence_length (当前: {})".format(
                getattr(self.config, 'max_sequence_length', 'unknown')))
            logger.error("4. 检查GPU内存使用情况")
            logger.error("5. 考虑使用更小的模型或更激进的量化")
            
        elif "unable to create tensor" in error_str or "padding" in error_str or "truncation" in error_str:
            logger.error("张量创建错误建议:")
            logger.error("1. 检查数据预处理是否正确生成了整数列表")
            logger.error("2. 确保所有 input_ids 和 labels 都是扁平的整数列表")
            logger.error("3. 验证分词器的 pad_token 设置")
            logger.error("4. 使用增强的数据整理器 (EnhancedDataCollatorForLanguageModeling)")
            logger.error("5. 检查数据集中是否有嵌套列表或非整数值")
            logger.error("6. 尝试使用 create_safe_data_collator 函数")
            
        elif "dataloader" in error_str:
            logger.error("数据加载错误建议:")
            logger.error("1. 检查数据文件是否存在且格式正确")
            logger.error("2. 减少 dataloader_num_workers")
            logger.error("3. 检查数据集大小和内容")
            
        elif "checkpoint" in error_str:
            logger.error("检查点相关错误建议:")
            logger.error("1. 检查输出目录的写入权限")
            logger.error("2. 检查磁盘空间是否充足")
            logger.error("3. 验证检查点文件的完整性")
            
        else:
            logger.error("通用错误建议:")
            logger.error("1. 检查模型和数据的兼容性")
            logger.error("2. 验证所有依赖包的版本")
            logger.error("3. 检查CUDA和PyTorch的兼容性")
            logger.error("4. 尝试使用更简单的配置")
    
    def _log_training_stats(self, train_result, training_duration) -> None:
        """
        记录训练统计信息
        
        Args:
            train_result: 训练结果
            training_duration: 训练持续时间
        """
        try:
            stats = {
                "training_loss": train_result.training_loss,
                "training_duration": str(training_duration),
                "total_steps": train_result.global_step,
                "epochs_completed": getattr(train_result, 'epoch', 'N/A'),
            }
            
            # 如果有评估结果，也记录
            if hasattr(train_result, 'eval_loss'):
                stats["eval_loss"] = train_result.eval_loss
            
            logger.info("训练统计信息:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
                
        except Exception as e:
            logger.warning(f"记录训练统计信息时出错: {e}")
    
    def save_model(self, trainer: Trainer, output_dir: Optional[str] = None) -> None:
        """
        实现带磁盘空间管理的检查点保存
        
        Args:
            trainer: 训练器实例
            output_dir: 可选的输出目录，如果为None则使用配置中的目录
        """
        save_dir = output_dir or self.config.output_dir
        logger.info(f"保存模型到 {save_dir}")
        
        try:
            # 检查磁盘空间
            self._check_disk_space(save_dir)
            
            # 保存模型
            trainer.save_model(save_dir)
            
            # 保存分词器
            if trainer.tokenizer is not None:
                trainer.tokenizer.save_pretrained(save_dir)
            
            # 保存训练状态
            trainer.save_state()
            
            # 如果是PEFT模型，额外保存适配器
            if isinstance(trainer.model, PeftModel):
                adapter_dir = os.path.join(save_dir, "adapter")
                trainer.model.save_pretrained(adapter_dir)
                logger.info(f"PEFT适配器已保存到 {adapter_dir}")
            
            logger.info("模型保存完成")
            
        except Exception as e:
            logger.error(f"保存模型时出错: {e}")
            raise RuntimeError(f"无法保存模型: {e}")
    
    def _check_disk_space(self, directory: str, required_gb: float = 5.0) -> None:
        """
        检查磁盘空间是否足够
        
        Args:
            directory: 要检查的目录
            required_gb: 需要的磁盘空间（GB）
        """
        try:
            import shutil
            
            # 确保目录存在
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # 获取磁盘使用情况
            total, used, free = shutil.disk_usage(directory)
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            
            logger.info(f"磁盘使用情况: 总计 {total_gb:.2f}GB, 已用 {used_gb:.2f}GB, 可用 {free_gb:.2f}GB")
            
            if free_gb < required_gb:
                logger.warning(f"磁盘空间不足: 可用 {free_gb:.2f}GB, 需要 {required_gb:.2f}GB")
                
                # 尝试清理旧的检查点
                freed_space = self._cleanup_old_checkpoints(directory)
                
                # 重新检查
                total, used, free = shutil.disk_usage(directory)
                free_gb = free / (1024**3)
                
                logger.info(f"清理后可用空间: {free_gb:.2f}GB (释放了 {freed_space:.2f}GB)")
                
                if free_gb < required_gb:
                    # 尝试清理临时文件
                    self._cleanup_temp_files(directory)
                    
                    # 最终检查
                    total, used, free = shutil.disk_usage(directory)
                    free_gb = free / (1024**3)
                    
                    if free_gb < required_gb:
                        raise RuntimeError(f"磁盘空间不足: 可用 {free_gb:.2f}GB, 需要 {required_gb:.2f}GB")
            
            logger.info(f"磁盘空间检查通过: 可用 {free_gb:.2f}GB")
            
        except Exception as e:
            logger.warning(f"磁盘空间检查失败: {e}")
    
    def _cleanup_old_checkpoints(self, directory: str) -> float:
        """
        清理旧的检查点文件
        
        Args:
            directory: 检查点目录
            
        Returns:
            float: 释放的磁盘空间（GB）
        """
        freed_space = 0.0
        
        try:
            checkpoint_dirs = []
            base_path = Path(directory)
            
            if not base_path.exists():
                return 0.0
            
            # 查找所有检查点目录
            for item in base_path.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    checkpoint_dirs.append(item)
            
            if not checkpoint_dirs:
                logger.info("未找到需要清理的检查点")
                return 0.0
            
            # 按修改时间排序，保留最新的几个
            checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 删除多余的检查点
            keep_count = self.config.save_total_limit
            deleted_count = 0
            
            for checkpoint_dir in checkpoint_dirs[keep_count:]:
                try:
                    # 计算目录大小
                    dir_size = self._get_directory_size(checkpoint_dir)
                    
                    # 删除目录
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                    
                    freed_space += dir_size
                    deleted_count += 1
                    logger.info(f"删除旧检查点: {checkpoint_dir.name} ({dir_size:.2f}GB)")
                    
                except Exception as e:
                    logger.warning(f"删除检查点 {checkpoint_dir} 时出错: {e}")
            
            if deleted_count > 0:
                logger.info(f"清理完成: 删除了 {deleted_count} 个检查点，释放 {freed_space:.2f}GB")
            else:
                logger.info("无需清理检查点")
                
        except Exception as e:
            logger.warning(f"清理检查点时出错: {e}")
        
        return freed_space
    
    def _cleanup_temp_files(self, directory: str) -> float:
        """
        清理临时文件和缓存
        
        Args:
            directory: 目录路径
            
        Returns:
            float: 释放的磁盘空间（GB）
        """
        freed_space = 0.0
        
        try:
            base_path = Path(directory)
            
            # 清理模式
            temp_patterns = [
                "*.tmp",
                "*.temp", 
                "*.cache",
                "*.log",
                "events.out.tfevents.*",  # TensorBoard日志
                "*.pyc",
                "__pycache__"
            ]
            
            for pattern in temp_patterns:
                if pattern == "__pycache__":
                    # 清理__pycache__目录
                    for pycache_dir in base_path.rglob("__pycache__"):
                        try:
                            dir_size = self._get_directory_size(pycache_dir)
                            import shutil
                            shutil.rmtree(pycache_dir)
                            freed_space += dir_size
                            logger.debug(f"删除缓存目录: {pycache_dir}")
                        except Exception as e:
                            logger.debug(f"删除缓存目录失败 {pycache_dir}: {e}")
                else:
                    # 清理匹配模式的文件
                    for temp_file in base_path.rglob(pattern):
                        if temp_file.is_file():
                            try:
                                file_size = temp_file.stat().st_size / (1024**3)
                                temp_file.unlink()
                                freed_space += file_size
                                logger.debug(f"删除临时文件: {temp_file}")
                            except Exception as e:
                                logger.debug(f"删除临时文件失败 {temp_file}: {e}")
            
            if freed_space > 0:
                logger.info(f"临时文件清理完成，释放 {freed_space:.2f}GB")
            
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")
        
        return freed_space
    
    def _get_directory_size(self, directory: Path) -> float:
        """
        计算目录大小
        
        Args:
            directory: 目录路径
            
        Returns:
            float: 目录大小（GB）
        """
        total_size = 0
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.debug(f"计算目录大小时出错 {directory}: {e}")
        
        return total_size / (1024**3)
    
    def get_disk_usage_report(self, directory: str) -> Dict[str, Any]:
        """
        获取磁盘使用报告
        
        Args:
            directory: 目录路径
            
        Returns:
            Dict[str, Any]: 磁盘使用报告
        """
        try:
            import shutil
            
            # 基本磁盘信息
            total, used, free = shutil.disk_usage(directory)
            
            report = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "usage_percentage": (used / total) * 100,
                "directory": directory
            }
            
            # 检查点信息
            base_path = Path(directory)
            if base_path.exists():
                checkpoint_dirs = [d for d in base_path.iterdir() 
                                 if d.is_dir() and d.name.startswith("checkpoint-")]
                
                checkpoint_sizes = []
                total_checkpoint_size = 0
                
                for checkpoint_dir in checkpoint_dirs:
                    size = self._get_directory_size(checkpoint_dir)
                    checkpoint_sizes.append({
                        "name": checkpoint_dir.name,
                        "size_gb": size,
                        "modified": datetime.fromtimestamp(checkpoint_dir.stat().st_mtime)
                    })
                    total_checkpoint_size += size
                
                report.update({
                    "checkpoint_count": len(checkpoint_dirs),
                    "total_checkpoint_size_gb": total_checkpoint_size,
                    "checkpoints": sorted(checkpoint_sizes, 
                                        key=lambda x: x["modified"], reverse=True)
                })
            
            return report
            
        except Exception as e:
            logger.error(f"生成磁盘使用报告时出错: {e}")
            return {"error": str(e)}
    
    def generate_final_training_report(self) -> Dict[str, Any]:
        """
        生成最终训练报告，包含性能指标和摘要
        
        Returns:
            Dict[str, Any]: 完整的训练报告
        """
        try:
            # 获取进度监控摘要 - 添加错误处理
            try:
                progress_summary = self.progress_monitor.generate_training_summary()
            except Exception as e:
                logger.warning(f"获取进度监控摘要失败: {e}")
                progress_summary = {
                    "error": f"进度监控摘要生成失败: {e}",
                    "training_summary": {
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "total_duration": "0:00:00",
                        "total_steps": 0,
                        "final_epoch": 0.0,
                        "final_loss": 0.0,
                        "final_learning_rate": 0.0
                    }
                }
            
            # 获取日志系统摘要
            try:
                log_summary = self.logging_system.get_log_summary()
            except Exception as e:
                logger.warning(f"获取日志系统摘要失败: {e}")
                log_summary = {"error": f"日志摘要生成失败: {e}"}
            
            # 获取磁盘使用报告
            try:
                disk_report = self.get_disk_usage_report(self.config.output_dir)
            except Exception as e:
                logger.warning(f"获取磁盘使用报告失败: {e}")
                disk_report = {"error": f"磁盘报告生成失败: {e}"}
            
            # 获取内存优化器状态
            try:
                final_memory_status = self.memory_optimizer.get_memory_status()
            except Exception as e:
                logger.warning(f"获取内存状态失败: {e}")
                final_memory_status = None
            
            # 组合最终报告
            final_report = {
                "training_engine_config": {
                    "output_dir": self.config.output_dir,
                    "max_memory_gb": self.config.max_memory_gb,
                    "batch_size": self.config.batch_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "learning_rate": self.config.learning_rate,
                    "num_epochs": self.config.num_epochs
                },
                "progress_summary": progress_summary,
                "logging_summary": log_summary,
                "disk_usage": disk_report,
                "final_memory_status": {
                    "allocated_gb": final_memory_status.allocated_gb if final_memory_status else 0.0,
                    "cached_gb": final_memory_status.cached_gb if final_memory_status else 0.0,
                    "available_gb": final_memory_status.available_gb if final_memory_status else 0.0,
                    "total_gb": final_memory_status.total_gb if final_memory_status else 0.0,
                    "is_safe": final_memory_status.is_safe if final_memory_status else True
                },
                "report_generated_at": datetime.now().isoformat()
            }
            
            # 记录到日志系统
            self.logging_system.info("最终训练报告生成完成", "TRAINING_ENGINE", 
                                    {"report_sections": list(final_report.keys())})
            
            return final_report
            
        except Exception as e:
            error_msg = f"生成最终训练报告失败: {e}"
            logger.error(error_msg)
            self.logging_system.error(error_msg, "TRAINING_ENGINE")
            return {"error": str(e), "generated_at": datetime.now().isoformat()}
    
    def save_final_report(self, output_path: Optional[str] = None) -> str:
        """
        保存最终训练报告到文件
        
        Args:
            output_path: 可选的输出路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "final_training_report.json")
        
        try:
            report = self.generate_final_training_report()
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用自定义JSON编码器处理datetime对象
            def json_serializer(obj):
                """JSON序列化器，处理datetime和其他特殊对象"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, timedelta):
                    return str(obj)
                elif hasattr(obj, 'isoformat'):  # 其他日期时间对象
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):  # 自定义对象
                    return obj.__dict__
                else:
                    return str(obj)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            self.logging_system.info(f"最终训练报告已保存到 {output_path}", "TRAINING_ENGINE")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"保存最终训练报告失败: {e}"
            logger.error(error_msg)
            self.logging_system.error(error_msg, "TRAINING_ENGINE")
            raise
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> str:
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            str: 验证后的检查点路径
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        logger.info(f"将从检查点恢复训练: {checkpoint_path}")
        return str(checkpoint_path)


class ProgressMonitoringCallback(TrainerCallback):
    """进度监控回调类"""
    
    def __init__(self, progress_monitor: ProgressMonitor):
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
            
            # 考虑梯度累积步数 - 这是关键修复
            gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
            if gradient_accumulation_steps > 1:
                # 梯度累积会减少优化器更新次数（global_step），但不会减少实际处理的批次数
                # 每 gradient_accumulation_steps 个批次才会进行一次优化器更新
                steps_per_epoch = math.ceil(dataloader_length / gradient_accumulation_steps)
                logger.info(f"考虑梯度累积 ({gradient_accumulation_steps} 步)，每轮优化器更新次数: {steps_per_epoch}")
            else:
                steps_per_epoch = dataloader_length
            
            total_steps = int(args.num_train_epochs * steps_per_epoch)
            
            logger.info(f"从数据加载器计算总步数: {total_steps} "
                       f"(数据加载器长度: {dataloader_length}, 梯度累积: {gradient_accumulation_steps}, "
                       f"每轮步数: {steps_per_epoch}, 轮数: {args.num_train_epochs})")
            
            return self._validate_total_steps(total_steps, args)
            
        except Exception as e:
            logger.error(f"从数据加载器计算总步数失败: {e}")
            return self._get_fallback_total_steps(args)
    
    def _calculate_from_dataset(self, args, train_dataset) -> int:
        """从数据集估算总步数"""
        try:
            dataset_size = len(train_dataset)
            if dataset_size == 0:
                logger.warning("数据集大小为0，使用默认步数")
                return self._get_fallback_total_steps(args)
            
            # 计算有效批次大小（不包含梯度累积）
            effective_batch_size = getattr(args, 'per_device_train_batch_size', 1)
            
            # 考虑多GPU训练
            world_size = getattr(args, 'world_size', 1)
            if world_size > 1:
                effective_batch_size *= world_size
                logger.info(f"检测到多GPU训练，world_size: {world_size}")
            
            # 计算每轮批次数（不考虑梯度累积）
            dataloader_drop_last = getattr(args, 'dataloader_drop_last', False)
            if dataloader_drop_last:
                batches_per_epoch = dataset_size // effective_batch_size
            else:
                batches_per_epoch = math.ceil(dataset_size / effective_batch_size)
            
            # 考虑梯度累积对步数的影响
            gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
            if gradient_accumulation_steps > 1:
                # 梯度累积会减少优化器更新次数（global_step），但不会减少实际处理的批次数
                # 每 gradient_accumulation_steps 个批次才会进行一次优化器更新
                steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
                logger.info(f"考虑梯度累积 ({gradient_accumulation_steps} 步)，每轮优化器更新次数: {steps_per_epoch}")
            else:
                steps_per_epoch = batches_per_epoch
            
            steps_per_epoch = max(1, steps_per_epoch)
            total_steps = int(args.num_train_epochs * steps_per_epoch)
            
            logger.info(f"根据数据集估算总步数: {total_steps} "
                       f"(数据集: {dataset_size}, 批次大小: {effective_batch_size}, "
                       f"每轮批次数: {batches_per_epoch}, 梯度累积: {gradient_accumulation_steps}, "
                       f"每轮步数: {steps_per_epoch}, 轮数: {args.num_train_epochs}, "
                       f"drop_last: {dataloader_drop_last})")
            
            return self._validate_total_steps(total_steps, args)
            
        except Exception as e:
            logger.error(f"从数据集估算总步数失败: {e}")
            return self._get_fallback_total_steps(args)
    
    def _get_fallback_total_steps(self, args) -> int:
        """获取备选的总步数"""
        try:
            # 基于批次大小的智能估算
            batch_size = getattr(args, 'per_device_train_batch_size', 4)
            estimated_steps_per_epoch = max(10, int(1000 / batch_size))
            total_steps = int(args.num_train_epochs * estimated_steps_per_epoch)
            
            logger.warning(f"无法计算准确的总步数，使用估算值: {total_steps} "
                          f"(每轮估算: {estimated_steps_per_epoch} 步)")
            
            return self._validate_total_steps(total_steps, args)
            
        except Exception as e:
            logger.error(f"获取备选总步数失败: {e}")
            # 最后的保底值
            return int(getattr(args, 'num_train_epochs', 10) * 50)
    
    def _validate_total_steps(self, total_steps: int, args) -> int:
        """
        验证总步数的合理性
        
        Args:
            total_steps: 计算得到的总步数
            args: 训练参数
            
        Returns:
            int: 验证后的总步数
        """
        try:
            # 基本合理性检查
            if total_steps <= 0:
                logger.error(f"计算得到的总步数无效: {total_steps}")
                return int(getattr(args, 'num_train_epochs', 10) * 100)
            
            # 检查是否过大（可能计算错误）
            num_epochs = getattr(args, 'num_train_epochs', 10)
            max_reasonable_steps = int(num_epochs * 10000)  # 每轮最多10000步
            if total_steps > max_reasonable_steps:
                logger.warning(f"计算得到的总步数可能过大: {total_steps}, 限制为: {max_reasonable_steps}")
                return max_reasonable_steps
            
            # 检查是否过小
            min_reasonable_steps = int(num_epochs)  # 每轮至少1步
            if total_steps < min_reasonable_steps:
                logger.warning(f"计算得到的总步数可能过小: {total_steps}, 调整为: {min_reasonable_steps}")
                return min_reasonable_steps
            
            return total_steps
            
        except Exception as e:
            logger.error(f"验证总步数时出错: {e}")
            return int(getattr(args, 'num_train_epochs', 10) * 100)
    
    def on_init_end(self, args, state, control, **kwargs):
        """初始化结束时的回调"""
        # 在这里开始监控，因为此时训练器已经完全初始化
        if not self.training_started:
            total_steps = self._calculate_total_steps(args, state, **kwargs)
            self.progress_monitor.start_monitoring(total_steps)
            self.training_started = True
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时的回调"""
        # 确保监控已经开始
        if not self.training_started:
            total_steps = self._calculate_total_steps(args, state, **kwargs)
            self.progress_monitor.start_monitoring(total_steps)
            self.training_started = True
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """训练轮次开始时的回调"""
        # 在每个epoch开始时更新进度
        try:
            if hasattr(state, 'epoch'):
                self.progress_monitor.update_progress(
                    epoch=state.epoch,
                    step=state.global_step,
                    loss=None,  # epoch开始时可能还没有loss
                    learning_rate=None
                )
        except Exception as e:
            logger.warning(f"epoch开始进度更新失败: {e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """训练轮次结束时的回调"""
        # 在每个epoch结束时更新进度
        try:
            if hasattr(state, 'epoch'):
                self.progress_monitor.update_progress(
                    epoch=state.epoch,
                    step=state.global_step,
                    loss=None,  # epoch结束时可能还没有最新的loss
                    learning_rate=None
                )
        except Exception as e:
            logger.warning(f"epoch结束进度更新失败: {e}")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """训练步骤开始时的回调"""
        try:
            # 在每个步骤开始时更新进度
            self.progress_monitor.update_progress(
                epoch=getattr(state, 'epoch', 0),
                step=state.global_step,
                loss=None,  # 步骤开始时还没有loss
                learning_rate=None
            )
        except Exception as e:
            logger.warning(f"步骤开始进度更新失败: {e}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """训练步骤结束时的回调"""
        try:
            # 在每个步骤结束时更新进度
            self.progress_monitor.update_progress(
                epoch=getattr(state, 'epoch', 0),
                step=state.global_step,
                loss=None,  # 步骤结束时可能还没有最新的loss
                learning_rate=None
            )
        except Exception as e:
            logger.warning(f"步骤结束进度更新失败: {e}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        """评估时的回调"""
        try:
            # 评估时更新进度
            # 安全地获取epoch值
            epoch_value = getattr(state, 'epoch', None)
            if epoch_value is None:
                epoch_value = 0.0
            else:
                try:
                    epoch_value = float(epoch_value)
                except (TypeError, ValueError):
                    epoch_value = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch_value,
                step=state.global_step,
                loss=None,
                learning_rate=None
            )
        except Exception as e:
            logger.warning(f"评估进度更新失败: {e}")
    
    def on_prediction_step(self, args, state, control, **kwargs):
        """预测步骤时的回调"""
        try:
            # 预测步骤时更新进度（如果需要的话）
            pass  # 通常预测步骤不需要更新训练进度
        except Exception as e:
            logger.warning(f"预测步骤进度更新失败: {e}")
    
    def on_save(self, args, state, control, **kwargs):
        """保存模型时的回调"""
        try:
            # 保存时更新进度
            # 安全地获取epoch值
            epoch_value = getattr(state, 'epoch', None)
            if epoch_value is None:
                epoch_value = 0.0
            else:
                try:
                    epoch_value = float(epoch_value)
                except (TypeError, ValueError):
                    epoch_value = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch_value,
                step=state.global_step,
                loss=None,
                learning_rate=None
            )
        except Exception as e:
            logger.warning(f"保存时进度更新失败: {e}")
    
    def on_substep_end(self, args, state, control, **kwargs):
        """子步骤结束时的回调"""
        try:
            # 子步骤结束时更新进度
            # 安全地获取epoch值
            epoch_value = getattr(state, 'epoch', None)
            if epoch_value is None:
                epoch_value = 0.0
            else:
                try:
                    epoch_value = float(epoch_value)
                except (TypeError, ValueError):
                    epoch_value = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch_value,
                step=state.global_step,
                loss=None,
                learning_rate=None
            )
        except Exception as e:
            logger.warning(f"子步骤结束进度更新失败: {e}")
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """优化器步骤前的回调"""
        try:
            # 优化器步骤前更新进度
            # 安全地获取epoch值
            epoch_value = getattr(state, 'epoch', None)
            if epoch_value is None:
                epoch_value = 0.0
            else:
                try:
                    epoch_value = float(epoch_value)
                except (TypeError, ValueError):
                    epoch_value = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch_value,
                step=state.global_step,
                loss=None,
                learning_rate=None
            )
        except Exception as e:
            logger.warning(f"优化器步骤前进度更新失败: {e}")
    
    def on_optimizer_step(self, args, state, control, **kwargs):
        """优化器步骤时的回调"""
        try:
            # 优化器步骤时更新进度
            # 安全地获取epoch值
            epoch_value = getattr(state, 'epoch', None)
            if epoch_value is None:
                epoch_value = 0.0
            else:
                try:
                    epoch_value = float(epoch_value)
                except (TypeError, ValueError):
                    epoch_value = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch_value,
                step=state.global_step,
                loss=None,
                learning_rate=None
            )
        except Exception as e:
            logger.warning(f"优化器步骤进度更新失败: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """日志记录时的回调"""
        if logs is not None:
            # 安全地获取和转换日志值
            epoch = logs.get("epoch", 0)
            loss = logs.get("loss", 0.0)
            learning_rate = logs.get("learning_rate", 0.0)
            
            # 确保类型正确
            try:
                epoch = float(epoch) if epoch is not None else 0.0
                loss = float(loss) if loss is not None else 0.0
                learning_rate = float(learning_rate) if learning_rate is not None else 0.0
            except (TypeError, ValueError):
                epoch = 0.0
                loss = 0.0
                learning_rate = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch,
                step=state.global_step,
                loss=loss,
                learning_rate=learning_rate
            )
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的回调"""
        self.progress_monitor.stop_monitoring()
        
        # 生成并保存训练摘要
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        self.progress_monitor.save_progress_report(summary_path)


class MemoryMonitoringCallback(TrainerCallback):
    """内存监控回调类"""
    
    def __init__(self, memory_optimizer: MemoryOptimizer, logging_system: Optional[LoggingSystem] = None):
        self.memory_optimizer = memory_optimizer
        self.logging_system = logging_system
        self.step_count = 0
        self.epoch_count = 0
        self.memory_history = []
        self.epoch_memory_history = []  # 轮次级别的内存历史
        self.last_cleanup_step = 0
        self.epoch_start_memory = None
        self.epoch_peak_memory = None
        self.epoch_cleanup_count = 0
    
    def on_init_end(self, args, state, control, **kwargs):
        """初始化结束时的回调"""
        # 在初始化结束时记录初始内存状态
        try:
            status = self.memory_optimizer.get_memory_status()
            logger.info(f"训练器初始化完成，内存状态: {status.allocated_gb:.2f}GB")
            
            if self.logging_system:
                self.logging_system.log_memory_status(status, 0, "InitEnd")
        except Exception as e:
            self._handle_callback_error(e, "on_init_end", {"step": 0})
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时的回调"""
        # 在训练开始时记录内存状态
        try:
            status = self.memory_optimizer.get_memory_status()
            logger.info(f"训练开始，内存状态: {status.allocated_gb:.2f}GB")
            
            if self.logging_system:
                self.logging_system.log_memory_status(status, 0, "TrainBegin")
        except Exception as e:
            self._handle_callback_error(e, "on_train_begin", {"step": 0})
    
    def on_step_begin(self, args, state, control, **kwargs):
        """训练步骤开始时的回调"""
        try:
            self._update_callback_state(step_increment=1)
            
            # 每10步检查一次内存
            if self.step_count % 10 == 0:
                status = self.memory_optimizer.get_memory_status()
                self.memory_history.append((self.step_count, status.allocated_gb))
                
                # 更新轮次峰值内存
                if self.epoch_peak_memory is None or status.allocated_gb > self.epoch_peak_memory:
                    self.epoch_peak_memory = status.allocated_gb
                
                # 使用工具方法记录内存状态
                self._log_memory_status(status, self.step_count, "StepBegin", {
                    "memory_check_interval": 10
                })
                
                # 使用工具方法检查是否需要清理
                if self._should_trigger_cleanup(status):
                    cleanup_msg = self._format_memory_log("StepBegin", status.allocated_gb, "触发内存清理")
                    logger.warning(cleanup_msg)
                    
                    self.memory_optimizer.cleanup_gpu_memory()
                    self.last_cleanup_step = self.step_count
                    self.epoch_cleanup_count += 1
                    
                    if self.logging_system:
                        self.logging_system.warning(cleanup_msg, "MEMORY_CALLBACK", {
                            "step": self.step_count,
                            "cleanup_triggered": True
                        })
                
                # 检测内存泄漏
                if len(self.memory_history) > 10:
                    self._check_memory_leak()
                
        except Exception as e:
            self._handle_callback_error(e, "on_step_begin", {
                "step": self.step_count,
                "memory_check_interval": 10
            })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """日志记录时的回调"""
        if logs is not None:
            try:
                status = self.memory_optimizer.get_memory_status()
                
                # 更新Transformers的日志
                logs.update({
                    "memory_allocated_gb": status.allocated_gb,
                    "memory_cached_gb": status.cached_gb,
                    "memory_available_gb": status.available_gb,
                    "memory_is_safe": status.is_safe
                })
                
                # 如果有日志系统，创建训练指标并记录
                if self.logging_system and logs.get("loss") is not None:
                    training_metrics = TrainingMetrics(
                        epoch=logs.get("epoch", 0),
                        step=state.global_step,
                        loss=logs.get("loss", 0.0),
                        learning_rate=logs.get("learning_rate", 0.0),
                        memory_usage=status,
                        timestamp=datetime.now()
                    )
                    self.logging_system.log_training_metrics(training_metrics)
                
                # 每50步记录详细内存状态
                if self.step_count % 50 == 0 and self.logging_system:
                    self.logging_system.log_memory_status(status, state.global_step, "Training")
                    
            except Exception as e:
                self._handle_callback_error(e, "on_log", {
                    "step": getattr(state, 'global_step', 0),
                    "has_logs": logs is not None
                })
    
    def on_save(self, args, state, control, **kwargs):
        """保存检查点时的回调"""
        try:
            # 在保存前进行内存清理
            self.memory_optimizer.cleanup_gpu_memory()
            info_msg = f"检查点保存前内存清理完成 (步骤 {self.step_count})"
            logger.info(info_msg)
            if self.logging_system:
                self.logging_system.info(info_msg, "MEMORY_CALLBACK", {"step": self.step_count})
        except Exception as e:
            self._handle_callback_error(e, "on_save", {"step": self.step_count})
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的回调"""
        try:
            # 训练结束时的最终内存清理
            self.memory_optimizer.cleanup_gpu_memory()
            
            # 记录内存使用统计
            if self.memory_history:
                max_memory = max(mem for _, mem in self.memory_history)
                avg_memory = sum(mem for _, mem in self.memory_history) / len(self.memory_history)
                stats_msg = f"训练内存统计: 最大 {max_memory:.2f}GB, 平均 {avg_memory:.2f}GB"
                logger.info(stats_msg)
                
                if self.logging_system:
                    self.logging_system.info(stats_msg, "MEMORY_CALLBACK", {
                        "max_memory_gb": max_memory,
                        "avg_memory_gb": avg_memory,
                        "total_steps": len(self.memory_history)
                    })
            
        except Exception as e:
            self._handle_callback_error(e, "on_train_end", {
                "total_steps": len(self.memory_history),
                "memory_history_length": len(self.memory_history)
            })
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """训练轮次开始时的回调"""
        try:
            self.epoch_count += 1
            status = self.memory_optimizer.get_memory_status()
            self.epoch_start_memory = status.allocated_gb
            self.epoch_peak_memory = status.allocated_gb  # 初始化峰值内存
            self.epoch_cleanup_count = 0  # 重置清理计数
            
            epoch_msg = f"轮次 {self.epoch_count} 开始，内存状态: {status.allocated_gb:.2f}GB"
            logger.info(epoch_msg)
            
            if self.logging_system:
                self.logging_system.log_memory_status(status, state.global_step, f"EpochBegin_{self.epoch_count}")
                self.logging_system.info(epoch_msg, "MEMORY_CALLBACK", {
                    "epoch": self.epoch_count,
                    "memory_gb": status.allocated_gb,
                    "step": state.global_step
                })
                
        except Exception as e:
            self._handle_callback_error(e, "on_epoch_begin", {
                "epoch": self.epoch_count,
                "step": getattr(state, 'global_step', 0)
            })
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """训练轮次结束时的回调"""
        try:
            status = self.memory_optimizer.get_memory_status()
            current_memory = status.allocated_gb
            
            # 计算轮次内存变化
            memory_delta = 0.0
            if self.epoch_start_memory is not None:
                memory_delta = current_memory - self.epoch_start_memory
            
            # 创建轮次内存指标
            if self.epoch_start_memory is not None and self.epoch_peak_memory is not None:
                epoch_metrics = EpochMemoryMetrics(
                    epoch=self.epoch_count,
                    start_memory_gb=self.epoch_start_memory,
                    end_memory_gb=current_memory,
                    peak_memory_gb=self.epoch_peak_memory,
                    memory_delta_gb=memory_delta,
                    cleanup_count=self.epoch_cleanup_count,
                    timestamp=datetime.now()
                )
                self.epoch_memory_history.append(epoch_metrics)
                
                # 记录轮次内存摘要
                self._log_epoch_memory_summary(epoch_metrics, state.global_step)
            
            epoch_msg = f"轮次 {self.epoch_count} 结束，内存状态: {current_memory:.2f}GB"
            if self.epoch_start_memory is not None:
                epoch_msg += f"，内存变化: {memory_delta:+.2f}GB，峰值: {self.epoch_peak_memory:.2f}GB"
            
            logger.info(epoch_msg)
            
            # 如果内存增长过多，进行清理
            if memory_delta > 0.5:  # 增长超过0.5GB
                cleanup_msg = f"轮次 {self.epoch_count} 内存增长过多 ({memory_delta:.2f}GB)，执行清理"
                logger.warning(cleanup_msg)
                self.memory_optimizer.cleanup_gpu_memory()
                self.epoch_cleanup_count += 1
                
                if self.logging_system:
                    self.logging_system.warning(cleanup_msg, "MEMORY_CALLBACK", {
                        "epoch": self.epoch_count,
                        "memory_delta_gb": memory_delta,
                        "cleanup_triggered": True
                    })
            
            if self.logging_system:
                self.logging_system.log_memory_status(status, state.global_step, f"EpochEnd_{self.epoch_count}")
                self.logging_system.info(epoch_msg, "MEMORY_CALLBACK", {
                    "epoch": self.epoch_count,
                    "memory_gb": current_memory,
                    "memory_delta_gb": memory_delta,
                    "peak_memory_gb": self.epoch_peak_memory,
                    "cleanup_count": self.epoch_cleanup_count,
                    "step": state.global_step
                })
                
        except Exception as e:
            self._handle_callback_error(e, "on_epoch_end", {
                "epoch": self.epoch_count,
                "step": getattr(state, 'global_step', 0),
                "epoch_start_memory": self.epoch_start_memory
            })
    
    def on_step_end(self, args, state, control, **kwargs):
        """训练步骤结束时的回调"""
        try:
            # 每20步检查一次步骤结束时的内存状态
            if self.step_count % 20 == 0:
                status = self.memory_optimizer.get_memory_status()
                
                step_msg = f"步骤 {self.step_count} 结束，内存状态: {status.allocated_gb:.2f}GB"
                logger.debug(step_msg)
                
                # 检查内存是否安全
                if not status.is_safe:
                    warning_msg = f"步骤 {self.step_count} 结束时内存使用过高 ({status.allocated_gb:.2f}GB)"
                    logger.warning(warning_msg)
                    
                    if self.logging_system:
                        self.logging_system.warning(warning_msg, "MEMORY_CALLBACK", {
                            "step": self.step_count,
                            "memory_gb": status.allocated_gb,
                            "memory_safe": status.is_safe
                        })
                
                # 每100步记录详细内存状态
                if self.step_count % 100 == 0 and self.logging_system:
                    self.logging_system.log_memory_status(status, state.global_step, f"StepEnd_{self.step_count}")
                    
        except Exception as e:
            self._handle_callback_error(e, "on_step_end", {
                "step": self.step_count,
                "global_step": getattr(state, 'global_step', 0)
            })
    
    def on_evaluate(self, args, state, control, **kwargs):
        """评估开始时的回调"""
        try:
            status = self.memory_optimizer.get_memory_status()
            
            # 使用工具方法记录内存状态
            self._log_memory_status(status, getattr(state, 'global_step', 0), "Evaluation", {
                "phase": "evaluation_begin"
            })
            
            # 使用更严格的阈值检查评估前是否需要清理（80%）
            if self._should_trigger_cleanup(status, status.total_gb * 0.8):
                cleanup_msg = self._format_memory_log("Evaluation", status.allocated_gb, "评估前执行预防性清理")
                logger.warning(cleanup_msg)
                self.memory_optimizer.cleanup_gpu_memory()
                
                if self.logging_system:
                    self.logging_system.warning(cleanup_msg, "MEMORY_CALLBACK", {
                        "phase": "evaluation",
                        "cleanup_triggered": True,
                        "cleanup_reason": "preventive_before_evaluation"
                    })
                
        except Exception as e:
            self._handle_callback_error(e, "on_evaluate", {
                "step": getattr(state, 'global_step', 0),
                "phase": "evaluation"
            })
    
    def on_prediction_step(self, args, state, control, **kwargs):
        """预测步骤时的回调"""
        try:
            # 每50个预测步骤检查一次内存
            if hasattr(state, 'prediction_step') and state.prediction_step % 50 == 0:
                status = self.memory_optimizer.get_memory_status()
                
                pred_msg = f"预测步骤 {state.prediction_step}，内存状态: {status.allocated_gb:.2f}GB"
                logger.debug(pred_msg)
                
                # 预测阶段内存监控更加严格
                if not status.is_safe or status.allocated_gb > status.total_gb * 0.75:
                    warning_msg = f"预测步骤 {state.prediction_step} 内存使用过高 ({status.allocated_gb:.2f}GB)"
                    logger.warning(warning_msg)
                    
                    # 预测阶段立即清理内存
                    self.memory_optimizer.cleanup_gpu_memory()
                    
                    if self.logging_system:
                        self.logging_system.warning(warning_msg, "MEMORY_CALLBACK", {
                            "phase": "prediction",
                            "prediction_step": getattr(state, 'prediction_step', 0),
                            "memory_gb": status.allocated_gb,
                            "cleanup_triggered": True
                        })
                
                # 每100个预测步骤记录详细状态
                if state.prediction_step % 100 == 0 and self.logging_system:
                    self.logging_system.log_memory_status(status, state.global_step, f"PredictionStep_{state.prediction_step}")
                    
        except Exception as e:
            self._handle_callback_error(e, "on_prediction_step", {
                "step": getattr(state, 'global_step', 0),
                "prediction_step": getattr(state, 'prediction_step', 0),
                "phase": "prediction"
            })
    
    def on_substep_end(self, args, state, control, **kwargs):
        """子步骤结束时的回调"""
        try:
            # 子步骤结束时进行轻量级内存检查
            if self.step_count % 5 == 0:  # 每5个子步骤检查一次
                status = self.memory_optimizer.get_memory_status()
                
                # 只在内存使用过高时记录
                if not status.is_safe:
                    substep_msg = f"子步骤结束，内存状态: {status.allocated_gb:.2f}GB (不安全)"
                    logger.debug(substep_msg)
                    
                    if self.logging_system:
                        self.logging_system.debug(substep_msg, "MEMORY_CALLBACK", {
                            "step": self.step_count,
                            "global_step": getattr(state, 'global_step', 0),
                            "memory_gb": status.allocated_gb,
                            "memory_safe": status.is_safe,
                            "phase": "substep_end"
                        })
                    
                    # 如果内存使用非常高，触发清理
                    if status.allocated_gb > status.total_gb * 0.9:
                        cleanup_msg = f"子步骤结束时内存使用过高 ({status.allocated_gb:.2f}GB)，执行清理"
                        logger.warning(cleanup_msg)
                        self.memory_optimizer.cleanup_gpu_memory()
                        
                        if self.logging_system:
                            self.logging_system.warning(cleanup_msg, "MEMORY_CALLBACK", {
                                "step": self.step_count,
                                "cleanup_triggered": True,
                                "cleanup_reason": "substep_high_memory"
                            })
                
        except Exception as e:
            self._handle_callback_error(e, "on_substep_end", {
                "step": self.step_count,
                "global_step": getattr(state, 'global_step', 0),
                "phase": "substep_end"
            })
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """优化器步骤前的回调"""
        try:
            # 优化器步骤前进行内存检查
            status = self.memory_optimizer.get_memory_status()
            
            # 记录优化器步骤前的内存状态
            pre_opt_msg = f"优化器步骤前，内存状态: {status.allocated_gb:.2f}GB"
            logger.debug(pre_opt_msg)
            
            if self.logging_system:
                self.logging_system.debug(pre_opt_msg, "MEMORY_CALLBACK", {
                    "step": self.step_count,
                    "global_step": getattr(state, 'global_step', 0),
                    "memory_gb": status.allocated_gb,
                    "memory_safe": status.is_safe,
                    "phase": "pre_optimizer_step"
                })
            
            # 如果内存使用过高，在优化器步骤前清理
            if not status.is_safe or status.allocated_gb > status.total_gb * 0.8:
                cleanup_msg = f"优化器步骤前内存使用过高 ({status.allocated_gb:.2f}GB)，执行预防性清理"
                logger.warning(cleanup_msg)
                self.memory_optimizer.cleanup_gpu_memory()
                
                if self.logging_system:
                    self.logging_system.warning(cleanup_msg, "MEMORY_CALLBACK", {
                        "step": self.step_count,
                        "cleanup_triggered": True,
                        "cleanup_reason": "pre_optimizer_high_memory"
                    })
                
        except Exception as e:
            self._handle_callback_error(e, "on_pre_optimizer_step", {
                "step": self.step_count,
                "global_step": getattr(state, 'global_step', 0),
                "phase": "pre_optimizer_step"
            })
    
    def _handle_callback_error(self, error: Exception, method_name: str, context: dict = None):
        """统一的回调错误处理方法"""
        try:
            # 构建错误消息
            error_msg = f"{method_name} 回调方法出错: {str(error)}"
            logger.warning(error_msg)
            
            # 添加上下文信息
            error_context = {
                "method": method_name,
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
            
            if context:
                error_context.update(context)
            
            # 记录到日志系统
            if self.logging_system:
                self.logging_system.error(error_msg, "MEMORY_CALLBACK", error_context)
            
            # 如果是内存相关错误，尝试紧急清理
            if any(keyword in str(error).lower() for keyword in ['memory', 'cuda', 'out of memory', 'oom']):
                emergency_msg = f"{method_name} 检测到内存相关错误，执行紧急清理"
                logger.warning(emergency_msg)
                
                try:
                    self.memory_optimizer.cleanup_gpu_memory()
                    if self.logging_system:
                        self.logging_system.warning(emergency_msg, "MEMORY_CALLBACK", {
                            "emergency_cleanup": True,
                            "original_error": str(error)
                        })
                except Exception as cleanup_error:
                    cleanup_error_msg = f"紧急内存清理失败: {cleanup_error}"
                    logger.error(cleanup_error_msg)
                    if self.logging_system:
                        self.logging_system.error(cleanup_error_msg, "MEMORY_CALLBACK")
                        
        except Exception as handler_error:
            # 如果错误处理器本身出错，只能使用基本日志
            logger.error(f"错误处理器失败: {handler_error}")
    
    def _log_epoch_memory_summary(self, epoch_metrics: EpochMemoryMetrics, global_step: int):
        """记录轮次内存摘要"""
        try:
            summary_msg = (
                f"轮次 {epoch_metrics.epoch} 内存摘要: "
                f"开始 {epoch_metrics.start_memory_gb:.2f}GB, "
                f"结束 {epoch_metrics.end_memory_gb:.2f}GB, "
                f"峰值 {epoch_metrics.peak_memory_gb:.2f}GB, "
                f"变化 {epoch_metrics.memory_delta_gb:+.2f}GB, "
                f"清理次数 {epoch_metrics.cleanup_count}, "
                f"效率 {epoch_metrics.memory_efficiency:.3f}"
            )
            
            logger.info(summary_msg)
            
            if self.logging_system:
                self.logging_system.info(summary_msg, "MEMORY_CALLBACK", {
                    "epoch": epoch_metrics.epoch,
                    "start_memory_gb": epoch_metrics.start_memory_gb,
                    "end_memory_gb": epoch_metrics.end_memory_gb,
                    "peak_memory_gb": epoch_metrics.peak_memory_gb,
                    "memory_delta_gb": epoch_metrics.memory_delta_gb,
                    "cleanup_count": epoch_metrics.cleanup_count,
                    "memory_efficiency": epoch_metrics.memory_efficiency,
                    "step": global_step
                })
                
        except Exception as e:
            logger.warning(f"记录轮次内存摘要失败: {e}")
    
    def _log_memory_status(self, status: MemoryStatus, step: int, phase: str, additional_context: dict = None):
        """标准化内存状态日志记录的工具方法"""
        try:
            # 构建基本消息
            msg = f"{phase} - 步骤 {step}: 内存 {status.allocated_gb:.2f}GB"
            if not status.is_safe:
                msg += " (警告: 内存使用过高)"
            
            # 选择日志级别
            if status.is_safe:
                logger.debug(msg)
            else:
                logger.warning(msg)
            
            # 记录到日志系统
            if self.logging_system:
                context = {
                    "phase": phase,
                    "step": step,
                    "memory_gb": status.allocated_gb,
                    "memory_cached_gb": status.cached_gb,
                    "memory_available_gb": status.available_gb,
                    "memory_safe": status.is_safe
                }
                
                if additional_context:
                    context.update(additional_context)
                
                self.logging_system.log_memory_status(status, step, phase)
                
                if status.is_safe:
                    self.logging_system.debug(msg, "MEMORY_CALLBACK", context)
                else:
                    self.logging_system.warning(msg, "MEMORY_CALLBACK", context)
                    
        except Exception as e:
            logger.warning(f"标准化内存日志记录失败: {e}")
    
    def _update_callback_state(self, step_increment: int = 0, epoch_increment: int = 0):
        """回调状态管理方法"""
        try:
            if step_increment > 0:
                self.step_count += step_increment
            
            if epoch_increment > 0:
                self.epoch_count += epoch_increment
                # 重置轮次相关状态
                self.epoch_start_memory = None
                self.epoch_peak_memory = None
                self.epoch_cleanup_count = 0
                
        except Exception as e:
            logger.warning(f"更新回调状态失败: {e}")
    
    def _should_trigger_cleanup(self, status: MemoryStatus, threshold_gb: float = None) -> bool:
        """内存清理触发逻辑"""
        try:
            # 默认阈值：总内存的75%
            if threshold_gb is None:
                threshold_gb = status.total_gb * 0.75
            
            # 检查多个条件
            conditions = [
                not status.is_safe,  # 内存不安全
                status.allocated_gb > threshold_gb,  # 超过阈值
                status.allocated_gb > status.total_gb * 0.9,  # 超过90%总内存
            ]
            
            return any(conditions)
            
        except Exception as e:
            logger.warning(f"内存清理触发检查失败: {e}")
            return False
    
    def _format_memory_log(self, phase: str, memory_gb: float, additional_info: str = "") -> str:
        """统一的日志格式化方法"""
        try:
            base_msg = f"[{phase}] 内存: {memory_gb:.2f}GB"
            if additional_info:
                base_msg += f" - {additional_info}"
            return base_msg
        except Exception as e:
            logger.warning(f"日志格式化失败: {e}")
            return f"[{phase}] 内存状态记录失败"
    
    def _check_memory_leak(self):
        """检测内存泄漏"""
        if len(self.memory_history) < 10:
            return
        
        # 取最近10个内存记录
        recent_memory = [mem for _, mem in self.memory_history[-10:]]
        
        # 检查内存是否持续增长
        increasing_count = 0
        for i in range(1, len(recent_memory)):
            if recent_memory[i] > recent_memory[i-1]:
                increasing_count += 1
        
        # 如果80%的时间内存都在增长，可能存在内存泄漏
        if increasing_count >= 8:
            current_memory = recent_memory[-1]
            initial_memory = recent_memory[0]
            growth = current_memory - initial_memory
            
            if growth > 1.0:  # 增长超过1GB
                leak_msg = f"检测到可能的内存泄漏: 内存增长 {growth:.2f}GB"
                logger.warning(leak_msg)
                if self.logging_system:
                    self.logging_system.warning(leak_msg, "MEMORY_CALLBACK", {
                        "memory_growth_gb": growth,
                        "current_memory_gb": current_memory,
                        "initial_memory_gb": initial_memory
                    })
                
                # 强制进行内存清理
                if self.step_count - self.last_cleanup_step > 50:
                    self.memory_optimizer.cleanup_gpu_memory()
                    self.last_cleanup_step = self.step_count


    def run_complete_training_pipeline(self,
                                      model: PreTrainedModel,
                                      train_dataset: Dataset,
                                      tokenizer: PreTrainedTokenizer,
                                      data_collator: Any,
                                      eval_dataset: Optional[Dataset] = None,
                                      resume_from_checkpoint: Optional[str] = None) -> Tuple[Trainer, Dict[str, Any]]:
        """
        运行完整的训练管道，包括监控、错误处理和保存
        
        Args:
            model: 训练模型
            train_dataset: 训练数据集
            tokenizer: 分词器
            data_collator: 数据整理器
            eval_dataset: 可选的评估数据集
            resume_from_checkpoint: 可选的检查点路径
            
        Returns:
            Tuple[Trainer, Dict[str, Any]]: 训练器和训练报告
        """
        logger.info("开始完整训练管道...")
        
        # 生成训练前报告
        pre_training_report = self._generate_pre_training_report(model, train_dataset)
        logger.info("训练前检查完成")
        
        try:
            # 1. 创建训练器
            trainer = self.create_trainer(
                model=model,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                eval_dataset=eval_dataset
            )
            
            # 2. 执行训练
            trainer = self.train_model(trainer, resume_from_checkpoint)
            
            # 3. 保存模型
            self.save_model(trainer)
            
            # 4. 生成训练后报告
            post_training_report = self._generate_post_training_report(trainer)
            
            # 5. 合并报告
            complete_report = {
                "pre_training": pre_training_report,
                "post_training": post_training_report,
                "training_config": self.config.__dict__,
                "success": True
            }
            
            logger.info("完整训练管道执行成功")
            return trainer, complete_report
            
        except Exception as e:
            logger.error(f"训练管道执行失败: {e}")
            
            # 生成错误报告
            error_report = {
                "pre_training": pre_training_report,
                "error": str(e),
                "training_config": self.config.__dict__,
                "success": False,
                "disk_usage": self.get_disk_usage_report(self.config.output_dir)
            }
            
            return None, error_report
    
    def _generate_pre_training_report(self, model: PreTrainedModel, dataset: Dataset) -> Dict[str, Any]:
        """
        生成训练前报告
        
        Args:
            model: 模型
            dataset: 数据集
            
        Returns:
            Dict[str, Any]: 训练前报告
        """
        try:
            # 内存状态
            memory_status = self.memory_optimizer.get_memory_status()
            
            # 模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 数据集信息
            dataset_info = {
                "size": len(dataset),
                "columns": dataset.column_names if hasattr(dataset, 'column_names') else []
            }
            
            # 磁盘使用情况
            disk_usage = self.get_disk_usage_report(self.config.output_dir)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "memory_status": {
                    "allocated_gb": memory_status.allocated_gb,
                    "available_gb": memory_status.available_gb,
                    "is_safe": memory_status.is_safe
                },
                "model_info": {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
                },
                "dataset_info": dataset_info,
                "disk_usage": disk_usage,
                "training_config": {
                    "batch_size": self.config.batch_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "learning_rate": self.config.learning_rate,
                    "num_epochs": self.config.num_epochs
                }
            }
            
            return report
            
        except Exception as e:
            logger.warning(f"生成训练前报告时出错: {e}")
            return {"error": str(e)}
    
    def _generate_post_training_report(self, trainer: Trainer) -> Dict[str, Any]:
        """
        生成训练后报告
        
        Args:
            trainer: 训练器
            
        Returns:
            Dict[str, Any]: 训练后报告
        """
        try:
            # 内存状态
            memory_status = self.memory_optimizer.get_memory_status()
            
            # 训练状态
            training_state = trainer.state
            
            # 磁盘使用情况
            disk_usage = self.get_disk_usage_report(self.config.output_dir)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "memory_status": {
                    "allocated_gb": memory_status.allocated_gb,
                    "available_gb": memory_status.available_gb,
                    "is_safe": memory_status.is_safe
                },
                "training_results": {
                    "global_step": training_state.global_step,
                    "epoch": training_state.epoch,
                    "train_loss": getattr(training_state, 'log_history', [{}])[-1].get('train_loss', None) if training_state.log_history else None,
                    "eval_loss": getattr(training_state, 'log_history', [{}])[-1].get('eval_loss', None) if training_state.log_history else None
                },
                "disk_usage": disk_usage,
                "model_saved": True
            }
            
            return report
            
        except Exception as e:
            logger.warning(f"生成训练后报告时出错: {e}")
            return {"error": str(e)}


def create_training_config(
    output_dir: str = "./qwen3-finetuned",
    max_memory_gb: float = 13.0,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-5,
    num_epochs: int = 10,
    **kwargs
) -> TrainingConfig:
    """
    创建训练配置的便捷函数
    
    Args:
        output_dir: 输出目录
        max_memory_gb: 最大内存限制
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        num_epochs: 训练轮数
        **kwargs: 其他配置参数
        
    Returns:
        TrainingConfig: 训练配置实例
    """
    config = TrainingConfig(
        output_dir=output_dir,
        max_memory_gb=max_memory_gb,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    # 更新其他参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # 测试训练引擎
    logging.basicConfig(level=logging.INFO)
    
    # 创建训练配置
    config = create_training_config(
        output_dir="./test_output",
        batch_size=2,
        num_epochs=1
    )
    
    # 创建训练引擎
    engine = TrainingEngine(config)
    
    # 创建训练参数
    training_args = engine.create_training_args()
    print(f"训练参数创建成功: {training_args.output_dir}")
