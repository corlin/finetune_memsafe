"""
训练引擎集成

与TrainingEngine训练引擎集成，支持验证集评估、实时性能监控和自动评估调度。
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from pathlib import Path

from .evaluation_engine import EvaluationEngine
from .training_monitor import TrainingMonitor, EarlyStoppingCallback
from .data_models import EvaluationConfig, EvaluationResult
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class TrainingEngineIntegration:
    """
    训练引擎集成器
    
    提供与现有训练引擎的集成功能：
    - 扩展TrainingEngine支持验证集评估
    - 集成实时性能监控到训练循环
    - 实现训练过程中的自动评估调度
    - 建立训练完成后的自动全面评估
    """
    
    def __init__(self, 
                 evaluation_config: Optional[EvaluationConfig] = None,
                 monitor_config: Optional[Dict[str, Any]] = None,
                 output_dir: str = "./training_evaluation"):
        """
        初始化训练引擎集成器
        
        Args:
            evaluation_config: 评估配置
            monitor_config: 监控配置
            output_dir: 输出目录
        """
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.evaluation_engine = EvaluationEngine(self.evaluation_config)
        self.training_monitor = TrainingMonitor(**(monitor_config or {}))
        self.metrics_calculator = MetricsCalculator()
        
        # 集成状态
        self.is_monitoring = False
        self.evaluation_thread = None
        self.evaluation_results = []
        
        logger.info("TrainingEngineIntegration初始化完成")
    
    def enhance_training_engine(self, training_engine_class):
        """
        增强现有的TrainingEngine类
        
        Args:
            training_engine_class: 现有的TrainingEngine类
            
        Returns:
            增强后的TrainingEngine类
        """
        class EnhancedTrainingEngine(training_engine_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # 集成评估组件
                eval_config = kwargs.get('evaluation_config')
                monitor_config = kwargs.get('monitor_config')
                output_dir = kwargs.get('evaluation_output_dir', './training_evaluation')
                
                self.integration = TrainingEngineIntegration(
                    evaluation_config=eval_config,
                    monitor_config=monitor_config,
                    output_dir=output_dir
                )
                
                # 添加早停回调
                self.early_stopping_callback = EarlyStoppingCallback(
                    self.integration.training_monitor
                )
                
                # 评估数据集
                self.eval_datasets = {}
            
            def set_evaluation_datasets(self, datasets: Dict[str, Any]):
                """
                设置评估数据集
                
                Args:
                    datasets: 评估数据集字典
                """
                self.eval_datasets = datasets
                logger.info(f"设置评估数据集: {list(datasets.keys())}")
            
            def train_with_evaluation(self, 
                                    train_dataset,
                                    val_dataset=None,
                                    eval_frequency: int = 1,
                                    save_best_model: bool = True,
                                    **kwargs):
                """
                带评估的训练
                
                Args:
                    train_dataset: 训练数据集
                    val_dataset: 验证数据集
                    eval_frequency: 评估频率（每N个epoch评估一次）
                    save_best_model: 是否保存最佳模型
                    **kwargs: 其他训练参数
                """
                return self.integration.train_with_evaluation(
                    training_engine=self,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    eval_frequency=eval_frequency,
                    save_best_model=save_best_model,
                    **kwargs
                )
            
            def evaluate_during_training(self, epoch: int, logs: Dict[str, float]):
                """
                训练过程中的评估
                
                Args:
                    epoch: 当前epoch
                    logs: 训练日志
                """
                return self.integration.evaluate_during_training(
                    training_engine=self,
                    epoch=epoch,
                    logs=logs,
                    eval_datasets=self.eval_datasets
                )
            
            def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> bool:
                """
                epoch结束时的回调
                
                Args:
                    epoch: epoch编号
                    logs: 训练日志
                    
                Returns:
                    是否应该停止训练
                """
                # 调用原始的epoch结束处理
                if hasattr(super(), 'on_epoch_end'):
                    super().on_epoch_end(epoch, logs)
                
                # 执行评估和监控
                should_stop = self.early_stopping_callback.on_epoch_end(epoch, logs)
                
                # 定期评估
                if hasattr(self, 'eval_datasets') and self.eval_datasets:
                    self.evaluate_during_training(epoch, logs)
                
                return should_stop
            
            def finalize_training(self):
                """
                完成训练后的处理
                """
                return self.integration.finalize_training(
                    training_engine=self,
                    eval_datasets=self.eval_datasets
                )
        
        return EnhancedTrainingEngine 
   
    def train_with_evaluation(self, 
                            training_engine,
                            train_dataset,
                            val_dataset=None,
                            eval_frequency: int = 1,
                            save_best_model: bool = True,
                            **kwargs):
        """
        带评估的训练
        
        Args:
            training_engine: 训练引擎实例
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            eval_frequency: 评估频率
            save_best_model: 是否保存最佳模型
            **kwargs: 其他训练参数
        """
        logger.info("开始带评估的训练")
        
        # 启动监控
        self.is_monitoring = True
        
        try:
            # 如果有验证数据集，设置为评估数据集
            if val_dataset:
                eval_datasets = {"validation": val_dataset}
                if hasattr(training_engine, 'set_evaluation_datasets'):
                    training_engine.set_evaluation_datasets(eval_datasets)
            
            # 开始训练（这里需要调用原始的训练方法）
            if hasattr(training_engine, 'train'):
                training_result = training_engine.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    **kwargs
                )
            else:
                logger.warning("训练引擎没有train方法")
                training_result = None
            
            # 训练完成后的全面评估
            final_evaluation = self.finalize_training(training_engine, eval_datasets if val_dataset else {})
            
            return {
                "training_result": training_result,
                "final_evaluation": final_evaluation,
                "training_history": self.training_monitor.history,
                "evaluation_results": self.evaluation_results
            }
            
        finally:
            self.is_monitoring = False
    
    def evaluate_during_training(self, 
                                training_engine,
                                epoch: int,
                                logs: Dict[str, float],
                                eval_datasets: Dict[str, Any]):
        """
        训练过程中的评估
        
        Args:
            training_engine: 训练引擎实例
            epoch: 当前epoch
            logs: 训练日志
            eval_datasets: 评估数据集
        """
        if not eval_datasets:
            return
        
        try:
            # 获取当前模型和分词器
            model = getattr(training_engine, 'model', None)
            tokenizer = getattr(training_engine, 'tokenizer', None)
            
            if model is None or tokenizer is None:
                logger.warning("无法获取模型或分词器，跳过评估")
                return
            
            # 执行快速评估
            eval_result = self.evaluation_engine.evaluate_model(
                model=model,
                tokenizer=tokenizer,
                datasets=eval_datasets,
                model_name=f"epoch_{epoch}"
            )
            
            # 记录评估结果
            self.evaluation_results.append({
                "epoch": epoch,
                "evaluation_result": eval_result,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Epoch {epoch} 评估完成，总分: {eval_result.metrics.get('overall_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"训练过程中评估失败: {e}")
    
    def finalize_training(self, 
                         training_engine,
                         eval_datasets: Dict[str, Any]) -> Optional[EvaluationResult]:
        """
        完成训练后的处理
        
        Args:
            training_engine: 训练引擎实例
            eval_datasets: 评估数据集
            
        Returns:
            最终评估结果
        """
        logger.info("开始训练完成后的全面评估")
        
        try:
            # 获取最终模型
            model = getattr(training_engine, 'model', None)
            tokenizer = getattr(training_engine, 'tokenizer', None)
            
            if model is None or tokenizer is None:
                logger.warning("无法获取模型或分词器，跳过最终评估")
                return None
            
            # 执行全面评估
            final_eval_result = self.evaluation_engine.evaluate_model(
                model=model,
                tokenizer=tokenizer,
                datasets=eval_datasets,
                model_name="final_model"
            )
            
            # 生成训练监控报告
            monitor_report = self.training_monitor.generate_monitoring_report()
            
            # 生成训练曲线
            curves_path = self.training_monitor.generate_training_curves(
                str(self.output_dir / "training_curves.png")
            )
            
            # 保存训练历史
            history_path = self.training_monitor.save_training_history(
                str(self.output_dir / "training_history.json")
            )
            
            # 保存最终评估结果
            self.evaluation_engine.save_evaluation_result(
                final_eval_result,
                str(self.output_dir / "final_evaluation.json")
            )
            
            logger.info("训练完成后的处理已完成")
            
            return final_eval_result
            
        except Exception as e:
            logger.error(f"训练完成后处理失败: {e}")
            return None
    
    def create_evaluation_scheduler(self, 
                                  training_engine,
                                  eval_datasets: Dict[str, Any],
                                  eval_interval: int = 100) -> threading.Thread:
        """
        创建评估调度器
        
        Args:
            training_engine: 训练引擎实例
            eval_datasets: 评估数据集
            eval_interval: 评估间隔（秒）
            
        Returns:
            评估线程
        """
        def evaluation_worker():
            """评估工作线程"""
            while self.is_monitoring:
                try:
                    # 检查是否需要评估
                    current_epoch = getattr(training_engine, 'current_epoch', 0)
                    
                    if current_epoch > 0:
                        # 执行评估
                        self.evaluate_during_training(
                            training_engine=training_engine,
                            epoch=current_epoch,
                            logs={},
                            eval_datasets=eval_datasets
                        )
                    
                    # 等待下次评估
                    time.sleep(eval_interval)
                    
                except Exception as e:
                    logger.error(f"评估调度器错误: {e}")
                    time.sleep(eval_interval)
        
        thread = threading.Thread(target=evaluation_worker, daemon=True)
        return thread
    
    def create_performance_monitor(self, 
                                 training_engine,
                                 monitor_interval: int = 60) -> threading.Thread:
        """
        创建性能监控器
        
        Args:
            training_engine: 训练引擎实例
            monitor_interval: 监控间隔（秒）
            
        Returns:
            监控线程
        """
        def monitor_worker():
            """监控工作线程"""
            while self.is_monitoring:
                try:
                    # 收集性能指标
                    performance_metrics = self._collect_performance_metrics(training_engine)
                    
                    # 记录到日志
                    if performance_metrics:
                        logger.info(f"性能监控: {performance_metrics}")
                    
                    # 等待下次监控
                    time.sleep(monitor_interval)
                    
                except Exception as e:
                    logger.error(f"性能监控器错误: {e}")
                    time.sleep(monitor_interval)
        
        thread = threading.Thread(target=monitor_worker, daemon=True)
        return thread
    
    def _collect_performance_metrics(self, training_engine) -> Dict[str, Any]:
        """
        收集性能指标
        
        Args:
            training_engine: 训练引擎实例
            
        Returns:
            性能指标字典
        """
        metrics = {}
        
        try:
            # GPU使用率
            if hasattr(training_engine, 'device') and 'cuda' in str(training_engine.device):
                try:
                    import torch
                    if torch.cuda.is_available():
                        metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                        metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3   # GB
                except:
                    pass
            
            # CPU和内存使用率
            try:
                import psutil
                process = psutil.Process()
                metrics["cpu_percent"] = process.cpu_percent()
                metrics["memory_mb"] = process.memory_info().rss / 1024**2
            except:
                pass
            
            # 训练状态
            metrics["current_epoch"] = getattr(training_engine, 'current_epoch', 0)
            metrics["is_training"] = getattr(training_engine, 'is_training', False)
            
        except Exception as e:
            logger.warning(f"收集性能指标失败: {e}")
        
        return metrics
    
    def setup_automatic_evaluation(self, 
                                 training_engine,
                                 eval_datasets: Dict[str, Any],
                                 eval_frequency: int = 1,
                                 eval_interval: int = 300):
        """
        设置自动评估
        
        Args:
            training_engine: 训练引擎实例
            eval_datasets: 评估数据集
            eval_frequency: 评估频率（每N个epoch）
            eval_interval: 评估间隔（秒）
        """
        logger.info("设置自动评估")
        
        # 启动评估调度器
        if eval_interval > 0:
            self.evaluation_thread = self.create_evaluation_scheduler(
                training_engine, eval_datasets, eval_interval
            )
            self.evaluation_thread.start()
        
        # 设置epoch级别的评估
        original_on_epoch_end = getattr(training_engine, 'on_epoch_end', None)
        
        def enhanced_on_epoch_end(epoch, logs):
            # 调用原始方法
            if original_on_epoch_end:
                result = original_on_epoch_end(epoch, logs)
            else:
                result = False
            
            # 执行评估
            if epoch % eval_frequency == 0:
                self.evaluate_during_training(training_engine, epoch, logs, eval_datasets)
            
            return result
        
        # 替换方法
        training_engine.on_epoch_end = enhanced_on_epoch_end
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        获取评估摘要
        
        Returns:
            评估摘要
        """
        if not self.evaluation_results:
            return {"message": "没有评估结果"}
        
        # 提取关键指标
        epochs = [result["epoch"] for result in self.evaluation_results]
        scores = [result["evaluation_result"].metrics.get("overall_score", 0) 
                 for result in self.evaluation_results]
        
        # 找到最佳性能
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_result = self.evaluation_results[best_idx]
        
        summary = {
            "total_evaluations": len(self.evaluation_results),
            "epochs_evaluated": epochs,
            "best_performance": {
                "epoch": best_result["epoch"],
                "score": scores[best_idx],
                "timestamp": best_result["timestamp"].isoformat()
            },
            "performance_trend": {
                "initial_score": scores[0] if scores else 0,
                "final_score": scores[-1] if scores else 0,
                "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0
            },
            "training_monitor_summary": self.training_monitor.generate_monitoring_report()
        }
        
        return summary


def create_enhanced_training_engine(base_training_engine_class=None):
    """
    创建增强的训练引擎类
    
    Args:
        base_training_engine_class: 基础训练引擎类，如果为None则创建新类
        
    Returns:
        增强的训练引擎类
    """
    if base_training_engine_class is None:
        # 创建基础训练引擎类
        class BaseTrainingEngine:
            def __init__(self, **kwargs):
                self.config = kwargs
                self.model = None
                self.tokenizer = None
                self.current_epoch = 0
                self.is_training = False
            
            def train(self, train_dataset, val_dataset=None, **kwargs):
                """基础训练方法"""
                self.is_training = True
                # 这里应该实现实际的训练逻辑
                pass
            
            def on_epoch_end(self, epoch, logs):
                """epoch结束回调"""
                self.current_epoch = epoch
                return False
    
        base_training_engine_class = BaseTrainingEngine
    
    # 使用集成器增强训练引擎
    integration = TrainingEngineIntegration()
    enhanced_class = integration.enhance_training_engine(base_training_engine_class)
    
    return enhanced_class