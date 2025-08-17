"""
训练监控器

实现训练过程中的实时评估、验证集性能监控、过拟合检测和早停建议系统。
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import json

from .data_models import convert_numpy_types

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    训练监控器
    
    提供训练监控功能：
    - 集成到训练过程中的实时评估
    - 验证集性能监控和趋势分析
    - 过拟合检测和早停建议系统
    - 训练曲线和性能趋势报告生成
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.001,
                 monitor_metric: str = "val_loss",
                 mode: str = "min",
                 output_dir: str = "./training_logs"):
        """
        初始化训练监控器
        
        Args:
            patience: 早停耐心值（epoch数）
            min_delta: 最小改进阈值
            monitor_metric: 监控的指标名称
            mode: 监控模式 ('min' 或 'max')
            output_dir: 输出目录
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 监控状态
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rate": [],
            "timestamp": []
        }
        
        # 早停相关
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
        # 过拟合检测
        self.overfitting_threshold = 0.1  # 训练和验证损失差异阈值
        self.overfitting_patience = 5     # 过拟合检测耐心值
        self.overfitting_count = 0
        
        # 性能趋势分析
        self.trend_window = 5  # 趋势分析窗口大小
        self.performance_trend = deque(maxlen=self.trend_window)
        
        logger.info(f"TrainingMonitor初始化完成，监控指标: {monitor_metric}, 模式: {mode}")
    
    def log_epoch(self, 
                  epoch: int,
                  train_loss: float,
                  val_loss: float,
                  train_metrics: Optional[Dict[str, float]] = None,
                  val_metrics: Optional[Dict[str, float]] = None,
                  learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        记录一个epoch的训练结果
        
        Args:
            epoch: epoch编号
            train_loss: 训练损失
            val_loss: 验证损失
            train_metrics: 训练指标
            val_metrics: 验证指标
            learning_rate: 学习率
            
        Returns:
            监控状态和建议
        """
        # 记录历史数据
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["train_metrics"].append(train_metrics or {})
        self.history["val_metrics"].append(val_metrics or {})
        self.history["learning_rate"].append(learning_rate)
        self.history["timestamp"].append(datetime.now().isoformat())
        
        # 获取当前监控指标值
        current_score = self._get_current_score(val_loss, val_metrics)
        
        # 更新性能趋势
        self.performance_trend.append(current_score)
        
        # 早停检查
        early_stop_info = self._check_early_stopping(current_score, epoch)
        
        # 过拟合检测
        overfitting_info = self._detect_overfitting(train_loss, val_loss)
        
        # 性能趋势分析
        trend_info = self._analyze_performance_trend()
        
        # 学习率建议
        lr_suggestion = self._suggest_learning_rate_adjustment(train_loss, val_loss)
        
        # 生成监控报告
        monitor_report = {
            "epoch": epoch,
            "current_score": float(current_score),
            "best_score": float(self.best_score) if self.best_score is not None else None,
            "best_epoch": self.best_epoch,
            "early_stopping": early_stop_info,
            "overfitting": overfitting_info,
            "performance_trend": trend_info,
            "learning_rate_suggestion": lr_suggestion,
            "should_stop": self.should_stop
        }
        
        # 保存监控日志
        self._save_monitoring_log(monitor_report)
        
        logger.info(f"Epoch {epoch}: {self.monitor_metric}={current_score:.4f}, "
                   f"Best={self.best_score:.4f} (Epoch {self.best_epoch})")
        
        if self.should_stop:
            logger.warning(f"建议早停：连续 {self.wait} 个epoch无改进")
        
        return monitor_report
    
    def _get_current_score(self, val_loss: float, val_metrics: Optional[Dict[str, float]]) -> float:
        """获取当前监控指标值"""
        if self.monitor_metric == "val_loss":
            return val_loss
        elif val_metrics and self.monitor_metric in val_metrics:
            return val_metrics[self.monitor_metric]
        else:
            # 默认使用验证损失
            return val_loss
    
    def _check_early_stopping(self, current_score: float, epoch: int) -> Dict[str, Any]:
        """检查早停条件"""
        early_stop_info = {
            "triggered": False,
            "reason": "",
            "wait_epochs": self.wait,
            "patience": self.patience
        }
        
        # 判断是否有改进
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            improved = True
        else:
            if self.mode == "min":
                improved = current_score < (self.best_score - self.min_delta)
            else:  # mode == "max"
                improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            early_stop_info["reason"] = "性能改进"
        else:
            self.wait += 1
            early_stop_info["reason"] = f"连续 {self.wait} 个epoch无改进"
            
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                early_stop_info["triggered"] = True
                early_stop_info["reason"] = f"达到耐心值 {self.patience}，建议早停"
        
        return early_stop_info
    
    def _detect_overfitting(self, train_loss: float, val_loss: float) -> Dict[str, Any]:
        """检测过拟合"""
        overfitting_info = {
            "detected": False,
            "severity": "none",
            "train_val_gap": float(val_loss - train_loss),
            "suggestion": ""
        }
        
        # 计算训练和验证损失的差异
        gap = val_loss - train_loss
        
        if gap > self.overfitting_threshold:
            self.overfitting_count += 1
            
            if gap > self.overfitting_threshold * 3:
                overfitting_info["severity"] = "severe"
                overfitting_info["suggestion"] = "严重过拟合，建议增加正则化或减少模型复杂度"
            elif gap > self.overfitting_threshold * 2:
                overfitting_info["severity"] = "moderate"
                overfitting_info["suggestion"] = "中度过拟合，建议调整正则化参数"
            else:
                overfitting_info["severity"] = "mild"
                overfitting_info["suggestion"] = "轻度过拟合，建议监控趋势"
            
            if self.overfitting_count >= self.overfitting_patience:
                overfitting_info["detected"] = True
        else:
            self.overfitting_count = max(0, self.overfitting_count - 1)
        
        return overfitting_info
    
    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """分析性能趋势"""
        trend_info = {
            "direction": "stable",
            "slope": 0.0,
            "confidence": 0.0,
            "suggestion": ""
        }
        
        if len(self.performance_trend) < 3:
            return trend_info
        
        # 计算趋势斜率
        x = np.arange(len(self.performance_trend))
        y = np.array(self.performance_trend)
        
        try:
            # 线性回归计算趋势
            slope, intercept = np.polyfit(x, y, 1)
            
            # 计算相关系数作为置信度
            correlation = np.corrcoef(x, y)[0, 1]
            confidence = abs(correlation)
            
            trend_info["slope"] = float(slope)
            trend_info["confidence"] = float(confidence)
            
            # 判断趋势方向
            if self.mode == "min":
                if slope < -0.001 and confidence > 0.5:
                    trend_info["direction"] = "improving"
                    trend_info["suggestion"] = "性能持续改善，继续训练"
                elif slope > 0.001 and confidence > 0.5:
                    trend_info["direction"] = "degrading"
                    trend_info["suggestion"] = "性能下降，考虑调整学习率或早停"
                else:
                    trend_info["direction"] = "stable"
                    trend_info["suggestion"] = "性能稳定，继续观察"
            else:  # mode == "max"
                if slope > 0.001 and confidence > 0.5:
                    trend_info["direction"] = "improving"
                    trend_info["suggestion"] = "性能持续改善，继续训练"
                elif slope < -0.001 and confidence > 0.5:
                    trend_info["direction"] = "degrading"
                    trend_info["suggestion"] = "性能下降，考虑调整学习率或早停"
                else:
                    trend_info["direction"] = "stable"
                    trend_info["suggestion"] = "性能稳定，继续观察"
                    
        except Exception as e:
            logger.warning(f"趋势分析失败: {e}")
        
        return trend_info
    
    def _suggest_learning_rate_adjustment(self, train_loss: float, val_loss: float) -> Dict[str, Any]:
        """建议学习率调整"""
        lr_suggestion = {
            "action": "maintain",
            "factor": 1.0,
            "reason": ""
        }
        
        # 检查最近几个epoch的损失变化
        if len(self.history["train_loss"]) >= 3:
            recent_train_losses = self.history["train_loss"][-3:]
            recent_val_losses = self.history["val_loss"][-3:]
            
            # 计算损失变化趋势
            train_trend = np.mean(np.diff(recent_train_losses))
            val_trend = np.mean(np.diff(recent_val_losses))
            
            # 如果训练损失不再下降，可能需要降低学习率
            if train_trend > 0.001:
                lr_suggestion["action"] = "decrease"
                lr_suggestion["factor"] = 0.5
                lr_suggestion["reason"] = "训练损失不再下降，建议降低学习率"
            
            # 如果验证损失持续上升，建议降低学习率
            elif val_trend > 0.01:
                lr_suggestion["action"] = "decrease"
                lr_suggestion["factor"] = 0.3
                lr_suggestion["reason"] = "验证损失持续上升，建议大幅降低学习率"
            
            # 如果损失下降很慢，可能可以提高学习率
            elif abs(train_trend) < 0.0001 and train_loss > 0.1:
                lr_suggestion["action"] = "increase"
                lr_suggestion["factor"] = 1.5
                lr_suggestion["reason"] = "损失下降缓慢，可以尝试提高学习率"
        
        return lr_suggestion
    
    def generate_training_curves(self, save_path: Optional[str] = None) -> str:
        """
        生成训练曲线图
        
        Args:
            save_path: 保存路径
            
        Returns:
            图表文件路径
        """
        if not self.history["epoch"]:
            logger.warning("没有训练历史数据，无法生成曲线图")
            return ""
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"training_curves_{timestamp}.png"
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('训练监控曲线', fontsize=16)
            
            epochs = self.history["epoch"]
            
            # 损失曲线
            axes[0, 0].plot(epochs, self.history["train_loss"], label='训练损失', color='blue')
            axes[0, 0].plot(epochs, self.history["val_loss"], label='验证损失', color='red')
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 标记最佳点
            if self.best_epoch > 0:
                axes[0, 0].axvline(x=self.best_epoch, color='green', linestyle='--', 
                                  label=f'最佳 (Epoch {self.best_epoch})')
                axes[0, 0].legend()
            
            # 学习率曲线
            if any(lr is not None for lr in self.history["learning_rate"]):
                lr_values = [lr for lr in self.history["learning_rate"] if lr is not None]
                lr_epochs = [epochs[i] for i, lr in enumerate(self.history["learning_rate"]) if lr is not None]
                axes[0, 1].plot(lr_epochs, lr_values, label='学习率', color='orange')
                axes[0, 1].set_title('学习率变化')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Learning Rate')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True)
            
            # 训练-验证损失差异
            loss_gap = [val - train for train, val in zip(self.history["train_loss"], self.history["val_loss"])]
            axes[1, 0].plot(epochs, loss_gap, label='验证-训练损失差', color='purple')
            axes[1, 0].axhline(y=self.overfitting_threshold, color='red', linestyle='--', 
                              label=f'过拟合阈值 ({self.overfitting_threshold})')
            axes[1, 0].set_title('过拟合监控')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Gap')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 性能趋势
            if len(self.performance_trend) > 1:
                trend_epochs = epochs[-len(self.performance_trend):]
                axes[1, 1].plot(trend_epochs, list(self.performance_trend), 
                               label=f'{self.monitor_metric}', color='green', marker='o')
                axes[1, 1].set_title('性能趋势')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel(self.monitor_metric)
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"训练曲线已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"生成训练曲线失败: {e}")
            return ""
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """
        生成监控报告
        
        Returns:
            监控报告字典
        """
        if not self.history["epoch"]:
            return {"error": "没有训练历史数据"}
        
        try:
            # 基本统计
            total_epochs = len(self.history["epoch"])
            final_train_loss = self.history["train_loss"][-1]
            final_val_loss = self.history["val_loss"][-1]
            
            # 最佳性能
            best_performance = {
                "best_score": float(self.best_score) if self.best_score is not None else None,
                "best_epoch": self.best_epoch,
                "improvement_from_start": None
            }
            
            if self.best_score is not None and self.history["val_loss"]:
                start_score = self.history["val_loss"][0]
                if self.mode == "min":
                    improvement = start_score - self.best_score
                else:
                    improvement = self.best_score - start_score
                best_performance["improvement_from_start"] = float(improvement)
            
            # 训练稳定性分析
            stability_analysis = self._analyze_training_stability()
            
            # 过拟合分析
            overfitting_analysis = self._analyze_overfitting_history()
            
            # 收敛分析
            convergence_analysis = self._analyze_convergence()
            
            # 建议
            recommendations = self._generate_recommendations()
            
            report = {
                "training_summary": {
                    "total_epochs": total_epochs,
                    "final_train_loss": float(final_train_loss),
                    "final_val_loss": float(final_val_loss),
                    "training_completed": not self.should_stop,
                    "early_stopped": self.should_stop,
                    "stopped_epoch": self.stopped_epoch if self.should_stop else None
                },
                "best_performance": best_performance,
                "stability_analysis": stability_analysis,
                "overfitting_analysis": overfitting_analysis,
                "convergence_analysis": convergence_analysis,
                "recommendations": recommendations,
                "monitoring_config": {
                    "monitor_metric": self.monitor_metric,
                    "mode": self.mode,
                    "patience": self.patience,
                    "min_delta": self.min_delta
                }
            }
            
            return convert_numpy_types(report)
            
        except Exception as e:
            logger.error(f"生成监控报告失败: {e}")
            return {"error": str(e)}
    
    def _analyze_training_stability(self) -> Dict[str, Any]:
        """分析训练稳定性"""
        if len(self.history["train_loss"]) < 5:
            return {"insufficient_data": True}
        
        train_losses = np.array(self.history["train_loss"])
        val_losses = np.array(self.history["val_loss"])
        
        # 计算损失的变异系数
        train_cv = np.std(train_losses) / np.mean(train_losses) if np.mean(train_losses) > 0 else 0
        val_cv = np.std(val_losses) / np.mean(val_losses) if np.mean(val_losses) > 0 else 0
        
        # 计算损失的平滑度（相邻epoch间的变化）
        train_smoothness = np.mean(np.abs(np.diff(train_losses)))
        val_smoothness = np.mean(np.abs(np.diff(val_losses)))
        
        stability_score = 1.0 / (1.0 + train_cv + val_cv)  # 0-1之间，越高越稳定
        
        return {
            "train_coefficient_variation": float(train_cv),
            "val_coefficient_variation": float(val_cv),
            "train_smoothness": float(train_smoothness),
            "val_smoothness": float(val_smoothness),
            "stability_score": float(stability_score),
            "is_stable": stability_score > 0.7
        }
    
    def _analyze_overfitting_history(self) -> Dict[str, Any]:
        """分析过拟合历史"""
        if len(self.history["train_loss"]) < 3:
            return {"insufficient_data": True}
        
        train_losses = np.array(self.history["train_loss"])
        val_losses = np.array(self.history["val_loss"])
        
        gaps = val_losses - train_losses
        
        # 统计过拟合情况
        overfitting_epochs = np.sum(gaps > self.overfitting_threshold)
        severe_overfitting_epochs = np.sum(gaps > self.overfitting_threshold * 3)
        
        # 计算平均gap和最大gap
        avg_gap = float(np.mean(gaps))
        max_gap = float(np.max(gaps))
        
        # 过拟合趋势
        gap_trend = float(np.mean(np.diff(gaps[-5:]))) if len(gaps) >= 5 else 0.0
        
        return {
            "average_gap": avg_gap,
            "max_gap": max_gap,
            "overfitting_epochs": int(overfitting_epochs),
            "severe_overfitting_epochs": int(severe_overfitting_epochs),
            "overfitting_ratio": float(overfitting_epochs / len(gaps)),
            "gap_trend": gap_trend,
            "is_overfitting": avg_gap > self.overfitting_threshold
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析收敛情况"""
        if len(self.history["val_loss"]) < 5:
            return {"insufficient_data": True}
        
        val_losses = np.array(self.history["val_loss"])
        
        # 检查最近几个epoch的变化
        recent_changes = np.abs(np.diff(val_losses[-5:]))
        avg_recent_change = float(np.mean(recent_changes))
        
        # 检查整体趋势
        if len(val_losses) >= 10:
            early_avg = np.mean(val_losses[:5])
            late_avg = np.mean(val_losses[-5:])
            overall_improvement = float(early_avg - late_avg) if self.mode == "min" else float(late_avg - early_avg)
        else:
            overall_improvement = 0.0
        
        # 判断是否收敛
        is_converged = avg_recent_change < self.min_delta
        
        return {
            "is_converged": is_converged,
            "recent_average_change": avg_recent_change,
            "overall_improvement": overall_improvement,
            "convergence_confidence": float(1.0 / (1.0 + avg_recent_change))
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成训练建议"""
        recommendations = []
        
        # 基于早停状态的建议
        if self.should_stop:
            recommendations.append(f"建议在第 {self.best_epoch} 个epoch停止训练，这是验证性能最佳的点")
        
        # 基于过拟合的建议
        if len(self.history["train_loss"]) > 0:
            recent_gap = self.history["val_loss"][-1] - self.history["train_loss"][-1]
            if recent_gap > self.overfitting_threshold * 2:
                recommendations.append("检测到严重过拟合，建议增加正则化（如dropout、weight decay）")
            elif recent_gap > self.overfitting_threshold:
                recommendations.append("检测到过拟合趋势，建议调整正则化参数或减少模型复杂度")
        
        # 基于性能趋势的建议
        if len(self.performance_trend) >= 3:
            recent_trend = list(self.performance_trend)[-3:]
            if self.mode == "min":
                if all(recent_trend[i] >= recent_trend[i-1] for i in range(1, len(recent_trend))):
                    recommendations.append("验证损失持续上升，建议降低学习率或检查数据质量")
            else:
                if all(recent_trend[i] <= recent_trend[i-1] for i in range(1, len(recent_trend))):
                    recommendations.append("验证指标持续下降，建议降低学习率或检查数据质量")
        
        # 基于学习率的建议
        if len(self.history["learning_rate"]) > 0 and self.history["learning_rate"][-1] is not None:
            current_lr = self.history["learning_rate"][-1]
            if len(self.history["train_loss"]) >= 3:
                recent_train_change = abs(self.history["train_loss"][-1] - self.history["train_loss"][-3])
                if recent_train_change < 0.001:
                    recommendations.append(f"训练损失变化很小，当前学习率 {current_lr:.6f} 可能过小")
        
        # 基于训练时长的建议
        if len(self.history["epoch"]) > 100 and not self.should_stop:
            recommendations.append("训练时间较长，建议检查是否需要调整模型架构或超参数")
        
        if not recommendations:
            recommendations.append("训练进展正常，继续当前设置")
        
        return recommendations
    
    def _save_monitoring_log(self, monitor_report: Dict[str, Any]):
        """保存监控日志"""
        try:
            log_file = self.output_dir / "monitoring_log.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(convert_numpy_types(monitor_report), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            logger.warning(f"保存监控日志失败: {e}")
    
    def save_training_history(self, save_path: Optional[str] = None) -> str:
        """
        保存训练历史
        
        Args:
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"training_history_{timestamp}.json"
        
        try:
            history_data = {
                "history": convert_numpy_types(self.history),
                "monitoring_config": {
                    "patience": self.patience,
                    "min_delta": self.min_delta,
                    "monitor_metric": self.monitor_metric,
                    "mode": self.mode
                },
                "final_state": {
                    "best_score": float(self.best_score) if self.best_score is not None else None,
                    "best_epoch": self.best_epoch,
                    "should_stop": self.should_stop,
                    "stopped_epoch": self.stopped_epoch
                }
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"训练历史已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")
            return ""
    
    def load_training_history(self, load_path: str) -> bool:
        """
        加载训练历史
        
        Args:
            load_path: 加载路径
            
        Returns:
            是否加载成功
        """
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复历史数据
            self.history = data.get("history", {})
            
            # 恢复状态
            final_state = data.get("final_state", {})
            self.best_score = final_state.get("best_score")
            self.best_epoch = final_state.get("best_epoch", 0)
            self.should_stop = final_state.get("should_stop", False)
            self.stopped_epoch = final_state.get("stopped_epoch", 0)
            
            # 重建性能趋势
            if self.history.get("val_loss"):
                recent_scores = self.history["val_loss"][-self.trend_window:]
                self.performance_trend = deque(recent_scores, maxlen=self.trend_window)
            
            logger.info(f"训练历史加载成功: {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载训练历史失败: {e}")
            return False
    
    def reset(self):
        """重置监控器状态"""
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rate": [],
            "timestamp": []
        }
        
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.overfitting_count = 0
        self.performance_trend.clear()
        
        logger.info("训练监控器已重置")


class EarlyStoppingCallback:
    """
    早停回调函数
    
    可以集成到各种训练框架中的早停回调。
    """
    
    def __init__(self, monitor: TrainingMonitor):
        """
        初始化早停回调
        
        Args:
            monitor: 训练监控器
        """
        self.monitor = monitor
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> bool:
        """
        在每个epoch结束时调用
        
        Args:
            epoch: epoch编号
            logs: 训练日志
            
        Returns:
            是否应该停止训练
        """
        train_loss = logs.get("loss", logs.get("train_loss", 0.0))
        val_loss = logs.get("val_loss", 0.0)
        
        # 提取训练和验证指标
        train_metrics = {k: v for k, v in logs.items() 
                        if k.startswith("train_") and k != "train_loss"}
        val_metrics = {k: v for k, v in logs.items() 
                      if k.startswith("val_") and k != "val_loss"}
        
        learning_rate = logs.get("lr", logs.get("learning_rate"))
        
        # 记录epoch结果
        monitor_report = self.monitor.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=learning_rate
        )
        
        return monitor_report["should_stop"]