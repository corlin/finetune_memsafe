#!/usr/bin/env python3
"""
增强的Qwen3优化微调系统主应用程序

基于现有main.py程序，集成数据拆分、模型训练和性能评估的完整流程。

新增功能：
- 增强的数据字段检测：智能识别不同格式的数据字段
- 批次数据处理修复：解决"批次数据为空"问题
- 自动错误恢复：多级降级处理机制
- 数据质量诊断：详细的数据验证和建议
- 灵活字段映射：支持自定义数据格式
- 实时处理监控：批次处理统计和性能监控
"""

import os
import sys
import argparse
import logging
import traceback
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import asdict
import json
from datetime import datetime

# 抑制常见的模型警告
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*past_key_value.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*")

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入现有组件
from main import QwenFineTuningApplication
from enhanced_config import EnhancedApplicationConfig, load_enhanced_config_from_yaml
from enhanced_error_recovery import ErrorRecoveryManager, ErrorCategory, ErrorSeverity

# 导入评估相关组件
from src.evaluation import (
    DataSplitter, EvaluationEngine, EvaluationConfig,
    ExperimentTracker, ReportGenerator,
    create_enhanced_evaluation_engine, load_evaluation_config
)
from datasets import Dataset


class EnhancedQwenFineTuningApplication(QwenFineTuningApplication):
    """
    增强的Qwen3微调应用程序
    
    基于现有QwenFineTuningApplication，添加数据拆分和评估功能。
    """
    
    def __init__(self, config: EnhancedApplicationConfig):
        """
        初始化增强应用程序
        
        Args:
            config: 增强应用程序配置
        """
        # 调用父类初始化
        super().__init__(config)
        
        self.enhanced_config = config
        
        # 增强组件
        self.data_splitter = None
        self.evaluation_engine = None
        self.experiment_tracker = None
        self.report_generator = None
        self.error_recovery_manager = None
        
        # 数据拆分结果
        self.data_split_result = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 评估结果
        self.evaluation_result = None
        self.experiment_id = None
        
        # 初始化增强组件
        self._initialize_enhanced_components()
        
        # 优化评估配置
        self._optimize_evaluation_config()
    
    def _initialize_enhanced_components(self):
        """初始化增强组件"""
        try:
            # 错误恢复管理器
            self.error_recovery_manager = ErrorRecoveryManager(
                fallback_mode=self.enhanced_config.fallback_to_basic_mode,
                log_errors=True
            )
            self.logger.info("错误恢复管理器初始化完成")
            
            # 数据拆分器
            if self.enhanced_config.enable_data_splitting:
                self.data_splitter = DataSplitter(
                    **self.enhanced_config.get_data_split_config()
                )
                self.logger.info("数据拆分器初始化完成")
            
            # 评估引擎 - 使用增强的评估引擎
            if self.enhanced_config.enable_comprehensive_evaluation:
                eval_config_dict = self.enhanced_config.get_evaluation_config()
                
                # 添加增强的数据处理配置
                eval_config_dict["data_processing"] = {
                    "field_mapping": {
                        "text_generation": {
                            "input_fields": ["text", "input", "prompt", "source", "content"],
                            "target_fields": ["target", "answer", "output", "response", "label"]
                        },
                        "question_answering": {
                            "input_fields": ["question", "query", "q"],
                            "context_fields": ["context", "passage", "document", "text"],
                            "target_fields": ["answer", "target", "a", "response"]
                        },
                        "classification": {
                            "input_fields": ["text", "input", "sentence", "content"],
                            "target_fields": ["label", "target", "class", "category"]
                        }
                    },
                    "validation": {
                        "min_valid_samples_ratio": 0.1,
                        "skip_empty_batches": True,
                        "enable_data_cleaning": True,
                        "enable_fallback": True
                    },
                    "diagnostics": {
                        "enable_detailed_logging": False,  # 避免过多日志
                        "log_batch_statistics": True,
                        "save_processing_report": True
                    }
                }
                
                # 使用增强的评估引擎
                self.evaluation_engine = create_enhanced_evaluation_engine(
                    config_data=eval_config_dict,
                    device="auto",
                    max_workers=8
                )
                self.logger.info("增强评估引擎初始化完成")
            
            # 实验跟踪器
            if self.enhanced_config.enable_experiment_tracking:
                self.experiment_tracker = ExperimentTracker(
                    experiment_dir=self.enhanced_config.experiments_output_dir
                )
                self.logger.info("实验跟踪器初始化完成")
            
            # 报告生成器
            self.report_generator = ReportGenerator(
                output_dir=self.enhanced_config.reports_output_dir
            )
            self.logger.info("报告生成器初始化完成")
            
        except Exception as e:
            self.logger.error(f"增强组件初始化失败: {e}")
            if not self.enhanced_config.fallback_to_basic_mode:
                raise
            else:
                self.logger.warning("回退到基础模式")
    
    def run_enhanced_pipeline(self) -> bool:
        """
        运行增强的微调流程
        
        Returns:
            bool: 是否成功完成
        """
        self.logger.info("开始增强Qwen3微调流程")
        self._display_enhanced_features_status()
        
        self.logging_system.info("增强应用程序启动", "APPLICATION", {
            "config": asdict(self.enhanced_config)
        })
        
        try:
            # 记录开始时间
            start_time = datetime.now()
            
            # 开始实验跟踪
            if self.experiment_tracker:
                self.experiment_id = self._start_experiment_tracking()
            
            # 1. 环境验证和设置
            if self.enhanced_config.verify_environment:
                self.logger.info("步骤1: 环境验证和设置")
                if not self._verify_and_setup_environment():
                    return False
            
            # 2. 内存优化和系统准备
            self.logger.info("步骤2: 内存优化和系统准备")
            self._prepare_system()
            
            # 3. 数据拆分（新增步骤）
            if self.enhanced_config.enable_data_splitting:
                self.logger.info("步骤3: 数据拆分")
                if not self._split_data():
                    return False
                
                # 更新实验进度
                if self.experiment_id and self.data_split_result:
                    self._update_experiment_progress("data_splitting", {
                        "train_samples": len(self.data_split_result.train_dataset),
                        "val_samples": len(self.data_split_result.val_dataset),
                        "test_samples": len(self.data_split_result.test_dataset)
                    })
            
            # 4. 模型加载和配置
            self.logger.info("步骤4: 模型加载和配置")
            if not self._load_and_configure_model():
                return False
            
            # 5. 数据准备（使用拆分后的训练集）
            self.logger.info("步骤5: 数据准备")
            if not self._prepare_enhanced_data():
                return False
            
            # 6. 训练配置和执行（支持验证集评估）
            self.logger.info("步骤6: 训练配置和执行")
            if not self._configure_and_train_enhanced():
                return False
            
            # 更新实验进度
            if self.experiment_id:
                training_progress = {
                    "epochs_completed": self.enhanced_config.num_epochs,
                    "model_saved": True
                }
                if hasattr(self, 'trainer') and self.trainer:
                    training_progress["final_step"] = getattr(self.trainer.state, 'global_step', 0)
                
                self._update_experiment_progress("training", training_progress)
            
            # 7. 全面模型评估（新增步骤）
            if self.enhanced_config.enable_comprehensive_evaluation:
                self.logger.info("步骤7: 全面模型评估")
                if not self._comprehensive_evaluation():
                    self.logger.warning("全面评估失败，但训练已完成")
                
                # 更新实验进度
                if self.experiment_id and self.evaluation_result:
                    self._update_experiment_progress("evaluation", {
                        "metrics": self.evaluation_result.metrics,
                        "efficiency_measured": bool(self.evaluation_result.efficiency_metrics),
                        "quality_analyzed": bool(self.evaluation_result.quality_scores)
                    })
            
            # 8. 推理测试（保持原有功能）
            if self.enhanced_config.enable_inference_test:
                self.logger.info("步骤8: 推理测试")
                if not self._test_inference():
                    self.logger.warning("推理测试失败，但训练已完成")
            
            # 9. 生成报告（新增步骤）
            self.logger.info("步骤9: 生成报告")
            self._generate_enhanced_reports()
            
            # 10. 完成实验跟踪
            if self.experiment_tracker and self.experiment_id:
                self._complete_experiment_tracking(start_time, True)
            
            # 10. 生成数据处理报告
            self.logger.info("步骤10: 生成数据处理报告")
            self._generate_data_processing_report()
            
            # 11. 生成最终报告
            self.logger.info("步骤11: 生成最终报告")
            self._generate_final_report(start_time)
            
            self.logger.info("增强Qwen3微调流程成功完成")
            return True
            
        except KeyboardInterrupt:
            self.logger.info("用户中断了微调流程")
            self._handle_interruption()
            return False
            
        except Exception as e:
            self.logger.error(f"增强微调流程失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 完成实验跟踪（失败）
            if self.experiment_tracker and self.experiment_id:
                self._complete_experiment_tracking(datetime.now(), False)
            
            self._handle_error(e)
            return False
        
        finally:
            # 生成错误报告
            if self.error_recovery_manager and self.error_recovery_manager.error_history:
                try:
                    error_report_path = Path(self.enhanced_config.output_dir) / "error_report.json"
                    self.error_recovery_manager.save_error_report(str(error_report_path))
                    
                    # 显示错误摘要
                    error_summary = self.error_recovery_manager.get_error_summary()
                    self.logger.info(f"执行过程中遇到 {error_summary['total_errors']} 个错误")
                    if error_summary['categories']:
                        self.logger.info(f"错误类别分布: {error_summary['categories']}")
                    
                except Exception as e:
                    self.logger.error(f"生成错误报告失败: {e}")
            
            self._cleanup()
    
    def _split_data(self) -> bool:
        """数据拆分步骤"""
        try:
            self.logging_system.info("开始数据拆分", "DATA_SPLIT")
            
            # 检查是否已存在拆分结果
            splits_dir = Path(self.enhanced_config.data_splits_output_dir)
            if (self.enhanced_config.skip_data_splitting_if_exists and 
                splits_dir.exists() and 
                (splits_dir / "split_info.json").exists()):
                
                self.logger.info("发现已存在的数据拆分结果，跳过拆分步骤")
                self.data_split_result = self.data_splitter.load_splits(str(splits_dir))
                
            else:
                # 加载原始数据
                from src.data_pipeline import DataPipeline
                data_pipeline = DataPipeline(
                    data_dir=self.enhanced_config.data_dir,
                    max_sequence_length=self.enhanced_config.max_sequence_length
                )
                
                # 加载QA数据
                qa_data = data_pipeline.load_qa_data_from_files()
                
                # 格式化数据
                raw_dataset = data_pipeline.format_for_qwen(qa_data)
                
                # 执行数据拆分
                self.data_split_result = self.data_splitter.split_data(
                    raw_dataset, 
                    output_dir=self.enhanced_config.data_splits_output_dir
                )
            
            # 提取拆分后的数据集
            self.train_dataset = self.data_split_result.train_dataset
            self.val_dataset = self.data_split_result.val_dataset
            self.test_dataset = self.data_split_result.test_dataset
            
            # 记录拆分信息
            split_info = self.data_split_result.split_info
            self.logging_system.info("数据拆分完成", "DATA_SPLIT", split_info)
            
            self.logger.info(f"数据拆分完成:")
            self.logger.info(f"  训练集: {len(self.train_dataset)} 样本")
            self.logger.info(f"  验证集: {len(self.val_dataset)} 样本")
            self.logger.info(f"  测试集: {len(self.test_dataset)} 样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据拆分失败: {e}")
            self.logging_system.error(f"数据拆分失败: {e}", "DATA_SPLIT")
            
            # 使用错误恢复管理器处理
            if self.error_recovery_manager:
                can_continue = self.error_recovery_manager.handle_error(
                    ErrorCategory.DATA_SPLIT, 
                    e,
                    context={
                        "data_dir": self.enhanced_config.data_dir,
                        "enable_splitting": self.enhanced_config.enable_data_splitting,
                        "train_ratio": self.enhanced_config.train_ratio
                    }
                )
                if can_continue:
                    self.logger.warning("数据拆分错误已恢复，继续执行")
                    return True
            
            return False
    
    def _prepare_enhanced_data(self) -> bool:
        """准备增强的训练数据"""
        try:
            self.logging_system.info("开始增强数据准备", "DATA")
            
            if self.enhanced_config.enable_data_splitting and self.train_dataset:
                # 使用拆分后的训练集
                dataset = self.train_dataset
                self.logger.info(f"使用拆分后的训练集，样本数: {len(dataset)}")
            else:
                # 回退到原始数据准备方法
                return self._prepare_data()
            
            # 创建数据管道进行分词
            from src.data_pipeline import DataPipeline
            self.data_pipeline = DataPipeline(
                data_dir=self.enhanced_config.data_dir,
                max_sequence_length=self.enhanced_config.max_sequence_length
            )
            
            # 分词
            self.dataset = self.data_pipeline.tokenize_dataset(dataset, self.tokenizer)
            
            # 获取数据统计
            data_stats = self.data_pipeline.get_dataset_stats(self.dataset)
            self.logging_system.info("增强数据准备完成", "DATA", data_stats)
            
            return True
            
        except Exception as e:
            self.logger.error(f"增强数据准备失败: {e}")
            self.logging_system.error(f"增强数据准备失败: {e}", "DATA")
            return False
    
    def _configure_and_train_enhanced(self) -> bool:
        """配置和执行增强训练"""
        try:
            self.logging_system.info("开始增强训练配置", "TRAINING")
            
            # 创建训练配置
            from src.training_engine import TrainingConfig
            training_config = TrainingConfig(
                output_dir=self.enhanced_config.output_dir,
                max_memory_gb=self.enhanced_config.max_memory_gb,
                batch_size=self.enhanced_config.batch_size,
                gradient_accumulation_steps=self.enhanced_config.gradient_accumulation_steps,
                learning_rate=self.enhanced_config.learning_rate,
                num_epochs=self.enhanced_config.num_epochs,
                max_sequence_length=self.enhanced_config.max_sequence_length,
                eval_steps=self.enhanced_config.validation_steps if self.enhanced_config.enable_validation_during_training else 0
            )
            
            # 创建训练引擎
            from src.training_engine import TrainingEngine
            self.training_engine = TrainingEngine(
                config=training_config,
                memory_optimizer=self.memory_optimizer,
                logging_system=self.logging_system,
                progress_monitor=self.progress_monitor
            )
            
            # 创建数据整理器
            data_collator = self.data_pipeline.create_data_collator(self.tokenizer)
            
            # 准备验证集（如果可用）
            eval_dataset = None
            if (self.enhanced_config.enable_validation_during_training and 
                self.val_dataset):
                
                # 分词验证集
                eval_dataset = self.data_pipeline.tokenize_dataset(
                    self.val_dataset, self.tokenizer
                )
                self.logger.info(f"准备验证集，样本数: {len(eval_dataset)}")
            
            # 创建训练器（包含验证集）
            self.trainer = self.training_engine.create_trainer(
                model=self.model,
                train_dataset=self.dataset,
                eval_dataset=eval_dataset,  # 传入验证集
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            # 执行训练
            self.trainer = self.training_engine.train_model(self.trainer)
            
            # 保存模型
            self.training_engine.save_model(self.trainer)
            
            # 记录验证集评估历史
            if (eval_dataset and 
                self.enhanced_config.save_validation_metrics and
                hasattr(self.trainer.state, 'log_history')):
                
                self._save_validation_metrics_history()
            
            self.logging_system.info("增强训练完成", "TRAINING")
            return True
            
        except Exception as e:
            self.logger.error(f"增强训练失败: {e}")
            self.logging_system.error(f"增强训练失败: {e}", "TRAINING")
            
            # 使用错误恢复管理器处理
            if self.error_recovery_manager:
                can_continue = self.error_recovery_manager.handle_error(
                    ErrorCategory.TRAINING,
                    e,
                    context={
                        "batch_size": self.enhanced_config.batch_size,
                        "num_epochs": self.enhanced_config.num_epochs,
                        "learning_rate": self.enhanced_config.learning_rate,
                        "max_memory_gb": self.enhanced_config.max_memory_gb
                    }
                )
                if can_continue:
                    self.logger.warning("训练错误已恢复，继续执行")
                    return True
            
            return False
    
    def _save_validation_metrics_history(self):
        """保存验证指标历史"""
        try:
            if not hasattr(self.trainer.state, 'log_history'):
                return
            
            # 提取验证指标历史
            validation_history = []
            for log_entry in self.trainer.state.log_history:
                if 'eval_loss' in log_entry:
                    validation_history.append({
                        'epoch': log_entry.get('epoch', 0),
                        'step': log_entry.get('step', 0),
                        'eval_loss': log_entry.get('eval_loss', 0),
                        'train_loss': log_entry.get('train_loss', 0),
                        **{k: v for k, v in log_entry.items() 
                           if k.startswith('eval_') and k != 'eval_loss'}
                    })
            
            # 保存验证历史
            if validation_history:
                history_path = Path(self.enhanced_config.output_dir) / "validation_history.json"
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(validation_history, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"验证指标历史已保存: {history_path}")
                
                # 记录最佳验证性能
                best_eval_loss = min(entry['eval_loss'] for entry in validation_history)
                self.logging_system.info("验证集训练历史", "VALIDATION", {
                    "total_evaluations": len(validation_history),
                    "best_eval_loss": best_eval_loss,
                    "final_eval_loss": validation_history[-1]['eval_loss'] if validation_history else None
                })
            
        except Exception as e:
            self.logger.error(f"保存验证指标历史失败: {e}")
    
    def _evaluate_on_validation_set(self):
        """在验证集上评估（独立评估，用于训练后的详细分析）"""
        try:
            if not self.val_dataset or not self.evaluation_engine:
                return
            
            # 分词验证集
            val_dataset_tokenized = self.data_pipeline.tokenize_dataset(
                self.val_dataset, self.tokenizer
            )
            
            # 执行评估
            val_result = self.evaluation_engine.evaluate_model(
                self.model,
                self.tokenizer,
                {"validation": val_dataset_tokenized},
                f"{self.enhanced_config.model_name}_validation"
            )
            
            # 记录验证结果
            self.logging_system.info("独立验证集评估完成", "VALIDATION", {
                "metrics": val_result.metrics
            })
            
            # 保存验证评估结果
            val_result_path = Path(self.enhanced_config.output_dir) / "validation_evaluation.json"
            with open(val_result_path, 'w', encoding='utf-8') as f:
                json.dump(val_result.get_summary(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"验证评估结果已保存: {val_result_path}")
            
        except Exception as e:
            self.logger.error(f"验证集评估失败: {e}")
    
    def _comprehensive_evaluation(self) -> bool:
        """全面模型评估 - 使用增强的数据处理能力"""
        try:
            if not self.evaluation_engine or not self.test_dataset:
                self.logger.warning("评估引擎或测试集不可用，跳过全面评估")
                return True
            
            self.logging_system.info("开始全面模型评估", "EVALUATION")
            
            # 准备原始数据集（不进行分词，让增强评估引擎自动处理）
            eval_datasets = {}
            
            # 为每个配置的评估任务准备数据集
            for task_name in self.enhanced_config.evaluation_tasks:
                eval_datasets[task_name] = self.test_dataset
                self.logger.info(f"为任务 {task_name} 准备测试集，样本数: {len(self.test_dataset)}")
            
            # 如果有验证集，也为每个任务添加验证集
            if self.val_dataset:
                # 为验证集创建单独的任务名称
                for task_name in self.enhanced_config.evaluation_tasks:
                    validation_task_name = f"{task_name}_validation"
                    eval_datasets[validation_task_name] = self.val_dataset
                    self.logger.info(f"为任务 {validation_task_name} 准备验证集，样本数: {len(self.val_dataset)}")
            
            self.logger.info(f"准备的评估数据集: {list(eval_datasets.keys())}")
            
            # 在评估前诊断数据集
            self._diagnose_evaluation_datasets(eval_datasets)
            
            # 执行增强评估（包含诊断信息）
            result = self.evaluation_engine.evaluate_model_with_diagnostics(
                self.model,
                self.tokenizer,
                eval_datasets,
                self.enhanced_config.model_name,
                save_diagnostics=True
            )
            
            self.evaluation_result = result["evaluation_result"]
            evaluation_diagnostics = result["diagnostics"]
            processing_stats = result["processing_stats"]
            
            # 记录诊断信息
            self.logger.info("评估诊断信息:")
            self.logger.info(f"  处理统计: {processing_stats}")
            
            if evaluation_diagnostics.get("report_path"):
                self.logger.info(f"  诊断报告已保存: {evaluation_diagnostics['report_path']}")
            
            # 保存详细评估结果
            self._save_detailed_evaluation_results()
            
            # 保存增强评估的诊断信息
            self._save_evaluation_diagnostics(evaluation_diagnostics, processing_stats)
            
            # 记录评估结果
            self.logging_system.info("全面评估完成", "EVALUATION", {
                "metrics": self.evaluation_result.metrics,
                "efficiency": asdict(self.evaluation_result.efficiency_metrics),
                "quality": asdict(self.evaluation_result.quality_scores),
                "tasks_evaluated": list(eval_datasets.keys()),
                "processing_stats": processing_stats
            })
            
            # 显示评估结果摘要
            self._display_evaluation_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"全面评估失败: {e}")
            self.logging_system.error(f"全面评估失败: {e}", "EVALUATION")
            
            # 尝试使用增强的错误处理
            context = {
                "evaluation_tasks": self.enhanced_config.evaluation_tasks,
                "evaluation_metrics": self.enhanced_config.evaluation_metrics,
                "batch_size": getattr(self.enhanced_config, 'evaluation_batch_size', 8),
                "num_samples": getattr(self.enhanced_config, 'evaluation_num_samples', 100),
                "datasets": eval_datasets if 'eval_datasets' in locals() else {}
            }
            
            if self._handle_evaluation_errors(e, context):
                self.logger.info("评估错误已通过增强处理修复，重试评估...")
                try:
                    # 重试评估
                    result = self.evaluation_engine.evaluate_model_with_diagnostics(
                        self.model,
                        self.tokenizer,
                        eval_datasets,
                        self.enhanced_config.model_name,
                        save_diagnostics=True
                    )
                    
                    self.evaluation_result = result["evaluation_result"]
                    self.logger.info("重试评估成功")
                    return True
                    
                except Exception as retry_error:
                    self.logger.error(f"重试评估仍然失败: {retry_error}")
            
            # 使用原有的错误恢复管理器作为后备
            if self.error_recovery_manager:
                can_continue = self.error_recovery_manager.handle_error(
                    ErrorCategory.EVALUATION,
                    e,
                    context=context
                )
                if can_continue:
                    self.logger.warning("评估错误已恢复，继续执行")
                    return True
            
            return False
    
    def _save_detailed_evaluation_results(self):
        """保存详细的评估结果"""
        try:
            if not self.evaluation_result:
                return
            
            # 保存完整评估结果
            eval_result_path = Path(self.enhanced_config.output_dir) / "comprehensive_evaluation.json"
            with open(eval_result_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_result.get_summary(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"详细评估结果已保存: {eval_result_path}")
            
            # 保存任务级别的详细结果
            for task_name, task_result in self.evaluation_result.task_results.items():
                task_result_path = Path(self.enhanced_config.output_dir) / f"evaluation_{task_name}.json"
                
                task_summary = {
                    "task_name": task_result.task_name,
                    "metrics": task_result.metrics,
                    "execution_time": task_result.execution_time,
                    "sample_count": len(task_result.samples),
                    "predictions_sample": task_result.predictions[:5] if task_result.predictions else [],
                    "references_sample": task_result.references[:5] if task_result.references else []
                }
                
                with open(task_result_path, 'w', encoding='utf-8') as f:
                    json.dump(task_summary, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"任务 {task_name} 评估结果已保存: {task_result_path}")
            
        except Exception as e:
            self.logger.error(f"保存详细评估结果失败: {e}")
    
    def _display_evaluation_summary(self):
        """显示评估结果摘要"""
        try:
            if not self.evaluation_result:
                return
            
            self.logger.info("=" * 60)
            self.logger.info("全面评估结果摘要")
            self.logger.info("=" * 60)
            
            # 显示整体指标
            self.logger.info("整体指标:")
            for metric, value in self.evaluation_result.metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {metric}: {value:.4f}")
                else:
                    self.logger.info(f"  {metric}: {value}")
            
            # 显示任务级别指标
            if self.evaluation_result.task_results:
                self.logger.info("\n任务级别指标:")
                for task_name, task_result in self.evaluation_result.task_results.items():
                    self.logger.info(f"  {task_name}:")
                    for metric, value in task_result.metrics.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"    {metric}: {value:.4f}")
                        else:
                            self.logger.info(f"    {metric}: {value}")
            
            # 显示效率指标
            if self.evaluation_result.efficiency_metrics:
                self.logger.info("\n效率指标:")
                eff_metrics = self.evaluation_result.efficiency_metrics
                self.logger.info(f"  推理延迟: {eff_metrics.inference_latency:.2f} ms")
                self.logger.info(f"  吞吐量: {eff_metrics.throughput:.2f} samples/sec")
                self.logger.info(f"  内存使用: {eff_metrics.memory_usage:.2f} GB")
                self.logger.info(f"  模型大小: {eff_metrics.model_size:.2f} MB")
            
            # 显示质量分数
            if self.evaluation_result.quality_scores:
                self.logger.info("\n质量分数:")
                quality_scores = self.evaluation_result.quality_scores
                self.logger.info(f"  流畅性: {quality_scores.fluency:.4f}")
                self.logger.info(f"  连贯性: {quality_scores.coherence:.4f}")
                self.logger.info(f"  相关性: {quality_scores.relevance:.4f}")
                self.logger.info(f"  事实性: {quality_scores.factuality:.4f}")
                self.logger.info(f"  整体质量: {quality_scores.overall:.4f}")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"显示评估摘要失败: {e}")
    
    def _diagnose_evaluation_datasets(self, eval_datasets: Dict[str, Dataset]):
        """诊断评估数据集，提前发现潜在问题"""
        try:
            self.logger.info("开始诊断评估数据集...")
            
            for task_name, dataset in eval_datasets.items():
                self.logger.info(f"诊断任务 {task_name} 的数据集:")
                
                # 使用增强评估引擎的诊断功能
                diagnosis = self.evaluation_engine.diagnose_dataset(dataset, task_name)
                
                # 记录诊断结果
                batch_info = diagnosis.get("batch_info", {})
                self.logger.info(f"  数据集大小: {batch_info.get('total_samples', 0)}")
                self.logger.info(f"  可用字段: {batch_info.get('available_fields', [])}")
                
                # 检查字段映射
                field_mapping_info = diagnosis.get("field_mapping_info", {})
                recommended_input = field_mapping_info.get("recommended_input_field")
                recommended_target = field_mapping_info.get("recommended_target_field")
                
                if recommended_input:
                    self.logger.info(f"  推荐输入字段: {recommended_input}")
                if recommended_target:
                    self.logger.info(f"  推荐目标字段: {recommended_target}")
                
                # 显示建议
                recommendations = diagnosis.get("recommendations", [])
                if recommendations:
                    self.logger.info(f"  数据处理建议:")
                    for i, rec in enumerate(recommendations[:3], 1):  # 只显示前3个建议
                        self.logger.info(f"    {i}. {rec}")
                
                # 检查验证结果
                validation_result = diagnosis.get("validation_result", {})
                if not validation_result.get("is_valid", True):
                    issues = validation_result.get("issues", [])
                    self.logger.warning(f"  数据验证问题: {issues}")
            
            self.logger.info("数据集诊断完成")
            
        except Exception as e:
            self.logger.error(f"数据集诊断失败: {e}")
    
    def _save_evaluation_diagnostics(self, diagnostics: Dict[str, Any], processing_stats: Dict[str, Any]):
        """保存评估诊断信息"""
        try:
            diagnostics_data = {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.enhanced_config.model_name,
                "evaluation_tasks": self.enhanced_config.evaluation_tasks,
                "diagnostics": diagnostics,
                "processing_stats": processing_stats
            }
            
            # 保存诊断信息
            diagnostics_path = Path(self.enhanced_config.output_dir) / "evaluation_diagnostics.json"
            with open(diagnostics_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostics_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"评估诊断信息已保存: {diagnostics_path}")
            
            # 记录关键统计信息
            if processing_stats:
                self.logger.info("数据处理统计:")
                self.logger.info(f"  成功率: {processing_stats.get('success_rate', 0):.2%}")
                self.logger.info(f"  有效样本率: {processing_stats.get('valid_sample_rate', 0):.2%}")
                self.logger.info(f"  处理的批次数: {processing_stats.get('total_batches_processed', 0)}")
                self.logger.info(f"  处理的样本数: {processing_stats.get('total_samples_processed', 0)}")
            
        except Exception as e:
            self.logger.error(f"保存评估诊断信息失败: {e}")
    
    def _optimize_evaluation_config(self):
        """优化评估配置以提高数据处理能力"""
        try:
            if not self.evaluation_engine:
                return
            
            # 根据数据集大小调整批次大小
            if hasattr(self.enhanced_config, 'evaluation_batch_size'):
                original_batch_size = self.enhanced_config.evaluation_batch_size
                
                # 根据可用内存和数据集大小优化批次大小
                if self.enhanced_config.max_memory_gb < 8:
                    # 低内存环境，使用较小的批次
                    optimized_batch_size = min(original_batch_size, 16)
                elif self.enhanced_config.max_memory_gb >= 16:
                    # 高内存环境，可以使用较大的批次
                    optimized_batch_size = min(original_batch_size * 2, 64)
                else:
                    optimized_batch_size = original_batch_size
                
                if optimized_batch_size != original_batch_size:
                    self.logger.info(f"优化评估批次大小: {original_batch_size} -> {optimized_batch_size}")
                    # 更新评估引擎配置
                    self.evaluation_engine.config.batch_size = optimized_batch_size
            
            # 配置增强数据处理选项
            if hasattr(self.evaluation_engine, 'configure_enhanced_processing'):
                self.evaluation_engine.configure_enhanced_processing(
                    enable_detailed_logging=False,  # 避免过多日志影响性能
                    enable_data_cleaning=True,      # 启用数据清洗
                    enable_fallback=True,           # 启用降级处理
                    min_valid_samples_ratio=0.05    # 较低的有效样本阈值
                )
                self.logger.info("已配置增强数据处理选项")
            
        except Exception as e:
            self.logger.error(f"优化评估配置失败: {e}")
    
    def _handle_evaluation_errors(self, error: Exception, context: Dict[str, Any]) -> bool:
        """处理评估过程中的错误"""
        try:
            self.logger.error(f"评估错误: {error}")
            
            # 检查是否是数据处理相关的错误
            error_str = str(error).lower()
            
            if any(keyword in error_str for keyword in ['batch', 'empty', 'field', 'data']):
                self.logger.info("检测到数据处理相关错误，尝试诊断和修复...")
                
                # 获取处理建议
                if self.evaluation_engine and hasattr(self.evaluation_engine, 'get_processing_recommendations'):
                    datasets = context.get('datasets', {})
                    if datasets:
                        recommendations = self.evaluation_engine.get_processing_recommendations(datasets)
                        
                        if recommendations:
                            self.logger.info("数据处理建议:")
                            for i, rec in enumerate(recommendations, 1):
                                self.logger.info(f"  {i}. {rec}")
                        
                        # 尝试应用自动修复
                        return self._apply_evaluation_fixes(datasets, recommendations)
            
            return False
            
        except Exception as e:
            self.logger.error(f"处理评估错误时发生异常: {e}")
            return False
    
    def _apply_evaluation_fixes(self, datasets: Dict[str, Dataset], recommendations: List[str]) -> bool:
        """应用评估修复建议"""
        try:
            # 检查建议中是否包含字段映射相关的修复
            for rec in recommendations:
                if "字段映射" in rec or "field mapping" in rec.lower():
                    self.logger.info("尝试应用字段映射修复...")
                    
                    # 重新配置字段映射
                    if hasattr(self.evaluation_engine, 'configure_enhanced_processing'):
                        self.evaluation_engine.configure_enhanced_processing(
                            enable_fallback=True,
                            enable_data_cleaning=True,
                            min_valid_samples_ratio=0.01  # 更低的阈值
                        )
                        return True
                
                elif "批次大小" in rec or "batch size" in rec.lower():
                    self.logger.info("尝试调整批次大小...")
                    
                    # 减小批次大小
                    if hasattr(self.evaluation_engine, 'config'):
                        original_batch_size = self.evaluation_engine.config.batch_size
                        new_batch_size = max(1, original_batch_size // 2)
                        self.evaluation_engine.config.batch_size = new_batch_size
                        self.logger.info(f"批次大小已调整: {original_batch_size} -> {new_batch_size}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"应用评估修复失败: {e}")
            return False
    
    def _generate_data_processing_report(self):
        """生成数据处理报告"""
        try:
            if not self.evaluation_engine:
                self.logger.warning("评估引擎不可用，跳过数据处理报告生成")
                return
            
            self.logger.info("开始生成数据处理报告...")
            
            # 获取数据处理统计信息
            if hasattr(self.evaluation_engine, 'get_diagnostic_statistics'):
                diagnostic_stats = self.evaluation_engine.get_diagnostic_statistics()
                
                # 生成报告
                report_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.enhanced_config.model_name,
                    "data_processing_summary": {
                        "enhanced_processing_enabled": True,
                        "diagnostic_statistics": diagnostic_stats,
                        "configuration": {
                            "batch_size": getattr(self.evaluation_engine.config, 'batch_size', 'unknown'),
                            "enable_data_cleaning": True,
                            "enable_fallback": True,
                            "min_valid_samples_ratio": 0.05
                        }
                    }
                }
                
                # 添加数据集信息
                if self.data_split_result:
                    report_data["data_split_info"] = {
                        "train_samples": len(self.data_split_result.train_dataset),
                        "val_samples": len(self.data_split_result.val_dataset),
                        "test_samples": len(self.data_split_result.test_dataset),
                        "split_ratios": self.data_split_result.split_info
                    }
                
                # 保存报告
                report_path = Path(self.enhanced_config.output_dir) / "data_processing_report.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"数据处理报告已保存: {report_path}")
                
                # 显示关键统计信息
                if diagnostic_stats:
                    self.logger.info("数据处理统计摘要:")
                    processing_stats = diagnostic_stats.get("processing_stats", {})
                    if processing_stats:
                        self.logger.info(f"  批次处理成功率: {processing_stats.get('success_rate', 0):.2%}")
                        self.logger.info(f"  有效样本率: {processing_stats.get('valid_sample_rate', 0):.2%}")
                        self.logger.info(f"  处理的总批次数: {processing_stats.get('total_batches_processed', 0)}")
            
        except Exception as e:
            self.logger.error(f"生成数据处理报告失败: {e}")
    
    def _display_enhanced_features_status(self):
        """显示增强功能状态"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("增强功能状态")
            self.logger.info("=" * 60)
            
            # 数据拆分功能
            if self.enhanced_config.enable_data_splitting:
                self.logger.info("✓ 数据拆分功能: 已启用")
                self.logger.info(f"  - 训练集比例: {getattr(self.enhanced_config, 'train_ratio', 0.7)}")
                self.logger.info(f"  - 验证集比例: {getattr(self.enhanced_config, 'val_ratio', 0.15)}")
                self.logger.info(f"  - 测试集比例: {getattr(self.enhanced_config, 'test_ratio', 0.15)}")
            else:
                self.logger.info("✗ 数据拆分功能: 已禁用")
            
            # 增强评估功能
            if self.enhanced_config.enable_comprehensive_evaluation:
                self.logger.info("✓ 增强评估功能: 已启用")
                self.logger.info("  - 智能字段检测: 已启用")
                self.logger.info("  - 批次数据处理修复: 已启用")
                self.logger.info("  - 自动错误恢复: 已启用")
                self.logger.info("  - 数据质量诊断: 已启用")
                self.logger.info("  - 实时处理监控: 已启用")
                
                if hasattr(self.enhanced_config, 'evaluation_tasks'):
                    self.logger.info(f"  - 评估任务: {self.enhanced_config.evaluation_tasks}")
            else:
                self.logger.info("✗ 增强评估功能: 已禁用")
            
            # 实验跟踪功能
            if self.enhanced_config.enable_experiment_tracking:
                self.logger.info("✓ 实验跟踪功能: 已启用")
            else:
                self.logger.info("✗ 实验跟踪功能: 已禁用")
            
            # 错误恢复功能
            if self.enhanced_config.fallback_to_basic_mode:
                self.logger.info("✓ 错误恢复功能: 已启用")
            else:
                self.logger.info("✗ 错误恢复功能: 已禁用")
            
            # 验证集训练功能
            if getattr(self.enhanced_config, 'enable_validation_during_training', False):
                self.logger.info("✓ 训练时验证: 已启用")
            else:
                self.logger.info("✗ 训练时验证: 已禁用")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"显示增强功能状态失败: {e}")
    
    def _generate_enhanced_reports(self):
        """生成增强报告"""
        try:
            if not self.report_generator:
                self.logger.warning("报告生成器不可用，跳过报告生成")
                return
            
            self.logger.info("开始生成增强报告")
            
            report_paths = {}
            
            # 生成评估报告
            if self.evaluation_result:
                self.logger.info("生成评估报告...")
                for format_type in self.enhanced_config.report_formats:
                    try:
                        report_path = self.report_generator.generate_evaluation_report(
                            self.evaluation_result,
                            format_type=format_type,
                            include_charts=self.enhanced_config.output_charts
                        )
                        if report_path:
                            report_paths[f"evaluation_{format_type}"] = report_path
                            self.logger.info(f"生成{format_type}评估报告: {report_path}")
                    except Exception as e:
                        self.logger.error(f"生成{format_type}评估报告失败: {e}")
            
            # 生成数据拆分分析报告
            if self.data_split_result and self.enhanced_config.enable_data_splitting:
                self.logger.info("生成数据拆分分析报告...")
                try:
                    split_report_path = self._generate_data_split_report()
                    if split_report_path:
                        report_paths["data_split_analysis"] = split_report_path
                        self.logger.info(f"生成数据拆分分析报告: {split_report_path}")
                except Exception as e:
                    self.logger.error(f"生成数据拆分报告失败: {e}")
            
            # 生成训练过程报告
            if hasattr(self, 'trainer') and self.trainer:
                self.logger.info("生成训练过程报告...")
                try:
                    training_report_path = self._generate_training_report()
                    if training_report_path:
                        report_paths["training_process"] = training_report_path
                        self.logger.info(f"生成训练过程报告: {training_report_path}")
                except Exception as e:
                    self.logger.error(f"生成训练过程报告失败: {e}")
            
            # 生成综合报告
            if len(report_paths) > 0:
                self.logger.info("生成综合报告...")
                try:
                    comprehensive_report_path = self._generate_comprehensive_report(report_paths)
                    if comprehensive_report_path:
                        report_paths["comprehensive"] = comprehensive_report_path
                        self.logger.info(f"生成综合报告: {comprehensive_report_path}")
                except Exception as e:
                    self.logger.error(f"生成综合报告失败: {e}")
            
            # 保存报告路径信息
            self._save_report_index(report_paths)
            
            self.logger.info(f"报告生成完成，共生成 {len(report_paths)} 个报告")
            
            # 记录到日志系统
            self.logging_system.info("增强报告生成完成", "REPORTS", {
                "report_count": len(report_paths),
                "report_types": list(report_paths.keys()),
                "formats": self.enhanced_config.report_formats
            })
            
        except Exception as e:
            self.logger.error(f"生成增强报告失败: {e}")
    
    def _generate_data_split_report(self) -> Optional[str]:
        """生成数据拆分分析报告"""
        try:
            if not self.data_split_result:
                return None
            
            # 创建数据拆分分析报告
            report_data = {
                "title": "数据拆分分析报告",
                "generated_at": datetime.now().isoformat(),
                "split_info": self.data_split_result.split_info,
                "distribution_analysis": asdict(self.data_split_result.distribution_analysis)
            }
            
            # 保存JSON格式报告
            report_path = Path(self.enhanced_config.reports_output_dir) / "data_split_analysis.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # 如果启用HTML报告，生成HTML版本
            if "html" in self.enhanced_config.report_formats:
                html_report_path = self._generate_data_split_html_report(report_data)
                return html_report_path
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"生成数据拆分报告失败: {e}")
            return None
    
    def _generate_data_split_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成数据拆分HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据拆分分析报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-name {{ color: #7f8c8d; font-size: 14px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .consistency-score {{ font-size: 36px; font-weight: bold; text-align: center; margin: 20px 0; }}
                .good {{ color: #4CAF50; }}
                .warning {{ color: #FF9800; }}
                .error {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据拆分分析报告</h1>
                <p>生成时间: {generated_at}</p>
            </div>
            
            <div class="section">
                <h2>拆分概览</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{total_samples}</div>
                        <div class="metric-name">总样本数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{train_samples}</div>
                        <div class="metric-name">训练集</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{val_samples}</div>
                        <div class="metric-name">验证集</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{test_samples}</div>
                        <div class="metric-name">测试集</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>分布一致性</h2>
                <div class="consistency-score {consistency_class}">
                    {consistency_score:.4f}
                </div>
                <p>分布一致性分数（1.0为完美一致）</p>
            </div>
            
            <div class="section">
                <h2>拆分配置</h2>
                <table>
                    <tr><th>配置项</th><th>值</th></tr>
                    <tr><td>拆分方法</td><td>{split_method}</td></tr>
                    <tr><td>随机种子</td><td>{random_seed}</td></tr>
                    <tr><td>分层字段</td><td>{stratify_by}</td></tr>
                    <tr><td>训练集比例</td><td>{train_ratio:.1%}</td></tr>
                    <tr><td>验证集比例</td><td>{val_ratio:.1%}</td></tr>
                    <tr><td>测试集比例</td><td>{test_ratio:.1%}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        # 提取数据
        split_info = report_data["split_info"]
        dist_analysis = report_data["distribution_analysis"]
        
        # 确定一致性分数的样式类
        consistency_score = dist_analysis.get("consistency_score", 0)
        if consistency_score >= 0.9:
            consistency_class = "good"
        elif consistency_score >= 0.7:
            consistency_class = "warning"
        else:
            consistency_class = "error"
        
        # 填充模板
        html_content = html_template.format(
            generated_at=report_data["generated_at"],
            total_samples=split_info.get("total_samples", 0),
            train_samples=split_info.get("train_samples", 0),
            val_samples=split_info.get("val_samples", 0),
            test_samples=split_info.get("test_samples", 0),
            consistency_score=consistency_score,
            consistency_class=consistency_class,
            split_method=split_info.get("split_method", "unknown"),
            random_seed=split_info.get("random_seed", 42),
            stratify_by=split_info.get("stratify_by", "None"),
            train_ratio=split_info.get("train_ratio_actual", 0),
            val_ratio=split_info.get("val_ratio_actual", 0),
            test_ratio=split_info.get("test_ratio_actual", 0)
        )
        
        # 保存HTML报告
        html_path = Path(self.enhanced_config.reports_output_dir) / "data_split_analysis.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _generate_training_report(self) -> Optional[str]:
        """生成训练过程报告"""
        try:
            if not hasattr(self, 'trainer') or not self.trainer:
                return None
            
            # 收集训练信息
            training_info = {
                "title": "训练过程报告",
                "generated_at": datetime.now().isoformat(),
                "model_name": self.enhanced_config.model_name,
                "training_config": {
                    "num_epochs": self.enhanced_config.num_epochs,
                    "batch_size": self.enhanced_config.batch_size,
                    "learning_rate": self.enhanced_config.learning_rate,
                    "gradient_accumulation_steps": self.enhanced_config.gradient_accumulation_steps
                }
            }
            
            # 添加训练历史（如果可用）
            if hasattr(self.trainer.state, 'log_history'):
                training_info["log_history"] = self.trainer.state.log_history
            
            # 添加最终训练状态
            if hasattr(self.trainer.state, 'epoch'):
                training_info["final_epoch"] = self.trainer.state.epoch
            if hasattr(self.trainer.state, 'global_step'):
                training_info["final_step"] = self.trainer.state.global_step
            
            # 保存训练报告
            report_path = Path(self.enhanced_config.reports_output_dir) / "training_process.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"生成训练报告失败: {e}")
            return None
    
    def _generate_comprehensive_report(self, report_paths: Dict[str, str]) -> Optional[str]:
        """生成综合报告"""
        try:
            comprehensive_data = {
                "title": "增强训练Pipeline综合报告",
                "generated_at": datetime.now().isoformat(),
                "pipeline_config": asdict(self.enhanced_config),
                "report_files": report_paths,
                "summary": {
                    "data_splitting_enabled": self.enhanced_config.enable_data_splitting,
                    "comprehensive_evaluation_enabled": self.enhanced_config.enable_comprehensive_evaluation,
                    "experiment_tracking_enabled": self.enhanced_config.enable_experiment_tracking,
                    "validation_during_training": self.enhanced_config.enable_validation_during_training
                }
            }
            
            # 添加关键结果摘要
            if self.evaluation_result:
                comprehensive_data["evaluation_summary"] = {
                    "overall_metrics": self.evaluation_result.metrics,
                    "efficiency_metrics": asdict(self.evaluation_result.efficiency_metrics),
                    "quality_scores": asdict(self.evaluation_result.quality_scores)
                }
            
            if self.data_split_result:
                comprehensive_data["data_split_summary"] = self.data_split_result.split_info
            
            # 保存综合报告
            report_path = Path(self.enhanced_config.reports_output_dir) / "comprehensive_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"生成综合报告失败: {e}")
            return None
    
    def _save_report_index(self, report_paths: Dict[str, str]):
        """保存报告索引"""
        try:
            index_data = {
                "generated_at": datetime.now().isoformat(),
                "pipeline_version": "enhanced_v1.0",
                "reports": report_paths,
                "access_info": {
                    "main_report": report_paths.get("comprehensive", ""),
                    "evaluation_report": report_paths.get("evaluation_html", ""),
                    "data_split_report": report_paths.get("data_split_analysis", ""),
                    "training_report": report_paths.get("training_process", "")
                }
            }
            
            index_path = Path(self.enhanced_config.reports_output_dir) / "report_index.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"报告索引已保存: {index_path}")
            
        except Exception as e:
            self.logger.error(f"保存报告索引失败: {e}")
    
    def _start_experiment_tracking(self) -> Optional[str]:
        """开始实验跟踪"""
        try:
            if not self.experiment_tracker:
                self.logger.warning("实验跟踪器不可用")
                return None
            
            from src.evaluation.data_models import ExperimentConfig
            
            # 生成实验名称（如果未指定）
            experiment_name = self.enhanced_config.experiment_name
            if not experiment_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"enhanced_pipeline_{timestamp}"
            
            # 创建实验配置
            experiment_config_dict = self.enhanced_config.get_experiment_config()
            experiment_config_dict["experiment_name"] = experiment_name
            
            experiment_config = ExperimentConfig(
                **experiment_config_dict
            )
            
            # 开始跟踪实验
            experiment_id = self.experiment_tracker.track_experiment(
                experiment_config, 
                None  # 结果稍后更新
            )
            
            self.logger.info(f"开始实验跟踪")
            self.logger.info(f"  实验ID: {experiment_id}")
            self.logger.info(f"  实验名称: {experiment_name}")
            self.logger.info(f"  实验标签: {self.enhanced_config.experiment_tags}")
            
            # 记录实验开始
            self.logging_system.info("实验跟踪开始", "EXPERIMENT", {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "tags": self.enhanced_config.experiment_tags,
                "config_summary": {
                    "model_name": self.enhanced_config.model_name,
                    "data_splitting": self.enhanced_config.enable_data_splitting,
                    "comprehensive_evaluation": self.enhanced_config.enable_comprehensive_evaluation,
                    "num_epochs": self.enhanced_config.num_epochs,
                    "batch_size": self.enhanced_config.batch_size
                }
            })
            
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"开始实验跟踪失败: {e}")
            if self.enhanced_config.fallback_to_basic_mode:
                self.logger.warning("实验跟踪失败，但继续执行")
                return None
            raise
    
    def _update_experiment_progress(self, phase: str, progress_data: Dict[str, Any]):
        """更新实验进度"""
        try:
            if not self.experiment_tracker or not self.experiment_id:
                return
            
            # 记录进度信息
            self.logging_system.info(f"实验进度更新: {phase}", "EXPERIMENT", {
                "experiment_id": self.experiment_id,
                "phase": phase,
                "progress": progress_data
            })
            
        except Exception as e:
            self.logger.error(f"更新实验进度失败: {e}")
    
    def _complete_experiment_tracking(self, start_time: datetime, success: bool):
        """完成实验跟踪"""
        try:
            if not self.experiment_tracker or not self.experiment_id:
                return
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # 准备实验结果摘要
            experiment_summary = {
                "success": success,
                "duration": str(duration),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # 添加数据拆分结果
            if self.data_split_result:
                experiment_summary["data_split"] = {
                    "train_samples": len(self.data_split_result.train_dataset),
                    "val_samples": len(self.data_split_result.val_dataset),
                    "test_samples": len(self.data_split_result.test_dataset),
                    "consistency_score": self.data_split_result.distribution_analysis.consistency_score
                }
            
            # 添加训练结果
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer.state, 'log_history'):
                training_history = self.trainer.state.log_history
                if training_history:
                    final_train_loss = None
                    final_eval_loss = None
                    
                    # 查找最终的损失值
                    for log_entry in reversed(training_history):
                        if final_train_loss is None and 'train_loss' in log_entry:
                            final_train_loss = log_entry['train_loss']
                        if final_eval_loss is None and 'eval_loss' in log_entry:
                            final_eval_loss = log_entry['eval_loss']
                        if final_train_loss is not None and final_eval_loss is not None:
                            break
                    
                    experiment_summary["training"] = {
                        "final_train_loss": final_train_loss,
                        "final_eval_loss": final_eval_loss,
                        "total_steps": getattr(self.trainer.state, 'global_step', 0),
                        "epochs_completed": getattr(self.trainer.state, 'epoch', 0)
                    }
            
            # 更新实验结果
            if self.evaluation_result:
                # 使用现有的update_experiment_result方法
                self.experiment_tracker.update_experiment_result(
                    self.experiment_id, 
                    self.evaluation_result
                )
                
                # 添加评估摘要
                experiment_summary["evaluation"] = {
                    "overall_metrics": self.evaluation_result.metrics,
                    "efficiency": {
                        "inference_latency": self.evaluation_result.efficiency_metrics.inference_latency,
                        "throughput": self.evaluation_result.efficiency_metrics.throughput,
                        "memory_usage": self.evaluation_result.efficiency_metrics.memory_usage
                    },
                    "quality_overall": self.evaluation_result.quality_scores.overall
                }
            
            # 记录实验完成
            self.logging_system.info("实验跟踪完成", "EXPERIMENT", {
                "experiment_id": self.experiment_id,
                "success": success,
                "duration_seconds": duration.total_seconds(),
                "summary": experiment_summary
            })
            
            self.logger.info(f"实验跟踪完成")
            self.logger.info(f"  实验ID: {self.experiment_id}")
            self.logger.info(f"  执行状态: {'成功' if success else '失败'}")
            self.logger.info(f"  总耗时: {duration}")
            
            # 保存实验摘要到文件
            self._save_experiment_summary(experiment_summary)
            
        except Exception as e:
            self.logger.error(f"完成实验跟踪失败: {e}")
    
    def _save_experiment_summary(self, experiment_summary: Dict[str, Any]):
        """保存实验摘要到文件"""
        try:
            if not self.experiment_id:
                return
            
            # 创建实验摘要文件
            summary_data = {
                "experiment_id": self.experiment_id,
                "experiment_name": self.enhanced_config.experiment_name,
                "config": asdict(self.enhanced_config),
                "summary": experiment_summary,
                "generated_at": datetime.now().isoformat()
            }
            
            # 保存到实验目录
            experiment_dir = Path(self.enhanced_config.experiments_output_dir) / self.experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = experiment_dir / "experiment_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"实验摘要已保存: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"保存实验摘要失败: {e}")


def create_enhanced_argument_parser() -> argparse.ArgumentParser:
    """创建增强的命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="增强的Qwen3优化微调系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础配置
    parser.add_argument("--model-name", type=str, 
                       default="Qwen/Qwen3-4B-Thinking-2507",
                       help="要微调的模型名称")
    parser.add_argument("--output-dir", type=str, 
                       default="./enhanced-qwen3-finetuned",
                       help="输出目录")
    parser.add_argument("--data-dir", type=str, 
                       default="data/raw",
                       help="训练数据目录")
    
    # 数据拆分配置
    parser.add_argument("--enable-data-splitting", action="store_true",
                       default=True,
                       help="启用数据拆分")
    parser.add_argument("--train-ratio", type=float, 
                       default=0.7,
                       help="训练集比例")
    parser.add_argument("--val-ratio", type=float, 
                       default=0.15,
                       help="验证集比例")
    parser.add_argument("--test-ratio", type=float, 
                       default=0.15,
                       help="测试集比例")
    parser.add_argument("--stratify-by", type=str,
                       help="分层抽样字段名称")
    
    # 评估配置
    parser.add_argument("--enable-comprehensive-evaluation", action="store_true",
                       default=True,
                       help="启用全面评估")
    parser.add_argument("--evaluation-tasks", nargs="+",
                       default=["text_generation"],
                       help="评估任务列表")
    parser.add_argument("--evaluation-metrics", nargs="+",
                       default=["bleu", "rouge", "accuracy"],
                       help="评估指标列表")
    
    # 实验跟踪配置
    parser.add_argument("--enable-experiment-tracking", action="store_true",
                       default=True,
                       help="启用实验跟踪")
    parser.add_argument("--experiment-name", type=str,
                       help="实验名称")
    parser.add_argument("--experiment-tags", nargs="+",
                       default=[],
                       help="实验标签列表")
    
    # 报告配置
    parser.add_argument("--report-formats", nargs="+",
                       default=["html", "json"],
                       help="报告格式列表")
    parser.add_argument("--enable-visualization", action="store_true",
                       default=True,
                       help="启用可视化")
    
    # 配置文件
    parser.add_argument("--config", type=str,
                       help="YAML配置文件路径")
    
    # 其他原有参数
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-memory-gb", type=float, default=16.0)
    
    return parser


def main():
    """主函数"""
    # 解析命令行参数
    parser = create_enhanced_argument_parser()
    args = parser.parse_args()
    
    try:
        # 创建配置
        if args.config:
            # 从YAML文件加载配置
            config = load_enhanced_config_from_yaml(args.config)
            print(f"从配置文件加载: {args.config}")
        else:
            # 从命令行参数创建配置
            config_dict = {
                "model_name": args.model_name,
                "output_dir": args.output_dir,
                "data_dir": args.data_dir,
                "enable_data_splitting": args.enable_data_splitting,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "stratify_by": args.stratify_by,
                "enable_comprehensive_evaluation": args.enable_comprehensive_evaluation,
                "evaluation_tasks": args.evaluation_tasks,
                "evaluation_metrics": args.evaluation_metrics,
                "enable_experiment_tracking": args.enable_experiment_tracking,
                "experiment_name": args.experiment_name,
                "experiment_tags": args.experiment_tags,
                "report_formats": args.report_formats,
                "enable_visualization": args.enable_visualization,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "max_memory_gb": args.max_memory_gb
            }
            
            config = EnhancedApplicationConfig.from_dict(config_dict)
        
        # 验证配置
        errors = config.validate_config()
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        print("增强Qwen3微调系统启动")
        print(f"模型: {config.model_name}")
        print(f"数据拆分: {config.enable_data_splitting}")
        print(f"全面评估: {config.enable_comprehensive_evaluation}")
        print(f"实验跟踪: {config.enable_experiment_tracking}")
        
        # 创建并运行增强应用程序
        app = EnhancedQwenFineTuningApplication(config)
        success = app.run_enhanced_pipeline()
        
        # 退出码
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n用户中断了程序")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
