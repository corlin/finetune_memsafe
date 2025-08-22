#!/usr/bin/env python3
"""
Qwen3优化微调系统主应用程序

这是一个集成的主应用程序，协调所有组件按顺序执行完整的微调流程。
包含错误处理、恢复机制和命令行界面。
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timedelta

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.memory_optimizer import MemoryOptimizer
from src.model_manager import ModelManager
from src.lora_adapter import LoRAAdapter
from src.data_pipeline import DataPipeline
from src.training_engine import TrainingEngine, TrainingConfig
from src.inference_tester import InferenceTester
from src.logging_system import LoggingSystem
from src.progress_monitor import ProgressMonitor
from src.environment_validator import EnvironmentValidator, DependencyInstaller, UVEnvironmentManager, EnvironmentSetupManager


@dataclass
class ApplicationConfig:
    """主应用程序配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    output_dir: str = "./qwen3-finetuned"
    
    # 内存配置
    max_memory_gb: float = 1.0
    
    # 训练配置
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    max_sequence_length: int = 256
    
    # LoRA配置
    lora_r: int = 6
    lora_alpha: int = 12
    lora_dropout: float = 0.1
    
    # 数据配置
    data_dir: str = "data/raw"
    
    # 系统配置
    log_dir: str = "./logs"
    enable_tensorboard: bool = True
    enable_inference_test: bool = True
    
    # 环境配置
    verify_environment: bool = True
    auto_install_deps: bool = False


class QwenFineTuningApplication:
    """
    Qwen3微调应用程序主协调器
    
    负责协调所有组件按顺序执行完整的微调流程，
    包含错误处理和恢复机制。
    """
    
    def __init__(self, config: ApplicationConfig):
        """
        初始化应用程序
        
        Args:
            config: 应用程序配置
        """
        self.config = config
        self.logger = None
        self.logging_system = None
        self.memory_optimizer = None
        self.progress_monitor = None
        
        # 组件实例
        self.model_manager = None
        self.lora_adapter = None
        self.data_pipeline = None
        self.training_engine = None
        self.inference_tester = None
        
        # 运行状态
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        # 初始化基础组件
        self._initialize_logging()
        self._initialize_core_components()
    
    def _initialize_logging(self):
        """初始化日志系统"""
        try:
            # 创建日志目录
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
            
            # 初始化日志系统
            self.logging_system = LoggingSystem(
                log_dir=self.config.log_dir,
                run_name=f"qwen3_finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                enable_async=True
            )
            
            # 设置标准日志记录器
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(Path(self.config.log_dir) / "application.log", encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info("日志系统初始化完成")
            
        except Exception as e:
            print(f"日志系统初始化失败: {e}")
            raise
    
    def _initialize_core_components(self):
        """初始化核心组件"""
        try:
            # 内存优化器
            self.memory_optimizer = MemoryOptimizer(
                max_memory_gb=self.config.max_memory_gb
            )
            
            # 进度监控器
            self.progress_monitor = ProgressMonitor(
                memory_optimizer=self.memory_optimizer,
                logging_system=self.logging_system,
                enable_rich_display=True
            )
            
            self.logger.info("核心组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"核心组件初始化失败: {e}")
            raise
    
    def run(self) -> bool:
        """
        运行完整的微调流程
        
        Returns:
            bool: 是否成功完成
        """
        self.logger.info("开始Qwen3优化微调流程")
        self.logging_system.info("应用程序启动", "APPLICATION", {"config": asdict(self.config)})
        
        try:
            # 记录开始时间
            start_time = datetime.now()
            
            # 1. 环境验证和设置
            if self.config.verify_environment:
                self.logger.info("步骤1: 环境验证和设置")
                if not self._verify_and_setup_environment():
                    return False
            
            # 2. 内存优化和系统准备
            self.logger.info("步骤2: 内存优化和系统准备")
            self._prepare_system()
            
            # 3. 模型加载和配置
            self.logger.info("步骤3: 模型加载和配置")
            if not self._load_and_configure_model():
                return False
            
            # 4. 数据准备
            self.logger.info("步骤4: 数据准备")
            if not self._prepare_data():
                return False
            
            # 5. 训练配置和执行
            self.logger.info("步骤5: 训练配置和执行")
            if not self._configure_and_train():
                return False
            
            # 6. 推理测试
            if self.config.enable_inference_test:
                self.logger.info("步骤6: 推理测试")
                if not self._test_inference():
                    self.logger.warning("推理测试失败，但训练已完成")
            
            # 7. 生成最终报告
            self.logger.info("步骤7: 生成最终报告")
            self._generate_final_report(start_time)
            
            self.logger.info("Qwen3优化微调流程成功完成")
            return True
            
        except KeyboardInterrupt:
            self.logger.info("用户中断了微调流程")
            self._handle_interruption()
            return False
            
        except Exception as e:
            self.logger.error(f"微调流程失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self._handle_error(e)
            return False
        
        finally:
            self._cleanup()
    
    def _verify_and_setup_environment(self) -> bool:
        """验证和设置环境"""
        try:
            self.logging_system.info("开始环境验证和设置", "ENVIRONMENT")
            
            # 使用环境设置管理器进行完整的环境设置
            setup_manager = EnvironmentSetupManager(
                auto_install=self.config.auto_install_deps,
                auto_setup_uv=True
            )
            
            # 执行完整的环境设置
            setup_success = setup_manager.setup_complete_environment()
            
            # 获取最终的环境状态
            validator = EnvironmentValidator()
            env_status = validator.validate_environment()
            
            # 记录环境状态
            self.logging_system.info("环境验证完成", "ENVIRONMENT", {
                "python_version": env_status.python_version,
                "cuda_available": env_status.cuda_available,
                "gpu_memory_gb": env_status.gpu_memory_gb,
                "disk_space_gb": env_status.disk_space_gb,
                "uv_available": env_status.uv_available,
                "missing_packages": env_status.missing_packages,
                "all_checks_passed": env_status.all_checks_passed,
                "setup_success": setup_success
            })
            
            # 检查是否通过所有验证
            if not env_status.all_checks_passed and not self.config.auto_install_deps:
                self.logger.error("环境验证未通过，请解决以上问题或启用自动安装")
                return False
            
            if setup_success:
                self.logging_system.info("环境验证和设置完成", "ENVIRONMENT")
            else:
                self.logging_system.warning("环境设置部分失败，但可以继续", "ENVIRONMENT")
            
            return True
            
        except Exception as e:
            self.logger.error(f"环境验证失败: {e}")
            return False
    

    
    def _prepare_system(self):
        """准备系统和内存优化"""
        try:
            self.logging_system.info("开始系统准备", "SYSTEM")
            
            # 内存优化
            self.memory_optimizer.optimize_for_training()
            
            # 记录初始内存状态
            initial_memory = self.memory_optimizer.get_memory_status()
            self.logging_system.log_memory_status(initial_memory, 0, "Initial")
            
            self.logging_system.info("系统准备完成", "SYSTEM")
            
        except Exception as e:
            self.logger.error(f"系统准备失败: {e}")
            raise
    
    def _load_and_configure_model(self) -> bool:
        """加载和配置模型"""
        try:
            self.logging_system.info("开始模型加载", "MODEL")
            
            # 创建模型管理器
            self.model_manager = ModelManager(max_memory_gb=self.config.max_memory_gb)
            
            # 加载模型和分词器
            self.model, self.tokenizer = self.model_manager.load_model_with_quantization(
                self.config.model_name
            )
            
            # 准备模型用于训练
            self.model = self.model_manager.prepare_for_training(self.model)
            
            # 创建LoRA适配器
            self.lora_adapter = LoRAAdapter(
                r=self.config.lora_r,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout
            )
            
            # 应用LoRA
            self.model = self.lora_adapter.setup_lora_for_model(self.model)
            
            # 获取模型信息
            model_info = self.lora_adapter.get_trainable_params_info(self.model)
            self.logging_system.info("模型配置完成", "MODEL", model_info)
            
            # 记录模型加载后的内存状态
            post_model_memory = self.memory_optimizer.get_memory_status()
            self.logging_system.log_memory_status(post_model_memory, 0, "PostModel")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.logging_system.error(f"模型加载失败: {e}", "MODEL")
            return False
    
    def _prepare_data(self) -> bool:
        """准备训练数据"""
        try:
            self.logging_system.info("开始数据准备", "DATA")
            
            # 创建数据管道
            self.data_pipeline = DataPipeline(
                data_dir=self.config.data_dir,
                max_sequence_length=self.config.max_sequence_length
            )
            
            # 加载QA数据
            qa_data = self.data_pipeline.load_qa_data_from_files()
            
            # 格式化数据
            dataset = self.data_pipeline.format_for_qwen(qa_data)
            
            # 分词
            self.dataset = self.data_pipeline.tokenize_dataset(dataset, self.tokenizer)
            
            # 获取数据统计
            data_stats = self.data_pipeline.get_dataset_stats(self.dataset)
            self.logging_system.info("数据准备完成", "DATA", data_stats)
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据准备失败: {e}")
            self.logging_system.error(f"数据准备失败: {e}", "DATA")
            return False
    
    def _configure_and_train(self) -> bool:
        """配置和执行训练"""
        try:
            self.logging_system.info("开始训练配置", "TRAINING")
            
            # 创建训练配置
            training_config = TrainingConfig(
                output_dir=self.config.output_dir,
                max_memory_gb=self.config.max_memory_gb,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                num_epochs=self.config.num_epochs,
                max_sequence_length=self.config.max_sequence_length
            )
            
            # 创建训练引擎
            self.training_engine = TrainingEngine(
                config=training_config,
                memory_optimizer=self.memory_optimizer,
                logging_system=self.logging_system,
                progress_monitor=self.progress_monitor
            )
            
            # 创建数据整理器
            data_collator = self.data_pipeline.create_data_collator(self.tokenizer)
            
            # 创建训练器
            self.trainer = self.training_engine.create_trainer(
                model=self.model,
                train_dataset=self.dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            # 执行训练
            self.trainer = self.training_engine.train_model(self.trainer)
            
            # 保存模型
            self.training_engine.save_model(self.trainer)
            
            self.logging_system.info("训练完成", "TRAINING")
            return True
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            self.logging_system.error(f"训练失败: {e}", "TRAINING")
            return False
    
    def _test_inference(self) -> bool:
        """测试推理"""
        try:
            self.logging_system.info("开始推理测试", "INFERENCE")
            
            # 创建推理测试器
            self.inference_tester = InferenceTester(
                memory_optimizer=self.memory_optimizer
            )
            
            # 加载微调后的模型
            model_path = self.config.output_dir
            self.inference_tester.load_finetuned_model(
                model_path=model_path,
                base_model_name=self.config.model_name
            )
            
            # 运行推理测试
            test_results = self.inference_tester.test_model_with_multiple_prompts()
            
            # 验证模型质量
            quality_results = self.inference_tester.validate_model_quality(test_results)
            
            self.logging_system.info("推理测试完成", "INFERENCE", quality_results)
            
            # 判断是否通过质量验证
            if quality_results.get("quality_passed", False):
                self.logger.info("模型质量验证通过")
                return True
            else:
                self.logger.warning("模型质量验证未通过，但推理测试完成")
                return True  # 不因为质量问题而失败
            
        except Exception as e:
            self.logger.error(f"推理测试失败: {e}")
            self.logging_system.error(f"推理测试失败: {e}", "INFERENCE")
            return False
    
    def _generate_final_report(self, start_time: datetime):
        """生成最终报告"""
        try:
            end_time = datetime.now()
            total_duration = end_time - start_time
            
            # 生成训练引擎报告
            training_report = {}
            if self.training_engine:
                training_report = self.training_engine.generate_final_training_report()
            
            # 生成应用程序报告
            app_report = {
                "application_config": asdict(self.config),
                "execution_summary": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "total_duration": str(total_duration),
                    "success": True
                },
                "training_report": training_report,
                "final_memory_status": asdict(self.memory_optimizer.get_memory_status()) if self.memory_optimizer else {},
                "generated_at": datetime.now().isoformat()
            }
            
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
            
            # 保存报告
            report_path = Path(self.config.output_dir) / "final_application_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(app_report, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            self.logger.info(f"最终报告已生成: {report_path}")
            self.logging_system.info(f"最终报告已生成: {report_path}", "APPLICATION")
            
        except Exception as e:
            self.logger.error(f"生成最终报告失败: {e}")
    
    def _handle_interruption(self):
        """处理用户中断"""
        try:
            self.logger.info("处理用户中断...")
            
            # 保存当前状态
            if self.trainer:
                try:
                    self.trainer.save_state()
                    self.logger.info("训练状态已保存")
                except Exception as e:
                    self.logger.error(f"保存训练状态失败: {e}")
            
            # 记录中断信息
            self.logging_system.info("用户中断了微调流程", "APPLICATION")
            
        except Exception as e:
            self.logger.error(f"处理中断失败: {e}")
    
    def _handle_error(self, error: Exception):
        """处理错误"""
        try:
            self.logger.error("处理应用程序错误...")
            
            # 记录错误信息
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            }
            
            self.logging_system.error("应用程序执行失败", "APPLICATION", error_info)
            
            # 尝试保存当前状态
            if self.trainer:
                try:
                    self.trainer.save_state()
                    self.logger.info("训练状态已保存")
                except Exception as save_error:
                    self.logger.error(f"保存训练状态失败: {save_error}")
            
            # 提供错误恢复建议
            self._provide_error_recovery_suggestions(error)
            
        except Exception as e:
            self.logger.error(f"处理错误失败: {e}")
    
    def _provide_error_recovery_suggestions(self, error: Exception):
        """提供错误恢复建议"""
        error_str = str(error).lower()
        
        self.logger.error("错误恢复建议:")
        
        if "out of memory" in error_str or "cuda" in error_str:
            self.logger.error("内存相关错误:")
            self.logger.error("1. 减少batch_size参数")
            self.logger.error("2. 增加gradient_accumulation_steps")
            self.logger.error("3. 减少max_sequence_length")
            self.logger.error("4. 降低max_memory_gb限制")
            
        elif "model" in error_str or "load" in error_str:
            self.logger.error("模型相关错误:")
            self.logger.error("1. 检查模型名称是否正确")
            self.logger.error("2. 验证网络连接")
            self.logger.error("3. 检查Hugging Face访问权限")
            self.logger.error("4. 尝试使用本地模型路径")
            
        elif "data" in error_str:
            self.logger.error("数据相关错误:")
            self.logger.error("1. 检查数据目录是否存在")
            self.logger.error("2. 验证数据文件格式")
            self.logger.error("3. 检查数据文件编码")
            
        else:
            self.logger.error("通用建议:")
            self.logger.error("1. 检查所有配置参数")
            self.logger.error("2. 验证环境依赖")
            self.logger.error("3. 查看详细错误日志")
            self.logger.error("4. 尝试使用更保守的配置")
    
    def _cleanup(self):
        """清理资源"""
        try:
            self.logger.info("清理应用程序资源...")
            
            # 清理推理测试器
            if self.inference_tester:
                self.inference_tester.cleanup()
            
            # 清理内存
            if self.memory_optimizer:
                self.memory_optimizer.cleanup_gpu_memory()
            
            # 关闭日志系统
            if self.logging_system:
                self.logging_system.close()
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            print(f"资源清理失败: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Qwen3优化微调系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型配置
    parser.add_argument("--model-name", type=str, 
                       default="Qwen/Qwen3-4B-Thinking-2507",
                       help="要微调的模型名称")
    parser.add_argument("--output-dir", type=str, 
                       default="./qwen3-finetuned",
                       help="输出目录")
    
    # 内存配置
    parser.add_argument("--max-memory-gb", type=float, 
                       default=16.0,
                       help="最大GPU内存限制(GB)")
    
    # 训练配置
    parser.add_argument("--batch-size", type=int, 
                       default=48,
                       help="批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, 
                       default=16,
                       help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, 
                       default=2e-4,
                       help="学习率")
    parser.add_argument("--num-epochs", type=int, 
                       default=10,
                       help="训练轮数")
    parser.add_argument("--max-sequence-length", type=int, 
                       default=512,
                       help="最大序列长度")
    
    # LoRA配置
    parser.add_argument("--lora-r", type=int, 
                       default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, 
                       default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, 
                       default=0.1,
                       help="LoRA dropout")
    
    # 数据配置
    parser.add_argument("--data-dir", type=str, 
                       default="data/raw",
                       help="训练数据目录")
    
    # 系统配置
    parser.add_argument("--log-dir", type=str, 
                       default="./logs",
                       help="日志目录")
    parser.add_argument("--no-tensorboard", action="store_true",
                       help="禁用TensorBoard")
    parser.add_argument("--no-inference-test", action="store_true",
                       help="禁用推理测试")
    parser.add_argument("--no-verify-environment", action="store_true",
                       help="跳过环境验证")
    parser.add_argument("--auto-install-deps", action="store_true",
                       help="自动安装缺少的依赖")
    
    # 配置文件
    parser.add_argument("--config", type=str,
                       help="配置文件路径(JSON格式)")
    
    return parser


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """从文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def main():
    """主函数"""
    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # 创建配置
        config_dict = {}
        
        # 从配置文件加载
        if args.config:
            config_dict.update(load_config_from_file(args.config))
        
        # 从命令行参数更新
        config_dict.update({
            "model_name": args.model_name,
            "output_dir": args.output_dir,
            "max_memory_gb": args.max_memory_gb,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_sequence_length": args.max_sequence_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "data_dir": args.data_dir,
            "log_dir": args.log_dir,
            "enable_tensorboard": not args.no_tensorboard,
            "enable_inference_test": not args.no_inference_test,
            "verify_environment": not args.no_verify_environment,
            "auto_install_deps": args.auto_install_deps
        })
        
        # 创建应用程序配置
        config = ApplicationConfig(**config_dict)
        
        # 创建并运行应用程序
        app = QwenFineTuningApplication(config)
        success = app.run()
        
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