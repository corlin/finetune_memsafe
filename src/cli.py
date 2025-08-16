"""
命令行接口模块

提供完整的命令行工具，支持配置文件、命令行参数、交互式配置向导等功能。
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

try:
    from .export_config import ConfigurationManager, create_default_config_file
    from .export_models import ExportConfiguration, QuantizationLevel, LogLevel
    from .model_export_controller import ModelExportController
    from .export_exceptions import ModelExportError
except ImportError:
    # For direct execution or testing
    from export_config import ConfigurationManager, create_default_config_file
    from export_models import ExportConfiguration, QuantizationLevel, LogLevel
    from model_export_controller import ModelExportController
    from export_exceptions import ModelExportError


class CLIInterface:
    """命令行接口类"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('model_export_cli')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            prog='model-export',
            description='模型导出优化工具 - 将LoRA checkpoint与基座模型合并并导出为多种格式',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )
        
        # 添加子命令
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # export 命令
        self._add_export_command(subparsers)
        
        # config 命令
        self._add_config_command(subparsers)
        
        # validate 命令
        self._add_validate_command(subparsers)
        
        # wizard 命令
        self._add_wizard_command(subparsers)
        
        return parser
    
    def _add_export_command(self, subparsers):
        """添加导出命令"""
        export_parser = subparsers.add_parser(
            'export',
            help='执行模型导出',
            description='将LoRA checkpoint与基座模型合并并导出为指定格式'
        )
        
        # 基本参数
        export_parser.add_argument(
            '--checkpoint-path', '-c',
            type=str,
            help='Checkpoint目录路径 (默认: qwen3-finetuned)'
        )
        
        export_parser.add_argument(
            '--base-model', '-m',
            type=str,
            help='基座模型名称 (默认: Qwen/Qwen3-4B-Thinking-2507)'
        )
        
        export_parser.add_argument(
            '--output-dir', '-o',
            type=str,
            help='输出目录 (默认: exported_models)'
        )
        
        # 优化参数
        export_parser.add_argument(
            '--quantization', '-q',
            choices=['none', 'fp16', 'int8', 'int4'],
            help='量化级别 (默认: int8)'
        )
        
        export_parser.add_argument(
            '--no-artifacts',
            action='store_true',
            help='不移除训练artifacts'
        )
        
        export_parser.add_argument(
            '--no-compression',
            action='store_true',
            help='不压缩权重'
        )
        
        # 导出格式
        export_parser.add_argument(
            '--pytorch',
            action='store_true',
            help='导出PyTorch格式 (默认启用)'
        )
        
        export_parser.add_argument(
            '--onnx',
            action='store_true',
            help='导出ONNX格式 (默认启用)'
        )
        
        export_parser.add_argument(
            '--tensorrt',
            action='store_true',
            help='导出TensorRT格式'
        )
        
        export_parser.add_argument(
            '--no-pytorch',
            action='store_true',
            help='不导出PyTorch格式'
        )
        
        export_parser.add_argument(
            '--no-onnx',
            action='store_true',
            help='不导出ONNX格式'
        )
        
        # ONNX参数
        export_parser.add_argument(
            '--onnx-opset',
            type=int,
            help='ONNX opset版本 (默认: 20)'
        )
        
        export_parser.add_argument(
            '--no-onnx-optimize',
            action='store_true',
            help='不优化ONNX图'
        )
        
        # 验证参数
        export_parser.add_argument(
            '--no-validation',
            action='store_true',
            help='跳过验证测试'
        )
        
        # 配置文件
        export_parser.add_argument(
            '--config', '-f',
            type=str,
            help='配置文件路径'
        )
        
        # 日志级别
        export_parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            help='日志级别 (默认: INFO)'
        )
        
        # 并行导出
        export_parser.add_argument(
            '--parallel',
            action='store_true',
            help='启用并行导出'
        )
        
        # 内存限制
        export_parser.add_argument(
            '--max-memory',
            type=float,
            help='最大内存使用量(GB) (默认: 16.0)'
        )
    
    def _add_config_command(self, subparsers):
        """添加配置命令"""
        config_parser = subparsers.add_parser(
            'config',
            help='配置管理',
            description='管理导出配置文件和模板'
        )
        
        config_subparsers = config_parser.add_subparsers(dest='config_action', help='配置操作')
        
        # create-template 子命令
        template_parser = config_subparsers.add_parser(
            'create-template',
            help='创建配置模板'
        )
        template_parser.add_argument(
            '--output', '-o',
            type=str,
            default='export_config_template.yaml',
            help='模板文件输出路径'
        )
        
        # validate 子命令
        validate_parser = config_subparsers.add_parser(
            'validate',
            help='验证配置文件'
        )
        validate_parser.add_argument(
            'config_file',
            type=str,
            help='要验证的配置文件路径'
        )
        
        # show 子命令
        show_parser = config_subparsers.add_parser(
            'show',
            help='显示当前配置'
        )
        show_parser.add_argument(
            '--config', '-f',
            type=str,
            help='配置文件路径'
        )
        
        # presets 子命令
        presets_parser = config_subparsers.add_parser(
            'presets',
            help='管理配置预设'
        )
        presets_subparsers = presets_parser.add_subparsers(dest='preset_action', help='预设操作')
        
        # list presets
        presets_subparsers.add_parser('list', help='列出可用预设')
        
        # create preset
        create_preset_parser = presets_subparsers.add_parser('create', help='创建预设')
        create_preset_parser.add_argument('name', help='预设名称')
        create_preset_parser.add_argument('--description', help='预设描述')
        
        # use preset
        use_preset_parser = presets_subparsers.add_parser('use', help='使用预设')
        use_preset_parser.add_argument('name', help='预设名称')
        use_preset_parser.add_argument('--output', '-o', help='输出配置文件路径')
    
    def _add_validate_command(self, subparsers):
        """添加验证命令"""
        validate_parser = subparsers.add_parser(
            'validate',
            help='验证导出的模型',
            description='验证已导出模型的功能性和一致性'
        )
        
        validate_parser.add_argument(
            'model_path',
            type=str,
            help='要验证的模型路径'
        )
        
        validate_parser.add_argument(
            '--format',
            choices=['pytorch', 'onnx', 'tensorrt'],
            default='pytorch',
            help='模型格式 (默认: pytorch)'
        )
        
        validate_parser.add_argument(
            '--test-samples',
            type=int,
            default=5,
            help='测试样本数量 (默认: 5)'
        )
        
        validate_parser.add_argument(
            '--compare-with',
            type=str,
            help='与另一个模型比较输出'
        )
        
        validate_parser.add_argument(
            '--benchmark',
            action='store_true',
            help='运行性能基准测试'
        )
        
        validate_parser.add_argument(
            '--output-report',
            type=str,
            help='验证报告输出路径'
        )
    
    def _add_wizard_command(self, subparsers):
        """添加配置向导命令"""
        wizard_parser = subparsers.add_parser(
            'wizard',
            help='交互式配置向导',
            description='通过交互式界面创建导出配置'
        )
        
        wizard_parser.add_argument(
            '--output', '-o',
            type=str,
            default='export_config.yaml',
            help='配置文件输出路径'
        )
        
        wizard_parser.add_argument(
            '--preset',
            type=str,
            help='基于预设开始配置'
        )
    
    def _get_usage_examples(self) -> str:
        """获取使用示例"""
        return """
使用示例:

  # 基本导出
  model-export export --checkpoint-path qwen3-finetuned --output-dir my_models

  # 使用配置文件导出
  model-export export --config export_config.yaml

  # 自定义量化和格式
  model-export export -c qwen3-finetuned -q int4 --onnx --tensorrt

  # 创建配置模板
  model-export config create-template --output my_config.yaml

  # 交互式配置向导
  model-export wizard --output my_config.yaml

  # 验证导出的模型
  model-export validate exported_models/qwen3_merged --benchmark

  # 显示当前配置
  model-export config show --config export_config.yaml

环境变量:
  EXPORT_CHECKPOINT_PATH    Checkpoint路径
  EXPORT_OUTPUT_DIR         输出目录
  EXPORT_QUANTIZATION_LEVEL 量化级别
  EXPORT_LOG_LEVEL          日志级别
  
更多信息请访问: https://github.com/your-repo/model-export
"""
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """运行CLI"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        try:
            if parsed_args.command == 'export':
                return self._handle_export_command(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config_command(parsed_args)
            elif parsed_args.command == 'validate':
                return self._handle_validate_command(parsed_args)
            elif parsed_args.command == 'wizard':
                return self._handle_wizard_command(parsed_args)
            else:
                self.logger.error(f"未知命令: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            self.logger.info("用户中断操作")
            return 130
        except Exception as e:
            self.logger.error(f"执行失败: {e}")
            if self.logger.level <= logging.DEBUG:
                import traceback
                traceback.print_exc()
            return 1
    
    def _handle_export_command(self, args) -> int:
        """处理导出命令"""
        try:
            # 构建配置参数
            config_kwargs = {}
            
            if args.checkpoint_path:
                config_kwargs['checkpoint_path'] = args.checkpoint_path
            if args.base_model:
                config_kwargs['base_model_name'] = args.base_model
            if args.output_dir:
                config_kwargs['output_directory'] = args.output_dir
            if args.quantization:
                config_kwargs['quantization_level'] = args.quantization
            if args.no_artifacts:
                config_kwargs['remove_training_artifacts'] = False
            if args.no_compression:
                config_kwargs['compress_weights'] = False
            if args.pytorch or not (args.no_pytorch or args.onnx or args.tensorrt):
                config_kwargs['export_pytorch'] = True
            if args.no_pytorch:
                config_kwargs['export_pytorch'] = False
            if args.onnx or not (args.no_onnx or args.pytorch or args.tensorrt):
                config_kwargs['export_onnx'] = True
            if args.no_onnx:
                config_kwargs['export_onnx'] = False
            if args.tensorrt:
                config_kwargs['export_tensorrt'] = True
            if args.onnx_opset:
                config_kwargs['onnx_opset_version'] = args.onnx_opset
            if args.no_onnx_optimize:
                config_kwargs['onnx_optimize_graph'] = False
            if args.no_validation:
                config_kwargs['run_validation_tests'] = False
            if args.log_level:
                config_kwargs['log_level'] = args.log_level
            if args.parallel:
                config_kwargs['enable_parallel_export'] = True
            if args.max_memory:
                config_kwargs['max_memory_usage_gb'] = args.max_memory
            
            # 加载配置
            self.config_manager.config_path = args.config if hasattr(args, 'config') else None
            config = self.config_manager.load_configuration(**config_kwargs)
            
            # 设置日志级别
            if config.log_level:
                self.logger.setLevel(getattr(logging, config.log_level.value))
            
            self.logger.info("开始模型导出...")
            self.logger.info(f"Checkpoint路径: {config.checkpoint_path}")
            self.logger.info(f"基座模型: {config.base_model_name}")
            self.logger.info(f"输出目录: {config.output_directory}")
            self.logger.info(f"量化级别: {config.quantization_level.value}")
            
            # 执行导出
            controller = ModelExportController(config)
            result = controller.export_model()
            
            if result.success:
                self.logger.info("模型导出成功!")
                self.logger.info(f"导出ID: {result.export_id}")
                if result.pytorch_model_path:
                    self.logger.info(f"PyTorch模型: {result.pytorch_model_path}")
                if result.onnx_model_path:
                    self.logger.info(f"ONNX模型: {result.onnx_model_path}")
                if result.tensorrt_model_path:
                    self.logger.info(f"TensorRT模型: {result.tensorrt_model_path}")
                
                self.logger.info(f"原始大小: {result.original_size_mb:.1f} MB")
                self.logger.info(f"优化后大小: {result.optimized_size_mb:.1f} MB")
                self.logger.info(f"压缩率: {result.size_reduction_percentage:.1f}%")
                
                return 0
            else:
                self.logger.error("模型导出失败!")
                if result.error_message:
                    self.logger.error(f"错误信息: {result.error_message}")
                return 1
                
        except Exception as e:
            self.logger.error(f"导出过程中发生错误: {e}")
            return 1
    
    def _handle_config_command(self, args) -> int:
        """处理配置命令"""
        if not args.config_action:
            self.logger.error("请指定配置操作")
            return 1
        
        try:
            if args.config_action == 'create-template':
                return self._create_config_template(args.output)
            elif args.config_action == 'validate':
                return self._validate_config_file(args.config_file)
            elif args.config_action == 'show':
                return self._show_config(getattr(args, 'config', None))
            elif args.config_action == 'presets':
                return self._handle_presets_command(args)
            else:
                self.logger.error(f"未知配置操作: {args.config_action}")
                return 1
                
        except Exception as e:
            self.logger.error(f"配置操作失败: {e}")
            return 1
    
    def _create_config_template(self, output_path: str) -> int:
        """创建配置模板"""
        try:
            create_default_config_file(output_path)
            self.logger.info(f"配置模板已创建: {output_path}")
            return 0
        except Exception as e:
            self.logger.error(f"创建配置模板失败: {e}")
            return 1
    
    def _validate_config_file(self, config_file: str) -> int:
        """验证配置文件"""
        try:
            config_manager = ConfigurationManager(config_file)
            config = config_manager.load_configuration()
            
            errors = config.validate()
            if errors:
                self.logger.error("配置验证失败:")
                for error in errors:
                    self.logger.error(f"  - {error}")
                return 1
            else:
                self.logger.info("配置验证通过")
                return 0
                
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return 1
    
    def _show_config(self, config_file: Optional[str]) -> int:
        """显示当前配置"""
        try:
            config_manager = ConfigurationManager(config_file)
            config = config_manager.load_configuration()
            
            print("当前配置:")
            print("=" * 50)
            
            # 基本配置
            print(f"Checkpoint路径: {config.checkpoint_path}")
            print(f"基座模型: {config.base_model_name}")
            print(f"输出目录: {config.output_directory}")
            print(f"量化级别: {config.quantization_level.value}")
            print(f"移除训练artifacts: {config.remove_training_artifacts}")
            print(f"压缩权重: {config.compress_weights}")
            
            # 导出格式
            print("\n导出格式:")
            print(f"  PyTorch: {config.export_pytorch}")
            print(f"  ONNX: {config.export_onnx}")
            print(f"  TensorRT: {config.export_tensorrt}")
            
            # ONNX配置
            if config.export_onnx:
                print(f"\nONNX配置:")
                print(f"  Opset版本: {config.onnx_opset_version}")
                print(f"  优化图: {config.onnx_optimize_graph}")
            
            # 验证配置
            print(f"\n验证测试: {config.run_validation_tests}")
            
            # 监控配置
            print(f"\n监控配置:")
            print(f"  进度监控: {config.enable_progress_monitoring}")
            print(f"  日志级别: {config.log_level.value}")
            print(f"  最大内存: {config.max_memory_usage_gb} GB")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"显示配置失败: {e}")
            return 1
    
    def _handle_presets_command(self, args) -> int:
        """处理预设命令"""
        preset_manager = ConfigPresetManager()
        
        if not args.preset_action:
            self.logger.error("请指定预设操作")
            return 1
        
        try:
            if args.preset_action == 'list':
                return preset_manager.list_presets()
            elif args.preset_action == 'create':
                return preset_manager.create_preset(args.name, getattr(args, 'description', None))
            elif args.preset_action == 'use':
                return preset_manager.use_preset(args.name, getattr(args, 'output', None))
            else:
                self.logger.error(f"未知预设操作: {args.preset_action}")
                return 1
                
        except Exception as e:
            self.logger.error(f"预设操作失败: {e}")
            return 1
    
    def _handle_validate_command(self, args) -> int:
        """处理验证命令"""
        try:
            from .validation_tester import ValidationTester
            
            self.logger.info(f"验证模型: {args.model_path}")
            
            # 创建验证器
            validator = ValidationTester()
            
            # 运行验证
            if args.format == 'pytorch':
                results = validator.test_pytorch_model_functionality(
                    args.model_path,
                    num_samples=args.test_samples
                )
            elif args.format == 'onnx':
                results = validator.test_onnx_model_functionality(
                    args.model_path,
                    num_samples=args.test_samples
                )
            else:
                self.logger.error(f"不支持的模型格式: {args.format}")
                return 1
            
            # 输出结果
            if results['success']:
                self.logger.info("模型验证通过")
                if args.benchmark:
                    self.logger.info(f"平均推理时间: {results.get('avg_inference_time', 'N/A')} ms")
                    self.logger.info(f"内存使用: {results.get('memory_usage', 'N/A')} MB")
            else:
                self.logger.error("模型验证失败")
                if 'error' in results:
                    self.logger.error(f"错误: {results['error']}")
                return 1
            
            # 保存报告
            if args.output_report:
                with open(args.output_report, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                self.logger.info(f"验证报告已保存: {args.output_report}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"验证失败: {e}")
            return 1
    
    def _handle_wizard_command(self, args) -> int:
        """处理配置向导命令"""
        try:
            wizard = ConfigurationWizard()
            config = wizard.run_wizard(preset=getattr(args, 'preset', None))
            
            # 保存配置
            config_manager = ConfigurationManager()
            config_manager.save_configuration(config, args.output)
            
            self.logger.info(f"配置已保存到: {args.output}")
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("配置向导已取消")
            return 130
        except Exception as e:
            self.logger.error(f"配置向导失败: {e}")
            return 1


class ConfigPresetManager:
    """配置预设管理器"""
    
    def __init__(self):
        self.presets_dir = Path.home() / '.model_export' / 'presets'
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def list_presets(self) -> int:
        """列出可用预设"""
        preset_files = list(self.presets_dir.glob('*.yaml'))
        
        if not preset_files:
            print("没有可用的配置预设")
            return 0
        
        print("可用的配置预设:")
        print("=" * 30)
        
        for preset_file in preset_files:
            preset_name = preset_file.stem
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = yaml.safe_load(f)
                    description = preset_data.get('description', '无描述')
                    print(f"{preset_name}: {description}")
            except Exception as e:
                print(f"{preset_name}: (读取失败: {e})")
        
        return 0
    
    def create_preset(self, name: str, description: Optional[str] = None) -> int:
        """创建预设"""
        preset_file = self.presets_dir / f"{name}.yaml"
        
        if preset_file.exists():
            response = input(f"预设 '{name}' 已存在，是否覆盖? (y/N): ")
            if response.lower() != 'y':
                print("操作已取消")
                return 0
        
        # 获取当前配置作为预设
        config_manager = ConfigurationManager()
        config = config_manager.load_configuration()
        
        preset_data = {
            'name': name,
            'description': description or f"用户创建的预设: {name}",
            'created_at': datetime.now().isoformat(),
            'config': config_manager._configuration_to_dict(config)
        }
        
        try:
            with open(preset_file, 'w', encoding='utf-8') as f:
                yaml.dump(preset_data, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            print(f"预设 '{name}' 已创建")
            return 0
            
        except Exception as e:
            self.logger.error(f"创建预设失败: {e}")
            return 1
    
    def use_preset(self, name: str, output_path: Optional[str] = None) -> int:
        """使用预设"""
        preset_file = self.presets_dir / f"{name}.yaml"
        
        if not preset_file.exists():
            self.logger.error(f"预设 '{name}' 不存在")
            return 1
        
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                preset_data = yaml.safe_load(f)
            
            config_dict = preset_data.get('config', {})
            
            # 创建配置对象
            config_manager = ConfigurationManager()
            config = config_manager._dict_to_configuration(config_dict)
            
            # 保存配置
            output_path = output_path or f"export_config_{name}.yaml"
            config_manager.save_configuration(config, output_path)
            
            print(f"预设 '{name}' 已应用到: {output_path}")
            return 0
            
        except Exception as e:
            self.logger.error(f"使用预设失败: {e}")
            return 1


class ConfigurationWizard:
    """交互式配置向导"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_wizard(self, preset: Optional[str] = None) -> ExportConfiguration:
        """运行配置向导"""
        print("=" * 60)
        print("模型导出配置向导")
        print("=" * 60)
        print()
        
        # 基础配置
        config_dict = {}
        
        if preset:
            # 从预设开始
            preset_manager = ConfigPresetManager()
            preset_file = preset_manager.presets_dir / f"{preset}.yaml"
            if preset_file.exists():
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = yaml.safe_load(f)
                    config_dict = preset_data.get('config', {})
                print(f"基于预设 '{preset}' 开始配置\n")
        
        # 基本设置
        print("1. 基本设置")
        print("-" * 20)
        
        config_dict['checkpoint_path'] = self._prompt_input(
            "Checkpoint路径",
            config_dict.get('checkpoint_path', 'qwen3-finetuned')
        )
        
        config_dict['base_model_name'] = self._prompt_input(
            "基座模型名称",
            config_dict.get('base_model_name', 'Qwen/Qwen3-4B-Thinking-2507')
        )
        
        config_dict['output_directory'] = self._prompt_input(
            "输出目录",
            config_dict.get('output_directory', 'exported_models')
        )
        
        # 优化设置
        print("\n2. 优化设置")
        print("-" * 20)
        
        config_dict['quantization_level'] = self._prompt_choice(
            "量化级别",
            ['none', 'fp16', 'int8', 'int4'],
            config_dict.get('quantization_level', 'int8')
        )
        
        config_dict['remove_training_artifacts'] = self._prompt_bool(
            "移除训练artifacts",
            config_dict.get('remove_training_artifacts', True)
        )
        
        config_dict['compress_weights'] = self._prompt_bool(
            "压缩权重",
            config_dict.get('compress_weights', True)
        )
        
        # 导出格式
        print("\n3. 导出格式")
        print("-" * 20)
        
        config_dict['export_pytorch'] = self._prompt_bool(
            "导出PyTorch格式",
            config_dict.get('export_pytorch', True)
        )
        
        config_dict['export_onnx'] = self._prompt_bool(
            "导出ONNX格式",
            config_dict.get('export_onnx', True)
        )
        
        config_dict['export_tensorrt'] = self._prompt_bool(
            "导出TensorRT格式",
            config_dict.get('export_tensorrt', False)
        )
        
        # ONNX设置
        if config_dict['export_onnx']:
            print("\n4. ONNX设置")
            print("-" * 20)
            
            config_dict['onnx_opset_version'] = int(self._prompt_input(
                "ONNX Opset版本",
                str(config_dict.get('onnx_opset_version', 20))
            ))
            
            config_dict['onnx_optimize_graph'] = self._prompt_bool(
                "优化ONNX图",
                config_dict.get('onnx_optimize_graph', True)
            )
        
        # 验证设置
        print("\n5. 验证设置")
        print("-" * 20)
        
        config_dict['run_validation_tests'] = self._prompt_bool(
            "运行验证测试",
            config_dict.get('run_validation_tests', True)
        )
        
        # 高级设置
        print("\n6. 高级设置")
        print("-" * 20)
        
        config_dict['log_level'] = self._prompt_choice(
            "日志级别",
            ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            config_dict.get('log_level', 'INFO')
        )
        
        config_dict['max_memory_usage_gb'] = float(self._prompt_input(
            "最大内存使用(GB)",
            str(config_dict.get('max_memory_usage_gb', 16.0))
        ))
        
        config_dict['enable_parallel_export'] = self._prompt_bool(
            "启用并行导出",
            config_dict.get('enable_parallel_export', False)
        )
        
        print("\n配置完成!")
        
        # 创建配置对象
        config_manager = ConfigurationManager()
        return config_manager._dict_to_configuration(config_dict)
    
    def _prompt_input(self, prompt: str, default: str) -> str:
        """输入提示"""
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    
    def _prompt_bool(self, prompt: str, default: bool) -> bool:
        """布尔值提示"""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        
        return response in ['y', 'yes', 'true', '1']
    
    def _prompt_choice(self, prompt: str, choices: List[str], default: str) -> str:
        """选择提示"""
        print(f"{prompt}:")
        for i, choice in enumerate(choices, 1):
            marker = " (默认)" if choice == default else ""
            print(f"  {i}. {choice}{marker}")
        
        while True:
            response = input("请选择 [1-{}]: ".format(len(choices))).strip()
            
            if not response:
                return default
            
            try:
                index = int(response) - 1
                if 0 <= index < len(choices):
                    return choices[index]
                else:
                    print("无效选择，请重试")
            except ValueError:
                # 尝试直接匹配选项名称
                if response.lower() in [c.lower() for c in choices]:
                    return next(c for c in choices if c.lower() == response.lower())
                print("无效输入，请重试")


def main():
    """主入口函数"""
    cli = CLIInterface()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())