"""
模型导出配置管理系统

本模块提供配置文件解析、环境变量支持和默认值管理功能。
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

try:
    from .export_models import ExportConfiguration, QuantizationLevel, LogLevel
except ImportError:
    from export_models import ExportConfiguration, QuantizationLevel, LogLevel


class ConfigurationManager:
    """配置管理器"""
    
    # 环境变量前缀
    ENV_PREFIX = "EXPORT_"
    
    # 默认配置文件名
    DEFAULT_CONFIG_FILES = [
        "export_config.yaml",
        "export_config.yml", 
        "export_config.json",
        ".export_config.yaml",
        ".export_config.yml",
        ".export_config.json"
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则自动搜索
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
    
    def load_configuration(self, **kwargs) -> ExportConfiguration:
        """
        加载配置，优先级：命令行参数 > 环境变量 > 配置文件 > 默认值
        
        Args:
            **kwargs: 命令行参数覆盖
            
        Returns:
            ExportConfiguration: 完整的配置对象
        """
        # 1. 加载默认配置
        config_dict = self._get_default_config()
        
        # 2. 加载配置文件
        file_config = self._load_config_file()
        if file_config:
            config_dict.update(file_config)
        
        # 3. 加载环境变量
        env_config = self._load_environment_variables()
        config_dict.update(env_config)
        
        # 4. 应用命令行参数
        config_dict.update(kwargs)
        
        # 5. 转换为配置对象
        return self._dict_to_configuration(config_dict)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'checkpoint_path': 'qwen3-finetuned',
            'base_model_name': 'Qwen/Qwen3-4B-Thinking-2507',
            'output_directory': 'exported_models',
            'quantization_level': 'int8',
            'remove_training_artifacts': True,
            'compress_weights': True,
            'export_pytorch': True,
            'export_onnx': True,
            'export_tensorrt': False,
            'onnx_opset_version': 20,
            'onnx_optimize_graph': True,
            'run_validation_tests': True,
            'enable_progress_monitoring': True,
            'log_level': 'INFO',
            'auto_detect_latest_checkpoint': True,
            'save_tokenizer': True,
            'naming_pattern': '{model_name}_{timestamp}',
            'max_memory_usage_gb': 16.0,
            'enable_parallel_export': False
        }
    
    def _load_config_file(self) -> Optional[Dict[str, Any]]:
        """加载配置文件"""
        config_file = self._find_config_file()
        if not config_file:
            self.logger.info("未找到配置文件，使用默认配置")
            return None
        
        try:
            self.logger.info(f"加载配置文件: {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
                    
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return None
    
    def _find_config_file(self) -> Optional[Path]:
        """查找配置文件"""
        if self.config_path:
            config_path = Path(self.config_path)
            if config_path.exists():
                return config_path
            else:
                self.logger.warning(f"指定的配置文件不存在: {self.config_path}")
        
        # 在当前目录搜索默认配置文件
        for filename in self.DEFAULT_CONFIG_FILES:
            config_path = Path(filename)
            if config_path.exists():
                return config_path
        
        return None
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """加载环境变量配置"""
        env_config = {}
        
        # 环境变量映射
        env_mappings = {
            'EXPORT_CHECKPOINT_PATH': 'checkpoint_path',
            'EXPORT_BASE_MODEL_NAME': 'base_model_name',
            'EXPORT_OUTPUT_DIR': 'output_directory',
            'EXPORT_QUANTIZATION_LEVEL': 'quantization_level',
            'EXPORT_REMOVE_ARTIFACTS': 'remove_training_artifacts',
            'EXPORT_COMPRESS_WEIGHTS': 'compress_weights',
            'EXPORT_PYTORCH': 'export_pytorch',
            'EXPORT_ONNX': 'export_onnx',
            'EXPORT_TENSORRT': 'export_tensorrt',
            'EXPORT_ONNX_OPSET': 'onnx_opset_version',
            'EXPORT_ONNX_OPTIMIZE': 'onnx_optimize_graph',
            'EXPORT_VALIDATION': 'run_validation_tests',
            'EXPORT_MONITORING': 'enable_progress_monitoring',
            'EXPORT_LOG_LEVEL': 'log_level',
            'EXPORT_AUTO_DETECT': 'auto_detect_latest_checkpoint',
            'EXPORT_SAVE_TOKENIZER': 'save_tokenizer',
            'EXPORT_NAMING_PATTERN': 'naming_pattern',
            'EXPORT_MAX_MEMORY_GB': 'max_memory_usage_gb',
            'EXPORT_PARALLEL': 'enable_parallel_export'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 类型转换
                converted_value = self._convert_env_value(value, config_key)
                if converted_value is not None:
                    env_config[config_key] = converted_value
                    self.logger.debug(f"从环境变量加载: {config_key} = {converted_value}")
        
        return env_config
    
    def _convert_env_value(self, value: str, config_key: str) -> Any:
        """转换环境变量值到正确的类型"""
        try:
            # 布尔值
            if config_key in ['remove_training_artifacts', 'compress_weights', 
                             'export_pytorch', 'export_onnx', 'export_tensorrt',
                             'onnx_optimize_graph', 'run_validation_tests',
                             'enable_progress_monitoring', 'auto_detect_latest_checkpoint',
                             'save_tokenizer', 'enable_parallel_export']:
                return value.lower() in ['true', '1', 'yes', 'on']
            
            # 整数值
            elif config_key in ['onnx_opset_version']:
                return int(value)
            
            # 浮点数值
            elif config_key in ['max_memory_usage_gb']:
                return float(value)
            
            # 字符串值
            else:
                return value
                
        except (ValueError, TypeError) as e:
            self.logger.warning(f"环境变量 {config_key} 值转换失败: {value}, 错误: {e}")
            return None
    
    def _dict_to_configuration(self, config_dict: Dict[str, Any]) -> ExportConfiguration:
        """将字典转换为配置对象"""
        # 处理枚举类型
        if 'quantization_level' in config_dict:
            quant_level = config_dict['quantization_level']
            if isinstance(quant_level, str):
                config_dict['quantization_level'] = QuantizationLevel(quant_level.lower())
        
        if 'log_level' in config_dict:
            log_level = config_dict['log_level']
            if isinstance(log_level, str):
                config_dict['log_level'] = LogLevel(log_level.upper())
        
        # 处理ONNX动态轴配置
        if 'onnx_dynamic_axes' in config_dict and config_dict['onnx_dynamic_axes'] is None:
            # 使用默认值，在ExportConfiguration.__post_init__中设置
            pass
        
        # 处理测试样本
        if 'test_input_samples' in config_dict and config_dict['test_input_samples'] is None:
            # 使用默认值，在ExportConfiguration.__post_init__中设置
            pass
        
        return ExportConfiguration(**config_dict)
    
    def save_configuration(self, config: ExportConfiguration, output_path: str):
        """保存配置到文件"""
        config_dict = self._configuration_to_dict(config)
        
        output_path = Path(output_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            
            self.logger.info(f"配置已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            raise
    
    def _configuration_to_dict(self, config: ExportConfiguration) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        config_dict = {}
        
        for field_name, field_value in config.__dict__.items():
            if isinstance(field_value, (QuantizationLevel, LogLevel)):
                config_dict[field_name] = field_value.value
            else:
                config_dict[field_name] = field_value
        
        return config_dict
    
    def create_config_template(self, output_path: str = "export_config_template.yaml"):
        """创建配置模板文件"""
        template_config = {
            'export': {
                'checkpoint': {
                    'path': 'qwen3-finetuned',
                    'auto_detect_latest': True
                },
                'base_model': {
                    'name': 'Qwen/Qwen3-4B-Thinking-2507',
                    'load_in_4bit': False
                },
                'optimization': {
                    'quantization': 'int8',  # none, fp16, int8, int4
                    'remove_artifacts': True,
                    'compress_weights': True
                },
                'formats': {
                    'pytorch': {
                        'enabled': True,
                        'save_tokenizer': True
                    },
                    'onnx': {
                        'enabled': True,
                        'opset_version': 20,
                        'dynamic_axes': True,
                        'optimize_graph': True
                    },
                    'tensorrt': {
                        'enabled': False
                    }
                },
                'validation': {
                    'enabled': True,
                    'test_samples': 5,
                    'compare_outputs': True,
                    'benchmark_performance': True
                },
                'output': {
                    'directory': 'exported_models',
                    'naming_pattern': '{model_name}_{timestamp}'
                },
                'monitoring': {
                    'enable_progress': True,
                    'log_level': 'INFO',
                    'max_memory_gb': 16.0
                },
                'advanced': {
                    'enable_parallel_export': False
                }
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(template_config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            self.logger.info(f"配置模板已创建: {output_path}")
            
        except Exception as e:
            self.logger.error(f"创建配置模板失败: {e}")
            raise
    
    def validate_configuration(self, config: ExportConfiguration) -> List[str]:
        """验证配置的完整性和有效性"""
        return config.validate()


def load_export_configuration(config_path: Optional[str] = None, **kwargs) -> ExportConfiguration:
    """
    便捷函数：加载导出配置
    
    Args:
        config_path: 配置文件路径
        **kwargs: 额外的配置参数
        
    Returns:
        ExportConfiguration: 配置对象
    """
    manager = ConfigurationManager(config_path)
    return manager.load_configuration(**kwargs)


def create_default_config_file(output_path: str = "export_config.yaml"):
    """
    便捷函数：创建默认配置文件
    
    Args:
        output_path: 输出文件路径
    """
    manager = ConfigurationManager()
    manager.create_config_template(output_path)