"""
配置预设管理模块

提供预定义的配置模板和预设方案，方便用户快速开始。
"""

from typing import Dict, Any
from pathlib import Path
import yaml
import json
from datetime import datetime

try:
    from .export_models import ExportConfiguration, QuantizationLevel, LogLevel
except ImportError:
    from export_models import ExportConfiguration, QuantizationLevel, LogLevel


class ConfigPresets:
    """配置预设类"""
    
    @staticmethod
    def get_quick_export_preset() -> Dict[str, Any]:
        """快速导出预设 - 适合快速测试"""
        return {
            'name': 'quick-export',
            'description': '快速导出预设，适合测试和验证',
            'config': {
                'checkpoint_path': 'qwen3-finetuned',
                'base_model_name': 'Qwen/Qwen3-4B-Thinking-2507',
                'output_directory': 'exported_models',
                'quantization_level': 'fp16',
                'remove_training_artifacts': True,
                'compress_weights': False,
                'export_pytorch': True,
                'export_onnx': False,
                'export_tensorrt': False,
                'onnx_opset_version': 20,
                'onnx_optimize_graph': True,
                'run_validation_tests': True,
                'enable_progress_monitoring': True,
                'log_level': 'INFO',
                'auto_detect_latest_checkpoint': True,
                'save_tokenizer': True,
                'naming_pattern': '{model_name}_quick_{timestamp}',
                'max_memory_usage_gb': 8.0,
                'enable_parallel_export': False
            }
        }
    
    @staticmethod
    def get_production_preset() -> Dict[str, Any]:
        """生产环境预设 - 完整优化"""
        return {
            'name': 'production',
            'description': '生产环境预设，完整优化和多格式导出',
            'config': {
                'checkpoint_path': 'qwen3-finetuned',
                'base_model_name': 'Qwen/Qwen3-4B-Thinking-2507',
                'output_directory': 'production_models',
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
                'naming_pattern': '{model_name}_prod_{timestamp}',
                'max_memory_usage_gb': 16.0,
                'enable_parallel_export': True
            }
        }
    
    @staticmethod
    def get_mobile_preset() -> Dict[str, Any]:
        """移动端预设 - 极致压缩"""
        return {
            'name': 'mobile',
            'description': '移动端预设，极致压缩适合资源受限环境',
            'config': {
                'checkpoint_path': 'qwen3-finetuned',
                'base_model_name': 'Qwen/Qwen3-4B-Thinking-2507',
                'output_directory': 'mobile_models',
                'quantization_level': 'int4',
                'remove_training_artifacts': True,
                'compress_weights': True,
                'export_pytorch': True,
                'export_onnx': True,
                'export_tensorrt': False,
                'onnx_opset_version': 20,
                'onnx_optimize_graph': True,
                'run_validation_tests': True,
                'enable_progress_monitoring': True,
                'log_level': 'WARNING',
                'auto_detect_latest_checkpoint': True,
                'save_tokenizer': True,
                'naming_pattern': '{model_name}_mobile_{timestamp}',
                'max_memory_usage_gb': 4.0,
                'enable_parallel_export': False
            }
        }
    
    @staticmethod
    def get_research_preset() -> Dict[str, Any]:
        """研究预设 - 保持最高精度"""
        return {
            'name': 'research',
            'description': '研究预设，保持最高精度用于研究和分析',
            'config': {
                'checkpoint_path': 'qwen3-finetuned',
                'base_model_name': 'Qwen/Qwen3-4B-Thinking-2507',
                'output_directory': 'research_models',
                'quantization_level': 'none',
                'remove_training_artifacts': False,
                'compress_weights': False,
                'export_pytorch': True,
                'export_onnx': True,
                'export_tensorrt': False,
                'onnx_opset_version': 20,
                'onnx_optimize_graph': False,
                'run_validation_tests': True,
                'enable_progress_monitoring': True,
                'log_level': 'DEBUG',
                'auto_detect_latest_checkpoint': True,
                'save_tokenizer': True,
                'naming_pattern': '{model_name}_research_{timestamp}',
                'max_memory_usage_gb': 32.0,
                'enable_parallel_export': False
            }
        }
    
    @staticmethod
    def get_all_presets() -> Dict[str, Dict[str, Any]]:
        """获取所有预设"""
        return {
            'quick-export': ConfigPresets.get_quick_export_preset(),
            'production': ConfigPresets.get_production_preset(),
            'mobile': ConfigPresets.get_mobile_preset(),
            'research': ConfigPresets.get_research_preset()
        }
    
    @staticmethod
    def create_preset_files(output_dir: str = "config_presets"):
        """创建预设文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        presets = ConfigPresets.get_all_presets()
        
        for preset_name, preset_data in presets.items():
            # 添加创建时间
            preset_data['created_at'] = datetime.now().isoformat()
            
            # 保存为YAML文件
            yaml_file = output_path / f"{preset_name}.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(preset_data, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            # 保存为JSON文件
            json_file = output_path / f"{preset_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
        
        print(f"预设文件已创建在: {output_path}")


class ConfigTemplates:
    """配置模板类"""
    
    @staticmethod
    def get_basic_template() -> str:
        """基础配置模板"""
        return """# 模型导出配置文件
# 基础配置模板

# Checkpoint配置
checkpoint_path: "qwen3-finetuned"
base_model_name: "Qwen/Qwen3-4B-Thinking-2507"
output_directory: "exported_models"

# 优化配置
quantization_level: "int8"  # none, fp16, int8, int4
remove_training_artifacts: true
compress_weights: true

# 导出格式
export_pytorch: true
export_onnx: true
export_tensorrt: false

# ONNX配置
onnx_opset_version: 20
onnx_optimize_graph: true

# 验证配置
run_validation_tests: true

# 监控配置
enable_progress_monitoring: true
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR

# 高级配置
auto_detect_latest_checkpoint: true
save_tokenizer: true
naming_pattern: "{model_name}_{timestamp}"
max_memory_usage_gb: 16.0
enable_parallel_export: false
"""
    
    @staticmethod
    def get_advanced_template() -> str:
        """高级配置模板"""
        return """# 模型导出配置文件
# 高级配置模板 - 包含所有可用选项

export:
  # Checkpoint配置
  checkpoint:
    path: "qwen3-finetuned"
    auto_detect_latest: true
    
  # 基座模型配置
  base_model:
    name: "Qwen/Qwen3-4B-Thinking-2507"
    load_in_4bit: false
    trust_remote_code: true
    
  # 优化配置
  optimization:
    quantization: "int8"  # none, fp16, int8, int4
    remove_artifacts: true
    compress_weights: true
    
  # 导出格式配置
  formats:
    pytorch:
      enabled: true
      save_tokenizer: true
      save_config: true
      
    onnx:
      enabled: true
      opset_version: 20
      dynamic_axes: true
      optimize_graph: true
      use_external_data_format: false
      
    tensorrt:
      enabled: false
      precision: "fp16"
      max_workspace_size: "1GB"
      
  # 验证配置
  validation:
    enabled: true
    test_samples: 5
    compare_outputs: true
    benchmark_performance: true
    tolerance: 1e-5
    
  # 输出配置
  output:
    directory: "exported_models"
    naming_pattern: "{model_name}_{timestamp}"
    create_subdirs: true
    
  # 监控配置
  monitoring:
    enable_progress: true
    log_level: "INFO"
    log_file: null
    max_memory_gb: 16.0
    
  # 高级配置
  advanced:
    enable_parallel_export: false
    retry_attempts: 3
    cleanup_temp_files: true
    save_intermediate_results: false

# 环境变量覆盖
# 可以通过环境变量覆盖任何配置项
# 例如: EXPORT_QUANTIZATION_LEVEL=int4
"""
    
    @staticmethod
    def get_docker_template() -> str:
        """Docker环境配置模板"""
        return """# Docker环境配置模板
# 适用于容器化部署

# 基本配置
checkpoint_path: "/app/checkpoints/qwen3-finetuned"
base_model_name: "Qwen/Qwen3-4B-Thinking-2507"
output_directory: "/app/output"

# 优化配置 - 适合容器环境
quantization_level: "int8"
remove_training_artifacts: true
compress_weights: true

# 导出格式
export_pytorch: true
export_onnx: true
export_tensorrt: false

# 资源限制 - 适合容器
max_memory_usage_gb: 8.0
enable_parallel_export: false

# 日志配置
log_level: "INFO"
enable_progress_monitoring: true

# 容器特定配置
auto_detect_latest_checkpoint: true
save_tokenizer: true
naming_pattern: "model_{timestamp}"

# 验证配置 - 简化以节省时间
run_validation_tests: true
"""
    
    @staticmethod
    def create_template_files(output_dir: str = "config_templates"):
        """创建模板文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        templates = {
            'basic_config.yaml': ConfigTemplates.get_basic_template(),
            'advanced_config.yaml': ConfigTemplates.get_advanced_template(),
            'docker_config.yaml': ConfigTemplates.get_docker_template()
        }
        
        for filename, content in templates.items():
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"配置模板已创建在: {output_path}")


def create_all_config_files():
    """创建所有配置文件和模板"""
    print("创建配置预设和模板...")
    
    # 创建预设文件
    ConfigPresets.create_preset_files()
    
    # 创建模板文件
    ConfigTemplates.create_template_files()
    
    print("所有配置文件创建完成!")


if __name__ == '__main__':
    create_all_config_files()