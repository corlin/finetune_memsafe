"""
配置验证器

验证评估配置的有效性，提供默认配置和配置示例。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    配置验证器
    
    验证数据处理配置的有效性，提供默认配置和配置建议。
    """
    
    def __init__(self):
        """初始化配置验证器"""
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_field_mapping_config(self, config: Dict[str, Any]) -> bool:
        """
        验证字段映射配置
        
        Args:
            config: 字段映射配置
            
        Returns:
            是否有效
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        if not isinstance(config, dict):
            self.validation_errors.append("字段映射配置必须是字典类型")
            return False
        
        # 验证每个任务的配置
        for task_name, task_config in config.items():
            if not isinstance(task_config, dict):
                self.validation_errors.append(f"任务 '{task_name}' 的配置必须是字典类型")
                continue
            
            # 验证字段列表
            for field_type, field_list in task_config.items():
                if not isinstance(field_list, list):
                    self.validation_errors.append(
                        f"任务 '{task_name}' 的字段类型 '{field_type}' 必须是列表类型"
                    )
                    continue
                
                # 检查字段名称
                for field_name in field_list:
                    if not isinstance(field_name, str):
                        self.validation_errors.append(
                            f"任务 '{task_name}' 的字段名称必须是字符串类型: {field_name}"
                        )
                    elif not field_name.strip():
                        self.validation_errors.append(
                            f"任务 '{task_name}' 包含空的字段名称"
                        )
        
        # 检查是否有推荐的任务类型
        recommended_tasks = ["text_generation", "question_answering", "classification"]
        for task in recommended_tasks:
            if task not in config:
                self.validation_warnings.append(f"建议添加任务 '{task}' 的字段映射配置")
        
        return len(self.validation_errors) == 0
    
    def validate_processing_config(self, config: Dict[str, Any]) -> bool:
        """
        验证数据处理配置
        
        Args:
            config: 数据处理配置
            
        Returns:
            是否有效
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        if not isinstance(config, dict):
            self.validation_errors.append("数据处理配置必须是字典类型")
            return False
        
        # 验证验证配置
        if "validation" in config:
            validation_config = config["validation"]
            if not isinstance(validation_config, dict):
                self.validation_errors.append("validation配置必须是字典类型")
            else:
                self._validate_validation_config(validation_config)
        
        # 验证诊断配置
        if "diagnostics" in config:
            diagnostics_config = config["diagnostics"]
            if not isinstance(diagnostics_config, dict):
                self.validation_errors.append("diagnostics配置必须是字典类型")
            else:
                self._validate_diagnostics_config(diagnostics_config)
        
        # 验证字段映射配置
        if "field_mapping" in config:
            field_mapping_config = config["field_mapping"]
            if not self.validate_field_mapping_config(field_mapping_config):
                # 错误已经添加到validation_errors中
                pass
        
        return len(self.validation_errors) == 0
    
    def _validate_validation_config(self, config: Dict[str, Any]):
        """验证validation配置"""
        # 验证min_valid_samples_ratio
        if "min_valid_samples_ratio" in config:
            ratio = config["min_valid_samples_ratio"]
            if not isinstance(ratio, (int, float)):
                self.validation_errors.append("min_valid_samples_ratio必须是数字类型")
            elif not 0 <= ratio <= 1:
                self.validation_errors.append("min_valid_samples_ratio必须在0-1之间")
        
        # 验证布尔值配置
        bool_configs = ["skip_empty_batches", "enable_data_cleaning", "enable_fallback"]
        for config_name in bool_configs:
            if config_name in config:
                value = config[config_name]
                if not isinstance(value, bool):
                    self.validation_errors.append(f"{config_name}必须是布尔类型")
    
    def _validate_diagnostics_config(self, config: Dict[str, Any]):
        """验证diagnostics配置"""
        bool_configs = ["enable_detailed_logging", "log_batch_statistics", "save_processing_report"]
        for config_name in bool_configs:
            if config_name in config:
                value = config[config_name]
                if not isinstance(value, bool):
                    self.validation_errors.append(f"{config_name}必须是布尔类型")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
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
                },
                "similarity": {
                    "input_fields": ["text1", "sentence1", "text_a"],
                    "input2_fields": ["text2", "sentence2", "text_b"],
                    "target_fields": ["label", "score", "similarity"]
                }
            },
            "validation": {
                "min_valid_samples_ratio": 0.1,
                "skip_empty_batches": True,
                "enable_data_cleaning": True,
                "enable_fallback": True
            },
            "diagnostics": {
                "enable_detailed_logging": False,
                "log_batch_statistics": True,
                "save_processing_report": True
            }
        }
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        获取配置模板（带注释说明）
        
        Returns:
            配置模板
        """
        return {
            "field_mapping": {
                "# 任务特定的字段映射配置": "每个任务可以定义自己的输入和目标字段",
                "text_generation": {
                    "input_fields": ["text", "input", "prompt"],
                    "target_fields": ["target", "answer", "output"]
                },
                "question_answering": {
                    "input_fields": ["question", "query"],
                    "context_fields": ["context", "passage"],
                    "target_fields": ["answer", "target"]
                }
            },
            "validation": {
                "# 数据验证配置": "控制数据质量检查和处理",
                "min_valid_samples_ratio": 0.1,  # 最小有效样本比例
                "skip_empty_batches": True,      # 是否跳过空批次
                "enable_data_cleaning": True,    # 是否启用数据清洗
                "enable_fallback": True          # 是否启用降级处理
            },
            "diagnostics": {
                "# 诊断配置": "控制日志和诊断信息的输出",
                "enable_detailed_logging": False,  # 是否启用详细日志
                "log_batch_statistics": True,      # 是否记录批次统计
                "save_processing_report": True     # 是否保存处理报告
            }
        }
    
    def merge_with_default(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将用户配置与默认配置合并
        
        Args:
            user_config: 用户配置
            
        Returns:
            合并后的配置
        """
        default_config = self.get_default_config()
        merged_config = deepcopy(default_config)
        
        # 递归合并配置
        self._deep_merge(merged_config, user_config)
        
        return merged_config
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """递归合并字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_and_fix_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        验证并修复配置
        
        Args:
            config: 原始配置
            
        Returns:
            (修复后的配置, 修复说明列表)
        """
        fixes_applied = []
        fixed_config = deepcopy(config)
        
        # 确保基本结构存在
        if "field_mapping" not in fixed_config:
            fixed_config["field_mapping"] = self.get_default_config()["field_mapping"]
            fixes_applied.append("添加了默认的字段映射配置")
        
        if "validation" not in fixed_config:
            fixed_config["validation"] = self.get_default_config()["validation"]
            fixes_applied.append("添加了默认的验证配置")
        
        if "diagnostics" not in fixed_config:
            fixed_config["diagnostics"] = self.get_default_config()["diagnostics"]
            fixes_applied.append("添加了默认的诊断配置")
        
        # 修复validation配置
        validation_config = fixed_config["validation"]
        if "min_valid_samples_ratio" not in validation_config:
            validation_config["min_valid_samples_ratio"] = 0.1
            fixes_applied.append("设置了默认的最小有效样本比例")
        elif not isinstance(validation_config["min_valid_samples_ratio"], (int, float)):
            validation_config["min_valid_samples_ratio"] = 0.1
            fixes_applied.append("修复了无效的最小有效样本比例")
        elif not 0 <= validation_config["min_valid_samples_ratio"] <= 1:
            validation_config["min_valid_samples_ratio"] = max(0, min(1, validation_config["min_valid_samples_ratio"]))
            fixes_applied.append("调整了最小有效样本比例到有效范围")
        
        # 修复布尔值配置
        bool_configs = {
            "validation": ["skip_empty_batches", "enable_data_cleaning", "enable_fallback"],
            "diagnostics": ["enable_detailed_logging", "log_batch_statistics", "save_processing_report"]
        }
        
        for section, config_names in bool_configs.items():
            section_config = fixed_config[section]
            for config_name in config_names:
                if config_name not in section_config:
                    section_config[config_name] = self.get_default_config()[section][config_name]
                    fixes_applied.append(f"添加了默认的{config_name}配置")
                elif not isinstance(section_config[config_name], bool):
                    section_config[config_name] = bool(section_config[config_name])
                    fixes_applied.append(f"修复了{config_name}的数据类型")
        
        return fixed_config, fixes_applied
    
    def get_validation_errors(self) -> List[str]:
        """获取验证错误列表"""
        return self.validation_errors.copy()
    
    def get_validation_warnings(self) -> List[str]:
        """获取验证警告列表"""
        return self.validation_warnings.copy()
    
    def suggest_improvements(self, config: Dict[str, Any]) -> List[str]:
        """
        建议配置改进
        
        Args:
            config: 当前配置
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        # 检查字段映射的完整性
        if "field_mapping" in config:
            field_mapping = config["field_mapping"]
            
            # 检查是否有足够的任务类型
            common_tasks = ["text_generation", "question_answering", "classification"]
            missing_tasks = [task for task in common_tasks if task not in field_mapping]
            if missing_tasks:
                suggestions.append(f"考虑添加以下任务的字段映射: {', '.join(missing_tasks)}")
            
            # 检查字段列表的丰富性
            for task_name, task_config in field_mapping.items():
                if isinstance(task_config, dict):
                    for field_type, field_list in task_config.items():
                        if isinstance(field_list, list) and len(field_list) < 2:
                            suggestions.append(f"任务 '{task_name}' 的 '{field_type}' 字段候选较少，考虑添加更多选项")
        
        # 检查验证配置的合理性
        if "validation" in config:
            validation_config = config["validation"]
            
            ratio = validation_config.get("min_valid_samples_ratio", 0.1)
            if ratio < 0.05:
                suggestions.append("最小有效样本比例过低，可能导致质量问题")
            elif ratio > 0.5:
                suggestions.append("最小有效样本比例过高，可能导致过多数据被拒绝")
            
            if not validation_config.get("enable_data_cleaning", True):
                suggestions.append("建议启用数据清洗以提高数据质量")
            
            if not validation_config.get("enable_fallback", True):
                suggestions.append("建议启用降级处理以提高数据处理的鲁棒性")
        
        # 检查诊断配置
        if "diagnostics" in config:
            diagnostics_config = config["diagnostics"]
            
            if not diagnostics_config.get("log_batch_statistics", True):
                suggestions.append("建议启用批次统计日志以便监控数据处理状态")
            
            if not diagnostics_config.get("save_processing_report", True):
                suggestions.append("建议启用处理报告保存以便问题诊断")
        
        return suggestions
    
    def create_config_documentation(self) -> str:
        """
        创建配置文档
        
        Returns:
            配置文档字符串
        """
        doc = """
# 数据处理配置文档

## 配置结构

```yaml
data_processing:
  field_mapping:      # 字段映射配置
    <task_name>:      # 任务名称
      input_fields:   # 输入字段候选列表
      target_fields:  # 目标字段候选列表
      
  validation:         # 数据验证配置
    min_valid_samples_ratio: 0.1    # 最小有效样本比例 (0-1)
    skip_empty_batches: true        # 是否跳过空批次
    enable_data_cleaning: true      # 是否启用数据清洗
    enable_fallback: true           # 是否启用降级处理
    
  diagnostics:        # 诊断配置
    enable_detailed_logging: false  # 是否启用详细日志
    log_batch_statistics: true      # 是否记录批次统计
    save_processing_report: true    # 是否保存处理报告
```

## 配置说明

### field_mapping
定义不同任务类型的字段映射规则。系统会按照优先级顺序尝试这些字段名称。

### validation
控制数据验证和处理行为：
- `min_valid_samples_ratio`: 批次中有效样本的最小比例，低于此值的批次会被标记为无效
- `skip_empty_batches`: 是否跳过完全为空的批次
- `enable_data_cleaning`: 是否对数据进行清洗（去除多余空白、过滤过短文本等）
- `enable_fallback`: 是否启用降级处理（当主要字段不可用时尝试其他字段）

### diagnostics
控制诊断信息的输出：
- `enable_detailed_logging`: 启用详细的调试日志（可能影响性能）
- `log_batch_statistics`: 记录每个批次的处理统计信息
- `save_processing_report`: 保存详细的数据处理报告

## 使用示例

```python
from src.evaluation.data_models import EvaluationConfig

config = EvaluationConfig(
    data_processing={
        "field_mapping": {
            "text_generation": {
                "input_fields": ["text", "prompt", "input"],
                "target_fields": ["target", "answer"]
            }
        },
        "validation": {
            "min_valid_samples_ratio": 0.2,
            "enable_data_cleaning": True
        }
    }
)
```
"""
        return doc.strip()