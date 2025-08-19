"""
配置加载器

提供配置文件加载、验证和合并功能。
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .config_validator import ConfigValidator
from .data_models import EvaluationConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    配置加载器
    
    负责从文件加载配置、验证配置有效性并与默认配置合并。
    """
    
    def __init__(self):
        """初始化配置加载器"""
        self.validator = ConfigValidator()
    
    def load_config_from_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            logger.info(f"成功加载配置文件: {config_path}")
            return config_data or {}
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def create_evaluation_config(self, 
                                config_data: Optional[Dict[str, Any]] = None,
                                config_file: Optional[Union[str, Path]] = None,
                                validate: bool = True,
                                auto_fix: bool = True) -> EvaluationConfig:
        """
        创建评估配置对象
        
        Args:
            config_data: 配置数据字典
            config_file: 配置文件路径
            validate: 是否验证配置
            auto_fix: 是否自动修复配置问题
            
        Returns:
            评估配置对象
        """
        # 加载配置数据
        if config_file:
            file_config = self.load_config_from_file(config_file)
            if config_data:
                # 合并配置
                merged_config = self._merge_configs(file_config, config_data)
            else:
                merged_config = file_config
        elif config_data:
            merged_config = config_data
        else:
            # 使用默认配置
            merged_config = {}
        
        # 确保有data_processing配置
        if "data_processing" not in merged_config:
            merged_config["data_processing"] = {}
        
        # 验证和修复配置
        if validate:
            data_processing_config = merged_config["data_processing"]
            
            if auto_fix:
                fixed_config, fixes = self.validator.validate_and_fix_config(data_processing_config)
                merged_config["data_processing"] = fixed_config
                
                if fixes:
                    logger.info(f"自动修复了配置问题: {'; '.join(fixes)}")
            else:
                is_valid = self.validator.validate_processing_config(data_processing_config)
                if not is_valid:
                    errors = self.validator.get_validation_errors()
                    logger.error(f"配置验证失败: {'; '.join(errors)}")
                    raise ValueError(f"配置验证失败: {'; '.join(errors)}")
                
                warnings = self.validator.get_validation_warnings()
                if warnings:
                    logger.warning(f"配置警告: {'; '.join(warnings)}")
        
        # 与默认配置合并
        final_config = self.validator.merge_with_default(merged_config.get("data_processing", {}))
        merged_config["data_processing"] = final_config
        
        # 创建EvaluationConfig对象
        try:
            evaluation_config = EvaluationConfig(**merged_config)
            logger.info("成功创建评估配置对象")
            return evaluation_config
        except Exception as e:
            logger.error(f"创建评估配置对象失败: {e}")
            raise
    
    def save_config_to_file(self, 
                           config: Union[EvaluationConfig, Dict[str, Any]], 
                           output_path: Union[str, Path],
                           format: str = "yaml") -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置对象或字典
            output_path: 输出文件路径
            format: 输出格式 ("yaml" 或 "json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        if isinstance(config, EvaluationConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                elif format.lower() == "json":
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的输出格式: {format}")
            
            logger.info(f"配置已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def validate_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        验证配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            验证结果字典
        """
        try:
            config_data = self.load_config_from_file(config_path)
            data_processing_config = config_data.get("data_processing", {})
            
            is_valid = self.validator.validate_processing_config(data_processing_config)
            errors = self.validator.get_validation_errors()
            warnings = self.validator.get_validation_warnings()
            suggestions = self.validator.suggest_improvements(data_processing_config)
            
            result = {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "config_path": str(config_path)
            }
            
            if is_valid:
                logger.info(f"配置文件验证通过: {config_path}")
            else:
                logger.error(f"配置文件验证失败: {config_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"验证配置文件时出错: {e}")
            return {
                "is_valid": False,
                "errors": [str(e)],
                "warnings": [],
                "suggestions": [],
                "config_path": str(config_path)
            }
    
    def create_config_template(self, output_path: Union[str, Path]) -> None:
        """
        创建配置模板文件
        
        Args:
            output_path: 输出文件路径
        """
        template = self.validator.get_config_template()
        
        # 移除注释键（以#开头的键）
        clean_template = self._remove_comment_keys(template)
        
        self.save_config_to_file({"data_processing": clean_template}, output_path)
        logger.info(f"配置模板已创建: {output_path}")
    
    def _merge_configs(self, base_config: Dict[str, Any], update_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并两个配置字典"""
        merged = base_config.copy()
        
        for key, value in update_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _remove_comment_keys(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """移除配置中的注释键"""
        clean_config = {}
        
        for key, value in config.items():
            if not key.startswith("#"):
                if isinstance(value, dict):
                    clean_config[key] = self._remove_comment_keys(value)
                else:
                    clean_config[key] = value
        
        return clean_config
    
    def get_config_documentation(self) -> str:
        """获取配置文档"""
        return self.validator.create_config_documentation()


# 便利函数
def load_evaluation_config(config_file: Optional[Union[str, Path]] = None,
                          config_data: Optional[Dict[str, Any]] = None,
                          validate: bool = True,
                          auto_fix: bool = True) -> EvaluationConfig:
    """
    便利函数：加载评估配置
    
    Args:
        config_file: 配置文件路径
        config_data: 配置数据字典
        validate: 是否验证配置
        auto_fix: 是否自动修复配置问题
        
    Returns:
        评估配置对象
    """
    loader = ConfigLoader()
    return loader.create_evaluation_config(
        config_data=config_data,
        config_file=config_file,
        validate=validate,
        auto_fix=auto_fix
    )


def validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    便利函数：验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        验证结果字典
    """
    loader = ConfigLoader()
    return loader.validate_config_file(config_path)