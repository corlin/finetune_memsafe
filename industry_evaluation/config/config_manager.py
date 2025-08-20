"""
配置管理系统
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    adapter_type: str
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1
    fallback_enabled: bool = False
    fallback_response: str = "模型暂时不可用"
    retry_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorConfig:
    """评估器配置"""
    evaluator_type: str
    enabled: bool = True
    weight: float = 1.0
    threshold: float = 0.5
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """系统配置"""
    max_workers: int = 4
    log_level: str = "INFO"
    log_file: Optional[str] = None
    temp_dir: str = "/tmp/industry_evaluation"
    cache_enabled: bool = True
    cache_ttl: int = 3600
    monitoring_enabled: bool = True
    metrics_port: int = 8080


@dataclass
class EvaluationSystemConfig:
    """评估系统完整配置"""
    version: str = "1.0.0"
    system: SystemConfig = field(default_factory=SystemConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    evaluators: Dict[str, EvaluatorConfig] = field(default_factory=dict)
    industry_domains: List[str] = field(default_factory=list)
    default_weights: Dict[str, float] = field(default_factory=dict)
    default_thresholds: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """
        验证模型配置
        
        Args:
            config: 模型配置
            
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        if not config.model_id:
            errors.append("model_id不能为空")
        
        if not config.adapter_type:
            errors.append("adapter_type不能为空")
        
        if config.adapter_type == "openai" and not config.api_key:
            errors.append("OpenAI适配器需要api_key")
        
        if config.adapter_type == "http" and not config.api_url:
            errors.append("HTTP适配器需要api_url")
        
        if config.timeout <= 0:
            errors.append("timeout必须大于0")
        
        if config.max_retries < 0:
            errors.append("max_retries不能小于0")
        
        if config.rate_limit_delay < 0:
            errors.append("rate_limit_delay不能小于0")
        
        return errors
    
    @staticmethod
    def validate_evaluator_config(config: EvaluatorConfig) -> List[str]:
        """
        验证评估器配置
        
        Args:
            config: 评估器配置
            
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        if not config.evaluator_type:
            errors.append("evaluator_type不能为空")
        
        if config.weight < 0:
            errors.append("weight不能小于0")
        
        if config.threshold < 0 or config.threshold > 1:
            errors.append("threshold必须在0-1之间")
        
        return errors
    
    @staticmethod
    def validate_system_config(config: SystemConfig) -> List[str]:
        """
        验证系统配置
        
        Args:
            config: 系统配置
            
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        if config.max_workers <= 0:
            errors.append("max_workers必须大于0")
        
        if config.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("log_level必须是有效的日志级别")
        
        if config.cache_ttl <= 0:
            errors.append("cache_ttl必须大于0")
        
        if config.metrics_port <= 0 or config.metrics_port > 65535:
            errors.append("metrics_port必须是有效的端口号")
        
        return errors
    
    @staticmethod
    def validate_full_config(config: EvaluationSystemConfig) -> List[str]:
        """
        验证完整配置
        
        Args:
            config: 完整配置
            
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        # 验证系统配置
        errors.extend(ConfigValidator.validate_system_config(config.system))
        
        # 验证模型配置
        for model_id, model_config in config.models.items():
            model_errors = ConfigValidator.validate_model_config(model_config)
            errors.extend([f"模型 {model_id}: {error}" for error in model_errors])
        
        # 验证评估器配置
        for evaluator_id, evaluator_config in config.evaluators.items():
            evaluator_errors = ConfigValidator.validate_evaluator_config(evaluator_config)
            errors.extend([f"评估器 {evaluator_id}: {error}" for error in evaluator_errors])
        
        # 验证权重总和
        if config.default_weights:
            total_weight = sum(config.default_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"默认权重总和应该为1.0，当前为{total_weight}")
        
        return errors


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    
    def __init__(self, config_manager):
        """
        初始化文件处理器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if not event.is_directory and event.src_path == str(self.config_manager.config_file):
            self.logger.info(f"配置文件已修改: {event.src_path}")
            try:
                self.config_manager.reload_config()
            except Exception as e:
                self.logger.error(f"重新加载配置失败: {str(e)}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Union[str, Path], auto_reload: bool = True):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
            auto_reload: 是否自动重新加载配置
        """
        self.config_file = Path(config_file)
        self.auto_reload = auto_reload
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 配置数据
        self._config: Optional[EvaluationSystemConfig] = None
        self._config_lock = threading.RLock()
        
        # 文件监控
        self._observer: Optional[Observer] = None
        self._file_handler: Optional[ConfigFileHandler] = None
        
        # 回调函数
        self._reload_callbacks: List[callable] = []
        
        # 加载初始配置
        self.load_config()
        
        # 启动文件监控
        if self.auto_reload:
            self.start_file_monitoring()
    
    def load_config(self) -> EvaluationSystemConfig:
        """
        加载配置文件
        
        Returns:
            EvaluationSystemConfig: 配置对象
        """
        with self._config_lock:
            try:
                if not self.config_file.exists():
                    self.logger.warning(f"配置文件不存在，创建默认配置: {self.config_file}")
                    self._config = self._create_default_config()
                    self.save_config()
                    return self._config
                
                # 根据文件扩展名选择解析器
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                elif self.config_file.suffix.lower() == '.json':
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {self.config_file.suffix}")
                
                # 转换为配置对象
                self._config = self._dict_to_config(config_data)
                
                # 验证配置
                errors = ConfigValidator.validate_full_config(self._config)
                if errors:
                    error_msg = "配置验证失败:\n" + "\n".join(f"  - {error}" for error in errors)
                    raise ValueError(error_msg)
                
                self.logger.info(f"配置加载成功: {self.config_file}")
                return self._config
                
            except Exception as e:
                self.logger.error(f"加载配置失败: {str(e)}")
                if self._config is None:
                    # 如果没有有效配置，创建默认配置
                    self._config = self._create_default_config()
                raise
    
    def save_config(self) -> bool:
        """
        保存配置到文件
        
        Returns:
            bool: 保存是否成功
        """
        with self._config_lock:
            try:
                if self._config is None:
                    raise ValueError("没有配置数据可保存")
                
                # 更新时间戳
                self._config.updated_at = datetime.now().isoformat()
                
                # 转换为字典
                config_data = asdict(self._config)
                
                # 确保目录存在
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 根据文件扩展名选择格式
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
                elif self.config_file.suffix.lower() == '.json':
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, ensure_ascii=False, indent=2)
                else:
                    raise ValueError(f"不支持的配置文件格式: {self.config_file.suffix}")
                
                self.logger.info(f"配置保存成功: {self.config_file}")
                return True
                
            except Exception as e:
                self.logger.error(f"保存配置失败: {str(e)}")
                return False
    
    def reload_config(self) -> bool:
        """
        重新加载配置
        
        Returns:
            bool: 重新加载是否成功
        """
        try:
            old_config = self._config
            self.load_config()
            
            # 触发回调
            for callback in self._reload_callbacks:
                try:
                    callback(old_config, self._config)
                except Exception as e:
                    self.logger.error(f"配置重新加载回调失败: {str(e)}")
            
            return True
        except Exception as e:
            self.logger.error(f"重新加载配置失败: {str(e)}")
            return False
    
    def get_config(self) -> EvaluationSystemConfig:
        """
        获取当前配置
        
        Returns:
            EvaluationSystemConfig: 当前配置
        """
        with self._config_lock:
            if self._config is None:
                self.load_config()
            return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        更新配置
        
        Args:
            updates: 更新的配置项
            
        Returns:
            bool: 更新是否成功
        """
        with self._config_lock:
            try:
                if self._config is None:
                    self.load_config()
                
                # 深度更新配置
                self._deep_update_config(self._config, updates)
                
                # 验证更新后的配置
                errors = ConfigValidator.validate_full_config(self._config)
                if errors:
                    raise ValueError("配置更新后验证失败:\n" + "\n".join(f"  - {error}" for error in errors))
                
                # 保存配置
                return self.save_config()
                
            except Exception as e:
                self.logger.error(f"更新配置失败: {str(e)}")
                return False
    
    def add_model(self, model_id: str, model_config: ModelConfig) -> bool:
        """
        添加模型配置
        
        Args:
            model_id: 模型ID
            model_config: 模型配置
            
        Returns:
            bool: 添加是否成功
        """
        with self._config_lock:
            try:
                if self._config is None:
                    self.load_config()
                
                # 验证模型配置
                errors = ConfigValidator.validate_model_config(model_config)
                if errors:
                    raise ValueError("模型配置验证失败:\n" + "\n".join(f"  - {error}" for error in errors))
                
                self._config.models[model_id] = model_config
                return self.save_config()
                
            except Exception as e:
                self.logger.error(f"添加模型配置失败: {str(e)}")
                return False
    
    def remove_model(self, model_id: str) -> bool:
        """
        移除模型配置
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 移除是否成功
        """
        with self._config_lock:
            try:
                if self._config is None:
                    self.load_config()
                
                if model_id in self._config.models:
                    del self._config.models[model_id]
                    return self.save_config()
                else:
                    self.logger.warning(f"模型配置不存在: {model_id}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"移除模型配置失败: {str(e)}")
                return False
    
    def add_evaluator(self, evaluator_id: str, evaluator_config: EvaluatorConfig) -> bool:
        """
        添加评估器配置
        
        Args:
            evaluator_id: 评估器ID
            evaluator_config: 评估器配置
            
        Returns:
            bool: 添加是否成功
        """
        with self._config_lock:
            try:
                if self._config is None:
                    self.load_config()
                
                # 验证评估器配置
                errors = ConfigValidator.validate_evaluator_config(evaluator_config)
                if errors:
                    raise ValueError("评估器配置验证失败:\n" + "\n".join(f"  - {error}" for error in errors))
                
                self._config.evaluators[evaluator_id] = evaluator_config
                return self.save_config()
                
            except Exception as e:
                self.logger.error(f"添加评估器配置失败: {str(e)}")
                return False
    
    def remove_evaluator(self, evaluator_id: str) -> bool:
        """
        移除评估器配置
        
        Args:
            evaluator_id: 评估器ID
            
        Returns:
            bool: 移除是否成功
        """
        with self._config_lock:
            try:
                if self._config is None:
                    self.load_config()
                
                if evaluator_id in self._config.evaluators:
                    del self._config.evaluators[evaluator_id]
                    return self.save_config()
                else:
                    self.logger.warning(f"评估器配置不存在: {evaluator_id}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"移除评估器配置失败: {str(e)}")
                return False
    
    def register_reload_callback(self, callback: callable):
        """
        注册配置重新加载回调函数
        
        Args:
            callback: 回调函数，接收 (old_config, new_config) 参数
        """
        self._reload_callbacks.append(callback)
    
    def start_file_monitoring(self):
        """启动文件监控"""
        if self._observer is not None:
            return
        
        try:
            self._file_handler = ConfigFileHandler(self)
            self._observer = Observer()
            self._observer.schedule(
                self._file_handler,
                str(self.config_file.parent),
                recursive=False
            )
            self._observer.start()
            self.logger.info("配置文件监控已启动")
        except Exception as e:
            self.logger.error(f"启动文件监控失败: {str(e)}")
    
    def stop_file_monitoring(self):
        """停止文件监控"""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._file_handler = None
            self.logger.info("配置文件监控已停止")
    
    def _create_default_config(self) -> EvaluationSystemConfig:
        """创建默认配置"""
        return EvaluationSystemConfig(
            system=SystemConfig(),
            models={
                "example_openai": ModelConfig(
                    model_id="example_openai",
                    adapter_type="openai",
                    api_key="your_api_key_here",
                    model_name="gpt-3.5-turbo",
                    timeout=30,
                    max_retries=3
                ),
                "example_local": ModelConfig(
                    model_id="example_local",
                    adapter_type="local",
                    model_name="local_model_path"
                )
            },
            evaluators={
                "knowledge": EvaluatorConfig(
                    evaluator_type="knowledge",
                    weight=0.4,
                    threshold=0.7
                ),
                "terminology": EvaluatorConfig(
                    evaluator_type="terminology",
                    weight=0.3,
                    threshold=0.6
                ),
                "reasoning": EvaluatorConfig(
                    evaluator_type="reasoning",
                    weight=0.3,
                    threshold=0.7
                )
            },
            industry_domains=["finance", "healthcare", "technology", "general"],
            default_weights={
                "knowledge": 0.4,
                "terminology": 0.3,
                "reasoning": 0.3
            },
            default_thresholds={
                "knowledge": 0.7,
                "terminology": 0.6,
                "reasoning": 0.7
            }
        )
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> EvaluationSystemConfig:
        """将字典转换为配置对象"""
        # 转换系统配置
        system_data = config_data.get("system", {})
        system_config = SystemConfig(**system_data)
        
        # 转换模型配置
        models = {}
        for model_id, model_data in config_data.get("models", {}).items():
            models[model_id] = ModelConfig(**model_data)
        
        # 转换评估器配置
        evaluators = {}
        for evaluator_id, evaluator_data in config_data.get("evaluators", {}).items():
            evaluators[evaluator_id] = EvaluatorConfig(**evaluator_data)
        
        return EvaluationSystemConfig(
            version=config_data.get("version", "1.0.0"),
            system=system_config,
            models=models,
            evaluators=evaluators,
            industry_domains=config_data.get("industry_domains", []),
            default_weights=config_data.get("default_weights", {}),
            default_thresholds=config_data.get("default_thresholds", {}),
            created_at=config_data.get("created_at", datetime.now().isoformat()),
            updated_at=config_data.get("updated_at", datetime.now().isoformat())
        )
    
    def _deep_update_config(self, config: EvaluationSystemConfig, updates: Dict[str, Any]):
        """深度更新配置对象"""
        for key, value in updates.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                    # 递归更新嵌套对象
                    self._deep_update_dict(getattr(config, key).__dict__, value)
                else:
                    setattr(config, key, value)
    
    def _deep_update_dict(self, target: Dict[str, Any], updates: Dict[str, Any]):
        """深度更新字典"""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update_dict(target[key], value)
            else:
                target[key] = value
    
    def __del__(self):
        """析构函数"""
        self.stop_file_monitoring()


class ConfigTemplate:
    """配置模板生成器"""
    
    @staticmethod
    def generate_finance_config() -> EvaluationSystemConfig:
        """生成金融行业配置模板"""
        return EvaluationSystemConfig(
            system=SystemConfig(
                max_workers=8,
                log_level="INFO"
            ),
            models={
                "finance_gpt4": ModelConfig(
                    model_id="finance_gpt4",
                    adapter_type="openai",
                    api_key="your_openai_api_key",
                    model_name="gpt-4",
                    timeout=60,
                    max_retries=3
                ),
                "finance_local": ModelConfig(
                    model_id="finance_local",
                    adapter_type="local",
                    model_name="/path/to/finance/model"
                )
            },
            evaluators={
                "knowledge": EvaluatorConfig(
                    evaluator_type="knowledge",
                    weight=0.5,
                    threshold=0.8,
                    parameters={"domain": "finance"}
                ),
                "terminology": EvaluatorConfig(
                    evaluator_type="terminology",
                    weight=0.3,
                    threshold=0.7,
                    parameters={"finance_terms": True}
                ),
                "reasoning": EvaluatorConfig(
                    evaluator_type="reasoning",
                    weight=0.2,
                    threshold=0.7,
                    parameters={"financial_logic": True}
                )
            },
            industry_domains=["finance"],
            default_weights={
                "knowledge": 0.5,
                "terminology": 0.3,
                "reasoning": 0.2
            },
            default_thresholds={
                "knowledge": 0.8,
                "terminology": 0.7,
                "reasoning": 0.7
            }
        )
    
    @staticmethod
    def generate_healthcare_config() -> EvaluationSystemConfig:
        """生成医疗行业配置模板"""
        return EvaluationSystemConfig(
            system=SystemConfig(
                max_workers=6,
                log_level="INFO"
            ),
            models={
                "medical_gpt": ModelConfig(
                    model_id="medical_gpt",
                    adapter_type="openai",
                    api_key="your_openai_api_key",
                    model_name="gpt-4",
                    timeout=90,
                    max_retries=3
                )
            },
            evaluators={
                "knowledge": EvaluatorConfig(
                    evaluator_type="knowledge",
                    weight=0.6,
                    threshold=0.85,
                    parameters={"domain": "healthcare"}
                ),
                "terminology": EvaluatorConfig(
                    evaluator_type="terminology",
                    weight=0.4,
                    threshold=0.8,
                    parameters={"medical_terms": True}
                )
            },
            industry_domains=["healthcare"],
            default_weights={
                "knowledge": 0.6,
                "terminology": 0.4
            },
            default_thresholds={
                "knowledge": 0.85,
                "terminology": 0.8
            }
        )
    
    @staticmethod
    def save_template(config: EvaluationSystemConfig, file_path: Union[str, Path]):
        """
        保存配置模板到文件
        
        Args:
            config: 配置对象
            file_path: 文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = asdict(config)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")


# 环境变量配置支持
class EnvironmentConfigLoader:
    """环境变量配置加载器"""
    
    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """
        从环境变量加载配置
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        config = {}
        
        # 系统配置
        if os.getenv("EVAL_MAX_WORKERS"):
            config.setdefault("system", {})["max_workers"] = int(os.getenv("EVAL_MAX_WORKERS"))
        
        if os.getenv("EVAL_LOG_LEVEL"):
            config.setdefault("system", {})["log_level"] = os.getenv("EVAL_LOG_LEVEL")
        
        if os.getenv("EVAL_LOG_FILE"):
            config.setdefault("system", {})["log_file"] = os.getenv("EVAL_LOG_FILE")
        
        # 模型配置
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            config.setdefault("models", {})["openai_default"] = {
                "model_id": "openai_default",
                "adapter_type": "openai",
                "api_key": openai_api_key,
                "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
            }
        
        return config
    
    @staticmethod
    def apply_env_overrides(config_manager: ConfigManager):
        """
        应用环境变量覆盖
        
        Args:
            config_manager: 配置管理器
        """
        env_config = EnvironmentConfigLoader.load_from_env()
        if env_config:
            config_manager.update_config(env_config)