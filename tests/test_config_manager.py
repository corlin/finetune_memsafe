"""
测试配置管理系统
"""

import pytest
import tempfile
import json
import yaml
import time
from pathlib import Path
from unittest.mock import Mock, patch
from industry_evaluation.config.config_manager import (
    ConfigManager,
    ConfigValidator,
    ConfigTemplate,
    EnvironmentConfigLoader,
    ModelConfig,
    EvaluatorConfig,
    SystemConfig,
    EvaluationSystemConfig
)


class TestConfigValidator:
    """测试配置验证器"""
    
    def test_validate_model_config_success(self):
        """测试模型配置验证成功"""
        config = ModelConfig(
            model_id="test_model",
            adapter_type="openai",
            api_key="test_key",
            timeout=30,
            max_retries=3
        )
        
        errors = ConfigValidator.validate_model_config(config)
        assert len(errors) == 0
    
    def test_validate_model_config_missing_fields(self):
        """测试模型配置缺少必需字段"""
        config = ModelConfig(
            model_id="",
            adapter_type="",
            timeout=0,
            max_retries=-1
        )
        
        errors = ConfigValidator.validate_model_config(config)
        assert len(errors) > 0
        assert any("model_id不能为空" in error for error in errors)
        assert any("adapter_type不能为空" in error for error in errors)
        assert any("timeout必须大于0" in error for error in errors)
        assert any("max_retries不能小于0" in error for error in errors)
    
    def test_validate_openai_config_missing_api_key(self):
        """测试OpenAI配置缺少API密钥"""
        config = ModelConfig(
            model_id="test_model",
            adapter_type="openai"
        )
        
        errors = ConfigValidator.validate_model_config(config)
        assert any("OpenAI适配器需要api_key" in error for error in errors)
    
    def test_validate_http_config_missing_url(self):
        """测试HTTP配置缺少URL"""
        config = ModelConfig(
            model_id="test_model",
            adapter_type="http"
        )
        
        errors = ConfigValidator.validate_model_config(config)
        assert any("HTTP适配器需要api_url" in error for error in errors)
    
    def test_validate_evaluator_config_success(self):
        """测试评估器配置验证成功"""
        config = EvaluatorConfig(
            evaluator_type="knowledge",
            weight=0.5,
            threshold=0.7
        )
        
        errors = ConfigValidator.validate_evaluator_config(config)
        assert len(errors) == 0
    
    def test_validate_evaluator_config_invalid_values(self):
        """测试评估器配置无效值"""
        config = EvaluatorConfig(
            evaluator_type="",
            weight=-0.1,
            threshold=1.5
        )
        
        errors = ConfigValidator.validate_evaluator_config(config)
        assert len(errors) > 0
        assert any("evaluator_type不能为空" in error for error in errors)
        assert any("weight不能小于0" in error for error in errors)
        assert any("threshold必须在0-1之间" in error for error in errors)
    
    def test_validate_system_config_success(self):
        """测试系统配置验证成功"""
        config = SystemConfig(
            max_workers=4,
            log_level="INFO",
            cache_ttl=3600,
            metrics_port=8080
        )
        
        errors = ConfigValidator.validate_system_config(config)
        assert len(errors) == 0
    
    def test_validate_system_config_invalid_values(self):
        """测试系统配置无效值"""
        config = SystemConfig(
            max_workers=0,
            log_level="INVALID",
            cache_ttl=-1,
            metrics_port=70000
        )
        
        errors = ConfigValidator.validate_system_config(config)
        assert len(errors) > 0
        assert any("max_workers必须大于0" in error for error in errors)
        assert any("log_level必须是有效的日志级别" in error for error in errors)
        assert any("cache_ttl必须大于0" in error for error in errors)
        assert any("metrics_port必须是有效的端口号" in error for error in errors)
    
    def test_validate_full_config_weight_sum(self):
        """测试完整配置权重总和验证"""
        config = EvaluationSystemConfig(
            default_weights={
                "knowledge": 0.6,
                "terminology": 0.5  # 总和为1.1，超过1.0
            }
        )
        
        errors = ConfigValidator.validate_full_config(config)
        assert any("默认权重总和应该为1.0" in error for error in errors)


class TestConfigManager:
    """测试配置管理器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_default_config(self):
        """测试创建默认配置"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        assert self.config_file.exists()
        config = config_manager.get_config()
        
        assert config.version == "1.0.0"
        assert config.system.max_workers == 4
        assert len(config.models) > 0
        assert len(config.evaluators) > 0
    
    def test_load_yaml_config(self):
        """测试加载YAML配置"""
        test_config = {
            "version": "1.0.0",
            "system": {
                "max_workers": 8,
                "log_level": "DEBUG"
            },
            "models": {
                "test_model": {
                    "model_id": "test_model",
                    "adapter_type": "local",
                    "timeout": 60
                }
            },
            "evaluators": {
                "knowledge": {
                    "evaluator_type": "knowledge",
                    "weight": 1.0,
                    "threshold": 0.8
                }
            },
            "industry_domains": ["test"],
            "default_weights": {"knowledge": 1.0},
            "default_thresholds": {"knowledge": 0.8}
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)
        
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        config = config_manager.get_config()
        
        assert config.version == "1.0.0"
        assert config.system.max_workers == 8
        assert config.system.log_level == "DEBUG"
        assert "test_model" in config.models
        assert config.models["test_model"].timeout == 60
    
    def test_load_json_config(self):
        """测试加载JSON配置"""
        json_config_file = Path(self.temp_dir) / "test_config.json"
        
        test_config = {
            "version": "1.0.0",
            "system": {
                "max_workers": 6,
                "log_level": "WARNING"
            },
            "models": {},
            "evaluators": {},
            "industry_domains": [],
            "default_weights": {},
            "default_thresholds": {}
        }
        
        with open(json_config_file, 'w', encoding='utf-8') as f:
            json.dump(test_config, f)
        
        config_manager = ConfigManager(json_config_file, auto_reload=False)
        config = config_manager.get_config()
        
        assert config.version == "1.0.0"
        assert config.system.max_workers == 6
        assert config.system.log_level == "WARNING"
    
    def test_save_config(self):
        """测试保存配置"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        # 修改配置
        config = config_manager.get_config()
        config.system.max_workers = 16
        
        # 保存配置
        success = config_manager.save_config()
        assert success == True
        
        # 重新加载验证
        new_config_manager = ConfigManager(self.config_file, auto_reload=False)
        new_config = new_config_manager.get_config()
        
        assert new_config.system.max_workers == 16
    
    def test_update_config(self):
        """测试更新配置"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        updates = {
            "system": {
                "max_workers": 12,
                "log_level": "ERROR"
            }
        }
        
        success = config_manager.update_config(updates)
        assert success == True
        
        config = config_manager.get_config()
        assert config.system.max_workers == 12
        assert config.system.log_level == "ERROR"
    
    def test_add_model(self):
        """测试添加模型配置"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        model_config = ModelConfig(
            model_id="new_model",
            adapter_type="openai",
            api_key="test_key"
        )
        
        success = config_manager.add_model("new_model", model_config)
        assert success == True
        
        config = config_manager.get_config()
        assert "new_model" in config.models
        assert config.models["new_model"].adapter_type == "openai"
    
    def test_remove_model(self):
        """测试移除模型配置"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        # 先添加一个模型
        model_config = ModelConfig(
            model_id="temp_model",
            adapter_type="local"
        )
        config_manager.add_model("temp_model", model_config)
        
        # 验证模型存在
        config = config_manager.get_config()
        assert "temp_model" in config.models
        
        # 移除模型
        success = config_manager.remove_model("temp_model")
        assert success == True
        
        # 验证模型已移除
        config = config_manager.get_config()
        assert "temp_model" not in config.models
    
    def test_add_evaluator(self):
        """测试添加评估器配置"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        evaluator_config = EvaluatorConfig(
            evaluator_type="new_evaluator",
            weight=0.5,
            threshold=0.6
        )
        
        success = config_manager.add_evaluator("new_evaluator", evaluator_config)
        assert success == True
        
        config = config_manager.get_config()
        assert "new_evaluator" in config.evaluators
        assert config.evaluators["new_evaluator"].weight == 0.5
    
    def test_remove_evaluator(self):
        """测试移除评估器配置"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        # 先添加一个评估器
        evaluator_config = EvaluatorConfig(
            evaluator_type="temp_evaluator",
            weight=0.3,
            threshold=0.5
        )
        config_manager.add_evaluator("temp_evaluator", evaluator_config)
        
        # 验证评估器存在
        config = config_manager.get_config()
        assert "temp_evaluator" in config.evaluators
        
        # 移除评估器
        success = config_manager.remove_evaluator("temp_evaluator")
        assert success == True
        
        # 验证评估器已移除
        config = config_manager.get_config()
        assert "temp_evaluator" not in config.evaluators
    
    def test_config_validation_error(self):
        """测试配置验证错误"""
        # 创建无效配置文件
        invalid_config = {
            "version": "1.0.0",
            "system": {
                "max_workers": -1,  # 无效值
                "log_level": "INVALID"  # 无效值
            },
            "models": {},
            "evaluators": {},
            "industry_domains": [],
            "default_weights": {},
            "default_thresholds": {}
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(invalid_config, f)
        
        # 应该抛出验证错误
        with pytest.raises(ValueError) as exc_info:
            ConfigManager(self.config_file, auto_reload=False)
        
        assert "配置验证失败" in str(exc_info.value)
    
    def test_reload_callback(self):
        """测试配置重新加载回调"""
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        callback_called = False
        old_config_received = None
        new_config_received = None
        
        def test_callback(old_config, new_config):
            nonlocal callback_called, old_config_received, new_config_received
            callback_called = True
            old_config_received = old_config
            new_config_received = new_config
        
        config_manager.register_reload_callback(test_callback)
        
        # 修改配置文件
        config = config_manager.get_config()
        config.system.max_workers = 20
        config_manager.save_config()
        
        # 重新加载配置
        config_manager.reload_config()
        
        # 验证回调被调用
        assert callback_called == True
        assert old_config_received is not None
        assert new_config_received is not None
        assert new_config_received.system.max_workers == 20


class TestConfigTemplate:
    """测试配置模板"""
    
    def test_generate_finance_config(self):
        """测试生成金融配置模板"""
        config = ConfigTemplate.generate_finance_config()
        
        assert config.system.max_workers == 8
        assert "finance_gpt4" in config.models
        assert "finance_local" in config.models
        assert config.models["finance_gpt4"].adapter_type == "openai"
        assert config.evaluators["knowledge"].weight == 0.5
        assert "finance" in config.industry_domains
    
    def test_generate_healthcare_config(self):
        """测试生成医疗配置模板"""
        config = ConfigTemplate.generate_healthcare_config()
        
        assert config.system.max_workers == 6
        assert "medical_gpt" in config.models
        assert config.evaluators["knowledge"].weight == 0.6
        assert config.evaluators["knowledge"].threshold == 0.85
        assert "healthcare" in config.industry_domains
    
    def test_save_template(self):
        """测试保存配置模板"""
        temp_dir = tempfile.mkdtemp()
        try:
            config = ConfigTemplate.generate_finance_config()
            template_file = Path(temp_dir) / "finance_template.yaml"
            
            ConfigTemplate.save_template(config, template_file)
            
            assert template_file.exists()
            
            # 验证保存的内容
            with open(template_file, 'r', encoding='utf-8') as f:
                saved_data = yaml.safe_load(f)
            
            assert saved_data["system"]["max_workers"] == 8
            assert "finance_gpt4" in saved_data["models"]
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestEnvironmentConfigLoader:
    """测试环境变量配置加载器"""
    
    @patch.dict('os.environ', {
        'EVAL_MAX_WORKERS': '16',
        'EVAL_LOG_LEVEL': 'DEBUG',
        'OPENAI_API_KEY': 'test_key_123',
        'OPENAI_MODEL_NAME': 'gpt-4'
    })
    def test_load_from_env(self):
        """测试从环境变量加载配置"""
        config = EnvironmentConfigLoader.load_from_env()
        
        assert config["system"]["max_workers"] == 16
        assert config["system"]["log_level"] == "DEBUG"
        assert "openai_default" in config["models"]
        assert config["models"]["openai_default"]["api_key"] == "test_key_123"
        assert config["models"]["openai_default"]["model_name"] == "gpt-4"
    
    @patch.dict('os.environ', {}, clear=True)
    def test_load_from_env_empty(self):
        """测试从空环境变量加载配置"""
        config = EnvironmentConfigLoader.load_from_env()
        
        assert config == {}
    
    @patch.dict('os.environ', {
        'EVAL_MAX_WORKERS': '8',
        'OPENAI_API_KEY': 'env_key'
    })
    def test_apply_env_overrides(self):
        """测试应用环境变量覆盖"""
        temp_dir = tempfile.mkdtemp()
        try:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_manager = ConfigManager(config_file, auto_reload=False)
            
            # 应用环境变量覆盖
            EnvironmentConfigLoader.apply_env_overrides(config_manager)
            
            config = config_manager.get_config()
            
            # 验证环境变量覆盖生效
            assert config.system.max_workers == 8
            assert "openai_default" in config.models
            assert config.models["openai_default"].api_key == "env_key"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])