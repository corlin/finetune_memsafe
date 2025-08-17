"""
配置管理器测试

测试ConfigManager类的功能。
"""

import pytest
import json
import yaml
from pathlib import Path

from evaluation.config_manager import ConfigManager
from evaluation.data_models import EvaluationConfig


class TestConfigManager:
    """配置管理器测试类"""
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        manager = ConfigManager()
        
        assert manager.config_dir == "configs"
        assert manager.default_config_file == "default_config.yaml"
        assert manager.validate_on_load == True
    
    def test_init_custom_params(self, temp_dir):
        """测试自定义参数初始化"""
        manager = ConfigManager(
            config_dir=str(temp_dir),
            default_config_file="custom_config.yaml",
            validate_on_load=False
        )
        
        assert manager.config_dir == str(temp_dir)
        assert manager.default_config_file == "custom_config.yaml"
        assert manager.validate_on_load == False
    
    def test_load_yaml_config(self, temp_dir):
        """测试加载YAML配置"""
        manager = ConfigManager()
        
        # 创建测试配置文件
        config_data = {
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "random_seed": 42
            },
            "evaluation": {
                "tasks": ["classification", "text_generation"],
                "metrics": ["accuracy", "bleu"],
                "batch_size": 8
            }
        }
        
        config_path = temp_dir / "test_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        loaded_config = manager.load_config(str(config_path))
        
        assert loaded_config["data_split"]["train_ratio"] == 0.7
        assert "classification" in loaded_config["evaluation"]["tasks"]
        assert loaded_config["evaluation"]["batch_size"] == 8
    
    def test_load_json_config(self, temp_dir):
        """测试加载JSON配置"""
        manager = ConfigManager()
        
        config_data = {
            "model": {
                "name": "test_model",
                "max_length": 512
            },
            "training": {
                "epochs": 10,
                "learning_rate": 0.001
            }
        }
        
        config_path = temp_dir / "test_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False)
        
        loaded_config = manager.load_config(str(config_path))
        
        assert loaded_config["model"]["name"] == "test_model"
        assert loaded_config["training"]["epochs"] == 10
    
    def test_save_yaml_config(self, temp_dir):
        """测试保存YAML配置"""
        manager = ConfigManager()
        
        config_data = {
            "test_section": {
                "param1": "value1",
                "param2": 42,
                "param3": [1, 2, 3]
            }
        }
        
        config_path = temp_dir / "saved_config.yaml"
        manager.save_config(config_data, str(config_path))
        
        # 验证文件是否创建
        assert config_path.exists()
        
        # 验证内容是否正确
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["test_section"]["param1"] == "value1"
        assert loaded_data["test_section"]["param2"] == 42
    
    def test_save_json_config(self, temp_dir):
        """测试保存JSON配置"""
        manager = ConfigManager()
        
        config_data = {
            "settings": {
                "debug": True,
                "timeout": 30.5,
                "items": ["a", "b", "c"]
            }
        }
        
        config_path = temp_dir / "saved_config.json"
        manager.save_config(config_data, str(config_path), format="json")
        
        assert config_path.exists()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["settings"]["debug"] == True
        assert loaded_data["settings"]["timeout"] == 30.5
    
    def test_validate_config_valid(self):
        """测试有效配置验证"""
        manager = ConfigManager()
        
        valid_config = {
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "evaluation": {
                "tasks": ["classification"],
                "metrics": ["accuracy"],
                "batch_size": 8
            }
        }
        
        # 应该不抛出异常
        is_valid = manager.validate_config(valid_config)
        assert is_valid == True
    
    def test_validate_config_invalid_ratios(self):
        """测试无效比例配置验证"""
        manager = ConfigManager()
        
        invalid_config = {
            "data_split": {
                "train_ratio": 0.5,
                "val_ratio": 0.3,
                "test_ratio": 0.3  # 总和 > 1.0
            }
        }
        
        is_valid = manager.validate_config(invalid_config)
        assert is_valid == False
    
    def test_validate_config_missing_required(self):
        """测试缺少必需字段的配置验证"""
        manager = ConfigManager()
        
        incomplete_config = {
            "evaluation": {
                "tasks": [],  # 空任务列表
                "batch_size": 0  # 无效批次大小
            }
        }
        
        is_valid = manager.validate_config(incomplete_config)
        assert is_valid == False
    
    def test_merge_configs(self):
        """测试配置合并"""
        manager = ConfigManager()
        
        base_config = {
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "evaluation": {
                "batch_size": 8,
                "metrics": ["accuracy"]
            }
        }
        
        override_config = {
            "evaluation": {
                "batch_size": 16,  # 覆盖
                "tasks": ["classification"]  # 新增
            },
            "new_section": {
                "param": "value"
            }
        }
        
        merged = manager.merge_configs(base_config, override_config)
        
        # 检查合并结果
        assert merged["data_split"]["train_ratio"] == 0.7  # 保持原值
        assert merged["evaluation"]["batch_size"] == 16  # 被覆盖
        assert merged["evaluation"]["metrics"] == ["accuracy"]  # 保持原值
        assert merged["evaluation"]["tasks"] == ["classification"]  # 新增
        assert merged["new_section"]["param"] == "value"  # 新增节
    
    def test_get_default_config(self):
        """测试获取默认配置"""
        manager = ConfigManager()
        
        default_config = manager.get_default_config()
        
        assert isinstance(default_config, dict)
        assert "data_split" in default_config
        assert "evaluation" in default_config
        
        # 检查默认值
        assert default_config["data_split"]["train_ratio"] == 0.7
        assert default_config["data_split"]["val_ratio"] == 0.15
        assert default_config["data_split"]["test_ratio"] == 0.15
    
    def test_create_evaluation_config(self):
        """测试创建评估配置对象"""
        manager = ConfigManager()
        
        config_dict = {
            "evaluation": {
                "tasks": ["classification", "text_generation"],
                "metrics": ["accuracy", "bleu", "rouge"],
                "batch_size": 16,
                "max_length": 256,
                "num_samples": 100
            }
        }
        
        eval_config = manager.create_evaluation_config(config_dict)
        
        assert isinstance(eval_config, EvaluationConfig)
        assert eval_config.tasks == ["classification", "text_generation"]
        assert eval_config.batch_size == 16
        assert eval_config.num_samples == 100
    
    def test_update_config_section(self, temp_dir):
        """测试更新配置节"""
        manager = ConfigManager()
        
        # 创建初始配置
        initial_config = {
            "section1": {"param1": "value1"},
            "section2": {"param2": "value2"}
        }
        
        config_path = temp_dir / "update_test.yaml"
        manager.save_config(initial_config, str(config_path))
        
        # 更新配置节
        new_section_data = {"param1": "new_value1", "param3": "value3"}
        manager.update_config_section(str(config_path), "section1", new_section_data)
        
        # 验证更新结果
        updated_config = manager.load_config(str(config_path))
        
        assert updated_config["section1"]["param1"] == "new_value1"
        assert updated_config["section1"]["param3"] == "value3"
        assert updated_config["section2"]["param2"] == "value2"  # 未改变
    
    def test_list_config_templates(self, temp_dir):
        """测试列出配置模板"""
        manager = ConfigManager(config_dir=str(temp_dir))
        
        # 创建一些模板文件
        templates = ["template1.yaml", "template2.json", "template3.yaml"]
        for template in templates:
            (temp_dir / template).write_text("test: config")
        
        # 创建非配置文件（应该被忽略）
        (temp_dir / "readme.txt").write_text("not a config")
        
        template_list = manager.list_config_templates()
        
        assert len(template_list) == 3
        assert "template1.yaml" in template_list
        assert "template2.json" in template_list
        assert "template3.yaml" in template_list
        assert "readme.txt" not in template_list
    
    def test_validate_config_schema(self):
        """测试配置模式验证"""
        manager = ConfigManager()
        
        # 定义配置模式
        schema = {
            "type": "object",
            "properties": {
                "data_split": {
                    "type": "object",
                    "properties": {
                        "train_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                        "val_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                        "test_ratio": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["train_ratio", "val_ratio", "test_ratio"]
                }
            },
            "required": ["data_split"]
        }
        
        # 有效配置
        valid_config = {
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            }
        }
        
        assert manager.validate_config_schema(valid_config, schema) == True
        
        # 无效配置
        invalid_config = {
            "data_split": {
                "train_ratio": 1.5,  # 超出范围
                "val_ratio": 0.15
                # 缺少 test_ratio
            }
        }
        
        assert manager.validate_config_schema(invalid_config, schema) == False
    
    def test_environment_variable_substitution(self, temp_dir):
        """测试环境变量替换"""
        import os
        
        manager = ConfigManager()
        
        # 设置环境变量
        os.environ["TEST_BATCH_SIZE"] = "32"
        os.environ["TEST_MODEL_NAME"] = "test_model"
        
        try:
            config_data = {
                "evaluation": {
                    "batch_size": "${TEST_BATCH_SIZE}",
                    "model_name": "${TEST_MODEL_NAME}",
                    "default_value": "${UNDEFINED_VAR:default}"
                }
            }
            
            config_path = temp_dir / "env_test.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            loaded_config = manager.load_config(str(config_path), substitute_env_vars=True)
            
            assert loaded_config["evaluation"]["batch_size"] == "32"
            assert loaded_config["evaluation"]["model_name"] == "test_model"
            assert loaded_config["evaluation"]["default_value"] == "default"
            
        finally:
            # 清理环境变量
            os.environ.pop("TEST_BATCH_SIZE", None)
            os.environ.pop("TEST_MODEL_NAME", None)
    
    def test_config_inheritance(self, temp_dir):
        """测试配置继承"""
        manager = ConfigManager()
        
        # 创建基础配置
        base_config = {
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "evaluation": {
                "batch_size": 8,
                "metrics": ["accuracy"]
            }
        }
        
        base_path = temp_dir / "base_config.yaml"
        manager.save_config(base_config, str(base_path))
        
        # 创建继承配置
        child_config = {
            "inherit_from": str(base_path),
            "evaluation": {
                "batch_size": 16,  # 覆盖
                "tasks": ["classification"]  # 新增
            }
        }
        
        child_path = temp_dir / "child_config.yaml"
        manager.save_config(child_config, str(child_path))
        
        # 加载继承配置
        loaded_config = manager.load_config(str(child_path), resolve_inheritance=True)
        
        # 检查继承结果
        assert loaded_config["data_split"]["train_ratio"] == 0.7  # 继承
        assert loaded_config["evaluation"]["batch_size"] == 16  # 覆盖
        assert loaded_config["evaluation"]["metrics"] == ["accuracy"]  # 继承
        assert loaded_config["evaluation"]["tasks"] == ["classification"]  # 新增
    
    def test_config_validation_with_custom_rules(self):
        """测试自定义验证规则"""
        manager = ConfigManager()
        
        def custom_validator(config):
            """自定义验证函数"""
            errors = []
            
            # 检查数据拆分比例总和
            if "data_split" in config:
                split = config["data_split"]
                total = split.get("train_ratio", 0) + split.get("val_ratio", 0) + split.get("test_ratio", 0)
                if abs(total - 1.0) > 0.001:
                    errors.append("数据拆分比例总和必须等于1.0")
            
            # 检查批次大小
            if "evaluation" in config and "batch_size" in config["evaluation"]:
                if config["evaluation"]["batch_size"] <= 0:
                    errors.append("批次大小必须大于0")
            
            return errors
        
        manager.add_custom_validator(custom_validator)
        
        # 测试有效配置
        valid_config = {
            "data_split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
            "evaluation": {"batch_size": 8}
        }
        
        assert manager.validate_config(valid_config) == True
        
        # 测试无效配置
        invalid_config = {
            "data_split": {"train_ratio": 0.5, "val_ratio": 0.3, "test_ratio": 0.3},
            "evaluation": {"batch_size": 0}
        }
        
        assert manager.validate_config(invalid_config) == False
    
    def test_error_handling_invalid_file(self):
        """测试无效文件的错误处理"""
        manager = ConfigManager()
        
        # 测试不存在的文件
        with pytest.raises(FileNotFoundError):
            manager.load_config("nonexistent_file.yaml")
        
        # 测试无效的YAML文件
        with pytest.raises(yaml.YAMLError):
            manager.load_config_from_string("invalid: yaml: content: [")
        
        # 测试无效的JSON文件
        with pytest.raises(json.JSONDecodeError):
            manager.load_config_from_string('{"invalid": json content}', format="json")
    
    def test_config_backup_and_restore(self, temp_dir):
        """测试配置备份和恢复"""
        manager = ConfigManager()
        
        original_config = {
            "test_section": {
                "param1": "original_value",
                "param2": 42
            }
        }
        
        config_path = temp_dir / "backup_test.yaml"
        manager.save_config(original_config, str(config_path))
        
        # 创建备份
        backup_path = manager.backup_config(str(config_path))
        assert Path(backup_path).exists()
        
        # 修改原配置
        modified_config = {
            "test_section": {
                "param1": "modified_value",
                "param2": 100
            }
        }
        manager.save_config(modified_config, str(config_path))
        
        # 恢复备份
        manager.restore_config(str(config_path), backup_path)
        
        # 验证恢复结果
        restored_config = manager.load_config(str(config_path))
        assert restored_config["test_section"]["param1"] == "original_value"
        assert restored_config["test_section"]["param2"] == 42