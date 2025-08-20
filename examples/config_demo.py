"""
Industry Evaluation System 配置管理演示

这个示例程序展示了配置管理系统的各种功能：
- 配置文件创建和加载
- 配置验证和更新
- 模型和评估器配置管理
- 配置模板生成
- 环境变量支持
"""

import os
import tempfile
import time
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from industry_evaluation.config.config_manager import (
    ConfigManager,
    ConfigTemplate,
    ConfigValidator,
    EnvironmentConfigLoader,
    ModelConfig,
    EvaluatorConfig,
    SystemConfig,
    EvaluationSystemConfig
)


class ConfigDemo:
    """配置管理演示类"""
    
    def __init__(self):
        """初始化演示环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"📁 临时目录: {self.temp_dir}")
    
    def demo_config_templates(self):
        """演示配置模板"""
        print("\n🎨 演示配置模板生成")
        print("-" * 40)
        
        # 生成金融行业配置模板
        print("📋 生成金融行业配置模板...")
        finance_config = ConfigTemplate.generate_finance_config()
        
        print(f"✅ 金融配置生成完成:")
        print(f"   - 最大工作线程: {finance_config.system.max_workers}")
        print(f"   - 模型数量: {len(finance_config.models)}")
        print(f"   - 评估器数量: {len(finance_config.evaluators)}")
        print(f"   - 支持行业: {', '.join(finance_config.industry_domains)}")
        
        # 保存金融配置模板
        finance_file = self.temp_dir / "finance_config.yaml"
        ConfigTemplate.save_template(finance_config, finance_file)
        print(f"💾 金融配置已保存到: {finance_file}")
        
        # 生成医疗行业配置模板
        print("\n🏥 生成医疗行业配置模板...")
        healthcare_config = ConfigTemplate.generate_healthcare_config()
        
        print(f"✅ 医疗配置生成完成:")
        print(f"   - 最大工作线程: {healthcare_config.system.max_workers}")
        print(f"   - 模型数量: {len(healthcare_config.models)}")
        print(f"   - 评估器数量: {len(healthcare_config.evaluators)}")
        print(f"   - 知识权重: {healthcare_config.default_weights.get('knowledge', 'N/A')}")
        
        # 保存医疗配置模板
        healthcare_file = self.temp_dir / "healthcare_config.yaml"
        ConfigTemplate.save_template(healthcare_config, healthcare_file)
        print(f"💾 医疗配置已保存到: {healthcare_file}")
        
        return finance_file, healthcare_file
    
    def demo_config_loading(self, config_file: Path):
        """演示配置加载"""
        print(f"\n📂 演示配置加载 ({config_file.name})")
        print("-" * 40)
        
        # 创建配置管理器
        print("🔧 创建配置管理器...")
        config_manager = ConfigManager(config_file, auto_reload=False)
        
        # 获取配置信息
        config = config_manager.get_config()
        
        print("✅ 配置加载成功:")
        print(f"   - 版本: {config.version}")
        print(f"   - 创建时间: {config.created_at}")
        print(f"   - 更新时间: {config.updated_at}")
        print(f"   - 系统配置: 最大工作线程 {config.system.max_workers}")
        print(f"   - 模型配置: {len(config.models)} 个模型")
        print(f"   - 评估器配置: {len(config.evaluators)} 个评估器")
        
        # 显示模型详情
        print("\n🤖 模型配置详情:")
        for model_id, model_config in list(config.models.items())[:3]:
            print(f"   - {model_id}:")
            print(f"     * 类型: {model_config.adapter_type}")
            print(f"     * 超时: {model_config.timeout}秒")
            print(f"     * 重试次数: {model_config.max_retries}")
        
        # 显示评估器详情
        print("\n📊 评估器配置详情:")
        for evaluator_id, evaluator_config in config.evaluators.items():
            print(f"   - {evaluator_id}:")
            print(f"     * 类型: {evaluator_config.evaluator_type}")
            print(f"     * 权重: {evaluator_config.weight}")
            print(f"     * 阈值: {evaluator_config.threshold}")
            print(f"     * 启用: {evaluator_config.enabled}")
        
        return config_manager
    
    def demo_config_validation(self):
        """演示配置验证"""
        print("\n✅ 演示配置验证")
        print("-" * 40)
        
        # 测试有效的模型配置
        print("🔍 测试有效的模型配置...")
        valid_model_config = ModelConfig(
            model_id="test_model",
            adapter_type="openai",
            api_key="valid_key",
            timeout=30,
            max_retries=3
        )
        
        errors = ConfigValidator.validate_model_config(valid_model_config)
        if not errors:
            print("✅ 模型配置验证通过")
        else:
            print(f"❌ 模型配置验证失败: {errors}")
        
        # 测试无效的模型配置
        print("\n🔍 测试无效的模型配置...")
        invalid_model_config = ModelConfig(
            model_id="",  # 空ID
            adapter_type="openai",
            timeout=0,  # 无效超时
            max_retries=-1  # 无效重试次数
        )
        
        errors = ConfigValidator.validate_model_config(invalid_model_config)
        if errors:
            print("✅ 成功检测到配置错误:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("❌ 未能检测到配置错误")
        
        # 测试评估器配置验证
        print("\n🔍 测试评估器配置验证...")
        valid_evaluator_config = EvaluatorConfig(
            evaluator_type="knowledge",
            weight=0.5,
            threshold=0.7
        )
        
        errors = ConfigValidator.validate_evaluator_config(valid_evaluator_config)
        if not errors:
            print("✅ 评估器配置验证通过")
        else:
            print(f"❌ 评估器配置验证失败: {errors}")
        
        # 测试系统配置验证
        print("\n🔍 测试系统配置验证...")
        invalid_system_config = SystemConfig(
            max_workers=0,  # 无效值
            log_level="INVALID",  # 无效日志级别
            cache_ttl=-1,  # 无效TTL
            metrics_port=70000  # 无效端口
        )
        
        errors = ConfigValidator.validate_system_config(invalid_system_config)
        if errors:
            print("✅ 成功检测到系统配置错误:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("❌ 未能检测到系统配置错误")
    
    def demo_config_updates(self, config_manager: ConfigManager):
        """演示配置更新"""
        print("\n🔄 演示配置更新")
        print("-" * 40)
        
        # 获取原始配置
        original_config = config_manager.get_config()
        original_workers = original_config.system.max_workers
        
        print(f"📋 原始配置: 最大工作线程 = {original_workers}")
        
        # 更新系统配置
        print("🔧 更新系统配置...")
        updates = {
            "system": {
                "max_workers": 16,
                "log_level": "DEBUG",
                "cache_enabled": True
            }
        }
        
        success = config_manager.update_config(updates)
        if success:
            updated_config = config_manager.get_config()
            print("✅ 配置更新成功:")
            print(f"   - 最大工作线程: {original_workers} → {updated_config.system.max_workers}")
            print(f"   - 日志级别: INFO → {updated_config.system.log_level}")
            print(f"   - 缓存启用: {updated_config.system.cache_enabled}")
        else:
            print("❌ 配置更新失败")
        
        # 添加新模型配置
        print("\n🤖 添加新模型配置...")
        new_model_config = ModelConfig(
            model_id="demo_model",
            adapter_type="local",
            model_name="/path/to/demo/model",
            timeout=60,
            max_retries=5,
            fallback_enabled=True,
            fallback_response="演示模型暂时不可用"
        )
        
        success = config_manager.add_model("demo_model", new_model_config)
        if success:
            print("✅ 新模型配置添加成功")
            updated_config = config_manager.get_config()
            print(f"   - 模型数量: {len(original_config.models)} → {len(updated_config.models)}")
            print(f"   - 新模型ID: {new_model_config.model_id}")
            print(f"   - 新模型类型: {new_model_config.adapter_type}")
        else:
            print("❌ 新模型配置添加失败")
        
        # 添加新评估器配置
        print("\n📊 添加新评估器配置...")
        new_evaluator_config = EvaluatorConfig(
            evaluator_type="demo_evaluator",
            enabled=True,
            weight=0.2,
            threshold=0.6,
            parameters={"demo_param": "demo_value"}
        )
        
        success = config_manager.add_evaluator("demo_evaluator", new_evaluator_config)
        if success:
            print("✅ 新评估器配置添加成功")
            updated_config = config_manager.get_config()
            print(f"   - 评估器数量: {len(original_config.evaluators)} → {len(updated_config.evaluators)}")
            print(f"   - 新评估器ID: {new_evaluator_config.evaluator_type}")
            print(f"   - 新评估器权重: {new_evaluator_config.weight}")
        else:
            print("❌ 新评估器配置添加失败")
        
        # 移除演示配置
        print("\n🗑️ 清理演示配置...")
        config_manager.remove_model("demo_model")
        config_manager.remove_evaluator("demo_evaluator")
        print("✅ 演示配置清理完成")
    
    def demo_environment_config(self):
        """演示环境变量配置"""
        print("\n🌍 演示环境变量配置")
        print("-" * 40)
        
        # 设置环境变量
        print("🔧 设置环境变量...")
        os.environ["EVAL_MAX_WORKERS"] = "12"
        os.environ["EVAL_LOG_LEVEL"] = "WARNING"
        os.environ["EVAL_LOG_FILE"] = "/tmp/evaluation.log"
        os.environ["OPENAI_API_KEY"] = "demo_api_key_123"
        os.environ["OPENAI_MODEL_NAME"] = "gpt-4"
        
        # 从环境变量加载配置
        env_config = EnvironmentConfigLoader.load_from_env()
        
        print("✅ 环境变量配置加载成功:")
        if "system" in env_config:
            system_config = env_config["system"]
            print(f"   - 最大工作线程: {system_config.get('max_workers', 'N/A')}")
            print(f"   - 日志级别: {system_config.get('log_level', 'N/A')}")
            print(f"   - 日志文件: {system_config.get('log_file', 'N/A')}")
        
        if "models" in env_config:
            models_config = env_config["models"]
            if "openai_default" in models_config:
                openai_config = models_config["openai_default"]
                print(f"   - OpenAI模型: {openai_config.get('model_name', 'N/A')}")
                print(f"   - API密钥: {openai_config.get('api_key', 'N/A')[:10]}...")
        
        # 清理环境变量
        for key in ["EVAL_MAX_WORKERS", "EVAL_LOG_LEVEL", "EVAL_LOG_FILE", "OPENAI_API_KEY", "OPENAI_MODEL_NAME"]:
            if key in os.environ:
                del os.environ[key]
        
        print("🧹 环境变量清理完成")
    
    def demo_config_monitoring(self):
        """演示配置监控"""
        print("\n👁️ 演示配置文件监控")
        print("-" * 40)
        
        # 创建测试配置文件
        test_config_file = self.temp_dir / "monitor_test_config.yaml"
        config = ConfigTemplate.generate_finance_config()
        ConfigTemplate.save_template(config, test_config_file)
        
        # 创建带监控的配置管理器
        print("🔧 创建带文件监控的配置管理器...")
        config_manager = ConfigManager(test_config_file, auto_reload=True)
        
        # 注册回调函数
        callback_called = False
        
        def config_change_callback(old_config, new_config):
            nonlocal callback_called
            callback_called = True
            print(f"📢 配置变更回调触发:")
            print(f"   - 旧版本最大工作线程: {old_config.system.max_workers}")
            print(f"   - 新版本最大工作线程: {new_config.system.max_workers}")
        
        config_manager.register_reload_callback(config_change_callback)
        
        # 获取初始配置
        initial_config = config_manager.get_config()
        initial_workers = initial_config.system.max_workers
        
        print(f"📋 初始配置: 最大工作线程 = {initial_workers}")
        
        # 修改配置文件
        print("✏️ 修改配置文件...")
        modified_config = config_manager.get_config()
        modified_config.system.max_workers = 20
        config_manager.save_config()
        
        # 手动触发重新加载（模拟文件监控）
        print("🔄 触发配置重新加载...")
        config_manager.reload_config()
        
        # 检查回调是否被调用
        if callback_called:
            print("✅ 配置变更回调成功触发")
        else:
            print("⚠️ 配置变更回调未触发")
        
        # 验证配置更新
        updated_config = config_manager.get_config()
        if updated_config.system.max_workers == 20:
            print("✅ 配置文件监控和更新成功")
        else:
            print("❌ 配置文件监控和更新失败")
        
        # 停止文件监控
        config_manager.stop_file_monitoring()
        print("⏹️ 文件监控已停止")
    
    def demo_config_performance(self):
        """演示配置性能测试"""
        print("\n⚡ 演示配置性能测试")
        print("-" * 40)
        
        # 创建大型配置
        print("🏗️ 创建大型配置...")
        large_config = EvaluationSystemConfig()
        
        # 添加大量模型配置
        for i in range(100):
            model_config = ModelConfig(
                model_id=f"model_{i}",
                adapter_type="demo",
                timeout=30,
                max_retries=3
            )
            large_config.models[f"model_{i}"] = model_config
        
        # 添加大量评估器配置
        for i in range(50):
            evaluator_config = EvaluatorConfig(
                evaluator_type=f"evaluator_{i}",
                weight=0.02,
                threshold=0.5
            )
            large_config.evaluators[f"evaluator_{i}"] = evaluator_config
        
        print(f"✅ 大型配置创建完成: {len(large_config.models)} 个模型, {len(large_config.evaluators)} 个评估器")
        
        # 保存性能测试
        large_config_file = self.temp_dir / "large_config.yaml"
        
        start_time = time.time()
        ConfigTemplate.save_template(large_config, large_config_file)
        save_time = time.time() - start_time
        
        print(f"💾 配置保存耗时: {save_time:.3f} 秒")
        
        # 加载性能测试
        start_time = time.time()
        config_manager = ConfigManager(large_config_file, auto_reload=False)
        loaded_config = config_manager.get_config()
        load_time = time.time() - start_time
        
        print(f"📂 配置加载耗时: {load_time:.3f} 秒")
        
        # 验证性能测试
        start_time = time.time()
        errors = ConfigValidator.validate_full_config(loaded_config)
        validate_time = time.time() - start_time
        
        print(f"✅ 配置验证耗时: {validate_time:.3f} 秒")
        print(f"🔍 验证结果: {len(errors)} 个错误")
        
        # 更新性能测试
        start_time = time.time()
        updates = {"system": {"max_workers": 32}}
        config_manager.update_config(updates)
        update_time = time.time() - start_time
        
        print(f"🔄 配置更新耗时: {update_time:.3f} 秒")
    
    def run_config_demo(self):
        """运行配置管理演示"""
        print("⚙️ Industry Evaluation System - 配置管理演示")
        print("=" * 60)
        
        try:
            # 1. 配置模板演示
            finance_file, healthcare_file = self.demo_config_templates()
            
            # 2. 配置加载演示
            config_manager = self.demo_config_loading(finance_file)
            
            # 3. 配置验证演示
            self.demo_config_validation()
            
            # 4. 配置更新演示
            self.demo_config_updates(config_manager)
            
            # 5. 环境变量配置演示
            self.demo_environment_config()
            
            # 6. 配置监控演示
            self.demo_config_monitoring()
            
            # 7. 配置性能测试
            self.demo_config_performance()
            
            print("\n🎉 配置管理演示完成!")
            print("=" * 60)
            print("✅ 已演示的功能:")
            print("  • 配置模板生成 (金融、医疗行业)")
            print("  • 配置文件加载和解析")
            print("  • 配置验证和错误检测")
            print("  • 配置动态更新")
            print("  • 模型和评估器配置管理")
            print("  • 环境变量配置支持")
            print("  • 配置文件监控和热更新")
            print("  • 配置性能测试")
            
            print(f"\n📁 配置文件位置: {self.temp_dir}")
            print("💡 提示: 可以查看生成的配置文件了解详细结构")
            
        except Exception as e:
            print(f"❌ 配置管理演示过程中发生错误: {str(e)}")
            raise


def main():
    """主函数"""
    demo = ConfigDemo()
    demo.run_config_demo()


if __name__ == "__main__":
    main()