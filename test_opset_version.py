#!/usr/bin/env python3
"""
测试ONNX opset版本更新

验证所有配置文件都使用opset版本18
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(".") / "src"))

from src.export_models import ExportConfiguration
from src.export_config import ConfigurationManager
from src.config_presets import ConfigPresets

def test_opset_version_updates():
    """测试ONNX opset版本更新"""
    print("🧪 测试ONNX opset版本更新")
    print("=" * 40)
    
    # 测试默认配置
    print("1. 测试默认配置...")
    default_config = ExportConfiguration(
        checkpoint_path="test",
        base_model_name="test",
        output_directory="test"
    )
    assert default_config.onnx_opset_version == 20, f"默认配置opset版本应为20，实际为{default_config.onnx_opset_version}"
    print("   ✅ 默认配置opset版本正确")
    
    # 测试配置管理器默认值
    print("2. 测试配置管理器默认值...")
    manager = ConfigurationManager()
    default_dict = manager._get_default_config()
    assert default_dict['onnx_opset_version'] == 20, f"配置管理器默认opset版本应为20，实际为{default_dict['onnx_opset_version']}"
    print("   ✅ 配置管理器默认值正确")
    
    # 测试预设配置
    print("3. 测试预设配置...")
    presets = ConfigPresets.get_all_presets()
    for preset_name, preset_data in presets.items():
        opset_version = preset_data['config']['onnx_opset_version']
        assert opset_version == 20, f"预设{preset_name}的opset版本应为20，实际为{opset_version}"
        print(f"   ✅ 预设{preset_name}的opset版本正确")
    
    print("\n🎉 所有ONNX opset版本测试通过！")
    print("所有配置都已更新为opset版本20")
    
    return True

if __name__ == "__main__":
    try:
        success = test_opset_version_updates()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)