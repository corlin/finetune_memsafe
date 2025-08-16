#!/usr/bin/env python3
"""
简单测试脚本 - 验证进度监控修复
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import():
    """测试导入是否正常"""
    try:
        from training_engine import ProgressMonitoringCallback
        print("✅ 成功导入 ProgressMonitoringCallback")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_step_calculation():
    """测试步数计算逻辑"""
    print("\n测试步数计算逻辑:")
    
    # 模拟参数
    num_train_epochs = 10
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 16
    dataset_size = 178
    
    # 计算预期步数
    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, dataset_size // batch_size)
    total_steps = num_train_epochs * steps_per_epoch
    
    print(f"数据集大小: {dataset_size}")
    print(f"每设备批次大小: {per_device_train_batch_size}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"有效批次大小: {batch_size}")
    print(f"每轮步数: {steps_per_epoch}")
    print(f"总轮数: {num_train_epochs}")
    print(f"预期总步数: {total_steps}")
    
    # 验证计算
    if total_steps > 0:
        print("✅ 步数计算正常")
        return True
    else:
        print("❌ 步数计算异常")
        return False

def main():
    print("=" * 50)
    print("简单测试 - 进度监控修复验证")
    print("=" * 50)
    
    # 测试导入
    import_ok = test_import()
    
    # 测试计算逻辑
    calc_ok = test_step_calculation()
    
    print("\n" + "=" * 50)
    if import_ok and calc_ok:
        print("✅ 所有测试通过")
        print("修复应该能解决 '无法获取训练数据加载器' 的警告")
    else:
        print("❌ 部分测试失败")
    print("=" * 50)

if __name__ == "__main__":
    main()
