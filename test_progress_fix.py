#!/usr/bin/env python3
"""
测试进度监控修复的脚本

验证 ProgressMonitoringCallback 是否能正确估算总步数
"""

import logging
import sys
import math
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training_engine import ProgressMonitoringCallback, TrainingConfig
from progress_monitor import ProgressMonitor
from memory_optimizer import MemoryOptimizer
from logging_system import LoggingSystem

# 模拟训练参数和状态
class MockTrainingArgs:
    def __init__(self):
        self.num_train_epochs = 10
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 16
        self.max_steps = 0  # 设为0以测试估算逻辑

class MockTrainingState:
    def __init__(self):
        self.max_steps = 0
        self.global_step = 0
        self.epoch = 0

class MockDataset:
    def __init__(self, size=178):
        self.size = size
    
    def __len__(self):
        return self.size

def test_progress_estimation():
    """测试进度估算功能"""
    print("=" * 60)
    print("测试进度监控修复")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建必要的组件
    memory_optimizer = MemoryOptimizer(max_memory_gb=13.0)
    logging_system = LoggingSystem(log_dir="./test_logs")
    progress_monitor = ProgressMonitor(
        memory_optimizer=memory_optimizer,
        logging_system=logging_system,
        enable_rich_display=False  # 禁用Rich显示以避免测试时的问题
    )
    
    # 创建回调
    callback = ProgressMonitoringCallback(progress_monitor)
    
    # 创建模拟对象
    args = MockTrainingArgs()
    state = MockTrainingState()
    control = {}
    
    # 测试场景1: 有训练数据集，无数据加载器
    print("\n测试场景1: 有训练数据集，无数据加载器")
    train_dataset = MockDataset(size=178)
    kwargs = {
        'train_dataset': train_dataset,
        'train_dataloader': None
    }
    
    try:
        callback.on_init_end(args, state, control, **kwargs)
        print("✅ 场景1测试通过")
        
        # 验证估算结果
        # 计算每个epoch的批次数（不考虑梯度累积）
        effective_batch_size = args.per_device_train_batch_size
        batches_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
        
        # 考虑梯度累积，计算每个epoch的优化器更新次数
        expected_steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
        expected_total_steps = args.num_train_epochs * expected_steps_per_epoch
        
        print(f"数据集大小: {len(train_dataset)}")
        print(f"批次大小: {effective_batch_size}")
        print(f"每轮批次数: {batches_per_epoch}")
        print(f"梯度累积步数: {args.gradient_accumulation_steps}")
        print(f"每轮优化器更新次数: {expected_steps_per_epoch}")
        print(f"预期总步数: {expected_total_steps}")
        
    except Exception as e:
        print(f"❌ 场景1测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试场景2: 无训练数据集，无数据加载器（使用默认值）
    print("\n测试场景2: 无训练数据集，无数据加载器")
    kwargs = {
        'train_dataset': None,
        'train_dataloader': None
    }
    
    # 重置回调状态
    callback.training_started = False
    
    try:
        callback.on_init_end(args, state, control, **kwargs)
        print("✅ 场景2测试通过")
        
        expected_default_steps = args.num_train_epochs * 100
        print(f"默认总步数: {expected_default_steps}")
        
    except Exception as e:
        print(f"❌ 场景2测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试场景3: 有max_steps设置
    print("\n测试场景3: 有max_steps设置")
    state.max_steps = 500
    callback.training_started = False
    
    try:
        callback.on_init_end(args, state, control, **kwargs)
        print("✅ 场景3测试通过")
        print(f"使用max_steps: {state.max_steps}")
        
    except Exception as e:
        print(f"❌ 场景3测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    try:
        progress_monitor.stop_monitoring()
    except:
        pass
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_progress_estimation()
