#!/usr/bin/env python3
"""
测试修复效果的简单脚本
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_progress_monitor_fix():
    """测试进度监控器的修复"""
    print("测试进度监控器修复...")
    
    try:
        from src.progress_monitor import ProgressMonitor
        from src.memory_optimizer import MemoryOptimizer
        
        # 创建内存优化器
        memory_optimizer = MemoryOptimizer(max_memory_gb=13.0)
        
        # 创建进度监控器
        progress_monitor = ProgressMonitor(
            memory_optimizer=memory_optimizer,
            logging_system=None,
            enable_rich_display=False  # 禁用Rich显示以避免测试中的问题
        )
        
        # 测试 None 值处理
        print("  测试 None 值处理...")
        progress_monitor.update_progress(
            epoch=None,  # 测试 None 值
            step=None,
            loss=None,
            learning_rate=None
        )
        
        # 验证值被正确设置为默认值
        assert progress_monitor.current_epoch == 0.0, f"期望 0.0，实际 {progress_monitor.current_epoch}"
        assert progress_monitor.current_step == 0, f"期望 0，实际 {progress_monitor.current_step}"
        assert progress_monitor.current_loss == 0.0, f"期望 0.0，实际 {progress_monitor.current_loss}"
        assert progress_monitor.current_lr == 0.0, f"期望 0.0，实际 {progress_monitor.current_lr}"
        
        # 测试正常值处理
        print("  测试正常值处理...")
        progress_monitor.update_progress(
            epoch=1.5,
            step=100,
            loss=0.5,
            learning_rate=1e-4
        )
        
        assert progress_monitor.current_epoch == 1.5
        assert progress_monitor.current_step == 100
        assert progress_monitor.current_loss == 0.5
        assert progress_monitor.current_lr == 1e-4
        
        # 测试字符串转换
        print("  测试字符串转换...")
        progress_monitor.update_progress(
            epoch="2.0",  # 字符串形式的数字
            step="200",
            loss="0.3",
            learning_rate="5e-5"
        )
        
        assert progress_monitor.current_epoch == 2.0
        assert progress_monitor.current_step == 200
        assert progress_monitor.current_loss == 0.3
        assert progress_monitor.current_lr == 5e-5
        
        print("  ✅ 进度监控器修复测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 进度监控器修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_engine_fix():
    """测试训练引擎的修复"""
    print("测试训练引擎修复...")
    
    try:
        from src.training_engine import ProgressMonitoringCallback
        from src.progress_monitor import ProgressMonitor
        from src.memory_optimizer import MemoryOptimizer
        
        # 创建模拟的训练状态
        class MockState:
            def __init__(self, epoch=None, global_step=100):
                self.epoch = epoch
                self.global_step = global_step
        
        # 创建进度监控器
        memory_optimizer = MemoryOptimizer(max_memory_gb=13.0)
        progress_monitor = ProgressMonitor(
            memory_optimizer=memory_optimizer,
            logging_system=None,
            enable_rich_display=False
        )
        
        # 创建回调
        callback = ProgressMonitoringCallback(progress_monitor)
        
        # 测试 None epoch 处理
        print("  测试 None epoch 处理...")
        mock_state = MockState(epoch=None, global_step=50)
        
        # 这应该不会抛出异常
        callback.on_epoch_begin(None, mock_state, None)
        callback.on_epoch_end(None, mock_state, None)
        
        # 测试正常 epoch 处理
        print("  测试正常 epoch 处理...")
        mock_state = MockState(epoch=1.0, global_step=100)
        callback.on_epoch_begin(None, mock_state, None)
        callback.on_epoch_end(None, mock_state, None)
        
        # 测试日志回调
        print("  测试日志回调...")
        logs = {
            "epoch": None,  # 测试 None 值
            "loss": None,
            "learning_rate": None
        }
        callback.on_log(None, mock_state, None, logs=logs)
        
        # 测试正常日志
        logs = {
            "epoch": 1.5,
            "loss": 0.4,
            "learning_rate": 2e-5
        }
        callback.on_log(None, mock_state, None, logs=logs)
        
        print("  ✅ 训练引擎修复测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 训练引擎修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试修复效果...\n")
    
    success = True
    
    # 测试进度监控器修复
    if not test_progress_monitor_fix():
        success = False
    
    print()
    
    # 测试训练引擎修复
    if not test_training_engine_fix():
        success = False
    
    print()
    
    if success:
        print("🎉 所有测试通过！修复成功。")
        print("\n现在可以安全地重新运行 main.py 进行训练。")
        return 0
    else:
        print("❌ 部分测试失败，请检查修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
