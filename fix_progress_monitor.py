#!/usr/bin/env python3
"""
修复进度监控器中的 NoneType 转换错误

这个脚本修复了 progress_monitor.py 中可能导致 "can't convert type 'NoneType' to numerator/denominator" 错误的问题。
"""

import os
import sys
from pathlib import Path

def fix_progress_monitor():
    """修复进度监控器中的类型转换问题"""
    
    # 读取原始文件
    progress_monitor_path = Path("src/progress_monitor.py")
    
    if not progress_monitor_path.exists():
        print(f"错误: 找不到文件 {progress_monitor_path}")
        return False
    
    print(f"正在修复 {progress_monitor_path}...")
    
    # 读取文件内容
    with open(progress_monitor_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: 在 update_progress 方法中添加类型检查
    old_update_progress = '''    def update_progress(self, epoch: float, step: int, loss: float, learning_rate: float) -> None:
        """
        更新进度信息
        
        Args:
            epoch: 当前轮次
            step: 当前步数
            loss: 当前损失
            learning_rate: 当前学习率
        """
        self.current_epoch = epoch
        self.current_step = step
        self.current_loss = loss
        self.current_lr = learning_rate'''
    
    new_update_progress = '''    def update_progress(self, epoch: float, step: int, loss: float, learning_rate: float) -> None:
        """
        更新进度信息
        
        Args:
            epoch: 当前轮次
            step: 当前步数
            loss: 当前损失
            learning_rate: 当前学习率
        """
        # 安全地处理可能为None的值
        self.current_epoch = float(epoch) if epoch is not None else 0.0
        self.current_step = int(step) if step is not None else 0
        self.current_loss = float(loss) if loss is not None else 0.0
        self.current_lr = float(learning_rate) if learning_rate is not None else 0.0'''
    
    # 修复2: 在 _create_progress_snapshot 方法中添加安全检查
    old_snapshot = '''        return ProgressSnapshot(
            timestamp=now,
            epoch=self.current_epoch,
            step=self.current_step,
            total_steps=self.total_steps,
            loss=self.current_loss,
            learning_rate=self.current_lr,
            memory_status=memory_status,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining
        )'''
    
    new_snapshot = '''        return ProgressSnapshot(
            timestamp=now,
            epoch=float(self.current_epoch) if self.current_epoch is not None else 0.0,
            step=int(self.current_step) if self.current_step is not None else 0,
            total_steps=int(self.total_steps) if self.total_steps is not None else 0,
            loss=float(self.current_loss) if self.current_loss is not None else 0.0,
            learning_rate=float(self.current_lr) if self.current_lr is not None else 0.0,
            memory_status=memory_status,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining
        )'''
    
    # 修复3: 在 generate_training_summary 方法中添加安全检查
    old_summary = '''        summary = {
            "training_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": str(total_time),
                "total_steps": self.current_step,
                "final_epoch": self.current_epoch,
                "final_loss": self.current_loss,
                "final_learning_rate": self.current_lr
            },'''
    
    new_summary = '''        summary = {
            "training_summary": {
                "start_time": self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": str(total_time),
                "total_steps": int(self.current_step) if self.current_step is not None else 0,
                "final_epoch": float(self.current_epoch) if self.current_epoch is not None else 0.0,
                "final_loss": float(self.current_loss) if self.current_loss is not None else 0.0,
                "final_learning_rate": float(self.current_lr) if self.current_lr is not None else 0.0
            },'''
    
    # 应用修复
    content = content.replace(old_update_progress, new_update_progress)
    content = content.replace(old_snapshot, new_snapshot)
    content = content.replace(old_summary, new_summary)
    
    # 创建备份
    backup_path = progress_monitor_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已创建备份文件: {backup_path}")
    
    # 写入修复后的内容
    with open(progress_monitor_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复 {progress_monitor_path}")
    return True

def fix_training_engine():
    """修复训练引擎中的回调函数问题"""
    
    training_engine_path = Path("src/training_engine.py")
    
    if not training_engine_path.exists():
        print(f"错误: 找不到文件 {training_engine_path}")
        return False
    
    print(f"正在修复 {training_engine_path}...")
    
    # 读取文件内容
    with open(training_engine_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复回调函数中的 epoch 访问
    old_callback_pattern = '''            self.progress_monitor.update_progress(
                epoch=getattr(state, 'epoch', 0),
                step=state.global_step,
                loss=None,
                learning_rate=None
            )'''
    
    new_callback_pattern = '''            # 安全地获取epoch值
            epoch_value = getattr(state, 'epoch', None)
            if epoch_value is None:
                epoch_value = 0.0
            else:
                try:
                    epoch_value = float(epoch_value)
                except (TypeError, ValueError):
                    epoch_value = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch_value,
                step=state.global_step,
                loss=None,
                learning_rate=None
            )'''
    
    # 应用修复
    content = content.replace(old_callback_pattern, new_callback_pattern)
    
    # 修复 on_log 回调中的类型转换
    old_log_callback = '''        if logs is not None:
            epoch = logs.get("epoch", 0)
            loss = logs.get("loss", 0.0)
            learning_rate = logs.get("learning_rate", 0.0)
            
            self.progress_monitor.update_progress(
                epoch=epoch,
                step=state.global_step,
                loss=loss,
                learning_rate=learning_rate
            )'''
    
    new_log_callback = '''        if logs is not None:
            # 安全地获取和转换日志值
            epoch = logs.get("epoch", 0)
            loss = logs.get("loss", 0.0)
            learning_rate = logs.get("learning_rate", 0.0)
            
            # 确保类型正确
            try:
                epoch = float(epoch) if epoch is not None else 0.0
                loss = float(loss) if loss is not None else 0.0
                learning_rate = float(learning_rate) if learning_rate is not None else 0.0
            except (TypeError, ValueError):
                epoch = 0.0
                loss = 0.0
                learning_rate = 0.0
            
            self.progress_monitor.update_progress(
                epoch=epoch,
                step=state.global_step,
                loss=loss,
                learning_rate=learning_rate
            )'''
    
    content = content.replace(old_log_callback, new_log_callback)
    
    # 创建备份
    backup_path = training_engine_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已创建备份文件: {backup_path}")
    
    # 写入修复后的内容
    with open(training_engine_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复 {training_engine_path}")
    return True

def main():
    """主函数"""
    print("开始修复 NoneType 转换错误...")
    
    success = True
    
    # 修复进度监控器
    if not fix_progress_monitor():
        success = False
    
    # 修复训练引擎
    if not fix_training_engine():
        success = False
    
    if success:
        print("\n✅ 修复完成！")
        print("\n修复内容:")
        print("1. 在 progress_monitor.py 中添加了类型安全检查")
        print("2. 在 training_engine.py 中修复了回调函数的类型转换")
        print("3. 确保所有 None 值都被正确处理")
        print("\n现在可以重新运行训练程序。")
    else:
        print("\n❌ 修复过程中出现错误，请检查上述消息。")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
