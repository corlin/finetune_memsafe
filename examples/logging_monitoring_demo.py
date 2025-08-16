#!/usr/bin/env python3
"""
日志记录和监控系统演示

展示如何使用日志记录系统和进度监控器来跟踪训练进度和内存使用情况。
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logging_system import LoggingSystem, TrainingMetrics
from src.progress_monitor import ProgressMonitor
from src.memory_optimizer import MemoryOptimizer, MemoryStatus


def simulate_training_step(step: int, total_steps: int) -> tuple:
    """
    模拟训练步骤
    
    Args:
        step: 当前步骤
        total_steps: 总步骤数
        
    Returns:
        tuple: (epoch, loss, learning_rate)
    """
    # 模拟训练进度
    epoch = step / (total_steps / 10)  # 假设10个epoch
    
    # 模拟损失下降
    base_loss = 2.0
    loss_decay = 0.95 ** (step / 100)
    noise = random.uniform(-0.1, 0.1)
    loss = base_loss * loss_decay + noise
    loss = max(0.1, loss)  # 确保损失不为负
    
    # 模拟学习率调度
    initial_lr = 5e-5
    lr_decay = 0.99 ** (step / 50)
    learning_rate = initial_lr * lr_decay
    
    return epoch, loss, learning_rate


def simulate_memory_usage(step: int, total_steps: int) -> MemoryStatus:
    """
    模拟内存使用情况
    
    Args:
        step: 当前步骤
        total_steps: 总步骤数
        
    Returns:
        MemoryStatus: 模拟的内存状态
    """
    # 模拟内存使用模式
    base_memory = 8.0  # 基础内存使用 8GB
    
    # 训练过程中内存逐渐增加
    memory_growth = (step / total_steps) * 2.0  # 最多增加2GB
    
    # 添加一些随机波动
    noise = random.uniform(-0.5, 0.5)
    
    allocated_gb = base_memory + memory_growth + noise
    allocated_gb = max(6.0, min(12.0, allocated_gb))  # 限制在6-12GB之间
    
    # 模拟其他内存指标
    cached_gb = allocated_gb * 0.3  # 缓存内存约为分配内存的30%
    total_gb = 16.0  # 假设总内存16GB
    available_gb = total_gb - allocated_gb
    is_safe = allocated_gb < 13.0  # 13GB以下认为安全
    
    return MemoryStatus(
        allocated_gb=allocated_gb,
        cached_gb=cached_gb,
        total_gb=total_gb,
        available_gb=available_gb,
        is_safe=is_safe,
        timestamp=datetime.now()
    )


def demo_logging_system():
    """演示日志记录系统"""
    print("🚀 日志记录系统演示")
    print("=" * 50)
    
    # 创建日志记录系统
    log_dir = "./demo_logs"
    logging_system = LoggingSystem(log_dir=log_dir, run_name="demo_run")
    
    try:
        # 记录训练开始
        config = {
            "model_name": "Qwen/Qwen3-4B-Thinking-2507",
            "batch_size": 4,
            "learning_rate": 5e-5,
            "num_epochs": 3
        }
        logging_system.log_training_start(config)
        
        # 模拟训练过程
        total_steps = 100
        for step in range(1, total_steps + 1):
            epoch, loss, learning_rate = simulate_training_step(step, total_steps)
            memory_status = simulate_memory_usage(step, total_steps)
            
            # 创建训练指标
            metrics = TrainingMetrics(
                epoch=epoch,
                step=step,
                loss=loss,
                learning_rate=learning_rate,
                memory_usage=memory_status,
                timestamp=datetime.now()
            )
            
            # 记录训练指标
            logging_system.log_training_metrics(metrics)
            
            # 每10步记录内存状态
            if step % 10 == 0:
                logging_system.log_memory_status(memory_status, step)
                print(f"步骤 {step:3d}/{total_steps}: 损失={loss:.4f}, 内存={memory_status.allocated_gb:.1f}GB")
            
            # 模拟一些警告和错误
            if step == 30:
                logging_system.warning("内存使用接近限制", "MEMORY_MONITOR", 
                                     {"memory_gb": memory_status.allocated_gb})
            
            if step == 60:
                logging_system.error("模拟错误恢复", "TRAINING", 
                                   {"error_type": "temporary_failure"})
            
            # 短暂延迟以模拟真实训练
            time.sleep(0.05)
        
        # 记录训练完成
        logging_system.info("训练演示完成", "DEMO", {"total_steps": total_steps})
        
        # 获取日志摘要
        summary = logging_system.get_log_summary()
        print(f"\n📊 日志摘要:")
        print(f"  运行名称: {summary['run_name']}")
        print(f"  日志目录: {summary['log_dir']}")
        print(f"  结构化日志条目: {summary['structured_log_entries']}")
        print(f"  TensorBoard指标: {len(summary['tensorboard_metrics'])} 个")
        
    finally:
        # 关闭日志系统
        logging_system.close()
        print(f"\n✅ 日志已保存到: {log_dir}")


def demo_progress_monitor():
    """演示进度监控系统"""
    print("\n🔍 进度监控系统演示")
    print("=" * 50)
    
    # 创建内存优化器（模拟）
    class MockMemoryOptimizer:
        def get_memory_status(self):
            return simulate_memory_usage(
                random.randint(1, 100), 100
            )
    
    memory_optimizer = MockMemoryOptimizer()
    
    # 创建进度监控器
    progress_monitor = ProgressMonitor(
        memory_optimizer=memory_optimizer,
        enable_rich_display=False  # 在演示中禁用Rich显示
    )
    
    try:
        # 开始监控
        total_steps = 50
        progress_monitor.start_monitoring(total_steps)
        
        print("开始模拟训练...")
        
        # 模拟训练过程
        for step in range(1, total_steps + 1):
            epoch, loss, learning_rate = simulate_training_step(step, total_steps)
            
            # 更新进度
            progress_monitor.update_progress(epoch, step, loss, learning_rate)
            
            # 每10步显示状态
            if step % 10 == 0:
                status = progress_monitor.get_current_status()
                print(f"步骤 {step:2d}/{total_steps}: "
                      f"进度={status['progress_percent']:.1f}%, "
                      f"损失={loss:.4f}, "
                      f"速度={status['steps_per_second']:.2f} 步/秒")
            
            # 短暂延迟
            time.sleep(0.1)
        
        # 停止监控
        progress_monitor.stop_monitoring()
        
        # 生成训练摘要
        summary = progress_monitor.generate_training_summary()
        
        print(f"\n📈 训练摘要:")
        training_summary = summary['training_summary']
        print(f"  总时长: {training_summary['total_duration']}")
        print(f"  总步数: {training_summary['total_steps']}")
        print(f"  最终损失: {training_summary['final_loss']:.4f}")
        
        performance = summary['performance_metrics']
        print(f"\n⚡ 性能指标:")
        print(f"  平均速度: {performance['avg_steps_per_second']:.2f} 步/秒")
        print(f"  峰值内存: {performance['peak_memory_gb']:.1f} GB")
        print(f"  平均内存: {performance['avg_memory_gb']:.1f} GB")
        print(f"  训练稳定性: {performance['training_stability']:.2f}")
        
        # 保存进度报告
        report_path = "./demo_progress_report.json"
        progress_monitor.save_progress_report(report_path)
        print(f"\n✅ 进度报告已保存到: {report_path}")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
    finally:
        progress_monitor.stop_monitoring()


def demo_integrated_system():
    """演示集成的日志记录和监控系统"""
    print("\n🔗 集成系统演示")
    print("=" * 50)
    
    # 创建内存优化器（模拟）
    class MockMemoryOptimizer:
        def get_memory_status(self):
            return simulate_memory_usage(
                random.randint(1, 100), 100
            )
    
    memory_optimizer = MockMemoryOptimizer()
    
    # 创建集成系统
    log_dir = "./demo_integrated_logs"
    logging_system = LoggingSystem(log_dir=log_dir, run_name="integrated_demo")
    progress_monitor = ProgressMonitor(
        memory_optimizer=memory_optimizer,
        logging_system=logging_system,
        enable_rich_display=False
    )
    
    try:
        # 开始集成演示
        total_steps = 30
        progress_monitor.start_monitoring(total_steps)
        
        config = {
            "demo_type": "integrated",
            "total_steps": total_steps,
            "features": ["logging", "monitoring", "memory_tracking"]
        }
        logging_system.log_training_start(config)
        
        print("开始集成系统演示...")
        
        # 模拟训练过程
        for step in range(1, total_steps + 1):
            epoch, loss, learning_rate = simulate_training_step(step, total_steps)
            memory_status = simulate_memory_usage(step, total_steps)
            
            # 更新进度监控
            progress_monitor.update_progress(epoch, step, loss, learning_rate)
            
            # 记录训练指标
            metrics = TrainingMetrics(
                epoch=epoch,
                step=step,
                loss=loss,
                learning_rate=learning_rate,
                memory_usage=memory_status,
                timestamp=datetime.now()
            )
            logging_system.log_training_metrics(metrics)
            
            # 每5步显示状态
            if step % 5 == 0:
                status = progress_monitor.get_current_status()
                print(f"步骤 {step:2d}/{total_steps}: "
                      f"损失={loss:.4f}, "
                      f"内存={memory_status.allocated_gb:.1f}GB, "
                      f"进度={status['progress_percent']:.1f}%")
            
            time.sleep(0.1)
        
        # 完成演示
        progress_monitor.stop_monitoring()
        logging_system.info("集成系统演示完成", "DEMO")
        
        # 生成综合报告
        progress_summary = progress_monitor.generate_training_summary()
        log_summary = logging_system.get_log_summary()
        
        print(f"\n📋 综合报告:")
        print(f"  日志条目: {log_summary['structured_log_entries']}")
        print(f"  监控快照: {progress_summary['progress_statistics']['total_snapshots']}")
        print(f"  内存报告: {len(progress_monitor.memory_reports)}")
        
        # 保存报告
        progress_report_path = "./demo_integrated_progress.json"
        progress_monitor.save_progress_report(progress_report_path)
        print(f"  进度报告: {progress_report_path}")
        
    finally:
        progress_monitor.stop_monitoring()
        logging_system.close()
        print(f"✅ 集成系统演示完成，日志保存到: {log_dir}")


def main():
    """主函数"""
    print("🎯 Qwen3 优化微调 - 日志记录和监控系统演示")
    print("=" * 60)
    
    try:
        # 演示1: 日志记录系统
        demo_logging_system()
        
        # 演示2: 进度监控系统
        demo_progress_monitor()
        
        # 演示3: 集成系统
        demo_integrated_system()
        
        print(f"\n🎉 所有演示完成！")
        print("检查生成的日志文件和报告以查看详细信息。")
        
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()