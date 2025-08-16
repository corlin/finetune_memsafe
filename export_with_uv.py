#!/usr/bin/env python3
"""
使用uv运行Qwen3模型导出

直接调用现有的基本导出示例
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 使用uv运行Qwen3模型导出")
    print("=" * 50)
    
    # 检查文件存在
    export_example = Path("examples/basic_export_example.py")
    if not export_example.exists():
        print(f"❌ 错误: 找不到 {export_example}")
        sys.exit(1)
    
    checkpoint_path = Path("qwen3-finetuned/checkpoint-30")
    if not checkpoint_path.exists():
        print(f"❌ 错误: Checkpoint目录不存在: {checkpoint_path}")
        sys.exit(1)
    
    print("✅ 文件检查通过")
    print("📦 使用uv运行导出...")
    
    try:
        # 使用uv运行基本导出示例
        cmd = ['uv', 'run', 'python', str(export_example)]
        result = subprocess.run(cmd, check=True)
        
        print("\n🎉 导出完成！")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 导出失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
        sys.exit(1)

if __name__ == "__main__":
    main()