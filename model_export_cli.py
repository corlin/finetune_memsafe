#!/usr/bin/env python3
"""
模型导出CLI工具主入口

使用方法:
    python model_export_cli.py export --checkpoint-path qwen3-finetuned
    python model_export_cli.py config create-template
    python model_export_cli.py wizard
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from cli_standalone import main

if __name__ == '__main__':
    sys.exit(main())