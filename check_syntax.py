#!/usr/bin/env python3
"""
语法检查脚本
"""

import ast
import sys
from pathlib import Path


def check_file_syntax(file_path):
    """检查单个文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析AST
        ast.parse(content)
        return True, None
        
    except SyntaxError as e:
        return False, f"语法错误在第{e.lineno}行: {e.msg}"
    except Exception as e:
        return False, f"其他错误: {str(e)}"


def main():
    """主函数"""
    print("🔍 Industry Evaluation System - 语法检查")
    print("=" * 50)
    
    # 检查关键文件
    files_to_check = [
        "industry_evaluation/models/data_models.py",
        "industry_evaluation/core/interfaces.py",
        "industry_evaluation/core/evaluation_engine.py",
        "industry_evaluation/core/batch_evaluator.py",
        "industry_evaluation/core/result_aggregator.py",
        "industry_evaluation/reporting/report_generator.py",
        "industry_evaluation/config/config_manager.py",
        "industry_evaluation/adapters/model_adapter.py",
    ]
    
    failed_files = []
    
    for file_path in files_to_check:
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ {file_path}: 文件不存在")
            failed_files.append(str(file_path))
            continue
        
        is_valid, error = check_file_syntax(file_path)
        
        if is_valid:
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: {error}")
            failed_files.append(str(file_path))
    
    print("\n" + "=" * 50)
    
    if failed_files:
        print(f"❌ {len(failed_files)} 个文件有语法错误:")
        for file_path in failed_files:
            print(f"   - {file_path}")
        return False
    else:
        print(f"✅ 所有 {len(files_to_check)} 个文件语法检查通过")
        return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 语法检查过程中发生错误: {str(e)}")
        sys.exit(1)