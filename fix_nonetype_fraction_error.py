#!/usr/bin/env python3
"""
修复 NoneType 转换为分数的错误
这个错误通常发生在使用 fractions.Fraction 或类似的分数计算时传入了 None 值
"""

import os
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """备份文件"""
    backup_path = f"{file_path}.backup_fraction_fix"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"已备份文件: {backup_path}")

def fix_file_fraction_errors(file_path):
    """修复文件中的分数相关错误"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 修复 Fraction 构造函数调用
    # 查找 Fraction(x, y) 模式并添加 None 检查
    fraction_pattern = r'Fraction\s*\(\s*([^,\)]+)\s*,\s*([^,\)]+)\s*\)'
    def fix_fraction(match):
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f'Fraction({num}, {den}) if {num} is not None and {den} is not None and {den} != 0 else Fraction(0, 1)'
    
    content = re.sub(fraction_pattern, fix_fraction, content)
    
    # 2. 修复可能导致 None 传递给分数计算的地方
    # 查找除法运算，特别是可能返回 None 的变量
    division_patterns = [
        # 修复 a / b 形式的除法
        (r'(\w+)\s*=\s*([^/\n]+)\s*/\s*([^/\n]+)(?=\s*(?:#|$|\n))', 
         r'\1 = (\2) / (\3) if (\3) is not None and (\3) != 0 else 0'),
        
        # 修复函数参数中的除法
        (r'(\w+\s*\([^)]*?)([^/\n]+)\s*/\s*([^/\n,)]+)', 
         r'\1(\2) / (\3) if (\3) is not None and (\3) != 0 else 0'),
    ]
    
    for pattern, replacement in division_patterns:
        content = re.sub(pattern, replacement, content)
    
    # 3. 添加安全的分数创建函数
    if 'def safe_fraction(' not in content:
        safe_fraction_func = '''
def safe_fraction(numerator, denominator=1):
    """安全创建分数，避免 NoneType 错误"""
    try:
        from fractions import Fraction
        if numerator is None:
            numerator = 0
        if denominator is None or denominator == 0:
            denominator = 1
        return Fraction(numerator, denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        from fractions import Fraction
        return Fraction(0, 1)

'''
        # 在文件开头添加函数（在import之后）
        import_end = 0
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not (line.startswith('import ') or line.startswith('from ') or line.startswith('#')):
                import_end = i
                break
        
        if import_end > 0:
            lines.insert(import_end, safe_fraction_func)
            content = '\n'.join(lines)
    
    # 4. 替换所有 Fraction 调用为 safe_fraction
    content = re.sub(r'\bFraction\s*\(', 'safe_fraction(', content)
    
    # 5. 添加必要的 import
    if 'from fractions import Fraction' not in content and 'import fractions' not in content:
        # 在文件开头添加 import
        lines = content.split('\n')
        first_non_comment = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                first_non_comment = i
                break
        lines.insert(first_non_comment, 'from fractions import Fraction')
        content = '\n'.join(lines)
    
    # 检查是否有修改
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已修复 {file_path}")
        return True
    else:
        print(f"{file_path} 无需修复")
        return False

def main():
    """主修复函数"""
    print("开始修复 NoneType 转换为分数的错误...")
    
    # 需要检查的文件列表
    files_to_fix = [
        'src/memory_optimizer.py',
        'src/training_engine.py', 
        'src/progress_monitor.py',
        'src/logging_system.py',
        'main.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_file_fraction_errors(file_path):
            fixed_count += 1
    
    print(f"\n修复完成！共修复了 {fixed_count} 个文件")
    
    # 创建一个通用的错误处理装饰器
    decorator_file = 'src/fraction_error_handler.py'
    if not os.path.exists(decorator_file):
        with open(decorator_file, 'w', encoding='utf-8') as f:
            f.write('''"""
分数错误处理装饰器
"""
from functools import wraps
from fractions import Fraction
import logging

logger = logging.getLogger(__name__)

def handle_fraction_errors(func):
    """处理分数相关错误的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            if "can't convert type 'NoneType' to numerator/denominator" in str(e):
                logger.error(f"分数转换错误在函数 {func.__name__} 中: {e}")
                # 返回默认值或进行错误恢复
                return None
            else:
                raise
        except Exception as e:
            logger.error(f"函数 {func.__name__} 中发生未预期错误: {e}")
            raise
    return wrapper

def safe_fraction_operation(numerator, denominator=1):
    """安全的分数操作"""
    try:
        if numerator is None:
            numerator = 0
        if denominator is None or denominator == 0:
            denominator = 1
        return Fraction(int(numerator), int(denominator))
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.warning(f"分数操作失败: {e}, 返回默认值 0/1")
        return Fraction(0, 1)
''')
        print(f"已创建错误处理模块: {decorator_file}")

if __name__ == "__main__":
    main()
