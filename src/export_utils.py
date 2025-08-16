"""
模型导出系统的工具函数

本模块提供模型导出过程中使用的各种工具函数和辅助类。
"""

import os
import shutil
import psutil
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import uuid

from .export_exceptions import DiskSpaceError, MemoryError


def generate_export_id() -> str:
    """生成唯一的导出ID"""
    return f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def format_timestamp(dt: datetime = None) -> str:
    """格式化时间戳"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y%m%d_%H%M%S')


def get_file_size_mb(file_path: str) -> float:
    """获取文件大小（MB）"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def get_directory_size_mb(directory: str) -> float:
    """获取目录大小（MB）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    continue
        return total_size / (1024 * 1024)
    except OSError:
        return 0.0


def check_disk_space(path: str, required_gb: float) -> Tuple[bool, float]:
    """
    检查磁盘空间是否足够
    
    Args:
        path: 检查路径
        required_gb: 需要的空间（GB）
        
    Returns:
        Tuple[bool, float]: (是否足够, 可用空间GB)
    """
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024 ** 3)
        return available_gb >= required_gb, available_gb
    except OSError:
        return False, 0.0


def check_memory_usage() -> Dict[str, float]:
    """
    检查当前内存使用情况
    
    Returns:
        Dict[str, float]: 内存使用信息（GB）
    """
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024 ** 3),
        'available_gb': memory.available / (1024 ** 3),
        'used_gb': memory.used / (1024 ** 3),
        'percent': memory.percent
    }


def ensure_memory_available(required_gb: float):
    """
    确保有足够的内存可用
    
    Args:
        required_gb: 需要的内存（GB）
        
    Raises:
        MemoryError: 内存不足时抛出
    """
    memory_info = check_memory_usage()
    if memory_info['available_gb'] < required_gb:
        raise MemoryError(
            f"内存不足，需要 {required_gb:.1f}GB，可用 {memory_info['available_gb']:.1f}GB",
            required_memory_gb=required_gb,
            available_memory_gb=memory_info['available_gb']
        )


def ensure_disk_space(path: str, required_gb: float):
    """
    确保有足够的磁盘空间
    
    Args:
        path: 检查路径
        required_gb: 需要的空间（GB）
        
    Raises:
        DiskSpaceError: 磁盘空间不足时抛出
    """
    sufficient, available_gb = check_disk_space(path, required_gb)
    if not sufficient:
        raise DiskSpaceError(
            f"磁盘空间不足，需要 {required_gb:.1f}GB，可用 {available_gb:.1f}GB",
            required_space_gb=required_gb,
            available_space_gb=available_gb
        )


def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ('md5', 'sha1', 'sha256')
        
    Returns:
        str: 哈希值
    """
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except OSError:
        return ""


def safe_remove_file(file_path: str) -> bool:
    """
    安全删除文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否成功删除
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except OSError:
        return False


def safe_remove_directory(directory: str) -> bool:
    """
    安全删除目录
    
    Args:
        directory: 目录路径
        
    Returns:
        bool: 是否成功删除
    """
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        return True
    except OSError:
        return False


def create_directory_structure(base_path: str, subdirs: List[str]) -> Dict[str, str]:
    """
    创建目录结构
    
    Args:
        base_path: 基础路径
        subdirs: 子目录列表
        
    Returns:
        Dict[str, str]: 目录名到完整路径的映射
    """
    paths = {}
    base_path = Path(base_path)
    
    # 创建基础目录
    base_path.mkdir(parents=True, exist_ok=True)
    paths['base'] = str(base_path)
    
    # 创建子目录
    for subdir in subdirs:
        subdir_path = base_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = str(subdir_path)
    
    return paths


def backup_file(file_path: str, backup_suffix: str = '.backup') -> Optional[str]:
    """
    备份文件
    
    Args:
        file_path: 原文件路径
        backup_suffix: 备份文件后缀
        
    Returns:
        Optional[str]: 备份文件路径，失败时返回None
    """
    try:
        if not os.path.exists(file_path):
            return None
        
        backup_path = f"{file_path}{backup_suffix}"
        shutil.copy2(file_path, backup_path)
        return backup_path
    except OSError:
        return None


def restore_from_backup(original_path: str, backup_path: str) -> bool:
    """
    从备份恢复文件
    
    Args:
        original_path: 原文件路径
        backup_path: 备份文件路径
        
    Returns:
        bool: 是否成功恢复
    """
    try:
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, original_path)
            return True
        return False
    except OSError:
        return False


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进级别
        
    Returns:
        bool: 是否成功保存
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except (OSError, TypeError):
        return False


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        Optional[Dict[str, Any]]: 加载的数据，失败时返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def format_duration(seconds: float) -> str:
    """
    格式化持续时间
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def format_size(size_mb: float) -> str:
    """
    格式化文件大小
    
    Args:
        size_mb: 大小（MB）
        
    Returns:
        str: 格式化的大小字符串
    """
    if size_mb < 1024:
        return f"{size_mb:.1f}MB"
    else:
        size_gb = size_mb / 1024
        return f"{size_gb:.1f}GB"


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger('model_export')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_model_name(model_name: str) -> bool:
    """
    验证模型名称格式
    
    Args:
        model_name: 模型名称
        
    Returns:
        bool: 是否有效
    """
    if not model_name:
        return False
    
    # 检查是否包含无效字符
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    return not any(char in model_name for char in invalid_chars)


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除无效字符
    
    Args:
        filename: 原文件名
        
    Returns:
        str: 清理后的文件名
    """
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        Dict[str, Any]: 系统信息
    """
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage('.')
    
    return {
        'platform': os.name,
        'cpu_count': psutil.cpu_count(),
        'memory': {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'percent': memory.percent
        },
        'disk': {
            'total_gb': disk.total / (1024 ** 3),
            'free_gb': disk.free / (1024 ** 3),
            'used_gb': disk.used / (1024 ** 3)
        }
    }


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int, description: str = "处理中"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
    
    def update(self, step: int = None, description: str = None):
        """更新进度"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if description:
            self.description = description
        
        progress = (self.current_step / self.total_steps) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = format_duration(eta)
        else:
            eta_str = "未知"
        
        self.logger.info(
            f"{self.description}: {self.current_step}/{self.total_steps} "
            f"({progress:.1f}%) - 预计剩余时间: {eta_str}"
        )
    
    def finish(self, description: str = "完成"):
        """完成进度跟踪"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{description} - 总耗时: {format_duration(total_time)}")


class TempFileManager:
    """临时文件管理器"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), 'temp_export')
        self.temp_files = []
        self.temp_dirs = []
        
        # 创建临时目录
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'temp_') -> str:
        """创建临时文件"""
        filename = f"{prefix}{uuid.uuid4().hex[:8]}{suffix}"
        filepath = os.path.join(self.base_dir, filename)
        self.temp_files.append(filepath)
        return filepath
    
    def create_temp_dir(self, prefix: str = 'temp_dir_') -> str:
        """创建临时目录"""
        dirname = f"{prefix}{uuid.uuid4().hex[:8]}"
        dirpath = os.path.join(self.base_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        self.temp_dirs.append(dirpath)
        return dirpath
    
    def cleanup(self):
        """清理所有临时文件和目录"""
        # 删除临时文件
        for filepath in self.temp_files:
            safe_remove_file(filepath)
        
        # 删除临时目录
        for dirpath in self.temp_dirs:
            safe_remove_directory(dirpath)
        
        # 删除基础临时目录（如果为空）
        try:
            if os.path.exists(self.base_dir) and not os.listdir(self.base_dir):
                os.rmdir(self.base_dir)
        except OSError:
            pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()