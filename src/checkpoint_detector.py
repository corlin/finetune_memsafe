"""
Checkpoint检测和验证组件

本模块负责自动检测、验证和管理训练完成的checkpoint文件。
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import re

from .export_models import CheckpointInfo
from .export_exceptions import CheckpointValidationError
from .export_utils import get_file_size_mb, get_directory_size_mb, load_json_file


class CheckpointDetector:
    """检测和验证训练checkpoint的组件"""
    
    # 必需的checkpoint文件
    REQUIRED_FILES = [
        'adapter_config.json',
        'adapter_model.safetensors'
    ]
    
    # 可选的checkpoint文件
    OPTIONAL_FILES = [
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt',
        'training_args.bin',
        'trainer_state.json'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_latest_checkpoint(self, checkpoint_dir: str) -> str:
        """
        自动检测最新的checkpoint目录
        
        Args:
            checkpoint_dir: checkpoint基础目录
            
        Returns:
            str: 最新checkpoint的路径
            
        Raises:
            CheckpointValidationError: 未找到有效checkpoint时抛出
        """
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            raise CheckpointValidationError(
                f"Checkpoint目录不存在: {checkpoint_dir}",
                checkpoint_path=checkpoint_dir
            )
        
        # 如果直接是checkpoint目录（包含adapter_config.json）
        if self._is_checkpoint_directory(checkpoint_path):
            self.logger.info(f"检测到直接checkpoint目录: {checkpoint_path}")
            return str(checkpoint_path)
        
        # 搜索子目录中的checkpoint
        checkpoint_candidates = []
        
        for item in checkpoint_path.iterdir():
            if item.is_dir():
                if self._is_checkpoint_directory(item):
                    # 获取目录的修改时间
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    checkpoint_candidates.append((item, mtime))
                
                # 检查是否有numbered checkpoint (checkpoint-N)
                if re.match(r'checkpoint-\d+', item.name):
                    if self._is_checkpoint_directory(item):
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        checkpoint_candidates.append((item, mtime))
        
        if not checkpoint_candidates:
            raise CheckpointValidationError(
                f"在目录 {checkpoint_dir} 中未找到有效的checkpoint",
                checkpoint_path=checkpoint_dir,
                missing_files=self.REQUIRED_FILES
            )
        
        # 按修改时间排序，返回最新的
        checkpoint_candidates.sort(key=lambda x: x[1], reverse=True)
        latest_checkpoint = checkpoint_candidates[0][0]
        
        self.logger.info(f"检测到最新checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)
    
    def list_available_checkpoints(self, checkpoint_dir: str) -> List[str]:
        """
        列出所有可用的checkpoint
        
        Args:
            checkpoint_dir: checkpoint基础目录
            
        Returns:
            List[str]: checkpoint路径列表
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoints = []
        
        if not checkpoint_path.exists():
            return checkpoints
        
        # 检查直接checkpoint目录
        if self._is_checkpoint_directory(checkpoint_path):
            checkpoints.append(str(checkpoint_path))
        
        # 搜索子目录
        for item in checkpoint_path.iterdir():
            if item.is_dir() and self._is_checkpoint_directory(item):
                checkpoints.append(str(item))
        
        # 按名称排序
        checkpoints.sort()
        return checkpoints
    
    def validate_checkpoint_integrity(self, checkpoint_path: str) -> bool:
        """
        验证checkpoint文件的完整性
        
        Args:
            checkpoint_path: checkpoint路径
            
        Returns:
            bool: 是否完整
        """
        try:
            checkpoint_info = self.get_checkpoint_metadata(checkpoint_path)
            return checkpoint_info.is_valid
        except Exception as e:
            self.logger.error(f"验证checkpoint完整性时出错: {e}")
            return False
    
    def get_checkpoint_metadata(self, checkpoint_path: str) -> CheckpointInfo:
        """
        获取checkpoint的元数据信息
        
        Args:
            checkpoint_path: checkpoint路径
            
        Returns:
            CheckpointInfo: checkpoint信息对象
        """
        checkpoint_path = Path(checkpoint_path)
        
        # 创建基础信息对象
        checkpoint_info = CheckpointInfo(
            path=str(checkpoint_path),
            timestamp=datetime.fromtimestamp(checkpoint_path.stat().st_mtime),
            size_mb=get_directory_size_mb(str(checkpoint_path)),
            is_valid=True
        )
        
        # 验证必需文件
        missing_files = []
        for required_file in self.REQUIRED_FILES:
            file_path = checkpoint_path / required_file
            if file_path.exists():
                if required_file == 'adapter_config.json':
                    checkpoint_info.has_adapter_config = True
                elif required_file == 'adapter_model.safetensors':
                    checkpoint_info.has_adapter_model = True
            else:
                missing_files.append(required_file)
                checkpoint_info.add_validation_error(f"缺失必需文件: {required_file}")
        
        # 检查可选文件
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json']
        if any((checkpoint_path / f).exists() for f in tokenizer_files):
            checkpoint_info.has_tokenizer = True
        
        # 加载配置文件
        try:
            adapter_config_path = checkpoint_path / 'adapter_config.json'
            if adapter_config_path.exists():
                checkpoint_info.adapter_config = load_json_file(str(adapter_config_path))
                if checkpoint_info.adapter_config is None:
                    checkpoint_info.add_validation_error("adapter_config.json格式无效")
        except Exception as e:
            checkpoint_info.add_validation_error(f"读取adapter_config.json失败: {e}")
        
        try:
            training_args_path = checkpoint_path / 'training_args.bin'
            if training_args_path.exists():
                # 尝试加载training_args（通常是pickle格式）
                # 这里我们只记录文件存在，不解析内容
                checkpoint_info.training_args = {'file_exists': True}
        except Exception as e:
            self.logger.warning(f"无法读取training_args.bin: {e}")
        
        # 验证adapter配置
        if checkpoint_info.adapter_config:
            self._validate_adapter_config(checkpoint_info)
        
        # 验证文件大小
        self._validate_file_sizes(checkpoint_path, checkpoint_info)
        
        return checkpoint_info
    
    def _is_checkpoint_directory(self, path: Path) -> bool:
        """检查是否为有效的checkpoint目录"""
        if not path.is_dir():
            return False
        
        # 检查必需文件是否存在
        for required_file in self.REQUIRED_FILES:
            if not (path / required_file).exists():
                return False
        
        return True
    
    def _validate_adapter_config(self, checkpoint_info: CheckpointInfo):
        """验证adapter配置的有效性"""
        config = checkpoint_info.adapter_config
        
        # 检查必需的配置字段
        required_fields = ['peft_type', 'task_type', 'r', 'lora_alpha']
        for field in required_fields:
            if field not in config:
                checkpoint_info.add_validation_error(f"adapter_config.json缺失字段: {field}")
        
        # 验证PEFT类型
        if config.get('peft_type') != 'LORA':
            checkpoint_info.add_validation_error(
                f"不支持的PEFT类型: {config.get('peft_type')}, 仅支持LORA"
            )
        
        # 验证LoRA参数
        r = config.get('r', 0)
        if not isinstance(r, int) or r <= 0:
            checkpoint_info.add_validation_error(f"无效的LoRA rank值: {r}")
        
        lora_alpha = config.get('lora_alpha', 0)
        if not isinstance(lora_alpha, (int, float)) or lora_alpha <= 0:
            checkpoint_info.add_validation_error(f"无效的LoRA alpha值: {lora_alpha}")
        
        # 检查目标模块
        target_modules = config.get('target_modules', [])
        if not target_modules:
            checkpoint_info.add_validation_error("target_modules为空")
    
    def _validate_file_sizes(self, checkpoint_path: Path, checkpoint_info: CheckpointInfo):
        """验证文件大小的合理性"""
        # 检查adapter_model.safetensors大小
        adapter_model_path = checkpoint_path / 'adapter_model.safetensors'
        if adapter_model_path.exists():
            size_mb = get_file_size_mb(str(adapter_model_path))
            
            # LoRA适配器文件通常应该在几MB到几百MB之间
            if size_mb < 0.1:
                checkpoint_info.add_validation_error(
                    f"adapter_model.safetensors文件过小: {size_mb:.1f}MB"
                )
            elif size_mb > 5000:  # 5GB
                checkpoint_info.add_validation_error(
                    f"adapter_model.safetensors文件过大: {size_mb:.1f}MB"
                )
        
        # 检查总目录大小
        if checkpoint_info.size_mb > 10000:  # 10GB
            checkpoint_info.add_validation_error(
                f"Checkpoint目录过大: {checkpoint_info.size_mb:.1f}MB"
            )
    
    def find_best_checkpoint(self, checkpoint_dir: str, criteria: str = 'latest') -> str:
        """
        根据指定标准查找最佳checkpoint
        
        Args:
            checkpoint_dir: checkpoint基础目录
            criteria: 选择标准 ('latest', 'largest', 'smallest')
            
        Returns:
            str: 最佳checkpoint路径
        """
        checkpoints = self.list_available_checkpoints(checkpoint_dir)
        
        if not checkpoints:
            raise CheckpointValidationError(
                f"在目录 {checkpoint_dir} 中未找到任何checkpoint",
                checkpoint_path=checkpoint_dir
            )
        
        if criteria == 'latest':
            return self.detect_latest_checkpoint(checkpoint_dir)
        
        elif criteria == 'largest':
            largest_checkpoint = None
            largest_size = 0
            
            for checkpoint_path in checkpoints:
                size = get_directory_size_mb(checkpoint_path)
                if size > largest_size:
                    largest_size = size
                    largest_checkpoint = checkpoint_path
            
            return largest_checkpoint
        
        elif criteria == 'smallest':
            smallest_checkpoint = None
            smallest_size = float('inf')
            
            for checkpoint_path in checkpoints:
                size = get_directory_size_mb(checkpoint_path)
                if size < smallest_size:
                    smallest_size = size
                    smallest_checkpoint = checkpoint_path
            
            return smallest_checkpoint
        
        else:
            raise ValueError(f"不支持的选择标准: {criteria}")
    
    def get_checkpoint_summary(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        获取checkpoint的摘要信息
        
        Args:
            checkpoint_path: checkpoint路径
            
        Returns:
            Dict[str, Any]: 摘要信息
        """
        checkpoint_info = self.get_checkpoint_metadata(checkpoint_path)
        
        summary = {
            'path': checkpoint_info.path,
            'timestamp': checkpoint_info.timestamp.isoformat(),
            'size_mb': checkpoint_info.size_mb,
            'is_valid': checkpoint_info.is_valid,
            'has_adapter_model': checkpoint_info.has_adapter_model,
            'has_adapter_config': checkpoint_info.has_adapter_config,
            'has_tokenizer': checkpoint_info.has_tokenizer,
            'validation_errors': checkpoint_info.validation_errors
        }
        
        # 添加adapter配置摘要
        if checkpoint_info.adapter_config:
            summary['adapter_config'] = {
                'peft_type': checkpoint_info.adapter_config.get('peft_type'),
                'task_type': checkpoint_info.adapter_config.get('task_type'),
                'r': checkpoint_info.adapter_config.get('r'),
                'lora_alpha': checkpoint_info.adapter_config.get('lora_alpha'),
                'target_modules': checkpoint_info.adapter_config.get('target_modules', [])
            }
        
        return summary
    
    def compare_checkpoints(self, checkpoint_paths: List[str]) -> Dict[str, Any]:
        """
        比较多个checkpoint
        
        Args:
            checkpoint_paths: checkpoint路径列表
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        comparison = {
            'checkpoints': [],
            'summary': {
                'total_count': len(checkpoint_paths),
                'valid_count': 0,
                'total_size_mb': 0
            }
        }
        
        for checkpoint_path in checkpoint_paths:
            try:
                info = self.get_checkpoint_metadata(checkpoint_path)
                checkpoint_data = {
                    'path': info.path,
                    'is_valid': info.is_valid,
                    'size_mb': info.size_mb,
                    'timestamp': info.timestamp.isoformat(),
                    'error_count': len(info.validation_errors)
                }
                
                comparison['checkpoints'].append(checkpoint_data)
                
                if info.is_valid:
                    comparison['summary']['valid_count'] += 1
                
                comparison['summary']['total_size_mb'] += info.size_mb
                
            except Exception as e:
                self.logger.error(f"比较checkpoint {checkpoint_path} 时出错: {e}")
                comparison['checkpoints'].append({
                    'path': checkpoint_path,
                    'is_valid': False,
                    'error': str(e)
                })
        
        return comparison