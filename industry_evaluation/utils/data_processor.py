"""
数据预处理模块
"""

import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from industry_evaluation.models.data_models import Dataset, DataSample
from industry_evaluation.core.interfaces import DataProcessor


class TextDataProcessor(DataProcessor):
    """文本数据处理器"""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        初始化文本数据处理器
        
        Args:
            encoding: 文件编码
        """
        self.encoding = encoding
        self.text_cleaners = [
            self._remove_extra_whitespace,
            self._normalize_punctuation,
            self._remove_special_chars
        ]
    
    def preprocess(self, raw_data: Any) -> Dataset:
        """
        数据预处理
        
        Args:
            raw_data: 原始数据，可以是文件路径、字典或列表
            
        Returns:
            Dataset: 处理后的数据集
        """
        if isinstance(raw_data, str):
            # 文件路径
            return self._process_file(raw_data)
        elif isinstance(raw_data, dict):
            # 字典格式数据
            return self._process_dict(raw_data)
        elif isinstance(raw_data, list):
            # 列表格式数据
            return self._process_list(raw_data)
        else:
            raise ValueError(f"不支持的数据类型: {type(raw_data)}")
    
    def validate(self, data: Any) -> bool:
        """
        数据验证
        
        Args:
            data: 待验证数据
            
        Returns:
            bool: 验证结果
        """
        if isinstance(data, Dataset):
            return data.validate()
        elif isinstance(data, dict):
            return self._validate_dict(data)
        elif isinstance(data, list):
            return self._validate_list(data)
        else:
            return False
    
    def _process_file(self, file_path: str) -> Dataset:
        """处理文件数据"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            return self._process_json_file(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._process_csv_file(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._process_txt_file(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def _process_json_file(self, file_path: Path) -> Dataset:
        """处理JSON文件"""
        with open(file_path, 'r', encoding=self.encoding) as f:
            data = json.load(f)
        return self._process_dict(data)
    
    def _process_csv_file(self, file_path: Path) -> Dataset:
        """处理CSV文件"""
        samples = []
        with open(file_path, 'r', encoding=self.encoding, newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                sample = DataSample(
                    sample_id=row.get('id', f'sample_{i:04d}'),
                    input_text=self._clean_text(row.get('input', '')),
                    expected_output=self._clean_text(row.get('expected', '')),
                    context=self._parse_context(row.get('context', '{}')),
                    metadata=self._parse_metadata(row)
                )
                samples.append(sample)
        
        return Dataset(
            name=file_path.stem,
            samples=samples,
            industry_domain="未指定",
            description=f"从CSV文件 {file_path.name} 导入"
        )
    
    def _process_txt_file(self, file_path: Path) -> Dataset:
        """处理文本文件"""
        with open(file_path, 'r', encoding=self.encoding) as f:
            content = f.read()
        
        # 简单的文本分割处理
        lines = content.strip().split('\n')
        samples = []
        
        for i, line in enumerate(lines):
            if line.strip():
                # 假设每行是一个输入，期望输出为空
                sample = DataSample(
                    sample_id=f'txt_{i:04d}',
                    input_text=self._clean_text(line),
                    expected_output="",
                    context={},
                    metadata={"line_number": i + 1}
                )
                samples.append(sample)
        
        return Dataset(
            name=file_path.stem,
            samples=samples,
            industry_domain="未指定",
            description=f"从文本文件 {file_path.name} 导入"
        )
    
    def _process_dict(self, data: Dict[str, Any]) -> Dataset:
        """处理字典数据"""
        if not self._validate_dict(data):
            raise ValueError("数据格式验证失败")
        
        samples = []
        for item in data.get('samples', []):
            sample = DataSample(
                sample_id=item.get('id', item.get('sample_id', '')),
                input_text=self._clean_text(item.get('input', item.get('input_text', ''))),
                expected_output=self._clean_text(item.get('expected', item.get('expected_output', ''))),
                context=item.get('context', {}),
                metadata=item.get('metadata', {})
            )
            samples.append(sample)
        
        return Dataset(
            name=data.get('name', '未命名数据集'),
            samples=samples,
            industry_domain=data.get('industry_domain', '未指定'),
            description=data.get('description', ''),
            version=data.get('version', '1.0')
        )
    
    def _process_list(self, data: List[Any]) -> Dataset:
        """处理列表数据"""
        if not self._validate_list(data):
            raise ValueError("数据格式验证失败")
        
        samples = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                sample = DataSample(
                    sample_id=item.get('id', item.get('sample_id', f'item_{i:04d}')),
                    input_text=self._clean_text(item.get('input', item.get('input_text', ''))),
                    expected_output=self._clean_text(item.get('expected', item.get('expected_output', ''))),
                    context=item.get('context', {}),
                    metadata=item.get('metadata', {})
                )
            else:
                # 假设是简单的文本列表
                sample = DataSample(
                    sample_id=f'item_{i:04d}',
                    input_text=self._clean_text(str(item)),
                    expected_output="",
                    context={},
                    metadata={"index": i}
                )
            samples.append(sample)
        
        return Dataset(
            name="列表数据集",
            samples=samples,
            industry_domain="未指定",
            description="从列表数据导入"
        )
    
    def _validate_dict(self, data: Dict[str, Any]) -> bool:
        """验证字典数据格式"""
        if not isinstance(data, dict):
            return False
        
        # 检查必需字段
        if 'samples' not in data:
            return False
        
        if not isinstance(data['samples'], list):
            return False
        
        # 验证样本格式
        for item in data['samples']:
            if not isinstance(item, dict):
                return False
            if 'input' not in item and 'input_text' not in item:
                return False
        
        return True
    
    def _validate_list(self, data: List[Any]) -> bool:
        """验证列表数据格式"""
        if not isinstance(data, list):
            return False
        
        if len(data) == 0:
            return False
        
        return True
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            text = str(text)
        
        # 应用所有文本清理器
        for cleaner in self.text_cleaners:
            text = cleaner(text)
        
        return text.strip()
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """移除多余空白字符"""
        # 将多个空白字符替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """标准化标点符号"""
        # 中文标点符号标准化
        punctuation_map = {
            '，': ',',
            '。': '.',
            '？': '?',
            '！': '!',
            '：': ':',
            '；': ';',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'"
        }
        
        for old, new in punctuation_map.items():
            text = text.replace(old, new)
        
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        # 保留中文、英文、数字、常用标点符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:\'\"()-]', '', text)
        return text
    
    def _parse_context(self, context_str: str) -> Dict[str, Any]:
        """解析上下文字符串"""
        try:
            if isinstance(context_str, str):
                return json.loads(context_str)
            elif isinstance(context_str, dict):
                return context_str
            else:
                return {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _parse_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """解析元数据"""
        metadata = {}
        
        # 排除标准字段，其余作为元数据
        standard_fields = {'id', 'input', 'expected', 'context'}
        for key, value in row.items():
            if key not in standard_fields:
                metadata[key] = value
        
        return metadata
    
    def add_text_cleaner(self, cleaner_func):
        """添加自定义文本清理器"""
        if callable(cleaner_func):
            self.text_cleaners.append(cleaner_func)
    
    def remove_text_cleaner(self, cleaner_func):
        """移除文本清理器"""
        if cleaner_func in self.text_cleaners:
            self.text_cleaners.remove(cleaner_func)


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_sample(sample: DataSample) -> List[str]:
        """
        验证单个样本
        
        Args:
            sample: 数据样本
            
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        if not sample.sample_id:
            errors.append("样本ID不能为空")
        
        if not sample.input_text.strip():
            errors.append("输入文本不能为空")
        
        if len(sample.input_text) > 10000:
            errors.append("输入文本过长（超过10000字符）")
        
        if len(sample.expected_output) > 10000:
            errors.append("期望输出过长（超过10000字符）")
        
        return errors
    
    @staticmethod
    def validate_dataset(dataset: Dataset) -> List[str]:
        """
        验证数据集
        
        Args:
            dataset: 数据集
            
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        if not dataset.name:
            errors.append("数据集名称不能为空")
        
        if not dataset.industry_domain:
            errors.append("行业领域不能为空")
        
        if not dataset.samples:
            errors.append("数据集不能为空")
        
        # 检查样本ID唯一性
        sample_ids = [sample.sample_id for sample in dataset.samples]
        if len(sample_ids) != len(set(sample_ids)):
            errors.append("存在重复的样本ID")
        
        # 验证每个样本
        for i, sample in enumerate(dataset.samples):
            sample_errors = DataValidator.validate_sample(sample)
            for error in sample_errors:
                errors.append(f"样本{i+1}: {error}")
        
        return errors
    
    @staticmethod
    def check_data_quality(dataset: Dataset) -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            dataset: 数据集
            
        Returns:
            Dict[str, Any]: 质量报告
        """
        report = {
            "total_samples": len(dataset.samples),
            "empty_inputs": 0,
            "empty_outputs": 0,
            "avg_input_length": 0,
            "avg_output_length": 0,
            "min_input_length": float('inf'),
            "max_input_length": 0,
            "min_output_length": float('inf'),
            "max_output_length": 0,
            "unique_inputs": 0,
            "unique_outputs": 0
        }
        
        if not dataset.samples:
            return report
        
        input_lengths = []
        output_lengths = []
        inputs = set()
        outputs = set()
        
        for sample in dataset.samples:
            input_len = len(sample.input_text)
            output_len = len(sample.expected_output)
            
            input_lengths.append(input_len)
            output_lengths.append(output_len)
            
            if not sample.input_text.strip():
                report["empty_inputs"] += 1
            else:
                inputs.add(sample.input_text)
            
            if not sample.expected_output.strip():
                report["empty_outputs"] += 1
            else:
                outputs.add(sample.expected_output)
            
            report["min_input_length"] = min(report["min_input_length"], input_len)
            report["max_input_length"] = max(report["max_input_length"], input_len)
            report["min_output_length"] = min(report["min_output_length"], output_len)
            report["max_output_length"] = max(report["max_output_length"], output_len)
        
        report["avg_input_length"] = sum(input_lengths) / len(input_lengths)
        report["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        report["unique_inputs"] = len(inputs)
        report["unique_outputs"] = len(outputs)
        
        # 处理无穷大值
        if report["min_input_length"] == float('inf'):
            report["min_input_length"] = 0
        if report["min_output_length"] == float('inf'):
            report["min_output_length"] = 0
        
        return report