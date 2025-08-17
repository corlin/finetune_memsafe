"""
数据管道集成

与DataPipeline数据管道集成，支持数据拆分功能、数据质量分析和缓存重用机制。
"""

import logging
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import json

from datasets import Dataset, DatasetDict

from .data_splitter import DataSplitter
from .quality_analyzer import QualityAnalyzer
from .data_models import DataSplitResult, DataQualityReport, convert_numpy_types

logger = logging.getLogger(__name__)


class DataPipelineIntegration:
    """
    数据管道集成器
    
    提供与现有数据管道的集成功能：
    - 修改DataPipeline类支持数据拆分功能
    - 集成数据质量分析到数据加载流程
    - 实现拆分数据的缓存和重用机制
    - 建立与现有数据格式的兼容性
    """
    
    def __init__(self, 
                 cache_dir: str = "./data_cache",
                 enable_quality_analysis: bool = True,
                 enable_caching: bool = True):
        """
        初始化数据管道集成器
        
        Args:
            cache_dir: 缓存目录
            enable_quality_analysis: 是否启用质量分析
            enable_caching: 是否启用缓存
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_quality_analysis = enable_quality_analysis
        self.enable_caching = enable_caching
        
        # 初始化组件
        self.data_splitter = None
        self.quality_analyzer = QualityAnalyzer() if enable_quality_analysis else None
        
        # 缓存管理
        self.cache_index = self._load_cache_index()
        
        logger.info(f"DataPipelineIntegration初始化完成，缓存目录: {cache_dir}")
    
    def enhance_data_pipeline(self, data_pipeline_class):
        """
        增强现有的DataPipeline类
        
        Args:
            data_pipeline_class: 现有的DataPipeline类
            
        Returns:
            增强后的DataPipeline类
        """
        class EnhancedDataPipeline(data_pipeline_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.integration = DataPipelineIntegration(
                    cache_dir=kwargs.get('cache_dir', './data_cache'),
                    enable_quality_analysis=kwargs.get('enable_quality_analysis', True),
                    enable_caching=kwargs.get('enable_caching', True)
                )
            
            def load_and_split_data(self, 
                                  data_path: str,
                                  train_ratio: float = 0.7,
                                  val_ratio: float = 0.15,
                                  test_ratio: float = 0.15,
                                  stratify_by: Optional[str] = None,
                                  random_seed: int = 42,
                                  force_reload: bool = False) -> DataSplitResult:
                """
                加载数据并进行拆分
                
                Args:
                    data_path: 数据路径
                    train_ratio: 训练集比例
                    val_ratio: 验证集比例
                    test_ratio: 测试集比例
                    stratify_by: 分层字段
                    random_seed: 随机种子
                    force_reload: 是否强制重新加载
                    
                Returns:
                    数据拆分结果
                """
                return self.integration.load_and_split_data(
                    data_path=data_path,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    stratify_by=stratify_by,
                    random_seed=random_seed,
                    force_reload=force_reload
                )
            
            def analyze_data_quality(self, 
                                   dataset: Union[Dataset, str],
                                   dataset_name: str = "dataset") -> DataQualityReport:
                """
                分析数据质量
                
                Args:
                    dataset: 数据集或数据路径
                    dataset_name: 数据集名称
                    
                Returns:
                    数据质量报告
                """
                return self.integration.analyze_data_quality(dataset, dataset_name)
            
            def get_cached_splits(self, data_path: str, config_hash: str) -> Optional[DataSplitResult]:
                """
                获取缓存的数据拆分结果
                
                Args:
                    data_path: 数据路径
                    config_hash: 配置哈希
                    
                Returns:
                    缓存的拆分结果
                """
                return self.integration.get_cached_splits(data_path, config_hash)
            
            def clear_cache(self, data_path: Optional[str] = None):
                """
                清理缓存
                
                Args:
                    data_path: 特定数据路径，如果为None则清理所有缓存
                """
                self.integration.clear_cache(data_path)
        
        return EnhancedDataPipeline
    
    def load_and_split_data(self, 
                           data_path: str,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           stratify_by: Optional[str] = None,
                           random_seed: int = 42,
                           force_reload: bool = False) -> DataSplitResult:
        """
        加载数据并进行拆分
        
        Args:
            data_path: 数据路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify_by: 分层字段
            random_seed: 随机种子
            force_reload: 是否强制重新加载
            
        Returns:
            数据拆分结果
        """
        logger.info(f"加载并拆分数据: {data_path}")
        
        # 生成配置哈希
        config = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "stratify_by": stratify_by,
            "random_seed": random_seed
        }
        config_hash = self._generate_config_hash(config)
        
        # 检查缓存
        if self.enable_caching and not force_reload:
            cached_result = self.get_cached_splits(data_path, config_hash)
            if cached_result:
                logger.info("使用缓存的数据拆分结果")
                return cached_result
        
        # 加载原始数据
        dataset = self._load_dataset(data_path)
        
        # 数据质量分析
        quality_report = None
        if self.enable_quality_analysis:
            quality_report = self.analyze_data_quality(dataset, Path(data_path).stem)
        
        # 初始化数据拆分器
        if self.data_splitter is None:
            self.data_splitter = DataSplitter(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                stratify_by=stratify_by,
                random_seed=random_seed,
                enable_quality_analysis=self.enable_quality_analysis
            )
        else:
            # 更新配置
            self.data_splitter.train_ratio = train_ratio
            self.data_splitter.val_ratio = val_ratio
            self.data_splitter.test_ratio = test_ratio
            self.data_splitter.stratify_by = stratify_by
            self.data_splitter.random_seed = random_seed
        
        # 执行数据拆分
        split_result = self.data_splitter.split_data(dataset)
        
        # 缓存结果
        if self.enable_caching:
            self._cache_split_result(data_path, config_hash, split_result, quality_report)
        
        return split_result
    
    def analyze_data_quality(self, 
                           dataset: Union[Dataset, str],
                           dataset_name: str = "dataset") -> Optional[DataQualityReport]:
        """
        分析数据质量
        
        Args:
            dataset: 数据集或数据路径
            dataset_name: 数据集名称
            
        Returns:
            数据质量报告
        """
        if not self.enable_quality_analysis or not self.quality_analyzer:
            return None
        
        # 如果是路径，先加载数据
        if isinstance(dataset, str):
            dataset = self._load_dataset(dataset)
        
        logger.info(f"分析数据质量: {dataset_name}")
        return self.quality_analyzer.analyze_data_quality(dataset, dataset_name)
    
    def get_cached_splits(self, data_path: str, config_hash: str) -> Optional[DataSplitResult]:
        """
        获取缓存的数据拆分结果
        
        Args:
            data_path: 数据路径
            config_hash: 配置哈希
            
        Returns:
            缓存的拆分结果
        """
        if not self.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(data_path, config_hash)
        
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            cache_file = self.cache_dir / cache_info["cache_file"]
            
            if cache_file.exists():
                try:
                    # 检查数据文件是否有更新
                    data_file = Path(data_path)
                    if data_file.exists():
                        data_mtime = data_file.stat().st_mtime
                        cache_mtime = cache_info["created_time"]
                        
                        if data_mtime > cache_mtime:
                            logger.info("数据文件已更新，缓存失效")
                            return None
                    
                    # 加载缓存结果
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    logger.info(f"加载缓存的拆分结果: {cache_key}")
                    return cached_data["split_result"]
                    
                except Exception as e:
                    logger.warning(f"加载缓存失败: {e}")
                    # 删除损坏的缓存
                    self._remove_cache_entry(cache_key)
        
        return None
    
    def _load_dataset(self, data_path: str) -> Dataset:
        """
        加载数据集
        
        Args:
            data_path: 数据路径
            
        Returns:
            数据集
        """
        data_path = Path(data_path)
        
        try:
            if data_path.is_dir():
                # 检查是否是Hugging Face数据集格式
                if (data_path / "dataset_info.json").exists() or (data_path / "state.json").exists():
                    return Dataset.load_from_disk(str(data_path))
                else:
                    # 处理包含多个文件的目录
                    import glob
                    
                    # 查找所有支持的文件
                    supported_extensions = [".json", ".jsonl", ".csv", ".tsv", ".txt", ".md"]
                    files = []
                    
                    for ext in supported_extensions:
                        files.extend(glob.glob(str(data_path / f"*{ext}")))
                    
                    if not files:
                        raise ValueError(f"在目录 {data_path} 中未找到支持的数据文件")
                    
                    # 读取所有文件内容
                    all_data = []
                    for file_path in files:
                        file_data = self._load_single_file(file_path)
                        all_data.extend(file_data)
                    
                    return Dataset.from_list(all_data)
            
            elif data_path.is_file():
                # 单个文件
                file_data = self._load_single_file(str(data_path))
                return Dataset.from_list(file_data)
            
            else:
                raise ValueError(f"数据路径不存在: {data_path}")
                
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def _load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据列表
        """
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        return [data]
            
            elif file_path.suffix.lower() == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                return data
            
            elif file_path.suffix.lower() in ['.csv', '.tsv']:
                import pandas as pd
                delimiter = '\t' if file_path.suffix.lower() == '.tsv' else ','
                df = pd.read_csv(file_path, delimiter=delimiter)
                return df.to_dict('records')
            
            elif file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return [{"text": content, "filename": file_path.name}]
            
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return []
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """
        生成配置哈希
        
        Args:
            config: 配置字典
            
        Returns:
            配置哈希
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _generate_cache_key(self, data_path: str, config_hash: str) -> str:
        """
        生成缓存键
        
        Args:
            data_path: 数据路径
            config_hash: 配置哈希
            
        Returns:
            缓存键
        """
        path_hash = hashlib.md5(str(Path(data_path).absolute()).encode()).hexdigest()
        return f"{path_hash}_{config_hash}"
    
    def _cache_split_result(self, 
                           data_path: str, 
                           config_hash: str, 
                           split_result: DataSplitResult,
                           quality_report: Optional[DataQualityReport]):
        """
        缓存拆分结果
        
        Args:
            data_path: 数据路径
            config_hash: 配置哈希
            split_result: 拆分结果
            quality_report: 质量报告
        """
        try:
            cache_key = self._generate_cache_key(data_path, config_hash)
            cache_filename = f"split_{cache_key}.pkl"
            cache_file = self.cache_dir / cache_filename
            
            # 准备缓存数据
            cache_data = {
                "split_result": split_result,
                "quality_report": quality_report,
                "data_path": data_path,
                "config_hash": config_hash,
                "created_time": datetime.now().timestamp()
            }
            
            # 保存到文件
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 更新缓存索引
            self.cache_index[cache_key] = {
                "cache_file": cache_filename,
                "data_path": data_path,
                "config_hash": config_hash,
                "created_time": cache_data["created_time"],
                "file_size": cache_file.stat().st_size
            }
            
            # 保存缓存索引
            self._save_cache_index()
            
            logger.info(f"缓存拆分结果: {cache_key}")
            
        except Exception as e:
            logger.error(f"缓存拆分结果失败: {e}")
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """
        加载缓存索引
        
        Returns:
            缓存索引字典
        """
        index_file = self.cache_dir / "cache_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}")
        
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        try:
            index_file = self.cache_dir / "cache_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """
        删除缓存条目
        
        Args:
            cache_key: 缓存键
        """
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            cache_file = self.cache_dir / cache_info["cache_file"]
            
            # 删除缓存文件
            if cache_file.exists():
                cache_file.unlink()
            
            # 从索引中删除
            del self.cache_index[cache_key]
            self._save_cache_index()
            
            logger.info(f"删除缓存条目: {cache_key}")
    
    def clear_cache(self, data_path: Optional[str] = None):
        """
        清理缓存
        
        Args:
            data_path: 特定数据路径，如果为None则清理所有缓存
        """
        if data_path is None:
            # 清理所有缓存
            for cache_file in self.cache_dir.glob("split_*.pkl"):
                cache_file.unlink()
            
            self.cache_index.clear()
            self._save_cache_index()
            
            logger.info("已清理所有缓存")
        else:
            # 清理特定数据路径的缓存
            path_hash = hashlib.md5(str(Path(data_path).absolute()).encode()).hexdigest()
            
            keys_to_remove = []
            for cache_key, cache_info in self.cache_index.items():
                if cache_key.startswith(path_hash):
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                self._remove_cache_entry(cache_key)
            
            logger.info(f"已清理数据路径 {data_path} 的缓存")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        total_files = len(self.cache_index)
        total_size = sum(info.get("file_size", 0) for info in self.cache_index.values())
        
        # 按数据路径分组
        path_groups = {}
        for cache_key, cache_info in self.cache_index.items():
            data_path = cache_info.get("data_path", "unknown")
            if data_path not in path_groups:
                path_groups[data_path] = {"count": 0, "size": 0}
            
            path_groups[data_path]["count"] += 1
            path_groups[data_path]["size"] += cache_info.get("file_size", 0)
        
        return {
            "total_cached_files": total_files,
            "total_cache_size_bytes": total_size,
            "total_cache_size_mb": total_size / (1024 * 1024),
            "cache_by_data_path": path_groups,
            "cache_directory": str(self.cache_dir)
        }
    
    def optimize_cache(self, max_size_mb: float = 1000, max_age_days: int = 30):
        """
        优化缓存
        
        Args:
            max_size_mb: 最大缓存大小（MB）
            max_age_days: 最大缓存年龄（天）
        """
        logger.info("开始优化缓存")
        
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_days * 24 * 3600
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # 收集需要删除的缓存
        to_remove = []
        total_size = 0
        
        # 按创建时间排序（最旧的优先删除）
        sorted_cache = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].get("created_time", 0)
        )
        
        for cache_key, cache_info in sorted_cache:
            cache_age = current_time - cache_info.get("created_time", current_time)
            cache_size = cache_info.get("file_size", 0)
            
            # 检查年龄
            if cache_age > max_age_seconds:
                to_remove.append(cache_key)
                continue
            
            # 检查总大小
            if total_size + cache_size > max_size_bytes:
                to_remove.append(cache_key)
                continue
            
            total_size += cache_size
        
        # 删除过期或超大小的缓存
        for cache_key in to_remove:
            self._remove_cache_entry(cache_key)
        
        logger.info(f"缓存优化完成，删除了 {len(to_remove)} 个缓存条目")
    
    def create_data_format_adapter(self, format_name: str, loader_func: callable):
        """
        创建数据格式适配器
        
        Args:
            format_name: 格式名称
            loader_func: 加载函数
        """
        # 这里可以扩展支持更多数据格式
        # 将加载函数注册到格式映射中
        if not hasattr(self, '_format_adapters'):
            self._format_adapters = {}
        
        self._format_adapters[format_name] = loader_func
        logger.info(f"注册数据格式适配器: {format_name}")
    
    def validate_data_compatibility(self, dataset: Dataset) -> Dict[str, Any]:
        """
        验证数据兼容性
        
        Args:
            dataset: 数据集
            
        Returns:
            兼容性报告
        """
        compatibility_report = {
            "is_compatible": True,
            "issues": [],
            "recommendations": [],
            "dataset_info": {
                "num_samples": len(dataset),
                "column_names": dataset.column_names if hasattr(dataset, 'column_names') else [],
                "features": str(dataset.features) if hasattr(dataset, 'features') else "unknown"
            }
        }
        
        try:
            # 检查必要的列
            required_columns = ["text", "input", "sentence"]
            has_text_column = any(col in dataset.column_names for col in required_columns)
            
            if not has_text_column:
                compatibility_report["is_compatible"] = False
                compatibility_report["issues"].append("缺少文本列（text, input, sentence）")
                compatibility_report["recommendations"].append("确保数据集包含文本内容列")
            
            # 检查数据类型
            if hasattr(dataset, 'features'):
                for column, feature in dataset.features.items():
                    if 'string' not in str(feature).lower() and 'text' in column.lower():
                        compatibility_report["issues"].append(f"列 {column} 可能不是文本类型")
            
            # 检查数据完整性
            if len(dataset) == 0:
                compatibility_report["is_compatible"] = False
                compatibility_report["issues"].append("数据集为空")
            
            # 检查样本格式
            if len(dataset) > 0:
                sample = dataset[0]
                if not isinstance(sample, dict):
                    compatibility_report["issues"].append("样本格式不是字典类型")
        
        except Exception as e:
            compatibility_report["is_compatible"] = False
            compatibility_report["issues"].append(f"兼容性检查失败: {e}")
        
        return compatibility_report


def create_enhanced_data_pipeline(base_pipeline_class=None):
    """
    创建增强的数据管道类
    
    Args:
        base_pipeline_class: 基础数据管道类，如果为None则创建新类
        
    Returns:
        增强的数据管道类
    """
    if base_pipeline_class is None:
        # 创建基础数据管道类
        class BaseDataPipeline:
            def __init__(self, **kwargs):
                self.config = kwargs
            
            def load_data(self, data_path: str):
                """基础数据加载方法"""
                pass
    
        base_pipeline_class = BaseDataPipeline
    
    # 使用集成器增强数据管道
    integration = DataPipelineIntegration()
    enhanced_class = integration.enhance_data_pipeline(base_pipeline_class)
    
    return enhanced_class