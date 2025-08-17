"""
基准测试管理器

实现基准数据集的下载、加载和管理，支持CLUE、FewCLUE、C-Eval等标准基准测试。
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import requests
from urllib.parse import urlparse

try:
    from datasets import Dataset, load_dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.warning("datasets library not available, some features will be disabled")

from .data_models import BenchmarkConfig, BenchmarkResult, TaskResult, convert_numpy_types
from .evaluation_engine import EvaluationEngine

logger = logging.getLogger(__name__)


class BenchmarkManager:
    """
    基准测试管理器
    
    提供基准数据集管理功能：
    - 基准数据集的下载和加载
    - CLUE、FewCLUE、C-Eval基准测试支持
    - 自定义基准数据集集成
    - 基准测试结果的标准化处理
    """
    
    def __init__(self, 
                 cache_dir: str = "./benchmarks",
                 auto_download: bool = True):
        """
        初始化基准测试管理器
        
        Args:
            cache_dir: 缓存目录
            auto_download: 是否自动下载数据集
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_download = auto_download
        
        # 预定义的基准测试配置
        self.benchmark_configs = self._load_benchmark_configs()
        
        logger.info(f"BenchmarkManager初始化完成，缓存目录: {cache_dir}")
    
    def _load_benchmark_configs(self) -> Dict[str, BenchmarkConfig]:
        """
        加载预定义的基准测试配置
        
        Returns:
            基准测试配置字典
        """
        configs = {}
        
        # CLUE基准测试
        configs["clue"] = BenchmarkConfig(
            name="CLUE",
            dataset_path="clue",
            tasks=[
                "tnews",      # 今日头条中文新闻（短文本）分类
                "afqmc",      # 蚂蚁金融语义相似度
                "cmnli",      # 中文自然语言推理
                "ocnli",      # 原创中文自然语言推理
                "wsc",        # 威诺格拉德模式挑战
                "csl"         # 中科院科学文献数据集
            ],
            evaluation_protocol="official",
            metrics=["accuracy", "f1"]
        )
        
        # FewCLUE基准测试
        configs["few_clue"] = BenchmarkConfig(
            name="FewCLUE",
            dataset_path="few_clue",
            tasks=["tnews", "afqmc", "cmnli"],
            evaluation_protocol="few_shot",
            metrics=["accuracy", "f1"],
            max_samples=1000
        )
        
        # C-Eval基准测试
        configs["c_eval"] = BenchmarkConfig(
            name="C-Eval",
            dataset_path="c_eval",
            tasks=[
                "high_school_physics",
                "high_school_chemistry", 
                "high_school_biology",
                "computer_science"
            ],
            evaluation_protocol="multiple_choice",
            metrics=["accuracy"]
        )
        
        # SuperGLUE基准测试（英文）
        configs["superglue"] = BenchmarkConfig(
            name="SuperGLUE",
            dataset_path="super_glue",
            tasks=["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"],
            evaluation_protocol="official",
            metrics=["accuracy", "f1"]
        )
        
        return configs
    
    def list_available_benchmarks(self) -> List[str]:
        """
        列出可用的基准测试
        
        Returns:
            基准测试名称列表
        """
        return list(self.benchmark_configs.keys())
    
    def get_benchmark_config(self, benchmark_name: str) -> Optional[BenchmarkConfig]:
        """
        获取基准测试配置
        
        Args:
            benchmark_name: 基准测试名称
            
        Returns:
            基准测试配置，如果不存在则返回None
        """
        return self.benchmark_configs.get(benchmark_name.lower())
    
    def load_benchmark_dataset(self, 
                             benchmark_name: str,
                             tasks: Optional[List[str]] = None,
                             split: str = "test") -> Dict[str, Dataset]:
        """
        加载基准数据集
        
        Args:
            benchmark_name: 基准测试名称
            tasks: 要加载的任务列表，如果为None则加载所有任务
            split: 数据集分割（train/validation/test）
            
        Returns:
            任务名称到数据集的映射
        """
        config = self.get_benchmark_config(benchmark_name)
        if not config:
            raise ValueError(f"未知的基准测试: {benchmark_name}")
        
        tasks_to_load = tasks or config.tasks
        datasets = {}
        
        logger.info(f"加载基准数据集: {benchmark_name}, 任务: {tasks_to_load}")
        
        for task_name in tasks_to_load:
            try:
                dataset = self._load_single_task_dataset(
                    benchmark_name, task_name, split, config
                )
                if dataset:
                    datasets[task_name] = dataset
                    logger.info(f"成功加载任务 {task_name}，样本数: {len(dataset)}")
                else:
                    logger.warning(f"无法加载任务: {task_name}")
            except Exception as e:
                logger.error(f"加载任务 {task_name} 失败: {e}")
        
        return datasets
    
    def _load_single_task_dataset(self, 
                                 benchmark_name: str,
                                 task_name: str, 
                                 split: str,
                                 config: BenchmarkConfig) -> Optional[Dataset]:
        """
        加载单个任务的数据集
        
        Args:
            benchmark_name: 基准测试名称
            task_name: 任务名称
            split: 数据集分割
            config: 基准测试配置
            
        Returns:
            数据集，如果加载失败则返回None
        """
        # 检查本地缓存
        cache_path = self.cache_dir / benchmark_name / task_name / f"{split}.json"
        
        if cache_path.exists():
            logger.info(f"从缓存加载: {cache_path}")
            return self._load_from_cache(cache_path)
        
        # 尝试从Hugging Face Hub加载
        if DATASETS_AVAILABLE and self.auto_download:
            try:
                dataset = self._load_from_huggingface(benchmark_name, task_name, split)
                if dataset:
                    # 保存到缓存
                    self._save_to_cache(dataset, cache_path)
                    return dataset
            except Exception as e:
                logger.warning(f"从Hugging Face加载失败: {e}")
        
        # 尝试从本地文件加载
        local_path = self.cache_dir / benchmark_name / task_name
        if local_path.exists():
            return self._load_from_local_files(local_path, split)
        
        logger.error(f"无法加载数据集: {benchmark_name}/{task_name}")
        return None
    
    def _load_from_huggingface(self, 
                              benchmark_name: str,
                              task_name: str, 
                              split: str) -> Optional[Dataset]:
        """
        从Hugging Face Hub加载数据集
        
        Args:
            benchmark_name: 基准测试名称
            task_name: 任务名称
            split: 数据集分割
            
        Returns:
            数据集
        """
        try:
            # 构建数据集名称
            if benchmark_name.lower() == "clue":
                dataset_name = f"clue/{task_name}"
            elif benchmark_name.lower() == "c_eval":
                dataset_name = "ceval/ceval-exam"
                # C-Eval需要特殊处理
                dataset = load_dataset(dataset_name, task_name, split=split)
                return dataset
            elif benchmark_name.lower() == "superglue":
                dataset_name = f"super_glue/{task_name}"
            else:
                dataset_name = f"{benchmark_name}/{task_name}"
            
            dataset = load_dataset(dataset_name, split=split)
            return dataset
            
        except Exception as e:
            logger.warning(f"从Hugging Face加载 {dataset_name} 失败: {e}")
            return None
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dataset]:
        """
        从缓存加载数据集
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            数据集
        """
        try:
            if DATASETS_AVAILABLE:
                return Dataset.load_from_disk(str(cache_path.parent))
            else:
                # 如果没有datasets库，尝试加载JSON
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._create_simple_dataset(data)
        except Exception as e:
            logger.error(f"从缓存加载失败: {e}")
            return None
    
    def _save_to_cache(self, dataset: Dataset, cache_path: Path):
        """
        保存数据集到缓存
        
        Args:
            dataset: 数据集
            cache_path: 缓存路径
        """
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            if DATASETS_AVAILABLE and hasattr(dataset, 'save_to_disk'):
                dataset.save_to_disk(str(cache_path.parent))
            else:
                # 保存为JSON格式
                data = []
                for item in dataset:
                    data.append(item)
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            logger.info(f"数据集已缓存到: {cache_path}")
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _load_from_local_files(self, local_path: Path, split: str) -> Optional[Dataset]:
        """
        从本地文件加载数据集
        
        Args:
            local_path: 本地路径
            split: 数据集分割
            
        Returns:
            数据集
        """
        try:
            # 查找可能的文件格式
            possible_files = [
                local_path / f"{split}.json",
                local_path / f"{split}.jsonl",
                local_path / f"{split}.csv",
                local_path / f"{split}.tsv"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    return self._load_file(file_path)
            
            logger.warning(f"在 {local_path} 中未找到 {split} 数据文件")
            return None
            
        except Exception as e:
            logger.error(f"从本地文件加载失败: {e}")
            return None
    
    def _load_file(self, file_path: Path) -> Optional[Dataset]:
        """
        加载单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据集
        """
        try:
            if DATASETS_AVAILABLE:
                if file_path.suffix == '.json':
                    return Dataset.from_json(str(file_path))
                elif file_path.suffix == '.jsonl':
                    return Dataset.from_json(str(file_path))
                elif file_path.suffix == '.csv':
                    return Dataset.from_csv(str(file_path))
                elif file_path.suffix == '.tsv':
                    return Dataset.from_csv(str(file_path), delimiter='\t')
            else:
                # 简单的JSON加载
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                
                return self._create_simple_dataset(data)
                
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {e}")
            return None
    
    def _create_simple_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        """
        创建简单的数据集对象
        
        Args:
            data: 数据列表
            
        Returns:
            数据集
        """
        if DATASETS_AVAILABLE:
            return Dataset.from_list(data)
        else:
            # 创建简单的数据集类
            class SimpleDataset:
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
                
                def __iter__(self):
                    return iter(self.data)
                
                @property
                def column_names(self):
                    if self.data:
                        return list(self.data[0].keys())
                    return []
            
            return SimpleDataset(data)
    
    def run_benchmark(self, 
                     benchmark_name: str,
                     model,
                     tokenizer,
                     evaluation_engine: EvaluationEngine,
                     tasks: Optional[List[str]] = None,
                     split: str = "test") -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            benchmark_name: 基准测试名称
            model: 要评估的模型
            tokenizer: 分词器
            evaluation_engine: 评估引擎
            tasks: 要运行的任务列表
            split: 数据集分割
            
        Returns:
            基准测试结果
        """
        logger.info(f"开始运行基准测试: {benchmark_name}")
        
        config = self.get_benchmark_config(benchmark_name)
        if not config:
            raise ValueError(f"未知的基准测试: {benchmark_name}")
        
        # 加载数据集
        datasets = self.load_benchmark_dataset(benchmark_name, tasks, split)
        if not datasets:
            raise ValueError(f"无法加载基准数据集: {benchmark_name}")
        
        # 运行评估
        evaluation_result = evaluation_engine.evaluate_model(
            model, tokenizer, datasets, f"{benchmark_name}_model"
        )
        
        # 计算基准测试特定的分数
        overall_score = self._calculate_benchmark_score(
            evaluation_result.task_results, config
        )
        
        # 获取排名信息（如果有历史数据）
        ranking_info = self._get_ranking_info(benchmark_name, overall_score)
        
        # 创建基准测试结果
        benchmark_result = BenchmarkResult(
            benchmark_name=config.name,
            model_name=evaluation_result.model_name,
            evaluation_time=evaluation_result.evaluation_time,
            task_results=evaluation_result.task_results,
            overall_score=overall_score,
            ranking_info=ranking_info
        )
        
        # 保存结果
        self._save_benchmark_result(benchmark_result)
        
        logger.info(f"基准测试完成: {benchmark_name}, 总分: {overall_score:.4f}")
        
        return benchmark_result
    
    def _calculate_benchmark_score(self, 
                                 task_results: Dict[str, TaskResult],
                                 config: BenchmarkConfig) -> float:
        """
        计算基准测试总分
        
        Args:
            task_results: 任务结果字典
            config: 基准测试配置
            
        Returns:
            总分
        """
        if not task_results:
            return 0.0
        
        scores = []
        
        for task_name, result in task_results.items():
            # 根据基准测试类型选择主要指标
            if "accuracy" in result.metrics:
                scores.append(result.metrics["accuracy"])
            elif "f1" in result.metrics:
                scores.append(result.metrics["f1"])
            elif "bleu" in result.metrics:
                scores.append(result.metrics["bleu"])
            else:
                # 使用第一个数值指标
                for metric_name, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        scores.append(value)
                        break
        
        # 计算平均分
        if scores:
            return float(sum(scores) / len(scores))
        else:
            return 0.0
    
    def _get_ranking_info(self, benchmark_name: str, score: float) -> Dict[str, Any]:
        """
        获取排名信息
        
        Args:
            benchmark_name: 基准测试名称
            score: 当前分数
            
        Returns:
            排名信息
        """
        # 这里可以实现与历史结果的比较
        # 目前返回基本信息
        return {
            "current_score": float(score),
            "benchmark": benchmark_name,
            "evaluation_date": datetime.now().isoformat(),
            "note": "排名功能需要历史数据支持"
        }
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """
        保存基准测试结果
        
        Args:
            result: 基准测试结果
        """
        try:
            results_dir = self.cache_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.benchmark_name}_{result.model_name}_{timestamp}.json"
            result_path = results_dir / filename
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result.get_summary(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"基准测试结果已保存: {result_path}")
            
        except Exception as e:
            logger.error(f"保存基准测试结果失败: {e}")
    
    def add_custom_benchmark(self, 
                           name: str,
                           config: BenchmarkConfig,
                           datasets: Dict[str, Dataset]):
        """
        添加自定义基准测试
        
        Args:
            name: 基准测试名称
            config: 基准测试配置
            datasets: 数据集字典
        """
        logger.info(f"添加自定义基准测试: {name}")
        
        # 保存配置
        self.benchmark_configs[name.lower()] = config
        
        # 保存数据集到缓存
        for task_name, dataset in datasets.items():
            cache_path = self.cache_dir / name / task_name / "test.json"
            self._save_to_cache(dataset, cache_path)
        
        logger.info(f"自定义基准测试 {name} 添加成功")
    
    def get_benchmark_history(self, benchmark_name: str) -> List[Dict[str, Any]]:
        """
        获取基准测试历史结果
        
        Args:
            benchmark_name: 基准测试名称
            
        Returns:
            历史结果列表
        """
        results = []
        results_dir = self.cache_dir / "results"
        
        if not results_dir.exists():
            return results
        
        try:
            for result_file in results_dir.glob(f"{benchmark_name}_*.json"):
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    results.append(result_data)
            
            # 按时间排序
            results.sort(key=lambda x: x.get("evaluation_time", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"获取基准测试历史失败: {e}")
        
        return results
    
    def compare_benchmark_results(self, 
                                results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        比较基准测试结果
        
        Args:
            results: 基准测试结果列表
            
        Returns:
            比较分析结果
        """
        if not results:
            return {"error": "没有提供基准测试结果"}
        
        comparison = {
            "benchmark_name": results[0].benchmark_name,
            "num_models": len(results),
            "models": [],
            "task_comparison": {},
            "overall_ranking": []
        }
        
        # 收集模型信息
        for result in results:
            model_info = {
                "name": result.model_name,
                "overall_score": result.overall_score,
                "evaluation_time": result.evaluation_time.isoformat(),
                "task_scores": {name: task.metrics.get("accuracy", task.metrics.get("f1", 0))
                              for name, task in result.task_results.items()}
            }
            comparison["models"].append(model_info)
        
        # 按总分排序
        comparison["overall_ranking"] = sorted(
            comparison["models"], 
            key=lambda x: x["overall_score"], 
            reverse=True
        )
        
        # 任务级别比较
        if results[0].task_results:
            for task_name in results[0].task_results.keys():
                task_scores = []
                for result in results:
                    if task_name in result.task_results:
                        score = result.task_results[task_name].metrics.get(
                            "accuracy", 
                            result.task_results[task_name].metrics.get("f1", 0)
                        )
                        task_scores.append({
                            "model": result.model_name,
                            "score": score
                        })
                
                # 按分数排序
                task_scores.sort(key=lambda x: x["score"], reverse=True)
                comparison["task_comparison"][task_name] = task_scores
        
        return convert_numpy_types(comparison)