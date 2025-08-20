"""
批量评估功能
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
from industry_evaluation.core.interfaces import EvaluationConfig, EvaluationResult
from industry_evaluation.core.evaluation_engine import IndustryEvaluationEngine, EvaluationStatus


@dataclass
class BatchEvaluationConfig:
    """批量评估配置"""
    batch_size: int = 100
    max_concurrent_tasks: int = 4
    max_concurrent_models: int = 2
    chunk_size: int = 1000
    save_intermediate_results: bool = True
    intermediate_results_dir: str = "./batch_results"
    resume_from_checkpoint: bool = True
    timeout_per_batch: int = 3600  # 1小时
    memory_limit_mb: int = 8192  # 8GB
    enable_parallel_processing: bool = True
    process_pool_size: Optional[int] = None


@dataclass
class BatchTask:
    """批量任务"""
    task_id: str
    model_ids: List[str]
    dataset_path: str
    evaluation_config: EvaluationConfig
    batch_config: BatchEvaluationConfig
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_samples: int = 0
    processed_samples: int = 0
    failed_samples: int = 0
    results: Dict[str, EvaluationResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class DatasetLoader:
    """数据集加载器"""
    
    @staticmethod
    def load_dataset(dataset_path: str, chunk_size: Optional[int] = None) -> Iterator[List[Dict[str, Any]]]:
        """
        加载数据集
        
        Args:
            dataset_path: 数据集路径
            chunk_size: 分块大小
            
        Yields:
            List[Dict[str, Any]]: 数据块
        """
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        if path.suffix.lower() == '.json':
            yield from DatasetLoader._load_json_dataset(path, chunk_size)
        elif path.suffix.lower() == '.jsonl':
            yield from DatasetLoader._load_jsonl_dataset(path, chunk_size)
        elif path.suffix.lower() == '.csv':
            yield from DatasetLoader._load_csv_dataset(path, chunk_size)
        else:
            raise ValueError(f"不支持的数据集格式: {path.suffix}")
    
    @staticmethod
    def _load_json_dataset(path: Path, chunk_size: Optional[int]) -> Iterator[List[Dict[str, Any]]]:
        """加载JSON数据集"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON数据集必须是数组格式")
        
        if chunk_size is None:
            yield data
        else:
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
    
    @staticmethod
    def _load_jsonl_dataset(path: Path, chunk_size: Optional[int]) -> Iterator[List[Dict[str, Any]]]:
        """加载JSONL数据集"""
        chunk = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        chunk.append(item)
                        
                        if chunk_size and len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    except json.JSONDecodeError as e:
                        logging.warning(f"跳过无效JSON行: {line[:100]}...")
        
        if chunk:
            yield chunk
    
    @staticmethod
    def _load_csv_dataset(path: Path, chunk_size: Optional[int]) -> Iterator[List[Dict[str, Any]]]:
        """加载CSV数据集"""
        import pandas as pd
        
        if chunk_size is None:
            df = pd.read_csv(path)
            yield df.to_dict('records')
        else:
            for chunk_df in pd.read_csv(path, chunksize=chunk_size):
                yield chunk_df.to_dict('records')
    
    @staticmethod
    def count_samples(dataset_path: str) -> int:
        """
        计算数据集样本数量
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            int: 样本数量
        """
        path = Path(dataset_path)
        
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1
        elif path.suffix.lower() == '.jsonl':
            count = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        elif path.suffix.lower() == '.csv':
            import pandas as pd
            return len(pd.read_csv(path))
        else:
            raise ValueError(f"不支持的数据集格式: {path.suffix}")


class BatchResultManager:
    """批量结果管理器"""
    
    def __init__(self, results_dir: str):
        """
        初始化结果管理器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_intermediate_result(self, 
                               task_id: str, 
                               model_id: str, 
                               batch_index: int, 
                               result: EvaluationResult):
        """
        保存中间结果
        
        Args:
            task_id: 任务ID
            model_id: 模型ID
            batch_index: 批次索引
            result: 评估结果
        """
        result_file = self.results_dir / f"{task_id}_{model_id}_batch_{batch_index}.json"
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"保存中间结果: {result_file}")
        except Exception as e:
            self.logger.error(f"保存中间结果失败: {str(e)}")
    
    def load_intermediate_results(self, task_id: str, model_id: str) -> List[EvaluationResult]:
        """
        加载中间结果
        
        Args:
            task_id: 任务ID
            model_id: 模型ID
            
        Returns:
            List[EvaluationResult]: 中间结果列表
        """
        results = []
        pattern = f"{task_id}_{model_id}_batch_*.json"
        
        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    result = EvaluationResult.from_dict(result_data)
                    results.append(result)
            except Exception as e:
                self.logger.error(f"加载中间结果失败 {result_file}: {str(e)}")
        
        return results
    
    def save_final_result(self, task_id: str, batch_task: BatchTask):
        """
        保存最终结果
        
        Args:
            task_id: 任务ID
            batch_task: 批量任务
        """
        result_file = self.results_dir / f"{task_id}_final.json"
        
        try:
            result_data = {
                "task_id": batch_task.task_id,
                "model_ids": batch_task.model_ids,
                "dataset_path": batch_task.dataset_path,
                "status": batch_task.status,
                "created_at": batch_task.created_at,
                "started_at": batch_task.started_at,
                "completed_at": batch_task.completed_at,
                "total_samples": batch_task.total_samples,
                "processed_samples": batch_task.processed_samples,
                "failed_samples": batch_task.failed_samples,
                "results": {model_id: result.to_dict() for model_id, result in batch_task.results.items()},
                "errors": batch_task.errors
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"保存最终结果: {result_file}")
        except Exception as e:
            self.logger.error(f"保存最终结果失败: {str(e)}")
    
    def cleanup_intermediate_results(self, task_id: str):
        """
        清理中间结果
        
        Args:
            task_id: 任务ID
        """
        pattern = f"{task_id}_*_batch_*.json"
        
        for result_file in self.results_dir.glob(pattern):
            try:
                result_file.unlink()
                self.logger.debug(f"删除中间结果: {result_file}")
            except Exception as e:
                self.logger.error(f"删除中间结果失败 {result_file}: {str(e)}")


class BatchEvaluator:
    """批量评估器"""
    
    def __init__(self, evaluation_engine: IndustryEvaluationEngine):
        """
        初始化批量评估器
        
        Args:
            evaluation_engine: 评估引擎
        """
        self.evaluation_engine = evaluation_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 任务管理
        self.batch_tasks: Dict[str, BatchTask] = {}
        self.result_manager: Optional[BatchResultManager] = None
        
        # 线程池
        self.thread_executor = ThreadPoolExecutor(max_workers=8)
        self.process_executor: Optional[ProcessPoolExecutor] = None
    
    def create_batch_task(self,
                         task_id: str,
                         model_ids: List[str],
                         dataset_path: str,
                         evaluation_config: EvaluationConfig,
                         batch_config: BatchEvaluationConfig) -> BatchTask:
        """
        创建批量任务
        
        Args:
            task_id: 任务ID
            model_ids: 模型ID列表
            dataset_path: 数据集路径
            evaluation_config: 评估配置
            batch_config: 批量配置
            
        Returns:
            BatchTask: 批量任务
        """
        # 计算总样本数
        total_samples = DatasetLoader.count_samples(dataset_path)
        
        batch_task = BatchTask(
            task_id=task_id,
            model_ids=model_ids,
            dataset_path=dataset_path,
            evaluation_config=evaluation_config,
            batch_config=batch_config,
            total_samples=total_samples
        )
        
        self.batch_tasks[task_id] = batch_task
        
        # 初始化结果管理器
        if batch_config.save_intermediate_results:
            self.result_manager = BatchResultManager(batch_config.intermediate_results_dir)
        
        self.logger.info(f"创建批量任务: {task_id}, 模型数: {len(model_ids)}, 样本数: {total_samples}")
        
        return batch_task
    
    def start_batch_evaluation(self, task_id: str) -> bool:
        """
        启动批量评估
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 启动是否成功
        """
        if task_id not in self.batch_tasks:
            self.logger.error(f"批量任务不存在: {task_id}")
            return False
        
        batch_task = self.batch_tasks[task_id]
        
        if batch_task.status != "pending":
            self.logger.error(f"批量任务状态不正确: {batch_task.status}")
            return False
        
        # 启动批量评估
        if batch_task.batch_config.enable_parallel_processing:
            future = self.thread_executor.submit(self._execute_parallel_batch_evaluation, batch_task)
        else:
            future = self.thread_executor.submit(self._execute_sequential_batch_evaluation, batch_task)
        
        batch_task.status = "running"
        batch_task.started_at = time.time()
        
        self.logger.info(f"启动批量评估: {task_id}")
        return True
    
    def _execute_sequential_batch_evaluation(self, batch_task: BatchTask):
        """
        执行顺序批量评估
        
        Args:
            batch_task: 批量任务
        """
        try:
            self.logger.info(f"开始顺序批量评估: {batch_task.task_id}")
            
            # 加载数据集
            dataset_chunks = list(DatasetLoader.load_dataset(
                batch_task.dataset_path, 
                batch_task.batch_config.chunk_size
            ))
            
            # 逐个模型评估
            for model_id in batch_task.model_ids:
                self.logger.info(f"评估模型: {model_id}")
                
                model_results = []
                
                # 逐批次评估
                for batch_index, data_chunk in enumerate(dataset_chunks):
                    try:
                        # 创建评估任务
                        eval_task_id = self.evaluation_engine.evaluate_model(
                            model_id=model_id,
                            dataset=data_chunk,
                            evaluation_config=batch_task.evaluation_config
                        )
                        
                        # 等待评估完成
                        result = self._wait_for_evaluation_completion(
                            eval_task_id, 
                            batch_task.batch_config.timeout_per_batch
                        )
                        
                        if result:
                            model_results.append(result)
                            
                            # 保存中间结果
                            if self.result_manager:
                                self.result_manager.save_intermediate_result(
                                    batch_task.task_id, 
                                    model_id, 
                                    batch_index, 
                                    result
                                )
                            
                            batch_task.processed_samples += len(data_chunk)
                        else:
                            batch_task.failed_samples += len(data_chunk)
                            batch_task.errors.append(f"模型 {model_id} 批次 {batch_index} 评估失败")
                        
                        self.logger.info(f"完成批次 {batch_index + 1}/{len(dataset_chunks)}")
                        
                    except Exception as e:
                        self.logger.error(f"批次 {batch_index} 评估失败: {str(e)}")
                        batch_task.failed_samples += len(data_chunk)
                        batch_task.errors.append(f"批次 {batch_index} 错误: {str(e)}")
                
                # 合并模型结果
                if model_results:
                    merged_result = self._merge_evaluation_results(model_results)
                    batch_task.results[model_id] = merged_result
            
            # 完成任务
            batch_task.status = "completed"
            batch_task.completed_at = time.time()
            
            # 保存最终结果
            if self.result_manager:
                self.result_manager.save_final_result(batch_task.task_id, batch_task)
                
                # 清理中间结果
                self.result_manager.cleanup_intermediate_results(batch_task.task_id)
            
            self.logger.info(f"批量评估完成: {batch_task.task_id}")
            
        except Exception as e:
            batch_task.status = "failed"
            batch_task.completed_at = time.time()
            batch_task.errors.append(f"批量评估失败: {str(e)}")
            
            self.logger.error(f"批量评估失败: {batch_task.task_id}, 错误: {str(e)}")
    
    def _execute_parallel_batch_evaluation(self, batch_task: BatchTask):
        """
        执行并行批量评估
        
        Args:
            batch_task: 批量任务
        """
        try:
            self.logger.info(f"开始并行批量评估: {batch_task.task_id}")
            
            # 初始化进程池
            if not self.process_executor:
                pool_size = batch_task.batch_config.process_pool_size or min(
                    batch_task.batch_config.max_concurrent_models,
                    mp.cpu_count()
                )
                self.process_executor = ProcessPoolExecutor(max_workers=pool_size)
            
            # 加载数据集
            dataset_chunks = list(DatasetLoader.load_dataset(
                batch_task.dataset_path, 
                batch_task.batch_config.chunk_size
            ))
            
            # 创建并行任务
            futures = []
            
            for model_id in batch_task.model_ids:
                for batch_index, data_chunk in enumerate(dataset_chunks):
                    future = self.thread_executor.submit(
                        self._evaluate_batch,
                        batch_task.task_id,
                        model_id,
                        batch_index,
                        data_chunk,
                        batch_task.evaluation_config,
                        batch_task.batch_config.timeout_per_batch
                    )
                    futures.append((future, model_id, batch_index, len(data_chunk)))
            
            # 收集结果
            model_results: Dict[str, List[EvaluationResult]] = {
                model_id: [] for model_id in batch_task.model_ids
            }
            
            for future, model_id, batch_index, chunk_size in futures:
                try:
                    result = future.result(timeout=batch_task.batch_config.timeout_per_batch)
                    
                    if result:
                        model_results[model_id].append(result)
                        
                        # 保存中间结果
                        if self.result_manager:
                            self.result_manager.save_intermediate_result(
                                batch_task.task_id, 
                                model_id, 
                                batch_index, 
                                result
                            )
                        
                        batch_task.processed_samples += chunk_size
                    else:
                        batch_task.failed_samples += chunk_size
                        batch_task.errors.append(f"模型 {model_id} 批次 {batch_index} 评估失败")
                    
                except Exception as e:
                    self.logger.error(f"并行任务失败: {str(e)}")
                    batch_task.failed_samples += chunk_size
                    batch_task.errors.append(f"模型 {model_id} 批次 {batch_index} 错误: {str(e)}")
            
            # 合并每个模型的结果
            for model_id, results in model_results.items():
                if results:
                    merged_result = self._merge_evaluation_results(results)
                    batch_task.results[model_id] = merged_result
            
            # 完成任务
            batch_task.status = "completed"
            batch_task.completed_at = time.time()
            
            # 保存最终结果
            if self.result_manager:
                self.result_manager.save_final_result(batch_task.task_id, batch_task)
                self.result_manager.cleanup_intermediate_results(batch_task.task_id)
            
            self.logger.info(f"并行批量评估完成: {batch_task.task_id}")
            
        except Exception as e:
            batch_task.status = "failed"
            batch_task.completed_at = time.time()
            batch_task.errors.append(f"并行批量评估失败: {str(e)}")
            
            self.logger.error(f"并行批量评估失败: {batch_task.task_id}, 错误: {str(e)}")
    
    def _evaluate_batch(self,
                       task_id: str,
                       model_id: str,
                       batch_index: int,
                       data_chunk: List[Dict[str, Any]],
                       evaluation_config: EvaluationConfig,
                       timeout: int) -> Optional[EvaluationResult]:
        """
        评估单个批次
        
        Args:
            task_id: 任务ID
            model_id: 模型ID
            batch_index: 批次索引
            data_chunk: 数据块
            evaluation_config: 评估配置
            timeout: 超时时间
            
        Returns:
            Optional[EvaluationResult]: 评估结果
        """
        try:
            # 创建评估任务
            eval_task_id = self.evaluation_engine.evaluate_model(
                model_id=model_id,
                dataset=data_chunk,
                evaluation_config=evaluation_config
            )
            
            # 等待评估完成
            result = self._wait_for_evaluation_completion(eval_task_id, timeout)
            
            return result
            
        except Exception as e:
            self.logger.error(f"批次评估失败 {model_id}_{batch_index}: {str(e)}")
            return None
    
    def _wait_for_evaluation_completion(self, 
                                      eval_task_id: str, 
                                      timeout: int) -> Optional[EvaluationResult]:
        """
        等待评估完成
        
        Args:
            eval_task_id: 评估任务ID
            timeout: 超时时间
            
        Returns:
            Optional[EvaluationResult]: 评估结果
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            progress = self.evaluation_engine.get_evaluation_progress(eval_task_id)
            
            if progress and progress.status == "completed":
                return self.evaluation_engine.get_evaluation_result(eval_task_id)
            elif progress and progress.status == "failed":
                self.logger.error(f"评估任务失败: {eval_task_id}")
                return None
            
            time.sleep(1)  # 等待1秒后重新检查
        
        # 超时，取消任务
        self.evaluation_engine.cancel_evaluation(eval_task_id)
        self.logger.error(f"评估任务超时: {eval_task_id}")
        return None
    
    def _merge_evaluation_results(self, results: List[EvaluationResult]) -> EvaluationResult:
        """
        合并评估结果
        
        Args:
            results: 结果列表
            
        Returns:
            EvaluationResult: 合并后的结果
        """
        if not results:
            raise ValueError("结果列表不能为空")
        
        if len(results) == 1:
            return results[0]
        
        # 合并详细结果
        all_detailed_results = []
        for result in results:
            all_detailed_results.extend(result.detailed_results)
        
        # 计算综合评分
        total_samples = len(all_detailed_results)
        dimension_scores = {}
        
        if total_samples > 0:
            # 计算各维度平均分
            for result in results:
                for dimension, score in result.dimension_scores.items():
                    if dimension not in dimension_scores:
                        dimension_scores[dimension] = []
                    dimension_scores[dimension].append(score * len(result.detailed_results))
            
            for dimension in dimension_scores:
                dimension_scores[dimension] = sum(dimension_scores[dimension]) / total_samples
            
            # 计算综合评分
            overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        else:
            overall_score = 0.0
        
        # 合并错误分析
        merged_error_analysis = self._merge_error_analysis([result.error_analysis for result in results])
        
        # 合并改进建议
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.improvement_suggestions)
        
        # 去重改进建议
        unique_suggestions = list(set(all_suggestions))
        
        return EvaluationResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            detailed_results=all_detailed_results,
            error_analysis=merged_error_analysis,
            improvement_suggestions=unique_suggestions
        )
    
    def _merge_error_analysis(self, error_analyses: List[Any]) -> Any:
        """
        合并错误分析
        
        Args:
            error_analyses: 错误分析列表
            
        Returns:
            Any: 合并后的错误分析
        """
        # 这里需要根据ErrorAnalysis的具体实现来合并
        # 暂时返回第一个非空的错误分析
        for analysis in error_analyses:
            if analysis:
                return analysis
        return None
    
    def get_batch_task_status(self, task_id: str) -> Optional[BatchTask]:
        """
        获取批量任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[BatchTask]: 任务状态
        """
        return self.batch_tasks.get(task_id)
    
    def cancel_batch_task(self, task_id: str) -> bool:
        """
        取消批量任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 取消是否成功
        """
        if task_id not in self.batch_tasks:
            return False
        
        batch_task = self.batch_tasks[task_id]
        
        if batch_task.status not in ["pending", "running"]:
            return False
        
        batch_task.status = "cancelled"
        batch_task.completed_at = time.time()
        
        self.logger.info(f"取消批量任务: {task_id}")
        return True
    
    def list_batch_tasks(self) -> List[BatchTask]:
        """
        列出所有批量任务
        
        Returns:
            List[BatchTask]: 任务列表
        """
        return list(self.batch_tasks.values())
    
    def shutdown(self):
        """关闭批量评估器"""
        self.logger.info("关闭批量评估器")
        
        self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)