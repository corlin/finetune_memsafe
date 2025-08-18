"""
实验跟踪器

实现实验管理系统，包括实验配置和结果的持久化存储、元数据管理、版本控制等功能。
"""

import logging
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import sqlite3
import pickle

from .data_models import (
    ExperimentConfig, EvaluationResult, BenchmarkResult, 
    ComparisonResult, convert_numpy_types
)

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    实验跟踪器
    
    提供实验管理功能：
    - 实验配置和结果的持久化存储
    - 实验元数据管理（模型参数、超参数、数据集信息）
    - 实验唯一标识和版本控制
    - 实验结果的查询和检索功能
    """
    
    def __init__(self, 
                 experiment_dir: str = "./experiments",
                 db_path: Optional[str] = None):
        """
        初始化实验跟踪器
        
        Args:
            experiment_dir: 实验结果目录
            db_path: 数据库路径，如果为None则使用默认路径
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据库路径
        if db_path is None:
            db_path = self.experiment_dir / "experiments.db"
        self.db_path = Path(db_path)
        
        # 初始化数据库
        self._init_database()
        
        logger.info(f"ExperimentTracker初始化完成，实验目录: {experiment_dir}")
    
    def _init_database(self):
        """初始化数据库表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建实验表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS experiments (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        config_hash TEXT,
                        status TEXT DEFAULT 'created',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        tags TEXT,
                        metadata TEXT
                    )
                ''')
                
                # 创建实验结果表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS experiment_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id TEXT,
                        result_type TEXT,
                        result_data TEXT,
                        metrics TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                    )
                ''')
                
                # 创建实验文件表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS experiment_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id TEXT,
                        file_path TEXT,
                        file_type TEXT,
                        file_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments (name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment_id ON experiment_results (experiment_id)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
            raise
    
    def create_experiment(self, 
                         name: str,
                         config: ExperimentConfig,
                         description: str = "",
                         tags: Optional[List[str]] = None) -> str:
        """
        创建新实验
        
        Args:
            name: 实验名称
            config: 实验配置
            description: 实验描述
            tags: 标签列表
            
        Returns:
            实验ID
        """
        # 生成唯一实验ID
        experiment_id = str(uuid.uuid4())
        
        # 计算配置哈希
        config_hash = self._calculate_config_hash(config)
        
        # 准备标签
        tags_str = json.dumps(tags or [])
        
        # 准备元数据
        metadata = {
            "model_config": config.model_config,
            "training_config": config.training_config,
            "evaluation_config": config.evaluation_config.to_dict(),
            "data_config": config.data_config
        }
        metadata_str = json.dumps(convert_numpy_types(metadata))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO experiments 
                    (id, name, description, config_hash, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (experiment_id, name, description, config_hash, tags_str, metadata_str))
                conn.commit()
            
            # 保存配置文件
            self._save_experiment_config(experiment_id, config)
            
            logger.info(f"创建实验成功: {name} (ID: {experiment_id})")
            return experiment_id
            
        except Exception as e:
            logger.error(f"创建实验失败: {e}")
            raise
    
    def track_experiment(self, 
                        config: ExperimentConfig,
                        result: Union[EvaluationResult, BenchmarkResult]) -> str:
        """
        跟踪实验和结果
        
        Args:
            config: 实验配置
            result: 实验结果
            
        Returns:
            实验ID
        """
        # 创建实验
        experiment_id = self.create_experiment(
            name=config.experiment_name,
            config=config,
            description=config.description,
            tags=config.tags
        )
        
        # 记录结果
        result_type = "evaluation" if isinstance(result, EvaluationResult) else "benchmark"
        self.log_result(experiment_id, result, result_type)
        
        return experiment_id

    def log_result(self, 
                  experiment_id: str,
                  result: Union[EvaluationResult, BenchmarkResult],
                  result_type: str = "evaluation"):
        """
        记录实验结果
        
        Args:
            experiment_id: 实验ID
            result: 实验结果
            result_type: 结果类型
        """
        try:
            # 序列化结果数据
            if hasattr(result, 'get_summary'):
                result_data = result.get_summary()
            else:
                result_data = result.__dict__
            
            result_data_str = json.dumps(convert_numpy_types(result_data))
            
            # 提取关键指标
            metrics = {}
            if hasattr(result, 'metrics'):
                metrics = result.metrics
            elif hasattr(result, 'overall_score'):
                metrics = {"overall_score": result.overall_score}
            
            metrics_str = json.dumps(convert_numpy_types(metrics))
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO experiment_results 
                    (experiment_id, result_type, result_data, metrics)
                    VALUES (?, ?, ?, ?)
                ''', (experiment_id, result_type, result_data_str, metrics_str))
                conn.commit()
            
            # 保存详细结果文件
            self._save_result_file(experiment_id, result, result_type)
            
            # 更新实验状态
            self._update_experiment_status(experiment_id, "completed")
            
            logger.info(f"记录实验结果成功: {experiment_id}")
            
        except Exception as e:
            logger.error(f"记录实验结果失败: {e}")
            raise
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        获取实验信息
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验信息字典
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, name, description, config_hash, status, 
                           created_at, updated_at, tags, metadata
                    FROM experiments WHERE id = ?
                ''', (experiment_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                experiment = {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "config_hash": row[3],
                    "status": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "tags": json.loads(row[7]) if row[7] else [],
                    "metadata": json.loads(row[8]) if row[8] else {}
                }
                
                # 获取实验结果
                cursor.execute('''
                    SELECT result_type, metrics, created_at
                    FROM experiment_results WHERE experiment_id = ?
                    ORDER BY created_at DESC
                ''', (experiment_id,))
                
                results = []
                for result_row in cursor.fetchall():
                    results.append({
                        "result_type": result_row[0],
                        "metrics": json.loads(result_row[1]) if result_row[1] else {},
                        "created_at": result_row[2]
                    })
                
                experiment["results"] = results
                
                return experiment
                
        except Exception as e:
            logger.error(f"获取实验信息失败: {e}")
            return None
    
    def list_experiments(self, 
                        tags: Optional[List[str]] = None,
                        status: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Dict[str, Any]]:
        """
        列出实验
        
        Args:
            tags: 标签过滤
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            实验列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if tags:
                    # 简单的标签匹配（实际应用中可能需要更复杂的逻辑）
                    for tag in tags:
                        conditions.append("tags LIKE ?")
                        params.append(f"%{tag}%")
                
                if status:
                    conditions.append("status = ?")
                    params.append(status)
                
                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)
                
                query = f'''
                    SELECT id, name, description, status, created_at, updated_at, tags
                    FROM experiments {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                '''
                
                params.extend([limit, offset])
                cursor.execute(query, params)
                
                experiments = []
                for row in cursor.fetchall():
                    experiments.append({
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "status": row[3],
                        "created_at": row[4],
                        "updated_at": row[5],
                        "tags": json.loads(row[6]) if row[6] else []
                    })
                
                return experiments
                
        except Exception as e:
            logger.error(f"列出实验失败: {e}")
            return []
    
    def search_experiments(self, 
                          query: str,
                          search_fields: List[str] = ["name", "description"]) -> List[Dict[str, Any]]:
        """
        搜索实验
        
        Args:
            query: 搜索查询
            search_fields: 搜索字段
            
        Returns:
            匹配的实验列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建搜索条件
                conditions = []
                params = []
                
                for field in search_fields:
                    if field in ["name", "description"]:
                        conditions.append(f"{field} LIKE ?")
                        params.append(f"%{query}%")
                
                if not conditions:
                    return []
                
                where_clause = "WHERE " + " OR ".join(conditions)
                
                sql_query = f'''
                    SELECT id, name, description, status, created_at, updated_at, tags
                    FROM experiments {where_clause}
                    ORDER BY created_at DESC
                '''
                
                cursor.execute(sql_query, params)
                
                experiments = []
                for row in cursor.fetchall():
                    experiments.append({
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "status": row[3],
                        "created_at": row[4],
                        "updated_at": row[5],
                        "tags": json.loads(row[6]) if row[6] else []
                    })
                
                return experiments
                
        except Exception as e:
            logger.error(f"搜索实验失败: {e}")
            return []
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        删除实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否删除成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 删除实验结果
                cursor.execute('DELETE FROM experiment_results WHERE experiment_id = ?', (experiment_id,))
                
                # 删除实验文件记录
                cursor.execute('DELETE FROM experiment_files WHERE experiment_id = ?', (experiment_id,))
                
                # 删除实验
                cursor.execute('DELETE FROM experiments WHERE id = ?', (experiment_id,))
                
                conn.commit()
            
            # 删除实验文件夹
            experiment_dir = self.experiment_dir / experiment_id
            if experiment_dir.exists():
                import shutil
                shutil.rmtree(experiment_dir)
            
            logger.info(f"删除实验成功: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除实验失败: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        获取实验的所有结果
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            结果列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT result_type, result_data, metrics, created_at
                    FROM experiment_results WHERE experiment_id = ?
                    ORDER BY created_at DESC
                ''', (experiment_id,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "result_type": row[0],
                        "result_data": json.loads(row[1]) if row[1] else {},
                        "metrics": json.loads(row[2]) if row[2] else {},
                        "created_at": row[3]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"获取实验结果失败: {e}")
            return []
    
    def compare_experiments(self, experiment_ids: List[str]) -> ComparisonResult:
        """
        比较多个实验
        
        Args:
            experiment_ids: 实验ID列表
            
        Returns:
            比较结果
        """
        try:
            experiments_data = []
            
            for exp_id in experiment_ids:
                experiment = self.get_experiment(exp_id)
                if experiment:
                    experiments_data.append(experiment)
            
            if not experiments_data:
                return ComparisonResult(
                    models=[],
                    metrics={},
                    statistical_tests={},
                    rankings={},
                    best_model={}
                )
            
            # 提取模型名称和指标
            models = [exp["name"] for exp in experiments_data]
            metrics = {}
            
            # 收集所有指标
            all_metric_names = set()
            for exp in experiments_data:
                for result in exp.get("results", []):
                    all_metric_names.update(result.get("metrics", {}).keys())
            
            # 为每个指标收集所有实验的值
            for metric_name in all_metric_names:
                metric_values = []
                for exp in experiments_data:
                    # 获取最新结果的指标值
                    latest_result = exp.get("results", [{}])[0] if exp.get("results") else {}
                    metric_value = latest_result.get("metrics", {}).get(metric_name, 0.0)
                    metric_values.append(float(metric_value))
                
                metrics[metric_name] = metric_values
            
            # 计算排名
            rankings = {}
            for metric_name, values in metrics.items():
                # 按值排序（降序）
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
                rankings[metric_name] = [models[i] for i in sorted_indices]
            
            # 找到最佳模型
            best_model = {}
            for metric_name, model_list in rankings.items():
                if model_list:
                    best_model[metric_name] = model_list[0]
            
            # 简单的统计检验（这里可以扩展更复杂的统计分析）
            statistical_tests = {
                "comparison_type": "descriptive",
                "num_experiments": len(experiments_data),
                "metrics_compared": list(all_metric_names)
            }
            
            return ComparisonResult(
                models=models,
                metrics=metrics,
                statistical_tests=statistical_tests,
                rankings=rankings,
                best_model=best_model
            )
            
        except Exception as e:
            logger.error(f"比较实验失败: {e}")
            return ComparisonResult(
                models=[],
                metrics={},
                statistical_tests={"error": str(e)},
                rankings={},
                best_model={}
            )
    
    def export_experiment(self, experiment_id: str, export_path: str) -> bool:
        """
        导出实验
        
        Args:
            experiment_id: 实验ID
            export_path: 导出路径
            
        Returns:
            是否导出成功
        """
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                logger.error(f"实验不存在: {experiment_id}")
                return False
            
            # 获取实验结果
            results = self.get_experiment_results(experiment_id)
            experiment["detailed_results"] = results
            
            # 保存到文件
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(experiment), f, indent=2, ensure_ascii=False)
            
            logger.info(f"导出实验成功: {experiment_id} -> {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出实验失败: {e}")
            return False
    
    def import_experiment(self, import_path: str) -> Optional[str]:
        """
        导入实验
        
        Args:
            import_path: 导入路径
            
        Returns:
            新实验ID，如果失败则返回None
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                experiment_data = json.load(f)
            
            # 创建新的实验ID
            new_experiment_id = str(uuid.uuid4())
            
            # 更新实验数据
            experiment_data["id"] = new_experiment_id
            experiment_data["name"] = f"{experiment_data['name']}_imported"
            
            # 插入实验记录
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO experiments 
                    (id, name, description, config_hash, status, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    new_experiment_id,
                    experiment_data["name"],
                    experiment_data.get("description", ""),
                    experiment_data.get("config_hash", ""),
                    experiment_data.get("status", "imported"),
                    json.dumps(experiment_data.get("tags", [])),
                    json.dumps(experiment_data.get("metadata", {}))
                ))
                
                # 插入结果记录
                for result in experiment_data.get("detailed_results", []):
                    cursor.execute('''
                        INSERT INTO experiment_results 
                        (experiment_id, result_type, result_data, metrics)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        new_experiment_id,
                        result.get("result_type", "unknown"),
                        json.dumps(result.get("result_data", {})),
                        json.dumps(result.get("metrics", {}))
                    ))
                
                conn.commit()
            
            logger.info(f"导入实验成功: {new_experiment_id}")
            return new_experiment_id
            
        except Exception as e:
            logger.error(f"导入实验失败: {e}")
            return None
    
    def _calculate_config_hash(self, config: ExperimentConfig) -> str:
        """
        计算配置哈希
        
        Args:
            config: 实验配置
            
        Returns:
            配置哈希值
        """
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_experiment_config(self, experiment_id: str, config: ExperimentConfig):
        """
        保存实验配置
        
        Args:
            experiment_id: 实验ID
            config: 实验配置
        """
        experiment_dir = self.experiment_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        config_path = experiment_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 记录文件
        self._record_file(experiment_id, str(config_path), "config", config_path.stat().st_size)
    
    def _save_result_file(self, 
                         experiment_id: str, 
                         result: Union[EvaluationResult, BenchmarkResult],
                         result_type: str):
        """
        保存结果文件
        
        Args:
            experiment_id: 实验ID
            result: 结果对象
            result_type: 结果类型
        """
        experiment_dir = self.experiment_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = experiment_dir / f"{result_type}_{timestamp}.pkl"
        
        # 保存为pickle文件以保持对象完整性
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        
        # 记录文件
        self._record_file(experiment_id, str(result_path), result_type, result_path.stat().st_size)
    
    def _record_file(self, experiment_id: str, file_path: str, file_type: str, file_size: int):
        """
        记录文件信息
        
        Args:
            experiment_id: 实验ID
            file_path: 文件路径
            file_type: 文件类型
            file_size: 文件大小
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO experiment_files 
                    (experiment_id, file_path, file_type, file_size)
                    VALUES (?, ?, ?, ?)
                ''', (experiment_id, file_path, file_type, file_size))
                conn.commit()
        except Exception as e:
            logger.warning(f"记录文件信息失败: {e}")
    
    def _update_experiment_status(self, experiment_id: str, status: str):
        """
        更新实验状态
        
        Args:
            experiment_id: 实验ID
            status: 新状态
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE experiments 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, experiment_id))
                conn.commit()
        except Exception as e:
            logger.warning(f"更新实验状态失败: {e}")
    
    def export_results(self, 
                      output_path: str,
                      format: str = "csv") -> str:
        """
        导出结果
        
        Args:
            output_path: 输出路径
            format: 导出格式
            
        Returns:
            导出文件路径
        """
        try:
            # 获取所有实验
            experiments = self.list_experiments()
            
            if format.lower() == "csv":
                return self._export_to_csv(experiments, output_path)
            elif format.lower() == "json":
                return self._export_to_json(experiments, output_path)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
                
        except Exception as e:
            logger.error(f"导出结果失败: {e}")
            return ""
    
    def _export_to_csv(self, experiments: List[Dict[str, Any]], output_path: str) -> str:
        """导出为CSV格式"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                "实验ID", "实验名称", "状态", "创建时间", 
                "标签", "描述", "结果类型", "指标数量", "主要指标"
            ])
            
            # 写入数据
            for exp in experiments:
                full_exp = self.get_experiment(exp['id'])
                if full_exp:
                    results = full_exp.get("results", [])
                    if results:
                        latest_result = results[0]
                        result_type = latest_result.get("result_type", "")
                        metrics = latest_result.get("metrics", {})
                        num_metrics = len(metrics)
                        main_metric = list(metrics.keys())[0] if metrics else ""
                        main_value = metrics.get(main_metric, "") if main_metric else ""
                    else:
                        result_type = ""
                        num_metrics = 0
                        main_metric = ""
                        main_value = ""
                    
                    writer.writerow([
                        full_exp["id"],
                        full_exp["name"],
                        full_exp["status"],
                        full_exp["created_at"],
                        ";".join(full_exp.get("tags", [])),
                        full_exp.get("description", ""),
                        result_type,
                        num_metrics,
                        f"{main_metric}: {main_value}" if main_metric else ""
                    ])
        
        logger.info(f"CSV导出完成: {output_path}")
        return output_path
    
    def _export_to_json(self, experiments: List[Dict[str, Any]], output_path: str) -> str:
        """导出为JSON格式"""
        # 获取完整的实验数据
        full_experiments = []
        for exp in experiments:
            full_exp = self.get_experiment(exp['id'])
            if full_exp:
                full_experiments.append(full_exp)
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_experiments": len(full_experiments),
            "experiments": full_experiments
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(export_data), f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON导出完成: {output_path}")
        return output_path

    def generate_leaderboard(self, metric: str = "overall_score") -> List[Dict[str, Any]]:
        """
        生成排行榜
        
        Args:
            metric: 排序指标
            
        Returns:
            排行榜列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取所有已完成实验的最新结果
                cursor.execute('''
                    SELECT e.id, e.name, e.created_at, e.tags, er.result_type, er.metrics, er.created_at as result_time
                    FROM experiments e
                    JOIN experiment_results er ON e.id = er.experiment_id
                    WHERE e.status = 'completed'
                    ORDER BY er.created_at DESC
                ''')
                
                leaderboard_entries = []
                
                # 按实验分组，只保留每个实验的最新结果
                latest_results = {}
                for row in cursor.fetchall():
                    exp_id, name, created_at, tags, result_type, metrics_str, result_time = row
                    
                    if exp_id not in latest_results or result_time > latest_results[exp_id]['result_time']:
                        metrics = json.loads(metrics_str) if metrics_str else {}
                        latest_results[exp_id] = {
                            "id": exp_id,
                            "name": name,
                            "created_at": created_at,
                            "tags": json.loads(tags) if tags else [],
                            "result_type": result_type,
                            "metrics": metrics,
                            "result_time": result_time
                        }
                
                # 构建排行榜条目
                for exp_id, result_data in latest_results.items():
                    metrics = result_data["metrics"]
                    
                    # 查找指定的指标
                    score = 0.0
                    if metric in metrics:
                        score = float(metrics[metric])
                    elif "overall_score" in metrics:
                        score = float(metrics["overall_score"])
                    elif "accuracy" in metrics:
                        score = float(metrics["accuracy"])
                    else:
                        # 如果没有找到指定指标，使用第一个可用的数值指标
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                score = float(value)
                                break
                    
                    leaderboard_entries.append({
                        "experiment_id": exp_id,
                        "model_name": result_data["name"],
                        "score": score,
                        "metric": metric,
                        "created_at": result_data["created_at"],
                        "tags": result_data["tags"],
                        "all_metrics": metrics
                    })
                
                # 按分数排序（降序）
                leaderboard_entries.sort(key=lambda x: x["score"], reverse=True)
                
                # 添加排名
                for i, entry in enumerate(leaderboard_entries, 1):
                    entry["rank"] = i
                
                logger.info(f"生成排行榜成功，指标: {metric}，条目数: {len(leaderboard_entries)}")
                return leaderboard_entries
                
        except Exception as e:
            logger.error(f"生成排行榜失败: {e}")
            return []

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """
        获取实验统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 总实验数
                cursor.execute('SELECT COUNT(*) FROM experiments')
                total_experiments = cursor.fetchone()[0]
                
                # 按状态统计
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM experiments 
                    GROUP BY status
                ''')
                status_counts = dict(cursor.fetchall())
                
                # 最近的实验
                cursor.execute('''
                    SELECT name, created_at 
                    FROM experiments 
                    ORDER BY created_at DESC 
                    LIMIT 5
                ''')
                recent_experiments = [
                    {"name": row[0], "created_at": row[1]} 
                    for row in cursor.fetchall()
                ]
                
                # 存储使用情况
                cursor.execute('SELECT SUM(file_size) FROM experiment_files')
                total_storage = cursor.fetchone()[0] or 0
                
                # 计算平均准确率
                cursor.execute('''
                    SELECT AVG(CAST(json_extract(metrics, '$.accuracy') AS REAL))
                    FROM experiment_results
                    WHERE json_extract(metrics, '$.accuracy') IS NOT NULL
                ''')
                avg_accuracy = cursor.fetchone()[0] or 0
                
                # 计算最佳准确率
                cursor.execute('''
                    SELECT MAX(CAST(json_extract(metrics, '$.accuracy') AS REAL))
                    FROM experiment_results
                    WHERE json_extract(metrics, '$.accuracy') IS NOT NULL
                ''')
                best_accuracy = cursor.fetchone()[0] or 0
                
                return {
                    "total_experiments": total_experiments,
                    "status_distribution": status_counts,
                    "recent_experiments": recent_experiments,
                    "total_storage_bytes": total_storage,
                    "total_storage_mb": total_storage / (1024 * 1024),
                    "avg_accuracy": avg_accuracy,
                    "best_accuracy": best_accuracy
                }
                
        except Exception as e:
            logger.error(f"获取实验统计信息失败: {e}")
            return {}
