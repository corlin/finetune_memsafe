"""
RESTful API接口
"""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
from werkzeug.utils import secure_filename

from industry_evaluation.core.evaluation_engine import IndustryEvaluationEngine
from industry_evaluation.core.batch_evaluator import BatchEvaluator, BatchEvaluationConfig
from industry_evaluation.core.result_aggregator import ResultAggregator
from industry_evaluation.adapters.model_adapter import ModelManager
from industry_evaluation.reporting.report_generator import ReportGenerator
from industry_evaluation.config.config_manager import ConfigManager
from industry_evaluation.core.interfaces import EvaluationConfig


class EvaluationAPI:
    """评估系统API"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化API
        
        Args:
            config_manager: 配置管理器
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化Flask应用
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
        CORS(self.app)
        
        # 初始化评估系统组件
        self._init_evaluation_system()
        
        # 注册路由
        self._register_routes()
        
        # 注册错误处理器
        self._register_error_handlers()
    
    def _init_evaluation_system(self):
        """初始化评估系统"""
        try:
            config = self.config_manager.get_config()
            
            # 初始化组件
            self.model_manager = ModelManager()
            self.evaluators = {}  # 这里应该根据配置初始化评估器
            self.result_aggregator = ResultAggregator()
            self.report_generator = ReportGenerator()
            
            # 初始化评估引擎
            self.evaluation_engine = IndustryEvaluationEngine(
                model_manager=self.model_manager,
                evaluators=self.evaluators,
                result_aggregator=self.result_aggregator,
                report_generator=self.report_generator,
                max_workers=config.system.max_workers
            )
            
            # 初始化批量评估器
            self.batch_evaluator = BatchEvaluator(self.evaluation_engine)
            
            self.logger.info("评估系统初始化成功")
            
        except Exception as e:
            self.logger.error(f"评估系统初始化失败: {str(e)}")
            raise
    
    def _register_routes(self):
        """注册API路由"""
        
        # 健康检查
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": self.config_manager.get_config().version
            })
        
        # 系统信息
        @self.app.route('/info', methods=['GET'])
        def system_info():
            """系统信息接口"""
            config = self.config_manager.get_config()
            return jsonify({
                "version": config.version,
                "system": {
                    "max_workers": config.system.max_workers,
                    "log_level": config.system.log_level
                },
                "models": list(config.models.keys()),
                "evaluators": list(config.evaluators.keys()),
                "industry_domains": config.industry_domains
            })
        
        # 模型管理
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """列出所有模型"""
            try:
                models = self.model_manager.list_models()
                return jsonify({
                    "success": True,
                    "data": models
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/models/<model_id>/test', methods=['POST'])
        def test_model(model_id):
            """测试模型可用性"""
            try:
                result = self.model_manager.test_model(model_id)
                return jsonify({
                    "success": True,
                    "data": result
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # 评估任务管理
        @self.app.route('/evaluations', methods=['POST'])
        def create_evaluation():
            """创建评估任务"""
            try:
                data = request.get_json()
                
                # 验证请求数据
                required_fields = ['model_id', 'dataset', 'config']
                for field in required_fields:
                    if field not in data:
                        raise BadRequest(f"缺少必需字段: {field}")
                
                # 解析评估配置
                config_data = data['config']
                evaluation_config = EvaluationConfig(
                    industry_domain=config_data.get('industry_domain', 'general'),
                    evaluation_dimensions=config_data.get('evaluation_dimensions', []),
                    weight_config=config_data.get('weight_config', {}),
                    threshold_config=config_data.get('threshold_config', {}),
                    auto_generate_report=config_data.get('auto_generate_report', False)
                )
                
                # 启动评估
                task_id = self.evaluation_engine.evaluate_model(
                    model_id=data['model_id'],
                    dataset=data['dataset'],
                    evaluation_config=evaluation_config
                )
                
                return jsonify({
                    "success": True,
                    "data": {
                        "task_id": task_id,
                        "status": "created"
                    }
                }), 201
                
            except BadRequest as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 400
            except Exception as e:
                self.logger.error(f"创建评估任务失败: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": "内部服务器错误"
                }), 500
        
        @self.app.route('/evaluations', methods=['GET'])
        def list_evaluations():
            """列出评估任务"""
            try:
                status_filter = request.args.get('status')
                evaluations = self.evaluation_engine.list_evaluations(status_filter)
                
                return jsonify({
                    "success": True,
                    "data": evaluations
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/evaluations/<task_id>', methods=['GET'])
        def get_evaluation(task_id):
            """获取评估任务详情"""
            try:
                # 获取进度信息
                progress = self.evaluation_engine.get_evaluation_progress(task_id)
                if not progress:
                    raise NotFound(f"评估任务不存在: {task_id}")
                
                response_data = {
                    "task_id": task_id,
                    "status": progress.status,
                    "progress": getattr(progress, 'progress', 0),
                    "created_at": getattr(progress, 'created_at', None),
                    "started_at": getattr(progress, 'started_at', None),
                    "completed_at": getattr(progress, 'completed_at', None)
                }
                
                # 如果任务完成，添加结果
                if progress.status == "completed":
                    result = self.evaluation_engine.get_evaluation_result(task_id)
                    if result:
                        response_data["result"] = {
                            "overall_score": result.overall_score,
                            "dimension_scores": result.dimension_scores,
                            "total_samples": len(result.detailed_results),
                            "improvement_suggestions": result.improvement_suggestions
                        }
                
                return jsonify({
                    "success": True,
                    "data": response_data
                })
                
            except NotFound as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 404
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/evaluations/<task_id>/cancel', methods=['POST'])
        def cancel_evaluation(task_id):
            """取消评估任务"""
            try:
                success = self.evaluation_engine.cancel_evaluation(task_id)
                
                if success:
                    return jsonify({
                        "success": True,
                        "message": "评估任务已取消"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "取消评估任务失败"
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/evaluations/<task_id>/report', methods=['GET'])
        def get_evaluation_report(task_id):
            """获取评估报告"""
            try:
                report_format = request.args.get('format', 'json')
                
                report = self.evaluation_engine.generate_report(task_id, report_format)
                
                if not report:
                    raise NotFound("报告不存在或任务未完成")
                
                if report_format == 'json':
                    # 如果是JSON格式，直接返回
                    if isinstance(report, str):
                        try:
                            report_data = json.loads(report)
                            return jsonify({
                                "success": True,
                                "data": report_data
                            })
                        except json.JSONDecodeError:
                            pass
                    
                    return jsonify({
                        "success": True,
                        "data": report
                    })
                else:
                    # 如果是文件格式，返回文件
                    if isinstance(report, str) and Path(report).exists():
                        return send_file(report, as_attachment=True)
                    else:
                        return jsonify({
                            "success": False,
                            "error": "报告文件不存在"
                        }), 404
                        
            except NotFound as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 404
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # 批量评估
        @self.app.route('/batch-evaluations', methods=['POST'])
        def create_batch_evaluation():
            """创建批量评估任务"""
            try:
                data = request.get_json()
                
                # 验证请求数据
                required_fields = ['task_id', 'model_ids', 'dataset_path', 'evaluation_config']
                for field in required_fields:
                    if field not in data:
                        raise BadRequest(f"缺少必需字段: {field}")
                
                # 解析评估配置
                eval_config_data = data['evaluation_config']
                evaluation_config = EvaluationConfig(
                    industry_domain=eval_config_data.get('industry_domain', 'general'),
                    evaluation_dimensions=eval_config_data.get('evaluation_dimensions', []),
                    weight_config=eval_config_data.get('weight_config', {}),
                    threshold_config=eval_config_data.get('threshold_config', {})
                )
                
                # 解析批量配置
                batch_config_data = data.get('batch_config', {})
                batch_config = BatchEvaluationConfig(
                    batch_size=batch_config_data.get('batch_size', 100),
                    max_concurrent_tasks=batch_config_data.get('max_concurrent_tasks', 4),
                    chunk_size=batch_config_data.get('chunk_size', 1000),
                    save_intermediate_results=batch_config_data.get('save_intermediate_results', True),
                    enable_parallel_processing=batch_config_data.get('enable_parallel_processing', True)
                )
                
                # 创建批量任务
                batch_task = self.batch_evaluator.create_batch_task(
                    task_id=data['task_id'],
                    model_ids=data['model_ids'],
                    dataset_path=data['dataset_path'],
                    evaluation_config=evaluation_config,
                    batch_config=batch_config
                )
                
                # 启动批量评估
                success = self.batch_evaluator.start_batch_evaluation(data['task_id'])
                
                if success:
                    return jsonify({
                        "success": True,
                        "data": {
                            "task_id": batch_task.task_id,
                            "status": "started",
                            "total_samples": batch_task.total_samples,
                            "model_count": len(batch_task.model_ids)
                        }
                    }), 201
                else:
                    return jsonify({
                        "success": False,
                        "error": "启动批量评估失败"
                    }), 500
                    
            except BadRequest as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 400
            except Exception as e:
                self.logger.error(f"创建批量评估任务失败: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": "内部服务器错误"
                }), 500
        
        @self.app.route('/batch-evaluations/<task_id>', methods=['GET'])
        def get_batch_evaluation(task_id):
            """获取批量评估任务状态"""
            try:
                batch_task = self.batch_evaluator.get_batch_task_status(task_id)
                
                if not batch_task:
                    raise NotFound(f"批量评估任务不存在: {task_id}")
                
                response_data = {
                    "task_id": batch_task.task_id,
                    "status": batch_task.status,
                    "model_ids": batch_task.model_ids,
                    "total_samples": batch_task.total_samples,
                    "processed_samples": batch_task.processed_samples,
                    "failed_samples": batch_task.failed_samples,
                    "created_at": batch_task.created_at,
                    "started_at": batch_task.started_at,
                    "completed_at": batch_task.completed_at,
                    "errors": batch_task.errors
                }
                
                # 如果任务完成，添加结果摘要
                if batch_task.status == "completed" and batch_task.results:
                    results_summary = {}
                    for model_id, result in batch_task.results.items():
                        results_summary[model_id] = {
                            "overall_score": result.overall_score,
                            "dimension_scores": result.dimension_scores,
                            "sample_count": len(result.detailed_results)
                        }
                    response_data["results_summary"] = results_summary
                
                return jsonify({
                    "success": True,
                    "data": response_data
                })
                
            except NotFound as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 404
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # 配置管理
        @self.app.route('/config', methods=['GET'])
        def get_config():
            """获取系统配置"""
            try:
                config = self.config_manager.get_config()
                
                # 隐藏敏感信息
                safe_config = {
                    "version": config.version,
                    "system": {
                        "max_workers": config.system.max_workers,
                        "log_level": config.system.log_level,
                        "cache_enabled": config.system.cache_enabled,
                        "monitoring_enabled": config.system.monitoring_enabled
                    },
                    "models": {
                        model_id: {
                            "model_id": model_config.model_id,
                            "adapter_type": model_config.adapter_type,
                            "timeout": model_config.timeout,
                            "max_retries": model_config.max_retries
                        }
                        for model_id, model_config in config.models.items()
                    },
                    "evaluators": {
                        evaluator_id: {
                            "evaluator_type": evaluator_config.evaluator_type,
                            "enabled": evaluator_config.enabled,
                            "weight": evaluator_config.weight,
                            "threshold": evaluator_config.threshold
                        }
                        for evaluator_id, evaluator_config in config.evaluators.items()
                    },
                    "industry_domains": config.industry_domains,
                    "default_weights": config.default_weights,
                    "default_thresholds": config.default_thresholds
                }
                
                return jsonify({
                    "success": True,
                    "data": safe_config
                })
                
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/config', methods=['PUT'])
        def update_config():
            """更新系统配置"""
            try:
                data = request.get_json()
                
                success = self.config_manager.update_config(data)
                
                if success:
                    return jsonify({
                        "success": True,
                        "message": "配置更新成功"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "配置更新失败"
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # 文件上传
        @self.app.route('/upload/dataset', methods=['POST'])
        def upload_dataset():
            """上传数据集文件"""
            try:
                if 'file' not in request.files:
                    raise BadRequest("没有上传文件")
                
                file = request.files['file']
                if file.filename == '':
                    raise BadRequest("文件名为空")
                
                # 验证文件类型
                allowed_extensions = {'.json', '.jsonl', '.csv'}
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in allowed_extensions:
                    raise BadRequest(f"不支持的文件类型: {file_ext}")
                
                # 保存文件
                filename = secure_filename(file.filename)
                upload_dir = Path("uploads/datasets")
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = upload_dir / filename
                file.save(str(file_path))
                
                return jsonify({
                    "success": True,
                    "data": {
                        "filename": filename,
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size
                    }
                }), 201
                
            except BadRequest as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 400
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
    
    def _register_error_handlers(self):
        """注册错误处理器"""
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                "success": False,
                "error": "请求参数错误",
                "details": str(error)
            }), 400
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                "success": False,
                "error": "资源不存在",
                "details": str(error)
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            self.logger.error(f"内部服务器错误: {str(error)}")
            self.logger.error(traceback.format_exc())
            return jsonify({
                "success": False,
                "error": "内部服务器错误"
            }), 500
        
        @self.app.errorhandler(Exception)
        def handle_exception(error):
            self.logger.error(f"未处理的异常: {str(error)}")
            self.logger.error(traceback.format_exc())
            return jsonify({
                "success": False,
                "error": "服务器内部错误"
            }), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        运行API服务器
        
        Args:
            host: 主机地址
            port: 端口号
            debug: 调试模式
        """
        self.logger.info(f"启动API服务器: {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def get_app(self):
        """获取Flask应用实例"""
        return self.app


def create_api_app(config_file: str) -> Flask:
    """
    创建API应用
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        Flask: Flask应用实例
    """
    config_manager = ConfigManager(config_file)
    api = EvaluationAPI(config_manager)
    return api.get_app()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="行业评估系统API服务器")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并运行API
    config_manager = ConfigManager(args.config)
    api = EvaluationAPI(config_manager)
    api.run(host=args.host, port=args.port, debug=args.debug)