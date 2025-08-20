"""
API文档生成器
"""

import json
from typing import Dict, Any
from flask import Flask
from flask_restx import Api, Resource, fields, Namespace
from industry_evaluation.api.rest_api import EvaluationAPI


def create_documented_api(config_file: str) -> Flask:
    """
    创建带文档的API应用
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        Flask: Flask应用实例
    """
    from industry_evaluation.config.config_manager import ConfigManager
    
    config_manager = ConfigManager(config_file)
    evaluation_api = EvaluationAPI(config_manager)
    app = evaluation_api.get_app()
    
    # 创建API文档
    api = Api(
        app,
        version='1.0',
        title='行业模型评估系统API',
        description='用于行业化模型微调效果评估的RESTful API',
        doc='/docs/',
        prefix='/api/v1'
    )
    
    # 定义数据模型
    _define_api_models(api)
    
    # 定义命名空间
    _define_namespaces(api)
    
    return app


def _define_api_models(api: Api):
    """定义API数据模型"""
    
    # 基础响应模型
    api.model('BaseResponse', {
        'success': fields.Boolean(required=True, description='请求是否成功'),
        'message': fields.String(description='响应消息'),
        'error': fields.String(description='错误信息')
    })
    
    # 评估配置模型
    api.model('EvaluationConfig', {
        'industry_domain': fields.String(required=True, description='行业领域', example='finance'),
        'evaluation_dimensions': fields.List(fields.String, required=True, description='评估维度', example=['knowledge', 'terminology']),
        'weight_config': fields.Raw(required=True, description='权重配置', example={'knowledge': 0.7, 'terminology': 0.3}),
        'threshold_config': fields.Raw(required=True, description='阈值配置', example={'knowledge': 0.6, 'terminology': 0.5}),
        'auto_generate_report': fields.Boolean(description='是否自动生成报告', default=False)
    })
    
    # 数据样本模型
    api.model('DataSample', {
        'id': fields.String(description='样本ID'),
        'input': fields.String(required=True, description='输入文本'),
        'expected_output': fields.String(required=True, description='期望输出'),
        'context': fields.Raw(description='上下文信息')
    })
    
    # 评估请求模型
    api.model('EvaluationRequest', {
        'model_id': fields.String(required=True, description='模型ID'),
        'dataset': fields.List(fields.Nested(api.models['DataSample']), required=True, description='数据集'),
        'config': fields.Nested(api.models['EvaluationConfig'], required=True, description='评估配置')
    })
    
    # 评估结果模型
    api.model('EvaluationResult', {
        'overall_score': fields.Float(description='综合得分'),
        'dimension_scores': fields.Raw(description='各维度得分'),
        'total_samples': fields.Integer(description='总样本数'),
        'improvement_suggestions': fields.List(fields.String, description='改进建议')
    })
    
    # 评估任务模型
    api.model('EvaluationTask', {
        'task_id': fields.String(description='任务ID'),
        'status': fields.String(description='任务状态', enum=['pending', 'running', 'completed', 'failed', 'cancelled']),
        'progress': fields.Float(description='进度百分比'),
        'created_at': fields.String(description='创建时间'),
        'started_at': fields.String(description='开始时间'),
        'completed_at': fields.String(description='完成时间'),
        'result': fields.Nested(api.models['EvaluationResult'], description='评估结果')
    })
    
    # 批量评估配置模型
    api.model('BatchEvaluationConfig', {
        'batch_size': fields.Integer(description='批次大小', default=100),
        'max_concurrent_tasks': fields.Integer(description='最大并发任务数', default=4),
        'chunk_size': fields.Integer(description='数据块大小', default=1000),
        'save_intermediate_results': fields.Boolean(description='是否保存中间结果', default=True),
        'enable_parallel_processing': fields.Boolean(description='是否启用并行处理', default=True)
    })
    
    # 批量评估请求模型
    api.model('BatchEvaluationRequest', {
        'task_id': fields.String(required=True, description='任务ID'),
        'model_ids': fields.List(fields.String, required=True, description='模型ID列表'),
        'dataset_path': fields.String(required=True, description='数据集文件路径'),
        'evaluation_config': fields.Nested(api.models['EvaluationConfig'], required=True, description='评估配置'),
        'batch_config': fields.Nested(api.models['BatchEvaluationConfig'], description='批量配置')
    })
    
    # 模型信息模型
    api.model('ModelInfo', {
        'model_id': fields.String(description='模型ID'),
        'adapter_type': fields.String(description='适配器类型'),
        'is_available': fields.Boolean(description='是否可用'),
        'config': fields.Raw(description='配置信息')
    })
    
    # 系统信息模型
    api.model('SystemInfo', {
        'version': fields.String(description='系统版本'),
        'system': fields.Raw(description='系统配置'),
        'models': fields.List(fields.String, description='可用模型列表'),
        'evaluators': fields.List(fields.String, description='可用评估器列表'),
        'industry_domains': fields.List(fields.String, description='支持的行业领域')
    })


def _define_namespaces(api: Api):
    """定义API命名空间"""
    
    # 系统命名空间
    system_ns = Namespace('system', description='系统相关接口')
    
    @system_ns.route('/health')
    class HealthCheck(Resource):
        @system_ns.doc('health_check')
        @system_ns.marshal_with(api.models['BaseResponse'])
        def get(self):
            """健康检查"""
            pass
    
    @system_ns.route('/info')
    class SystemInfo(Resource):
        @system_ns.doc('system_info')
        @system_ns.marshal_with(api.model('SystemInfoResponse', {
            'success': fields.Boolean(),
            'data': fields.Nested(api.models['SystemInfo'])
        }))
        def get(self):
            """获取系统信息"""
            pass
    
    # 模型管理命名空间
    models_ns = Namespace('models', description='模型管理接口')
    
    @models_ns.route('/')
    class ModelList(Resource):
        @models_ns.doc('list_models')
        @models_ns.marshal_with(api.model('ModelListResponse', {
            'success': fields.Boolean(),
            'data': fields.List(fields.Nested(api.models['ModelInfo']))
        }))
        def get(self):
            """列出所有模型"""
            pass
    
    @models_ns.route('/<string:model_id>/test')
    class ModelTest(Resource):
        @models_ns.doc('test_model')
        @models_ns.marshal_with(api.model('ModelTestResponse', {
            'success': fields.Boolean(),
            'data': fields.Raw(description='测试结果')
        }))
        def post(self, model_id):
            """测试模型可用性"""
            pass
    
    # 评估任务命名空间
    evaluations_ns = Namespace('evaluations', description='评估任务接口')
    
    @evaluations_ns.route('/')
    class EvaluationList(Resource):
        @evaluations_ns.doc('list_evaluations')
        @evaluations_ns.marshal_with(api.model('EvaluationListResponse', {
            'success': fields.Boolean(),
            'data': fields.List(fields.Nested(api.models['EvaluationTask']))
        }))
        @evaluations_ns.param('status', '状态过滤器', enum=['pending', 'running', 'completed', 'failed', 'cancelled'])
        def get(self):
            """列出评估任务"""
            pass
        
        @evaluations_ns.doc('create_evaluation')
        @evaluations_ns.expect(api.models['EvaluationRequest'])
        @evaluations_ns.marshal_with(api.model('EvaluationCreateResponse', {
            'success': fields.Boolean(),
            'data': fields.Raw(description='创建结果')
        }))
        def post(self):
            """创建评估任务"""
            pass
    
    @evaluations_ns.route('/<string:task_id>')
    class EvaluationDetail(Resource):
        @evaluations_ns.doc('get_evaluation')
        @evaluations_ns.marshal_with(api.model('EvaluationDetailResponse', {
            'success': fields.Boolean(),
            'data': fields.Nested(api.models['EvaluationTask'])
        }))
        def get(self, task_id):
            """获取评估任务详情"""
            pass
    
    @evaluations_ns.route('/<string:task_id>/cancel')
    class EvaluationCancel(Resource):
        @evaluations_ns.doc('cancel_evaluation')
        @evaluations_ns.marshal_with(api.models['BaseResponse'])
        def post(self, task_id):
            """取消评估任务"""
            pass
    
    @evaluations_ns.route('/<string:task_id>/report')
    class EvaluationReport(Resource):
        @evaluations_ns.doc('get_evaluation_report')
        @evaluations_ns.param('format', '报告格式', enum=['json', 'html', 'pdf'], default='json')
        def get(self, task_id):
            """获取评估报告"""
            pass
    
    # 批量评估命名空间
    batch_ns = Namespace('batch-evaluations', description='批量评估接口')
    
    @batch_ns.route('/')
    class BatchEvaluationList(Resource):
        @batch_ns.doc('create_batch_evaluation')
        @batch_ns.expect(api.models['BatchEvaluationRequest'])
        @batch_ns.marshal_with(api.model('BatchEvaluationCreateResponse', {
            'success': fields.Boolean(),
            'data': fields.Raw(description='创建结果')
        }))
        def post(self):
            """创建批量评估任务"""
            pass
    
    @batch_ns.route('/<string:task_id>')
    class BatchEvaluationDetail(Resource):
        @batch_ns.doc('get_batch_evaluation')
        @batch_ns.marshal_with(api.model('BatchEvaluationDetailResponse', {
            'success': fields.Boolean(),
            'data': fields.Raw(description='批量评估任务详情')
        }))
        def get(self, task_id):
            """获取批量评估任务状态"""
            pass
    
    # 配置管理命名空间
    config_ns = Namespace('config', description='配置管理接口')
    
    @config_ns.route('/')
    class ConfigManagement(Resource):
        @config_ns.doc('get_config')
        @config_ns.marshal_with(api.model('ConfigResponse', {
            'success': fields.Boolean(),
            'data': fields.Raw(description='系统配置')
        }))
        def get(self):
            """获取系统配置"""
            pass
        
        @config_ns.doc('update_config')
        @config_ns.expect(api.model('ConfigUpdateRequest', {
            'config': fields.Raw(required=True, description='配置更新数据')
        }))
        @config_ns.marshal_with(api.models['BaseResponse'])
        def put(self):
            """更新系统配置"""
            pass
    
    # 文件上传命名空间
    upload_ns = Namespace('upload', description='文件上传接口')
    
    @upload_ns.route('/dataset')
    class DatasetUpload(Resource):
        @upload_ns.doc('upload_dataset')
        @upload_ns.marshal_with(api.model('UploadResponse', {
            'success': fields.Boolean(),
            'data': fields.Raw(description='上传结果')
        }))
        def post(self):
            """上传数据集文件"""
            pass
    
    # 注册命名空间
    api.add_namespace(system_ns, path='/system')
    api.add_namespace(models_ns, path='/models')
    api.add_namespace(evaluations_ns, path='/evaluations')
    api.add_namespace(batch_ns, path='/batch-evaluations')
    api.add_namespace(config_ns, path='/config')
    api.add_namespace(upload_ns, path='/upload')


def generate_openapi_spec(config_file: str) -> Dict[str, Any]:
    """
    生成OpenAPI规范
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        Dict[str, Any]: OpenAPI规范
    """
    app = create_documented_api(config_file)
    
    with app.app_context():
        # 获取API实例
        api = app.extensions['restx']
        
        # 生成OpenAPI规范
        spec = api.__schema__
        
        # 添加额外信息
        spec['info']['contact'] = {
            'name': 'API Support',
            'email': 'support@example.com'
        }
        
        spec['info']['license'] = {
            'name': 'MIT',
            'url': 'https://opensource.org/licenses/MIT'
        }
        
        spec['servers'] = [
            {
                'url': 'http://localhost:5000/api/v1',
                'description': '开发服务器'
            },
            {
                'url': 'https://api.example.com/v1',
                'description': '生产服务器'
            }
        ]
        
        # 添加安全定义
        spec['components']['securitySchemes'] = {
            'ApiKeyAuth': {
                'type': 'apiKey',
                'in': 'header',
                'name': 'X-API-Key'
            },
            'BearerAuth': {
                'type': 'http',
                'scheme': 'bearer',
                'bearerFormat': 'JWT'
            }
        }
        
        return spec


def save_openapi_spec(config_file: str, output_file: str):
    """
    保存OpenAPI规范到文件
    
    Args:
        config_file: 配置文件路径
        output_file: 输出文件路径
    """
    spec = generate_openapi_spec(config_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API文档生成器")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--output", help="OpenAPI规范输出文件路径")
    parser.add_argument("--serve", action="store_true", help="启动文档服务器")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    
    args = parser.parse_args()
    
    if args.output:
        # 生成OpenAPI规范文件
        save_openapi_spec(args.config, args.output)
        print(f"OpenAPI规范已保存到: {args.output}")
    
    if args.serve:
        # 启动文档服务器
        app = create_documented_api(args.config)
        print(f"API文档服务器启动: http://{args.host}:{args.port}/docs/")
        app.run(host=args.host, port=args.port, debug=True)