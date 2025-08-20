"""
测试REST API
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from industry_evaluation.api.rest_api import EvaluationAPI, create_api_app
from industry_evaluation.config.config_manager import ConfigManager


class TestEvaluationAPI:
    """测试评估API"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # 创建测试配置
        self.config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        # 创建API实例
        with patch('industry_evaluation.api.rest_api.IndustryEvaluationEngine'), \
             patch('industry_evaluation.api.rest_api.BatchEvaluator'), \
             patch('industry_evaluation.api.rest_api.ModelManager'), \
             patch('industry_evaluation.api.rest_api.ResultAggregator'), \
             patch('industry_evaluation.api.rest_api.ReportGenerator'):
            
            self.api = EvaluationAPI(self.config_manager)
            self.client = self.api.app.test_client()
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = self.client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_system_info(self):
        """测试系统信息接口"""
        response = self.client.get('/info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'version' in data
        assert 'system' in data
        assert 'models' in data
        assert 'evaluators' in data
        assert 'industry_domains' in data
    
    def test_list_models(self):
        """测试列出模型接口"""
        # 模拟模型管理器返回
        self.api.model_manager.list_models = Mock(return_value=[
            {
                "model_id": "test_model",
                "adapter_type": "openai",
                "is_available": True,
                "config": {}
            }
        ])
        
        response = self.client.get('/models')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert len(data['data']) == 1
        assert data['data'][0]['model_id'] == 'test_model'
    
    def test_test_model(self):
        """测试模型测试接口"""
        # 模拟模型测试结果
        self.api.model_manager.test_model = Mock(return_value={
            "available": True,
            "response_time": 0.5,
            "test_output": "测试输出"
        })
        
        response = self.client.post('/models/test_model/test')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['data']['available'] == True
        assert 'response_time' in data['data']
    
    def test_create_evaluation(self):
        """测试创建评估任务接口"""
        # 模拟评估引擎返回
        self.api.evaluation_engine.evaluate_model = Mock(return_value="task_123")
        
        request_data = {
            "model_id": "test_model",
            "dataset": [
                {
                    "id": "sample_1",
                    "input": "测试输入",
                    "expected_output": "期望输出"
                }
            ],
            "config": {
                "industry_domain": "finance",
                "evaluation_dimensions": ["knowledge"],
                "weight_config": {"knowledge": 1.0},
                "threshold_config": {"knowledge": 0.7}
            }
        }
        
        response = self.client.post(
            '/evaluations',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['data']['task_id'] == 'task_123'
        assert data['data']['status'] == 'created'
    
    def test_create_evaluation_missing_fields(self):
        """测试创建评估任务缺少必需字段"""
        request_data = {
            "model_id": "test_model"
            # 缺少dataset和config字段
        }
        
        response = self.client.post(
            '/evaluations',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['success'] == False
        assert '缺少必需字段' in data['error']
    
    def test_list_evaluations(self):
        """测试列出评估任务接口"""
        # 模拟评估引擎返回
        self.api.evaluation_engine.list_evaluations = Mock(return_value=[
            {
                "task_id": "task_1",
                "status": "completed",
                "progress": 1.0,
                "created_at": "2023-01-01T00:00:00"
            },
            {
                "task_id": "task_2",
                "status": "running",
                "progress": 0.5,
                "created_at": "2023-01-01T01:00:00"
            }
        ])
        
        response = self.client.get('/evaluations')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert len(data['data']) == 2
        assert data['data'][0]['task_id'] == 'task_1'
    
    def test_list_evaluations_with_filter(self):
        """测试带状态过滤的列出评估任务"""
        self.api.evaluation_engine.list_evaluations = Mock(return_value=[
            {
                "task_id": "task_1",
                "status": "completed",
                "progress": 1.0
            }
        ])
        
        response = self.client.get('/evaluations?status=completed')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert len(data['data']) == 1
        
        # 验证调用参数
        self.api.evaluation_engine.list_evaluations.assert_called_with('completed')
    
    def test_get_evaluation(self):
        """测试获取评估任务详情接口"""
        # 模拟进度信息
        mock_progress = Mock()
        mock_progress.status = "completed"
        mock_progress.progress = 1.0
        mock_progress.created_at = "2023-01-01T00:00:00"
        mock_progress.started_at = "2023-01-01T00:01:00"
        mock_progress.completed_at = "2023-01-01T00:05:00"
        
        self.api.evaluation_engine.get_evaluation_progress = Mock(return_value=mock_progress)
        
        # 模拟评估结果
        mock_result = Mock()
        mock_result.overall_score = 0.85
        mock_result.dimension_scores = {"knowledge": 0.85}
        mock_result.detailed_results = [Mock(), Mock()]  # 2个样本
        mock_result.improvement_suggestions = ["建议1", "建议2"]
        
        self.api.evaluation_engine.get_evaluation_result = Mock(return_value=mock_result)
        
        response = self.client.get('/evaluations/task_123')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['data']['task_id'] == 'task_123'
        assert data['data']['status'] == 'completed'
        assert data['data']['result']['overall_score'] == 0.85
        assert data['data']['result']['total_samples'] == 2
    
    def test_get_evaluation_not_found(self):
        """测试获取不存在的评估任务"""
        self.api.evaluation_engine.get_evaluation_progress = Mock(return_value=None)
        
        response = self.client.get('/evaluations/nonexistent_task')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        
        assert data['success'] == False
        assert '评估任务不存在' in data['error']
    
    def test_cancel_evaluation(self):
        """测试取消评估任务接口"""
        self.api.evaluation_engine.cancel_evaluation = Mock(return_value=True)
        
        response = self.client.post('/evaluations/task_123/cancel')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert '评估任务已取消' in data['message']
    
    def test_cancel_evaluation_failed(self):
        """测试取消评估任务失败"""
        self.api.evaluation_engine.cancel_evaluation = Mock(return_value=False)
        
        response = self.client.post('/evaluations/task_123/cancel')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['success'] == False
        assert '取消评估任务失败' in data['error']
    
    def test_get_evaluation_report_json(self):
        """测试获取JSON格式评估报告"""
        mock_report = {
            "overall_score": 0.85,
            "dimension_scores": {"knowledge": 0.85},
            "summary": "评估报告摘要"
        }
        
        self.api.evaluation_engine.generate_report = Mock(
            return_value=json.dumps(mock_report)
        )
        
        response = self.client.get('/evaluations/task_123/report?format=json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['data']['overall_score'] == 0.85
    
    def test_get_evaluation_report_not_found(self):
        """测试获取不存在的评估报告"""
        self.api.evaluation_engine.generate_report = Mock(return_value=None)
        
        response = self.client.get('/evaluations/task_123/report')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        
        assert data['success'] == False
        assert '报告不存在或任务未完成' in data['error']
    
    def test_create_batch_evaluation(self):
        """测试创建批量评估任务接口"""
        # 模拟批量任务
        mock_batch_task = Mock()
        mock_batch_task.task_id = "batch_123"
        mock_batch_task.total_samples = 100
        mock_batch_task.model_ids = ["model1", "model2"]
        
        self.api.batch_evaluator.create_batch_task = Mock(return_value=mock_batch_task)
        self.api.batch_evaluator.start_batch_evaluation = Mock(return_value=True)
        
        request_data = {
            "task_id": "batch_123",
            "model_ids": ["model1", "model2"],
            "dataset_path": "/path/to/dataset.json",
            "evaluation_config": {
                "industry_domain": "finance",
                "evaluation_dimensions": ["knowledge"],
                "weight_config": {"knowledge": 1.0},
                "threshold_config": {"knowledge": 0.7}
            },
            "batch_config": {
                "batch_size": 50,
                "max_concurrent_tasks": 2
            }
        }
        
        response = self.client.post(
            '/batch-evaluations',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['data']['task_id'] == 'batch_123'
        assert data['data']['status'] == 'started'
        assert data['data']['total_samples'] == 100
        assert data['data']['model_count'] == 2
    
    def test_get_batch_evaluation(self):
        """测试获取批量评估任务状态接口"""
        # 模拟批量任务状态
        mock_batch_task = Mock()
        mock_batch_task.task_id = "batch_123"
        mock_batch_task.status = "completed"
        mock_batch_task.model_ids = ["model1", "model2"]
        mock_batch_task.total_samples = 100
        mock_batch_task.processed_samples = 100
        mock_batch_task.failed_samples = 0
        mock_batch_task.created_at = "2023-01-01T00:00:00"
        mock_batch_task.started_at = "2023-01-01T00:01:00"
        mock_batch_task.completed_at = "2023-01-01T00:10:00"
        mock_batch_task.errors = []
        
        # 模拟结果
        mock_result1 = Mock()
        mock_result1.overall_score = 0.8
        mock_result1.dimension_scores = {"knowledge": 0.8}
        mock_result1.detailed_results = [Mock(), Mock()]
        
        mock_result2 = Mock()
        mock_result2.overall_score = 0.75
        mock_result2.dimension_scores = {"knowledge": 0.75}
        mock_result2.detailed_results = [Mock(), Mock()]
        
        mock_batch_task.results = {
            "model1": mock_result1,
            "model2": mock_result2
        }
        
        self.api.batch_evaluator.get_batch_task_status = Mock(return_value=mock_batch_task)
        
        response = self.client.get('/batch-evaluations/batch_123')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['data']['task_id'] == 'batch_123'
        assert data['data']['status'] == 'completed'
        assert data['data']['total_samples'] == 100
        assert data['data']['processed_samples'] == 100
        assert 'results_summary' in data['data']
        assert 'model1' in data['data']['results_summary']
        assert 'model2' in data['data']['results_summary']
    
    def test_get_config(self):
        """测试获取系统配置接口"""
        response = self.client.get('/config')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert 'version' in data['data']
        assert 'system' in data['data']
        assert 'models' in data['data']
        assert 'evaluators' in data['data']
        
        # 验证敏感信息被隐藏
        if data['data']['models']:
            for model_config in data['data']['models'].values():
                assert 'api_key' not in model_config
    
    def test_update_config(self):
        """测试更新系统配置接口"""
        self.api.config_manager.update_config = Mock(return_value=True)
        
        request_data = {
            "system": {
                "max_workers": 8
            }
        }
        
        response = self.client.put(
            '/config',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert '配置更新成功' in data['message']
    
    def test_upload_dataset(self):
        """测试上传数据集文件接口"""
        # 创建测试文件
        test_data = [
            {"input": "测试输入", "expected_output": "期望输出"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = self.client.post(
                    '/upload/dataset',
                    data={'file': (f, 'test_dataset.json')},
                    content_type='multipart/form-data'
                )
            
            assert response.status_code == 201
            data = json.loads(response.data)
            
            assert data['success'] == True
            assert 'filename' in data['data']
            assert 'file_path' in data['data']
            assert 'file_size' in data['data']
            
        finally:
            import os
            os.unlink(temp_file_path)
    
    def test_upload_dataset_invalid_format(self):
        """测试上传无效格式的数据集文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("这是一个文本文件")
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = self.client.post(
                    '/upload/dataset',
                    data={'file': (f, 'test_file.txt')},
                    content_type='multipart/form-data'
                )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            
            assert data['success'] == False
            assert '不支持的文件类型' in data['error']
            
        finally:
            import os
            os.unlink(temp_file_path)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 模拟内部错误
        self.api.model_manager.list_models = Mock(side_effect=Exception("内部错误"))
        
        response = self.client.get('/models')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        
        assert data['success'] == False
        assert 'error' in data


class TestAPIFactory:
    """测试API工厂函数"""
    
    def test_create_api_app(self):
        """测试创建API应用"""
        temp_dir = tempfile.mkdtemp()
        try:
            config_file = Path(temp_dir) / "test_config.yaml"
            
            with patch('industry_evaluation.api.rest_api.IndustryEvaluationEngine'), \
                 patch('industry_evaluation.api.rest_api.BatchEvaluator'), \
                 patch('industry_evaluation.api.rest_api.ModelManager'), \
                 patch('industry_evaluation.api.rest_api.ResultAggregator'), \
                 patch('industry_evaluation.api.rest_api.ReportGenerator'):
                
                app = create_api_app(str(config_file))
                
                assert app is not None
                assert hasattr(app, 'test_client')
                
                # 测试基本功能
                client = app.test_client()
                response = client.get('/health')
                assert response.status_code == 200
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])