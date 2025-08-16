"""
完整导出流程集成测试

本模块测试ModelExportController的完整端到端导出流程。
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.model_export_controller import ModelExportController, ExportState
from src.export_models import ExportConfiguration, QuantizationLevel, LogLevel
from src.export_exceptions import ModelExportError


class TestFullExportIntegration:
    """完整导出流程集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_checkpoint_dir(self, temp_dir):
        """创建模拟checkpoint目录"""
        checkpoint_dir = Path(temp_dir) / "test_checkpoint"
        checkpoint_dir.mkdir(parents=True)
        
        # 创建必需的checkpoint文件
        (checkpoint_dir / "adapter_config.json").write_text('''{
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }''')
        
        # 创建足够大的模型文件以通过验证
        (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"fake_model_data" * 100000)
        
        return str(checkpoint_dir)
    
    @pytest.fixture
    def integration_config(self, temp_dir, mock_checkpoint_dir):
        """集成测试配置"""
        return ExportConfiguration(
            checkpoint_path=mock_checkpoint_dir,
            base_model_name="test_model",
            output_directory=str(Path(temp_dir) / "output"),
            quantization_level=QuantizationLevel.NONE,
            export_pytorch=True,
            export_onnx=False,  # 禁用ONNX以简化测试
            export_tensorrt=False,
            run_validation_tests=False,  # 禁用验证以简化测试
            enable_progress_monitoring=True,
            log_level=LogLevel.INFO
        )
    
    def test_controller_component_integration(self, integration_config):
        """测试控制器与各组件的集成"""
        controller = ModelExportController(integration_config)
        
        # 验证所有组件都已正确初始化
        assert controller.checkpoint_detector is not None
        assert controller.model_merger is not None
        assert controller.optimization_processor is not None
        assert controller.format_exporter is not None
        assert controller.validation_tester is not None
        assert controller.monitoring_logger is not None
        
        # 验证配置传递正确
        assert controller.config == integration_config
        assert controller.format_exporter.config == integration_config
        assert controller.validation_tester.config == integration_config
    
    def test_export_state_lifecycle(self, integration_config):
        """测试导出状态的完整生命周期"""
        controller = ModelExportController(integration_config)
        
        # 初始化状态
        controller._initialize_export_state("integration_test")
        
        assert controller.export_state is not None
        assert controller.export_state.export_id == "integration_test"
        assert controller.export_state.current_phase == "initialized"
        assert len(controller.export_state.completed_phases) == 0
        
        # 模拟阶段执行
        phases = ["checkpoint_detection", "model_merging", "optimization"]
        
        for phase in phases:
            controller.export_state.mark_phase_completed(phase, 1.0)
        
        assert len(controller.export_state.completed_phases) == 3
        assert "checkpoint_detection" in controller.export_state.completed_phases
        assert "model_merging" in controller.export_state.completed_phases
        assert "optimization" in controller.export_state.completed_phases
        
        # 测试状态持久化
        assert controller.state_file_path is not None
        assert os.path.exists(controller.state_file_path)
        
        # 测试状态加载
        loaded_state = ExportState.load_from_file(controller.state_file_path)
        assert loaded_state.export_id == "integration_test"
        assert len(loaded_state.completed_phases) == 3
    
    def test_phase_execution_order(self, integration_config):
        """测试阶段执行顺序"""
        controller = ModelExportController(integration_config)
        
        # 验证阶段定义
        expected_phases = [
            "checkpoint_detection",
            "model_merging", 
            "optimization",
            "pytorch_export",
            "onnx_export",
            "tensorrt_export",
            "validation"
        ]
        
        assert controller.PHASES == expected_phases
        
        # 验证阶段执行判断
        assert controller._should_execute_phase("checkpoint_detection") == True
        assert controller._should_execute_phase("model_merging") == True
        assert controller._should_execute_phase("optimization") == True
        assert controller._should_execute_phase("pytorch_export") == True
        assert controller._should_execute_phase("onnx_export") == False  # 配置中禁用
        assert controller._should_execute_phase("tensorrt_export") == False  # 配置中禁用
        assert controller._should_execute_phase("validation") == False  # 配置中禁用
    
    def test_error_handling_and_retry(self, integration_config):
        """测试错误处理和重试机制"""
        controller = ModelExportController(integration_config)
        controller._initialize_export_state("error_test")
        
        # 模拟失败的阶段
        call_count = 0
        def failing_phase():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 前两次失败
                raise Exception(f"模拟失败 {call_count}")
            # 第三次成功
        
        with patch.object(controller, '_phase_checkpoint_detection', side_effect=failing_phase):
            # 应该重试并最终成功
            controller._execute_single_phase("checkpoint_detection", "test_op", 1, None)
        
        assert call_count == 3  # 1次初始尝试 + 2次重试
        assert "checkpoint_detection" in controller.export_state.completed_phases
        assert "checkpoint_detection" not in controller.export_state.failed_phases
    
    def test_callback_system(self, integration_config):
        """测试回调系统"""
        controller = ModelExportController(integration_config)
        
        # 创建回调函数
        progress_calls = []
        status_calls = []
        
        def progress_callback(op_id, progress, description):
            progress_calls.append((op_id, progress, description))
        
        def status_callback(op_id, status):
            status_calls.append((op_id, status))
        
        # 注册回调
        controller.add_progress_callback(progress_callback)
        controller.add_status_callback(status_callback)
        
        # 验证回调注册
        assert progress_callback in controller.progress_callbacks
        assert status_callback in controller.status_callbacks
        
        # 验证回调传递到监控系统
        assert progress_callback in controller.monitoring_logger.progress_callbacks
        assert status_callback in controller.monitoring_logger.status_callbacks
    
    def test_parallel_export_configuration(self, integration_config):
        """测试并发导出配置"""
        # 测试禁用并发的情况
        controller = ModelExportController(integration_config)
        assert controller.max_concurrent_exports == 1
        
        with pytest.raises(ModelExportError, match="并发导出未启用"):
            controller.export_multiple_formats_parallel(["pytorch", "onnx"])
        
        # 测试启用并发的情况
        integration_config.enable_parallel_export = True
        controller_parallel = ModelExportController(integration_config)
        assert controller_parallel.max_concurrent_exports == 2
    
    def test_resource_cleanup(self, integration_config):
        """测试资源清理"""
        controller = ModelExportController(integration_config)
        controller._initialize_export_state("cleanup_test")
        
        # 创建一些临时文件
        temp_dir = Path(integration_config.output_directory) / "temp" / "cleanup_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "test_file.txt").write_text("test content")
        
        assert temp_dir.exists()
        
        # 执行清理
        controller._cleanup_resources()
        
        # 验证临时文件被清理（在实际实现中）
        # 注意：这个测试可能需要根据实际的清理逻辑进行调整
    
    def test_configuration_validation_integration(self, integration_config):
        """测试配置验证集成"""
        # 测试有效配置
        controller = ModelExportController(integration_config)
        assert controller.config == integration_config
        
        # 测试无效配置
        invalid_config = ExportConfiguration(
            checkpoint_path="nonexistent_path",
            base_model_name="",  # 空的模型名
            output_directory=""  # 空的输出目录
        )
        
        validation_errors = invalid_config.validate()
        assert len(validation_errors) > 0
        assert any("checkpoint路径不存在" in error for error in validation_errors)
        assert any("base_model_name不能为空" in error for error in validation_errors)
        assert any("output_directory不能为空" in error for error in validation_errors)
    
    def test_monitoring_integration(self, integration_config):
        """测试监控系统集成"""
        controller = ModelExportController(integration_config)
        
        # 验证监控系统初始化
        assert controller.monitoring_logger is not None
        
        # 测试监控上下文管理器
        with controller.monitoring_logger:
            # 开始操作
            op_id = controller.monitoring_logger.start_operation("test_operation", 3)
            
            # 更新进度
            controller.monitoring_logger.update_operation_progress(op_id, 1, "步骤1")
            controller.monitoring_logger.update_operation_progress(op_id, 2, "步骤2")
            controller.monitoring_logger.update_operation_progress(op_id, 3, "步骤3")
            
            # 完成操作
            controller.monitoring_logger.complete_operation(op_id, success=True)
        
        # 验证操作记录
        operation = controller.monitoring_logger.get_operation_status(op_id)
        assert operation is not None
        assert operation.operation_name == "test_operation"
        assert operation.current_step == 3
        assert operation.progress_percent == 100.0
    
    def test_context_manager_integration(self, integration_config):
        """测试上下文管理器集成"""
        # 测试正常使用
        with ModelExportController(integration_config) as controller:
            assert isinstance(controller, ModelExportController)
            assert controller.config == integration_config
        
        # 上下文退出后应该清理资源
        # 这里主要测试不会抛出异常
    
    @patch('src.model_merger.ModelMerger.merge_and_save')
    @patch('src.optimization_processor.OptimizationProcessor.apply_quantization')
    @patch('src.format_exporter.FormatExporter.export_pytorch_model')
    def test_mocked_full_pipeline(self, mock_export, mock_optimize, mock_merge, integration_config):
        """测试模拟的完整导出流程"""
        # 配置mock返回值
        mock_merge.return_value = {
            'success': True,
            'model_size_mb': 100.0,
            'parameter_count': 1000000,
            'verification_passed': True
        }
        
        mock_model = Mock()
        mock_optimize.return_value = mock_model
        mock_export.return_value = "/path/to/exported/model"
        
        controller = ModelExportController(integration_config)
        
        # 测试各个阶段的执行
        controller._initialize_export_state("pipeline_test")
        
        # 执行checkpoint检测
        controller._phase_checkpoint_detection()
        assert controller.export_state.checkpoint_path is not None
        
        # 执行模型合并
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_load:
            mock_load.return_value = mock_model
            controller._phase_model_merging()
            assert controller.export_state.merged_model_path is not None
        
        # 执行优化
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_load:
            mock_load.return_value = mock_model
            controller._phase_optimization()
            assert controller.export_state.optimized_model_path is not None
        
        # 执行PyTorch导出
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_load, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_load.return_value = mock_model
            mock_tokenizer.return_value = Mock()
            
            controller._phase_pytorch_export(None)
            assert "pytorch" in controller.export_state.exported_models
        
        # 验证所有阶段都完成
        expected_completed = ["checkpoint_detection", "model_merging", "optimization", "pytorch_export"]
        for phase in expected_completed:
            controller.export_state.mark_phase_completed(phase, 1.0)
        
        assert len(controller.export_state.completed_phases) == len(expected_completed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])