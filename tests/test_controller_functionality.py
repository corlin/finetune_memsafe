"""
控制器功能性测试

专注于测试ModelExportController的核心功能，避免复杂的mock设置。
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.model_export_controller import ModelExportController, ExportState
from src.export_models import ExportConfiguration, QuantizationLevel, LogLevel
from src.export_exceptions import ModelExportError


class TestControllerFunctionality:
    """控制器功能性测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def valid_checkpoint_dir(self, temp_dir):
        """创建有效的checkpoint目录"""
        checkpoint_dir = Path(temp_dir) / "valid_checkpoint"
        checkpoint_dir.mkdir(parents=True)
        
        # 创建必需的checkpoint文件
        (checkpoint_dir / "adapter_config.json").write_text('''{
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }''')
        
        # 创建足够大的模型文件
        (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"fake_model_data" * 100000)
        
        return str(checkpoint_dir)
    
    @pytest.fixture
    def basic_config(self, temp_dir, valid_checkpoint_dir):
        """基本配置"""
        return ExportConfiguration(
            checkpoint_path=valid_checkpoint_dir,
            base_model_name="test_model",
            output_directory=str(Path(temp_dir) / "output"),
            quantization_level=QuantizationLevel.NONE,
            export_pytorch=True,
            export_onnx=False,
            export_tensorrt=False,
            run_validation_tests=False,
            enable_progress_monitoring=False,
            log_level=LogLevel.INFO
        )
    
    def test_controller_initialization_success(self, basic_config):
        """测试控制器成功初始化"""
        controller = ModelExportController(basic_config)
        
        # 验证基本属性
        assert controller.config == basic_config
        assert controller.export_state is None
        assert controller.state_file_path is None
        
        # 验证组件初始化
        assert controller.checkpoint_detector is not None
        assert controller.model_merger is not None
        assert controller.optimization_processor is not None
        assert controller.format_exporter is not None
        assert controller.validation_tester is not None
        assert controller.monitoring_logger is not None
        
        # 验证阶段定义
        expected_phases = [
            "checkpoint_detection", "model_merging", "optimization",
            "pytorch_export", "onnx_export", "tensorrt_export", "validation"
        ]
        assert controller.PHASES == expected_phases
    
    def test_export_state_management(self, basic_config):
        """测试导出状态管理"""
        controller = ModelExportController(basic_config)
        
        # 初始化状态
        export_id = "test_state_management"
        controller._initialize_export_state(export_id)
        
        # 验证状态初始化
        assert controller.export_state is not None
        assert controller.export_state.export_id == export_id
        assert controller.export_state.current_phase == "initialized"
        assert len(controller.export_state.completed_phases) == 0
        assert len(controller.export_state.failed_phases) == 0
        
        # 验证状态文件创建
        assert controller.state_file_path is not None
        assert os.path.exists(controller.state_file_path)
        
        # 测试状态更新
        controller.export_state.mark_phase_completed("checkpoint_detection", 1.5)
        assert "checkpoint_detection" in controller.export_state.completed_phases
        assert controller.export_state.phase_durations["checkpoint_detection"] == 1.5
        
        # 测试失败状态
        controller.export_state.mark_phase_failed("model_merging", "测试失败")
        assert "model_merging" in controller.export_state.failed_phases
        assert controller.export_state.last_error == "测试失败"
        assert controller.export_state.retry_count == 1
    
    def test_phase_execution_logic(self, basic_config):
        """测试阶段执行逻辑"""
        controller = ModelExportController(basic_config)
        
        # 测试阶段执行判断
        assert controller._should_execute_phase("checkpoint_detection") == True
        assert controller._should_execute_phase("model_merging") == True
        assert controller._should_execute_phase("optimization") == True
        assert controller._should_execute_phase("pytorch_export") == True
        assert controller._should_execute_phase("onnx_export") == False
        assert controller._should_execute_phase("tensorrt_export") == False
        assert controller._should_execute_phase("validation") == False
        
        # 测试格式导出判断
        assert controller._should_export_format("pytorch") == True
        assert controller._should_export_format("onnx") == False
        assert controller._should_export_format("tensorrt") == False
        assert controller._should_export_format("unknown") == False
    
    def test_checkpoint_detection_phase(self, basic_config):
        """测试checkpoint检测阶段"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("checkpoint_test")
        
        # 执行checkpoint检测
        controller._phase_checkpoint_detection()
        
        # 验证结果
        assert controller.export_state.checkpoint_path is not None
        assert controller.export_state.checkpoint_path == basic_config.checkpoint_path
    
    def test_error_handling_and_retry_mechanism(self, basic_config):
        """测试错误处理和重试机制"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("retry_test")
        
        # 模拟会重试成功的场景
        call_count = 0
        def mock_failing_phase():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 前两次失败
                raise Exception(f"模拟失败 {call_count}")
            # 第三次成功
        
        with patch.object(controller, '_phase_checkpoint_detection', side_effect=mock_failing_phase):
            # 应该重试并最终成功
            controller._execute_single_phase("checkpoint_detection", "test_op", 1, None)
        
        # 验证重试次数和最终成功
        assert call_count == 3
        assert "checkpoint_detection" in controller.export_state.completed_phases
        assert "checkpoint_detection" not in controller.export_state.failed_phases
    
    def test_callback_system(self, basic_config):
        """测试回调系统"""
        controller = ModelExportController(basic_config)
        
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
    
    def test_parallel_export_configuration(self, basic_config):
        """测试并发导出配置"""
        # 测试默认配置（禁用并发）
        controller = ModelExportController(basic_config)
        assert controller.max_concurrent_exports == 1
        
        # 测试并发导出检查
        with pytest.raises(ModelExportError, match="并发导出未启用"):
            controller.export_multiple_formats_parallel(["pytorch", "onnx"])
        
        # 测试启用并发
        basic_config.enable_parallel_export = True
        controller_parallel = ModelExportController(basic_config)
        assert controller_parallel.max_concurrent_exports == 2
    
    def test_export_status_reporting(self, basic_config):
        """测试导出状态报告"""
        controller = ModelExportController(basic_config)
        
        # 未初始化时应返回None
        assert controller.get_export_status() is None
        
        # 初始化后应返回状态
        controller._initialize_export_state("status_test")
        status = controller.get_export_status()
        
        assert status is not None
        assert status['export_id'] == "status_test"
        assert status['current_phase'] == "initialized"
        assert status['completed_phases'] == []
        assert status['failed_phases'] == []
        assert status['progress_percent'] == 0.0
        assert 'start_time' in status
        
        # 更新状态后再次检查
        controller.export_state.mark_phase_completed("checkpoint_detection", 1.0)
        controller.export_state.mark_phase_completed("model_merging", 2.0)
        
        updated_status = controller.get_export_status()
        assert len(updated_status['completed_phases']) == 2
        assert updated_status['progress_percent'] > 0
    
    def test_context_manager_functionality(self, basic_config):
        """测试上下文管理器功能"""
        # 测试正常使用
        with ModelExportController(basic_config) as controller:
            assert isinstance(controller, ModelExportController)
            assert controller.config == basic_config
            
            # 在上下文中执行一些操作
            controller._initialize_export_state("context_test")
            assert controller.export_state is not None
        
        # 上下文退出后应该清理资源（不应抛出异常）
    
    def test_configuration_integration(self, basic_config):
        """测试配置集成"""
        controller = ModelExportController(basic_config)
        
        # 验证配置传递到各组件
        assert controller.format_exporter.config == basic_config
        assert controller.validation_tester.config == basic_config
        
        # 验证配置影响行为
        assert controller._should_execute_phase("pytorch_export") == basic_config.export_pytorch
        assert controller._should_execute_phase("onnx_export") == basic_config.export_onnx
        assert controller._should_execute_phase("validation") == basic_config.run_validation_tests
    
    def test_monitoring_integration(self, basic_config):
        """测试监控系统集成"""
        controller = ModelExportController(basic_config)
        
        # 验证监控系统初始化
        assert controller.monitoring_logger is not None
        
        # 测试监控级别设置
        if basic_config.log_level == LogLevel.DEBUG:
            from src.monitoring_logger import MonitoringLevel
            expected_level = MonitoringLevel.DETAILED
        else:
            from src.monitoring_logger import MonitoringLevel
            expected_level = MonitoringLevel.STANDARD
        
        assert controller.monitoring_logger.monitoring_level == expected_level
    
    def test_cleanup_functionality(self, basic_config):
        """测试清理功能"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("cleanup_test")
        
        # 创建一些临时状态
        temp_dir = Path(basic_config.output_directory) / "temp" / "cleanup_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "test_file.txt").write_text("test content")
        
        # 执行清理
        controller._cleanup_resources()
        
        # 验证清理不会抛出异常
        # 实际的文件清理可能需要根据具体实现进行验证
    
    def test_export_state_persistence(self, basic_config):
        """测试导出状态持久化"""
        controller = ModelExportController(basic_config)
        export_id = "persistence_test"
        
        # 初始化并更新状态
        controller._initialize_export_state(export_id)
        controller.export_state.mark_phase_completed("checkpoint_detection", 1.0)
        controller.export_state.mark_phase_completed("model_merging", 2.0)
        controller.export_state.checkpoint_path = "/test/checkpoint/path"
        
        # 保存状态
        state_file = controller.state_file_path
        assert os.path.exists(state_file)
        
        # 加载状态
        loaded_state = ExportState.load_from_file(state_file)
        assert loaded_state.export_id == export_id
        assert len(loaded_state.completed_phases) == 2
        assert "checkpoint_detection" in loaded_state.completed_phases
        assert "model_merging" in loaded_state.completed_phases
        assert loaded_state.checkpoint_path == "/test/checkpoint/path"
    
    def test_cancel_export_functionality(self, basic_config):
        """测试取消导出功能"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("cancel_test")
        
        # 取消导出
        controller.cancel_export()
        
        # 验证状态更新
        assert controller.export_state.current_phase == "cancelled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])