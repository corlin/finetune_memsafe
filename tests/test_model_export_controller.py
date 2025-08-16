"""
模型导出控制器集成测试

本模块包含ModelExportController的完整集成测试，验证端到端的导出流程。
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.model_export_controller import ModelExportController, ExportState
from src.export_models import ExportConfiguration, QuantizationLevel, LogLevel
from src.export_exceptions import ModelExportError, CheckpointValidationError


class TestModelExportController:
    """模型导出控制器测试类"""
    
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
        
        (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"fake_model_data" * 100000)  # Make it larger to pass size validation
        
        return str(checkpoint_dir)
    
    @pytest.fixture
    def basic_config(self, temp_dir, mock_checkpoint_dir):
        """基本配置"""
        return ExportConfiguration(
            checkpoint_path=mock_checkpoint_dir,
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
    
    @pytest.fixture
    def full_config(self, temp_dir, mock_checkpoint_dir):
        """完整配置"""
        return ExportConfiguration(
            checkpoint_path=mock_checkpoint_dir,
            base_model_name="test_model",
            output_directory=str(Path(temp_dir) / "output"),
            quantization_level=QuantizationLevel.INT8,
            export_pytorch=True,
            export_onnx=True,
            export_tensorrt=False,
            run_validation_tests=True,
            enable_progress_monitoring=True,
            log_level=LogLevel.DEBUG,
            enable_parallel_export=True
        )
    
    def test_controller_initialization(self, basic_config):
        """测试控制器初始化"""
        controller = ModelExportController(basic_config)
        
        assert controller.config == basic_config
        assert controller.checkpoint_detector is not None
        assert controller.model_merger is not None
        assert controller.optimization_processor is not None
        assert controller.format_exporter is not None
        assert controller.validation_tester is not None
        assert controller.monitoring_logger is not None
        assert controller.export_state is None
        assert controller.max_concurrent_exports == 1  # 默认不启用并发
    
    def test_controller_initialization_with_parallel(self, full_config):
        """测试启用并发的控制器初始化"""
        controller = ModelExportController(full_config)
        
        assert controller.max_concurrent_exports == 2  # 启用并发
    
    def test_export_state_initialization(self, basic_config):
        """测试导出状态初始化"""
        controller = ModelExportController(basic_config)
        
        export_id = "test_export_123"
        controller._initialize_export_state(export_id)
        
        assert controller.export_state is not None
        assert controller.export_state.export_id == export_id
        assert controller.export_state.current_phase == "initialized"
        assert controller.export_state.completed_phases == []
        assert controller.export_state.start_time is not None
        assert controller.state_file_path is not None
        assert os.path.exists(controller.state_file_path)
    
    def test_export_state_save_and_load(self, temp_dir):
        """测试导出状态保存和加载"""
        state = ExportState(
            export_id="test_123",
            current_phase="model_merging",
            completed_phases=["checkpoint_detection"],
            checkpoint_path="/test/path"
        )
        
        state_file = Path(temp_dir) / "test_state.pkl"
        state.save_to_file(str(state_file))
        
        assert state_file.exists()
        
        loaded_state = ExportState.load_from_file(str(state_file))
        assert loaded_state.export_id == state.export_id
        assert loaded_state.current_phase == state.current_phase
        assert loaded_state.completed_phases == state.completed_phases
        assert loaded_state.checkpoint_path == state.checkpoint_path
    
    def test_checkpoint_detection_phase(self, basic_config):
        """测试checkpoint检测阶段"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        # 执行checkpoint检测阶段
        controller._phase_checkpoint_detection()
        
        assert controller.export_state.checkpoint_path is not None
        assert controller.export_state.checkpoint_path == basic_config.checkpoint_path
    
    def test_checkpoint_detection_phase_invalid_checkpoint(self, temp_dir):
        """测试无效checkpoint的检测"""
        # 创建无效的checkpoint目录（缺少必需文件）
        invalid_checkpoint = Path(temp_dir) / "invalid_checkpoint"
        invalid_checkpoint.mkdir()
        
        config = ExportConfiguration(
            checkpoint_path=str(invalid_checkpoint),
            base_model_name="test_model",
            output_directory=str(Path(temp_dir) / "output")
        )
        
        controller = ModelExportController(config)
        controller._initialize_export_state("test_export")
        
        with pytest.raises(CheckpointValidationError):
            controller._phase_checkpoint_detection()
    
    def test_model_merging_phase(self, basic_config):
        """测试模型合并阶段"""
        # Mock模型合并器的返回值
        mock_merge_result = {
            'success': True,
            'model_size_mb': 100.0,
            'parameter_count': 1000000,
            'verification_passed': True
        }
        
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        controller.export_state.checkpoint_path = basic_config.checkpoint_path
        
        with patch.object(controller.model_merger, 'merge_and_save', return_value=mock_merge_result):
            controller._phase_model_merging()
        
        assert controller.export_state.merged_model_path is not None
        assert "merged_model" in controller.export_state.merged_model_path
    
    @patch('transformers.AutoModelForCausalLM')
    @patch('transformers.AutoTokenizer')
    def test_optimization_phase(self, mock_tokenizer, mock_model, basic_config):
        """测试模型优化阶段"""
        # 创建mock模型
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        # 设置合并后的模型路径
        merged_path = Path(basic_config.output_directory) / "temp" / "test_export" / "merged_model"
        merged_path.mkdir(parents=True, exist_ok=True)
        controller.export_state.merged_model_path = str(merged_path)
        
        # Mock优化处理器方法
        with patch.object(controller.optimization_processor, 'apply_quantization', return_value=mock_model_instance), \
             patch.object(controller.optimization_processor, 'remove_training_artifacts', return_value=mock_model_instance), \
             patch.object(controller.optimization_processor, 'compress_model_weights', return_value=mock_model_instance):
            
            controller._phase_optimization()
        
        assert controller.export_state.optimized_model_path is not None
        assert "optimized_model" in controller.export_state.optimized_model_path
    
    @patch('transformers.AutoModelForCausalLM')
    @patch('transformers.AutoTokenizer')
    def test_pytorch_export_phase(self, mock_tokenizer, mock_model, basic_config):
        """测试PyTorch导出阶段"""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        # 设置优化后的模型路径
        optimized_path = Path(basic_config.output_directory) / "temp" / "test_export" / "optimized_model"
        optimized_path.mkdir(parents=True, exist_ok=True)
        controller.export_state.optimized_model_path = str(optimized_path)
        
        # Mock格式导出器
        expected_pytorch_path = str(Path(basic_config.output_directory) / "pytorch_model")
        with patch.object(controller.format_exporter, 'export_pytorch_model', return_value=expected_pytorch_path):
            controller._phase_pytorch_export(None)
        
        assert controller.export_state.exported_models["pytorch"] == expected_pytorch_path
    
    @patch('transformers.AutoModelForCausalLM')
    @patch('transformers.AutoTokenizer')
    def test_onnx_export_phase(self, mock_tokenizer, mock_model, basic_config):
        """测试ONNX导出阶段"""
        # 启用ONNX导出
        basic_config.export_onnx = True
        
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        # 设置优化后的模型路径
        optimized_path = Path(basic_config.output_directory) / "temp" / "test_export" / "optimized_model"
        optimized_path.mkdir(parents=True, exist_ok=True)
        controller.export_state.optimized_model_path = str(optimized_path)
        
        # Mock格式导出器
        expected_onnx_path = str(Path(basic_config.output_directory) / "onnx_model")
        with patch.object(controller.format_exporter, 'export_onnx_model', return_value=expected_onnx_path):
            controller._phase_onnx_export(None)
        
        assert controller.export_state.exported_models["onnx"] == expected_onnx_path
    
    def test_validation_phase(self, basic_config):
        """测试验证阶段"""
        # 启用验证测试
        basic_config.run_validation_tests = True
        
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        # 设置导出的模型路径
        controller.export_state.exported_models = {
            "pytorch": "/path/to/pytorch_model",
            "onnx": "/path/to/onnx_model"
        }
        
        # Mock验证测试器
        mock_report = Mock()
        mock_report.success_rate = 0.9
        
        with patch.object(controller.validation_tester, 'run_comprehensive_validation', return_value=mock_report), \
             patch.object(controller.validation_tester, 'generate_validation_report'):
            
            result = Mock()
            controller._phase_validation(result)
        
        assert result.validation_passed == True
        assert result.output_consistency_score == 0.9
    
    def test_should_execute_phase(self, basic_config):
        """测试阶段执行判断"""
        controller = ModelExportController(basic_config)
        
        # 测试基本阶段（总是执行）
        assert controller._should_execute_phase("checkpoint_detection") == True
        assert controller._should_execute_phase("model_merging") == True
        assert controller._should_execute_phase("optimization") == True
        
        # 测试条件阶段
        assert controller._should_execute_phase("pytorch_export") == True  # 配置中启用
        assert controller._should_execute_phase("onnx_export") == False   # 配置中禁用
        assert controller._should_execute_phase("tensorrt_export") == False  # 配置中禁用
        assert controller._should_execute_phase("validation") == False    # 配置中禁用
    
    def test_should_export_format(self, basic_config):
        """测试格式导出判断"""
        controller = ModelExportController(basic_config)
        
        assert controller._should_export_format("pytorch") == True
        assert controller._should_export_format("onnx") == False
        assert controller._should_export_format("tensorrt") == False
        assert controller._should_export_format("unknown") == False
    
    def test_single_phase_execution_with_retry(self, basic_config):
        """测试单阶段执行和重试机制"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        # Mock一个会失败然后成功的阶段
        call_count = 0
        def mock_phase():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("第一次失败")
            # 第二次成功
        
        with patch.object(controller, '_phase_checkpoint_detection', side_effect=mock_phase):
            # 应该重试并最终成功
            controller._execute_single_phase("checkpoint_detection", "test_op", 1, None)
        
        assert call_count == 2
        assert "checkpoint_detection" in controller.export_state.completed_phases
    
    def test_single_phase_execution_max_retries(self, basic_config):
        """测试单阶段执行达到最大重试次数"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        # Mock一个总是失败的阶段
        def mock_phase():
            raise Exception("总是失败")
        
        with patch.object(controller, '_phase_checkpoint_detection', side_effect=mock_phase):
            with pytest.raises(ModelExportError):
                controller._execute_single_phase("checkpoint_detection", "test_op", 1, None)
        
        assert "checkpoint_detection" in controller.export_state.failed_phases
    
    def test_get_export_status(self, basic_config):
        """测试获取导出状态"""
        controller = ModelExportController(basic_config)
        
        # 未初始化时应返回None
        assert controller.get_export_status() is None
        
        # 初始化后应返回状态信息
        controller._initialize_export_state("test_export")
        status = controller.get_export_status()
        
        assert status is not None
        assert status['export_id'] == "test_export"
        assert status['current_phase'] == "initialized"
        assert status['completed_phases'] == []
        assert status['progress_percent'] == 0.0
    
    def test_cancel_export(self, basic_config):
        """测试取消导出"""
        controller = ModelExportController(basic_config)
        controller._initialize_export_state("test_export")
        
        controller.cancel_export()
        
        assert controller.export_state.current_phase == "cancelled"
    
    def test_list_available_checkpoints(self, basic_config):
        """测试列出可用检查点"""
        controller = ModelExportController(basic_config)
        
        # 创建几个测试状态文件
        state_dir = Path(basic_config.output_directory) / "export_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试状态
        state1 = ExportState(export_id="export_1", current_phase="completed")
        state1.start_time = datetime.now()
        state1.save_to_file(str(state_dir / "export_1_state.pkl"))
        
        state2 = ExportState(export_id="export_2", current_phase="failed")
        state2.start_time = datetime.now()
        state2.failed_phases = ["model_merging"]
        state2.save_to_file(str(state_dir / "export_2_state.pkl"))
        
        checkpoints = controller.list_available_checkpoints()
        
        assert len(checkpoints) == 2
        assert any(cp['export_id'] == 'export_1' for cp in checkpoints)
        assert any(cp['export_id'] == 'export_2' for cp in checkpoints)
    
    def test_context_manager(self, basic_config):
        """测试上下文管理器"""
        with ModelExportController(basic_config) as controller:
            assert controller is not None
            assert isinstance(controller, ModelExportController)
        
        # 上下文退出后应该清理资源
        # 这里主要测试不会抛出异常
    
    def test_callback_registration(self, basic_config):
        """测试回调函数注册"""
        controller = ModelExportController(basic_config)
        
        progress_callback = Mock()
        status_callback = Mock()
        
        controller.add_progress_callback(progress_callback)
        controller.add_status_callback(status_callback)
        
        assert progress_callback in controller.progress_callbacks
        assert status_callback in controller.status_callbacks
    
    @patch('src.model_export_controller.ThreadPoolExecutor')
    def test_parallel_export_not_enabled(self, mock_executor, basic_config):
        """测试未启用并发导出时的错误"""
        controller = ModelExportController(basic_config)
        
        with pytest.raises(ModelExportError, match="并发导出未启用"):
            controller.export_multiple_formats_parallel(["pytorch", "onnx"])
    
    def test_parallel_export_enabled(self, full_config):
        """测试启用并发导出"""
        controller = ModelExportController(full_config)
        
        # Mock所有必要的方法
        with patch.object(controller, '_execute_preprocessing_phases'), \
             patch.object(controller, '_export_single_format') as mock_export:
            
            # Mock导出结果
            mock_result = Mock()
            mock_result.success = True
            mock_export.return_value = mock_result
            
            results = controller.export_multiple_formats_parallel(["pytorch", "onnx"])
        
        assert len(results) == 2
        assert "pytorch" in results
        assert "onnx" in results


class TestExportState:
    """导出状态测试类"""
    
    def test_export_state_creation(self):
        """测试导出状态创建"""
        state = ExportState(export_id="test_123")
        
        assert state.export_id == "test_123"
        assert state.current_phase == "initialized"
        assert state.completed_phases == []
        assert state.failed_phases == []
        assert state.retry_count == 0
        assert state.max_retries == 3
    
    def test_mark_phase_completed(self):
        """测试标记阶段完成"""
        state = ExportState(export_id="test_123")
        
        state.mark_phase_completed("checkpoint_detection", 1.5)
        
        assert "checkpoint_detection" in state.completed_phases
        assert state.phase_durations["checkpoint_detection"] == 1.5
    
    def test_mark_phase_failed(self):
        """测试标记阶段失败"""
        state = ExportState(export_id="test_123")
        
        state.mark_phase_failed("model_merging", "合并失败")
        
        assert "model_merging" in state.failed_phases
        assert state.last_error == "合并失败"
        assert state.retry_count == 1
    
    def test_can_retry(self):
        """测试重试判断"""
        state = ExportState(export_id="test_123")
        
        assert state.can_retry() == True
        
        # 达到最大重试次数
        state.retry_count = 3
        assert state.can_retry() == False
    
    def test_mark_phase_completed_removes_failed(self):
        """测试标记完成会移除失败记录"""
        state = ExportState(export_id="test_123")
        
        # 先标记失败
        state.mark_phase_failed("checkpoint_detection", "失败")
        assert "checkpoint_detection" in state.failed_phases
        
        # 再标记完成
        state.mark_phase_completed("checkpoint_detection", 1.0)
        assert "checkpoint_detection" not in state.failed_phases
        assert "checkpoint_detection" in state.completed_phases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])