"""
CLI功能测试

测试命令行接口的各种功能，包括参数解析、配置管理、预设管理等。
"""

import os
import sys
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cli import CLIInterface, ConfigPresetManager, ConfigurationWizard
from config_presets import ConfigPresets, ConfigTemplates
from export_models import ExportConfiguration, QuantizationLevel, LogLevel


class TestCLIInterface:
    """CLI接口测试"""
    
    def setup_method(self):
        """测试设置"""
        self.cli = CLIInterface()
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """测试清理"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parser_creation(self):
        """测试解析器创建"""
        parser = self.cli.create_parser()
        
        # 测试基本结构
        assert parser.prog == 'model-export'
        assert 'export' in parser._subparsers._group_actions[0].choices
        assert 'config' in parser._subparsers._group_actions[0].choices
        assert 'validate' in parser._subparsers._group_actions[0].choices
        assert 'wizard' in parser._subparsers._group_actions[0].choices
    
    def test_export_command_parsing(self):
        """测试导出命令解析"""
        parser = self.cli.create_parser()
        
        # 测试基本导出命令
        args = parser.parse_args(['export', '--checkpoint-path', 'test-checkpoint'])
        assert args.command == 'export'
        assert args.checkpoint_path == 'test-checkpoint'
        
        # 测试量化参数
        args = parser.parse_args(['export', '--quantization', 'int4'])
        assert args.quantization == 'int4'
        
        # 测试格式参数
        args = parser.parse_args(['export', '--onnx', '--tensorrt'])
        assert args.onnx is True
        assert args.tensorrt is True
        
        # 测试否定参数
        args = parser.parse_args(['export', '--no-pytorch', '--no-validation'])
        assert args.no_pytorch is True
        assert args.no_validation is True
    
    def test_config_command_parsing(self):
        """测试配置命令解析"""
        parser = self.cli.create_parser()
        
        # 测试创建模板
        args = parser.parse_args(['config', 'create-template', '--output', 'test.yaml'])
        assert args.command == 'config'
        assert args.config_action == 'create-template'
        assert args.output == 'test.yaml'
        
        # 测试验证配置
        args = parser.parse_args(['config', 'validate', 'config.yaml'])
        assert args.config_action == 'validate'
        assert args.config_file == 'config.yaml'
        
        # 测试预设管理
        args = parser.parse_args(['config', 'presets', 'list'])
        assert args.config_action == 'presets'
        assert args.preset_action == 'list'
    
    def test_validate_command_parsing(self):
        """测试验证命令解析"""
        parser = self.cli.create_parser()
        
        args = parser.parse_args([
            'validate', 'model_path',
            '--format', 'onnx',
            '--test-samples', '10',
            '--benchmark',
            '--output-report', 'report.json'
        ])
        
        assert args.command == 'validate'
        assert args.model_path == 'model_path'
        assert args.format == 'onnx'
        assert args.test_samples == 10
        assert args.benchmark is True
        assert args.output_report == 'report.json'
    
    def test_wizard_command_parsing(self):
        """测试向导命令解析"""
        parser = self.cli.create_parser()
        
        args = parser.parse_args(['wizard', '--output', 'config.yaml', '--preset', 'production'])
        assert args.command == 'wizard'
        assert args.output == 'config.yaml'
        assert args.preset == 'production'
    
    @patch('cli.ModelExportController')
    def test_export_command_execution(self, mock_controller_class):
        """测试导出命令执行"""
        # 模拟成功的导出结果
        mock_result = Mock()
        mock_result.success = True
        mock_result.export_id = 'test-export-123'
        mock_result.pytorch_model_path = 'output/pytorch_model'
        mock_result.onnx_model_path = 'output/onnx_model'
        mock_result.tensorrt_model_path = None
        mock_result.original_size_mb = 1000.0
        mock_result.optimized_size_mb = 500.0
        mock_result.size_reduction_percentage = 50.0
        
        mock_controller = Mock()
        mock_controller.export_model.return_value = mock_result
        mock_controller_class.return_value = mock_controller
        
        # 执行导出命令
        result = self.cli.run(['export', '--checkpoint-path', 'test-checkpoint'])
        
        assert result == 0
        mock_controller_class.assert_called_once()
        mock_controller.export_model.assert_called_once()
    
    @patch('cli.ModelExportController')
    def test_export_command_failure(self, mock_controller_class):
        """测试导出命令失败"""
        # 模拟失败的导出结果
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Test error"
        
        mock_controller = Mock()
        mock_controller.export_model.return_value = mock_result
        mock_controller_class.return_value = mock_controller
        
        # 执行导出命令
        result = self.cli.run(['export', '--checkpoint-path', 'test-checkpoint'])
        
        assert result == 1
    
    def test_config_create_template(self):
        """测试创建配置模板"""
        result = self.cli.run(['config', 'create-template', '--output', 'test_template.yaml'])
        
        assert result == 0
        assert Path('test_template.yaml').exists()
        
        # 验证模板内容
        with open('test_template.yaml', 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)
        
        assert 'export' in template_data
        assert 'checkpoint' in template_data['export']
        assert 'optimization' in template_data['export']
    
    def test_config_validate_valid_file(self):
        """测试验证有效配置文件"""
        # 创建有效配置文件
        config_data = {
            'checkpoint_path': 'test-checkpoint',
            'base_model_name': 'test-model',
            'output_directory': 'output',
            'quantization_level': 'int8'
        }
        
        with open('valid_config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        result = self.cli.run(['config', 'validate', 'valid_config.yaml'])
        assert result == 0
    
    def test_config_validate_invalid_file(self):
        """测试验证无效配置文件"""
        # 创建无效配置文件
        with open('invalid_config.yaml', 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: content:')
        
        result = self.cli.run(['config', 'validate', 'invalid_config.yaml'])
        assert result == 1
    
    def test_config_show(self):
        """测试显示配置"""
        # 创建配置文件
        config_data = {
            'checkpoint_path': 'test-checkpoint',
            'base_model_name': 'test-model',
            'output_directory': 'output'
        }
        
        with open('show_config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        result = self.cli.run(['config', 'show', '--config', 'show_config.yaml'])
        assert result == 0
    
    @patch('cli.ValidationTester')
    def test_validate_command(self, mock_validator_class):
        """测试验证命令"""
        # 模拟验证结果
        mock_results = {
            'success': True,
            'avg_inference_time': 100.5,
            'memory_usage': 512.0
        }
        
        mock_validator = Mock()
        mock_validator.test_pytorch_model_functionality.return_value = mock_results
        mock_validator_class.return_value = mock_validator
        
        result = self.cli.run([
            'validate', 'test_model',
            '--format', 'pytorch',
            '--test-samples', '5',
            '--benchmark'
        ])
        
        assert result == 0
        mock_validator.test_pytorch_model_functionality.assert_called_once()
    
    def test_help_display(self):
        """测试帮助显示"""
        # 测试主帮助
        result = self.cli.run([])
        assert result == 1  # 应该显示帮助并退出
        
        # 测试子命令帮助
        with pytest.raises(SystemExit):
            self.cli.run(['export', '--help'])


class TestConfigPresetManager:
    """配置预设管理器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = ConfigPresetManager()
        self.preset_manager.presets_dir = Path(self.temp_dir) / 'presets'
        self.preset_manager.presets_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """测试清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_list_empty_presets(self):
        """测试列出空预设"""
        result = self.preset_manager.list_presets()
        assert result == 0
    
    def test_create_preset(self):
        """测试创建预设"""
        with patch('cli.ConfigurationManager') as mock_config_manager:
            mock_config = Mock()
            mock_manager = Mock()
            mock_manager.load_configuration.return_value = mock_config
            mock_manager._configuration_to_dict.return_value = {'test': 'config'}
            mock_config_manager.return_value = mock_manager
            
            result = self.preset_manager.create_preset('test-preset', 'Test description')
            assert result == 0
            
            preset_file = self.preset_manager.presets_dir / 'test-preset.yaml'
            assert preset_file.exists()
    
    def test_use_preset(self):
        """测试使用预设"""
        # 创建预设文件
        preset_data = {
            'name': 'test-preset',
            'description': 'Test preset',
            'config': {
                'checkpoint_path': 'test-checkpoint',
                'quantization_level': 'int8'
            }
        }
        
        preset_file = self.preset_manager.presets_dir / 'test-preset.yaml'
        with open(preset_file, 'w', encoding='utf-8') as f:
            yaml.dump(preset_data, f)
        
        with patch('cli.ConfigurationManager') as mock_config_manager:
            mock_manager = Mock()
            mock_config_manager.return_value = mock_manager
            
            result = self.preset_manager.use_preset('test-preset', 'output_config.yaml')
            assert result == 0
            
            mock_manager._dict_to_configuration.assert_called_once()
            mock_manager.save_configuration.assert_called_once()
    
    def test_use_nonexistent_preset(self):
        """测试使用不存在的预设"""
        result = self.preset_manager.use_nonexistent_preset('nonexistent')
        assert result == 1


class TestConfigurationWizard:
    """配置向导测试"""
    
    def setup_method(self):
        """测试设置"""
        self.wizard = ConfigurationWizard()
    
    @patch('builtins.input')
    def test_wizard_basic_flow(self, mock_input):
        """测试向导基本流程"""
        # 模拟用户输入
        mock_input.side_effect = [
            '',  # checkpoint_path (使用默认值)
            '',  # base_model_name (使用默认值)
            'my_output',  # output_directory
            '2',  # quantization_level (fp16)
            'y',  # remove_training_artifacts
            'n',  # compress_weights
            'y',  # export_pytorch
            'y',  # export_onnx
            'n',  # export_tensorrt
            '14',  # onnx_opset_version
            'y',  # onnx_optimize_graph
            'y',  # run_validation_tests
            '2',  # log_level (INFO)
            '8.0',  # max_memory_usage_gb
            'n'   # enable_parallel_export
        ]
        
        config = self.wizard.run_wizard()
        
        assert isinstance(config, ExportConfiguration)
        assert config.output_directory == 'my_output'
        assert config.quantization_level == QuantizationLevel.FP16
        assert config.remove_training_artifacts is True
        assert config.compress_weights is False
    
    def test_prompt_input_with_default(self):
        """测试输入提示（使用默认值）"""
        with patch('builtins.input', return_value=''):
            result = self.wizard._prompt_input('Test prompt', 'default_value')
            assert result == 'default_value'
    
    def test_prompt_input_with_custom(self):
        """测试输入提示（自定义值）"""
        with patch('builtins.input', return_value='custom_value'):
            result = self.wizard._prompt_input('Test prompt', 'default_value')
            assert result == 'custom_value'
    
    def test_prompt_bool_default_true(self):
        """测试布尔提示（默认True）"""
        with patch('builtins.input', return_value=''):
            result = self.wizard._prompt_bool('Test prompt', True)
            assert result is True
    
    def test_prompt_bool_explicit_false(self):
        """测试布尔提示（明确False）"""
        with patch('builtins.input', return_value='n'):
            result = self.wizard._prompt_bool('Test prompt', True)
            assert result is False
    
    def test_prompt_choice_by_number(self):
        """测试选择提示（按数字）"""
        choices = ['option1', 'option2', 'option3']
        with patch('builtins.input', return_value='2'):
            result = self.wizard._prompt_choice('Test prompt', choices, 'option1')
            assert result == 'option2'
    
    def test_prompt_choice_by_name(self):
        """测试选择提示（按名称）"""
        choices = ['option1', 'option2', 'option3']
        with patch('builtins.input', return_value='option3'):
            result = self.wizard._prompt_choice('Test prompt', choices, 'option1')
            assert result == 'option3'
    
    def test_prompt_choice_default(self):
        """测试选择提示（使用默认值）"""
        choices = ['option1', 'option2', 'option3']
        with patch('builtins.input', return_value=''):
            result = self.wizard._prompt_choice('Test prompt', choices, 'option2')
            assert result == 'option2'


class TestConfigPresets:
    """配置预设测试"""
    
    def test_get_quick_export_preset(self):
        """测试快速导出预设"""
        preset = ConfigPresets.get_quick_export_preset()
        
        assert preset['name'] == 'quick-export'
        assert 'description' in preset
        assert 'config' in preset
        
        config = preset['config']
        assert config['quantization_level'] == 'fp16'
        assert config['export_pytorch'] is True
        assert config['export_onnx'] is False
        assert config['max_memory_usage_gb'] == 8.0
    
    def test_get_production_preset(self):
        """测试生产环境预设"""
        preset = ConfigPresets.get_production_preset()
        
        assert preset['name'] == 'production'
        config = preset['config']
        assert config['quantization_level'] == 'int8'
        assert config['export_pytorch'] is True
        assert config['export_onnx'] is True
        assert config['enable_parallel_export'] is True
    
    def test_get_mobile_preset(self):
        """测试移动端预设"""
        preset = ConfigPresets.get_mobile_preset()
        
        assert preset['name'] == 'mobile'
        config = preset['config']
        assert config['quantization_level'] == 'int4'
        assert config['max_memory_usage_gb'] == 4.0
        assert config['log_level'] == 'WARNING'
    
    def test_get_research_preset(self):
        """测试研究预设"""
        preset = ConfigPresets.get_research_preset()
        
        assert preset['name'] == 'research'
        config = preset['config']
        assert config['quantization_level'] == 'none'
        assert config['remove_training_artifacts'] is False
        assert config['compress_weights'] is False
        assert config['log_level'] == 'DEBUG'
    
    def test_get_all_presets(self):
        """测试获取所有预设"""
        presets = ConfigPresets.get_all_presets()
        
        assert 'quick-export' in presets
        assert 'production' in presets
        assert 'mobile' in presets
        assert 'research' in presets
        
        assert len(presets) == 4
    
    def test_create_preset_files(self):
        """测试创建预设文件"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            ConfigPresets.create_preset_files(temp_dir)
            
            # 检查文件是否创建
            preset_dir = Path(temp_dir)
            yaml_files = list(preset_dir.glob('*.yaml'))
            json_files = list(preset_dir.glob('*.json'))
            
            assert len(yaml_files) == 4
            assert len(json_files) == 4
            
            # 检查文件内容
            quick_export_file = preset_dir / 'quick-export.yaml'
            assert quick_export_file.exists()
            
            with open(quick_export_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                assert data['name'] == 'quick-export'
                assert 'created_at' in data
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestConfigTemplates:
    """配置模板测试"""
    
    def test_get_basic_template(self):
        """测试基础模板"""
        template = ConfigTemplates.get_basic_template()
        
        assert isinstance(template, str)
        assert 'checkpoint_path' in template
        assert 'quantization_level' in template
        assert 'export_pytorch' in template
    
    def test_get_advanced_template(self):
        """测试高级模板"""
        template = ConfigTemplates.get_advanced_template()
        
        assert isinstance(template, str)
        assert 'export:' in template
        assert 'checkpoint:' in template
        assert 'optimization:' in template
        assert 'formats:' in template
    
    def test_get_docker_template(self):
        """测试Docker模板"""
        template = ConfigTemplates.get_docker_template()
        
        assert isinstance(template, str)
        assert '/app/checkpoints' in template
        assert '/app/output' in template
        assert 'max_memory_usage_gb: 8.0' in template
    
    def test_create_template_files(self):
        """测试创建模板文件"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            ConfigTemplates.create_template_files(temp_dir)
            
            # 检查文件是否创建
            template_dir = Path(temp_dir)
            template_files = list(template_dir.glob('*.yaml'))
            
            assert len(template_files) == 3
            
            # 检查文件内容
            basic_template_file = template_dir / 'basic_config.yaml'
            assert basic_template_file.exists()
            
            with open(basic_template_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert 'checkpoint_path' in content
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCLIIntegration:
    """CLI集成测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """测试清理"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_config_workflow(self):
        """测试完整配置工作流"""
        cli = CLIInterface()
        
        # 1. 创建配置模板
        result = cli.run(['config', 'create-template', '--output', 'template.yaml'])
        assert result == 0
        assert Path('template.yaml').exists()
        
        # 2. 验证配置
        result = cli.run(['config', 'validate', 'template.yaml'])
        assert result == 0
        
        # 3. 显示配置
        result = cli.run(['config', 'show', '--config', 'template.yaml'])
        assert result == 0
    
    @patch('cli.ModelExportController')
    def test_export_with_config_file(self, mock_controller_class):
        """测试使用配置文件导出"""
        # 创建配置文件
        config_data = {
            'checkpoint_path': 'test-checkpoint',
            'base_model_name': 'test-model',
            'output_directory': 'test-output',
            'quantization_level': 'int8'
        }
        
        with open('export_config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        # 模拟成功导出
        mock_result = Mock()
        mock_result.success = True
        mock_result.export_id = 'test-123'
        mock_result.pytorch_model_path = 'output/model'
        mock_result.onnx_model_path = None
        mock_result.tensorrt_model_path = None
        mock_result.original_size_mb = 1000.0
        mock_result.optimized_size_mb = 800.0
        mock_result.size_reduction_percentage = 20.0
        
        mock_controller = Mock()
        mock_controller.export_model.return_value = mock_result
        mock_controller_class.return_value = mock_controller
        
        cli = CLIInterface()
        result = cli.run(['export', '--config', 'export_config.yaml'])
        
        assert result == 0
        mock_controller_class.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])