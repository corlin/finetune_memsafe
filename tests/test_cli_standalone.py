"""
独立CLI功能测试

测试独立CLI工具的各种功能。
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cli_standalone import StandaloneCLI


class TestStandaloneCLI:
    """独立CLI测试"""
    
    def setup_method(self):
        """测试设置"""
        self.cli = StandaloneCLI()
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
        
        # 检查子命令是否存在
        subparsers_actions = [
            action for action in parser._actions 
            if hasattr(action, 'choices')
        ]
        
        if subparsers_actions:
            choices = subparsers_actions[0].choices
            assert 'export' in choices
            assert 'config' in choices
            assert 'validate' in choices
            assert 'wizard' in choices
    
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
        
        # 测试默认值
        args = parser.parse_args(['export'])
        assert args.checkpoint_path == 'qwen3-finetuned'
        assert args.base_model == 'Qwen/Qwen3-4B-Thinking-2507'
        assert args.output_dir == 'exported_models'
        assert args.quantization == 'int8'
        assert args.log_level == 'INFO'
    
    def test_config_command_parsing(self):
        """测试配置命令解析"""
        parser = self.cli.create_parser()
        
        # 测试创建模板
        args = parser.parse_args(['config', 'create-template', '--output', 'test.yaml'])
        assert args.command == 'config'
        assert args.config_action == 'create-template'
        assert args.output == 'test.yaml'
        
        # 测试显示配置
        args = parser.parse_args(['config', 'show'])
        assert args.config_action == 'show'
    
    def test_validate_command_parsing(self):
        """测试验证命令解析"""
        parser = self.cli.create_parser()
        
        args = parser.parse_args([
            'validate', 'model_path',
            '--format', 'onnx',
            '--test-samples', '10',
            '--benchmark'
        ])
        
        assert args.command == 'validate'
        assert args.model_path == 'model_path'
        assert args.format == 'onnx'
        assert args.test_samples == 10
        assert args.benchmark is True
    
    def test_wizard_command_parsing(self):
        """测试向导命令解析"""
        parser = self.cli.create_parser()
        
        args = parser.parse_args(['wizard', '--output', 'config.yaml'])
        assert args.command == 'wizard'
        assert args.output == 'config.yaml'
    
    def test_help_display(self):
        """测试帮助显示"""
        # 测试主帮助
        result = self.cli.run([])
        assert result == 1  # 应该显示帮助并退出
    
    def test_export_command_execution(self):
        """测试导出命令执行"""
        result = self.cli.run(['export', '--checkpoint-path', 'test-checkpoint'])
        assert result == 0
    
    def test_export_command_with_options(self):
        """测试带选项的导出命令"""
        result = self.cli.run([
            'export',
            '--checkpoint-path', 'test-checkpoint',
            '--quantization', 'int4',
            '--onnx',
            '--tensorrt',
            '--parallel'
        ])
        assert result == 0
    
    def test_config_create_template(self):
        """测试创建配置模板"""
        result = self.cli.run(['config', 'create-template', '--output', 'test_template.yaml'])
        
        assert result == 0
        assert Path('test_template.yaml').exists()
        
        # 验证模板内容
        with open('test_template.yaml', 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'export:' in content
            assert 'checkpoint:' in content
            assert 'optimization:' in content
    
    def test_config_show(self):
        """测试显示配置"""
        result = self.cli.run(['config', 'show'])
        assert result == 0
    
    def test_validate_command_execution(self):
        """测试验证命令执行"""
        result = self.cli.run(['validate', 'test_model', '--format', 'pytorch'])
        assert result == 0
    
    def test_validate_command_with_benchmark(self):
        """测试带基准测试的验证命令"""
        result = self.cli.run([
            'validate', 'test_model',
            '--format', 'onnx',
            '--test-samples', '10',
            '--benchmark'
        ])
        assert result == 0
    
    def test_wizard_command_execution(self):
        """测试向导命令执行"""
        result = self.cli.run(['wizard', '--output', 'wizard_config.yaml'])
        
        assert result == 0
        assert Path('wizard_config.yaml').exists()
        
        # 验证配置内容
        with open('wizard_config.yaml', 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'checkpoint_path' in content
            assert 'quantization_level' in content
    
    def test_unknown_command(self):
        """测试未知命令"""
        result = self.cli.run(['unknown-command'])
        assert result == 1
    
    def test_config_unknown_action(self):
        """测试未知配置操作"""
        result = self.cli.run(['config', 'unknown-action'])
        assert result == 1
    
    def test_config_no_action(self):
        """测试配置命令无操作"""
        result = self.cli.run(['config'])
        assert result == 1
    
    def test_export_format_logic(self):
        """测试导出格式逻辑"""
        # 测试默认格式（应该是PyTorch）
        parser = self.cli.create_parser()
        args = parser.parse_args(['export'])
        
        # 模拟格式选择逻辑
        formats = []
        if args.pytorch or not (args.no_pytorch or args.onnx or args.tensorrt):
            formats.append('PyTorch')
        if args.onnx and not args.no_onnx:
            formats.append('ONNX')
        if args.tensorrt:
            formats.append('TensorRT')
        
        assert 'PyTorch' in formats
        
        # 测试明确指定ONNX
        args = parser.parse_args(['export', '--onnx'])
        formats = []
        if args.pytorch or not (args.no_pytorch or args.onnx or args.tensorrt):
            formats.append('PyTorch')
        if args.onnx and not args.no_onnx:
            formats.append('ONNX')
        if args.tensorrt:
            formats.append('TensorRT')
        
        assert 'ONNX' in formats
    
    def test_usage_examples(self):
        """测试使用示例"""
        examples = self.cli._get_usage_examples()
        
        assert 'python cli_standalone.py export' in examples
        assert 'checkpoint-path' in examples
        assert 'quantization' in examples
        assert 'EXPORT_CHECKPOINT_PATH' in examples


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
    
    def test_full_workflow(self):
        """测试完整工作流"""
        cli = StandaloneCLI()
        
        # 1. 创建配置模板
        result = cli.run(['config', 'create-template', '--output', 'template.yaml'])
        assert result == 0
        assert Path('template.yaml').exists()
        
        # 2. 显示配置
        result = cli.run(['config', 'show'])
        assert result == 0
        
        # 3. 模拟导出
        result = cli.run(['export', '--checkpoint-path', 'test-checkpoint'])
        assert result == 0
        
        # 4. 模拟验证
        result = cli.run(['validate', 'test-model', '--benchmark'])
        assert result == 0
        
        # 5. 创建向导配置
        result = cli.run(['wizard', '--output', 'wizard.yaml'])
        assert result == 0
        assert Path('wizard.yaml').exists()
    
    def test_different_export_scenarios(self):
        """测试不同导出场景"""
        cli = StandaloneCLI()
        
        scenarios = [
            ['export', '--quantization', 'fp16'],
            ['export', '--quantization', 'int8', '--onnx'],
            ['export', '--quantization', 'int4', '--tensorrt'],
            ['export', '--quantization', 'none', '--no-compression'],
            ['export', '--parallel', '--log-level', 'DEBUG']
        ]
        
        for scenario in scenarios:
            result = cli.run(scenario)
            assert result == 0
    
    def test_validation_scenarios(self):
        """测试不同验证场景"""
        cli = StandaloneCLI()
        
        scenarios = [
            ['validate', 'model1', '--format', 'pytorch'],
            ['validate', 'model2', '--format', 'onnx', '--benchmark'],
            ['validate', 'model3', '--test-samples', '20'],
            ['validate', 'model4', '--format', 'tensorrt', '--benchmark']
        ]
        
        for scenario in scenarios:
            result = cli.run(scenario)
            assert result == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])