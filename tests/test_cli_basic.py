"""
CLI基础功能测试

测试CLI的基本功能，如参数解析、帮助系统等。
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cli import CLIInterface


class TestCLIBasic:
    """CLI基础测试"""
    
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])