"""
测试运行器

运行所有测试套件并生成报告。
"""

import pytest
import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRunner:
    """测试运行器类"""
    
    def __init__(self, test_dir="tests", output_dir="test_results"):
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_all_tests(self, verbose=True, coverage=True):
        """运行所有测试"""
        print("🚀 开始运行评估系统测试套件...")
        
        start_time = time.time()
        
        # 构建pytest参数
        pytest_args = [
            str(self.test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            f"--junitxml={self.output_dir}/test_results.xml",
            f"--html={self.output_dir}/test_report.html",
            "--self-contained-html"
        ]
        
        if coverage:
            pytest_args.extend([
                "--cov=evaluation",
                f"--cov-report=html:{self.output_dir}/coverage_html",
                f"--cov-report=xml:{self.output_dir}/coverage.xml",
                f"--cov-report=term-missing"
            ])
        
        # 运行测试
        exit_code = pytest.main(pytest_args)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 生成测试摘要
        self._generate_test_summary(exit_code, duration)
        
        return exit_code
    
    def run_unit_tests(self):
        """只运行单元测试"""
        print("🧪 运行单元测试...")
        
        unit_test_files = [
            "test_data_splitter.py",
            "test_evaluation_engine.py", 
            "test_metrics_calculator.py",
            "test_quality_analyzer.py",
            "test_benchmark_manager.py",
            "test_experiment_tracker.py",
            "test_report_generator.py",
            "test_efficiency_analyzer.py",
            "test_config_manager.py"
        ]
        
        pytest_args = [str(self.test_dir / test_file) for test_file in unit_test_files]
        pytest_args.extend(["-v", "--tb=short"])
        
        return pytest.main(pytest_args)
    
    def run_integration_tests(self):
        """只运行集成测试"""
        print("🔗 运行集成测试...")
        
        pytest_args = [
            str(self.test_dir / "test_integration_evaluation.py"),
            "-v",
            "--tb=short"
        ]
        
        return pytest.main(pytest_args)
    
    def run_performance_tests(self):
        """只运行性能测试"""
        print("⚡ 运行性能测试...")
        
        pytest_args = [
            str(self.test_dir / "test_performance_benchmarks.py"),
            "-v",
            "--tb=short",
            "-m", "performance"
        ]
        
        return pytest.main(pytest_args)
    
    def run_specific_test(self, test_file, test_function=None):
        """运行特定测试"""
        test_path = str(self.test_dir / test_file)
        
        if test_function:
            test_path += f"::{test_function}"
        
        pytest_args = [test_path, "-v", "--tb=long"]
        
        return pytest.main(pytest_args)
    
    def run_failed_tests(self):
        """重新运行失败的测试"""
        print("🔄 重新运行失败的测试...")
        
        pytest_args = [
            str(self.test_dir),
            "--lf",  # last-failed
            "-v",
            "--tb=short"
        ]
        
        return pytest.main(pytest_args)
    
    def run_tests_with_markers(self, markers):
        """运行带特定标记的测试"""
        print(f"🏷️  运行标记为 {markers} 的测试...")
        
        pytest_args = [
            str(self.test_dir),
            "-m", markers,
            "-v",
            "--tb=short"
        ]
        
        return pytest.main(pytest_args)
    
    def _generate_test_summary(self, exit_code, duration):
        """生成测试摘要"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "exit_code": exit_code,
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "test_files": self._get_test_files(),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(Path.cwd())
            }
        }
        
        # 保存摘要
        summary_path = self.output_dir / "test_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print(f"\n{'='*60}")
        print(f"📊 测试摘要")
        print(f"{'='*60}")
        print(f"状态: {'✅ 通过' if exit_code == 0 else '❌ 失败'}")
        print(f"耗时: {duration:.2f} 秒")
        print(f"测试文件数: {len(summary['test_files'])}")
        print(f"报告位置: {self.output_dir}")
        print(f"{'='*60}")
    
    def _get_test_files(self):
        """获取所有测试文件"""
        return [f.name for f in self.test_dir.glob("test_*.py")]
    
    def generate_test_matrix(self):
        """生成测试矩阵"""
        test_matrix = {
            "unit_tests": {
                "data_splitter": "test_data_splitter.py",
                "evaluation_engine": "test_evaluation_engine.py",
                "metrics_calculator": "test_metrics_calculator.py",
                "quality_analyzer": "test_quality_analyzer.py",
                "benchmark_manager": "test_benchmark_manager.py",
                "experiment_tracker": "test_experiment_tracker.py",
                "report_generator": "test_report_generator.py",
                "efficiency_analyzer": "test_efficiency_analyzer.py",
                "config_manager": "test_config_manager.py"
            },
            "integration_tests": {
                "evaluation_pipeline": "test_integration_evaluation.py"
            },
            "performance_tests": {
                "benchmarks": "test_performance_benchmarks.py"
            }
        }
        
        matrix_path = self.output_dir / "test_matrix.json"
        with open(matrix_path, 'w', encoding='utf-8') as f:
            json.dump(test_matrix, f, indent=2, ensure_ascii=False)
        
        return test_matrix
    
    def check_test_coverage(self):
        """检查测试覆盖率"""
        print("📈 检查测试覆盖率...")
        
        # 运行覆盖率分析
        pytest_args = [
            str(self.test_dir),
            "--cov=evaluation",
            "--cov-report=term-missing",
            "--cov-report=json",
            "--cov-fail-under=80",  # 要求至少80%覆盖率
            "-q"
        ]
        
        return pytest.main(pytest_args)
    
    def validate_test_environment(self):
        """验证测试环境"""
        print("🔍 验证测试环境...")
        
        required_packages = [
            "pytest",
            "pytest-cov",
            "pytest-html",
            "datasets",
            "numpy",
            "pandas"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 缺少必需的包: {', '.join(missing_packages)}")
            print("请运行: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ 测试环境验证通过")
        return True
    
    def clean_test_artifacts(self):
        """清理测试产物"""
        print("🧹 清理测试产物...")
        
        # 清理缓存目录
        cache_dirs = [
            "__pycache__",
            ".pytest_cache",
            ".coverage"
        ]
        
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                import shutil
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                else:
                    cache_path.unlink()
        
        # 清理测试输出目录
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir()
        
        print("✅ 清理完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="评估系统测试运行器")
    parser.add_argument("--type", choices=["all", "unit", "integration", "performance"], 
                       default="all", help="测试类型")
    parser.add_argument("--coverage", action="store_true", default=True, help="生成覆盖率报告")
    parser.add_argument("--verbose", action="store_true", default=True, help="详细输出")
    parser.add_argument("--clean", action="store_true", help="清理测试产物")
    parser.add_argument("--validate", action="store_true", help="验证测试环境")
    parser.add_argument("--file", help="运行特定测试文件")
    parser.add_argument("--function", help="运行特定测试函数")
    parser.add_argument("--markers", help="运行带特定标记的测试")
    parser.add_argument("--failed", action="store_true", help="重新运行失败的测试")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.clean:
        runner.clean_test_artifacts()
        return
    
    if args.validate:
        if not runner.validate_test_environment():
            sys.exit(1)
        return
    
    # 生成测试矩阵
    runner.generate_test_matrix()
    
    exit_code = 0
    
    if args.file:
        exit_code = runner.run_specific_test(args.file, args.function)
    elif args.failed:
        exit_code = runner.run_failed_tests()
    elif args.markers:
        exit_code = runner.run_tests_with_markers(args.markers)
    elif args.type == "unit":
        exit_code = runner.run_unit_tests()
    elif args.type == "integration":
        exit_code = runner.run_integration_tests()
    elif args.type == "performance":
        exit_code = runner.run_performance_tests()
    else:
        exit_code = runner.run_all_tests(args.verbose, args.coverage)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()