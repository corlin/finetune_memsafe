#!/usr/bin/env python3
"""
UV-based Final Integration Test Executor

This script uses uv to manage dependencies and execute the final integration tests
for the model export optimization system.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_uv_installation():
    """Check if uv is installed and available"""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ UV is available: {result.stdout.strip()}")
            return True
        else:
            print("✗ UV is installed but not working properly")
            return False
    except FileNotFoundError:
        print("✗ UV is not installed")
        print("Please install uv from: https://docs.astral.sh/uv/getting-started/installation/")
        return False


def setup_uv_environment():
    """Setup UV environment and install dependencies"""
    print("Setting up UV environment...")
    
    # Check if pyproject.toml exists
    if not os.path.exists("pyproject.toml"):
        print("Creating pyproject.toml for test dependencies...")
        create_test_pyproject()
    
    # Sync dependencies
    print("Syncing dependencies with uv...")
    try:
        result = subprocess.run(["uv", "sync"], check=True, capture_output=True, text=True)
        print("✓ Dependencies synced successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to sync dependencies: {e}")
        print("Stderr:", e.stderr)
        return False


def create_test_pyproject():
    """Create a pyproject.toml file with test dependencies"""
    pyproject_content = '''[project]
name = "model-export-optimization-tests"
version = "0.1.0"
description = "Final integration tests for model export optimization"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    "psutil>=5.9.0",
    "numpy>=1.24.0",
    "pytest>=7.0.0",
    "accelerate>=0.20.0",
    "peft>=0.4.0",
    "safetensors>=0.3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.uv]
dev-dependencies = [
    "pytest-xdist>=3.0.0",
    "pytest-cov>=4.0.0",
]
'''
    
    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    print("✓ Created pyproject.toml with test dependencies")


def run_basic_validation():
    """Run basic validation tests using uv"""
    print("\n" + "="*60)
    print("RUNNING BASIC VALIDATION TESTS")
    print("="*60)
    
    try:
        result = subprocess.run([
            "uv", "run", "python", "tests/test_final_validation_basic.py"
        ], check=True, capture_output=False, text=True)
        
        print("✓ Basic validation tests completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Basic validation tests failed with exit code: {e.returncode}")
        return False


def run_integration_tests():
    """Run full integration tests using uv"""
    print("\n" + "="*60)
    print("RUNNING FULL INTEGRATION TESTS")
    print("="*60)
    
    try:
        result = subprocess.run([
            "uv", "run", "python", "tests/run_final_integration_tests.py", 
            "--integration"
        ], check=True, capture_output=False, text=True)
        
        print("✓ Integration tests completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Integration tests failed with exit code: {e.returncode}")
        return False


def run_performance_tests():
    """Run performance benchmark tests using uv"""
    print("\n" + "="*60)
    print("RUNNING PERFORMANCE BENCHMARK TESTS")
    print("="*60)
    
    try:
        result = subprocess.run([
            "uv", "run", "python", "tests/run_final_integration_tests.py", 
            "--performance"
        ], check=True, capture_output=False, text=True)
        
        print("✓ Performance tests completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Performance tests failed with exit code: {e.returncode}")
        return False


def run_compatibility_tests():
    """Run compatibility tests using uv"""
    print("\n" + "="*60)
    print("RUNNING COMPATIBILITY TESTS")
    print("="*60)
    
    try:
        result = subprocess.run([
            "uv", "run", "python", "tests/run_final_integration_tests.py", 
            "--compatibility"
        ], check=True, capture_output=False, text=True)
        
        print("✓ Compatibility tests completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Compatibility tests failed with exit code: {e.returncode}")
        return False


def run_all_tests():
    """Run all test suites using uv"""
    print("\n" + "="*60)
    print("RUNNING ALL TEST SUITES")
    print("="*60)
    
    try:
        result = subprocess.run([
            "uv", "run", "python", "tests/run_final_integration_tests.py"
        ], check=True, capture_output=False, text=True)
        
        print("✓ All test suites completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Test suites failed with exit code: {e.returncode}")
        return False


def run_pytest_tests():
    """Run tests using pytest through uv"""
    print("\n" + "="*60)
    print("RUNNING PYTEST TEST SUITE")
    print("="*60)
    
    try:
        result = subprocess.run([
            "uv", "run", "pytest", "tests/", "-v", "--tb=short"
        ], check=True, capture_output=False, text=True)
        
        print("✓ Pytest tests completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Pytest tests failed with exit code: {e.returncode}")
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="UV-based Final Integration Test Executor")
    parser.add_argument("--basic", action="store_true", help="Run basic validation tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--compatibility", action="store_true", help="Run compatibility tests only")
    parser.add_argument("--pytest", action="store_true", help="Run pytest test suite")
    parser.add_argument("--setup-only", action="store_true", help="Setup environment only")
    
    args = parser.parse_args()
    
    print("Model Export Optimization - Final Integration Tests")
    print("Using UV for dependency management")
    print("="*80)
    
    # Check UV installation
    if not check_uv_installation():
        sys.exit(1)
    
    # Setup environment
    if not setup_uv_environment():
        print("Failed to setup UV environment")
        sys.exit(1)
    
    if args.setup_only:
        print("✓ Environment setup completed")
        sys.exit(0)
    
    # Determine which tests to run
    test_results = {}
    
    if args.basic:
        test_results["basic"] = run_basic_validation()
    elif args.integration:
        test_results["integration"] = run_integration_tests()
    elif args.performance:
        test_results["performance"] = run_performance_tests()
    elif args.compatibility:
        test_results["compatibility"] = run_compatibility_tests()
    elif args.pytest:
        test_results["pytest"] = run_pytest_tests()
    else:
        # Run all tests
        test_results["basic"] = run_basic_validation()
        if test_results["basic"]:  # Only continue if basic tests pass
            test_results["all"] = run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name.upper()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
    
    # Exit with appropriate code
    if all(test_results.values()):
        print("🎉 All tests PASSED! System ready for production.")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED. Please review and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()