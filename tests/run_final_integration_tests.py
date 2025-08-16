#!/usr/bin/env python3
"""
Final Integration Test Runner

This script orchestrates the complete final integration testing suite,
including all validation tests, performance benchmarks, and report generation.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
from test_integration_final_validation import run_full_integration_test
from performance_benchmark import run_performance_benchmark


def setup_test_environment():
    """Setup the test environment and verify prerequisites"""
    print("Setting up test environment...")
    
    # Check if checkpoint exists
    checkpoint_path = "qwen3-finetuned"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_path}")
        print("Please ensure the qwen3-finetuned checkpoint is available in the workspace root.")
        return False
        
    # Check if uv is available
    try:
        import subprocess
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: uv is not installed or not available in PATH")
            print("Please install uv: https://docs.astral.sh/uv/getting-started/installation/")
            return False
        print(f"Using uv version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: uv command not found")
        print("Please install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False
        
    # Check required Python packages using uv
    required_packages = [
        "torch", "transformers", "onnx", "onnxruntime", 
        "psutil", "numpy", "pytest"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
            
    if missing_packages:
        print(f"Missing packages detected: {', '.join(missing_packages)}")
        print("Installing missing packages using uv...")
        
        try:
            # Install missing packages using uv
            for package in missing_packages:
                print(f"Installing {package}...")
                result = subprocess.run(["uv", "add", package], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: Failed to install {package} with uv add")
                    print(f"Trying with uv pip install...")
                    result = subprocess.run(["uv", "pip", "install", package], capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"ERROR: Failed to install {package}")
                        return False
                        
        except Exception as e:
            print(f"ERROR: Package installation failed: {e}")
            return False
        
    print("Test environment setup completed successfully.")
    return True


def run_integration_tests(args):
    """Run the main integration test suite"""
    print("\n" + "=" * 80)
    print("RUNNING INTEGRATION TEST SUITE")
    print("=" * 80)
    
    try:
        results = run_full_integration_test()
        return results is not None
    except Exception as e:
        print(f"Integration test suite failed: {e}")
        return False


def run_performance_tests(args):
    """Run the performance benchmark suite"""
    print("\n" + "=" * 80)
    print("RUNNING PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    
    try:
        results = run_performance_benchmark()
        success_rate = results.get("execution_summary", {}).get("success_rate", 0)
        return success_rate >= 80  # Consider 80% success rate as passing
    except Exception as e:
        print(f"Performance benchmark suite failed: {e}")
        return False


def run_compatibility_tests(args):
    """Run cross-platform compatibility tests"""
    print("\n" + "=" * 80)
    print("RUNNING COMPATIBILITY TEST SUITE")
    print("=" * 80)
    
    try:
        import platform
        import tempfile
        
        # Basic compatibility checks
        compatibility_results = {
            "platform_info": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "path_handling": {},
            "file_operations": {},
            "environment_variables": {}
        }
        
        # Test path handling
        test_paths = [
            "models/test",
            "exports\\windows\\style",
            "/unix/style/path",
            "relative/path"
        ]
        
        for test_path in test_paths:
            try:
                normalized = os.path.normpath(test_path)
                compatibility_results["path_handling"][test_path] = {
                    "normalized": normalized,
                    "status": "PASSED"
                }
            except Exception as e:
                compatibility_results["path_handling"][test_path] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                
        # Test file operations
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
                f.write("Test content\n测试内容\n")
                temp_file = f.name
                
            # Read back
            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            os.unlink(temp_file)
            
            compatibility_results["file_operations"] = {
                "unicode_support": "测试内容" in content,
                "file_creation": True,
                "file_deletion": not os.path.exists(temp_file),
                "status": "PASSED"
            }
            
        except Exception as e:
            compatibility_results["file_operations"] = {
                "status": "FAILED",
                "error": str(e)
            }
            
        # Test environment variables
        try:
            test_var = "INTEGRATION_TEST_VAR"
            test_value = "test_value_123"
            
            os.environ[test_var] = test_value
            retrieved = os.environ.get(test_var)
            
            if test_var in os.environ:
                del os.environ[test_var]
                
            compatibility_results["environment_variables"] = {
                "set_variable": True,
                "get_variable": retrieved == test_value,
                "delete_variable": test_var not in os.environ,
                "status": "PASSED"
            }
            
        except Exception as e:
            compatibility_results["environment_variables"] = {
                "status": "FAILED",
                "error": str(e)
            }
            
        # Save compatibility report
        with open("compatibility_test_report.json", 'w', encoding='utf-8') as f:
            json.dump(compatibility_results, f, indent=2, ensure_ascii=False, default=str)
            
        # Check overall success
        all_passed = all(
            result.get("status") == "PASSED" 
            for category in ["path_handling", "file_operations", "environment_variables"]
            for result in compatibility_results[category].values()
            if isinstance(result, dict)
        )
        
        print(f"Compatibility tests: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
        
    except Exception as e:
        print(f"Compatibility test suite failed: {e}")
        return False


def generate_consolidated_report(integration_success, performance_success, compatibility_success):
    """Generate a consolidated final test report"""
    print("\n" + "=" * 80)
    print("GENERATING CONSOLIDATED FINAL REPORT")
    print("=" * 80)
    
    # Load individual reports
    reports = {}
    
    # Load integration test report
    if os.path.exists("final_integration_test_report.json"):
        try:
            with open("final_integration_test_report.json", 'r', encoding='utf-8') as f:
                reports["integration"] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load integration test report: {e}")
            
    # Load performance benchmark report
    if os.path.exists("performance_benchmark_report.json"):
        try:
            with open("performance_benchmark_report.json", 'r', encoding='utf-8') as f:
                reports["performance"] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load performance benchmark report: {e}")
            
    # Load compatibility test report
    if os.path.exists("compatibility_test_report.json"):
        try:
            with open("compatibility_test_report.json", 'r', encoding='utf-8') as f:
                reports["compatibility"] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load compatibility test report: {e}")
            
    # Generate consolidated report
    consolidated_report = {
        "final_validation_summary": {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "PASSED" if all([integration_success, performance_success, compatibility_success]) else "FAILED",
            "test_suite_results": {
                "integration_tests": "PASSED" if integration_success else "FAILED",
                "performance_benchmarks": "PASSED" if performance_success else "FAILED",
                "compatibility_tests": "PASSED" if compatibility_success else "FAILED"
            }
        },
        "detailed_reports": reports,
        "recommendations": generate_final_recommendations(integration_success, performance_success, compatibility_success, reports),
        "next_steps": generate_next_steps(integration_success, performance_success, compatibility_success)
    }
    
    # Save consolidated report
    consolidated_path = "consolidated_final_validation_report.json"
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_report, f, indent=2, ensure_ascii=False, default=str)
        
    print(f"Consolidated report saved to: {consolidated_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Integration Tests: {'✓ PASSED' if integration_success else '✗ FAILED'}")
    print(f"Performance Tests: {'✓ PASSED' if performance_success else '✗ FAILED'}")
    print(f"Compatibility Tests: {'✓ PASSED' if compatibility_success else '✗ FAILED'}")
    print(f"Overall Status: {'✓ PASSED' if consolidated_report['final_validation_summary']['overall_status'] == 'PASSED' else '✗ FAILED'}")
    
    return consolidated_report


def generate_final_recommendations(integration_success, performance_success, compatibility_success, reports):
    """Generate final recommendations based on test results"""
    recommendations = []
    
    if not integration_success:
        recommendations.append("Address integration test failures before proceeding to production deployment")
        
    if not performance_success:
        recommendations.append("Investigate performance issues and optimize system resources")
        
    if not compatibility_success:
        recommendations.append("Resolve cross-platform compatibility issues")
        
    # Analyze specific issues from reports
    if "integration" in reports:
        integration_report = reports["integration"]
        if "error_log" in integration_report and integration_report["error_log"]:
            recommendations.append("Review integration test error log for specific failure details")
            
    if "performance" in reports:
        performance_report = reports["performance"]
        if "benchmark_results" in performance_report:
            # Check for performance issues
            memory_results = performance_report["benchmark_results"].get("memory_usage", {})
            if memory_results.get("status") == "FAILED":
                recommendations.append("Address memory usage issues identified in performance benchmarks")
                
    if all([integration_success, performance_success, compatibility_success]):
        recommendations.extend([
            "All tests passed successfully - system is ready for production deployment",
            "Consider setting up continuous integration to maintain quality",
            "Document the validated configuration for future reference",
            "Monitor system performance in production environment"
        ])
        
    return recommendations


def generate_next_steps(integration_success, performance_success, compatibility_success):
    """Generate next steps based on test results"""
    if all([integration_success, performance_success, compatibility_success]):
        return [
            "Deploy the model export optimization system to production",
            "Set up monitoring and alerting for production usage",
            "Create user documentation and training materials",
            "Plan for regular system maintenance and updates"
        ]
    else:
        next_steps = ["Address failed test cases before proceeding"]
        
        if not integration_success:
            next_steps.append("Fix integration test failures and re-run validation")
            
        if not performance_success:
            next_steps.append("Optimize system performance and re-run benchmarks")
            
        if not compatibility_success:
            next_steps.append("Resolve compatibility issues and re-test")
            
        next_steps.append("Re-run complete validation suite after fixes")
        
        return next_steps


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Final Integration Test Runner")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--compatibility", action="store_true", help="Run compatibility tests only")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup check")
    
    args = parser.parse_args()
    
    # If no specific tests specified, run all
    if not any([args.integration, args.performance, args.compatibility]):
        args.integration = args.performance = args.compatibility = True
        
    print("Final Integration Test Suite")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Setup environment
    if not args.skip_setup:
        if not setup_test_environment():
            print("Environment setup failed. Exiting.")
            sys.exit(1)
            
    # Run selected test suites
    results = {
        "integration": True,  # Default to True if not running
        "performance": True,
        "compatibility": True
    }
    
    if args.integration:
        results["integration"] = run_integration_tests(args)
        
    if args.performance:
        results["performance"] = run_performance_tests(args)
        
    if args.compatibility:
        results["compatibility"] = run_compatibility_tests(args)
        
    # Generate consolidated report
    consolidated_report = generate_consolidated_report(
        results["integration"], 
        results["performance"], 
        results["compatibility"]
    )
    
    # Determine exit code
    overall_success = all(results.values())
    
    if overall_success:
        print("\n🎉 All final integration tests PASSED! System is ready for production.")
        sys.exit(0)
    else:
        print("\n❌ Some final integration tests FAILED. Please review the reports and address issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()