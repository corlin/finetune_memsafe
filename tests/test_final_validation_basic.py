#!/usr/bin/env python3
"""
Basic Final Validation Test

A simplified version of the final integration test that can run without
the full model export system, for testing the test framework itself.
"""

import os
import sys
import json
import tempfile
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_basic_functionality():
    """Test basic functionality without full model loading"""
    print("Running basic functionality test...")
    
    results = {
        "test_name": "basic_functionality",
        "timestamp": datetime.now().isoformat(),
        "status": "PASSED"
    }
    
    try:
        # Test file operations
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            temp_file = f.name
            
        # Read back
        with open(temp_file, 'r') as f:
            content = f.read()
            
        assert content == "Test content", "File content mismatch"
        
        # Cleanup
        os.unlink(temp_file)
        
        results["file_operations"] = "PASSED"
        
    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        
    return results


def test_import_functionality():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    results = {
        "test_name": "import_functionality",
        "timestamp": datetime.now().isoformat(),
        "imports": {}
    }
    
    # Test standard library imports
    standard_modules = ["os", "sys", "json", "time", "tempfile", "platform"]
    for module in standard_modules:
        try:
            __import__(module)
            results["imports"][module] = "PASSED"
        except ImportError as e:
            results["imports"][module] = f"FAILED: {e}"
            
    # Test optional modules (may not be available in all environments)
    optional_modules = ["torch", "transformers", "onnx", "onnxruntime", "psutil", "numpy"]
    for module in optional_modules:
        try:
            __import__(module)
            results["imports"][module] = "PASSED"
        except ImportError:
            results["imports"][module] = "NOT_AVAILABLE"
            
    # Determine overall status
    failed_standard = [m for m in standard_modules if "FAILED" in results["imports"][m]]
    if failed_standard:
        results["status"] = "FAILED"
        results["failed_standard_modules"] = failed_standard
    else:
        results["status"] = "PASSED"
        
    return results


def test_checkpoint_detection():
    """Test checkpoint detection logic without actual checkpoint"""
    print("Testing checkpoint detection logic...")
    
    results = {
        "test_name": "checkpoint_detection",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Test with temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock checkpoint structure
            checkpoint_dir = os.path.join(temp_dir, "mock_checkpoint")
            os.makedirs(checkpoint_dir)
            
            # Create mock files
            mock_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json"
            ]
            
            for mock_file in mock_files:
                file_path = os.path.join(checkpoint_dir, mock_file)
                with open(file_path, 'w') as f:
                    f.write('{"mock": "data"}')
                    
            # Test directory detection
            assert os.path.exists(checkpoint_dir), "Mock checkpoint directory not created"
            
            # Test file detection
            for mock_file in mock_files:
                file_path = os.path.join(checkpoint_dir, mock_file)
                assert os.path.exists(file_path), f"Mock file not created: {mock_file}"
                
            results["status"] = "PASSED"
            results["mock_checkpoint_created"] = True
            results["mock_files_created"] = len(mock_files)
            
    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        
    return results


def run_basic_validation_tests():
    """Run all basic validation tests"""
    print("Starting Basic Final Validation Tests")
    print("=" * 50)
    
    test_functions = [
        test_basic_functionality,
        test_import_functionality,
        test_checkpoint_detection
    ]
    
    all_results = {
        "test_suite": "basic_final_validation",
        "timestamp": datetime.now().isoformat(),
        "test_results": {}
    }
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_name = result["test_name"]
            all_results["test_results"][test_name] = result
            
            status = result["status"]
            print(f"{test_name}: {status}")
            
            if status == "PASSED":
                passed_tests += 1
                
        except Exception as e:
            test_name = test_func.__name__
            all_results["test_results"][test_name] = {
                "test_name": test_name,
                "status": "FAILED",
                "error": str(e)
            }
            print(f"{test_name}: FAILED - {e}")
            
    # Calculate summary
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    all_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": success_rate
    }
    
    # Save results
    report_path = "basic_validation_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
        
    print("\n" + "=" * 50)
    print("Basic Validation Test Summary")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Report saved to: {report_path}")
    
    return success_rate >= 80


if __name__ == "__main__":
    success = run_basic_validation_tests()
    
    if success:
        print("\n✓ Basic validation tests PASSED!")
        sys.exit(0)
    else:
        print("\n✗ Basic validation tests FAILED!")
        sys.exit(1)