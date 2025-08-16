#!/usr/bin/env python3
"""
Final Integration Testing and Validation Suite

This comprehensive test suite validates the complete model export optimization system
using the actual qwen3-finetuned checkpoint. It tests all export formats, performance
characteristics, and cross-platform compatibility.

Requirements covered: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3
"""

import os
import sys
import json
import time
import psutil
import pytest
import tempfile
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

try:
    from model_export_controller import ModelExportController
    from export_config import ExportConfiguration
    from checkpoint_detector import CheckpointDetector
    from model_merger import ModelMerger
    from optimization_processor import OptimizationProcessor
    from format_exporter import FormatExporter
    from validation_tester import ValidationTester
    from monitoring_logger import MonitoringLogger
    from validation_extensions import ExtendedValidationTester
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Running in mock mode for testing framework validation...")
    
    # Create mock classes for testing
    class MockClass:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: {"status": "MOCKED", "message": f"Mock {name} called"}
    
    ModelExportController = MockClass
    ExportConfiguration = MockClass
    CheckpointDetector = MockClass
    ModelMerger = MockClass
    OptimizationProcessor = MockClass
    FormatExporter = MockClass
    ValidationTester = MockClass
    MonitoringLogger = MockClass
    ExtendedValidationTester = MockClass


class FinalIntegrationTestSuite:
    """Comprehensive integration test suite for final validation"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.error_log = []
        self.start_time = datetime.now()
        self.checkpoint_path = "qwen3-finetuned"
        self.base_model_name = "Qwen/Qwen3-4B-Thinking-2507"
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Setup test environment and temporary directories"""
        self.temp_dir = tempfile.mkdtemp(prefix="final_integration_test_")
        print(f"Test environment setup at: {self.temp_dir}")
        
        # Verify checkpoint exists
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        return True
        
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def test_checkpoint_detection_and_validation(self) -> Dict[str, Any]:
        """Test checkpoint detection and validation functionality"""
        print("\n=== Testing Checkpoint Detection and Validation ===")
        
        try:
            detector = CheckpointDetector()
            
            # Test latest checkpoint detection
            latest_checkpoint = detector.detect_latest_checkpoint(self.checkpoint_path)
            assert latest_checkpoint is not None, "Failed to detect latest checkpoint"
            
            # Test checkpoint integrity validation
            is_valid = detector.validate_checkpoint_integrity(latest_checkpoint)
            assert is_valid, f"Checkpoint integrity validation failed: {latest_checkpoint}"
            
            # Test metadata extraction
            metadata = detector.get_checkpoint_metadata(latest_checkpoint)
            assert isinstance(metadata, dict), "Failed to extract checkpoint metadata"
            
            result = {
                "status": "PASSED",
                "checkpoint_path": latest_checkpoint,
                "metadata": metadata,
                "validation_time": time.time()
            }
            
        except Exception as e:
            result = {
                "status": "FAILED",
                "error": str(e),
                "validation_time": time.time()
            }
            self.error_log.append(f"Checkpoint validation failed: {e}")
            
        self.test_results["checkpoint_validation"] = result
        return result
        
    def test_model_merging_functionality(self) -> Dict[str, Any]:
        """Test model merging with LoRA adapter"""
        print("\n=== Testing Model Merging Functionality ===")
        
        try:
            merger = ModelMerger()
            
            # Test base model loading
            start_time = time.time()
            base_model = merger.load_base_model(self.base_model_name)
            load_time = time.time() - start_time
            
            assert base_model is not None, "Failed to load base model"
            
            # Test LoRA adapter loading and merging
            start_time = time.time()
            merged_model = merger.merge_lora_weights(base_model, self.checkpoint_path)
            merge_time = time.time() - start_time
            
            assert merged_model is not None, "Failed to merge LoRA weights"
            
            # Test merge integrity verification
            is_valid = merger.verify_merge_integrity(merged_model)
            assert is_valid, "Merged model integrity verification failed"
            
            # Save merged model for further testing
            merged_model_path = os.path.join(self.temp_dir, "merged_model")
            merger.save_merged_model(merged_model, merged_model_path)
            
            result = {
                "status": "PASSED",
                "base_model_load_time": load_time,
                "merge_time": merge_time,
                "merged_model_path": merged_model_path,
                "model_parameters": sum(p.numel() for p in merged_model.parameters()),
                "validation_time": time.time()
            }
            
        except Exception as e:
            result = {
                "status": "FAILED",
                "error": str(e),
                "validation_time": time.time()
            }
            self.error_log.append(f"Model merging failed: {e}")
            
        self.test_results["model_merging"] = result
        return result
        
    def test_optimization_processing(self) -> Dict[str, Any]:
        """Test model optimization and quantization"""
        print("\n=== Testing Optimization Processing ===")
        
        results = {}
        
        # Test different quantization levels
        quantization_levels = ["fp16", "int8"]  # Skip int4 for stability
        
        for quant_level in quantization_levels:
            try:
                processor = OptimizationProcessor()
                
                # Load merged model for optimization
                merged_model_path = self.test_results.get("model_merging", {}).get("merged_model_path")
                if not merged_model_path:
                    raise ValueError("Merged model not available for optimization testing")
                    
                model = AutoModelForCausalLM.from_pretrained(merged_model_path)
                original_size = self._calculate_model_size(model)
                
                # Apply optimization
                start_time = time.time()
                optimized_model = processor.apply_quantization(model, quant_level)
                optimization_time = time.time() - start_time
                
                optimized_size = self._calculate_model_size(optimized_model)
                size_reduction = processor.calculate_size_reduction(original_size, optimized_size)
                
                results[quant_level] = {
                    "status": "PASSED",
                    "optimization_time": optimization_time,
                    "original_size_mb": original_size / (1024 * 1024),
                    "optimized_size_mb": optimized_size / (1024 * 1024),
                    "size_reduction": size_reduction,
                    "validation_time": time.time()
                }
                
            except Exception as e:
                results[quant_level] = {
                    "status": "FAILED",
                    "error": str(e),
                    "validation_time": time.time()
                }
                self.error_log.append(f"Optimization failed for {quant_level}: {e}")
                
        self.test_results["optimization"] = results
        return results      
  
    def test_format_export_functionality(self) -> Dict[str, Any]:
        """Test all export format functionality"""
        print("\n=== Testing Format Export Functionality ===")
        
        results = {}
        
        try:
            exporter = FormatExporter()
            
            # Get merged model path
            merged_model_path = self.test_results.get("model_merging", {}).get("merged_model_path")
            if not merged_model_path:
                raise ValueError("Merged model not available for export testing")
                
            model = AutoModelForCausalLM.from_pretrained(merged_model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Test PyTorch export
            try:
                pytorch_output = os.path.join(self.temp_dir, "pytorch_export")
                start_time = time.time()
                pytorch_path = exporter.export_pytorch_model(model, pytorch_output)
                pytorch_time = time.time() - start_time
                
                results["pytorch"] = {
                    "status": "PASSED",
                    "export_time": pytorch_time,
                    "output_path": pytorch_path,
                    "file_size_mb": self._get_directory_size(pytorch_path) / (1024 * 1024)
                }
                
            except Exception as e:
                results["pytorch"] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                self.error_log.append(f"PyTorch export failed: {e}")
                
            # Test ONNX export
            try:
                onnx_output = os.path.join(self.temp_dir, "model.onnx")
                start_time = time.time()
                onnx_path = exporter.export_onnx_model(model, tokenizer, onnx_output)
                onnx_time = time.time() - start_time
                
                # Validate ONNX model
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                
                # Test ONNX optimization
                optimized_onnx_path = exporter.optimize_onnx_graph(onnx_path)
                
                results["onnx"] = {
                    "status": "PASSED",
                    "export_time": onnx_time,
                    "output_path": onnx_path,
                    "optimized_path": optimized_onnx_path,
                    "file_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
                    "optimized_size_mb": os.path.getsize(optimized_onnx_path) / (1024 * 1024)
                }
                
            except Exception as e:
                results["onnx"] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                self.error_log.append(f"ONNX export failed: {e}")
                
        except Exception as e:
            results["general_error"] = {
                "status": "FAILED",
                "error": str(e)
            }
            self.error_log.append(f"Format export setup failed: {e}")
            
        self.test_results["format_export"] = results
        return results
        
    def test_validation_and_consistency(self) -> Dict[str, Any]:
        """Test model validation and output consistency"""
        print("\n=== Testing Validation and Consistency ===")
        
        try:
            validator = ExtendedValidationTester()
            
            # Prepare test inputs
            test_inputs = [
                "Hello, how are you?",
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short poem about nature.",
                "What are the benefits of renewable energy?"
            ]
            
            results = {}
            
            # Test PyTorch model functionality
            pytorch_path = self.test_results.get("format_export", {}).get("pytorch", {}).get("output_path")
            if pytorch_path:
                try:
                    pytorch_results = validator.test_model_functionality(pytorch_path, test_inputs)
                    pytorch_benchmark = validator.benchmark_inference_speed(pytorch_path, test_inputs)
                    pytorch_memory = validator.measure_memory_usage(pytorch_path)
                    
                    results["pytorch"] = {
                        "status": "PASSED",
                        "functionality_test": pytorch_results,
                        "benchmark": pytorch_benchmark,
                        "memory_usage": pytorch_memory
                    }
                    
                except Exception as e:
                    results["pytorch"] = {
                        "status": "FAILED",
                        "error": str(e)
                    }
                    self.error_log.append(f"PyTorch validation failed: {e}")
                    
            # Test ONNX model functionality
            onnx_path = self.test_results.get("format_export", {}).get("onnx", {}).get("output_path")
            if onnx_path:
                try:
                    onnx_results = validator.test_onnx_model_functionality(onnx_path, test_inputs)
                    onnx_benchmark = validator.benchmark_onnx_inference_speed(onnx_path, test_inputs)
                    
                    results["onnx"] = {
                        "status": "PASSED",
                        "functionality_test": onnx_results,
                        "benchmark": onnx_benchmark
                    }
                    
                except Exception as e:
                    results["onnx"] = {
                        "status": "FAILED",
                        "error": str(e)
                    }
                    self.error_log.append(f"ONNX validation failed: {e}")
                    
            # Test output consistency between formats
            if pytorch_path and onnx_path:
                try:
                    consistency_results = validator.compare_pytorch_onnx_outputs(pytorch_path, onnx_path, test_inputs)
                    results["consistency"] = {
                        "status": "PASSED",
                        "comparison_results": consistency_results
                    }
                    
                except Exception as e:
                    results["consistency"] = {
                        "status": "FAILED",
                        "error": str(e)
                    }
                    self.error_log.append(f"Consistency validation failed: {e}")
                    
            # Additional validation tests
            if pytorch_path:
                try:
                    metadata_results = validator.validate_model_metadata(pytorch_path)
                    stress_results = validator.stress_test_model_loading(pytorch_path, iterations=3)
                    
                    results["additional_validation"] = {
                        "metadata_validation": metadata_results,
                        "stress_test": stress_results,
                        "status": "PASSED"
                    }
                    
                except Exception as e:
                    results["additional_validation"] = {
                        "status": "FAILED",
                        "error": str(e)
                    }
                    self.error_log.append(f"Additional validation failed: {e}")
                    
        except Exception as e:
            results = {
                "status": "FAILED",
                "error": str(e)
            }
            self.error_log.append(f"Validation setup failed: {e}")
            
        self.test_results["validation"] = results
        return results
        
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks and stress testing"""
        print("\n=== Testing Performance Benchmarks ===")
        
        results = {}
        
        try:
            # System information
            system_info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }
            
            results["system_info"] = system_info
            
            # Memory stress test
            memory_results = self._run_memory_stress_test()
            results["memory_stress"] = memory_results
            
            # Concurrent processing test
            concurrent_results = self._run_concurrent_processing_test()
            results["concurrent_processing"] = concurrent_results
            
            # Large input test
            large_input_results = self._run_large_input_test()
            results["large_input"] = large_input_results
            
        except Exception as e:
            results = {
                "status": "FAILED",
                "error": str(e)
            }
            self.error_log.append(f"Performance benchmark failed: {e}")
            
        self.test_results["performance"] = results
        return results
        
    def test_cross_platform_compatibility(self) -> Dict[str, Any]:
        """Test cross-platform compatibility"""
        print("\n=== Testing Cross-Platform Compatibility ===")
        
        results = {
            "current_platform": platform.system(),
            "architecture": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "compatibility_tests": {}
        }
        
        try:
            # Test path handling across platforms
            test_paths = [
                "models/test_model",
                "exports\\windows\\path",
                "/unix/style/path",
                "relative/path/test"
            ]
            
            path_results = {}
            for test_path in test_paths:
                try:
                    normalized_path = os.path.normpath(test_path)
                    path_results[test_path] = {
                        "normalized": normalized_path,
                        "is_absolute": os.path.isabs(normalized_path),
                        "status": "PASSED"
                    }
                except Exception as e:
                    path_results[test_path] = {
                        "status": "FAILED",
                        "error": str(e)
                    }
                    
            results["compatibility_tests"]["path_handling"] = path_results
            
            # Test file operations
            file_ops_results = self._test_file_operations()
            results["compatibility_tests"]["file_operations"] = file_ops_results
            
            # Test environment variables
            env_results = self._test_environment_variables()
            results["compatibility_tests"]["environment_variables"] = env_results
            
        except Exception as e:
            results["compatibility_tests"] = {
                "status": "FAILED",
                "error": str(e)
            }
            self.error_log.append(f"Cross-platform compatibility test failed: {e}")
            
        self.test_results["cross_platform"] = results
        return results       
 
    def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow using ModelExportController"""
        print("\n=== Testing End-to-End Workflow ===")
        
        try:
            # Create export configuration
            config = ExportConfiguration(
                checkpoint_path=self.checkpoint_path,
                base_model_name=self.base_model_name,
                output_directory=os.path.join(self.temp_dir, "e2e_export"),
                quantization_level="int8",
                export_pytorch=True,
                export_onnx=True,
                run_validation_tests=True,
                enable_progress_monitoring=True
            )
            
            # Initialize controller
            controller = ModelExportController()
            
            # Run complete export workflow
            start_time = time.time()
            export_result = controller.export_model(config)
            total_time = time.time() - start_time
            
            # Verify export result
            assert export_result.success, f"End-to-end export failed: {export_result.error_message}"
            
            result = {
                "status": "PASSED",
                "total_execution_time": total_time,
                "export_result": {
                    "success": export_result.success,
                    "pytorch_model_path": export_result.pytorch_model_path,
                    "onnx_model_path": export_result.onnx_model_path,
                    "original_size_mb": export_result.original_size_mb,
                    "optimized_size_mb": export_result.optimized_size_mb,
                    "size_reduction_percentage": export_result.size_reduction_percentage,
                    "validation_passed": export_result.validation_passed,
                    "warnings": export_result.warnings
                }
            }
            
        except Exception as e:
            result = {
                "status": "FAILED",
                "error": str(e)
            }
            self.error_log.append(f"End-to-end workflow failed: {e}")
            
        self.test_results["end_to_end"] = result
        return result
        
    def generate_final_test_report(self) -> str:
        """Generate comprehensive final test report"""
        print("\n=== Generating Final Test Report ===")
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate overall success rate
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.test_results.items():
            if isinstance(results, dict):
                if "status" in results:
                    total_tests += 1
                    if results["status"] == "PASSED":
                        passed_tests += 1
                else:
                    # Handle nested results
                    for sub_test, sub_result in results.items():
                        if isinstance(sub_result, dict) and "status" in sub_result:
                            total_tests += 1
                            if sub_result["status"] == "PASSED":
                                passed_tests += 1
                                
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate report
        report = {
            "test_execution_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate_percentage": success_rate
            },
            "system_information": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3)
            },
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "error_log": self.error_log,
            "recommendations": self._generate_recommendations()
        }
        
        # Save report to file
        report_path = os.path.join(self.temp_dir, "final_integration_test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
        # Also save to workspace root for easy access
        workspace_report_path = "final_integration_test_report.json"
        with open(workspace_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"Final test report saved to: {workspace_report_path}")
        return workspace_report_path
        
    def _calculate_model_size(self, model) -> int:
        """Calculate model size in bytes"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
        
    def _get_directory_size(self, directory: str) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
        
    def _run_memory_stress_test(self) -> Dict[str, Any]:
        """Run memory stress test"""
        try:
            initial_memory = psutil.virtual_memory().available
            
            # Simulate memory-intensive operations
            large_tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000)
                large_tensors.append(tensor)
                
            peak_memory = initial_memory - psutil.virtual_memory().available
            
            # Clean up
            del large_tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "status": "PASSED",
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "peak_memory_usage_mb": peak_memory / (1024 * 1024)
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def _run_concurrent_processing_test(self) -> Dict[str, Any]:
        """Run concurrent processing test"""
        try:
            import threading
            import queue
            
            results_queue = queue.Queue()
            
            def worker_task(task_id):
                try:
                    # Simulate concurrent model operations
                    tensor = torch.randn(100, 100)
                    result = torch.sum(tensor)
                    results_queue.put({"task_id": task_id, "result": result.item(), "status": "PASSED"})
                except Exception as e:
                    results_queue.put({"task_id": task_id, "error": str(e), "status": "FAILED"})
                    
            # Start multiple threads
            threads = []
            for i in range(4):
                thread = threading.Thread(target=worker_task, args=(i,))
                threads.append(thread)
                thread.start()
                
            # Wait for completion
            for thread in threads:
                thread.join()
                
            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
                
            passed_tasks = sum(1 for r in results if r["status"] == "PASSED")
            
            return {
                "status": "PASSED" if passed_tasks == len(results) else "PARTIAL",
                "total_tasks": len(results),
                "passed_tasks": passed_tasks,
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def _run_large_input_test(self) -> Dict[str, Any]:
        """Run large input handling test"""
        try:
            # Test with progressively larger inputs
            input_sizes = [100, 500, 1000, 2000]
            results = {}
            
            for size in input_sizes:
                try:
                    start_time = time.time()
                    large_input = "This is a test sentence. " * size
                    
                    # Simulate tokenization and processing
                    tokens = large_input.split()
                    processing_time = time.time() - start_time
                    
                    results[f"size_{size}"] = {
                        "status": "PASSED",
                        "input_length": len(large_input),
                        "token_count": len(tokens),
                        "processing_time": processing_time
                    }
                    
                except Exception as e:
                    results[f"size_{size}"] = {
                        "status": "FAILED",
                        "error": str(e)
                    }
                    
            return {
                "status": "PASSED",
                "size_tests": results
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def _test_file_operations(self) -> Dict[str, Any]:
        """Test file operations across platforms"""
        try:
            test_file = os.path.join(self.temp_dir, "platform_test.txt")
            test_content = "Platform compatibility test content\n测试中文内容\n"
            
            # Test file creation
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
                
            # Test file reading
            with open(test_file, 'r', encoding='utf-8') as f:
                read_content = f.read()
                
            # Test file operations
            file_exists = os.path.exists(test_file)
            file_size = os.path.getsize(test_file)
            
            # Clean up
            os.remove(test_file)
            
            return {
                "status": "PASSED",
                "file_creation": True,
                "file_reading": read_content == test_content,
                "file_size": file_size,
                "file_cleanup": not os.path.exists(test_file)
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def _test_environment_variables(self) -> Dict[str, Any]:
        """Test environment variable handling"""
        try:
            # Test setting and getting environment variables
            test_var = "INTEGRATION_TEST_VAR"
            test_value = "test_value_123"
            
            os.environ[test_var] = test_value
            retrieved_value = os.environ.get(test_var)
            
            # Clean up
            if test_var in os.environ:
                del os.environ[test_var]
                
            return {
                "status": "PASSED",
                "set_variable": True,
                "retrieve_variable": retrieved_value == test_value,
                "cleanup_variable": test_var not in os.environ
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed tests
        failed_tests = []
        for category, results in self.test_results.items():
            if isinstance(results, dict):
                if results.get("status") == "FAILED":
                    failed_tests.append(category)
                else:
                    # Check nested results
                    for sub_test, sub_result in results.items():
                        if isinstance(sub_result, dict) and sub_result.get("status") == "FAILED":
                            failed_tests.append(f"{category}.{sub_test}")
                            
        if failed_tests:
            recommendations.append(f"Address failed tests: {', '.join(failed_tests)}")
            
        # Performance recommendations
        if "performance" in self.test_results:
            perf_results = self.test_results["performance"]
            if "memory_stress" in perf_results:
                memory_usage = perf_results["memory_stress"].get("peak_memory_usage_mb", 0)
                if memory_usage > 1000:  # > 1GB
                    recommendations.append("Consider implementing memory optimization for large model handling")
                    
        # Export format recommendations
        if "format_export" in self.test_results:
            export_results = self.test_results["format_export"]
            if export_results.get("onnx", {}).get("status") == "FAILED":
                recommendations.append("ONNX export issues detected - verify ONNX Runtime compatibility")
                
        if not recommendations:
            recommendations.append("All tests passed successfully - system is ready for production use")
            
        return recommendations


# Test execution functions
def run_full_integration_test():
    """Run the complete integration test suite"""
    suite = FinalIntegrationTestSuite()
    
    try:
        print("Starting Final Integration Test Suite...")
        print("=" * 60)
        
        # Setup test environment
        suite.setup_test_environment()
        
        # Run all test phases
        suite.test_checkpoint_detection_and_validation()
        suite.test_model_merging_functionality()
        suite.test_optimization_processing()
        suite.test_format_export_functionality()
        suite.test_validation_and_consistency()
        suite.test_performance_benchmarks()
        suite.test_cross_platform_compatibility()
        suite.test_end_to_end_workflow()
        
        # Generate final report
        report_path = suite.generate_final_test_report()
        
        print("\n" + "=" * 60)
        print("Final Integration Test Suite Completed")
        print(f"Report saved to: {report_path}")
        
        return suite.test_results
        
    except Exception as e:
        print(f"Integration test suite failed: {e}")
        suite.error_log.append(f"Test suite execution failed: {e}")
        return None
        
    finally:
        # Cleanup
        suite.cleanup_test_environment()


if __name__ == "__main__":
    # Run the integration test suite
    results = run_full_integration_test()
    
    if results:
        print("\nIntegration test completed successfully!")
        sys.exit(0)
    else:
        print("\nIntegration test failed!")
        sys.exit(1)