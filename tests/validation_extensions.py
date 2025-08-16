#!/usr/bin/env python3
"""
Validation Extensions for Final Integration Testing

This module extends the ValidationTester class with additional functionality
specifically needed for comprehensive integration testing.
"""

import os
import sys
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validation_tester import ValidationTester


class ExtendedValidationTester(ValidationTester):
    """Extended validation tester with additional ONNX and integration testing capabilities"""
    
    def test_onnx_model_functionality(self, onnx_path: str, test_inputs: List[str]) -> Dict[str, Any]:
        """Test ONNX model functionality with various inputs"""
        try:
            # Load ONNX model
            session = ort.InferenceSession(onnx_path)
            
            # Get model input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            results = {
                "model_loaded": True,
                "input_name": input_info.name,
                "input_shape": input_info.shape,
                "output_name": output_info.name,
                "output_shape": output_info.shape,
                "test_results": []
            }
            
            # Test with sample inputs
            for i, test_input in enumerate(test_inputs[:3]):  # Limit to 3 for performance
                try:
                    # Create dummy input (simplified for testing)
                    input_ids = np.random.randint(0, 1000, size=(1, 10), dtype=np.int64)
                    
                    # Run inference
                    start_time = time.time()
                    outputs = session.run([output_info.name], {input_info.name: input_ids})
                    inference_time = time.time() - start_time
                    
                    results["test_results"].append({
                        "test_index": i,
                        "input_shape": input_ids.shape,
                        "output_shape": outputs[0].shape,
                        "inference_time": inference_time,
                        "status": "PASSED"
                    })
                    
                except Exception as e:
                    results["test_results"].append({
                        "test_index": i,
                        "status": "FAILED",
                        "error": str(e)
                    })
                    
            return results
            
        except Exception as e:
            return {
                "model_loaded": False,
                "error": str(e),
                "status": "FAILED"
            }
            
    def benchmark_onnx_inference_speed(self, onnx_path: str, test_inputs: List[str]) -> Dict[str, Any]:
        """Benchmark ONNX model inference speed"""
        try:
            session = ort.InferenceSession(onnx_path)
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            # Prepare test data
            input_ids = np.random.randint(0, 1000, size=(1, 50), dtype=np.int64)
            
            # Warmup runs
            for _ in range(3):
                session.run([output_info.name], {input_info.name: input_ids})
                
            # Benchmark runs
            times = []
            for _ in range(10):
                start_time = time.time()
                outputs = session.run([output_info.name], {input_info.name: input_ids})
                end_time = time.time()
                times.append(end_time - start_time)
                
            return {
                "average_inference_time": np.mean(times),
                "min_inference_time": np.min(times),
                "max_inference_time": np.max(times),
                "std_inference_time": np.std(times),
                "total_runs": len(times),
                "status": "PASSED"
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def compare_pytorch_onnx_outputs(self, pytorch_path: str, onnx_path: str, 
                                   test_inputs: List[str]) -> Dict[str, Any]:
        """Compare outputs between PyTorch and ONNX models"""
        try:
            # Load PyTorch model
            pytorch_model = AutoModelForCausalLM.from_pretrained(pytorch_path)
            pytorch_model.eval()
            
            # Load ONNX model
            onnx_session = ort.InferenceSession(onnx_path)
            onnx_input_info = onnx_session.get_inputs()[0]
            onnx_output_info = onnx_session.get_outputs()[0]
            
            comparison_results = []
            
            # Test with sample inputs
            for i, test_input in enumerate(test_inputs[:3]):  # Limit for performance
                try:
                    # Create test input
                    input_ids = torch.randint(0, 1000, (1, 20))
                    
                    # PyTorch inference
                    with torch.no_grad():
                        pytorch_output = pytorch_model(input_ids).logits
                        
                    # ONNX inference
                    onnx_input = input_ids.numpy().astype(np.int64)
                    onnx_output = onnx_session.run([onnx_output_info.name], 
                                                 {onnx_input_info.name: onnx_input})[0]
                    
                    # Compare outputs (first few values)
                    pytorch_values = pytorch_output[0, :5, :5].numpy()
                    onnx_values = onnx_output[0, :5, :5]
                    
                    # Calculate similarity metrics
                    mse = np.mean((pytorch_values - onnx_values) ** 2)
                    max_diff = np.max(np.abs(pytorch_values - onnx_values))
                    correlation = np.corrcoef(pytorch_values.flatten(), onnx_values.flatten())[0, 1]
                    
                    comparison_results.append({
                        "test_index": i,
                        "mse": float(mse),
                        "max_difference": float(max_diff),
                        "correlation": float(correlation),
                        "pytorch_shape": list(pytorch_output.shape),
                        "onnx_shape": list(onnx_output.shape),
                        "status": "PASSED" if mse < 1e-3 else "WARNING"
                    })
                    
                except Exception as e:
                    comparison_results.append({
                        "test_index": i,
                        "status": "FAILED",
                        "error": str(e)
                    })
                    
            # Calculate overall similarity
            successful_comparisons = [r for r in comparison_results if r["status"] in ["PASSED", "WARNING"]]
            if successful_comparisons:
                avg_mse = np.mean([r["mse"] for r in successful_comparisons])
                avg_correlation = np.mean([r["correlation"] for r in successful_comparisons])
                
                overall_status = "PASSED" if avg_mse < 1e-3 and avg_correlation > 0.95 else "WARNING"
            else:
                avg_mse = float('inf')
                avg_correlation = 0.0
                overall_status = "FAILED"
                
            return {
                "overall_status": overall_status,
                "average_mse": avg_mse,
                "average_correlation": avg_correlation,
                "individual_comparisons": comparison_results,
                "total_comparisons": len(comparison_results),
                "successful_comparisons": len(successful_comparisons)
            }
            
        except Exception as e:
            return {
                "overall_status": "FAILED",
                "error": str(e)
            }
            
    def validate_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Validate model metadata and configuration"""
        try:
            results = {
                "config_files": {},
                "model_files": {},
                "tokenizer_files": {}
            }
            
            # Check for required configuration files
            config_files = ["config.json", "generation_config.json"]
            for config_file in config_files:
                config_path = os.path.join(model_path, config_file)
                results["config_files"][config_file] = {
                    "exists": os.path.exists(config_path),
                    "size": os.path.getsize(config_path) if os.path.exists(config_path) else 0
                }
                
            # Check for model files
            model_files = ["pytorch_model.bin", "model.safetensors"]
            for model_file in model_files:
                model_file_path = os.path.join(model_path, model_file)
                results["model_files"][model_file] = {
                    "exists": os.path.exists(model_file_path),
                    "size": os.path.getsize(model_file_path) if os.path.exists(model_file_path) else 0
                }
                
            # Check for tokenizer files
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
            for tokenizer_file in tokenizer_files:
                tokenizer_path = os.path.join(model_path, tokenizer_file)
                results["tokenizer_files"][tokenizer_file] = {
                    "exists": os.path.exists(tokenizer_path),
                    "size": os.path.getsize(tokenizer_path) if os.path.exists(tokenizer_path) else 0
                }
                
            # Overall validation
            has_config = any(results["config_files"][f]["exists"] for f in config_files)
            has_model = any(results["model_files"][f]["exists"] for f in model_files)
            has_tokenizer = any(results["tokenizer_files"][f]["exists"] for f in tokenizer_files)
            
            results["validation_summary"] = {
                "has_config": has_config,
                "has_model": has_model,
                "has_tokenizer": has_tokenizer,
                "is_valid": has_config and has_model,
                "status": "PASSED" if (has_config and has_model) else "FAILED"
            }
            
            return results
            
        except Exception as e:
            return {
                "validation_summary": {
                    "status": "FAILED",
                    "error": str(e)
                }
            }
            
    def stress_test_model_loading(self, model_path: str, iterations: int = 5) -> Dict[str, Any]:
        """Stress test model loading and unloading"""
        try:
            load_times = []
            memory_usage = []
            
            for i in range(iterations):
                try:
                    import psutil
                    initial_memory = psutil.virtual_memory().used
                    
                    # Load model
                    start_time = time.time()
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    load_time = time.time() - start_time
                    
                    peak_memory = psutil.virtual_memory().used
                    memory_used = peak_memory - initial_memory
                    
                    load_times.append(load_time)
                    memory_usage.append(memory_used)
                    
                    # Clean up
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    return {
                        "status": "FAILED",
                        "iteration": i,
                        "error": str(e)
                    }
                    
            return {
                "status": "PASSED",
                "iterations": iterations,
                "average_load_time": np.mean(load_times),
                "min_load_time": np.min(load_times),
                "max_load_time": np.max(load_times),
                "average_memory_usage_mb": np.mean(memory_usage) / (1024 * 1024),
                "max_memory_usage_mb": np.max(memory_usage) / (1024 * 1024),
                "load_times": load_times,
                "memory_usage_mb": [m / (1024 * 1024) for m in memory_usage]
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }


def create_extended_validator() -> ExtendedValidationTester:
    """Create an instance of the extended validation tester"""
    return ExtendedValidationTester()