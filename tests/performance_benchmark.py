#!/usr/bin/env python3
"""
Performance Benchmark Suite for Final Integration Testing

This module provides comprehensive performance benchmarking capabilities
for the model export optimization system.
"""

import os
import sys
import time
import json
import psutil
import platform
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._collect_system_info()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmarking context"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_count": multiprocessing.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
    def benchmark_memory_usage(self, test_name: str = "memory_benchmark") -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        print(f"Running memory usage benchmark: {test_name}")
        
        try:
            initial_memory = psutil.virtual_memory()
            process = psutil.Process()
            initial_process_memory = process.memory_info()
            
            memory_snapshots = []
            
            # Baseline measurement
            memory_snapshots.append({
                "stage": "baseline",
                "system_memory_used_gb": (initial_memory.total - initial_memory.available) / (1024**3),
                "process_memory_mb": initial_process_memory.rss / (1024**2),
                "timestamp": time.time()
            })
            
            # Memory allocation test
            large_tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000)
                large_tensors.append(tensor)
                
                current_memory = psutil.virtual_memory()
                current_process_memory = process.memory_info()
                
                memory_snapshots.append({
                    "stage": f"allocation_{i}",
                    "system_memory_used_gb": (current_memory.total - current_memory.available) / (1024**3),
                    "process_memory_mb": current_process_memory.rss / (1024**2),
                    "timestamp": time.time()
                })
                
            # Peak memory measurement
            peak_memory = psutil.virtual_memory()
            peak_process_memory = process.memory_info()
            
            memory_snapshots.append({
                "stage": "peak",
                "system_memory_used_gb": (peak_memory.total - peak_memory.available) / (1024**3),
                "process_memory_mb": peak_process_memory.rss / (1024**2),
                "timestamp": time.time()
            })
            
            # Cleanup and final measurement
            del large_tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            final_memory = psutil.virtual_memory()
            final_process_memory = process.memory_info()
            
            memory_snapshots.append({
                "stage": "cleanup",
                "system_memory_used_gb": (final_memory.total - final_memory.available) / (1024**3),
                "process_memory_mb": final_process_memory.rss / (1024**2),
                "timestamp": time.time()
            })
            
            # Calculate metrics
            peak_usage = max(s["process_memory_mb"] for s in memory_snapshots)
            baseline_usage = memory_snapshots[0]["process_memory_mb"]
            memory_increase = peak_usage - baseline_usage
            
            return {
                "status": "PASSED",
                "baseline_memory_mb": baseline_usage,
                "peak_memory_mb": peak_usage,
                "memory_increase_mb": memory_increase,
                "cleanup_effective": memory_snapshots[-1]["process_memory_mb"] < peak_usage,
                "memory_snapshots": memory_snapshots,
                "system_info": self.system_info
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def benchmark_cpu_performance(self, test_name: str = "cpu_benchmark") -> Dict[str, Any]:
        """Benchmark CPU performance characteristics"""
        print(f"Running CPU performance benchmark: {test_name}")
        
        try:
            # CPU-intensive computation benchmark
            def cpu_intensive_task(size: int) -> float:
                start_time = time.time()
                # Matrix multiplication
                a = np.random.randn(size, size)
                b = np.random.randn(size, size)
                c = np.dot(a, b)
                return time.time() - start_time
                
            # Test different matrix sizes
            sizes = [100, 200, 500, 1000]
            cpu_results = {}
            
            for size in sizes:
                times = []
                for _ in range(3):  # Multiple runs for average
                    exec_time = cpu_intensive_task(size)
                    times.append(exec_time)
                    
                cpu_results[f"matrix_{size}x{size}"] = {
                    "average_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times)
                }
                
            # Multi-threading performance test
            def threaded_task():
                return cpu_intensive_task(200)
                
            # Single-threaded baseline
            single_start = time.time()
            for _ in range(4):
                threaded_task()
            single_time = time.time() - single_start
            
            # Multi-threaded test
            multi_start = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(threaded_task) for _ in range(4)]
                for future in futures:
                    future.result()
            multi_time = time.time() - multi_start
            
            threading_efficiency = single_time / multi_time if multi_time > 0 else 0
            
            return {
                "status": "PASSED",
                "matrix_computations": cpu_results,
                "threading_performance": {
                    "single_threaded_time": single_time,
                    "multi_threaded_time": multi_time,
                    "efficiency_ratio": threading_efficiency
                },
                "cpu_info": {
                    "cpu_count": multiprocessing.cpu_count(),
                    "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def benchmark_io_performance(self, test_name: str = "io_benchmark", 
                                temp_dir: str = None) -> Dict[str, Any]:
        """Benchmark I/O performance characteristics"""
        print(f"Running I/O performance benchmark: {test_name}")
        
        if temp_dir is None:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="io_benchmark_")
            
        try:
            io_results = {}
            
            # File write performance
            test_data_sizes = [1, 10, 100]  # MB
            
            for size_mb in test_data_sizes:
                test_data = b"x" * (size_mb * 1024 * 1024)
                test_file = os.path.join(temp_dir, f"test_{size_mb}mb.dat")
                
                # Write test
                start_time = time.time()
                with open(test_file, 'wb') as f:
                    f.write(test_data)
                write_time = time.time() - start_time
                
                # Read test
                start_time = time.time()
                with open(test_file, 'rb') as f:
                    read_data = f.read()
                read_time = time.time() - start_time
                
                # Verify data integrity
                data_integrity = len(read_data) == len(test_data)
                
                io_results[f"{size_mb}mb_file"] = {
                    "write_time": write_time,
                    "read_time": read_time,
                    "write_speed_mbps": size_mb / write_time if write_time > 0 else 0,
                    "read_speed_mbps": size_mb / read_time if read_time > 0 else 0,
                    "data_integrity": data_integrity
                }
                
                # Cleanup
                os.remove(test_file)
                
            # Directory operations
            start_time = time.time()
            test_dirs = []
            for i in range(100):
                dir_path = os.path.join(temp_dir, f"test_dir_{i}")
                os.makedirs(dir_path)
                test_dirs.append(dir_path)
            dir_create_time = time.time() - start_time
            
            start_time = time.time()
            for dir_path in test_dirs:
                os.rmdir(dir_path)
            dir_delete_time = time.time() - start_time
            
            return {
                "status": "PASSED",
                "file_operations": io_results,
                "directory_operations": {
                    "create_100_dirs_time": dir_create_time,
                    "delete_100_dirs_time": dir_delete_time,
                    "create_rate_per_sec": 100 / dir_create_time if dir_create_time > 0 else 0,
                    "delete_rate_per_sec": 100 / dir_delete_time if dir_delete_time > 0 else 0
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
    def benchmark_torch_operations(self, test_name: str = "torch_benchmark") -> Dict[str, Any]:
        """Benchmark PyTorch operations performance"""
        print(f"Running PyTorch operations benchmark: {test_name}")
        
        try:
            torch_results = {}
            
            # Tensor operations benchmark
            sizes = [1000, 2000, 5000]
            
            for size in sizes:
                # Tensor creation
                start_time = time.time()
                tensor_a = torch.randn(size, size)
                tensor_b = torch.randn(size, size)
                creation_time = time.time() - start_time
                
                # Matrix multiplication
                start_time = time.time()
                result = torch.mm(tensor_a, tensor_b)
                matmul_time = time.time() - start_time
                
                # Element-wise operations
                start_time = time.time()
                result = tensor_a + tensor_b
                result = torch.relu(result)
                result = torch.sigmoid(result)
                elementwise_time = time.time() - start_time
                
                torch_results[f"tensor_{size}x{size}"] = {
                    "creation_time": creation_time,
                    "matmul_time": matmul_time,
                    "elementwise_time": elementwise_time,
                    "total_elements": size * size
                }
                
            # CUDA performance (if available)
            cuda_results = {}
            if torch.cuda.is_available():
                device = torch.device("cuda")
                
                for size in [1000, 2000]:
                    # CPU to GPU transfer
                    cpu_tensor = torch.randn(size, size)
                    start_time = time.time()
                    gpu_tensor = cpu_tensor.to(device)
                    transfer_to_gpu_time = time.time() - start_time
                    
                    # GPU computation
                    gpu_tensor_b = torch.randn(size, size, device=device)
                    start_time = time.time()
                    gpu_result = torch.mm(gpu_tensor, gpu_tensor_b)
                    gpu_compute_time = time.time() - start_time
                    
                    # GPU to CPU transfer
                    start_time = time.time()
                    cpu_result = gpu_result.cpu()
                    transfer_to_cpu_time = time.time() - start_time
                    
                    cuda_results[f"cuda_{size}x{size}"] = {
                        "transfer_to_gpu_time": transfer_to_gpu_time,
                        "gpu_compute_time": gpu_compute_time,
                        "transfer_to_cpu_time": transfer_to_cpu_time,
                        "total_gpu_time": transfer_to_gpu_time + gpu_compute_time + transfer_to_cpu_time
                    }
                    
            return {
                "status": "PASSED",
                "cpu_operations": torch_results,
                "cuda_operations": cuda_results,
                "cuda_available": torch.cuda.is_available(),
                "torch_version": torch.__version__
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def benchmark_concurrent_operations(self, test_name: str = "concurrent_benchmark") -> Dict[str, Any]:
        """Benchmark concurrent operations performance"""
        print(f"Running concurrent operations benchmark: {test_name}")
        
        try:
            def compute_task(task_id: int, size: int = 500) -> Dict[str, Any]:
                start_time = time.time()
                tensor = torch.randn(size, size)
                result = torch.sum(tensor)
                compute_time = time.time() - start_time
                
                return {
                    "task_id": task_id,
                    "compute_time": compute_time,
                    "result": result.item()
                }
                
            # Sequential execution baseline
            sequential_start = time.time()
            sequential_results = []
            for i in range(8):
                result = compute_task(i)
                sequential_results.append(result)
            sequential_time = time.time() - sequential_start
            
            # Thread-based concurrent execution
            thread_start = time.time()
            thread_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(compute_task, i) for i in range(8)]
                for future in futures:
                    thread_results.append(future.result())
            thread_time = time.time() - thread_start
            
            # Process-based concurrent execution
            process_start = time.time()
            process_results = []
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(compute_task, i) for i in range(8)]
                for future in futures:
                    process_results.append(future.result())
            process_time = time.time() - process_start
            
            return {
                "status": "PASSED",
                "sequential_execution": {
                    "total_time": sequential_time,
                    "average_task_time": sequential_time / 8,
                    "results_count": len(sequential_results)
                },
                "thread_execution": {
                    "total_time": thread_time,
                    "speedup_ratio": sequential_time / thread_time if thread_time > 0 else 0,
                    "results_count": len(thread_results)
                },
                "process_execution": {
                    "total_time": process_time,
                    "speedup_ratio": sequential_time / process_time if process_time > 0 else 0,
                    "results_count": len(process_results)
                },
                "system_info": {
                    "cpu_count": multiprocessing.cpu_count(),
                    "thread_count": threading.active_count()
                }
            }
            
        except Exception as e:
            return {
                "status": "FAILED",
                "error": str(e)
            }
            
    def run_comprehensive_benchmark(self, temp_dir: str = None) -> Dict[str, Any]:
        """Run all benchmark tests and generate comprehensive report"""
        print("Starting Comprehensive Performance Benchmark Suite")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Run all benchmarks
        benchmarks = {
            "memory_usage": self.benchmark_memory_usage,
            "cpu_performance": self.benchmark_cpu_performance,
            "io_performance": lambda: self.benchmark_io_performance(temp_dir=temp_dir),
            "torch_operations": self.benchmark_torch_operations,
            "concurrent_operations": self.benchmark_concurrent_operations
        }
        
        results = {}
        
        for benchmark_name, benchmark_func in benchmarks.items():
            try:
                print(f"\nRunning {benchmark_name}...")
                result = benchmark_func()
                results[benchmark_name] = result
                status = result.get("status", "UNKNOWN")
                print(f"{benchmark_name}: {status}")
                
            except Exception as e:
                print(f"{benchmark_name}: FAILED - {e}")
                results[benchmark_name] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        passed_benchmarks = sum(1 for r in results.values() if r.get("status") == "PASSED")
        total_benchmarks = len(results)
        
        summary = {
            "execution_summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_benchmarks": total_benchmarks,
                "passed_benchmarks": passed_benchmarks,
                "failed_benchmarks": total_benchmarks - passed_benchmarks,
                "success_rate": (passed_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 0
            },
            "system_information": self.system_info,
            "benchmark_results": results
        }
        
        print(f"\nBenchmark Suite Completed")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Success Rate: {summary['execution_summary']['success_rate']:.1f}%")
        
        return summary
        
    def save_benchmark_report(self, results: Dict[str, Any], output_path: str = "performance_benchmark_report.json"):
        """Save benchmark results to file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Benchmark report saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Failed to save benchmark report: {e}")
            return None


def run_performance_benchmark(temp_dir: str = None) -> Dict[str, Any]:
    """Run the complete performance benchmark suite"""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark(temp_dir)
    
    # Save report
    report_path = benchmark.save_benchmark_report(results)
    
    return results


if __name__ == "__main__":
    # Run performance benchmark
    results = run_performance_benchmark()
    
    success_rate = results["execution_summary"]["success_rate"]
    if success_rate >= 80:
        print(f"\nPerformance benchmark completed successfully! ({success_rate:.1f}% success rate)")
        sys.exit(0)
    else:
        print(f"\nPerformance benchmark completed with issues ({success_rate:.1f}% success rate)")
        sys.exit(1)