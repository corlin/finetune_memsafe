"""
性能测试

测试评估系统在不同负载下的性能表现。
"""

import unittest
import time
import statistics
from datasets import Dataset

from src.evaluation.data_preprocessor import DataPreprocessor
from src.evaluation.data_models import EvaluationConfig


class TestPerformance(unittest.TestCase):
    """性能测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = EvaluationConfig(
            batch_size=32,
            data_processing={
                "validation": {
                    "min_valid_samples_ratio": 0.1,
                    "enable_data_cleaning": True
                },
                "diagnostics": {
                    "enable_detailed_logging": False,  # 禁用详细日志以提高性能
                    "log_batch_statistics": False
                }
            }
        )
        self.preprocessor = DataPreprocessor(self.config)
    
    def test_small_batch_performance(self):
        """测试小批次性能"""
        dataset = Dataset.from_dict({
            "text": [f"Small batch test {i}" for i in range(100)],
            "target": [f"Response {i}" for i in range(100)]
        })
        
        times = []
        batch_size = 10
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            start_time = time.time()
            result = self.preprocessor.preprocess_batch(batch, "text_generation")
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
        
        # 计算性能统计
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"小批次性能 (批次大小={batch_size}):")
        print(f"  平均时间: {avg_time*1000:.2f}ms")
        print(f"  最大时间: {max_time*1000:.2f}ms")
        print(f"  最小时间: {min_time*1000:.2f}ms")
        print(f"  处理速度: {batch_size/avg_time:.1f} samples/s")
        
        # 性能断言
        self.assertLess(avg_time, 0.1)  # 平均处理时间不超过100ms
        self.assertLess(max_time, 0.5)  # 最大处理时间不超过500ms
    
    def test_medium_batch_performance(self):
        """测试中等批次性能"""
        dataset = Dataset.from_dict({
            "text": [f"Medium batch test {i}" for i in range(1000)],
            "target": [f"Response {i}" for i in range(1000)]
        })
        
        times = []
        batch_size = 50
        
        for i in range(0, min(500, len(dataset)), batch_size):
            batch = dataset[i:i+batch_size]
            
            start_time = time.time()
            result = self.preprocessor.preprocess_batch(batch, "text_generation")
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
        
        # 计算性能统计
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"中等批次性能 (批次大小={batch_size}):")
        print(f"  平均时间: {avg_time*1000:.2f}ms")
        print(f"  最大时间: {max_time*1000:.2f}ms")
        print(f"  最小时间: {min_time*1000:.2f}ms")
        print(f"  处理速度: {batch_size/avg_time:.1f} samples/s")
        
        # 性能断言
        self.assertLess(avg_time, 0.2)  # 平均处理时间不超过200ms
        self.assertLess(max_time, 1.0)  # 最大处理时间不超过1s
    
    def test_large_batch_performance(self):
        """测试大批次性能"""
        dataset = Dataset.from_dict({
            "text": [f"Large batch test {i}" * 5 for i in range(2000)],  # 更长的文本
            "target": [f"Response {i}" for i in range(2000)]
        })
        
        times = []
        batch_size = 100
        
        for i in range(0, min(1000, len(dataset)), batch_size):
            batch = dataset[i:i+batch_size]
            
            start_time = time.time()
            result = self.preprocessor.preprocess_batch(batch, "text_generation")
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
        
        # 计算性能统计
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"大批次性能 (批次大小={batch_size}):")
        print(f"  平均时间: {avg_time*1000:.2f}ms")
        print(f"  最大时间: {max_time*1000:.2f}ms")
        print(f"  最小时间: {min_time*1000:.2f}ms")
        print(f"  处理速度: {batch_size/avg_time:.1f} samples/s")
        
        # 性能断言
        self.assertLess(avg_time, 0.5)  # 平均处理时间不超过500ms
        self.assertLess(max_time, 2.0)  # 最大处理时间不超过2s
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理大量数据
        dataset = Dataset.from_dict({
            "text": [f"Memory test {i}" * 20 for i in range(5000)],
            "target": [f"Response {i}" for i in range(5000)]
        })
        
        batch_size = 100
        memory_measurements = []
        
        for i in range(0, min(2000, len(dataset)), batch_size):
            batch = dataset[i:i+batch_size]
            result = self.preprocessor.preprocess_batch(batch, "text_generation")
            
            # 测量内存使用
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = max(memory_measurements)
        memory_increase = final_memory - initial_memory
        peak_increase = max_memory - initial_memory
        
        print(f"内存效率测试:")
        print(f"  初始内存: {initial_memory:.1f}MB")
        print(f"  最终内存: {final_memory:.1f}MB")
        print(f"  峰值内存: {max_memory:.1f}MB")
        print(f"  内存增长: {memory_increase:.1f}MB")
        print(f"  峰值增长: {peak_increase:.1f}MB")
        
        # 内存效率断言
        self.assertLess(memory_increase, 200)  # 内存增长不超过200MB
        self.assertLess(peak_increase, 300)    # 峰值增长不超过300MB
    
    def test_scalability_with_different_batch_sizes(self):
        """测试不同批次大小的可扩展性"""
        dataset = Dataset.from_dict({
            "text": [f"Scalability test {i}" for i in range(1000)],
            "target": [f"Response {i}" for i in range(1000)]
        })
        
        batch_sizes = [10, 25, 50, 100, 200]
        results = []
        
        for batch_size in batch_sizes:
            times = []
            
            # 处理固定数量的样本
            samples_to_process = 500
            
            for i in range(0, samples_to_process, batch_size):
                batch = dataset[i:i+batch_size]
                
                start_time = time.time()
                result = self.preprocessor.preprocess_batch(batch, "text_generation")
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time
            
            results.append({
                "batch_size": batch_size,
                "avg_time_ms": avg_time * 1000,
                "throughput": throughput
            })
            
            print(f"批次大小 {batch_size:3d}: {avg_time*1000:6.2f}ms, {throughput:6.1f} samples/s")
        
        # 验证可扩展性
        # 较大的批次应该有更好的吞吐量
        throughputs = [r["throughput"] for r in results]
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)
        
        # 最大吞吐量应该至少是最小吞吐量的1.5倍
        self.assertGreater(max_throughput / min_throughput, 1.5)
    
    def test_data_quality_impact_on_performance(self):
        """测试数据质量对性能的影响"""
        # 高质量数据
        high_quality_dataset = Dataset.from_dict({
            "text": [f"High quality sample {i}" for i in range(500)],
            "target": [f"Quality response {i}" for i in range(500)]
        })
        
        # 低质量数据（包含很多空值和无效数据）
        low_quality_dataset = Dataset.from_dict({
            "text": [
                f"Low quality sample {i}" if i % 3 == 0 else 
                "" if i % 3 == 1 else 
                None
                for i in range(500)
            ],
            "target": [f"Response {i}" for i in range(500)]
        })
        
        batch_size = 50
        
        # 测试高质量数据性能
        high_quality_times = []
        for i in range(0, len(high_quality_dataset), batch_size):
            batch = high_quality_dataset[i:i+batch_size]
            
            start_time = time.time()
            result = self.preprocessor.preprocess_batch(batch, "text_generation")
            end_time = time.time()
            
            high_quality_times.append(end_time - start_time)
        
        # 测试低质量数据性能
        low_quality_times = []
        for i in range(0, len(low_quality_dataset), batch_size):
            batch = low_quality_dataset[i:i+batch_size]
            
            start_time = time.time()
            result = self.preprocessor.preprocess_batch(batch, "text_generation")
            end_time = time.time()
            
            low_quality_times.append(end_time - start_time)
        
        high_quality_avg = statistics.mean(high_quality_times)
        low_quality_avg = statistics.mean(low_quality_times)
        
        print(f"数据质量对性能的影响:")
        print(f"  高质量数据平均时间: {high_quality_avg*1000:.2f}ms")
        print(f"  低质量数据平均时间: {low_quality_avg*1000:.2f}ms")
        print(f"  性能差异: {(low_quality_avg/high_quality_avg-1)*100:.1f}%")
        
        # 低质量数据的处理时间不应该显著增加
        self.assertLess(low_quality_avg / high_quality_avg, 2.0)  # 不超过2倍
    
    def test_concurrent_processing_performance(self):
        """测试并发处理性能"""
        import threading
        import queue
        
        dataset = Dataset.from_dict({
            "text": [f"Concurrent test {i}" for i in range(1000)],
            "target": [f"Response {i}" for i in range(1000)]
        })
        
        def process_batches(start_idx, end_idx, results_queue):
            times = []
            preprocessor = DataPreprocessor(self.config)
            
            for i in range(start_idx, end_idx, 50):
                batch = dataset[i:i+50]
                
                start_time = time.time()
                result = preprocessor.preprocess_batch(batch, "text_generation")
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            results_queue.put(times)
        
        # 测试串行处理
        start_time = time.time()
        serial_times = []
        for i in range(0, 500, 50):
            batch = dataset[i:i+50]
            
            batch_start = time.time()
            result = self.preprocessor.preprocess_batch(batch, "text_generation")
            batch_end = time.time()
            
            serial_times.append(batch_end - batch_start)
        serial_total_time = time.time() - start_time
        
        # 测试并行处理
        start_time = time.time()
        results_queue = queue.Queue()
        threads = []
        
        num_threads = 4
        samples_per_thread = 500 // num_threads
        
        for i in range(num_threads):
            start_idx = i * samples_per_thread
            end_idx = start_idx + samples_per_thread
            
            thread = threading.Thread(
                target=process_batches,
                args=(start_idx, end_idx, results_queue)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        parallel_total_time = time.time() - start_time
        
        # 收集并行处理结果
        parallel_times = []
        while not results_queue.empty():
            thread_times = results_queue.get()
            parallel_times.extend(thread_times)
        
        print(f"并发处理性能:")
        print(f"  串行总时间: {serial_total_time:.2f}s")
        print(f"  并行总时间: {parallel_total_time:.2f}s")
        print(f"  加速比: {serial_total_time/parallel_total_time:.2f}x")
        print(f"  串行平均批次时间: {statistics.mean(serial_times)*1000:.2f}ms")
        print(f"  并行平均批次时间: {statistics.mean(parallel_times)*1000:.2f}ms")
        
        # 并行处理应该更快
        self.assertLess(parallel_total_time, serial_total_time)


if __name__ == '__main__':
    unittest.main(verbosity=2)