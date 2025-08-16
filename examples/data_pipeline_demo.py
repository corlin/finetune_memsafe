#!/usr/bin/env python3
"""
数据管道演示脚本

展示DataPipeline类的完整功能，包括数据加载、格式化、分词和统计。
"""

import logging
from pathlib import Path
from transformers import AutoTokenizer

from src.data_pipeline import DataPipeline


def main():
    """主演示函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Qwen3 数据管道演示 ===\n")
    
    # 1. 初始化数据管道
    print("1. 初始化数据管道...")
    pipeline = DataPipeline(data_dir="data/raw", max_sequence_length=256)
    print(f"   数据目录: {pipeline.data_dir}")
    print(f"   最大序列长度: {pipeline.max_sequence_length}")
    
    # 2. 加载QA数据
    print("\n2. 加载QA数据...")
    qa_data = pipeline.load_qa_data_from_files()
    print(f"   成功加载 {len(qa_data)} 个QA对")
    
    if qa_data:
        print(f"   示例QA对:")
        print(f"     问题: {qa_data[0].question}")
        print(f"     答案: {qa_data[0].answer[:100]}...")
        print(f"     来源: {qa_data[0].source}")
    
    # 3. 格式化为Qwen3对话格式
    print("\n3. 格式化为Qwen3对话格式...")
    dataset = pipeline.format_for_qwen(qa_data)
    print(f"   格式化了 {len(dataset)} 个对话")
    
    if len(dataset) > 0:
        print(f"   示例对话格式:")
        print(f"     {dataset[0]['text'][:200]}...")
    
    # 4. 分词处理
    print("\n4. 加载分词器并进行分词...")
    try:
        # 使用较小的Qwen模型进行演示
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        print("   分词器加载成功")
        
        # 分词数据集
        tokenized_dataset = pipeline.tokenize_dataset(dataset, tokenizer)
        print(f"   分词完成，保留 {len(tokenized_dataset)} 个样本")
        
        # 5. 创建数据整理器
        print("\n5. 创建数据整理器...")
        data_collator = pipeline.create_data_collator(tokenizer)
        print("   数据整理器创建成功")
        
        # 6. 获取数据集统计信息
        print("\n6. 数据集统计信息:")
        stats = pipeline.get_dataset_stats(tokenized_dataset)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            elif isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # 7. 演示数据保存和加载
        print("\n7. 演示数据保存和加载...")
        output_path = "data/processed_dataset"
        
        try:
            pipeline.save_processed_data(tokenized_dataset, output_path)
            print(f"   数据集已保存到 {output_path}")
            
            # 重新加载验证
            loaded_dataset = pipeline.load_processed_data(output_path)
            print(f"   重新加载验证: {len(loaded_dataset)} 个样本")
            
        except Exception as e:
            print(f"   保存/加载演示跳过: {e}")
        
    except Exception as e:
        print(f"   分词器加载失败: {e}")
        print("   跳过分词演示")
    
    print("\n=== 演示完成 ===")
    print("\nDataPipeline 主要功能:")
    print("✓ 从多种markdown格式加载QA数据")
    print("✓ 数据验证和错误处理")
    print("✓ 回退到示例数据")
    print("✓ 格式化为Qwen3对话格式")
    print("✓ 内存优化的分词处理")
    print("✓ 序列长度限制和过滤")
    print("✓ 数据统计和分析")
    print("✓ 数据保存和加载")


if __name__ == "__main__":
    main()