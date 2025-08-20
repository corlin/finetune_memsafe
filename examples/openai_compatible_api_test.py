#!/usr/bin/env python3
"""
OpenAI兼容API测试示例

演示如何使用industry_evaluation系统测试不同的OpenAI兼容API提供商，
包括BigModel GLM、智谱AI、DeepSeek等。

使用方法:
python examples/openai_compatible_api_test.py --provider bigmodel --api-key YOUR_API_KEY --model glm-4.5
"""

import sys
import os
import argparse
import json
import time
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from industry_evaluation.adapters.model_adapter import ModelAdapterFactory, ModelManager
from industry_evaluation.core.interfaces import EvaluationConfig
from industry_evaluation.models.data_models import EvaluationDimension


def create_test_dataset() -> List[Dict[str, Any]]:
    """创建测试数据集"""
    return [
        {
            "id": "test_1",
            "input": "什么是人工智能？请简要介绍。",
            "expected_output": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "context": {
                "domain": "technology",
                "difficulty": "basic"
            }
        },
        {
            "id": "test_2", 
            "input": "中国大模型行业在2025年将面临哪些机遇和挑战？",
            "expected_output": "2025年中国大模型行业将面临技术创新、市场竞争、监管政策等多方面的机遇和挑战。",
            "context": {
                "domain": "industry_analysis",
                "difficulty": "advanced"
            }
        },
        {
            "id": "test_3",
            "input": "请解释什么是机器学习中的过拟合现象。",
            "expected_output": "过拟合是指模型在训练数据上表现很好，但在新数据上泛化能力差的现象。",
            "context": {
                "domain": "machine_learning",
                "difficulty": "intermediate"
            }
        }
    ]


def test_model_adapter(provider: str, api_key: str, model_name: str, 
                      base_url: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    测试模型适配器
    
    Args:
        provider: 提供商名称
        api_key: API密钥
        model_name: 模型名称
        base_url: 自定义API基础URL
        verbose: 是否显示详细信息
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    print(f"\n=== 测试 {provider} 提供商的 {model_name} 模型 ===")
    
    try:
        # 创建模型管理器
        model_manager = ModelManager()
        
        # 准备配置
        config_kwargs = {}
        if base_url:
            config_kwargs["base_url"] = base_url
        
        # 创建适配器
        if provider in ModelAdapterFactory.get_supported_providers():
            # 使用预定义的提供商配置
            adapter = ModelAdapterFactory.create_openai_compatible_adapter(
                model_id=f"{provider}_{model_name}",
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                **config_kwargs
            )
        else:
            # 使用通用OpenAI兼容配置
            config = {
                "api_key": api_key,
                "model_name": model_name,
                "base_url": base_url or "https://api.openai.com/v1",
                "provider": provider,
                **config_kwargs
            }
            adapter = ModelAdapterFactory.create_adapter(
                "openai_compatible", 
                f"{provider}_{model_name}", 
                config
            )
        
        # 注册模型
        model_manager.register_model(
            f"{provider}_{model_name}",
            "openai_compatible",
            adapter.config
        )
        
        # 测试结果
        results = {
            "provider": provider,
            "model_name": model_name,
            "model_id": f"{provider}_{model_name}",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "average_response_time": 0.0
            }
        }
        
        # 创建测试数据
        test_dataset = create_test_dataset()
        total_response_time = 0.0
        
        print(f"开始测试 {len(test_dataset)} 个样本...")
        
        for i, sample in enumerate(test_dataset, 1):
            print(f"\n--- 测试样本 {i}/{len(test_dataset)} ---")
            print(f"输入: {sample['input'][:100]}...")
            
            test_result = {
                "sample_id": sample["id"],
                "input": sample["input"],
                "success": False,
                "response": "",
                "response_time": 0.0,
                "error": None
            }
            
            try:
                start_time = time.time()
                
                # 调用模型
                response = adapter.predict(
                    sample["input"],
                    {
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                test_result.update({
                    "success": True,
                    "response": response,
                    "response_time": response_time
                })
                
                total_response_time += response_time
                results["summary"]["successful_tests"] += 1
                
                print(f"✅ 成功 (耗时: {response_time:.2f}s)")
                if verbose:
                    print(f"响应: {response[:200]}...")
                
            except Exception as e:
                test_result.update({
                    "success": False,
                    "error": str(e)
                })
                
                results["summary"]["failed_tests"] += 1
                print(f"❌ 失败: {str(e)}")
            
            results["tests"].append(test_result)
            results["summary"]["total_tests"] += 1
            
            # 避免请求过于频繁
            if i < len(test_dataset):
                time.sleep(1)
        
        # 计算平均响应时间
        if results["summary"]["successful_tests"] > 0:
            results["summary"]["average_response_time"] = total_response_time / results["summary"]["successful_tests"]
        
        # 打印总结
        print(f"\n=== 测试总结 ===")
        print(f"总测试数: {results['summary']['total_tests']}")
        print(f"成功: {results['summary']['successful_tests']}")
        print(f"失败: {results['summary']['failed_tests']}")
        print(f"成功率: {results['summary']['successful_tests']/results['summary']['total_tests']*100:.1f}%")
        if results["summary"]["average_response_time"] > 0:
            print(f"平均响应时间: {results['summary']['average_response_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return {
            "provider": provider,
            "model_name": model_name,
            "error": str(e),
            "success": False
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OpenAI兼容API测试工具")
    parser.add_argument("--provider", required=True, 
                       help="API提供商 (bigmodel, zhipu, deepseek, moonshot, openai, 或自定义)")
    parser.add_argument("--api-key", required=True, help="API密钥")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--base-url", help="自定义API基础URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细输出")
    parser.add_argument("--output", "-o", help="保存结果到JSON文件")
    
    args = parser.parse_args()
    
    print("OpenAI兼容API测试工具")
    print("=" * 50)
    
    # 显示支持的提供商
    supported_providers = ModelAdapterFactory.get_supported_providers()
    print(f"支持的预配置提供商: {', '.join(supported_providers)}")
    
    if args.provider in supported_providers:
        print(f"✅ 使用预配置的 {args.provider} 提供商设置")
    else:
        print(f"⚠️  使用自定义提供商 {args.provider}，需要提供 --base-url")
        if not args.base_url:
            print("❌ 自定义提供商需要指定 --base-url 参数")
            return 1
    
    # 执行测试
    results = test_model_adapter(
        provider=args.provider,
        api_key=args.api_key,
        model_name=args.model,
        base_url=args.base_url,
        verbose=args.verbose
    )
    
    # 保存结果
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n📁 结果已保存到: {args.output}")
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")
    
    # 返回退出码
    if results.get("success", True) and results.get("summary", {}).get("failed_tests", 0) == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())