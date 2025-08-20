#!/usr/bin/env python3
"""
BigModel GLM API 演示示例

这个示例演示如何使用industry_evaluation系统测试BigModel的GLM系列模型，
包括GLM-4.5等模型的API调用和评估。

使用方法:
1. 设置环境变量: export BIGMODEL_API_KEY="your_api_key_here"
2. 运行示例: python examples/bigmodel_glm_demo.py

或者直接传递API密钥:
python examples/bigmodel_glm_demo.py --api-key your_api_key_here
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


def create_bigmodel_test_dataset() -> List[Dict[str, Any]]:
    """创建针对BigModel GLM的测试数据集"""
    return [
        {
            "id": "glm_test_1",
            "input": "什么是人工智能？请简要介绍其发展历程和主要应用领域。",
            "expected_output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "context": {
                "domain": "technology",
                "difficulty": "basic",
                "max_tokens": 500,
                "temperature": 0.7
            }
        },
        {
            "id": "glm_test_2",
            "input": "中国大模型行业在2025年将面临哪些机遇和挑战？请从技术、市场、政策等角度分析。",
            "expected_output": "2025年中国大模型行业将面临多重机遇和挑战。机遇方面包括技术创新加速、应用场景扩展、产业数字化需求增长等。",
            "context": {
                "domain": "industry_analysis",
                "difficulty": "advanced",
                "max_tokens": 800,
                "temperature": 0.6,
                "system_prompt": "你是一个专业的行业分析师，请提供深入、客观的分析。"
            }
        },
        {
            "id": "glm_test_3",
            "input": "请解释什么是Transformer架构，以及它在大语言模型中的重要作用。",
            "expected_output": "Transformer是一种基于注意力机制的神经网络架构，由Google在2017年提出。",
            "context": {
                "domain": "machine_learning",
                "difficulty": "intermediate",
                "max_tokens": 600,
                "temperature": 0.5
            }
        },
        {
            "id": "glm_test_4",
            "input": "如何评估大语言模型在金融领域的应用效果？需要考虑哪些关键指标？",
            "expected_output": "评估大语言模型在金融领域的应用效果需要考虑多个维度的指标。",
            "context": {
                "domain": "finance",
                "difficulty": "advanced",
                "max_tokens": 700,
                "temperature": 0.4,
                "system_prompt": "你是一个金融科技专家，请提供专业的评估建议。"
            }
        },
        {
            "id": "glm_test_5",
            "input": "请用简单易懂的语言解释什么是机器学习中的过拟合现象，并提供解决方案。",
            "expected_output": "过拟合是指机器学习模型在训练数据上表现很好，但在新数据上泛化能力差的现象。",
            "context": {
                "domain": "education",
                "difficulty": "intermediate",
                "max_tokens": 500,
                "temperature": 0.8,
                "system_prompt": "你是一个教育专家，请用通俗易懂的语言解释复杂概念。"
            }
        }
    ]


def test_bigmodel_glm_api(api_key: str, model_name: str = "glm-4.5", verbose: bool = False) -> Dict[str, Any]:
    """
    测试BigModel GLM API
    
    Args:
        api_key: BigModel API密钥
        model_name: 模型名称，默认为glm-4.5
        verbose: 是否显示详细信息
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    print(f"\n🚀 开始测试 BigModel {model_name} API")
    print("=" * 60)
    
    try:
        # 创建模型管理器
        model_manager = ModelManager()
        
        # 创建BigModel GLM适配器
        adapter = ModelAdapterFactory.create_openai_compatible_adapter(
            model_id=f"bigmodel_{model_name.replace('-', '_')}",
            provider="bigmodel",
            api_key=api_key,
            model_name=model_name,
            timeout=60,  # 增加超时时间
            custom_headers={
                "User-Agent": "Industry-Evaluation-Demo/1.0"
            }
        )
        
        # 注册模型
        model_manager.register_model(
            f"bigmodel_{model_name.replace('-', '_')}",
            "openai_compatible",
            adapter.config
        )
        
        print(f"✅ 成功创建 {model_name} 适配器")
        print(f"📡 API端点: {adapter.base_url}")
        print(f"🔑 认证方式: {adapter.auth_type}")
        
        # 健康检查
        print(f"\n🔍 执行健康检查...")
        health_status = adapter.get_health_status()
        print(f"健康状态: {'✅ 健康' if health_status['is_healthy'] else '❌ 不健康'}")
        
        # 快速连通性测试
        print(f"\n⚡ 执行快速连通性测试...")
        try:
            quick_test = adapter.predict("你好", {"max_tokens": 10})
            print(f"✅ 连通性测试成功: {quick_test[:50]}...")
        except Exception as e:
            print(f"❌ 连通性测试失败: {str(e)}")
            return {"success": False, "error": f"连通性测试失败: {str(e)}"}
        
        # 创建测试数据集
        test_dataset = create_bigmodel_test_dataset()
        
        # 测试结果统计
        results = {
            "model_name": model_name,
            "provider": "bigmodel",
            "api_endpoint": adapter.base_url,
            "tests": [],
            "summary": {
                "total_tests": len(test_dataset),
                "successful_tests": 0,
                "failed_tests": 0,
                "total_response_time": 0.0,
                "average_response_time": 0.0,
                "total_tokens_used": 0
            }
        }
        
        print(f"\n📋 开始执行 {len(test_dataset)} 个测试用例...")
        
        for i, sample in enumerate(test_dataset, 1):
            print(f"\n--- 测试用例 {i}/{len(test_dataset)}: {sample['id']} ---")
            print(f"📝 领域: {sample['context'].get('domain', 'general')}")
            print(f"🎯 难度: {sample['context'].get('difficulty', 'normal')}")
            print(f"❓ 问题: {sample['input'][:100]}{'...' if len(sample['input']) > 100 else ''}")
            
            test_result = {
                "sample_id": sample["id"],
                "domain": sample['context'].get('domain', 'general'),
                "difficulty": sample['context'].get('difficulty', 'normal'),
                "input": sample["input"],
                "success": False,
                "response": "",
                "response_time": 0.0,
                "tokens_used": 0,
                "error": None
            }
            
            try:
                start_time = time.time()
                
                # 调用模型
                response = adapter.predict(sample["input"], sample["context"])
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # 估算token使用量（简单估算）
                estimated_tokens = len(sample["input"]) // 4 + len(response) // 4
                
                test_result.update({
                    "success": True,
                    "response": response,
                    "response_time": response_time,
                    "tokens_used": estimated_tokens
                })
                
                results["summary"]["successful_tests"] += 1
                results["summary"]["total_response_time"] += response_time
                results["summary"]["total_tokens_used"] += estimated_tokens
                
                print(f"✅ 成功 (耗时: {response_time:.2f}s, 约{estimated_tokens}tokens)")
                
                if verbose:
                    print(f"📄 回复内容:")
                    print("-" * 40)
                    print(response[:300] + ("..." if len(response) > 300 else ""))
                    print("-" * 40)
                else:
                    print(f"📄 回复预览: {response[:100]}{'...' if len(response) > 100 else ''}")
                
            except Exception as e:
                test_result.update({
                    "success": False,
                    "error": str(e)
                })
                
                results["summary"]["failed_tests"] += 1
                print(f"❌ 失败: {str(e)}")
            
            results["tests"].append(test_result)
            
            # 避免请求过于频繁
            if i < len(test_dataset):
                print("⏳ 等待1秒避免频率限制...")
                time.sleep(1)
        
        # 计算平均响应时间
        if results["summary"]["successful_tests"] > 0:
            results["summary"]["average_response_time"] = (
                results["summary"]["total_response_time"] / results["summary"]["successful_tests"]
            )
        
        # 打印测试总结
        print(f"\n📊 测试总结")
        print("=" * 60)
        print(f"🎯 模型: BigModel {model_name}")
        print(f"📈 总测试数: {results['summary']['total_tests']}")
        print(f"✅ 成功: {results['summary']['successful_tests']}")
        print(f"❌ 失败: {results['summary']['failed_tests']}")
        print(f"📊 成功率: {results['summary']['successful_tests']/results['summary']['total_tests']*100:.1f}%")
        
        if results["summary"]["average_response_time"] > 0:
            print(f"⏱️  平均响应时间: {results['summary']['average_response_time']:.2f}s")
        
        if results["summary"]["total_tokens_used"] > 0:
            print(f"🔢 总Token使用量: ~{results['summary']['total_tokens_used']}")
        
        # 按领域统计
        domain_stats = {}
        for test in results["tests"]:
            domain = test["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "success": 0}
            domain_stats[domain]["total"] += 1
            if test["success"]:
                domain_stats[domain]["success"] += 1
        
        print(f"\n📋 按领域统计:")
        for domain, stats in domain_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            print(f"  {domain}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        results["domain_stats"] = domain_stats
        results["success"] = True
        
        return results
        
    except Exception as e:
        print(f"💥 测试过程中发生错误: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name,
            "provider": "bigmodel"
        }


def demonstrate_curl_equivalent(api_key: str, model_name: str = "glm-4.5"):
    """演示等效的curl命令"""
    print(f"\n🔧 等效的curl命令:")
    print("=" * 60)
    
    curl_command = f'''curl --request POST \\
  --url https://open.bigmodel.cn/api/paas/v4/chat/completions \\
  --header 'Authorization: Bearer {api_key}' \\
  --header 'Content-Type: application/json' \\
  --data '{{
    "model": "{model_name}",
    "messages": [
      {{
        "role": "user",
        "content": "中国大模型行业在2025年将面临哪些机遇和挑战？"
      }}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }}'
'''
    
    print(curl_command)
    print("\n💡 使用industry_evaluation系统的优势:")
    print("  ✅ 自动重试和错误处理")
    print("  ✅ 统一的接口适配多个提供商")
    print("  ✅ 内置的评估和分析功能")
    print("  ✅ 批量测试和性能统计")
    print("  ✅ 配置管理和环境变量支持")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BigModel GLM API 演示")
    parser.add_argument("--api-key", help="BigModel API密钥 (也可通过环境变量BIGMODEL_API_KEY设置)")
    parser.add_argument("--model", default="glm-4.5", help="模型名称 (默认: glm-4.5)")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细输出")
    parser.add_argument("--output", "-o", help="保存结果到JSON文件")
    parser.add_argument("--show-curl", action="store_true", help="显示等效的curl命令")
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = args.api_key or os.getenv("BIGMODEL_API_KEY")
    
    if not api_key:
        print("❌ 错误: 未提供API密钥")
        print("请通过以下方式之一提供API密钥:")
        print("  1. 命令行参数: --api-key your_api_key_here")
        print("  2. 环境变量: export BIGMODEL_API_KEY=your_api_key_here")
        print("\n💡 获取API密钥:")
        print("  访问 https://open.bigmodel.cn 注册账号并获取API密钥")
        return 1
    
    print("BigModel GLM API 演示")
    print("=" * 60)
    print(f"🎯 目标模型: {args.model}")
    print(f"🔑 API密钥: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    
    # 显示curl命令示例
    if args.show_curl:
        demonstrate_curl_equivalent(api_key, args.model)
        print()
    
    # 执行测试
    results = test_bigmodel_glm_api(
        api_key=api_key,
        model_name=args.model,
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
    
    # 提供后续建议
    if results.get("success", False):
        print(f"\n🎉 测试完成！后续建议:")
        print("  📊 查看详细结果文件了解更多信息")
        print("  🔧 调整temperature和max_tokens参数优化输出")
        print("  📈 集成到完整的评估流程中进行深度分析")
        print("  🔄 定期运行测试监控模型性能")
    
    # 返回退出码
    return 0 if results.get("success", False) else 1


if __name__ == "__main__":
    exit(main())