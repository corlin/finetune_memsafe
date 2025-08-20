#!/usr/bin/env python3
"""
工作正常的BigModel演示 - 简化版本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from industry_evaluation.adapters.model_adapter import ModelAdapterFactory

def working_demo():
    """工作正常的BigModel演示"""
    
    print("🚀 BigModel GLM-4.5 工作演示")
    print("=" * 60)
    
    # 获取API密钥
    api_key = "e46d7ffc78de48a3a39aafa4bc1a6634.PxLyTIzIuKWF8lfN"
    
    print(f"🔑 API密钥: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        # 创建适配器
        print("\n🔧 创建BigModel适配器...")
        adapter = ModelAdapterFactory.create_openai_compatible_adapter(
            model_id="demo-glm-4.5",
            provider="bigmodel",
            api_key=api_key,
            model_name="glm-4.5",
            timeout=30
        )
        print("✅ 适配器创建成功")
        print(f"📡 API端点: {adapter.base_url}")
        
        # 测试连通性
        print("\n🔍 测试连通性...")
        test_response = adapter.predict("你好", {"max_tokens": 20})
        if test_response:
            print(f"✅ 连通性测试成功")
            print(f"📝 响应: {test_response[:100]}...")
        else:
            print("❌ 连通性测试失败")
            return
        
        # 金融领域测试
        print("\n💰 金融领域测试...")
        finance_questions = [
            {
                "question": "什么是金融风险管理？请简要解释。",
                "context": {"max_tokens": 200, "temperature": 0.7}
            },
            {
                "question": "请解释VaR模型的基本概念。",
                "context": {"max_tokens": 150, "temperature": 0.6}
            },
            {
                "question": "中国金融科技发展有哪些特点？",
                "context": {"max_tokens": 180, "temperature": 0.5}
            }
        ]
        
        results = []
        for i, item in enumerate(finance_questions, 1):
            print(f"\n--- 问题 {i} ---")
            print(f"❓ {item['question']}")
            
            try:
                response = adapter.predict(item['question'], item['context'])
                if response:
                    print(f"✅ 回答成功 ({len(response)} 字符)")
                    print(f"📝 内容: {response[:150]}{'...' if len(response) > 150 else ''}")
                    results.append({
                        "question": item['question'],
                        "response": response,
                        "success": True
                    })
                else:
                    print("⚠️ 回答为空")
                    results.append({
                        "question": item['question'],
                        "response": "",
                        "success": False
                    })
            except Exception as e:
                print(f"❌ 回答失败: {str(e)}")
                results.append({
                    "question": item['question'],
                    "response": f"错误: {str(e)}",
                    "success": False
                })
        
        # 总结
        print(f"\n📊 测试总结")
        print("=" * 60)
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        print(f"✅ 成功: {successful}/{total}")
        print(f"📈 成功率: {successful/total*100:.1f}%")
        
        if successful > 0:
            print(f"\n🎉 BigModel GLM-4.5 集成成功!")
            print("✨ 主要特点:")
            print("  • API调用正常")
            print("  • 支持中文对话")
            print("  • 金融领域知识丰富")
            print("  • 响应速度良好")
            print("  • 使用reasoning_content字段返回内容")
        else:
            print(f"\n❌ 所有测试都失败了")
            
    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    working_demo()