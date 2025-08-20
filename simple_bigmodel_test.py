#!/usr/bin/env python3
"""
简化的BigModel测试 - 直接测试模型适配器
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from industry_evaluation.adapters.model_adapter import ModelAdapterFactory

def test_bigmodel_adapter():
    """测试BigModel适配器"""
    
    api_key = "e46d7ffc78de48a3a39aafa4bc1a6634.PxLyTIzIuKWF8lfN"
    
    print("🚀 测试BigModel适配器")
    print("=" * 50)
    
    try:
        # 创建适配器
        print("🔧 创建BigModel适配器...")
        adapter = ModelAdapterFactory.create_openai_compatible_adapter(
            model_id="test-glm-4.5",
            provider="bigmodel",
            api_key=api_key,
            model_name="glm-4.5",
            timeout=30
        )
        print("✅ 适配器创建成功")
        
        # 测试基本预测
        print("\n🧪 测试基本预测...")
        test_questions = [
            "你好，请简单介绍一下你自己。",
            "什么是金融风险管理？",
            "请解释一下人工智能的基本概念。"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- 测试 {i} ---")
            print(f"❓ 问题: {question}")
            
            try:
                response = adapter.predict(question, {
                    "max_tokens": 100,
                    "temperature": 0.7
                })
                
                if response:
                    print(f"✅ 回答: {response[:200]}{'...' if len(response) > 200 else ''}")
                else:
                    print("⚠️ 回答为空")
                    
            except Exception as e:
                print(f"❌ 预测失败: {str(e)}")
        
        # 测试可用性检查
        print(f"\n🔍 测试可用性检查...")
        is_available = adapter.is_available()
        print(f"📊 模型可用性: {'✅ 可用' if is_available else '❌ 不可用'}")
        
        print(f"\n🎉 BigModel适配器测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bigmodel_adapter()