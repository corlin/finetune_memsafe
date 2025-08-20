#!/usr/bin/env python3
"""
直接测试BigModel API的简单脚本
"""

import requests
import json
import os

def test_bigmodel_api():
    """直接测试BigModel API"""
    
    # API配置
    api_key = "e46d7ffc78de48a3a39aafa4bc1a6634.PxLyTIzIuKWF8lfN"
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    endpoint = "/chat/completions"
    
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Industry-Evaluation-Demo/1.0"
    }
    
    # 请求数据
    data = {
        "model": "glm-4.5",
        "messages": [
            {"role": "user", "content": "Hello, please introduce yourself briefly."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print("🔍 测试BigModel API...")
    print(f"📡 URL: {base_url}{endpoint}")
    print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"📝 请求数据: {json.dumps(data, indent=2)}")
    
    try:
        # 发送请求
        response = requests.post(
            f"{base_url}{endpoint}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"\n📊 响应状态码: {response.status_code}")
        print(f"📋 响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 响应成功!")
            print(f"📄 完整响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 尝试解析内容
            if "choices" in result and result["choices"]:
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                reasoning_content = message.get("reasoning_content", "")
                
                print(f"💬 content字段: '{content}'")
                print(f"🧠 reasoning_content字段: '{reasoning_content}'")
                
                # 使用优先级：content > reasoning_content
                final_content = content if content and content.strip() else reasoning_content
                print(f"✨ 最终内容: '{final_content}'")
                
                if final_content and final_content.strip():
                    print("✅ API调用成功，内容不为空")
                else:
                    print("⚠️ API调用成功，但内容为空")
            else:
                print("❌ 响应格式不符合预期")
        else:
            print(f"❌ API调用失败")
            print(f"📄 错误响应: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {str(e)}")
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")

if __name__ == "__main__":
    test_bigmodel_api()