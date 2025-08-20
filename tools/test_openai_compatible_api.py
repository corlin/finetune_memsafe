#!/usr/bin/env python3
"""
OpenAI兼容API快速测试工具

这是一个简单的命令行工具，用于快速测试OpenAI兼容的API，
类似于curl命令，但提供了更友好的界面和错误处理。

使用示例:
# 测试BigModel GLM-4.5
python tools/test_openai_compatible_api.py \
  --url "https://open.bigmodel.cn/api/paas/v4/chat/completions" \
  --token "your_api_key" \
  --model "glm-4.5" \
  --message "中国大模型行业在2025年将面临哪些机遇和挑战？"

# 测试DeepSeek
python tools/test_openai_compatible_api.py \
  --url "https://api.deepseek.com/v1/chat/completions" \
  --token "your_api_key" \
  --model "deepseek-chat" \
  --message "什么是人工智能？"

# 从文件读取消息
python tools/test_openai_compatible_api.py \
  --url "https://api.openai.com/v1/chat/completions" \
  --token "your_api_key" \
  --model "gpt-3.5-turbo" \
  --message-file "test_message.txt"
"""

import sys
import os
import argparse
import json
import time
import requests
from typing import Dict, Any, Optional


def send_chat_request(url: str, token: str, model: str, message: str,
                     system_prompt: Optional[str] = None,
                     max_tokens: int = 1000,
                     temperature: float = 0.7,
                     timeout: int = 30,
                     custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    发送聊天请求到OpenAI兼容API
    
    Args:
        url: API端点URL
        token: API令牌
        model: 模型名称
        message: 用户消息
        system_prompt: 系统提示词
        max_tokens: 最大令牌数
        temperature: 温度参数
        timeout: 超时时间
        custom_headers: 自定义请求头
        
    Returns:
        Dict[str, Any]: API响应结果
    """
    # 构建请求头
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    if custom_headers:
        headers.update(custom_headers)
    
    # 构建消息
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    
    # 构建请求数据
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        print(f"🚀 发送请求到: {url}")
        print(f"📝 模型: {model}")
        print(f"💬 消息: {message[:100]}{'...' if len(message) > 100 else ''}")
        print(f"⚙️  参数: max_tokens={max_tokens}, temperature={temperature}")
        if system_prompt:
            print(f"🎯 系统提示: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}")
        print()
        
        start_time = time.time()
        
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=timeout
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  响应时间: {response_time:.2f}秒")
        print(f"📊 状态码: {response.status_code}")
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            
            # 提取回复内容
            if "choices" in result and result["choices"]:
                content = result["choices"][0].get("message", {}).get("content", "")
                usage = result.get("usage", {})
                
                print("✅ 请求成功!")
                print(f"📄 回复内容:")
                print("-" * 50)
                print(content)
                print("-" * 50)
                
                if usage:
                    print(f"📈 令牌使用: 输入={usage.get('prompt_tokens', 'N/A')}, "
                          f"输出={usage.get('completion_tokens', 'N/A')}, "
                          f"总计={usage.get('total_tokens', 'N/A')}")
                
                return {
                    "success": True,
                    "content": content,
                    "response_time": response_time,
                    "usage": usage,
                    "raw_response": result
                }
            else:
                print("❌ 响应格式异常: 缺少choices字段")
                return {
                    "success": False,
                    "error": "响应格式异常",
                    "raw_response": result
                }
        else:
            # 处理错误响应
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "未知错误")
            except:
                error_message = response.text or f"HTTP {response.status_code}"
            
            print(f"❌ 请求失败: {error_message}")
            
            return {
                "success": False,
                "error": error_message,
                "status_code": response.status_code,
                "response_time": response_time
            }
            
    except requests.exceptions.Timeout:
        print(f"⏰ 请求超时 ({timeout}秒)")
        return {"success": False, "error": "请求超时"}
    
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 连接错误: {str(e)}")
        return {"success": False, "error": f"连接错误: {str(e)}"}
    
    except requests.exceptions.RequestException as e:
        print(f"📡 请求错误: {str(e)}")
        return {"success": False, "error": f"请求错误: {str(e)}"}
    
    except json.JSONDecodeError as e:
        print(f"📋 JSON解析错误: {str(e)}")
        return {"success": False, "error": f"JSON解析错误: {str(e)}"}
    
    except Exception as e:
        print(f"💥 未知错误: {str(e)}")
        return {"success": False, "error": f"未知错误: {str(e)}"}


def load_message_from_file(file_path: str) -> str:
    """从文件加载消息内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="OpenAI兼容API快速测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

# 测试BigModel GLM-4.5
python tools/test_openai_compatible_api.py \\
  --url "https://open.bigmodel.cn/api/paas/v4/chat/completions" \\
  --token "your_api_key" \\
  --model "glm-4.5" \\
  --message "中国大模型行业在2025年将面临哪些机遇和挑战？"

# 测试DeepSeek
python tools/test_openai_compatible_api.py \\
  --url "https://api.deepseek.com/v1/chat/completions" \\
  --token "your_api_key" \\
  --model "deepseek-chat" \\
  --message "什么是人工智能？"

# 使用系统提示词
python tools/test_openai_compatible_api.py \\
  --url "https://api.openai.com/v1/chat/completions" \\
  --token "your_api_key" \\
  --model "gpt-3.5-turbo" \\
  --message "解释量子计算" \\
  --system "你是一个专业的科学解释员，请用简单易懂的语言解释复杂概念。"
        """
    )
    
    parser.add_argument("--url", required=True, help="API端点URL")
    parser.add_argument("--token", required=True, help="API令牌")
    parser.add_argument("--model", required=True, help="模型名称")
    
    # 消息输入方式（二选一）
    message_group = parser.add_mutually_exclusive_group(required=True)
    message_group.add_argument("--message", help="用户消息内容")
    message_group.add_argument("--message-file", help="从文件读取消息内容")
    
    # 可选参数
    parser.add_argument("--system", help="系统提示词")
    parser.add_argument("--max-tokens", type=int, default=1000, help="最大令牌数 (默认: 1000)")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数 (默认: 0.7)")
    parser.add_argument("--timeout", type=int, default=30, help="超时时间秒数 (默认: 30)")
    parser.add_argument("--header", action="append", help="自定义请求头，格式: Key:Value")
    parser.add_argument("--output", "-o", help="保存结果到JSON文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    print("OpenAI兼容API测试工具")
    print("=" * 60)
    
    try:
        # 获取消息内容
        if args.message:
            message = args.message
        else:
            message = load_message_from_file(args.message_file)
        
        # 解析自定义请求头
        custom_headers = {}
        if args.header:
            for header in args.header:
                if ":" not in header:
                    print(f"❌ 无效的请求头格式: {header} (应为 Key:Value)")
                    return 1
                key, value = header.split(":", 1)
                custom_headers[key.strip()] = value.strip()
        
        # 发送请求
        result = send_chat_request(
            url=args.url,
            token=args.token,
            model=args.model,
            message=message,
            system_prompt=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            custom_headers=custom_headers if custom_headers else None
        )
        
        # 显示详细信息
        if args.verbose:
            print("\n🔍 详细信息:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 保存结果
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n📁 结果已保存到: {args.output}")
            except Exception as e:
                print(f"❌ 保存结果失败: {str(e)}")
        
        # 返回退出码
        return 0 if result.get("success", False) else 1
        
    except Exception as e:
        print(f"💥 程序错误: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())