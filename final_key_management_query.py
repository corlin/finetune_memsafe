#!/usr/bin/env python3
"""
最终版本：询问秘钥管理问题的完整程序
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    model_path = "exported_models/qwen3_merged_lightweight/Qwen_Qwen3-4B-Thinking-2507_pytorch_20250816_225620"
    
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return
    
    try:
        logger.info("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            return_dict=True
        )
        
        model.config.return_dict = True
        device = next(model.parameters()).device
        logger.info(f"模型已加载到设备: {device}")
        
        # 问题
        question = "什么是秘钥管理？请简要说明其概念和重要性。"
        prompt = f"问题：{question}\n回答："
        
        logger.info("正在生成回答...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 提取回答
        generated_tokens = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print("\n" + "="*60)
        print("问题:", question)
        print("="*60)
        print("模型回答:")
        print(response.strip())
        print("="*60)
        print("\n程序执行完成！")
        
    except Exception as e:
        logger.error(f"运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()