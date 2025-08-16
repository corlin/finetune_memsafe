"""
推理测试器 - 用于验证微调模型的推理能力
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass
from .memory_optimizer import MemoryOptimizer

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """推理结果数据类"""
    prompt: str
    response: str
    generation_time: float
    token_count: int
    success: bool
    error_message: Optional[str] = None


class InferenceTester:
    """用于模型验证的推理测试器类"""
    
    def __init__(self, memory_optimizer: Optional[MemoryOptimizer] = None):
        """
        初始化推理测试器
        
        Args:
            memory_optimizer: 可选的内存优化器实例
        """
        self.memory_optimizer = memory_optimizer or MemoryOptimizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"InferenceTester初始化完成，设备: {self.device}")
    
    def load_finetuned_model(self, model_path: str, base_model_name: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载带PEFT适配器的微调模型
        
        Args:
            model_path: 微调模型路径
            base_model_name: 基础模型名称，如果为None则从模型路径推断
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: 加载的模型和分词器
        """
        logger.info(f"开始加载微调模型: {model_path}")
        
        try:
            # 检查模型路径是否存在
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            # 清理内存
            self.memory_optimizer.cleanup_gpu_memory()
            
            # 尝试从adapter_config.json获取基础模型信息
            adapter_config_path = model_path / "adapter_config.json"
            if adapter_config_path.exists() and base_model_name is None:
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get('base_model_name_or_path')
                    logger.info(f"从adapter_config.json获取基础模型: {base_model_name}")
            
            if base_model_name is None:
                raise ValueError("无法确定基础模型名称，请提供base_model_name参数")
            
            # 加载基础模型
            logger.info(f"加载基础模型: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # 加载PEFT适配器
            logger.info("加载PEFT适配器...")
            model = PeftModel.from_pretrained(
                base_model,
                str(model_path),
                torch_dtype=torch.bfloat16
            )
            
            # 合并适配器权重以提高推理速度
            logger.info("合并适配器权重...")
            model = model.merge_and_unload()
            
            # 加载分词器
            logger.info("加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                padding_side="left"  # 推理时使用左填充
            )
            
            # 配置pad token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})
            
            # 设置为评估模式
            model.eval()
            
            # 存储模型和分词器
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info("微调模型加载完成")
            self.memory_optimizer.log_memory_status("模型加载后")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"微调模型加载失败: {str(e)}")
            raise RuntimeError(f"无法加载微调模型 {model_path}: {str(e)}")
    
    def _format_prompt_for_qwen(self, prompt: str, system_message: str = None) -> str:
        """
        为Qwen3格式化提示
        
        Args:
            prompt: 用户提示
            system_message: 可选的系统消息
            
        Returns:
            str: 格式化后的提示
        """
        if system_message is None:
            system_message = "你是一个有用的AI助手。"
        
        # Qwen3对话格式
        formatted_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        return formatted_prompt
    
    def test_inference(self, prompt: str, model: AutoModelForCausalLM = None, 
                      tokenizer: AutoTokenizer = None, system_message: str = None,
                      max_new_tokens: int = 256, temperature: float = 0.7,
                      top_p: float = 0.9, do_sample: bool = True) -> str:
        """
        实现带适当提示格式化的推理测试
        
        Args:
            prompt: 输入提示
            model: 可选的模型实例，如果为None则使用已加载的模型
            tokenizer: 可选的分词器实例，如果为None则使用已加载的分词器
            system_message: 可选的系统消息
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            top_p: Top-p采样参数
            do_sample: 是否使用采样
            
        Returns:
            str: 生成的响应
        """
        # 使用提供的模型和分词器，或使用已加载的
        model = model or self.model
        tokenizer = tokenizer or self.tokenizer
        
        if model is None or tokenizer is None:
            raise ValueError("模型或分词器未加载，请先调用load_finetuned_model或提供模型参数")
        
        logger.info(f"开始推理测试，提示长度: {len(prompt)}")
        
        try:
            # 格式化提示
            formatted_prompt = self._format_prompt_for_qwen(prompt, system_message)
            
            # 分词
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            )
            
            # 移动到设备
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 检查内存安全
            self.memory_optimizer.check_memory_safety()
            
            # 生成响应
            logger.info("开始生成响应...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
            
            generation_time = time.time() - start_time
            
            # 解码响应
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分（移除原始提示）
            response = full_response[len(formatted_prompt):].strip()
            
            # 清理响应中的特殊标记
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
            
            logger.info(f"推理完成，生成时间: {generation_time:.2f}秒")
            logger.info(f"响应长度: {len(response)}")
            
            return response
            
        except Exception as e:
            logger.error(f"推理测试失败: {str(e)}")
            raise RuntimeError(f"推理测试失败: {str(e)}")
    
    def generate_with_optimized_params(self, prompt: str, **kwargs) -> InferenceResult:
        """
        创建带优化参数的响应生成
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Returns:
            InferenceResult: 推理结果对象
        """
        logger.info("使用优化参数生成响应")
        
        # 默认优化参数
        default_params = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "system_message": "你是一个有用、准确且简洁的AI助手。"
        }
        
        # 合并用户提供的参数
        params = {**default_params, **kwargs}
        
        start_time = time.time()
        success = True
        error_message = None
        response = ""
        
        try:
            # 执行推理
            response = self.test_inference(prompt, **params)
            
            # 计算token数量
            token_count = len(self.tokenizer.encode(response))
            
        except Exception as e:
            success = False
            error_message = str(e)
            token_count = 0
            logger.error(f"优化参数生成失败: {error_message}")
        
        generation_time = time.time() - start_time
        
        result = InferenceResult(
            prompt=prompt,
            response=response,
            generation_time=generation_time,
            token_count=token_count,
            success=success,
            error_message=error_message
        )
        
        logger.info(f"优化参数生成完成，成功: {success}, 时间: {generation_time:.2f}秒")
        
        return result
    
    def batch_inference(self, prompts: List[str], **kwargs) -> List[InferenceResult]:
        """
        批量推理测试
        
        Args:
            prompts: 提示列表
            **kwargs: 生成参数
            
        Returns:
            List[InferenceResult]: 推理结果列表
        """
        logger.info(f"开始批量推理，提示数量: {len(prompts)}")
        
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"处理提示 {i+1}/{len(prompts)}")
            
            try:
                result = self.generate_with_optimized_params(prompt, **kwargs)
                results.append(result)
                
                # 每次推理后清理内存
                if i % 5 == 0:  # 每5次清理一次
                    self.memory_optimizer.cleanup_gpu_memory()
                    
            except Exception as e:
                logger.error(f"批量推理第{i+1}个提示失败: {str(e)}")
                # 创建失败结果
                failed_result = InferenceResult(
                    prompt=prompt,
                    response="",
                    generation_time=0.0,
                    token_count=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)
        
        logger.info(f"批量推理完成，成功: {sum(1 for r in results if r.success)}/{len(results)}")
        
        return results
    
    def test_model_with_multiple_prompts(self, test_prompts: List[str] = None, **kwargs) -> List[InferenceResult]:
        """
        编写使用多个提示测试模型的函数
        
        Args:
            test_prompts: 测试提示列表，如果为None则使用默认测试集
            **kwargs: 生成参数
            
        Returns:
            List[InferenceResult]: 测试结果列表
        """
        logger.info("开始多提示模型测试")
        
        # 默认测试提示集
        if test_prompts is None:
            test_prompts = [
                "请解释什么是机器学习？",
                "如何优化深度学习模型的性能？",
                "Python中如何处理异常？",
                "请写一个简单的排序算法。",
                "什么是RESTful API？",
                "解释一下什么是数据结构中的栈？",
                "如何在Python中读取CSV文件？",
                "什么是版本控制系统？",
                "请解释面向对象编程的基本概念。",
                "如何优化数据库查询性能？"
            ]
        
        logger.info(f"使用{len(test_prompts)}个测试提示")
        
        # 执行批量推理
        results = self.batch_inference(test_prompts, **kwargs)
        
        # 记录测试统计
        successful_tests = sum(1 for r in results if r.success)
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"多提示测试完成:")
        logger.info(f"  成功: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        logger.info(f"  平均生成时间: {sum(r.generation_time for r in results if r.success) / successful_tests:.2f}秒")
        
        return results
    
    def validate_response_quality(self, response: str) -> Dict[str, Any]:
        """
        创建生成响应的基本质量检查
        
        Args:
            response: 要验证的响应
            
        Returns:
            Dict[str, Any]: 质量验证结果
        """
        logger.debug(f"验证响应质量，长度: {len(response)}")
        
        quality_metrics = {
            "length": len(response),
            "word_count": len(response.split()),
            "has_content": len(response.strip()) > 0,
            "min_length_met": len(response.strip()) >= 10,
            "max_length_reasonable": len(response) <= 2000,
            "no_repetition": self._check_repetition(response),
            "coherent_structure": self._check_coherence(response),
            "appropriate_language": self._check_language_appropriateness(response),
            "no_errors": self._check_for_errors(response),
            "overall_score": 0.0
        }
        
        # 计算总体质量分数
        quality_metrics["overall_score"] = self._calculate_quality_score(quality_metrics)
        
        logger.debug(f"质量验证完成，总分: {quality_metrics['overall_score']:.2f}")
        
        return quality_metrics
    
    def _check_repetition(self, response: str) -> bool:
        """检查响应是否有过度重复"""
        words = response.split()
        if len(words) < 5:
            return True
        
        # 检查连续重复的词
        consecutive_repeats = 0
        max_consecutive = 0
        
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_repeats += 1
                max_consecutive = max(max_consecutive, consecutive_repeats + 1)  # +1 因为我们从第二个重复开始计数
            else:
                consecutive_repeats = 0
        
        # 如果连续重复超过3个词，认为有问题
        return max_consecutive <= 3
    
    def _check_coherence(self, response: str) -> bool:
        """检查响应的连贯性"""
        # 基本连贯性检查
        sentences = response.split('。')
        
        if len(sentences) <= 2:
            return True  # 单句或两句认为是连贯的
        
        # 检查是否有完整的句子结构
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 3)
        
        return complete_sentences >= len(sentences) * 0.6
    
    def _check_language_appropriateness(self, response: str) -> bool:
        """检查语言的适当性"""
        # 检查是否包含不当内容的基本标记
        inappropriate_markers = [
            "抱歉，我不能",
            "我无法回答",
            "这个问题不合适",
            "ERROR",
            "FAILED",
            "<|im_start|>",
            "<|im_end|>"
        ]
        
        response_lower = response.lower()
        
        for marker in inappropriate_markers:
            if marker.lower() in response_lower:
                return False
        
        return True
    
    def _check_for_errors(self, response: str) -> bool:
        """检查明显的错误"""
        # 检查是否包含明显的错误标记
        error_markers = [
            "traceback",
            "exception",
            "error:",
            "failed:",
            "null",
            "undefined",
            "nan"
        ]
        
        response_lower = response.lower()
        
        for marker in error_markers:
            if marker in response_lower:
                return False
        
        return True
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """计算总体质量分数"""
        score = 0.0
        
        # 基本内容检查 (30%) - 如果没有内容，分数为0
        if not metrics["has_content"]:
            return 0.0
        
        if metrics["min_length_met"]:
            score += 0.30
        
        # 长度合理性 (10%)
        if metrics["max_length_reasonable"]:
            score += 0.10
        
        # 重复检查 (20%)
        if metrics["no_repetition"]:
            score += 0.20
        
        # 连贯性 (20%)
        if metrics["coherent_structure"]:
            score += 0.20
        
        # 语言适当性 (10%)
        if metrics["appropriate_language"]:
            score += 0.10
        
        # 错误检查 (10%)
        if metrics["no_errors"]:
            score += 0.10
        
        return round(score, 2)
    
    def validate_model_quality(self, test_results: List[InferenceResult] = None, 
                              quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        验证模型整体质量
        
        Args:
            test_results: 测试结果列表，如果为None则运行默认测试
            quality_threshold: 质量阈值
            
        Returns:
            Dict[str, Any]: 模型质量验证结果
        """
        logger.info("开始模型质量验证")
        
        # 如果没有提供测试结果，运行默认测试
        if test_results is None:
            test_results = self.test_model_with_multiple_prompts()
        
        # 验证每个响应的质量
        quality_results = []
        successful_results = [r for r in test_results if r.success]
        
        for result in successful_results:
            quality_metrics = self.validate_response_quality(result.response)
            quality_results.append(quality_metrics)
        
        # 计算整体统计
        if not quality_results:
            logger.warning("没有成功的测试结果用于质量验证")
            return {
                "overall_quality": 0.0,
                "passed_tests": 0,
                "total_tests": len(test_results),
                "success_rate": 0.0,
                "quality_passed": False,
                "recommendations": ["模型推理完全失败，需要检查模型加载和配置"]
            }
        
        # 统计指标
        total_tests = len(test_results)
        successful_tests = len(successful_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # 质量分数统计
        quality_scores = [q["overall_score"] for q in quality_results]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        high_quality_count = sum(1 for score in quality_scores if score >= quality_threshold)
        quality_pass_rate = (high_quality_count / len(quality_scores)) * 100 if quality_scores else 0
        
        # 详细指标统计
        avg_length = sum(q["length"] for q in quality_results) / len(quality_results)
        avg_word_count = sum(q["word_count"] for q in quality_results) / len(quality_results)
        coherence_rate = sum(1 for q in quality_results if q["coherent_structure"]) / len(quality_results) * 100
        repetition_rate = sum(1 for q in quality_results if q["no_repetition"]) / len(quality_results) * 100
        
        # 生成建议
        recommendations = self._generate_quality_recommendations(
            avg_quality, success_rate, quality_pass_rate, quality_results
        )
        
        validation_result = {
            "overall_quality": round(avg_quality, 2),
            "passed_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": round(success_rate, 1),
            "quality_pass_rate": round(quality_pass_rate, 1),
            "quality_passed": avg_quality >= quality_threshold,
            "detailed_metrics": {
                "avg_response_length": round(avg_length, 1),
                "avg_word_count": round(avg_word_count, 1),
                "coherence_rate": round(coherence_rate, 1),
                "repetition_control_rate": round(repetition_rate, 1),
                "quality_scores": quality_scores
            },
            "recommendations": recommendations
        }
        
        # 记录验证结果
        logger.info(f"模型质量验证完成:")
        logger.info(f"  整体质量分数: {avg_quality:.2f}")
        logger.info(f"  成功率: {success_rate:.1f}%")
        logger.info(f"  质量通过率: {quality_pass_rate:.1f}%")
        logger.info(f"  质量验证: {'通过' if validation_result['quality_passed'] else '未通过'}")
        
        return validation_result
    
    def _generate_quality_recommendations(self, avg_quality: float, success_rate: float, 
                                        quality_pass_rate: float, quality_results: List[Dict]) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        # 成功率建议
        if success_rate < 90:
            recommendations.append(f"推理成功率较低({success_rate:.1f}%)，建议检查模型加载和推理配置")
        
        # 质量分数建议
        if avg_quality < 0.5:
            recommendations.append("整体质量分数较低，建议增加训练数据或调整训练参数")
        elif avg_quality < 0.7:
            recommendations.append("质量分数中等，建议优化训练数据质量或增加训练轮数")
        
        # 具体问题建议
        coherence_issues = sum(1 for q in quality_results if not q["coherent_structure"])
        if coherence_issues > len(quality_results) * 0.3:
            recommendations.append("响应连贯性问题较多，建议改进训练数据的对话质量")
        
        repetition_issues = sum(1 for q in quality_results if not q["no_repetition"])
        if repetition_issues > len(quality_results) * 0.2:
            recommendations.append("存在重复问题，建议调整生成参数(temperature, repetition_penalty)")
        
        length_issues = sum(1 for q in quality_results if not q["min_length_met"])
        if length_issues > len(quality_results) * 0.2:
            recommendations.append("响应长度过短，建议调整max_new_tokens参数或改进提示格式")
        
        # 如果没有问题，给出积极建议
        if not recommendations:
            recommendations.append("模型质量良好，可以考虑在更复杂的任务上进行测试")
        
        return recommendations
    
    def handle_inference_failure(self, error: Exception, prompt: str) -> Dict[str, Any]:
        """
        实现推理失败的错误处理
        
        Args:
            error: 推理错误
            prompt: 失败的提示
            
        Returns:
            Dict[str, Any]: 错误处理结果
        """
        logger.error(f"推理失败处理: {str(error)}")
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "prompt": prompt,
            "timestamp": time.time(),
            "recovery_attempted": False,
            "recovery_successful": False,
            "suggestions": []
        }
        
        # 根据错误类型提供不同的处理策略
        if "out of memory" in str(error).lower() or "cuda" in str(error).lower():
            error_info["suggestions"].extend([
                "减少max_new_tokens参数",
                "清理GPU内存",
                "使用更小的batch_size",
                "检查GPU内存使用情况"
            ])
            
            # 尝试内存清理恢复
            try:
                self.memory_optimizer.cleanup_gpu_memory()
                error_info["recovery_attempted"] = True
                error_info["recovery_successful"] = True
                logger.info("内存清理恢复成功")
            except Exception as recovery_error:
                logger.error(f"内存清理恢复失败: {recovery_error}")
        
        elif "model" in str(error).lower() or "load" in str(error).lower():
            error_info["suggestions"].extend([
                "检查模型路径是否正确",
                "验证模型文件完整性",
                "确认基础模型名称正确",
                "重新加载模型"
            ])
        
        elif "tokenizer" in str(error).lower():
            error_info["suggestions"].extend([
                "检查分词器配置",
                "验证特殊token设置",
                "确认输入文本编码正确"
            ])
        
        else:
            error_info["suggestions"].extend([
                "检查输入参数格式",
                "验证模型和分词器状态",
                "查看详细错误日志",
                "尝试使用更简单的提示"
            ])
        
        logger.info(f"错误处理完成，建议数量: {len(error_info['suggestions'])}")
        
        return error_info
    
    def run_comprehensive_test(self, model_path: str, base_model_name: str = None,
                             custom_prompts: List[str] = None, 
                             quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        运行综合测试，包括模型加载、推理测试和质量验证
        
        Args:
            model_path: 模型路径
            base_model_name: 基础模型名称
            custom_prompts: 自定义测试提示
            quality_threshold: 质量阈值
            
        Returns:
            Dict[str, Any]: 综合测试结果
        """
        logger.info("开始综合模型测试")
        
        test_result = {
            "model_path": model_path,
            "base_model_name": base_model_name,
            "load_successful": False,
            "inference_results": [],
            "quality_validation": {},
            "errors": [],
            "overall_success": False
        }
        
        try:
            # 1. 加载模型
            logger.info("步骤1: 加载模型")
            self.load_finetuned_model(model_path, base_model_name)
            test_result["load_successful"] = True
            logger.info("模型加载成功")
            
            # 2. 运行推理测试
            logger.info("步骤2: 运行推理测试")
            test_prompts = custom_prompts or None
            inference_results = self.test_model_with_multiple_prompts(test_prompts)
            test_result["inference_results"] = [
                {
                    "prompt": r.prompt,
                    "response": r.response,
                    "success": r.success,
                    "generation_time": r.generation_time,
                    "token_count": r.token_count,
                    "error_message": r.error_message
                }
                for r in inference_results
            ]
            
            # 3. 质量验证
            logger.info("步骤3: 质量验证")
            quality_validation = self.validate_model_quality(inference_results, quality_threshold)
            test_result["quality_validation"] = quality_validation
            
            # 4. 判断整体成功
            test_result["overall_success"] = (
                test_result["load_successful"] and
                quality_validation["success_rate"] > 80 and
                quality_validation["quality_passed"]
            )
            
            logger.info(f"综合测试完成，整体成功: {test_result['overall_success']}")
            
        except Exception as e:
            logger.error(f"综合测试失败: {str(e)}")
            error_info = self.handle_inference_failure(e, "综合测试")
            test_result["errors"].append(error_info)
        
        finally:
            # 清理资源
            self.cleanup()
        
        return test_result
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理推理测试器资源...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理GPU内存
        self.memory_optimizer.cleanup_gpu_memory()
        
        logger.info("推理测试器资源清理完成")