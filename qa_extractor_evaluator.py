#!/usr/bin/env python3
"""
QA数据提取器和评估器 - 从markdown文件中提取QA对，生成评估用例，测试checkpoint
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """QA对数据结构"""
    id: str
    question: str
    answer: str
    category: str
    source_file: str
    difficulty: str = "medium"  # easy, medium, hard
    keywords: List[str] = None

@dataclass
class EvaluationCase:
    """评估用例数据结构"""
    id: str
    qa_pair: QAPair
    model_answer: str = ""
    expected_answer: str = ""
    evaluation_score: float = 0.0
    evaluation_details: Dict[str, Any] = None
    timestamp: str = ""

class QAExtractor:
    """QA数据提取器"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.qa_pairs = []
    
    def extract_all_qa_pairs(self) -> List[QAPair]:
        """提取所有QA对"""
        qa_files = list(self.data_dir.glob("QA*.md"))
        logger.info(f"发现 {len(qa_files)} 个QA文件")
        
        all_qa_pairs = []
        for qa_file in qa_files:
            pairs = self._extract_from_file(qa_file)
            all_qa_pairs.extend(pairs)
            logger.info(f"从 {qa_file.name} 提取了 {len(pairs)} 个QA对")
        
        self.qa_pairs = all_qa_pairs
        logger.info(f"总共提取了 {len(all_qa_pairs)} 个QA对")
        return all_qa_pairs
    
    def _extract_from_file(self, file_path: Path) -> List[QAPair]:
        """从单个文件提取QA对"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        qa_pairs = []
        current_category = "未分类"
        
        # 使用正则表达式匹配QA对
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 检测分类标题
            if line.startswith('##') and not line.startswith('###'):
                current_category = line.replace('##', '').strip()
                i += 1
                continue
            
            # 检测问题
            question_match = re.match(r'###\s*Q(\d+):\s*(.+)', line)
            if question_match:
                q_id = question_match.group(1)
                question = question_match.group(2).strip()
                i += 1
                
                # 查找对应的答案
                answer = ""
                while i < len(lines):
                    line = lines[i].strip()
                    answer_match = re.match(r'A\d+:\s*(.+)', line)
                    if answer_match:
                        answer = answer_match.group(1).strip()
                        break
                    i += 1
                
                if answer:
                    # 提取关键词
                    keywords = self._extract_keywords(question, answer)
                    
                    # 判断难度
                    difficulty = self._assess_difficulty(question, answer)
                    
                    qa_pair = QAPair(
                        id=f"{file_path.stem}_Q{q_id}",
                        question=question,
                        answer=answer,
                        category=current_category,
                        source_file=file_path.name,
                        difficulty=difficulty,
                        keywords=keywords
                    )
                    qa_pairs.append(qa_pair)
            
            i += 1
        
        return qa_pairs
    
    def _extract_keywords(self, question: str, answer: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取逻辑
        text = question + " " + answer
        
        # 密码学相关关键词
        crypto_keywords = [
            "密码", "加密", "解密", "密钥", "算法", "签名", "认证", "完整性", 
            "机密性", "真实性", "不可否认性", "杂凑", "对称", "非对称", 
            "数字证书", "PKI", "SM2", "SM3", "SM4", "AES", "RSA", "SHA"
        ]
        
        found_keywords = []
        for keyword in crypto_keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        return found_keywords[:5]  # 最多返回5个关键词
    
    def _assess_difficulty(self, question: str, answer: str) -> str:
        """评估问题难度"""
        # 简单的难度评估逻辑
        answer_length = len(answer)
        question_complexity = len(question.split('、')) + len(question.split('，'))
        
        if answer_length < 50 and question_complexity <= 2:
            return "easy"
        elif answer_length > 200 or question_complexity > 5:
            return "hard"
        else:
            return "medium"
    
    def filter_by_category(self, categories: List[str]) -> List[QAPair]:
        """按分类筛选QA对"""
        if not categories:
            return self.qa_pairs
        
        filtered = [qa for qa in self.qa_pairs if qa.category in categories]
        logger.info(f"按分类筛选后剩余 {len(filtered)} 个QA对")
        return filtered
    
    def filter_by_difficulty(self, difficulties: List[str]) -> List[QAPair]:
        """按难度筛选QA对"""
        if not difficulties:
            return self.qa_pairs
        
        filtered = [qa for qa in self.qa_pairs if qa.difficulty in difficulties]
        logger.info(f"按难度筛选后剩余 {len(filtered)} 个QA对")
        return filtered
    
    def sample_qa_pairs(self, n: int, strategy: str = "random") -> List[QAPair]:
        """采样QA对"""
        if n >= len(self.qa_pairs):
            return self.qa_pairs
        
        if strategy == "random":
            return random.sample(self.qa_pairs, n)
        elif strategy == "balanced":
            # 按分类平衡采样
            categories = list(set(qa.category for qa in self.qa_pairs))
            per_category = max(1, n // len(categories))
            
            sampled = []
            for category in categories:
                category_pairs = [qa for qa in self.qa_pairs if qa.category == category]
                sampled.extend(random.sample(category_pairs, min(per_category, len(category_pairs))))
            
            # 如果还需要更多，随机补充
            if len(sampled) < n:
                remaining = [qa for qa in self.qa_pairs if qa not in sampled]
                sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))
            
            return sampled[:n]
        else:
            return self.qa_pairs[:n]
    
    def save_qa_pairs(self, output_file: str):
        """保存QA对到文件"""
        data = [asdict(qa) for qa in self.qa_pairs]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"QA对已保存到 {output_file}")

class CheckpointEvaluator:
    """Checkpoint评估器"""
    
    def __init__(self, checkpoint_path: str = "enhanced-qwen3-finetuned/checkpoint-450"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.evaluation_cases = []
    
    def load_model(self):
        """加载模型"""
        logger.info(f"正在加载模型: {self.checkpoint_path}")
        
        try:
            # 检查checkpoint路径是否存在
            if not os.path.exists(self.checkpoint_path):
                logger.error(f"Checkpoint路径不存在: {self.checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint路径不存在: {self.checkpoint_path}")
            
            # 检查设备可用性
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"使用GPU设备: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("使用CPU设备")
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_path,
                trust_remote_code=True,
                local_files_only=True  # 只使用本地文件
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("设置pad_token为eos_token")
            
            # 加载模型 - 修复设备不匹配问题和return_dict问题
            logger.info("加载模型...")
            if device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True  # 只使用本地文件
                )
            else:
                # CPU模式
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True  # 只使用本地文件
                )
            
            # 修复return_dict问题 - 强制设置为True
            if hasattr(self.model.config, 'return_dict'):
                self.model.config.return_dict = True
                logger.info("设置模型config.return_dict = True")
            
            # 设置模型为评估模式
            self.model.eval()
            
            # 确保模型在正确的设备上
            self.device = device
            logger.info(f"模型加载成功，设备: {device}")
            logger.info(f"模型类型: {type(self.model)}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def generate_answer(self, question: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """生成答案"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 构建提示词
        prompt = f"问题：{question}\n答案："
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            # 确保输入张量在正确的设备上
            if hasattr(self, 'device') and self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                # 清理GPU缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # 修复模型生成参数，避免tuple错误
                generation_config = {
                    "max_new_tokens": max_length,  # 使用max_new_tokens而不是max_length
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.9,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "return_dict_in_generate": False,  # 确保返回tensor而不是dict
                    "output_scores": False,  # 不返回scores避免tuple问题
                    "output_attentions": False,  # 不返回attention
                    "output_hidden_states": False,  # 不返回hidden_states
                    "use_cache": True
                }
                
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    **generation_config
                )
            
            # 确保outputs是tensor格式
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 如果是tuple，取第一个元素
            
            # 解码回答
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取答案部分
            if "答案：" in response:
                answer = response.split("答案：", 1)[1].strip()
            else:
                answer = response.replace(prompt, "").strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"生成答案时出错: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return f"生成失败: {str(e)}"
    
    def test_model(self) -> bool:
        """测试模型是否正常工作"""
        logger.info("测试模型...")
        try:
            test_question = "什么是密码学？"
            test_answer = self.generate_answer(test_question, max_length=50)
            
            if "生成失败" in test_answer:
                logger.error(f"模型测试失败: {test_answer}")
                return False
            else:
                logger.info(f"模型测试成功，回答: {test_answer[:100]}...")
                return True
        except Exception as e:
            logger.error(f"模型测试出错: {e}")
            return False
    
    def evaluate_qa_pairs(self, qa_pairs: List[QAPair]) -> List[EvaluationCase]:
        """评估QA对"""
        logger.info(f"开始评估 {len(qa_pairs)} 个QA对")
        
        # 先测试模型
        if not self.test_model():
            logger.error("模型测试失败，停止评估")
            return []
        
        evaluation_cases = []
        
        for i, qa_pair in enumerate(qa_pairs):
            logger.info(f"评估进度: {i+1}/{len(qa_pairs)} - {qa_pair.category}")
            
            try:
                # 生成模型答案
                model_answer = self.generate_answer(qa_pair.question)
                
                # 计算评估分数
                score, details = self._calculate_score(qa_pair.answer, model_answer, qa_pair)
                
                # 创建评估用例
                eval_case = EvaluationCase(
                    id=f"eval_{qa_pair.id}",
                    qa_pair=qa_pair,
                    model_answer=model_answer,
                    expected_answer=qa_pair.answer,
                    evaluation_score=score,
                    evaluation_details=details,
                    timestamp=datetime.now().isoformat()
                )
                
                evaluation_cases.append(eval_case)
                logger.info(f"评分: {score:.3f}")
                
            except Exception as e:
                logger.error(f"评估QA对 {qa_pair.id} 时出错: {e}")
                continue
        
        self.evaluation_cases = evaluation_cases
        logger.info(f"评估完成，共 {len(evaluation_cases)} 个用例")
        return evaluation_cases
    
    def _calculate_score(self, expected: str, actual: str, qa_pair: QAPair) -> Tuple[float, Dict[str, Any]]:
        """计算评估分数"""
        details = {}
        
        # 1. 长度相似度 (0-1)
        len_expected = len(expected)
        len_actual = len(actual)
        length_ratio = min(len_actual, len_expected) / max(len_actual, len_expected) if max(len_actual, len_expected) > 0 else 0
        details["length_similarity"] = length_ratio
        
        # 2. 关键词匹配度 (0-1)
        expected_keywords = set(expected.split())
        actual_keywords = set(actual.split())
        
        if len(expected_keywords) > 0:
            keyword_overlap = len(expected_keywords.intersection(actual_keywords)) / len(expected_keywords)
        else:
            keyword_overlap = 0
        details["keyword_overlap"] = keyword_overlap
        
        # 3. 专业术语匹配度 (0-1)
        if qa_pair.keywords:
            expected_terms = set(qa_pair.keywords)
            actual_terms = set()
            for keyword in qa_pair.keywords:
                if keyword in actual:
                    actual_terms.add(keyword)
            
            if len(expected_terms) > 0:
                term_match = len(actual_terms) / len(expected_terms)
            else:
                term_match = 1.0
        else:
            term_match = 0.5  # 默认分数
        details["terminology_match"] = term_match
        
        # 4. 结构相似度 (0-1)
        # 简单检查是否包含数字、标点等结构特征
        expected_has_numbers = bool(re.search(r'\d', expected))
        actual_has_numbers = bool(re.search(r'\d', actual))
        expected_has_punctuation = bool(re.search(r'[，。；：]', expected))
        actual_has_punctuation = bool(re.search(r'[，。；：]', actual))
        
        structure_score = 0
        if expected_has_numbers == actual_has_numbers:
            structure_score += 0.5
        if expected_has_punctuation == actual_has_punctuation:
            structure_score += 0.5
        details["structure_similarity"] = structure_score
        
        # 5. 答案完整性 (0-1)
        # 检查答案是否被截断或过短
        if len(actual) < 10:
            completeness = 0.3
        elif "生成失败" in actual or "出错" in actual:
            completeness = 0.1
        elif len(actual) < len(expected) * 0.3:
            completeness = 0.5
        else:
            completeness = 1.0
        details["completeness"] = completeness
        
        # 综合评分 (加权平均)
        weights = {
            "keyword_overlap": 0.3,
            "terminology_match": 0.25,
            "length_similarity": 0.15,
            "structure_similarity": 0.15,
            "completeness": 0.15
        }
        
        total_score = sum(details[key] * weights[key] for key in weights.keys())
        details["total_score"] = total_score
        details["weights"] = weights
        
        return total_score, details
    
    def generate_report(self, output_file: str = "qa_evaluation_report.json"):
        """生成评估报告"""
        if not self.evaluation_cases:
            logger.warning("没有评估用例，无法生成报告")
            return
        
        # 计算统计信息
        scores = [case.evaluation_score for case in self.evaluation_cases]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # 按分类统计
        category_stats = {}
        for case in self.evaluation_cases:
            category = case.qa_pair.category
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(case.evaluation_score)
        
        category_averages = {
            cat: sum(scores) / len(scores) 
            for cat, scores in category_stats.items()
        }
        
        # 按难度统计
        difficulty_stats = {}
        for case in self.evaluation_cases:
            difficulty = case.qa_pair.difficulty
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = []
            difficulty_stats[difficulty].append(case.evaluation_score)
        
        difficulty_averages = {
            diff: sum(scores) / len(scores) 
            for diff, scores in difficulty_stats.items()
        }
        
        # 生成报告数据
        report_data = {
            "evaluation_summary": {
                "total_cases": len(self.evaluation_cases),
                "average_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "checkpoint_path": self.checkpoint_path,
                "evaluation_time": datetime.now().isoformat()
            },
            "category_performance": category_averages,
            "difficulty_performance": difficulty_averages,
            "detailed_results": []
        }
        
        # 添加详细结果
        for case in self.evaluation_cases:
            case_data = {
                "id": case.id,
                "question": case.qa_pair.question[:100] + "..." if len(case.qa_pair.question) > 100 else case.qa_pair.question,
                "expected_answer": case.expected_answer[:100] + "..." if len(case.expected_answer) > 100 else case.expected_answer,
                "model_answer": case.model_answer[:100] + "..." if len(case.model_answer) > 100 else case.model_answer,
                "category": case.qa_pair.category,
                "difficulty": case.qa_pair.difficulty,
                "score": case.evaluation_score,
                "evaluation_details": case.evaluation_details
            }
            report_data["detailed_results"].append(case_data)
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估报告已保存到 {output_file}")
        
        # 打印摘要
        self._print_summary(report_data)
    
    def _print_summary(self, report_data: Dict[str, Any]):
        """打印评估摘要"""
        summary = report_data["evaluation_summary"]
        category_perf = report_data["category_performance"]
        difficulty_perf = report_data["difficulty_performance"]
        
        print("\n" + "="*60)
        print("QA评估报告摘要")
        print("="*60)
        print(f"Checkpoint: {summary['checkpoint_path']}")
        print(f"评估用例数: {summary['total_cases']}")
        print(f"平均分数: {summary['average_score']:.3f}")
        print(f"最高分数: {summary['max_score']:.3f}")
        print(f"最低分数: {summary['min_score']:.3f}")
        
        print("\n分类表现:")
        for category, score in category_perf.items():
            print(f"  {category}: {score:.3f}")
        
        print("\n难度表现:")
        for difficulty, score in difficulty_perf.items():
            print(f"  {difficulty}: {score:.3f}")
        
        print("="*60)

def find_available_checkpoint():
    """查找可用的checkpoint路径"""
    possible_paths = [
        "exported_models/qwen3_merged_lightweight/Qwen_Qwen3-4B-Thinking-2507_pytorch_20250823_143214",
        
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"找到可用的checkpoint: {path}")
            return path
    
    logger.error("未找到任何可用的checkpoint路径")
    return None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QA数据提取和Checkpoint评估")
    parser.add_argument("--data-dir", default="data/raw", help="QA数据目录")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint路径")
    parser.add_argument("--output-qa", default="extracted_qa_pairs.json", help="提取的QA对输出文件")
    parser.add_argument("--output-report", default="qa_evaluation_report.json", help="评估报告输出文件")
    parser.add_argument("--max-samples", type=int, default=50, help="最大评估样本数")
    parser.add_argument("--categories", nargs="*", help="指定评估的分类")
    parser.add_argument("--difficulties", nargs="*", choices=["easy", "medium", "hard"], help="指定评估的难度")
    parser.add_argument("--sample-strategy", choices=["random", "balanced", "sequential"], default="balanced", help="采样策略")
    parser.add_argument("--extract-only", action="store_true", help="仅提取QA对，不进行评估")
    
    args = parser.parse_args()
    
    # 如果没有指定checkpoint，自动查找
    if not args.checkpoint:
        args.checkpoint = find_available_checkpoint()
        if not args.checkpoint:
            logger.error("无法找到可用的checkpoint，请使用--checkpoint参数指定路径")
            return
    
    # 1. 提取QA对
    logger.info("开始提取QA对...")
    extractor = QAExtractor(args.data_dir)
    qa_pairs = extractor.extract_all_qa_pairs()
    
    # 筛选
    if args.categories:
        qa_pairs = extractor.filter_by_category(args.categories)
    if args.difficulties:
        qa_pairs = extractor.filter_by_difficulty(args.difficulties)
    
    # 采样
    if args.max_samples > 0 and len(qa_pairs) > args.max_samples:
        qa_pairs = extractor.sample_qa_pairs(args.max_samples, args.sample_strategy)
    
    # 保存QA对
    extractor.qa_pairs = qa_pairs
    extractor.save_qa_pairs(args.output_qa)
    
    if args.extract_only:
        logger.info("仅提取模式，程序结束")
        return
    
    # 2. 评估Checkpoint
    logger.info("开始评估Checkpoint...")
    evaluator = CheckpointEvaluator(args.checkpoint)
    evaluator.load_model()
    
    evaluation_cases = evaluator.evaluate_qa_pairs(qa_pairs)
    evaluator.generate_report(args.output_report)

if __name__ == "__main__":
    main()