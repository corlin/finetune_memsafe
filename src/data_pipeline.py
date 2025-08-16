"""
数据处理管道模块

提供QA数据加载、格式化和分词功能，支持从markdown文件解析QA数据，
并将其转换为适合Qwen3训练的格式。
"""

import os
import re
import json
import logging
import torch
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class QAData:
    """QA数据结构"""
    question: str
    answer: str
    source: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedDataCollatorForLanguageModeling:
    """增强的数据整理器，具有强大的张量创建和错误处理功能"""
    
    def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
        # 确保tokenizer有pad_token
        self._setup_pad_token()
    
    def _setup_pad_token(self):
        """确保分词器有pad_token"""
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"设置 pad_token 为 eos_token: {self.tokenizer.eos_token}")
            else:
                # 添加新的pad token
                self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                logger.info("添加新的 pad_token: <|pad|>")
    
    def __call__(self, features):
        """处理批次，具有增强的错误处理"""
        try:
            # 调试信息
            logger.debug(f"处理包含 {len(features)} 个样本的批次")
            
            # 验证输入特征
            self._validate_features(features)
            
            # 提取和验证序列
            input_ids = [f["input_ids"] for f in features]
            labels = [f["labels"] for f in features]
            
            # 记录形状信息用于调试
            self._log_shape_info(input_ids, labels)
            
            # 将序列填充到相同长度
            padded_inputs = self._pad_sequences(input_ids, self.tokenizer.pad_token_id)
            padded_labels = self._pad_sequences(labels, -100)  # -100是忽略索引
            
            # 创建张量
            batch = {
                "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long)
            }
            
            # 添加注意力掩码
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
            
            # 最终验证
            self._validate_batch_tensors(batch)
            
            logger.debug(f"成功创建批次，形状: {self._get_batch_shapes(batch)}")
            return batch
            
        except Exception as e:
            logger.error(f"数据整理器错误: {e}")
            self._provide_debugging_info(features, e)
            raise RuntimeError(f"创建批次失败: {e}")
    
    def _validate_features(self, features):
        """验证输入特征"""
        if not features:
            raise ValueError("特征列表为空")
        
        required_keys = ["input_ids", "labels"]
        for i, feature in enumerate(features):
            for key in required_keys:
                if key not in feature:
                    raise ValueError(f"特征 {i} 缺少必需的键: {key}")
                
                if not isinstance(feature[key], list):
                    raise ValueError(f"特征 {i}[{key}] 必须是列表，得到 {type(feature[key])}")
                
                if not feature[key]:
                    raise ValueError(f"特征 {i}[{key}] 为空")
    
    def _pad_sequences(self, sequences, pad_value):
        """将序列填充到相同长度"""
        if not sequences:
            return []
        
        max_length = max(len(seq) for seq in sequences)
        
        # 如果指定了pad_to_multiple_of，应用它
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded = []
        for seq in sequences:
            padded_seq = seq + [pad_value] * (max_length - len(seq))
            padded.append(padded_seq)
        
        return padded
    
    def _validate_batch_tensors(self, batch):
        """验证最终批次张量"""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        # 检查形状匹配
        if input_ids.shape != labels.shape:
            raise ValueError(f"形状不匹配: input_ids {input_ids.shape} != labels {labels.shape}")
        
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"形状不匹配: input_ids {input_ids.shape} != attention_mask {attention_mask.shape}")
        
        # 检查数据类型
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids 必须是 torch.long，得到 {input_ids.dtype}")
        
        if labels.dtype != torch.long:
            raise ValueError(f"labels 必须是 torch.long，得到 {labels.dtype}")
    
    def _log_shape_info(self, input_ids, labels):
        """记录形状信息用于调试"""
        if logger.isEnabledFor(logging.DEBUG):
            input_lengths = [len(seq) for seq in input_ids]
            label_lengths = [len(seq) for seq in labels]
            
            logger.debug(f"输入长度: 最小={min(input_lengths)}, 最大={max(input_lengths)}, 平均={sum(input_lengths)/len(input_lengths):.1f}")
            logger.debug(f"标签长度: 最小={min(label_lengths)}, 最大={max(label_lengths)}, 平均={sum(label_lengths)/len(label_lengths):.1f}")
    
    def _get_batch_shapes(self, batch):
        """获取批次张量形状用于记录"""
        return {key: tensor.shape for key, tensor in batch.items()}
    
    def _provide_debugging_info(self, features, error):
        """提供详细的调试信息"""
        logger.error("=== 数据整理器调试信息 ===")
        logger.error(f"错误: {error}")
        logger.error(f"特征数量: {len(features) if features else 0}")
        
        if features:
            # 样本特征分析
            sample_feature = features[0]
            logger.error(f"样本特征键: {list(sample_feature.keys())}")
            
            if "input_ids" in sample_feature:
                input_ids = sample_feature["input_ids"]
                logger.error(f"样本 input_ids 类型: {type(input_ids)}")
                logger.error(f"样本 input_ids 长度: {len(input_ids) if hasattr(input_ids, '__len__') else 'N/A'}")
                logger.error(f"样本 input_ids 前几个: {input_ids[:5] if isinstance(input_ids, list) else '不是列表'}")
            
            if "labels" in sample_feature:
                labels = sample_feature["labels"]
                logger.error(f"样本 labels 类型: {type(labels)}")
                logger.error(f"样本 labels 长度: {len(labels) if hasattr(labels, '__len__') else 'N/A'}")
                logger.error(f"样本 labels 前几个: {labels[:5] if isinstance(labels, list) else '不是列表'}")
        
        logger.error("=== 调试信息结束 ===")


class TensorCreationErrorHandler:
    """处理张量创建错误的自动修复功能"""
    
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def fix_tokenized_data(self, tokenized_data):
        """修复分词数据中的常见问题"""
        fixed_data = {}
        
        for key, values in tokenized_data.items():
            if key in ["input_ids", "labels"]:
                fixed_values = []
                for value in values:
                    fixed_value = self._fix_sequence(value)
                    fixed_values.append(fixed_value)
                fixed_data[key] = fixed_values
            else:
                fixed_data[key] = values
        
        return fixed_data
    
    def _fix_sequence(self, sequence):
        """修复单个序列的问题"""
        # 处理嵌套列表
        if isinstance(sequence, list) and len(sequence) > 0:
            if isinstance(sequence[0], list):
                # 展平嵌套列表
                sequence = [item for sublist in sequence for item in sublist]
        
        # 确保所有元素都是整数
        sequence = [int(x) if isinstance(x, (int, float)) else x for x in sequence]
        
        # 移除非整数元素
        sequence = [x for x in sequence if isinstance(x, int)]
        
        # 如果太长则截断
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # 确保最小长度
        if len(sequence) < 1:
            sequence = [self.tokenizer.eos_token_id or 0]
        
        return sequence
    
    def fix_features_batch(self, features):
        """修复特征批次中的问题"""
        fixed_features = []
        
        for i, feature in enumerate(features):
            try:
                fixed_feature = {}
                
                # 修复input_ids
                if "input_ids" in feature:
                    fixed_feature["input_ids"] = self._fix_sequence(feature["input_ids"])
                
                # 修复labels
                if "labels" in feature:
                    fixed_feature["labels"] = self._fix_sequence(feature["labels"])
                
                # 复制其他字段
                for key, value in feature.items():
                    if key not in ["input_ids", "labels"]:
                        fixed_feature[key] = value
                
                # 验证修复后的特征
                self._validate_fixed_feature(fixed_feature, i)
                
                fixed_features.append(fixed_feature)
                
            except Exception as e:
                logger.warning(f"无法修复特征 {i}: {e}，跳过此样本")
                continue
        
        if not fixed_features:
            raise RuntimeError("所有特征都无法修复")
        
        logger.info(f"成功修复 {len(fixed_features)}/{len(features)} 个特征")
        return fixed_features
    
    def _validate_fixed_feature(self, feature, index):
        """验证修复后的特征"""
        required_keys = ["input_ids", "labels"]
        
        for key in required_keys:
            if key not in feature:
                raise ValueError(f"修复后的特征 {index} 缺少键: {key}")
            
            if not isinstance(feature[key], list):
                raise ValueError(f"修复后的特征 {index}[{key}] 不是列表")
            
            if not feature[key]:
                raise ValueError(f"修复后的特征 {index}[{key}] 为空")
            
            if not all(isinstance(x, int) for x in feature[key]):
                raise ValueError(f"修复后的特征 {index}[{key}] 包含非整数值")
        
        # 检查长度匹配
        if len(feature["input_ids"]) != len(feature["labels"]):
            raise ValueError(f"修复后的特征 {index}: input_ids 和 labels 长度不匹配")


class DataPipeline:
    """
    数据处理管道类
    
    负责从多种格式的markdown文件解析QA数据，提供数据验证和错误处理，
    并将数据格式化为适合Qwen3训练的对话格式。
    """
    
    def __init__(self, data_dir: str = "data/raw", max_sequence_length: int = 256):
        """
        初始化数据管道
        
        Args:
            data_dir: 数据文件目录
            max_sequence_length: 最大序列长度
        """
        self.data_dir = Path(data_dir)
        self.max_sequence_length = max_sequence_length
        self.qa_data: List[QAData] = []
        
    def load_qa_data_from_files(self) -> List[QAData]:
        """
        从多种格式的markdown文件解析QA数据
        
        Returns:
            解析的QA数据列表
        """
        logger.info(f"开始从 {self.data_dir} 加载QA数据")
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            logger.warning(f"数据目录 {self.data_dir} 不存在，使用示例数据")
            return self._get_example_data()
        
        # 查找所有markdown文件
        md_files = list(self.data_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"在 {self.data_dir} 中未找到markdown文件，使用示例数据")
            return self._get_example_data()
        
        all_qa_data = []
        
        for md_file in md_files:
            if md_file.name.lower() == "readme.md":
                continue  # 跳过README文件
                
            try:
                qa_data = self._parse_markdown_file(md_file)
                if qa_data:
                    all_qa_data.extend(qa_data)
                    logger.info(f"从 {md_file.name} 加载了 {len(qa_data)} 个QA对")
                else:
                    logger.warning(f"文件 {md_file.name} 中未找到有效的QA数据")
            except Exception as e:
                logger.error(f"解析文件 {md_file.name} 时出错: {e}")
                continue
        
        if not all_qa_data:
            logger.warning("未能从任何文件中加载有效数据，使用示例数据")
            return self._get_example_data()
        
        # 数据验证和清理
        validated_data = self._validate_and_clean_data(all_qa_data)
        
        logger.info(f"总共加载了 {len(validated_data)} 个有效的QA对")
        self.qa_data = validated_data
        return validated_data
    
    def _parse_markdown_file(self, file_path: Path) -> List[QAData]:
        """
        解析单个markdown文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析的QA数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"无法读取文件 {file_path}: {e}")
                return []
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {e}")
            return []
        
        # 检测文件格式并解析 (格式2优先，因为它更具体)
        if self._is_format_2(content):
            return self._parse_format_2(content, file_path.name)
        elif self._is_format_1(content):
            return self._parse_format_1(content, file_path.name)
        else:
            logger.warning(f"无法识别文件 {file_path.name} 的格式")
            return []
    
    def _is_format_1(self, content: str) -> bool:
        """检测是否为格式1 (Q1: ... A1: ...)"""
        return bool(re.search(r'Q\d+:', content) and re.search(r'A\d+:', content))
    
    def _is_format_2(self, content: str) -> bool:
        """检测是否为格式2 (### Q1: ... A1: ...)"""
        return bool(re.search(r'###\s*Q\d+:', content))
    
    def _parse_format_1(self, content: str, source: str) -> List[QAData]:
        """
        解析格式1: Q1: question A1: answer
        """
        qa_pairs = []
        
        # 使用正则表达式匹配Q和A对
        pattern = r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=Q\d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            q_num, question, answer = match
            
            # 清理问题和答案
            question = self._clean_text(question)
            answer = self._clean_text(answer)
            
            if question and answer:
                qa_pairs.append(QAData(
                    question=question,
                    answer=answer,
                    source=source,
                    metadata={"question_number": q_num}
                ))
        
        return qa_pairs
    
    def _parse_format_2(self, content: str, source: str) -> List[QAData]:
        """
        解析格式2: ### Q1: question A1: answer
        """
        qa_pairs = []
        
        # 使用正则表达式匹配Q和A对
        pattern = r'###\s*Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=###\s*Q\d+:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            q_num, question, answer = match
            
            # 清理问题和答案
            question = self._clean_text(question)
            answer = self._clean_text(answer)
            
            if question and answer:
                qa_pairs.append(QAData(
                    question=question,
                    answer=answer,
                    source=source,
                    metadata={"question_number": q_num}
                ))
        
        return qa_pairs
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除markdown标记
        text = re.sub(r'[*_`]', '', text)
        
        # 移除多余的标点符号
        text = re.sub(r'[：:]\s*$', '', text)
        
        return text.strip()
    
    def _validate_and_clean_data(self, qa_data: List[QAData]) -> List[QAData]:
        """
        验证和清理QA数据
        
        Args:
            qa_data: 原始QA数据列表
            
        Returns:
            验证后的QA数据列表
        """
        validated_data = []
        
        for qa in qa_data:
            # 检查问题和答案是否为空
            if not qa.question or not qa.answer:
                logger.debug(f"跳过空的QA对: {qa.question[:50] if qa.question else 'None'}...")
                continue
            
            # 检查长度是否合理 (降低最小长度要求)
            if len(qa.question) < 2 or len(qa.answer) < 2:
                logger.debug(f"跳过过短的QA对: {qa.question[:50]}...")
                continue
            
            # 检查是否包含有效内容
            if qa.question.strip() == qa.answer.strip():
                logger.debug(f"跳过问答相同的QA对: {qa.question[:50]}...")
                continue
            
            validated_data.append(qa)
        
        return validated_data
    
    def _get_example_data(self) -> List[QAData]:
        """
        获取示例QA数据作为回退
        
        Returns:
            示例QA数据列表
        """
        example_data = [
            QAData(
                question="什么是信息系统密码应用的基本要求？",
                answer="信息系统密码应用基本要求是指从信息系统的物理和环境安全、网络和通信安全、设备和计算安全、应用和数据安全四个技术层面提出的密码应用技术要求，以及从管理制度、人员管理、建设运行和应急处置四个方面提出的密码应用管理要求。",
                source="example_data",
                metadata={"type": "example"}
            ),
            QAData(
                question="密码安全功能维度有哪些？",
                answer="密码安全功能维度包括：机密性、完整性、真实性、不可否认性四个维度。",
                source="example_data",
                metadata={"type": "example"}
            ),
            QAData(
                question="什么是密钥管理？",
                answer="密钥管理是指根据安全策略，对密钥的产生、分发、存储、使用、更新、归档、撤销、备份、恢复和销毁等密钥全生存周期的管理。",
                source="example_data",
                metadata={"type": "example"}
            )
        ]
        
        logger.info(f"使用 {len(example_data)} 个示例QA对")
        return example_data
    
    def format_for_qwen(self, qa_data: Optional[List[QAData]] = None) -> Dataset:
        """
        将QA数据格式化为Qwen3对话格式
        
        Args:
            qa_data: QA数据列表，如果为None则使用已加载的数据
            
        Returns:
            格式化后的Dataset
        """
        if qa_data is None:
            qa_data = self.qa_data
        
        if not qa_data:
            logger.warning("没有可用的QA数据进行格式化")
            return Dataset.from_dict({"text": []})
        
        formatted_data = []
        
        for qa in qa_data:
            # 使用Qwen3的对话格式
            conversation = self._format_qwen_conversation(qa.question, qa.answer)
            
            # 过滤过长的对话
            if len(conversation) > self.max_sequence_length * 4:  # 粗略估计token数量
                logger.debug(f"跳过过长的对话: {qa.question[:50]}...")
                continue
            
            formatted_data.append({
                "text": conversation,
                "question": qa.question,
                "answer": qa.answer,
                "source": qa.source,
                "metadata": qa.metadata
            })
        
        logger.info(f"格式化了 {len(formatted_data)} 个对话")
        return Dataset.from_dict({
            "text": [item["text"] for item in formatted_data],
            "question": [item["question"] for item in formatted_data],
            "answer": [item["answer"] for item in formatted_data],
            "source": [item["source"] for item in formatted_data],
            "metadata": [item["metadata"] for item in formatted_data]
        })
    
    def _format_qwen_conversation(self, question: str, answer: str) -> str:
        """
        格式化为Qwen3对话格式
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            格式化的对话字符串
        """
        # Qwen3的对话格式
        conversation = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        return conversation
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
        """
        对数据集进行分词，包含序列长度限制的内存优化
        
        Args:
            dataset: 要分词的数据集
            tokenizer: 分词器
            
        Returns:
            分词后的数据集
        """
        logger.info("开始对数据集进行分词")
        
        def tokenize_function(examples):
            """增强的分词函数，确保标签格式正确"""
            try:
                # 批量分词
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,  # 不在这里padding，在DataCollator中处理
                    max_length=self.max_sequence_length,
                    return_tensors=None,  # 返回列表而不是张量
                    add_special_tokens=True
                )
                
                # 创建标签作为扁平的整数列表（不是嵌套的）
                labels = []
                for input_ids in tokenized["input_ids"]:
                    # 确保标签是扁平的整数列表
                    if isinstance(input_ids, list):
                        labels.append(input_ids.copy())
                    else:
                        # 如果不是列表，转换为列表
                        labels.append(input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids))
                
                tokenized["labels"] = labels
                
                # 验证分词结果
                self._validate_tokenized_batch(tokenized)
                
                return tokenized
                
            except Exception as e:
                logger.error(f"分词过程中出错: {e}")
                # 提供调试信息
                if examples and "text" in examples:
                    logger.error(f"样本数量: {len(examples['text'])}")
                    logger.error(f"第一个样本: {examples['text'][0][:100] if examples['text'] else 'None'}...")
                raise RuntimeError(f"分词失败: {e}")
        
        # 应用分词函数
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,  # 移除原始文本列以节省内存
            desc="分词中"
        )
        
        # 过滤过短的序列
        def filter_short_sequences(example):
            return len(example["input_ids"]) >= 10  # 至少10个token
        
        filtered_dataset = tokenized_dataset.filter(
            filter_short_sequences,
            desc="过滤短序列"
        )
        
        logger.info(f"分词完成，保留 {len(filtered_dataset)} 个样本")
        return filtered_dataset
    
    def _validate_tokenized_batch(self, tokenized):
        """验证分词批次的一致性"""
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        
        # 检查所有序列都是整数列表
        for i, (inp, lab) in enumerate(zip(input_ids, labels)):
            if not isinstance(inp, list) or not isinstance(lab, list):
                raise ValueError(f"样本 {i}: input_ids 和 labels 必须是列表，得到 input_ids: {type(inp)}, labels: {type(lab)}")
            
            if len(inp) != len(lab):
                raise ValueError(f"样本 {i}: input_ids 长度 {len(inp)} != labels 长度 {len(lab)}")
            
            # 检查所有元素都是整数
            if not all(isinstance(x, int) for x in inp):
                non_int_types = [type(x) for x in inp if not isinstance(x, int)]
                raise ValueError(f"样本 {i}: input_ids 包含非整数值，类型: {set(non_int_types)}")
            
            if not all(isinstance(x, int) for x in lab):
                non_int_types = [type(x) for x in lab if not isinstance(x, int)]
                raise ValueError(f"样本 {i}: labels 包含非整数值，类型: {set(non_int_types)}")
        
        logger.debug(f"分词验证通过: {len(input_ids)} 个样本，平均长度: {sum(len(seq) for seq in input_ids) / len(input_ids):.1f}")
    
    def create_data_collator(self, tokenizer: PreTrainedTokenizer):
        """
        创建增强的数据整理器，具有强大的错误处理和张量创建功能
        
        Args:
            tokenizer: 分词器
            
        Returns:
            增强的数据整理器
        """
        # 创建增强的数据整理器
        data_collator = EnhancedDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 不使用掩码语言模型
            pad_to_multiple_of=8,  # 为了提高效率，填充到8的倍数
            return_tensors="pt",  # 明确指定返回PyTorch张量
        )
        
        logger.info("增强数据整理器创建完成，配置:")
        logger.info(f"  pad_token: {tokenizer.pad_token}")
        logger.info(f"  pad_token_id: {tokenizer.pad_token_id}")
        logger.info(f"  pad_to_multiple_of: 8")
        logger.info(f"  增强错误处理: 启用")
        
        return data_collator
    
    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Args:
            dataset: 数据集
            
        Returns:
            统计信息字典
        """
        if len(dataset) == 0:
            return {"total_samples": 0}
        
        stats = {
            "total_samples": len(dataset),
            "columns": dataset.column_names,
        }
        
        # 如果有input_ids列，计算序列长度统计
        if "input_ids" in dataset.column_names:
            lengths = [len(example["input_ids"]) for example in dataset]
            stats.update({
                "avg_sequence_length": sum(lengths) / len(lengths),
                "min_sequence_length": min(lengths),
                "max_sequence_length": max(lengths),
                "total_tokens": sum(lengths)
            })
        
        # 如果有source列，计算来源统计
        if "source" in dataset.column_names:
            sources = [example["source"] for example in dataset]
            source_counts = {}
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            stats["source_distribution"] = source_counts
        
        return stats
    
    def save_processed_data(self, dataset: Dataset, output_path: str) -> None:
        """
        保存处理后的数据集
        
        Args:
            dataset: 要保存的数据集
            output_path: 输出路径
        """
        try:
            dataset.save_to_disk(output_path)
            logger.info(f"数据集已保存到 {output_path}")
        except Exception as e:
            logger.error(f"保存数据集时出错: {e}")
            raise
    
    def load_processed_data(self, input_path: str) -> Dataset:
        """
        加载已处理的数据集
        
        Args:
            input_path: 输入路径
            
        Returns:
            加载的数据集
        """
        try:
            dataset = Dataset.load_from_disk(input_path)
            logger.info(f"从 {input_path} 加载了数据集，包含 {len(dataset)} 个样本")
            return dataset
        except Exception as e:
            logger.error(f"加载数据集时出错: {e}")
            raise


def create_sample_data_files():
    """
    创建示例数据文件用于测试
    """
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例QA文件
    sample_qa = """# 密码学基础QA

## 基础概念

### Q1: 什么是对称加密？

A1: 对称加密是指加密和解密使用相同密钥的加密方式。常见的对称加密算法包括AES、DES等。

### Q2: 什么是非对称加密？

A2: 非对称加密是指加密和解密使用不同密钥的加密方式，包括公钥和私钥。常见的非对称加密算法包括RSA、ECC等。

### Q3: 什么是数字签名？

A3: 数字签名是使用私钥对数据进行签名，使用公钥进行验证的技术，用于确保数据的完整性和真实性。
"""
    
    sample_file = data_dir / "sample_qa.md"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_qa)
    
    print(f"示例数据文件已创建: {sample_file}")


def create_safe_data_collator(tokenizer, max_length=256, pad_to_multiple_of=8):
    """
    创建具有自动错误恢复功能的安全数据整理器
    
    Args:
        tokenizer: 分词器
        max_length: 最大序列长度
        pad_to_multiple_of: 填充到的倍数
        
    Returns:
        安全的数据整理器函数
    """
    # 创建基础数据整理器
    base_collator = EnhancedDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt"
    )
    
    # 创建错误处理器
    error_handler = TensorCreationErrorHandler(tokenizer, max_length)
    
    def safe_collator(features):
        """安全的数据整理器，具有自动错误恢复"""
        try:
            # 首先尝试正常处理
            return base_collator(features)
        except Exception as e:
            logger.warning(f"数据整理器失败，尝试恢复: {e}")
            
            # 尝试修复特征
            try:
                fixed_features = error_handler.fix_features_batch(features)
                logger.info(f"特征修复成功，重新尝试数据整理")
                return base_collator(fixed_features)
            except Exception as recovery_error:
                logger.error(f"恢复失败: {recovery_error}")
                
                # 提供详细的错误信息
                logger.error("=== 数据整理器最终错误报告 ===")
                logger.error(f"原始错误: {e}")
                logger.error(f"恢复错误: {recovery_error}")
                logger.error(f"特征数量: {len(features) if features else 0}")
                
                if features:
                    logger.error("样本特征分析:")
                    for i, feature in enumerate(features[:3]):  # 只显示前3个
                        logger.error(f"  特征 {i}: {list(feature.keys()) if isinstance(feature, dict) else type(feature)}")
                        if isinstance(feature, dict) and "input_ids" in feature:
                            input_ids = feature["input_ids"]
                            logger.error(f"    input_ids 类型: {type(input_ids)}, 长度: {len(input_ids) if hasattr(input_ids, '__len__') else 'N/A'}")
                
                logger.error("=== 错误报告结束 ===")
                
                raise RuntimeError(f"数据整理器即使在恢复后也失败: {recovery_error}")
    
    return safe_collator


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据文件
    create_sample_data_files()
    
    # 测试数据管道
    pipeline = DataPipeline()
    
    # 加载数据
    qa_data = pipeline.load_qa_data_from_files()
    print(f"加载了 {len(qa_data)} 个QA对")
    
    # 格式化数据
    dataset = pipeline.format_for_qwen(qa_data)
    print(f"格式化了 {len(dataset)} 个对话")
    
    # 显示统计信息
    stats = pipeline.get_dataset_stats(dataset)
    print("数据集统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")