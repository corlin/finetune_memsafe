#!/usr/bin/env python3
"""
Demo program to test enhanced-qwen3-finetuned checkpoint-450 using QA data
and industry model evaluation capabilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAItem:
    question: str
    answer: str
    category: str

@dataclass
class EvaluationResult:
    question: str
    expected_answer: str
    model_answer: str
    category: str
    score: float
    evaluation_details: Dict[str, Any]

class QADataLoader:
    """Load and parse QA data from markdown files"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
    
    def load_qa_data(self) -> List[QAItem]:
        """Load all QA data from markdown files"""
        qa_items = []
        
        # Find all QA markdown files
        qa_files = list(self.data_dir.glob("QA*.md"))
        logger.info(f"Found {len(qa_files)} QA files")
        
        for qa_file in qa_files:
            items = self._parse_qa_file(qa_file)
            qa_items.extend(items)
            logger.info(f"Loaded {len(items)} QA items from {qa_file.name}")
        
        return qa_items
    
    def _parse_qa_file(self, file_path: Path) -> List[QAItem]:
        """Parse a single QA markdown file"""
        qa_items = []
        current_category = "未分类"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect category headers
            if line.startswith('##') and not line.startswith('###'):
                current_category = line.replace('##', '').strip()
                i += 1
                continue
            
            # Detect questions
            if line.startswith('### Q'):
                question = line.split(':', 1)[1].strip() if ':' in line else line
                i += 1
                
                # Find corresponding answer
                while i < len(lines) and not lines[i].strip().startswith('A'):
                    i += 1
                
                if i < len(lines):
                    answer = lines[i].strip()
                    if answer.startswith('A'):
                        answer = answer.split(':', 1)[1].strip() if ':' in answer else answer
                    
                    qa_items.append(QAItem(
                        question=question,
                        answer=answer,
                        category=current_category
                    ))
            
            i += 1
        
        return qa_items

class CheckpointModelLoader:
    """Load and manage the fine-tuned checkpoint model"""
    
    def __init__(self, checkpoint_path: str = "enhanced-qwen3-finetuned/checkpoint-450"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the checkpoint model and tokenizer"""
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_answer(self, question: str, max_length: int = 512) -> str:
        """Generate answer for a given question"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format the prompt
        prompt = f"问题：{question}\n答案："
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "答案：" in response:
            answer = response.split("答案：", 1)[1].strip()
        else:
            answer = response.replace(prompt, "").strip()
        
        return answer

class IndustryEvaluator:
    """Industry-specific evaluation using the evaluation framework"""
    
    def __init__(self):
        # Try to import industry evaluation modules
        try:
            from industry_evaluation.evaluators.knowledge_evaluator import KnowledgeEvaluator
            from industry_evaluation.evaluators.terminology_evaluator import TerminologyEvaluator
            from industry_evaluation.evaluators.reasoning_evaluator import ReasoningEvaluator
            
            self.knowledge_evaluator = KnowledgeEvaluator()
            self.terminology_evaluator = TerminologyEvaluator()
            self.reasoning_evaluator = ReasoningEvaluator()
            self.evaluators_available = True
            
        except ImportError as e:
            logger.warning(f"Industry evaluators not available: {e}")
            self.evaluators_available = False
    
    def evaluate_answer(self, question: str, expected_answer: str, 
                       model_answer: str, category: str) -> Dict[str, Any]:
        """Evaluate model answer using industry evaluation framework"""
        
        if not self.evaluators_available:
            # Fallback to simple evaluation
            return self._simple_evaluation(expected_answer, model_answer)
        
        evaluation_results = {}
        
        try:
            # Knowledge accuracy evaluation
            knowledge_score = self.knowledge_evaluator.evaluate(
                question, model_answer, expected_answer, {"category": category}
            )
            evaluation_results["knowledge_accuracy"] = knowledge_score.score
            
            # Terminology evaluation
            terminology_score = self.terminology_evaluator.evaluate(
                question, model_answer, expected_answer, {"category": category}
            )
            evaluation_results["terminology_accuracy"] = terminology_score.score
            
            # Reasoning evaluation
            reasoning_score = self.reasoning_evaluator.evaluate(
                question, model_answer, expected_answer, {"category": category}
            )
            evaluation_results["reasoning_quality"] = reasoning_score.score
            
            # Calculate overall score
            overall_score = (
                evaluation_results["knowledge_accuracy"] * 0.4 +
                evaluation_results["terminology_accuracy"] * 0.3 +
                evaluation_results["reasoning_quality"] * 0.3
            )
            evaluation_results["overall_score"] = overall_score
            
        except Exception as e:
            logger.warning(f"Industry evaluation failed: {e}")
            evaluation_results = self._simple_evaluation(expected_answer, model_answer)
        
        return evaluation_results
    
    def _simple_evaluation(self, expected_answer: str, model_answer: str) -> Dict[str, Any]:
        """Simple fallback evaluation based on text similarity"""
        # Simple keyword matching score
        expected_keywords = set(expected_answer.split())
        model_keywords = set(model_answer.split())
        
        if len(expected_keywords) == 0:
            similarity = 0.0
        else:
            intersection = expected_keywords.intersection(model_keywords)
            similarity = len(intersection) / len(expected_keywords)
        
        return {
            "overall_score": similarity,
            "keyword_similarity": similarity,
            "evaluation_method": "simple_fallback"
        }

class DemoEvaluationRunner:
    """Main demo runner that orchestrates the evaluation process"""
    
    def __init__(self, checkpoint_path: str = "enhanced-qwen3-finetuned/checkpoint-450"):
        self.qa_loader = QADataLoader()
        self.model_loader = CheckpointModelLoader(checkpoint_path)
        self.evaluator = IndustryEvaluator()
        self.results = []
    
    def run_demo(self, max_samples: int = 20) -> List[EvaluationResult]:
        """Run the complete demo evaluation"""
        logger.info("Starting demo evaluation...")
        
        # Load QA data
        qa_items = self.qa_loader.load_qa_data()
        logger.info(f"Loaded {len(qa_items)} QA items")
        
        # Limit samples for demo
        if max_samples > 0:
            qa_items = qa_items[:max_samples]
            logger.info(f"Limited to {len(qa_items)} samples for demo")
        
        # Load model
        self.model_loader.load_model()
        
        # Process each QA item
        for i, qa_item in enumerate(qa_items):
            logger.info(f"Processing item {i+1}/{len(qa_items)}: {qa_item.category}")
            
            try:
                # Generate model answer
                model_answer = self.model_loader.generate_answer(qa_item.question)
                
                # Evaluate answer
                evaluation_details = self.evaluator.evaluate_answer(
                    qa_item.question,
                    qa_item.answer,
                    model_answer,
                    qa_item.category
                )
                
                # Store result
                result = EvaluationResult(
                    question=qa_item.question,
                    expected_answer=qa_item.answer,
                    model_answer=model_answer,
                    category=qa_item.category,
                    score=evaluation_details.get("overall_score", 0.0),
                    evaluation_details=evaluation_details
                )
                
                self.results.append(result)
                
                logger.info(f"Score: {result.score:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing item {i+1}: {e}")
                continue
        
        return self.results
    
    def generate_report(self, output_file: str = "demo_evaluation_report.json"):
        """Generate evaluation report"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Calculate statistics
        scores = [r.score for r in self.results]
        avg_score = sum(scores) / len(scores)
        
        # Group by category
        category_stats = {}
        for result in self.results:
            cat = result.category
            if cat not in category_stats:
                category_stats[cat] = []
            category_stats[cat].append(result.score)
        
        # Calculate category averages
        category_averages = {
            cat: sum(scores) / len(scores) 
            for cat, scores in category_stats.items()
        }
        
        # Prepare report data
        report_data = {
            "summary": {
                "total_samples": len(self.results),
                "average_score": avg_score,
                "max_score": max(scores),
                "min_score": min(scores),
                "category_averages": category_averages
            },
            "detailed_results": [
                {
                    "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
                    "expected_answer": r.expected_answer[:100] + "..." if len(r.expected_answer) > 100 else r.expected_answer,
                    "model_answer": r.model_answer[:100] + "..." if len(r.model_answer) > 100 else r.model_answer,
                    "category": r.category,
                    "score": r.score,
                    "evaluation_details": r.evaluation_details
                }
                for r in self.results
            ]
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("DEMO EVALUATION SUMMARY")
        print("="*50)
        print(f"Total samples: {len(self.results)}")
        print(f"Average score: {avg_score:.3f}")
        print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
        print("\nCategory Performance:")
        for cat, avg in category_averages.items():
            print(f"  {cat}: {avg:.3f}")
        print("="*50)

def main():
    """Main function to run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo evaluation of enhanced-qwen3-finetuned checkpoint-450")
    parser.add_argument("--checkpoint", default="enhanced-qwen3-finetuned/checkpoint-450", 
                       help="Path to checkpoint directory")
    parser.add_argument("--max-samples", type=int, default=20, 
                       help="Maximum number of samples to evaluate (0 for all)")
    parser.add_argument("--output", default="demo_evaluation_report.json", 
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Run demo
    runner = DemoEvaluationRunner(args.checkpoint)
    results = runner.run_demo(args.max_samples)
    runner.generate_report(args.output)

if __name__ == "__main__":
    main()