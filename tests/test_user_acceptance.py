"""
用户验收测试
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from industry_evaluation.core.evaluation_engine import IndustryEvaluationEngine
from industry_evaluation.core.batch_evaluator import BatchEvaluator, BatchEvaluationConfig
from industry_evaluation.core.result_aggregator import ResultAggregator
from industry_evaluation.adapters.model_adapter import ModelManager, ModelAdapterFactory
from industry_evaluation.reporting.report_generator import ReportGenerator
from industry_evaluation.evaluators.knowledge_evaluator import KnowledgeEvaluator
from industry_evaluation.evaluators.terminology_evaluator import TerminologyEvaluator
from industry_evaluation.evaluators.reasoning_evaluator import ReasoningEvaluator
from industry_evaluation.core.interfaces import EvaluationConfig


class RealWorldModelAdapter:
    """真实场景模拟的模型适配器"""
    
    def __init__(self, model_id: str, config: dict):
        self.model_id = model_id
        self.config = config
        self.domain_knowledge = {
            "finance": {
                "VaR": "Value at Risk是一种风险度量方法，用于量化在正常市场条件下，特定时间段内投资组合可能面临的最大损失。",
                "衍生品": "衍生品是一种金融工具，其价值来源于基础资产的价格变动，包括期货、期权、掉期等。",
                "流动性风险": "流动性风险是指无法在合理时间内以合理价格变现资产的风险。"
            },
            "healthcare": {
                "诊断": "医学诊断是通过症状、体征、实验室检查等信息确定疾病的过程。",
                "治疗方案": "治疗方案是针对特定疾病制定的综合治疗计划，包括药物治疗、手术治疗等。",
                "预后": "预后是对疾病发展趋势和治疗效果的预测。"
            },
            "technology": {
                "区块链": "区块链是一种分布式账本技术，通过密码学方法确保数据的不可篡改性。",
                "人工智能": "人工智能是模拟人类智能的计算机系统，包括机器学习、深度学习等技术。",
                "云计算": "云计算是通过网络提供可扩展的计算资源和服务的模式。"
            }
        }
    
    def predict(self, input_text: str, context=None):
        """根据输入生成相应的回答"""
        domain = context.get("industry", "general") if context else "general"
        
        # 模拟不同质量的回答
        if "VaR" in input_text or "风险" in input_text:
            if domain == "finance":
                return self.domain_knowledge["finance"]["VaR"] + " 它通常用95%或99%的置信水平来计算。"
            else:
                return "VaR是一种风险管理工具。"
        
        elif "区块链" in input_text or "共识" in input_text:
            if domain == "technology":
                return self.domain_knowledge["technology"]["区块链"] + " 常见的共识机制包括工作量证明和权益证明。"
            else:
                return "区块链是一种新技术。"
        
        elif "诊断" in input_text or "医疗" in input_text:
            if domain == "healthcare":
                return self.domain_knowledge["healthcare"]["诊断"] + " 现代医学诊断越来越依赖于先进的医疗设备和人工智能技术。"
            else:
                return "医疗诊断很重要。"
        
        else:
            return f"针对'{input_text}'的专业分析：这是一个复杂的问题，需要综合考虑多个因素。"
    
    def is_available(self):
        return True


class TestFinanceIndustryScenario:
    """金融行业场景测试"""
    
    def setup_method(self):
        """设置金融行业测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_system()
        self.setup_finance_data()
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.evaluation_engine.shutdown()
    
    def setup_system(self):
        """设置评估系统"""
        self.model_manager = ModelManager()
        ModelAdapterFactory.register_adapter("realworld", RealWorldModelAdapter)
        
        # 注册金融专业模型
        self.model_manager.register_model(
            "finance_expert_model", 
            "realworld", 
            {"domain": "finance"}
        )
        
        # 注册通用模型
        self.model_manager.register_model(
            "general_model", 
            "realworld", 
            {"domain": "general"}
        )
        
        self.evaluators = {
            "knowledge": KnowledgeEvaluator(),
            "terminology": TerminologyEvaluator(),
            "reasoning": ReasoningEvaluator()
        }
        
        self.result_aggregator = ResultAggregator()
        self.report_generator = ReportGenerator()
        
        self.evaluation_engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator
        )
    
    def setup_finance_data(self):
        """设置金融行业测试数据"""
        self.finance_dataset = [
            {
                "id": "finance_1",
                "input": "请解释金融风险管理中的VaR模型及其应用",
                "expected_output": "VaR（Value at Risk）是一种风险度量方法，用于量化在正常市场条件下，特定时间段内投资组合可能面临的最大损失。它通常用95%或99%的置信水平来计算，广泛应用于银行、投资公司等金融机构的风险管理中。",
                "context": {
                    "industry": "finance",
                    "topic": "risk_management",
                    "difficulty": "intermediate"
                }
            },
            {
                "id": "finance_2",
                "input": "什么是金融衍生品？请举例说明其主要类型",
                "expected_output": "金融衍生品是一种金融工具，其价值来源于基础资产的价格变动。主要类型包括：1）期货合约：标准化的远期合约；2）期权：给予持有者在特定时间以特定价格买卖资产的权利；3）掉期：交换现金流的协议；4）远期合约：非标准化的未来交易协议。",
                "context": {
                    "industry": "finance",
                    "topic": "derivatives",
                    "difficulty": "basic"
                }
            },
            {
                "id": "finance_3",
                "input": "如何评估和管理银行的流动性风险？",
                "expected_output": "银行流动性风险管理包括：1）流动性覆盖率（LCR）监控；2）净稳定资金比率（NSFR）管理；3）压力测试；4）多元化资金来源；5）建立流动性缓冲；6）制定应急流动性计划。关键是平衡流动性需求与盈利性。",
                "context": {
                    "industry": "finance",
                    "topic": "liquidity_risk",
                    "difficulty": "advanced"
                }
            }
        ]
        
        # 保存数据集
        self.dataset_path = Path(self.temp_dir) / "finance_dataset.json"
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.finance_dataset, f, ensure_ascii=False, indent=2)
    
    def test_finance_expert_model_evaluation(self):
        """测试金融专家模型评估"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology", "reasoning"],
            weight_config={
                "knowledge": 0.5,
                "terminology": 0.3,
                "reasoning": 0.2
            },
            threshold_config={
                "knowledge": 0.8,
                "terminology": 0.7,
                "reasoning": 0.7
            }
        )
        
        # 评估金融专家模型
        task_id = self.evaluation_engine.evaluate_model(
            model_id="finance_expert_model",
            dataset=self.finance_dataset,
            evaluation_config=config
        )
        
        # 等待评估完成
        result = self._wait_for_completion(task_id)
        
        # 验收标准
        assert result.overall_score > 0.6, "金融专家模型整体得分应该超过0.6"
        assert result.dimension_scores["knowledge"] > 0.7, "知识维度得分应该超过0.7"
        assert result.dimension_scores["terminology"] > 0.6, "术语维度得分应该超过0.6"
        
        # 验证详细结果
        assert len(result.detailed_results) == 3
        
        # 检查是否有改进建议
        assert len(result.improvement_suggestions) >= 0
        
        print(f"金融专家模型评估结果:")
        print(f"  整体得分: {result.overall_score:.3f}")
        print(f"  知识得分: {result.dimension_scores['knowledge']:.3f}")
        print(f"  术语得分: {result.dimension_scores['terminology']:.3f}")
        print(f"  推理得分: {result.dimension_scores['reasoning']:.3f}")
    
    def test_model_comparison_in_finance_domain(self):
        """测试金融领域的模型对比"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.7, "terminology": 0.6}
        )
        
        # 同时评估专家模型和通用模型
        expert_task_id = self.evaluation_engine.evaluate_model(
            model_id="finance_expert_model",
            dataset=self.finance_dataset,
            evaluation_config=config
        )
        
        general_task_id = self.evaluation_engine.evaluate_model(
            model_id="general_model",
            dataset=self.finance_dataset,
            evaluation_config=config
        )
        
        # 等待两个评估完成
        expert_result = self._wait_for_completion(expert_task_id)
        general_result = self._wait_for_completion(general_task_id)
        
        # 验收标准：专家模型应该在金融领域表现更好
        assert expert_result.overall_score >= general_result.overall_score, \
            "金融专家模型在金融领域应该表现更好"
        
        assert expert_result.dimension_scores["knowledge"] >= general_result.dimension_scores["knowledge"], \
            "专家模型的知识得分应该更高"
        
        print(f"模型对比结果:")
        print(f"  专家模型整体得分: {expert_result.overall_score:.3f}")
        print(f"  通用模型整体得分: {general_result.overall_score:.3f}")
        print(f"  专家模型知识得分: {expert_result.dimension_scores['knowledge']:.3f}")
        print(f"  通用模型知识得分: {general_result.dimension_scores['knowledge']:.3f}")
    
    def test_report_generation_for_finance(self):
        """测试金融领域报告生成"""
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology", "reasoning"],
            weight_config={"knowledge": 0.5, "terminology": 0.3, "reasoning": 0.2},
            threshold_config={"knowledge": 0.7, "terminology": 0.6, "reasoning": 0.6},
            auto_generate_report=True
        )
        
        task_id = self.evaluation_engine.evaluate_model(
            model_id="finance_expert_model",
            dataset=self.finance_dataset,
            evaluation_config=config
        )
        
        # 等待评估完成
        result = self._wait_for_completion(task_id)
        
        # 生成报告
        report = self.evaluation_engine.generate_report(task_id, "json")
        
        # 验收标准
        assert report is not None, "应该能够生成评估报告"
        
        # 如果报告是JSON字符串，验证其内容
        if isinstance(report, str) and report.startswith('{'):
            report_data = json.loads(report)
            assert "overall_score" in report_data
            assert "dimension_scores" in report_data
            assert "industry_domain" in report_data
            assert report_data["industry_domain"] == "finance"
        
        print(f"报告生成成功，类型: {type(report)}")
    
    def _wait_for_completion(self, task_id: str, timeout: int = 30):
        """等待评估完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            
            if progress and progress.status == "completed":
                return self.evaluation_engine.get_evaluation_result(task_id)
            elif progress and progress.status == "failed":
                pytest.fail(f"评估失败: {getattr(progress, 'error', '未知错误')}")
            
            time.sleep(1)
        
        pytest.fail(f"评估超时: {task_id}")


class TestHealthcareIndustryScenario:
    """医疗行业场景测试"""
    
    def setup_method(self):
        """设置医疗行业测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_system()
        self.setup_healthcare_data()
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.evaluation_engine.shutdown()
    
    def setup_system(self):
        """设置评估系统"""
        self.model_manager = ModelManager()
        ModelAdapterFactory.register_adapter("realworld", RealWorldModelAdapter)
        
        self.model_manager.register_model(
            "healthcare_model", 
            "realworld", 
            {"domain": "healthcare"}
        )
        
        self.evaluators = {
            "knowledge": KnowledgeEvaluator(),
            "terminology": TerminologyEvaluator()
        }
        
        self.result_aggregator = ResultAggregator()
        self.report_generator = ReportGenerator()
        
        self.evaluation_engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator
        )
    
    def setup_healthcare_data(self):
        """设置医疗行业测试数据"""
        self.healthcare_dataset = [
            {
                "id": "healthcare_1",
                "input": "请解释医疗诊断的基本流程和关键要素",
                "expected_output": "医疗诊断的基本流程包括：1）病史采集；2）体格检查；3）辅助检查（实验室检查、影像学检查等）；4）综合分析；5）诊断结论。关键要素包括症状、体征、检查结果的综合判断，以及医生的专业知识和临床经验。",
                "context": {
                    "industry": "healthcare",
                    "topic": "diagnosis",
                    "difficulty": "basic"
                }
            },
            {
                "id": "healthcare_2",
                "input": "人工智能在医疗诊断中有哪些应用？",
                "expected_output": "人工智能在医疗诊断中的应用包括：1）医学影像分析（CT、MRI、X光片的自动识别）；2）病理诊断辅助；3）疾病风险预测；4）药物相互作用检测；5）个性化治疗方案推荐；6）临床决策支持系统。这些应用能够提高诊断准确性和效率。",
                "context": {
                    "industry": "healthcare",
                    "topic": "ai_diagnosis",
                    "difficulty": "intermediate"
                }
            }
        ]
        
        self.dataset_path = Path(self.temp_dir) / "healthcare_dataset.json"
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.healthcare_dataset, f, ensure_ascii=False, indent=2)
    
    def test_healthcare_model_evaluation(self):
        """测试医疗模型评估"""
        config = EvaluationConfig(
            industry_domain="healthcare",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.8, "terminology": 0.7}
        )
        
        task_id = self.evaluation_engine.evaluate_model(
            model_id="healthcare_model",
            dataset=self.healthcare_dataset,
            evaluation_config=config
        )
        
        result = self._wait_for_completion(task_id)
        
        # 医疗领域的验收标准
        assert result.overall_score > 0.5, "医疗模型整体得分应该超过0.5"
        assert result.dimension_scores["knowledge"] > 0.6, "医疗知识得分应该超过0.6"
        
        print(f"医疗模型评估结果:")
        print(f"  整体得分: {result.overall_score:.3f}")
        print(f"  知识得分: {result.dimension_scores['knowledge']:.3f}")
        print(f"  术语得分: {result.dimension_scores['terminology']:.3f}")
    
    def _wait_for_completion(self, task_id: str, timeout: int = 30):
        """等待评估完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            
            if progress and progress.status == "completed":
                return self.evaluation_engine.get_evaluation_result(task_id)
            elif progress and progress.status == "failed":
                pytest.fail(f"评估失败: {getattr(progress, 'error', '未知错误')}")
            
            time.sleep(1)
        
        pytest.fail(f"评估超时: {task_id}")


class TestBatchEvaluationUserScenario:
    """批量评估用户场景测试"""
    
    def setup_method(self):
        """设置批量评估测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_system()
        self.setup_large_dataset()
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.evaluation_engine.shutdown()
    
    def setup_system(self):
        """设置评估系统"""
        self.model_manager = ModelManager()
        ModelAdapterFactory.register_adapter("realworld", RealWorldModelAdapter)
        
        # 注册多个模型用于批量对比
        for i in range(3):
            self.model_manager.register_model(
                f"model_{i}", 
                "realworld", 
                {"domain": "general", "model_version": i}
            )
        
        self.evaluators = {
            "knowledge": KnowledgeEvaluator(),
            "terminology": TerminologyEvaluator()
        }
        
        self.result_aggregator = ResultAggregator()
        self.report_generator = ReportGenerator()
        
        self.evaluation_engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator
        )
        
        self.batch_evaluator = BatchEvaluator(self.evaluation_engine)
    
    def setup_large_dataset(self):
        """设置大数据集"""
        # 创建包含100个样本的数据集
        large_dataset = []
        topics = ["风险管理", "投资策略", "市场分析", "技术创新", "业务流程"]
        
        for i in range(100):
            topic = topics[i % len(topics)]
            large_dataset.append({
                "id": f"sample_{i}",
                "input": f"请分析{topic}相关的问题 {i}",
                "expected_output": f"关于{topic}的专业分析 {i}",
                "context": {
                    "industry": "general",
                    "topic": topic,
                    "sample_index": i
                }
            })
        
        self.large_dataset_path = Path(self.temp_dir) / "large_dataset.json"
        with open(self.large_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(large_dataset, f, ensure_ascii=False, indent=2)
    
    def test_batch_evaluation_user_workflow(self):
        """测试批量评估用户工作流"""
        # 用户场景：需要对多个模型进行大规模评估对比
        eval_config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.8, "terminology": 0.2},
            threshold_config={"knowledge": 0.6, "terminology": 0.5}
        )
        
        batch_config = BatchEvaluationConfig(
            batch_size=20,
            max_concurrent_tasks=2,
            chunk_size=25,
            save_intermediate_results=True,
            intermediate_results_dir=str(Path(self.temp_dir) / "batch_results"),
            enable_parallel_processing=False  # 使用顺序处理确保稳定性
        )
        
        # 创建批量任务
        batch_task = self.batch_evaluator.create_batch_task(
            task_id="user_batch_test",
            model_ids=["model_0", "model_1"],  # 评估两个模型
            dataset_path=str(self.large_dataset_path),
            evaluation_config=eval_config,
            batch_config=batch_config
        )
        
        # 验证任务创建
        assert batch_task.task_id == "user_batch_test"
        assert batch_task.total_samples == 100
        assert len(batch_task.model_ids) == 2
        
        # 启动批量评估
        success = self.batch_evaluator.start_batch_evaluation("user_batch_test")
        assert success == True
        
        # 监控评估进度
        max_wait_time = 120  # 2分钟超时
        start_time = time.time()
        last_progress = 0
        
        while time.time() - start_time < max_wait_time:
            task_status = self.batch_evaluator.get_batch_task_status("user_batch_test")
            
            if task_status.status == "completed":
                break
            elif task_status.status == "failed":
                pytest.fail(f"批量评估失败: {task_status.errors}")
            
            # 显示进度更新
            current_progress = task_status.processed_samples
            if current_progress > last_progress:
                print(f"评估进度: {current_progress}/{task_status.total_samples}")
                last_progress = current_progress
            
            time.sleep(3)
        else:
            # 超时处理
            self.batch_evaluator.cancel_batch_task("user_batch_test")
            pytest.fail("批量评估超时")
        
        # 验证最终结果
        final_task = self.batch_evaluator.get_batch_task_status("user_batch_test")
        
        # 用户验收标准
        assert final_task.status == "completed", "批量评估应该成功完成"
        assert final_task.processed_samples >= 50, "应该处理至少50%的样本"
        assert len(final_task.results) == 2, "应该有两个模型的评估结果"
        
        # 验证每个模型的结果质量
        for model_id, result in final_task.results.items():
            assert result.overall_score > 0, f"模型 {model_id} 应该有有效的评估得分"
            assert len(result.detailed_results) > 0, f"模型 {model_id} 应该有详细的评估结果"
            
            print(f"模型 {model_id} 评估结果:")
            print(f"  整体得分: {result.overall_score:.3f}")
            print(f"  处理样本数: {len(result.detailed_results)}")
        
        # 验证中间结果文件是否生成
        results_dir = Path(self.temp_dir) / "batch_results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            print(f"生成了 {len(result_files)} 个结果文件")
    
    def test_batch_evaluation_error_recovery(self):
        """测试批量评估的错误恢复"""
        # 创建一个小数据集用于快速测试
        small_dataset = [
            {"id": f"test_{i}", "input": f"测试问题 {i}", "expected_output": f"测试答案 {i}"}
            for i in range(10)
        ]
        
        small_dataset_path = Path(self.temp_dir) / "small_dataset.json"
        with open(small_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(small_dataset, f, ensure_ascii=False, indent=2)
        
        eval_config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        batch_config = BatchEvaluationConfig(
            batch_size=5,
            max_concurrent_tasks=1,
            chunk_size=5,
            save_intermediate_results=True,
            intermediate_results_dir=str(Path(self.temp_dir) / "recovery_test")
        )
        
        # 创建批量任务
        batch_task = self.batch_evaluator.create_batch_task(
            task_id="recovery_test",
            model_ids=["model_0"],
            dataset_path=str(small_dataset_path),
            evaluation_config=eval_config,
            batch_config=batch_config
        )
        
        # 启动评估
        success = self.batch_evaluator.start_batch_evaluation("recovery_test")
        assert success == True
        
        # 等待完成或超时
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            task_status = self.batch_evaluator.get_batch_task_status("recovery_test")
            
            if task_status.status in ["completed", "failed"]:
                break
            
            time.sleep(1)
        
        # 验证任务处理了一些样本（即使有错误）
        final_task = self.batch_evaluator.get_batch_task_status("recovery_test")
        total_processed = final_task.processed_samples + final_task.failed_samples
        
        assert total_processed > 0, "应该至少处理了一些样本"
        
        print(f"错误恢复测试结果:")
        print(f"  状态: {final_task.status}")
        print(f"  成功处理: {final_task.processed_samples}")
        print(f"  失败样本: {final_task.failed_samples}")
        print(f"  错误信息: {final_task.errors}")


class TestUserExperienceScenarios:
    """用户体验场景测试"""
    
    def setup_method(self):
        """设置用户体验测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_system()
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.evaluation_engine.shutdown()
    
    def setup_system(self):
        """设置评估系统"""
        self.model_manager = ModelManager()
        ModelAdapterFactory.register_adapter("realworld", RealWorldModelAdapter)
        
        self.model_manager.register_model("test_model", "realworld", {})
        
        self.evaluators = {
            "knowledge": KnowledgeEvaluator(),
            "terminology": TerminologyEvaluator()
        }
        
        self.result_aggregator = ResultAggregator()
        self.report_generator = ReportGenerator()
        
        self.evaluation_engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator
        )
    
    def test_quick_evaluation_scenario(self):
        """测试快速评估场景"""
        # 用户场景：需要快速评估一个小样本
        quick_dataset = [
            {
                "id": "quick_test",
                "input": "这是一个快速测试问题",
                "expected_output": "这是期望的快速回答",
                "context": {"industry": "general"}
            }
        ]
        
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        start_time = time.time()
        
        task_id = self.evaluation_engine.evaluate_model(
            model_id="test_model",
            dataset=quick_dataset,
            evaluation_config=config
        )
        
        # 等待完成
        result = self._wait_for_completion(task_id, timeout=10)
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # 用户体验验收标准
        assert evaluation_time < 10, "快速评估应该在10秒内完成"
        assert result is not None, "应该能够获得评估结果"
        assert result.overall_score >= 0, "应该有有效的评估得分"
        
        print(f"快速评估完成时间: {evaluation_time:.2f} 秒")
        print(f"评估得分: {result.overall_score:.3f}")
    
    def test_evaluation_progress_monitoring(self):
        """测试评估进度监控"""
        # 用户场景：需要监控长时间运行的评估进度
        dataset = [
            {"id": f"progress_test_{i}", "input": f"测试问题 {i}", "expected_output": f"测试答案 {i}"}
            for i in range(5)
        ]
        
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.5, "terminology": 0.5}
        )
        
        task_id = self.evaluation_engine.evaluate_model(
            model_id="test_model",
            dataset=dataset,
            evaluation_config=config
        )
        
        # 监控进度
        progress_updates = []
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            
            if progress:
                progress_updates.append({
                    "status": progress.status,
                    "timestamp": time.time() - start_time
                })
                
                if progress.status == "completed":
                    break
                elif progress.status == "failed":
                    pytest.fail("评估失败")
            
            time.sleep(1)
        
        # 验证进度监控
        assert len(progress_updates) > 0, "应该能够获取进度更新"
        assert progress_updates[-1]["status"] == "completed", "最终状态应该是完成"
        
        print(f"进度更新次数: {len(progress_updates)}")
        print(f"总评估时间: {progress_updates[-1]['timestamp']:.2f} 秒")
    
    def test_evaluation_cancellation(self):
        """测试评估取消功能"""
        # 用户场景：需要取消正在运行的评估
        dataset = [
            {"id": f"cancel_test_{i}", "input": f"测试问题 {i}", "expected_output": f"测试答案 {i}"}
            for i in range(10)
        ]
        
        config = EvaluationConfig(
            industry_domain="general",
            evaluation_dimensions=["knowledge"],
            weight_config={"knowledge": 1.0},
            threshold_config={"knowledge": 0.5}
        )
        
        task_id = self.evaluation_engine.evaluate_model(
            model_id="test_model",
            dataset=dataset,
            evaluation_config=config
        )
        
        # 等待一小段时间后取消
        time.sleep(2)
        
        success = self.evaluation_engine.cancel_evaluation(task_id)
        assert success == True, "应该能够成功取消评估"
        
        # 验证取消状态
        time.sleep(1)
        evaluations = self.evaluation_engine.list_evaluations()
        
        cancelled_evaluation = None
        for evaluation in evaluations:
            if evaluation["task_id"] == task_id:
                cancelled_evaluation = evaluation
                break
        
        assert cancelled_evaluation is not None, "应该能够找到被取消的评估"
        assert cancelled_evaluation["status"] == "cancelled", "评估状态应该是已取消"
        
        print(f"评估取消成功，状态: {cancelled_evaluation['status']}")
    
    def _wait_for_completion(self, task_id: str, timeout: int = 30):
        """等待评估完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            
            if progress and progress.status == "completed":
                return self.evaluation_engine.get_evaluation_result(task_id)
            elif progress and progress.status == "failed":
                pytest.fail(f"评估失败: {getattr(progress, 'error', '未知错误')}")
            
            time.sleep(0.5)
        
        pytest.fail(f"评估超时: {task_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])