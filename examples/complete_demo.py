"""
Industry Evaluation System 完整功能演示

这个示例程序展示了行业评估系统的所有核心功能：
1. 配置管理
2. 模型适配器
3. 评估器
4. 评估引擎
5. 批量评估
6. 报告生成
7. API接口
"""

import asyncio
import json
import logging
import tempfile
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入系统组件
from industry_evaluation.config.config_manager import (
    ConfigManager, ConfigTemplate, ModelConfig, EvaluatorConfig
)
from industry_evaluation.adapters.model_adapter import (
    ModelManager, ModelAdapterFactory
)
from industry_evaluation.core.evaluation_engine import IndustryEvaluationEngine
from industry_evaluation.core.batch_evaluator import BatchEvaluator, BatchEvaluationConfig
from industry_evaluation.core.result_aggregator import ResultAggregator
from industry_evaluation.reporting.report_generator import ReportGenerator
from industry_evaluation.evaluators.knowledge_evaluator import KnowledgeEvaluator
from industry_evaluation.evaluators.terminology_evaluator import TerminologyEvaluator
from industry_evaluation.evaluators.reasoning_evaluator import ReasoningEvaluator
from industry_evaluation.evaluators.long_text_evaluator import LongTextEvaluator
from industry_evaluation.core.interfaces import EvaluationConfig
from industry_evaluation.api.rest_api import EvaluationAPI


class MockModelAdapter:
    """演示用的模拟模型适配器"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.config = config
        self.quality = config.get("quality", "good")
        self.domain = config.get("domain", "general")
        
        # 模拟不同质量的知识库
        self.knowledge_base = {
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
        
        # 根据质量和领域生成不同质量的回答
        if self.quality == "excellent" and domain in self.knowledge_base:
            # 高质量回答：准确且详细
            for keyword, definition in self.knowledge_base[domain].items():
                if keyword in input_text:
                    return f"{definition} 这是一个在{domain}领域中非常重要的概念，需要深入理解其应用场景和影响因素。"
            
            return f"针对{domain}领域的问题'{input_text}'，这是一个复杂的专业问题，需要综合考虑多个因素进行分析。"
        
        elif self.quality == "good":
            # 中等质量回答：基本准确
            for keyword, definition in self.knowledge_base.get(domain, {}).items():
                if keyword in input_text:
                    return f"{definition}"
            
            return f"关于'{input_text}'的问题，这涉及到{domain}领域的专业知识。"
        
        else:
            # 低质量回答：简单或可能有错误
            return f"关于'{input_text}'，这是一个{domain}相关的问题。"
    
    def is_available(self):
        return True


class IndustryEvaluationDemo:
    """行业评估系统完整演示"""
    
    def __init__(self):
        """初始化演示环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "demo_config.yaml"
        
        logger.info(f"演示环境初始化完成，临时目录: {self.temp_dir}")
        
        # 初始化组件
        self.setup_configuration()
        self.setup_models()
        self.setup_evaluators()
        self.setup_evaluation_engine()
        self.setup_test_data()
    
    def setup_configuration(self):
        """设置配置管理"""
        logger.info("🔧 设置配置管理系统...")
        
        # 创建金融行业配置模板
        config = ConfigTemplate.generate_finance_config()
        
        # 添加演示模型配置
        config.models = {
            "finance_expert": ModelConfig(
                model_id="finance_expert",
                adapter_type="demo",
                timeout=30,
                max_retries=3,
                retry_config={
                    "strategy": "exponential_backoff",
                    "base_delay": 1.0,
                    "max_delay": 10.0
                }
            ),
            "general_model": ModelConfig(
                model_id="general_model",
                adapter_type="demo",
                timeout=30,
                max_retries=2
            ),
            "poor_model": ModelConfig(
                model_id="poor_model",
                adapter_type="demo",
                timeout=30,
                max_retries=1
            )
        }
        
        # 保存配置
        ConfigTemplate.save_template(config, self.config_file)
        
        # 创建配置管理器
        self.config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        logger.info("✅ 配置管理系统设置完成")
    
    def setup_models(self):
        """设置模型管理"""
        logger.info("🤖 设置模型管理系统...")
        
        # 注册演示模型适配器
        ModelAdapterFactory.register_adapter("demo", MockModelAdapter)
        
        # 创建模型管理器
        self.model_manager = ModelManager()
        
        # 注册不同质量的模型
        self.model_manager.register_model(
            "finance_expert", 
            "demo", 
            {"quality": "excellent", "domain": "finance"}
        )
        
        self.model_manager.register_model(
            "general_model", 
            "demo", 
            {"quality": "good", "domain": "general"}
        )
        
        self.model_manager.register_model(
            "poor_model", 
            "demo", 
            {"quality": "poor", "domain": "general"}
        )
        
        logger.info("✅ 模型管理系统设置完成")
    
    def setup_evaluators(self):
        """设置评估器"""
        logger.info("📊 设置评估器系统...")
        
        self.evaluators = {
            "knowledge": KnowledgeEvaluator(),
            "terminology": TerminologyEvaluator(),
            "reasoning": ReasoningEvaluator(),
            "long_text": LongTextEvaluator()
        }
        
        logger.info("✅ 评估器系统设置完成")
    
    def setup_evaluation_engine(self):
        """设置评估引擎"""
        logger.info("🚀 设置评估引擎...")
        
        # 创建核心组件
        self.result_aggregator = ResultAggregator()
        self.report_generator = ReportGenerator()
        
        # 创建评估引擎
        self.evaluation_engine = IndustryEvaluationEngine(
            model_manager=self.model_manager,
            evaluators=self.evaluators,
            result_aggregator=self.result_aggregator,
            report_generator=self.report_generator,
            max_workers=2
        )
        
        # 创建批量评估器
        self.batch_evaluator = BatchEvaluator(self.evaluation_engine)
        
        logger.info("✅ 评估引擎设置完成")
    
    def setup_test_data(self):
        """设置测试数据"""
        logger.info("📝 准备测试数据...")
        
        # 金融领域测试数据
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
        
        # 创建大数据集用于批量测试
        self.large_dataset = []
        topics = ["风险管理", "投资策略", "市场分析", "金融创新", "监管合规"]
        
        for i in range(50):
            topic = topics[i % len(topics)]
            self.large_dataset.append({
                "id": f"large_sample_{i}",
                "input": f"请分析{topic}相关的问题 {i}：如何在当前市场环境下优化策略？",
                "expected_output": f"关于{topic}的专业分析 {i}：需要综合考虑市场环境、风险因素和监管要求。",
                "context": {
                    "industry": "finance",
                    "topic": topic.replace(" ", "_"),
                    "sample_index": i
                }
            })
        
        # 保存大数据集到文件
        self.large_dataset_path = self.temp_dir / "large_dataset.json"
        with open(self.large_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.large_dataset, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ 测试数据准备完成")
    
    async def demo_single_evaluation(self):
        """演示单模型评估"""
        logger.info("\n" + "="*60)
        logger.info("🎯 演示1: 单模型评估")
        logger.info("="*60)
        
        # 配置评估参数
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology", "reasoning"],
            weight_config={
                "knowledge": 0.5,
                "terminology": 0.3,
                "reasoning": 0.2
            },
            threshold_config={
                "knowledge": 0.7,
                "terminology": 0.6,
                "reasoning": 0.7
            },
            auto_generate_report=True
        )
        
        logger.info("📋 评估配置:")
        logger.info(f"  - 行业领域: {config.industry_domain}")
        logger.info(f"  - 评估维度: {config.evaluation_dimensions}")
        logger.info(f"  - 权重配置: {config.weight_config}")
        
        # 启动评估
        logger.info("🚀 启动金融专家模型评估...")
        task_id = self.evaluation_engine.evaluate_model(
            model_id="finance_expert",
            dataset=self.finance_dataset,
            evaluation_config=config
        )
        
        logger.info(f"📝 评估任务ID: {task_id}")
        
        # 监控评估进度
        await self._monitor_evaluation_progress(task_id)
        
        # 获取评估结果
        result = self.evaluation_engine.get_evaluation_result(task_id)
        
        if result:
            logger.info("📊 评估结果:")
            logger.info(f"  - 综合得分: {result.overall_score:.3f}")
            logger.info(f"  - 知识得分: {result.dimension_scores.get('knowledge', 0):.3f}")
            logger.info(f"  - 术语得分: {result.dimension_scores.get('terminology', 0):.3f}")
            logger.info(f"  - 推理得分: {result.dimension_scores.get('reasoning', 0):.3f}")
            logger.info(f"  - 处理样本数: {len(result.detailed_results)}")
            
            if result.improvement_suggestions:
                logger.info("💡 改进建议:")
                for suggestion in result.improvement_suggestions[:3]:
                    logger.info(f"  - {suggestion}")
        
        return task_id
    
    async def demo_model_comparison(self):
        """演示模型对比评估"""
        logger.info("\n" + "="*60)
        logger.info("⚖️ 演示2: 模型对比评估")
        logger.info("="*60)
        
        config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.6, "terminology": 0.5}
        )
        
        models_to_compare = ["finance_expert", "general_model", "poor_model"]
        results = {}
        
        logger.info(f"🔄 开始对比 {len(models_to_compare)} 个模型...")
        
        # 并行评估多个模型
        tasks = []
        for model_id in models_to_compare:
            logger.info(f"🚀 启动模型 {model_id} 的评估...")
            task_id = self.evaluation_engine.evaluate_model(
                model_id=model_id,
                dataset=self.finance_dataset[:2],  # 使用较小的数据集
                evaluation_config=config
            )
            tasks.append((model_id, task_id))
        
        # 等待所有评估完成
        for model_id, task_id in tasks:
            await self._monitor_evaluation_progress(task_id, model_name=model_id)
            result = self.evaluation_engine.get_evaluation_result(task_id)
            if result:
                results[model_id] = result
        
        # 显示对比结果
        logger.info("\n📊 模型对比结果:")
        logger.info("-" * 80)
        logger.info(f"{'模型名称':<15} {'综合得分':<10} {'知识得分':<10} {'术语得分':<10}")
        logger.info("-" * 80)
        
        for model_id, result in results.items():
            logger.info(
                f"{model_id:<15} "
                f"{result.overall_score:<10.3f} "
                f"{result.dimension_scores.get('knowledge', 0):<10.3f} "
                f"{result.dimension_scores.get('terminology', 0):<10.3f}"
            )
        
        # 找出最佳模型
        if results:
            best_model = max(results.items(), key=lambda x: x[1].overall_score)
            logger.info(f"\n🏆 最佳模型: {best_model[0]} (得分: {best_model[1].overall_score:.3f})")
        
        return results
    
    async def demo_batch_evaluation(self):
        """演示批量评估"""
        logger.info("\n" + "="*60)
        logger.info("📦 演示3: 批量评估")
        logger.info("="*60)
        
        # 配置批量评估
        eval_config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.8, "terminology": 0.2},
            threshold_config={"knowledge": 0.6, "terminology": 0.5}
        )
        
        batch_config = BatchEvaluationConfig(
            batch_size=10,
            max_concurrent_tasks=2,
            chunk_size=20,
            save_intermediate_results=True,
            intermediate_results_dir=str(self.temp_dir / "batch_results"),
            enable_parallel_processing=False  # 使用顺序处理以便观察
        )
        
        logger.info("📋 批量评估配置:")
        logger.info(f"  - 数据集大小: {len(self.large_dataset)} 样本")
        logger.info(f"  - 批次大小: {batch_config.batch_size}")
        logger.info(f"  - 评估模型: ['finance_expert', 'general_model']")
        
        # 创建批量任务
        batch_task = self.batch_evaluator.create_batch_task(
            task_id="demo_batch_evaluation",
            model_ids=["finance_expert", "general_model"],
            dataset_path=str(self.large_dataset_path),
            evaluation_config=eval_config,
            batch_config=batch_config
        )
        
        logger.info(f"📝 批量任务ID: {batch_task.task_id}")
        logger.info(f"📊 总样本数: {batch_task.total_samples}")
        
        # 启动批量评估
        success = self.batch_evaluator.start_batch_evaluation("demo_batch_evaluation")
        
        if success:
            logger.info("🚀 批量评估已启动...")
            
            # 监控批量评估进度
            await self._monitor_batch_evaluation_progress("demo_batch_evaluation")
            
            # 获取最终结果
            final_task = self.batch_evaluator.get_batch_task_status("demo_batch_evaluation")
            
            if final_task and final_task.status == "completed":
                logger.info("\n📊 批量评估结果:")
                logger.info(f"  - 状态: {final_task.status}")
                logger.info(f"  - 处理样本: {final_task.processed_samples}/{final_task.total_samples}")
                logger.info(f"  - 失败样本: {final_task.failed_samples}")
                
                if final_task.results:
                    logger.info("  - 模型结果:")
                    for model_id, result in final_task.results.items():
                        logger.info(f"    * {model_id}: {result.overall_score:.3f} (样本数: {len(result.detailed_results)})")
        
        return batch_task
    
    async def demo_report_generation(self, task_id: str):
        """演示报告生成"""
        logger.info("\n" + "="*60)
        logger.info("📄 演示4: 报告生成")
        logger.info("="*60)
        
        logger.info("📝 生成评估报告...")
        
        # 生成JSON格式报告
        json_report = self.evaluation_engine.generate_report(task_id, "json")
        
        if json_report:
            logger.info("✅ JSON报告生成成功")
            
            # 解析并显示报告摘要
            if isinstance(json_report, str):
                try:
                    report_data = json.loads(json_report)
                    logger.info("📊 报告摘要:")
                    logger.info(f"  - 综合得分: {report_data.get('overall_score', 'N/A')}")
                    logger.info(f"  - 评估维度: {list(report_data.get('dimension_scores', {}).keys())}")
                    logger.info(f"  - 行业领域: {report_data.get('industry_domain', 'N/A')}")
                    
                    # 保存报告到文件
                    report_file = self.temp_dir / f"evaluation_report_{task_id}.json"
                    with open(report_file, 'w', encoding='utf-8') as f:
                        json.dump(report_data, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"💾 报告已保存到: {report_file}")
                    
                except json.JSONDecodeError:
                    logger.warning("⚠️ 报告格式解析失败")
        else:
            logger.warning("⚠️ 报告生成失败")
    
    async def demo_api_interface(self):
        """演示API接口"""
        logger.info("\n" + "="*60)
        logger.info("🌐 演示5: API接口")
        logger.info("="*60)
        
        try:
            # 创建API实例
            api = EvaluationAPI(self.config_manager)
            app = api.get_app()
            
            # 创建测试客户端
            with app.test_client() as client:
                logger.info("🔍 测试API端点...")
                
                # 测试健康检查
                response = client.get('/health')
                logger.info(f"  - 健康检查: {response.status_code} - {json.loads(response.data)['status']}")
                
                # 测试系统信息
                response = client.get('/info')
                if response.status_code == 200:
                    info = json.loads(response.data)
                    logger.info(f"  - 系统信息: 版本 {info.get('version', 'N/A')}")
                
                # 测试模型列表
                response = client.get('/models')
                if response.status_code == 200:
                    models = json.loads(response.data)
                    logger.info(f"  - 模型列表: {len(models.get('data', []))} 个模型")
                
                # 测试配置获取
                response = client.get('/config')
                if response.status_code == 200:
                    config = json.loads(response.data)
                    logger.info(f"  - 配置信息: {len(config.get('data', {}).get('models', {}))} 个配置模型")
                
                logger.info("✅ API接口测试完成")
                
        except Exception as e:
            logger.error(f"❌ API接口测试失败: {str(e)}")
    
    async def demo_configuration_management(self):
        """演示配置管理"""
        logger.info("\n" + "="*60)
        logger.info("⚙️ 演示6: 配置管理")
        logger.info("="*60)
        
        logger.info("📋 当前配置信息:")
        config = self.config_manager.get_config()
        logger.info(f"  - 版本: {config.version}")
        logger.info(f"  - 最大工作线程: {config.system.max_workers}")
        logger.info(f"  - 日志级别: {config.system.log_level}")
        logger.info(f"  - 模型数量: {len(config.models)}")
        logger.info(f"  - 评估器数量: {len(config.evaluators)}")
        
        # 演示配置更新
        logger.info("🔄 演示配置更新...")
        updates = {
            "system": {
                "max_workers": 8,
                "log_level": "DEBUG"
            }
        }
        
        success = self.config_manager.update_config(updates)
        if success:
            logger.info("✅ 配置更新成功")
            updated_config = self.config_manager.get_config()
            logger.info(f"  - 新的最大工作线程: {updated_config.system.max_workers}")
            logger.info(f"  - 新的日志级别: {updated_config.system.log_level}")
        else:
            logger.warning("⚠️ 配置更新失败")
        
        # 演示模型配置管理
        logger.info("🤖 演示模型配置管理...")
        new_model_config = ModelConfig(
            model_id="demo_new_model",
            adapter_type="demo",
            timeout=60,
            max_retries=5
        )
        
        success = self.config_manager.add_model("demo_new_model", new_model_config)
        if success:
            logger.info("✅ 新模型配置添加成功")
            
            # 移除演示模型
            success = self.config_manager.remove_model("demo_new_model")
            if success:
                logger.info("✅ 演示模型配置移除成功")
    
    async def _monitor_evaluation_progress(self, task_id: str, model_name: str = None):
        """监控评估进度"""
        model_info = f" ({model_name})" if model_name else ""
        logger.info(f"⏳ 监控评估进度{model_info}...")
        
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress = self.evaluation_engine.get_evaluation_progress(task_id)
            
            if progress:
                if progress.status == "completed":
                    logger.info(f"✅ 评估完成{model_info}")
                    break
                elif progress.status == "failed":
                    logger.error(f"❌ 评估失败{model_info}")
                    break
                elif progress.status == "running":
                    logger.info(f"🔄 评估进行中{model_info}...")
            
            await asyncio.sleep(1)
        else:
            logger.warning(f"⏰ 评估监控超时{model_info}")
    
    async def _monitor_batch_evaluation_progress(self, task_id: str):
        """监控批量评估进度"""
        logger.info("⏳ 监控批量评估进度...")
        
        max_wait_time = 60
        start_time = time.time()
        last_progress = 0
        
        while time.time() - start_time < max_wait_time:
            task_status = self.batch_evaluator.get_batch_task_status(task_id)
            
            if task_status:
                if task_status.status == "completed":
                    logger.info("✅ 批量评估完成")
                    break
                elif task_status.status == "failed":
                    logger.error("❌ 批量评估失败")
                    if task_status.errors:
                        for error in task_status.errors[:3]:
                            logger.error(f"  - {error}")
                    break
                elif task_status.status == "running":
                    current_progress = task_status.processed_samples
                    if current_progress > last_progress:
                        logger.info(f"🔄 批量评估进度: {current_progress}/{task_status.total_samples}")
                        last_progress = current_progress
            
            await asyncio.sleep(2)
        else:
            logger.warning("⏰ 批量评估监控超时")
    
    async def run_complete_demo(self):
        """运行完整演示"""
        logger.info("🎬 开始 Industry Evaluation System 完整功能演示")
        logger.info("=" * 80)
        
        try:
            # 1. 单模型评估
            task_id = await self.demo_single_evaluation()
            
            # 2. 模型对比评估
            await self.demo_model_comparison()
            
            # 3. 批量评估
            await self.demo_batch_evaluation()
            
            # 4. 报告生成
            if task_id:
                await self.demo_report_generation(task_id)
            
            # 5. API接口
            await self.demo_api_interface()
            
            # 6. 配置管理
            await self.demo_configuration_management()
            
            logger.info("\n" + "=" * 80)
            logger.info("🎉 完整功能演示结束")
            logger.info("=" * 80)
            
            # 显示演示总结
            self._show_demo_summary()
            
        except Exception as e:
            logger.error(f"❌ 演示过程中发生错误: {str(e)}")
            raise
        finally:
            # 清理资源
            self.cleanup()
    
    def _show_demo_summary(self):
        """显示演示总结"""
        logger.info("\n📋 演示总结:")
        logger.info("✅ 已演示的功能:")
        logger.info("  1. 🔧 配置管理 - 灵活的配置文件管理和热更新")
        logger.info("  2. 🤖 模型适配器 - 支持多种模型类型和异常处理")
        logger.info("  3. 📊 评估器系统 - 多维度专业评估能力")
        logger.info("  4. 🎯 单模型评估 - 详细的评估结果和改进建议")
        logger.info("  5. ⚖️ 模型对比 - 多模型并行评估和对比分析")
        logger.info("  6. 📦 批量评估 - 大规模数据集的高效处理")
        logger.info("  7. 📄 报告生成 - 专业的评估报告输出")
        logger.info("  8. 🌐 API接口 - RESTful API和自动文档")
        
        logger.info(f"\n📁 演示文件位置: {self.temp_dir}")
        logger.info("💡 提示: 可以查看临时目录中的配置文件和报告文件")
    
    def cleanup(self):
        """清理演示环境"""
        try:
            # 关闭评估引擎
            if hasattr(self, 'evaluation_engine'):
                self.evaluation_engine.shutdown()
            
            logger.info("🧹 演示环境清理完成")
        except Exception as e:
            logger.warning(f"⚠️ 清理过程中发生错误: {str(e)}")


async def main():
    """主函数"""
    print("🚀 Industry Evaluation System - 完整功能演示")
    print("=" * 80)
    print("本演示将展示行业评估系统的所有核心功能：")
    print("• 配置管理和模型适配器")
    print("• 单模型评估和模型对比")
    print("• 批量评估和报告生成")
    print("• API接口和配置管理")
    print("=" * 80)
    
    # 创建并运行演示
    demo = IndustryEvaluationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # 运行异步演示
    asyncio.run(main())