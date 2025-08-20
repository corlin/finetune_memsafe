"""
Industry Evaluation System 简化演示 - 智谱模型版本

这个示例程序展示了如何使用智谱模型（BigModel GLM）进行行业评估系统的演示，
适合快速了解系统与智谱模型的集成使用方法。

使用方法:
1. 设置环境变量: export BIGMODEL_API_KEY="your_api_key_here"
2. 运行示例: python examples/simple_demo.py

或者在代码中直接设置API密钥（不推荐用于生产环境）
"""

import json
import logging
import tempfile
import time
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 导入系统组件
from industry_evaluation.config.config_manager import ConfigManager, ConfigTemplate, ModelConfig
from industry_evaluation.adapters.model_adapter import ModelManager, ModelAdapterFactory
from industry_evaluation.core.evaluation_engine import IndustryEvaluationEngine
from industry_evaluation.core.result_aggregator import ResultAggregator
from industry_evaluation.reporting.report_generator import ReportGenerator
from industry_evaluation.evaluators.knowledge_evaluator import KnowledgeEvaluator
from industry_evaluation.evaluators.terminology_evaluator import TerminologyEvaluator
from industry_evaluation.core.interfaces import EvaluationConfig


def simple_evaluation_demo():
    """使用智谱模型的简化评估演示"""
    
    print("🚀 Industry Evaluation System - 智谱模型演示")
    print("=" * 50)
    
    # 获取智谱API密钥
    api_key = os.getenv("BIGMODEL_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到智谱API密钥")
        print("请通过以下方式设置API密钥:")
        print("  1. 环境变量: export BIGMODEL_API_KEY=your_api_key_here")
        print("  2. 或在代码中直接设置 (不推荐用于生产环境)")
        print("\n💡 获取API密钥:")
        print("  访问 https://open.bigmodel.cn 注册账号并获取API密钥")
        return
    
    print(f"🔑 API密钥: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    
    # 1. 设置临时环境
    temp_dir = Path(tempfile.mkdtemp())
    config_file = temp_dir / "simple_config.yaml"
    
    print(f"📁 临时目录: {temp_dir}")
    
    try:
        # 2. 创建配置
        print("\n🔧 设置配置...")
        config = ConfigTemplate.generate_finance_config()
        ConfigTemplate.save_template(config, config_file)
        config_manager = ConfigManager(config_file, auto_reload=False)
        print("✅ 配置创建完成")
        
        # 3. 设置智谱模型
        print("\n🤖 设置智谱模型...")
        model_manager = ModelManager()
        
        # 创建智谱GLM-4.5模型适配器
        try:
            glm_adapter = ModelAdapterFactory.create_openai_compatible_adapter(
                model_id="myglm-4.5",
                provider="bigmodel",
                api_key=api_key,
                model_name="glm-4.5",
                timeout=60,
                custom_headers={
                    "User-Agent": "Industry-Evaluation-Demo/1.0"
                }
            )
            
            # 注册智谱模型
            model_manager.register_model(
                "glm-4.5",
                "openai_compatible",
                glm_adapter.config
            )
            
            print("✅ 智谱GLM-4.5模型设置完成")
            print(f"📡 API端点: {glm_adapter.base_url}")
            
            # 快速连通性测试
            print("🔍 执行连通性测试...")
            try:
                test_response = glm_adapter.predict("你好，请简单介绍一下你自己。", {"max_tokens": 50})
                if test_response:
                    print(f"✅ 连通性测试成功: {test_response[:100]}...")
                else:
                    print("⚠️ 连通性测试返回空结果，但连接正常")
            except Exception as test_error:
                print(f"⚠️ 连通性测试失败: {str(test_error)}")
                print("继续尝试评估...")
            
        except Exception as e:
            print(f"❌ 智谱模型设置失败: {str(e)}")
            print("请检查API密钥是否正确，或网络连接是否正常")
            return
        
        # 4. 设置评估器
        print("\n📊 设置评估器...")
        evaluators = {
            "knowledge": KnowledgeEvaluator(),
            "terminology": TerminologyEvaluator()
        }
        print("✅ 评估器设置完成")
        
        # 5. 创建评估引擎
        print("\n🚀 创建评估引擎...")
        result_aggregator = ResultAggregator()
        report_generator = ReportGenerator()
        
        evaluation_engine = IndustryEvaluationEngine(
            model_manager=model_manager,
            evaluators=evaluators,
            result_aggregator=result_aggregator,
            report_generator=report_generator,
            max_workers=1
        )
        print("✅ 评估引擎创建完成")
        
        # 6. 准备测试数据
        print("\n📝 准备测试数据...")
        test_dataset = [
            {
                "id": "zhipu_test_1",
                "input": "什么是金融风险管理？请详细解释其核心概念和主要方法。",
                "expected_output": "金融风险管理是识别、评估和控制金融风险的过程。",
                "context": {
                    "industry": "finance", 
                    "topic": "risk_management",
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "system_prompt": "你是一个专业的金融风险管理专家，请提供准确、专业的回答。"
                }
            },
            {
                "id": "zhipu_test_2", 
                "input": "请解释VaR模型的原理、计算方法和在风险管理中的应用。",
                "expected_output": "VaR（Value at Risk）是一种风险度量方法。",
                "context": {
                    "industry": "finance", 
                    "topic": "risk_models",
                    "max_tokens": 600,
                    "temperature": 0.6,
                    "system_prompt": "你是一个量化金融专家，请详细解释VaR模型的技术细节。"
                }
            },
            {
                "id": "zhipu_test_3",
                "input": "中国金融科技行业的发展趋势和监管政策有哪些？",
                "expected_output": "中国金融科技行业正在快速发展，监管政策也在不断完善。",
                "context": {
                    "industry": "fintech",
                    "topic": "industry_analysis",
                    "max_tokens": 700,
                    "temperature": 0.5,
                    "system_prompt": "你是一个金融科技行业分析师，请提供客观、全面的分析。"
                }
            }
        ]
        print(f"✅ 准备了 {len(test_dataset)} 个智谱模型测试样本")
        
        # 7. 执行评估
        print("\n🎯 开始评估...")
        
        eval_config = EvaluationConfig(
            industry_domain="finance",
            evaluation_dimensions=["knowledge", "terminology"],
            weight_config={"knowledge": 0.7, "terminology": 0.3},
            threshold_config={"knowledge": 0.6, "terminology": 0.5}
        )
        
        # 评估智谱GLM-4.5模型
        print("🔄 评估智谱GLM-4.5模型...")
        zhipu_task_id = evaluation_engine.evaluate_model(
            model_id="glm-4.5",
            dataset=test_dataset,
            evaluation_config=eval_config
        )
        
        # 等待评估完成
        wait_for_completion(evaluation_engine, zhipu_task_id, "智谱GLM-4.5")
        zhipu_result = evaluation_engine.get_evaluation_result(zhipu_task_id)
        
        # 8. 显示结果
        print("\n📊 智谱GLM-4.5评估结果:")
        print("-" * 60)
        print(f"{'指标':<15} {'得分':<10} {'详情':<35}")
        print("-" * 60)
        
        if zhipu_result:
            print(f"{'综合得分':<15} {zhipu_result.overall_score:<10.3f} {'整体表现评估':<35}")
            print(f"{'知识准确性':<15} {zhipu_result.dimension_scores.get('knowledge', 0):<10.3f} {'专业知识掌握程度':<35}")
            print(f"{'术语使用':<15} {zhipu_result.dimension_scores.get('terminology', 0):<10.3f} {'专业术语使用准确性':<35}")
            
            # 显示详细的测试结果
            print(f"\n📋 详细测试结果:")
            for i, sample in enumerate(test_dataset, 1):
                print(f"\n--- 测试样本 {i}: {sample['id']} ---")
                print(f"📝 问题: {sample['input'][:80]}{'...' if len(sample['input']) > 80 else ''}")
                print(f"🎯 领域: {sample['context'].get('industry', 'general')}")
                # 这里可以添加更多详细信息，如果评估结果包含单个样本的详情
        else:
            print("❌ 未获取到智谱模型评估结果")
        
        print("-" * 60)
        
        # 9. 生成报告
        print("\n📄 生成智谱模型评估报告...")
        if zhipu_result:
            report = evaluation_engine.generate_report(zhipu_task_id, "json")
            if report:
                report_file = temp_dir / "zhipu_evaluation_report.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report if isinstance(report, str) else json.dumps(report, ensure_ascii=False, indent=2))
                print(f"✅ 智谱模型评估报告已保存到: {report_file}")
        
        # 10. 显示总结
        print("\n🎉 智谱模型演示完成!")
        print("=" * 60)
        print("✅ 已演示的功能:")
        print("  • 智谱GLM-4.5模型集成")
        print("  • OpenAI兼容API适配器")
        print("  • 金融领域专业评估")
        print("  • 知识准确性评估")
        print("  • 术语使用评估")
        print("  • 详细报告生成")
        print(f"\n📁 查看详细结果: {temp_dir}")
        print(f"\n💡 后续建议:")
        print("  🔧 调整temperature和max_tokens参数优化输出")
        print("  📊 添加更多评估维度进行深度分析")
        print("  🔄 定期运行测试监控模型性能")
        print("  📈 集成到完整的评估流程中")
        
    except Exception as e:
        logger.error(f"❌ 演示过程中发生错误: {str(e)}")
        raise
    
    finally:
        # 清理资源
        try:
            evaluation_engine.shutdown()
        except:
            pass


def wait_for_completion(evaluation_engine, task_id, model_name):
    """等待评估完成"""
    max_wait = 120  # 增加等待时间，因为API调用可能需要更长时间
    start_time = time.time()
    
    print(f"⏳ 等待{model_name}评估完成...")
    
    while time.time() - start_time < max_wait:
        progress = evaluation_engine.get_evaluation_progress(task_id)
        
        if progress:
            if progress.status == "completed":
                print(f"✅ {model_name}评估完成")
                return
            elif progress.status == "failed":
                print(f"❌ {model_name}评估失败")
                if hasattr(progress, 'error_message'):
                    print(f"   错误信息: {progress.error_message}")
                return
            elif progress.status == "running":
                elapsed = time.time() - start_time
                print(f"🔄 {model_name}评估进行中... (已用时: {elapsed:.1f}s)")
        
        time.sleep(2)  # 增加检查间隔
    
    print(f"⏰ {model_name}评估超时 (超过{max_wait}秒)")


def set_api_key_if_needed():
    """如果环境变量中没有API密钥，提供设置选项"""
    if not os.getenv("BIGMODEL_API_KEY"):
        print("🔑 未检测到智谱API密钥环境变量")
        print("请选择设置方式:")
        print("1. 设置环境变量 (推荐): export BIGMODEL_API_KEY=your_api_key_here")
        print("2. 在此处临时设置 (仅用于演示)")
        
        choice = input("\n请输入选择 (1/2): ").strip()
        
        if choice == "2":
            api_key = input("请输入您的智谱API密钥: ").strip()
            if api_key:
                os.environ["BIGMODEL_API_KEY"] = api_key
                print("✅ API密钥已临时设置")
            else:
                print("❌ 未输入有效的API密钥")
                return False
        elif choice == "1":
            print("请在终端中运行: export BIGMODEL_API_KEY=your_api_key_here")
            print("然后重新运行此程序")
            return False
        else:
            print("❌ 无效选择")
            return False
    
    return True


if __name__ == "__main__":
    print("🚀 智谱模型演示程序启动")
    print("=" * 60)
    
    # 检查并设置API密钥
    if set_api_key_if_needed():
        simple_evaluation_demo()
    else:
        print("\n💡 获取智谱API密钥:")
        print("  1. 访问 https://open.bigmodel.cn")
        print("  2. 注册账号并获取API密钥")
        print("  3. 设置环境变量或在程序中输入密钥")