"""
Industry Evaluation System API 演示

这个示例程序展示了如何使用REST API接口进行评估。
"""

import json
import requests
import time
import threading
import sys
from pathlib import Path
import tempfile

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from industry_evaluation.config.config_manager import ConfigManager, ConfigTemplate
from industry_evaluation.api.rest_api import EvaluationAPI


class APIDemo:
    """API演示类"""
    
    def __init__(self):
        """初始化API演示"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "api_config.yaml"
        self.base_url = "http://localhost:5001"
        self.server_thread = None
        self.api_instance = None
        
        print(f"📁 临时目录: {self.temp_dir}")
    
    def setup_api_server(self):
        """设置API服务器"""
        print("🔧 设置API服务器...")
        
        # 创建配置
        config = ConfigTemplate.generate_finance_config()
        ConfigTemplate.save_template(config, self.config_file)
        
        # 创建配置管理器
        config_manager = ConfigManager(self.config_file, auto_reload=False)
        
        # 创建API实例
        self.api_instance = EvaluationAPI(config_manager)
        
        print("✅ API服务器设置完成")
    
    def start_server(self):
        """启动API服务器"""
        print("🚀 启动API服务器...")
        
        def run_server():
            self.api_instance.run(host='127.0.0.1', port=5001, debug=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # 等待服务器启动
        time.sleep(3)
        print(f"✅ API服务器已启动: {self.base_url}")
    
    def test_health_check(self):
        """测试健康检查"""
        print("\n🔍 测试健康检查...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 健康检查成功: {data['status']}")
                print(f"   版本: {data.get('version', 'N/A')}")
                print(f"   时间: {data.get('timestamp', 'N/A')}")
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 健康检查请求失败: {str(e)}")
    
    def test_system_info(self):
        """测试系统信息"""
        print("\n📋 测试系统信息...")
        
        try:
            response = requests.get(f"{self.base_url}/info", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 系统信息获取成功:")
                print(f"   版本: {data.get('version', 'N/A')}")
                print(f"   最大工作线程: {data.get('system', {}).get('max_workers', 'N/A')}")
                print(f"   支持的行业: {', '.join(data.get('industry_domains', []))}")
            else:
                print(f"❌ 系统信息获取失败: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 系统信息请求失败: {str(e)}")
    
    def test_model_management(self):
        """测试模型管理"""
        print("\n🤖 测试模型管理...")
        
        try:
            # 获取模型列表
            response = requests.get(f"{self.base_url}/models", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    models = data['data']
                    print(f"✅ 模型列表获取成功: {len(models)} 个模型")
                    
                    for model in models[:3]:  # 显示前3个模型
                        print(f"   - {model.get('model_id', 'N/A')}: {model.get('adapter_type', 'N/A')}")
                else:
                    print(f"❌ 模型列表获取失败: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ 模型列表请求失败: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 模型管理请求失败: {str(e)}")
    
    def test_evaluation_creation(self):
        """测试评估任务创建"""
        print("\n🎯 测试评估任务创建...")
        
        # 准备评估请求数据
        evaluation_request = {
            "model_id": "finance_gpt4",
            "dataset": [
                {
                    "id": "api_test_1",
                    "input": "什么是金融风险管理？",
                    "expected_output": "金融风险管理是识别、评估和控制金融风险的过程。",
                    "context": {"industry": "finance", "topic": "risk_management"}
                },
                {
                    "id": "api_test_2",
                    "input": "请解释VaR模型的应用",
                    "expected_output": "VaR模型广泛应用于银行和投资机构的风险管理中。",
                    "context": {"industry": "finance", "topic": "risk_models"}
                }
            ],
            "config": {
                "industry_domain": "finance",
                "evaluation_dimensions": ["knowledge", "terminology"],
                "weight_config": {"knowledge": 0.7, "terminology": 0.3},
                "threshold_config": {"knowledge": 0.6, "terminology": 0.5},
                "auto_generate_report": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/evaluations",
                json=evaluation_request,
                timeout=10
            )
            
            if response.status_code == 201:
                data = response.json()
                if data['success']:
                    task_id = data['data']['task_id']
                    print(f"✅ 评估任务创建成功")
                    print(f"   任务ID: {task_id}")
                    print(f"   状态: {data['data']['status']}")
                    return task_id
                else:
                    print(f"❌ 评估任务创建失败: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ 评估任务创建请求失败: {response.status_code}")
                if response.text:
                    print(f"   错误信息: {response.text}")
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ 评估任务创建请求失败: {str(e)}")
        
        return None
    
    def test_evaluation_monitoring(self, task_id: str):
        """测试评估任务监控"""
        if not task_id:
            print("\n⚠️ 跳过评估监控测试（无有效任务ID）")
            return
        
        print(f"\n⏳ 测试评估任务监控 (任务ID: {task_id})...")
        
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.base_url}/evaluations/{task_id}", timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['success']:
                        task_info = data['data']
                        status = task_info['status']
                        progress = task_info.get('progress', 0)
                        
                        print(f"🔄 任务状态: {status}, 进度: {progress:.1%}")
                        
                        if status == "completed":
                            print("✅ 评估任务完成")
                            result = task_info.get('result', {})
                            if result:
                                print(f"   综合得分: {result.get('overall_score', 'N/A')}")
                                print(f"   处理样本数: {result.get('total_samples', 'N/A')}")
                            break
                        elif status == "failed":
                            print("❌ 评估任务失败")
                            break
                    else:
                        print(f"❌ 获取任务状态失败: {data.get('error', 'Unknown error')}")
                        break
                elif response.status_code == 404:
                    print("❌ 任务不存在")
                    break
                else:
                    print(f"❌ 获取任务状态请求失败: {response.status_code}")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ 监控请求失败: {str(e)}")
                break
            
            time.sleep(2)
        else:
            print("⏰ 评估监控超时")
    
    def test_evaluation_list(self):
        """测试评估任务列表"""
        print("\n📋 测试评估任务列表...")
        
        try:
            response = requests.get(f"{self.base_url}/evaluations", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    evaluations = data['data']
                    print(f"✅ 评估任务列表获取成功: {len(evaluations)} 个任务")
                    
                    for eval_task in evaluations[:3]:  # 显示前3个任务
                        print(f"   - {eval_task.get('task_id', 'N/A')}: {eval_task.get('status', 'N/A')}")
                else:
                    print(f"❌ 评估任务列表获取失败: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ 评估任务列表请求失败: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 评估任务列表请求失败: {str(e)}")
    
    def test_configuration_management(self):
        """测试配置管理"""
        print("\n⚙️ 测试配置管理...")
        
        try:
            # 获取配置
            response = requests.get(f"{self.base_url}/config", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    config = data['data']
                    print(f"✅ 配置获取成功:")
                    print(f"   版本: {config.get('version', 'N/A')}")
                    print(f"   最大工作线程: {config.get('system', {}).get('max_workers', 'N/A')}")
                    print(f"   模型数量: {len(config.get('models', {}))}")
                    print(f"   评估器数量: {len(config.get('evaluators', {}))}")
                else:
                    print(f"❌ 配置获取失败: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ 配置获取请求失败: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 配置管理请求失败: {str(e)}")
    
    def test_file_upload(self):
        """测试文件上传"""
        print("\n📤 测试文件上传...")
        
        # 创建测试数据集文件
        test_dataset = [
            {"input": "测试问题1", "expected_output": "测试答案1"},
            {"input": "测试问题2", "expected_output": "测试答案2"}
        ]
        
        test_file = self.temp_dir / "test_dataset.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, ensure_ascii=False, indent=2)
        
        try:
            with open(test_file, 'rb') as f:
                files = {'file': ('test_dataset.json', f, 'application/json')}
                response = requests.post(
                    f"{self.base_url}/upload/dataset",
                    files=files,
                    timeout=10
                )
            
            if response.status_code == 201:
                data = response.json()
                if data['success']:
                    upload_info = data['data']
                    print(f"✅ 文件上传成功:")
                    print(f"   文件名: {upload_info.get('filename', 'N/A')}")
                    print(f"   文件大小: {upload_info.get('file_size', 'N/A')} 字节")
                else:
                    print(f"❌ 文件上传失败: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ 文件上传请求失败: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 文件上传请求失败: {str(e)}")
    
    def run_api_demo(self):
        """运行API演示"""
        print("🌐 Industry Evaluation System - API 演示")
        print("=" * 60)
        
        try:
            # 设置并启动API服务器
            self.setup_api_server()
            self.start_server()
            
            # 运行各种API测试
            self.test_health_check()
            self.test_system_info()
            self.test_model_management()
            
            # 创建评估任务并监控
            task_id = self.test_evaluation_creation()
            self.test_evaluation_monitoring(task_id)
            
            self.test_evaluation_list()
            self.test_configuration_management()
            self.test_file_upload()
            
            print("\n🎉 API演示完成!")
            print("=" * 60)
            print("✅ 已测试的API端点:")
            print("  • GET  /health - 健康检查")
            print("  • GET  /info - 系统信息")
            print("  • GET  /models - 模型列表")
            print("  • POST /evaluations - 创建评估任务")
            print("  • GET  /evaluations/{id} - 获取评估状态")
            print("  • GET  /evaluations - 评估任务列表")
            print("  • GET  /config - 获取配置")
            print("  • POST /upload/dataset - 上传数据集")
            
            print(f"\n📁 API文档: {self.base_url}/docs/")
            print(f"📁 临时文件: {self.temp_dir}")
            
        except Exception as e:
            print(f"❌ API演示过程中发生错误: {str(e)}")
            raise
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.api_instance:
                # 这里可以添加API实例的清理逻辑
                pass
        except Exception as e:
            print(f"⚠️ 清理过程中发生错误: {str(e)}")


def main():
    """主函数"""
    demo = APIDemo()
    
    try:
        demo.run_api_demo()
        
        # 保持服务器运行一段时间以便手动测试
        print(f"\n💡 服务器将继续运行30秒，您可以手动测试API:")
        print(f"   curl {demo.base_url}/health")
        print(f"   curl {demo.base_url}/info")
        print("   按 Ctrl+C 提前退出")
        
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示失败: {str(e)}")
    finally:
        demo.cleanup()
        print("🧹 演示结束")


if __name__ == "__main__":
    main()