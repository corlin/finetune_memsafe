"""
测试配置文件

定义测试运行的配置参数和常量。
"""

import os
from pathlib import Path

# 测试目录配置
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
TEMP_TEST_DIR = TEST_DIR / "temp"

# 测试数据配置
TEST_DATA_DIR = TEMP_TEST_DIR / "test_data"
TEST_OUTPUT_DIR = TEMP_TEST_DIR / "test_output"

# 模型配置（用于测试）
TEST_MODEL_NAME = "test-model"
TEST_MAX_MEMORY_GB = 8.0
TEST_BATCH_SIZE = 1
TEST_MAX_SEQUENCE_LENGTH = 128

# 测试QA数据
TEST_QA_DATA = [
    {
        "question": "什么是机器学习？",
        "answer": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。"
    },
    {
        "question": "什么是深度学习？", 
        "answer": "深度学习是机器学习的一个子集，使用多层神经网络来学习复杂的数据表示。"
    },
    {
        "question": "什么是自然语言处理？",
        "answer": "自然语言处理是计算机科学和人工智能的一个分支，专注于计算机与人类语言的交互。"
    }
]

# 测试环境配置
SKIP_CUDA_TESTS = not os.environ.get("RUN_CUDA_TESTS", "false").lower() == "true"
SKIP_SLOW_TESTS = os.environ.get("SKIP_SLOW_TESTS", "true").lower() == "true"

# 日志配置
TEST_LOG_LEVEL = "WARNING"  # 减少测试输出

# 内存测试配置
MOCK_GPU_MEMORY_GB = 16.0
MOCK_ALLOCATED_MEMORY_GB = 8.0
MOCK_CACHED_MEMORY_GB = 10.0

def setup_test_environment():
    """设置测试环境"""
    # 创建临时测试目录
    TEMP_TEST_DIR.mkdir(exist_ok=True)
    TEST_DATA_DIR.mkdir(exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

def cleanup_test_environment():
    """清理测试环境"""
    import shutil
    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR)

# 测试标记
UNIT_TEST_MARK = "unit"
INTEGRATION_TEST_MARK = "integration"
SLOW_TEST_MARK = "slow"
CUDA_TEST_MARK = "cuda"