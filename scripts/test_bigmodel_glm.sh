#!/bin/bash

# BigModel GLM API 快速测试脚本
# 
# 使用方法:
#   ./scripts/test_bigmodel_glm.sh YOUR_API_KEY
#   ./scripts/test_bigmodel_glm.sh YOUR_API_KEY glm-4.5
#   ./scripts/test_bigmodel_glm.sh YOUR_API_KEY glm-4.5 --verbose

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查参数
if [ $# -lt 1 ]; then
    print_error "缺少API密钥参数"
    echo
    echo "使用方法:"
    echo "  $0 YOUR_API_KEY [MODEL_NAME] [OPTIONS]"
    echo
    echo "示例:"
    echo "  $0 your_api_key_here"
    echo "  $0 your_api_key_here glm-4.5"
    echo "  $0 your_api_key_here glm-4.5 --verbose"
    echo
    echo "获取API密钥:"
    echo "  访问 https://open.bigmodel.cn 注册账号并获取API密钥"
    exit 1
fi

API_KEY="$1"
MODEL_NAME="${2:-glm-4.5}"
EXTRA_ARGS="${@:3}"

# 检查Python环境
print_info "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    print_error "未找到python3命令"
    exit 1
fi

# 检查项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_ROOT/examples/bigmodel_glm_demo.py" ]; then
    print_error "未找到演示脚本: $PROJECT_ROOT/examples/bigmodel_glm_demo.py"
    exit 1
fi

print_success "Python环境检查通过"

# 设置环境变量
export BIGMODEL_API_KEY="$API_KEY"

print_info "开始测试 BigModel $MODEL_NAME API..."
echo

# 运行Python演示脚本
cd "$PROJECT_ROOT"

python3 examples/bigmodel_glm_demo.py \
    --model "$MODEL_NAME" \
    --show-curl \
    $EXTRA_ARGS

exit_code=$?

echo
if [ $exit_code -eq 0 ]; then
    print_success "测试完成！"
    echo
    print_info "更多测试选项:"
    echo "  --verbose     显示详细输出"
    echo "  --output FILE 保存结果到文件"
    echo "  --show-curl   显示等效curl命令"
    echo
    print_info "其他工具:"
    echo "  python3 tools/test_openai_compatible_api.py --help"
    echo "  python3 examples/openai_compatible_api_test.py --help"
else
    print_error "测试失败，退出码: $exit_code"
fi

exit $exit_code