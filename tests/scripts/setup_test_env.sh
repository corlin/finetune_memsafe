#!/bin/bash
# 测试环境设置脚本

set -e

echo "🚀 设置评估系统测试环境..."

# 检查uv是否已安装
if ! command -v uv &> /dev/null; then
    echo "📦 安装uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "✅ uv版本: $(uv --version)"

# 创建虚拟环境并安装依赖
echo "📦 安装测试依赖..."
uv sync --extra dev

# 验证安装
echo "🔍 验证测试环境..."
uv run python -c "import pytest; print(f'pytest版本: {pytest.__version__}')"
uv run python -c "import numpy; print(f'numpy版本: {numpy.__version__}')"
uv run python -c "import pandas; print(f'pandas版本: {pandas.__version__}')"

# 创建测试结果目录
mkdir -p test_results

echo "✅ 测试环境设置完成！"
echo ""
echo "运行测试："
echo "  uv run pytest tests/"
echo "  uv run python tests/test_runner.py --type all"
echo ""
echo "查看帮助："
echo "  uv run python tests/test_runner.py --help"