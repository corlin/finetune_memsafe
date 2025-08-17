#!/bin/bash
# 测试运行脚本

set -e

# 默认参数
TEST_TYPE="all"
COVERAGE=true
VERBOSE=true
PARALLEL=false
MARKERS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --markers)
            MARKERS="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --type TYPE        测试类型 (all|unit|integration|performance)"
            echo "  --no-coverage      不生成覆盖率报告"
            echo "  --quiet           静默模式"
            echo "  --parallel        并行运行测试"
            echo "  --markers MARKERS 运行特定标记的测试"
            echo "  --help            显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

echo "🧪 运行评估系统测试..."
echo "测试类型: $TEST_TYPE"
echo "覆盖率: $COVERAGE"
echo "详细输出: $VERBOSE"
echo "并行运行: $PARALLEL"

# 构建pytest命令
PYTEST_ARGS="tests/"

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -v"
else
    PYTEST_ARGS="$PYTEST_ARGS -q"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=evaluation --cov-report=html:test_results/coverage_html --cov-report=xml:test_results/coverage.xml --cov-report=term-missing"
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -n auto"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m $MARKERS"
fi

# 添加输出格式
PYTEST_ARGS="$PYTEST_ARGS --junitxml=test_results/test_results.xml --html=test_results/test_report.html --self-contained-html"

# 根据测试类型调整参数
case $TEST_TYPE in
    unit)
        PYTEST_ARGS="$PYTEST_ARGS -m 'not integration and not performance'"
        ;;
    integration)
        PYTEST_ARGS="$PYTEST_ARGS -m integration"
        ;;
    performance)
        PYTEST_ARGS="$PYTEST_ARGS -m performance"
        ;;
    all)
        # 运行所有测试
        ;;
    *)
        echo "❌ 无效的测试类型: $TEST_TYPE"
        echo "支持的类型: all, unit, integration, performance"
        exit 1
        ;;
esac

# 创建结果目录
mkdir -p test_results

# 运行测试
echo "🚀 执行命令: uv run pytest $PYTEST_ARGS"
echo ""

if uv run pytest $PYTEST_ARGS; then
    echo ""
    echo "✅ 测试完成！"
    echo "📊 查看报告:"
    echo "  HTML报告: test_results/test_report.html"
    if [ "$COVERAGE" = true ]; then
        echo "  覆盖率报告: test_results/coverage_html/index.html"
    fi
else
    echo ""
    echo "❌ 测试失败！"
    echo "📋 查看详细报告: test_results/test_report.html"
    exit 1
fi