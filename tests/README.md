# 评估系统测试套件

本目录包含了Qwen3优化微调系统数据拆分和评估功能的完整测试套件。

## 测试结构

### 单元测试
- `test_data_splitter.py` - 数据拆分器测试
- `test_evaluation_engine.py` - 评估引擎测试
- `test_metrics_calculator.py` - 指标计算器测试
- `test_quality_analyzer.py` - 质量分析器测试
- `test_benchmark_manager.py` - 基准管理器测试
- `test_experiment_tracker.py` - 实验跟踪器测试
- `test_report_generator.py` - 报告生成器测试
- `test_efficiency_analyzer.py` - 效率分析器测试
- `test_config_manager.py` - 配置管理器测试

### 集成测试
- `test_integration_evaluation.py` - 评估系统集成测试

### 性能测试
- `test_performance_benchmarks.py` - 性能基准测试

### 测试配置
- `conftest.py` - 测试配置和夹具
- `pytest.ini` - pytest配置
- `requirements-test.txt` - 测试依赖
- `test_runner.py` - 测试运行器

## 运行测试

### 安装测试依赖

使用uv安装依赖（推荐）：
```bash
# 安装uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装测试依赖
uv sync --extra dev --extra gpu  # 包含开发和GPU依赖
# 或者只安装基础测试依赖
uv sync
```

或使用传统pip方式：
```bash
pip install -r tests/requirements-test.txt
```

### 运行所有测试

使用uv运行：
```bash
uv run python tests/test_runner.py --type all
```

或使用pytest直接运行：
```bash
uv run pytest tests/
```

### 运行特定类型的测试
```bash
# 单元测试
uv run python tests/test_runner.py --type unit

# 集成测试
uv run python tests/test_runner.py --type integration

# 性能测试
uv run python tests/test_runner.py --type performance
```

### 运行特定测试文件
```bash
uv run python tests/test_runner.py --file test_data_splitter.py
# 或
uv run pytest tests/test_data_splitter.py
```

### 运行特定测试函数
```bash
uv run python tests/test_runner.py --file test_data_splitter.py --function test_split_data_basic
# 或
uv run pytest tests/test_data_splitter.py::TestDataSplitter::test_split_data_basic
```

### 重新运行失败的测试
```bash
uv run python tests/test_runner.py --failed
# 或
uv run pytest --lf
```

### 运行带特定标记的测试
```bash
uv run python tests/test_runner.py --markers performance
# 或
uv run pytest -m performance
```

### 生成覆盖率报告
```bash
uv run python tests/test_runner.py --coverage
# 或
uv run pytest --cov=evaluation --cov-report=html
```

## 测试覆盖范围

### 数据拆分功能测试
- ✅ 基本数据拆分功能
- ✅ 分层抽样
- ✅ 数据分布一致性验证
- ✅ 数据泄露检测
- ✅ 可重现性测试
- ✅ 边界条件处理
- ✅ 错误处理

### 评估引擎测试
- ✅ 多任务评估
- ✅ 批量推理
- ✅ 并行评估
- ✅ 错误恢复
- ✅ 内存优化
- ✅ 配置管理

### 指标计算测试
- ✅ BLEU分数计算
- ✅ ROUGE分数计算
- ✅ BERTScore计算
- ✅ 分类指标计算
- ✅ 语义相似度计算
- ✅ 统计显著性检验

### 质量分析测试
- ✅ 数据质量分析
- ✅ 响应质量分析
- ✅ 问题检测
- ✅ 改进建议
- ✅ 报告生成

### 基准管理测试
- ✅ 基准数据集加载
- ✅ CLUE评估
- ✅ FewCLUE评估
- ✅ C-Eval评估
- ✅ 自定义基准
- ✅ 结果对比

### 实验跟踪测试
- ✅ 实验记录
- ✅ 实验查询
- ✅ 实验对比
- ✅ 排行榜生成
- ✅ 结果导出
- ✅ 历史管理

### 报告生成测试
- ✅ HTML报告生成
- ✅ JSON报告生成
- ✅ CSV导出
- ✅ Excel导出
- ✅ LaTeX表格生成
- ✅ 图表生成
- ✅ 多语言支持

### 效率分析测试
- ✅ 推理延迟测量
- ✅ 吞吐量测量
- ✅ 内存使用监控
- ✅ FLOPs计算
- ✅ 模型大小统计
- ✅ 批处理效率分析
- ✅ 性能对比

### 配置管理测试
- ✅ YAML配置加载
- ✅ JSON配置加载
- ✅ 配置验证
- ✅ 配置合并
- ✅ 环境变量替换
- ✅ 配置继承
- ✅ 备份恢复

### 集成测试
- ✅ 完整评估流水线
- ✅ 基准评估工作流
- ✅ 多模型对比工作流
- ✅ 数据质量集成
- ✅ 训练监控集成
- ✅ 错误恢复测试
- ✅ 并行处理测试

### 性能测试
- ✅ 数据拆分性能
- ✅ 指标计算性能
- ✅ 评估引擎性能
- ✅ 批处理效率
- ✅ 并行评估性能
- ✅ 内存效率
- ✅ 实验跟踪性能
- ✅ 报告生成性能
- ✅ 扩展性测试

## 测试数据

测试使用模拟数据和夹具，包括：
- 样本数据集（不同大小）
- 模拟模型和分词器
- 临时目录和文件
- 配置文件模板
- 预定义的测试常量

## 测试报告

测试运行后会在 `test_results/` 目录生成以下报告：
- `test_report.html` - HTML测试报告
- `test_results.xml` - JUnit XML报告
- `coverage_html/` - HTML覆盖率报告
- `coverage.xml` - XML覆盖率报告
- `test_summary.json` - 测试摘要
- `test_matrix.json` - 测试矩阵

## 持续集成

测试套件支持在CI/CD环境中运行：
- 支持并行测试
- 生成标准格式报告
- 覆盖率检查
- 性能回归检测

## 故障排除

### 常见问题

1. **导入错误**
   - 确保已安装所有依赖包
   - 检查Python路径配置

2. **测试超时**
   - 调整pytest.ini中的timeout设置
   - 检查系统资源使用情况

3. **内存不足**
   - 减少大数据集测试的样本数量
   - 使用更小的批次大小

4. **GPU测试失败**
   - 确保有可用的GPU
   - 安装正确的CUDA版本

### 调试技巧

1. **运行单个测试**
   ```bash
   uv run pytest tests/test_data_splitter.py::TestDataSplitter::test_split_data_basic -v
   ```

2. **查看详细输出**
   ```bash
   uv run pytest tests/test_data_splitter.py -v -s
   ```

3. **使用调试器**
   ```bash
   uv run pytest tests/test_data_splitter.py --pdb
   ```

4. **查看覆盖率详情**
   ```bash
   uv run pytest tests/ --cov=evaluation --cov-report=term-missing
   ```

## 贡献指南

### 添加新测试

1. 在相应的测试文件中添加测试函数
2. 使用描述性的函数名（以`test_`开头）
3. 添加适当的文档字符串
4. 使用合适的断言
5. 添加必要的标记（@pytest.mark.xxx）

### 测试最佳实践

1. **独立性** - 每个测试应该独立运行
2. **可重复性** - 测试结果应该一致
3. **清晰性** - 测试意图应该明确
4. **完整性** - 覆盖正常和异常情况
5. **效率性** - 避免不必要的计算

### 代码风格

- 遵循PEP 8代码风格
- 使用有意义的变量名
- 添加适当的注释
- 保持函数简洁

## 许可证

本测试套件遵循与主项目相同的许可证。