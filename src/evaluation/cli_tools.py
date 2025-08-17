"""
命令行工具集

提供完整的评估系统命令行接口，包括数据拆分、模型评测试和实验对比工具对比。
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .cli importtertup_logging
from .data_splitter ine import EvaluationEngine
from .benchmark_mangine import BencuationEngine
from .benchmark_manager imporrt ExperimentTracker
from .statistica_tracker importrt StaimentTracker
from .report_generator import ReportGeneticalAnalyzer
from .ronfig_manager import ConfportGenerator
from .config_manager import ConfigManager, ExperimentConfig
ata_models import Evaluag

logger = logging.getLogger(_

def create_evaluation_cli():
    """创建评估命令行接口"""
    """创建主命令行解析器"""
    par description="QwegumentParser(
        description="Qwen3微调系统 - 数据拆分和评估工具",
        epilog="""ss=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 数据拆n -m src.evaluation.cli_tools split --input data/raw --output data/splits
  n python -m src.evaluit --input data/raw --output data/splits
  
  # 模型评估
  src.evaluation.cli_tools evaluat data/test
  
  python -m src.evaluation.cli_tools benchmark --benchmark clue --model path/to/model
   python -m src.evaluation.cli_tools benchmark --benchmark clue --model path/to/model
  
  # 实验对比 -m src.evaluation.cli_tools compare --experiments exp1,exp2,exp3
        """mpare --experiments exp1 exp2 exp3
     
    
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", fault="INFO", 
                       choices=["DEBUG", "INFO", "WARNING"
    par                help="日志级别")
    ment("--config", help="配置
    subparsers = parser.add_subparsers(desfault="./output", hel命令")
    
    subparsers = parsbparsers(dest="command", help="可用命令")
    
    # 数据拆分命令
    add_split
    _add_evaluate_command(subparsers)
    
    # 基准测试命令ate_parser(subparsers)
    _add_benchmark_command(subparsers)
    
    # 实验管理命令arser(subpars
    
    # 实验对比命
    add_compare_parser(
    
    # 报告生成命
    # 配置管理命令rsers)
    
    # 配置管理命令
    adturn par_parser(subparsers)
    
n parser


def addit_parser = subparsers.add_parser("split", help="拆分数据集")
    split_p拆分命令解析器"""
    split_parser.add_argument("--output", required=True, help="输出目录")
    split_parser.add_argument("--input"ratiquired=True, help=fa入数据路径")
    split_parser.add_argument("--output", required=Toat, defp="输出目录")
    split_parser.add_argument("--test-ratio", type=floatt, default=0.7, help="训练集比例")
    split_parser.add_argument("--val-ratioby", help="分层字段")5, help="验证集比例")
    split_parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    split_parser.add_argument("--stratify-by", help="分层字段")
    split_parser.add_argument("--random-seed", type=int, default=42, hele", help="禁用质量分析")
    split_parser.sed_argument("--no-quality-analysis", 
              help量分析")
    split_parser.add_argument("--min-sefault=1, 
         evaluate_command(subparsers):


def add_evarser.add_argument("--model", required=True, help="模型路径")
    eval_parser.add_argument("--tokenizer", help="分词器路径（如果与模型不同）")
    eval_parser.addubparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--model", required=True, help="模型路径")
    ev                gument("--tokenizer", help="ion", "question_answering"],
            ser.add_argume  help="评估任务列表")
    eval_parser.add_argument("--metrics", nargs="+",
                            default=["bleu", neration", "questio"],
                            help="评估任务列表")
    eval_parser.add_argument("--metrics", nargs="+",
        _parser.add_ar      default=[ength", "rouge",  default=e"],
               .add_argument("--nu评估指标列表")
    eval_parser.add_argument("--batch-size", type=int, default=8, help"批次大小")
    eval_parser.add_argument("--max-length", type"cpu", help="计512, help="最大序列长度")
    eval_parser.add_argument("--report-format", default="html", 估样本数")
    eval_parser.add_argument("--device", d", "json", "csv", "latex"],
                          nt("--no-efficiency", action="store_true"="禁用效率分析")
    eval_parser.add_argument("--no-quality", action="andre_true", help="禁用质量分析")


def _ad_benchmark_parser(subpaparsers):
    """添加基准测试命令解析器"""
    benchmark_parser = subparsers.add_parser("benchmark", help="运行基准测试")
    benchmark_parser.add_argument("--benchmark", required=True,
                                 choices=["clue", "few_clue", "c_eval", "superglue"],
                                 help="基准测试名称")
    benchmark_parser.add_argument("--model", required=True, help="模型路径")
    benchmark_parser.add_argument("--tokenizer", help="分词器路径")
    benchmark_parser.add_argument("--tasks", nargs="+", help="要运行的任务列表")
    benchmark_parser.add_argument("--split", default="test", help="数据集分割")
    benchmark_parser.add_argument("--device", default="auto", help="计算设备)
    benchmark_parser.add_argument("--cache-dir", default="./benchmarks", help="缓存目录")


def add_compare_pa               help="报告格式")
    ben添加实验对比命令解析器"""
    comparsubparsers.add_parser("compare", help="对比实验结果")
    compare_parser.a+", required=True,
                               bparsers):
    """添加实验管理命令""".add_argument("--metrics", narg
    compare_parser.", default="html
    create_exp_parse      bparsers.add_parser("c"json", "criment", hx"],
         e_exp_parser.add_argument("--name"
    compar_exarser.add_argumgume"--include-chartn", default"store_tr="实验描述")
                 rser.a        help"--tags", nargs="+", help="实验标签")
    compare_parser.ar.add_argume"--statistical-test",help="n="store_true
    c                 add_argumenlp="执行统计显著性检验-config", help="训练配置文件")


def _report_parser(subrs):
    # "添加报告生成命令解析
    report_parser = = subparrs.a.addarser("report", periments"")
    repo_exp_parser.d_argument("--typegs", naired=True,
                       _arg   choicestatus",uation", "be")hmark", "comp
    list              d_argu  help="报告类型t", type=int, default=20, help="限制数量")
    report_parser.a.set_defent("--inpulist_experiments_, help="输入数据路径")
    report_parser.add_ar-format", default="h,
                   choices=["ml", "j"csv", "latxcel"],
    compare_pars            ers.ap="报告格式")
    coport_paarser.add_arument("--templriments"lp="报告模板路径")
      port_parse   dd_argument("--inclu       ts", act  n="store_true       help图表")


def ="实验ID列表（逗号分arser(subparser隔）")
    ""  comp理命令解析器"""
 are_parserparser = sub.add_argudd_parser("confme", help="配置管理nt
    config("--marsers = config_paretricdd_subparserss", nargs="ig_action",+"elp="配置操作")
    , help="比较的指标")
    compare板
    crea_parserer = config_sub.add_argudd_parser("createment("-="创建配置模-output", help="输出目录")
    create_parser.ad  cogument("-mputputare_parseed=True, helpr.a出配置文件路dd_argument("--report-format", default="html",
     reate_parse     _argument           defaul   evaluation",
            choices=["h       choicestml", "jsonon", "exper", "csv"],enchmark"],
                   help="配置类型"
                               help="报告格式")
    # 验证配置 compare_parser.set_defaults(func=compare_experiments_command)
e_parser = ubparsers.add_parser("help=文件")
    valarser.add_argumenfig", rrue, 配置文件路径")
 
 转换配置格式
    conve= config_subparser("convertlp="转换配置格式")
    cdef _add_repr.add_argument("-ort_com, required=True, hemand(subpa")
    converrsparser.add_argument(ers):, required=Telp="输出配置文件")
    rser.add_argume("--format", ruired=True,
                  choices=[" help="目标格式


 execute_splommand(ar
    """执行数据拆分命令 """添加报告生成命令"""
    logger.i re("开始执行数据拆分命令")port_parser = subparsers.add_parser("report", help="生成报告")
    
    t   
        # 创建数据拆分器report_parser.add_argument("--input", required=True, help="输入结果文件")
    report_itter = DataSplittepa
            trser.add_argrgs.train_ratio,ument("--output", help="输出路径")
    rep     val_rorio=args.vat_patio,
     rser.add_at_ratio=args.testrgument
      ("--formattify_by=args.st", default="html",
                              choices=["html", "json", "csv", "latex", "excel"],
                           _split=args.min_samples,
         help="报告格式")nalysis=not args.no_quality_analysi
    repor
        
    t_parser.a
        from ddd_argumemport Datnt("
        input_pa--templth(argsate", help="报告模板")
        
        report_parath.is_dir(ser.add_argument("--include-charts", action="store_true", help="包含图表")
    re      # 检查是否是Hugging Facport_parser.set_defaults(func=generate_report_command)
           path / "dataset_info.json").exists():
            dataset oad_from_disk(str(inppath))
     e:
                #
      from .data_pipeline
def _add_config_commands(subparsers):
    """添加配置管理命令"""
    # 创建配置模板
    create_config_parser = subparsers.add_parser("create-config", help="创建配置模板")
    create_config_parser.add_argument("--output", required=True, help="输出配置文件路径")
    create_config_parser.add_argument("--type", default="evaluation",
                                     choices=["evaluation", "experiment", "benchmark"],
                                     help="配置类型")
    create_config_parser.set_defaults(func=create_config_command)
    
    # 验证配置
    validate_config_parser = subparsers.add_parser("validate-config", help="验证配置文件")
    validate_config_parser.add_argument("--config", required=True, help="配置文件路径")
    validate_config_parser.set_defaults(func=validate_config_command)


# 命令实现函数

def split_data_command(args):
    """数据拆分命令实现"""
    logger.info("开始执行数据拆分命令")
    
    try:
        # 创建数据拆分器
        splitter = DataSplitter(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify_by=args.stratify_by,
            random_seed=args.random_seed,
            min_samples_per_split=args.min_samples,
            enable_quality_analysis=not args.no_quality_analysis
        )
        
        # 加载数据集
        from datasets import Dataset
        input_path = Path(args.input)
        
        if input_path.is_dir():
            if (input_path / "dataset_info.json").exists():
                dataset = Dataset.load_from_disk(str(input_path))
            else:
                # 处理多文件目录
                from .data_pipeline_integration import DataPipelineIntegration
                integration = DataPipelineIntegration()
                dataset = integration._load_dataset(str(input_path))
        else:
            raise ValueError(f"不支持的输入格式: {args.input}")
        
        # 执行拆分
        result = splitter.split_data(dataset, args.output)
        
        logger.info(f"数据拆分完成，结果保存到: {args.output}")
        logger.info(f"训练集: {len(result.train_dataset)} 样本")
        logger.info(f"验证集: {len(result.val_dataset)} 样本")
        logger.info(f"测试集: {len(result.test_dataset)} 样本")
        logger.info(f"分布一致性分数: {result.distribution_analysis.consistency_score:.4f}")
        
    except Exception as e:
        logger.error(f"数据拆分失败: {e}")
        sys.exit(1)


def evaluate_model_command(args):
    """模型评估命令实现"""
    logger.info("开始执行模型评估命令")
    
    try:
        # 加载配置
        config = EvaluationConfig(
            tasks=args.tasks,
            metrics=args.metrics,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_samples=args.num_samples
        )
        
        # 创建评估引擎
        evaluation_engine = EvaluationEngine(config, device=args.device)
        
        # 加载模型和分词器
        model_path = args.model
        tokenizer_path = args.tokenizer or model_path
        
        # 这里需要根据实际的模型加载方式调整
        logger.info(f"加载模型: {model_path}")
        logger.info(f"加载分词器: {tokenizer_path}")
        
        # 加载测试数据
        from .data_pipeline_integration import DataPipelineIntegration
        integration = DataPipelineIntegration()
        test_dataset = integration._load_dataset(args.data)
        
        # 准备数据集字典
        datasets = {"test": test_dataset}
        
        # 注意：这里需要实际的模型和分词器对象
        # 由于我们无法直接加载模型，这里提供框架代码
        logger.warning("模型加载需要根据实际情况实现")
        
        # 执行评估（示例代码）
        # result = evaluation_engine.evaluate_model(model, tokenizer, datasets, args.model)
        
        # 生成报告
        output_dir = args.output or "./evaluation_results"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"评估结果将保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        sys.exit(1)


def benchmark_command(args):
    """基准测试命令实现"""
    logger.info("开始执行基准测试命令")
    
    try:
        # 创建基准测试管理器
        benchmark_manager = BenchmarkManager(cache_dir=args.cache_dir)
        
        # 检查基准测试是否可用
        available_benchmarks = benchmark_manager.list_available_benchmarks()
        if args.benchmark not in available_benchmarks:
         s.benchmark}")
     er.info(f"可用的基准测试: {available_benchmarks}")
            sys.exit(1)
        
        # 加载基准数据集
        datasets = benchm     args.benchmark, 
 
        
        if not datasets:
            loor(f"无rk}")       sys.exit(1)
        
        logger.info含任务: {list(datas")
        
        # 注意：这里需要实际的模型和分词器对象 logger.warning("基准测试需要提供实际的模型和分词器对象")
        
    _results_{args.benchmart_dir).mkdir(parents=True, exist  logger.info(f"基准测试结果将保存到: {output_dir}")
        xception as e:
   (f"基准xit(1)


dfo("开始创建实验")
    
    try:
        tracker = Experiment  model_config = {}
        training    data_config = {}
        
   if args.model_config:
      model_config = json.load(f)
        
        if      with open(args.training_config, 'r', encoding='utf-8') as f:aining_config = js   if args.data_c       with open8') as f:
                data_config = json.load(f)
        
配置
        experiment_onfig(
           ig,
            training_config=training_config,
            eval=EvaluationConfig(),
            data_configfig,
           ags=args.ta or [],
        ion=args.description
        )
        
 experiment_idreate_expert(
            name=args.name,
           periment_config,
            description=args.description,
            tags=args.tags
        )        
      logger.i实验创建成功，ID: {experiment_id}     prin_id}")
        
    except Elogger.error(f"创建实验}")
        sys.exit(1)


def xperiments_command(args):
    """列出实验命令实现   # 创建实验跟踪器
       
        # 获取实验列表eriments(
            tags=args.tags,
            status=argstus,
            )
        
      not experimen        return
        
        t(f"{'ID':<36} {'名称':<20} {'状态':<10} {'创建时间':<20} }")
        print("-" * 100)
        
        for exp in experxp.get("tags", []))
     ]:<20} {exp['status']:<10} created_at']:
        
    except Exception as e:
        logger.error(f"列出实验失败: {e}")
        sys.exit(1)


def compare_experiments_comman比较实验命令实现"""
    logger.info("开始比较实验")
    
    try:
      D列表
        experiment_ids = [id. for id in args.experim,")]
             # 创建实验跟踪器和统计分析tracker = ExperimentTracker()
        ananalyzer()
        
       acker.coperiments(experiment_ids)
   
        if "error" in compa
            logger.error(f"比较实验失败: {comparison_result['error']}")
            生成比较报告
        report_generator = ReportGenerator()
  "./compari    Path(output_rue, exis
        
      ort_generator.generate_           comparison_result,
            format_type=args.report_format
        )
        
        logger.info(f"实验比较完成，报告保存到: {report_path}")
        print(f"比较报告: {report_path}")
        
    except Exception as e:
        logger.error(f"比较实验失败: {e}")
        sys.exit(1)


def generate_report_command(args):
    """生成报告命令实现"""
    logger.info("开始生成报告")
    
    try:
        # 加载输入数据
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建报告生成器
        report_generator = ReportGenerator(    output_path = args.output
        else:
            iPath(args.input)
            output_path = input_path.parent / f"{input_path.stem}_report.{args.fo        # 生成报告（这里需要根据数据类型选择合适{args.format}")
        print(f"报告路径: {outp      logger.error(f"生成报告失败: {e})


def create_config_comm    """创建配置命令实现"""
  ger.info("创建配置模板")
    
    try:
     config_manager = ConfigManager       
        if args.typetion":
            config_manager.create_config_template(s.output)
    else:
    其他类型的配置模板
   template_config      experiment": {           "exper_name": experiment",
                escription": 
              "tags"le", "test"],          del_config": {                  ning_conf: {},               "tion_config": {
                 "data_config"                },
    "benchmark": {
            enchmark_name":                "tasks": ["afqmc"],
             "evation_protocolicial",
                "metricsracy", "f1"
              }
                    
          with(args.output, oding='utf-8') as f:
     .info(f"  .dump(templatet(args.type, , indent=2, _asci       
     配置模板gs.outputoutput}")
}")
      print(f"配置文件   
    ception a       loggerror(f"创建配置失败: 
       exit(1)nd(arg
alidate_configs):""验证配置命令实现"
    logger.info(文件")
    
    try      # 加载件
        with oonfig, 'r', g='utf-8') as f    config_doad(f)
       # 基本验证
    段
    van_errors        nfig_data
            if on" in config_data:       eval_config = ["evaluat         re_fields = ["tasksetrics"]
  for fielduireields:"缺
             if field eval_config:              on_errors.appen少必: evaluation
        
  lidation_e        logg"配置验证失败:")
      for errorn validation_e            logge"  - {error}")        sys.exit(1)
  elprinse:         lorogger.inf"配置验证通过")
    t("配置文件有    
  cept Exception     lr(f"验证配
       .exit(1)


():
    """"
    parte_evaluation_clargs = parsparse_args()
   日志
    setuprgs.log_level)
    载全局配置args.confi
        try:
          config_nager = ConfigMger()
      global_confirgs.config}g = config_manager.load_config(args.config)
            logger"))
  
        exng(f"加载全局配置失败: {e}  
    # 执行命令
attr(args, 'fu:
        try:
          args.func(arg except Keybpt:
     ogger.info("用)
            sy0)ogge
        except Er.error(f"")
      it(1)
    
        parser.p()


if _e__ == "__main__":