#!/usr/bin/env python3
"""
验证框架: 确保每次修改都深入实际运行路径，而非停留在 import 或表面。

三步验证流程:
1. IMPORT_CHECK: 模块是否被导入？是否被调用？
2. RUNTIME_CHECK: 实例化后属性值是否正确传递？
3. EXECUTION_CHECK: 执行路径中代码是否真的运行？

用法:
  python tests/verify_changes.py --check all
  python tests/verify_changes.py --check league_params
  python tests/verify_changes.py --check pipeline_flow
"""

import sys
import os
import inspect
import argparse
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# 断言工具
# ============================================================
passed = 0
failed = 0
warnings = 0

def check(name: str, condition: bool, detail: str = ""):
    """断言检查，失败时打印详细信息"""
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {detail}")


def warn(name: str, detail: str = ""):
    global warnings
    warnings += 1
    print(f"  ⚠️  {name}: {detail}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 检查1: 联赛独立参数
# ============================================================
def check_league_params():
    section("检查1: 联赛独立参数是否真正独立")
    from src.config.league_params import get_league_params

    leagues = [
        ('premier_league', '英超'),
        ('la_liga', '西甲'),
        ('bundesliga', '德甲'),
        ('serie_a', '意甲'),
        ('ligue_1', '法甲'),
    ]

    params = {}
    for league_id, name in leagues:
        p = get_league_params(league_id)
        params[league_id] = {
            'factor_scale': p.factor_scale,
            'fusion_weight': p.fusion_weight,
            'calibration_discount': p.calibration_discount,
            'confidence_threshold': p.confidence_threshold,
        }

    # 验证: 五个联赛的参数不能完全相同
    all_factor_scales = set(p['factor_scale'] for p in params.values())
    check(
        "五个联赛的 factor_scale 不完全相同",
        len(all_factor_scales) > 1,
        f"所有联赛 factor_scale 相同: {all_factor_scales}"
    )

    all_fusion = set(p['fusion_weight'] for p in params.values())
    check(
        "五个联赛的 fusion_weight 不完全相同",
        len(all_fusion) > 1,
        f"所有联赛 fusion_weight 相同: {all_fusion}"
    )

    all_calib = set(p['calibration_discount'] for p in params.values())
    check(
        "五个联赛的 calibration_discount 不完全相同",
        len(all_calib) > 1,
        f"所有联赛 calibration_discount 相同: {all_calib}"
    )

    all_conf = set(p['confidence_threshold'] for p in params.values())
    check(
        "五个联赛的 confidence_threshold 不完全相同",
        len(all_conf) > 1,
        f"所有联赛 confidence_threshold 相同: {all_conf}"
    )

    # 打印每个联赛的参数
    print(f"\n  {'联赛':12s} {'factor_scale':>13s} {'fusion_weight':>14s} {'calib':>10s} {'conf':>10s}")
    print(f"  {'-'*12} {'-'*13} {'-'*14} {'-'*10} {'-'*10}")
    for league_id, name in leagues:
        p = params[league_id]
        print(f"  {name:12s} {p['factor_scale']:13.2f} {p['fusion_weight']:14.2f} {p['calibration_discount']:10.2f} {p['confidence_threshold']:10.2f}")


# ============================================================
# 检查2: Pipeline 是否真正使用联赛参数
# ============================================================
def check_pipeline_uses_params():
    section("检查2: Pipeline 是否真正读取联赛参数")
    from src.pipeline.orchestrator import GameFlowPipeline

    leagues = [
        ('premier_league', '英超'),
        ('la_liga', '西甲'),
        ('bundesliga', '德甲'),
        ('serie_a', '意甲'),
        ('ligue_1', '法甲'),
    ]

    pipelines = {}
    for league_id, name in leagues:
        p = GameFlowPipeline(league_id)
        pipelines[league_id] = p

        check(
            f"{name}: pipeline.factor_scale 匹配 league_params",
            abs(p.factor_scale - p.params.factor_scale) < 0.001,
            f"pipeline={p.factor_scale:.2f} vs params={p.params.factor_scale:.2f}"
        )

        check(
            f"{name}: pipeline.fusion_weight 匹配 league_params",
            abs(p.fusion_weight - p.params.fusion_weight) < 0.001,
            f"pipeline={p.fusion_weight:.2f} vs params={p.params.fusion_weight:.2f}"
        )

        check(
            f"{name}: pipeline.calibration_discount 匹配 league_params",
            abs(p.calibration_discount - p.params.calibration_discount) < 0.001,
            f"pipeline={p.calibration_discount:.2f} vs params={p.params.calibration_discount:.2f}"
        )

        check(
            f"{name}: pipeline.confidence_threshold 匹配 league_params",
            abs(p.confidence_threshold - p.params.confidence_threshold) < 0.001,
            f"pipeline={p.confidence_threshold:.2f} vs params={p.params.confidence_threshold:.2f}"
        )

    # 验证: 五个联赛的 pipeline 参数互不相同
    all_factor_scales = set(p.factor_scale for p in pipelines.values())
    check(
        "五个 pipeline 的 factor_scale 互不相同",
        len(all_factor_scales) > 1,
        f"所有 pipeline factor_scale 相同: {all_factor_scales}"
    )


# ============================================================
# 检查3: 执行路径追踪 — 确认参数在 run_stages_1_5 中被使用
# ============================================================
def check_execution_path():
    section("检查3: 执行路径追踪 — 参数是否在 run_stages 中被使用")
    from src.pipeline.orchestrator import GameFlowPipeline

    src = inspect.getsource(GameFlowPipeline.run_stages_1_5)

    # 检查 self.factor_scale 在 run_stages 中被引用
    check(
        "self.factor_scale 在 run_stages_1_5 中被引用",
        'self.factor_scale' in src,
        "factor_scale 未在 run_stages 中引用, 可能是死参数"
    )

    # 检查 self.fusion_weight 在 run_stages 中被引用
    check(
        "self.fusion_weight 在 run_stages_1_5 中被引用",
        'self.fusion_weight' in src,
        "fusion_weight 未在 run_stages 中引用, 可能是死参数"
    )

    # 检查 calibration_discount 在 run_stages 中被引用
    check(
        "self.calibration_discount 在 run_stages_1_5 中被引用",
        'self.calibration_discount' in src,
        "calibration_discount 未在 run_stages 中引用, 可能是死参数"
    )

    # 检查 confidence_threshold 在 run_stages_6_9 中被引用
    src69 = inspect.getsource(GameFlowPipeline.run_stages_6_9)
    check(
        "self.confidence_threshold 在 run_stages_6_9 中被引用",
        'self.confidence_threshold' in src69,
        "confidence_threshold 未在 run_stages_6_9 中引用, 可能是死参数"
    )

    # 检查硬编码是否被移除
    check(
        "run_stages_1_5 中无硬编码 fusion_weight=0.6",
        'fusion_weight = 0.6' not in src,
        "仍有硬编码的 fusion_weight=0.6"
    )

    check(
        "run_stages_6_9 中无硬编码 base_confidence_threshold=0.35",
        'base_confidence_threshold = 0.35' not in src69,
        "仍有硬编码的 confidence_threshold=0.35"
    )


# ============================================================
# 检查4: 死代码检测
# ============================================================
def check_dead_code():
    section("检查4: 死代码检测")
    import glob

    # 检测 src/factors/ 下是否有文件未被导入
    factor_files = glob.glob('src/factors/*.py')
    factor_files = [f for f in factor_files if '__pycache__' not in f and '__init__' not in f]

    # 检查 orchestrator.py 中的导入
    with open('src/pipeline/orchestrator.py', 'r') as f:
        orchestrator_src = f.read()

    for ff in factor_files:
        fname = os.path.basename(ff).replace('.py', '')
        if fname in ['compute_v2', 'v2_enhancement']:
            # 这些应该在 archive/ 中
            check(
                f"{fname}.py 已移至 archive/",
                not os.path.exists(ff),
                f"死代码仍在 src/factors/ 目录中"
            )
        else:
            # 检查是否被导入
            imported = fname in orchestrator_src
            if not imported:
                warn(f"{fname}.py 未被 orchestrator 导入", "可能是死代码")


# ============================================================
# 检查5: 概率校准是否真正使用联赛参数
# ============================================================
def check_probability_calibration():
    section("检查5: 概率校准链路 — calibration_discount 是否真正传递")
    from src.pipeline.orchestrator import GameFlowPipeline
    from src.engine.probability import ProbabilityEngine

    # 检查 ProbabilityEngine.calculate_value 是否接受 calibration_discount 参数
    sig = inspect.signature(ProbabilityEngine.calculate_value)
    check(
        "ProbabilityEngine.calculate_value 接受 calibration_discount 参数",
        'calibration_discount' in sig.parameters,
        f"当前参数: {list(sig.parameters.keys())}"
    )

    # 检查 orchestrator 调用 calculate_value 时是否传递了 calibration_discount
    src = inspect.getsource(GameFlowPipeline.run_stages_1_5)
    check(
        "orchestrator 调用 calculate_value 时传递 calibration_discount",
        'calibration_discount=self.calibration_discount' in src,
        "calculate_value 未收到 calibration_discount"
    )

    # 检查概率引擎源码中是否使用了 calibration_discount
    prob_src = inspect.getsource(ProbabilityEngine.calculate_value)
    check(
        "calculate_value 内部使用 calibration_discount 变量",
        'calibration_discount' in prob_src,
        "参数接收但未使用"
    )

    check(
        "calculate_value 未硬编码 0.85/0.15 校准系数",
        'model_p * 0.85 + implied * 0.15' not in prob_src,
        "仍有硬编码的校准系数"
    )


# ============================================================
# 主函数
# ============================================================
def main():
    global passed, failed, warnings
    parser = argparse.ArgumentParser(description='GTO-GameFlow 变更验证框架')
    parser.add_argument('--check', choices=['all', 'league_params', 'pipeline_flow', 'execution', 'dead_code', 'calibration'],
                        default='all', help='检查类型')
    args = parser.parse_args()

    print("=" * 60)
    print("  GTO-GameFlow 变更验证框架")
    print("  验证每次修改是否真正深入实际运行路径")
    print("=" * 60)

    checks = {
        'league_params': check_league_params,
        'pipeline_flow': check_pipeline_uses_params,
        'execution': check_execution_path,
        'dead_code': check_dead_code,
        'calibration': check_probability_calibration,
    }

    if args.check == 'all':
        for check_fn in checks.values():
            try:
                check_fn()
            except Exception as e:
                failed += 1
                print(f"  ❌ 检查异常: {e}")
    else:
        try:
            checks[args.check]()
        except Exception as e:
            failed += 1
            print(f"  ❌ 检查异常: {e}")

    # 总结
    print(f"\n{'='*60}")
    total = passed + failed
    if failed == 0:
        print(f"  ✅ 全部通过: {passed}/{total}")
    else:
        print(f"  ❌ {failed}/{total} 项检查失败", end="")
        if warnings:
            print(f", {warnings} 项警告")
        else:
            print()
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()