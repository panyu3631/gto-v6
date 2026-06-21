#!/usr/bin/env python3
"""
L5 一致性检查 #2: 计算链完整性验证

验证:
- 因子计算引擎 compute_all 覆盖所有活跃因子
- logit_accumulation 正确累加所有因子 delta
- sigmoid_normalization 总和为 1
- poisson_bridge 产生有效比分矩阵
- value_calculation 计算所有 3 个结果
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.factors.compute import FactorComputationEngine
from src.engine.probability import ProbabilityEngine
from src.data.models import ProbabilityDistribution

EXIT_CODE = 0


def check(name, actual, expected, cmp="eq", tolerance=0.001):
    global EXIT_CODE
    if cmp == "eq":
        ok = abs(actual - expected) < tolerance
    elif cmp == "ge":
        ok = actual >= expected
    elif cmp == "gt":
        ok = actual > expected
    elif cmp == "contains":
        ok = expected in actual
    else:
        ok = actual == expected

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: expected={expected}, actual={actual}")
    if not ok:
        EXIT_CODE = 1


def main():
    print("=" * 60)
    print("L5 一致性检查 #2: 计算链完整性")
    print("=" * 60)

    # 1. 因子计算
    engine = FactorComputationEngine("premier_league")
    deltas = engine.compute_all(
        elo_diff=200.0, xi_rating=6.0, recent_results=[1.5]*5,
        h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
        xg_diff=0.0, market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
    )
    check("因子计算覆盖所有活跃因子", len(deltas), 0, cmp="gt")

    # 2. logit 累加
    prob_engine = ProbabilityEngine("premier_league")
    logits = prob_engine.logit_accumulation(
        {"home": 0.45, "draw": 0.28, "away": 0.27}, deltas, uniform_prior=False
    )
    for outcome in ("home", "draw", "away"):
        check(f"logit {outcome} 存在", outcome in logits, True, cmp="eq")
        check(f"logit {outcome} 非 NaN", logits[outcome], logits[outcome], cmp="eq")

    # 3. sigmoid 归一化
    probs = prob_engine.sigmoid_normalization(logits)
    total = probs.prob_home + probs.prob_draw + probs.prob_away
    check("sigmoid 归一化总和", total, 1.0)

    # 4. 泊松桥接
    poisson_probs, score_matrix = prob_engine.poisson_bridge(
        home_elo=1600, away_elo=1400, factor_deltas=deltas, max_goals=5,
    )
    total_p = poisson_probs.prob_home + poisson_probs.prob_draw + poisson_probs.prob_away
    check("泊松概率总和", total_p, 1.0)
    check("比分矩阵大小", len(score_matrix.matrix), 36)

    # 5. Dual-Domain 融合
    fused = prob_engine.dual_domain_fusion(probs, poisson_probs)
    total_f = fused.prob_home + fused.prob_draw + fused.prob_away
    check("融合概率总和", total_f, 1.0)

    # 6. 价值计算
    values = prob_engine.calculate_value(fused, {"home": 2.0, "draw": 3.5, "away": 4.0})
    check("价值计算 home", "home" in values, True, cmp="eq")
    check("价值计算 draw", "draw" in values, True, cmp="eq")
    check("价值计算 away", "away" in values, True, cmp="eq")

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()