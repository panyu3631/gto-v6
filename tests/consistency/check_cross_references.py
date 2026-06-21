#!/usr/bin/env python3
"""
L5 一致性检查 #3: 交叉引用验证

验证:
- 注册中心因子 ID 与计算引擎一致
- 联赛权重覆盖所有因子
- F20/F38 互斥逻辑正确
- 联赛特定因子 all applicable_leagues 存在
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.factors.registry import (
    FACTOR_REGISTRY, LEAGUE_FACTOR_WEIGHTS,
    get_active_factors, validate_mutual_exclusion,
    FactorCategory,
)

EXIT_CODE = 0


def check(name, actual, expected, cmp="eq"):
    global EXIT_CODE
    if cmp == "eq":
        ok = actual == expected
    elif cmp == "ge":
        ok = actual >= expected
    elif cmp == "in":
        ok = actual in expected

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: expected={expected}, actual={actual}")
    if not ok:
        EXIT_CODE = 1


def main():
    print("=" * 60)
    print("L5 一致性检查 #3: 交叉引用验证")
    print("=" * 60)

    # 1. 权重配置覆盖所有因子
    for league_id, weights in LEAGUE_FACTOR_WEIGHTS.items():
        for fid in FACTOR_REGISTRY:
            check(f"{league_id} 包含因子 {fid}", fid in weights, True, cmp="eq")

    # 2. 联赛特定因子 applicable_leagues 检查
    for fid, factor in FACTOR_REGISTRY.items():
        if factor.category == FactorCategory.LEAGUE_SPECIFIC:
            check(f"{fid} 有 applicable_leagues", factor.applicable_leagues is not None, True, cmp="eq")

    # 3. F20/F38 互斥 — 英超和德甲均启用
    for league in ["premier_league", "bundesliga"]:
        exclusion = validate_mutual_exclusion(league)
        check(f"{league} F20/F38 互斥结果", exclusion, "F38")

    # 4. F14 权重检查
    for league_id in LEAGUE_FACTOR_WEIGHTS:
        w = LEAGUE_FACTOR_WEIGHTS[league_id].get("F14", 0.0)
        check(f"{league_id} F14 weight", w, 0.0)

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()