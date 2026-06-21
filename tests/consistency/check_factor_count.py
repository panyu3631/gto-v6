#!/usr/bin/env python3
"""
L5 一致性检查 #1: 因子数量验证

验证:
- 总注册因子数 = 42
- 基础因子 = 17 (F1-F18, 排除 F14)
- 增强因子 = 14 (F19-F32)
- 联赛特定因子 = 10 (F33-F42)
- 5 联赛活跃因子数在合理范围
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.factors.registry import (
    FACTOR_REGISTRY, FactorCategory,
    get_factor_ids_by_category, get_factor_count,
)

EXIT_CODE = 0


def check(name, actual, expected, cmp="eq"):
    global EXIT_CODE
    if cmp == "eq":
        ok = actual == expected
    elif cmp == "ge":
        ok = actual >= expected
    elif cmp == "le":
        ok = actual <= expected
    else:
        ok = actual == expected

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: expected={expected}, actual={actual}")
    if not ok:
        EXIT_CODE = 1


def main():
    print("=" * 60)
    print("L5 一致性检查 #1: 因子数量验证")
    print("=" * 60)

    # 总注册因子数
    check("总注册因子数", len(FACTOR_REGISTRY), 42)

    # 基础因子
    base = get_factor_ids_by_category(FactorCategory.BASE)
    check("基础因子 (F1-F18 排除 F14)", len(base), 17)

    # 增强因子
    enhanced = get_factor_ids_by_category(FactorCategory.ENHANCED)
    check("增强因子 (F19-F32)", len(enhanced), 14)

    # 联赛特定因子
    league = get_factor_ids_by_category(FactorCategory.LEAGUE_SPECIFIC)
    check("联赛特定因子 (F33-F41)", len(league), 9)

    # 总因子 = 17 + 14 + 9 = 40 (v5.5.1: F42合并到F18)
    total_active = len(base) + len(enhanced) + len(league)
    check("活跃因子总数", total_active, 40)

    # 5 联赛活跃因子数
    for league_id in ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]:
        count = get_factor_count(league_id)
        check(f"{league_id} 活跃因子数", count, 30, cmp="ge")

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()