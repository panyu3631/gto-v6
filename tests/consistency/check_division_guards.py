#!/usr/bin/env python3
"""
L5 一致性检查 #4: 除法守卫验证

验证:
- Kelly 公式: 赔率 ≤ 0 返回 0
- overround 计算: 赔率 = 0 不崩溃
- EWMA 空列表返回默认值
- 概率归一化: 总和 = 0 不崩溃
- Logit 边界值保护
"""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.bankroll import BankrollManager
from src.engine.probability import ProbabilityEngine
from src.factors.compute import FactorComputationEngine
from src.data.models import ProbabilityDistribution

EXIT_CODE = 0


def check(name, condition, detail=""):
    global EXIT_CODE
    ok = bool(condition)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        EXIT_CODE = 1


def main():
    print("=" * 60)
    print("L5 一致性检查 #4: 除法守卫验证")
    print("=" * 60)

    # 1. Kelly 公式: 赔率 = 0
    mgr = BankrollManager(10000)
    f = mgr.compute_kelly(model_prob=0.55, odds=0.0)
    check("Kelly with odds=0", f == 0.0, f"f={f}")

    # 2. Kelly 公式: 赔率 = 1.0 (b=0)
    f = mgr.compute_kelly(model_prob=0.99, odds=1.0)
    check("Kelly with odds=1.0 (b=0)", f == 0.0, f"f={f}")

    # 3. Kelly 公式: 极端概率
    f = mgr.compute_kelly(model_prob=0.0, odds=2.0)
    check("Kelly with prob=0", f == 0.0, f"f={f}")

    f = mgr.compute_kelly(model_prob=1.0, odds=2.0)
    check("Kelly with prob=1", not math.isnan(f) and not math.isinf(f), f"f={f}")

    # 4. overround: 赔率 = 0
    engine = ProbabilityEngine("premier_league")
    model = engine.sigmoid_normalization({"home": 0.0, "draw": 0.0, "away": 0.0})
    result = engine.calculate_value(model, {"home": 0.0, "draw": 3.0, "away": 3.0})
    check("overround with odds=0", "home" in result and "value" in result["home"])

    # 5. EWMA 空列表
    val = FactorComputationEngine._compute_ewma([], 0.5)
    check("EWMA empty list", val == 1.5, f"val={val}")

    # 6. H2H 空列表
    val = FactorComputationEngine._compute_h2h_advantage([])
    check("H2H empty list", val == 0.0, f"val={val}")

    # 7. Logit 边界
    v = FactorComputationEngine._logit(0.0)
    check("logit(0)", not math.isnan(v) and not math.isinf(v), f"v={v}")

    v = FactorComputationEngine._logit(1.0)
    check("logit(1)", not math.isnan(v) and not math.isinf(v), f"v={v}")

    # 8. 概率分布: 总和为 0 防御
    try:
        pd = ProbabilityDistribution(0.0, 0.0, 0.0)
        check("ProbabilityDistribution(0,0,0)", False, "应抛出异常")
    except ValueError:
        check("ProbabilityDistribution(0,0,0)", True, "正确抛出 ValueError")

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()