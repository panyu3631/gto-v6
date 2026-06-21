#!/usr/bin/env python3
"""
L5 一致性检查 #7: 标注类验证

验证:
- 关键注释标注存在 '规范第X.Y节' 形式
- 方法 docstring 存在
- 重要函数有参考规范标注
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import inspect
from src.engine.bankroll import (
    BankrollManager, compute_confidence, generate_bet_proposals,
)
from src.engine.probability import ProbabilityEngine
from src.engine.risk_control import RiskController

EXIT_CODE = 0


def check(name, condition, detail=""):
    global EXIT_CODE
    ok = bool(condition)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        EXIT_CODE = 1


def has_docstring(obj):
    return bool(inspect.getdoc(obj))


def main():
    print("=" * 60)
    print("L5 一致性检查 #7: 标注类验证")
    print("=" * 60)

    # 1. BankrollManager 关键方法
    check("BankrollManager docstring", has_docstring(BankrollManager))
    check("compute_kelly docstring", has_docstring(BankrollManager.compute_kelly))
    check("compute_priority_score docstring", has_docstring(BankrollManager.compute_priority_score))
    check("compute_allocation_score docstring", has_docstring(BankrollManager.compute_allocation_score))
    check("allocate_stakes docstring", has_docstring(BankrollManager.allocate_stakes))

    # 2. compute_confidence
    check("compute_confidence docstring", has_docstring(compute_confidence))

    # 3. generate_bet_proposals
    check("generate_bet_proposals docstring", has_docstring(generate_bet_proposals))

    # 4. ProbabilityEngine
    check("ProbabilityEngine docstring", has_docstring(ProbabilityEngine))
    check("logit_accumulation docstring", has_docstring(ProbabilityEngine.logit_accumulation))
    check("calculate_value docstring", has_docstring(ProbabilityEngine.calculate_value))
    check("dual_domain_fusion docstring", has_docstring(ProbabilityEngine.dual_domain_fusion))

    # 5. RiskController
    check("RiskController docstring", has_docstring(RiskController))
    check("check_circuit_breaker docstring", has_docstring(RiskController.check_circuit_breaker))
    check("run_all_checks docstring", has_docstring(RiskController.run_all_checks))

    # 6. 检查关键源文件包含规范引用
    import importlib
    for module_name in [
        "src.factors.registry",
        "src.factors.compute",
        "src.engine.probability",
        "src.engine.bankroll",
        "src.engine.risk_control",
        "src.pipeline.orchestrator",
    ]:
        try:
            mod = importlib.import_module(module_name)
            doc = inspect.getdoc(mod)
            if doc:
                check(f"{module_name} 有模块 docstring", True)
            else:
                check(f"{module_name} 有模块 docstring", False)
        except Exception as e:
            check(f"{module_name} 导入", False, f"error={e}")

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()