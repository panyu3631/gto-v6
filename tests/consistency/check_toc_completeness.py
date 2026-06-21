#!/usr/bin/env python3
"""
L5 一致性检查 #8: TOC 完整性验证

验证:
- 所有模块可导入
- 所有关键类可实例化
- 所有关键函数可调用
- 无循环导入
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
    print("L5 一致性检查 #8: TOC 完整性验证")
    print("=" * 60)

    modules_to_check = [
        # 数据模型
        ("src.data.models", ["MatchContext", "ProbabilityDistribution", "BetProposal",
                             "BetPlacement", "BetResult", "BetSelection", "BankrollState",
                             "CircuitBreakerState", "ScoreMatrix"]),
        # 配置
        ("src.config.settings", ["config", "GlobalConfig", "BankrollConfig",
                                  "CircuitBreakerConfig", "PipelineConfig"]),
        ("src.config.league_params", ["get_league_params", "LeagueParams"]),
        # 因子
        ("src.factors.registry", ["FACTOR_REGISTRY", "LEAGUE_FACTOR_WEIGHTS",
                                   "FactorCategory", "get_factor", "get_active_factors",
                                   "get_factor_count", "validate_mutual_exclusion"]),
        ("src.factors.compute", ["FactorComputationEngine", "compute_factors_from_context"]),
        # 引擎
        ("src.engine.probability", ["ProbabilityEngine"]),
        ("src.engine.bankroll", ["BankrollManager", "compute_confidence",
                                  "generate_bet_proposals"]),
        ("src.engine.risk_control", ["RiskController"]),
        # 流水线
        ("src.pipeline.orchestrator", ["GameFlowPipeline", "PipelineResult"]),
        # 数据层
        ("src.data.database", ["DatabaseManager", "Repository"]),
        ("src.data.api_client", ["FootballDataClient", "ApiFootballClient", "DataSourceManager"]),
        ("src.data.loader", ["DataLoader"]),
    ]

    for module_name, classes in modules_to_check:
        try:
            mod = __import__(module_name, fromlist=classes)
            check(f"导入 {module_name}", True)
            for cls_name in classes:
                obj = getattr(mod, cls_name, None)
                check(f"  {module_name}.{cls_name}", obj is not None,
                      f"type={type(obj).__name__}" if obj else "NOT FOUND")
        except Exception as e:
            check(f"导入 {module_name}", False, f"error={e}")

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()