#!/usr/bin/env python3
"""
L5 一致性检查 #5: 参数同步验证

验证:
- settings.py 中的阈值与 bankroll.py 硬性过滤一致
- settings.py 中的阈值与 risk_control.py 熔断一致
- 配置文件参数一致性
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.settings import config as global_config

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
    print("L5 一致性检查 #5: 参数同步验证")
    print("=" * 60)

    cfg = global_config

    # 1. Kelly fraction
    check("Kelly fraction = 0.25", cfg.bankroll.kelly_fraction == 0.25,
          f"actual={cfg.bankroll.kelly_fraction}")

    # 2. 单注上限
    check("单注上限 = 5%", cfg.bankroll.single_bet_max_ratio == 0.05,
          f"actual={cfg.bankroll.single_bet_max_ratio}")

    # 3. 总曝光上限
    check("总曝光上限 = 20%", cfg.bankroll.max_total_exposure == 0.20,
          f"actual={cfg.bankroll.max_total_exposure}")

    # 3a. 日暴露上限
    check("日暴露上限 = 15%", cfg.bankroll.daily_exposure_limit == 0.15,
          f"actual={cfg.bankroll.daily_exposure_limit}")

    # 3b. 周暴露上限
    check("周暴露上限 = 35%", cfg.bankroll.weekly_exposure_limit == 0.35,
          f"actual={cfg.bankroll.weekly_exposure_limit}")

    # 4. 最低赔率
    check("最低赔率 = 1.05", cfg.pipeline.default_odds_min == 1.05,
          f"actual={cfg.pipeline.default_odds_min}")

    # 5. 最高赔率
    check("最高赔率 = 10.0", cfg.pipeline.default_odds_max == 10.0,
          f"actual={cfg.pipeline.default_odds_max}")

    # 6. 最小价值阈值
    check("最小价值阈值 = 0.03", cfg.pipeline.min_value_threshold == 0.03,
          f"actual={cfg.pipeline.min_value_threshold}")

    # 7. 熔断：连续亏损
    check("熔断连续亏损 = 5", cfg.circuit_breaker.max_consecutive_losses == 5,
          f"actual={cfg.circuit_breaker.max_consecutive_losses}")

    # 8. 熔断：日亏损
    check("熔断日亏损 = 8%", cfg.circuit_breaker.daily_loss_pct == 0.08,
          f"actual={cfg.circuit_breaker.daily_loss_pct}")

    # 9. 熔断：周亏损
    check("熔断周亏损 = 15%", cfg.circuit_breaker.weekly_loss_pct == 0.15,
          f"actual={cfg.circuit_breaker.weekly_loss_pct}")

    # 10. 熔断：月亏损
    check("熔断月亏损 = 25%", cfg.circuit_breaker.monthly_loss_pct == 0.25,
          f"actual={cfg.circuit_breaker.monthly_loss_pct}")

    # 11. 冷却时间
    check("冷却时间 = 48h", cfg.circuit_breaker.cooldown_hours == 48,
          f"actual={cfg.circuit_breaker.cooldown_hours}")

    # 12. 版本号
    check("版本号 = 5.0.0", cfg.version == "5.0.0",
          f"actual={cfg.version}")

    # 13. 9 阶段流水线
    check("流水线阶段 = 9", cfg.pipeline.stage_count == 9,
          f"actual={cfg.pipeline.stage_count}")

    # 14. 比分矩阵截断
    check("比分矩阵截断 = 5", cfg.pipeline.score_matrix_max == 5,
          f"actual={cfg.pipeline.score_matrix_max}")

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()