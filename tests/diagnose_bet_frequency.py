"""
GTO-GameFlow v5.2 投注频率诊断 (简化版)
直接追踪 generate_bet_proposals 的过滤过程
"""
import sys
import os
import random
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import MatchContext, BetSelection, BetResult
from src.pipeline.orchestrator import GameFlowPipeline
from src.config.settings import config as global_config
from src.engine.bankroll import generate_bet_proposals, compute_confidence
from src.engine.probability import ProbabilityEngine
from src.config.league_params import get_league_params

from tests.test_backtesting_real import (
    LEAGUE_CONFIG, generate_realistic_odds,
)


def deep_diagnose():
    """深入诊断单场比赛的所有过滤层"""
    print("=" * 70)
    print("GTO-GameFlow v5.2 投注频率深度诊断")
    print("=" * 70)

    # 测试多组 Elo 差异
    test_cases = [
        ("强强对话", 1920, 1850),
        ("中强 vs 中弱", 1750, 1550),
        ("强 vs 弱", 1900, 1420),
        ("中游对等", 1640, 1630),
        ("弱 vs 强", 1420, 1900),
        ("弱弱对话", 1500, 1480),
        ("中游 vs 中游", 1600, 1580),
        ("强 vs 中游", 1850, 1600),
    ]

    league_id = "premier_league"
    params = get_league_params(league_id)

    for label, home_elo, away_elo in test_cases:
        print(f"\n{'─'*60}")
        print(f"  [{label}] Elo: {home_elo} vs {away_elo} (diff={home_elo-away_elo:+d})")

        odds_h, odds_d, odds_a = generate_realistic_odds(home_elo, away_elo, seed=42)

        # 隐含概率 (去 margin)
        impl_h = 1.0 / odds_h
        impl_d = 1.0 / odds_d
        impl_a = 1.0 / odds_a
        total_imp = impl_h + impl_d + impl_a
        impl_h /= total_imp
        impl_d /= total_imp
        impl_a /= total_imp

        print(f"  赔率: {odds_h:.2f} / {odds_d:.2f} / {odds_a:.2f}")
        print(f"  隐含概率: {impl_h:.4f} / {impl_d:.4f} / {impl_a:.4f}")

        match = MatchContext(
            match_id=f"diag_{label}",
            league_id=league_id,
            season="2023/24",
            matchday=1,
            kickoff_time=datetime(2023, 8, 11, 20, 0),
            home_team="Home FC",
            away_team="Away FC",
            home_elo=home_elo,
            away_elo=away_elo,
            odds_home=odds_h,
            odds_draw=odds_d,
            odds_away=odds_a,
        )

        # 运行完整流水线
        pipeline = GameFlowPipeline(league_id, initial_bankroll=10000.0)
        extra = {
            "elo_diff": home_elo - away_elo,
            "recent_results": [3.0, 3.0, 1.0, 0.0, 3.0],
            "rank_diff": int((home_elo - away_elo) / 20),
            "goal_diff": (home_elo - away_elo) / 20,
            "xg_diff": (home_elo - away_elo) / 200,
            "streak_momentum": 0.3,
            "streak_momentum_league": 0.4,
            "data_source_count": 5,
            "match_phase": 1.0,
        }
        result = pipeline.run_full(match, extra_data=extra)

        if result.fused_probs:
            mp = result.fused_probs
            print(f"  模型概率: {mp.prob_home:.4f} / {mp.prob_draw:.4f} / {mp.prob_away:.4f}")
            vh = mp.prob_home - impl_h
            vd = mp.prob_draw - impl_d
            va = mp.prob_away - impl_a
            print(f"  value:     {vh:+.4f} / {vd:+.4f} / {va:+.4f}")

        if result.proposals:
            print(f"  提案数: {len(result.proposals)}")
            for p in result.proposals:
                print(f"    -> {p.selection}: odds={p.odds:.2f} value={p.value:.4f} "
                      f"stake={p.adjusted_stake:.2f} "
                      f"priority={p.priority_score:.4f}")
        else:
            print(f"  ❌ 无提案 (全部被过滤)")

            # 手动检查每一层过滤
            print(f"\n  逐层过滤分析:")
            prob_engine = ProbabilityEngine(league_id)
            probs = prob_engine.compute_probabilities(match, extra)
            mp = probs

            selections = [
                ("home", mp.prob_home, odds_h, impl_h),
                ("draw", mp.prob_draw, odds_d, impl_d),
                ("away", mp.prob_away, odds_a, impl_a),
            ]

            for name, model_p, odds, imp in selections:
                value = model_p - imp
                pass_value = "✓" if value >= 0.03 else "✗"
                pass_odds = "✓" if 1.05 <= odds <= 10.0 else "✗"

                # 手动算 confidence
                conf = compute_confidence(
                    data_completeness=0.8,
                    factor_activation_rate=0.5,
                    odds_std=0.05,
                    match_phase=1.0,
                )
                pass_conf = "✓" if conf >= 0.6 else "✗"

                # Kelly
                b = odds - 1.0
                p = max(0.0001, min(0.9999, model_p))
                q = 1.0 - p
                f_kelly = (b * p - q) / b if b > 1e-6 else 0
                pass_kelly = "✓" if f_kelly > 0 else "✗"

                print(f"    {name:6s}: model={model_p:.4f} implied={imp:.4f} "
                      f"value={value:+.4f}[{pass_value}] "
                      f"odds={odds:.2f}[{pass_odds}] "
                      f"conf={conf:.2f}[{pass_conf}] "
                      f"f_kelly={f_kelly:.4f}[{pass_kelly}]")


if __name__ == "__main__":
    deep_diagnose()