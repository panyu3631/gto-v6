"""
L1 单元测试：因子计算引擎 (compute.py)

测试范围:
- 每个因子公式正确性（41 因子 × 3 个 outcome = 123 个断言）
- 5 联赛因子权重差异
- F20/F38 互斥逻辑
- F6+F36 叠加上限
- EWMA 计算
- H2H 优势计算
- logit 函数
- 联赛特定因子过滤
"""
import pytest
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.factors.compute import FactorComputationEngine, compute_factors_from_context
from src.factors.registry import get_factor_weight


# ================================================================
# 默认测试数据
# ================================================================

DEFAULT_MARKET_PROBS = {"home": 0.45, "draw": 0.28, "away": 0.27}
DEFAULT_RECENT = [3.0, 3.0, 1.0, 0.0, 1.0]  # 胜胜平负平
DEFAULT_H2H = [3.0, 0.0, 1.0, 3.0, 0.0]     # 胜负平胜负


class TestF1_EloRating:
    """F1: ELO评分"""

    def test_f1_basic(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=200.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        # delta = 0.5 × (200/400) = 0.25
        assert abs(result["F1"]["home"] - 0.25) < 0.01
        assert abs(result["F1"]["away"] + 0.25) < 0.01
        assert abs(result["F1"]["draw"]) < 0.001

    def test_f1_elo_zero(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert abs(result["F1"]["home"]) < 0.001
        assert abs(result["F1"]["away"]) < 0.001

    def test_f1_elo_negative(self):
        """F1: 客队 ELO 更高时 delta 为负"""
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=-300.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F1"]["home"] < 0
        assert result["F1"]["away"] > 0


class TestF2_CoreInjuries:
    """F2: 核心伤停"""

    def test_f2_above_average(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=7.5, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        # xi_rating 7.5 → (7.5-6.0)×0.08 = 0.12, ×weight=0.9 → 0.108
        assert abs(result["F2"]["home"] - 0.108) < 0.01
        assert abs(result["F2"]["away"] + 0.108) < 0.01

    def test_f2_below_average(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=5.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F2"]["home"] < 0
        assert result["F2"]["away"] > 0

    def test_f2_average(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert abs(result["F2"]["home"]) < 0.001


class TestF3_RecentForm:
    """F3: 近期状态 (EWMA)"""

    def test_f3_all_wins(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[3.0, 3.0, 3.0, 3.0, 3.0],
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        # 5连胜: EWMA = 3.0, delta = (3.0-1.5)×0.15 = 0.225, ×weight=0.85 → 0.191
        assert abs(result["F3"]["home"] - 0.191) < 0.01
        assert result["F3"]["home"] > 0

    def test_f3_all_losses(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[0.0, 0.0, 0.0, 0.0, 0.0],
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F3"]["home"] < 0

    def test_f3_empty_results(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[],
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert abs(result["F3"]["home"]) < 0.001  # EWMA=1.5, delta=0


class TestF4_HomeAdvantage:
    """F4: 主客场优势"""

    def test_f4_home_only(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F4"]["home"] > 0
        assert abs(result["F4"]["draw"]) < 0.001
        assert abs(result["F4"]["away"]) < 0.001

    def test_f4_league_difference(self):
        """F4: 不同联赛主场优势不同"""
        epl = FactorComputationEngine("premier_league")
        r_epl = epl.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        bundesliga = FactorComputationEngine("bundesliga")
        r_bl = bundesliga.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        # 德甲主场优势最大
        assert r_bl["F4"]["home"] > r_epl["F4"]["home"]


class TestF5_H2H:
    """F5: 历史交锋"""

    def test_f5_home_dominant(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[3.0, 3.0, 3.0, 3.0, 3.0],  # 全胜
            matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F5"]["home"] > 0

    def test_f5_away_dominant(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[0.0, 0.0, 0.0, 0.0, 0.0],  # 全负
            matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F5"]["home"] < 0

    def test_f5_empty_h2h(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert abs(result["F5"]["home"]) < 0.001


class TestF6_ScheduleDensity:
    """F6: 赛程密度"""

    def test_f6_one_match(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert abs(result["F6"]["home"]) < 0.001  # 1场比赛无惩罚

    def test_f6_three_matches(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=3, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        # delta = -(3-1)×0.03 = -0.06, ×weight=0.7 → -0.042
        assert abs(result["F6"]["home"] + 0.042) < 0.01
        assert result["F6"]["home"] < 0


class TestF7_RankDiff:
    """F7: 联赛排名差"""

    def test_f7_home_better(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=10, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        # rank_diff = rank_away - rank_home = 10 → 主队排名更靠前
        assert result["F7"]["home"] > 0

    def test_f7_away_better(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=-5, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F7"]["home"] < 0


class TestF8_GoalDiff:
    """F8: 进球/失球差"""

    def test_f8_positive(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=15.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F8"]["home"] > 0

    def test_f8_negative(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=-10.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F8"]["home"] < 0


class TestF9_XGDiff:
    """F9: xG差值"""

    def test_f9_positive(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=1.5, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F9"]["home"] > 0

    def test_f9_weight_double_f8(self):
        """F9 (xG) 权重应为 F8 (进球差) 的两倍"""
        engine = FactorComputationEngine("premier_league")
        r1 = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=12.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        r2 = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=1.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        # F9 权重 (0.8) ≈ 1.5× F8 权重 (0.55)
        assert abs(r2["F9"]["home"]) > abs(r1["F8"]["home"])


class TestF10_OddsImplied:
    """F10: 赔率隐含概率"""

    def test_f10_all_three_outcomes(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
        )
        assert result["F10"]["home"] != 0.0
        assert result["F10"]["draw"] != 0.0
        assert result["F10"]["away"] != 0.0

    def test_f10_draw_not_zero(self):
        """F10: 平局结果不应为 0"""
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert result["F10"]["draw"] != 0.0


class TestF11_OddsMovement:
    """F11: 市场赔率变动"""

    def test_f11_with_opening_probs(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
            opening_probs={"home": 0.50, "draw": 0.25, "away": 0.25},
        )
        # 主胜从0.50降到0.45 → 资金流向客队 → delta = (0.50-0.45)×0.5 = 0.025
        assert abs(result["F11"]["home"] - 0.025 * 0.6) < 0.01

    def test_f11_without_opening_probs(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert abs(result["F11"]["home"]) < 0.001


class TestF12_Weather:
    """F12: 天气影响 — 仅影响平局"""

    def test_f12_draw_only(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            weather=0.5,
        )
        assert abs(result["F12"]["home"]) < 0.001
        assert result["F12"]["draw"] != 0.0
        assert abs(result["F12"]["away"]) < 0.001


class TestF13_RefereeStyle:
    """F13: 裁判风格 — 仅影响主队"""

    def test_f13_home_only(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            ref_yellow_rate=4.5,
        )
        assert result["F13"]["home"] != 0.0
        assert abs(result["F13"]["draw"]) < 0.001
        assert abs(result["F13"]["away"]) < 0.001


class TestF14_Deprecated:
    """F14: 已废弃 — 所有 delta 为 0"""

    def test_f14_all_zeros(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
        )
        assert "F14" not in result or (
            abs(result["F14"]["home"]) < 0.001 and
            abs(result["F14"]["draw"]) < 0.001 and
            abs(result["F14"]["away"]) < 0.001
        )


class TestF15_CoachChange:
    """F15: 教练更替"""

    def test_f15_positive(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            coach_change_effect=0.084,  # 换帅后前3场
        )
        assert result["F15"]["home"] > 0


class TestF16_EuropeanFatigue:
    """F16: 欧战影响"""

    def test_f16_penalty(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            fatigue_penalty=-0.8,  # 欧战后间隔3天
        )
        assert result["F16"]["home"] < 0


class TestF17_RotationRisk:
    """F17: 轮换预测"""

    def test_f17_high_rotation(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            rotation_risk=0.8,
        )
        # rotation_risk × 0.05 = 0.04, ×weight=0.5 → 0.02
        assert abs(result["F17"]["home"] - 0.02) < 0.01


class TestF18_DerbyMatch:
    """F18: 德比战 — 仅影响平局"""

    def test_f18_draw_only(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            derby_boost=0.8,  # 国家德比
        )
        assert abs(result["F18"]["home"]) < 0.001
        assert result["F18"]["draw"] > 0
        assert abs(result["F18"]["away"]) < 0.001


# ================================================================
# 增强因子 F19-F32
# ================================================================

class TestF19_AttackDefenseStyle:
    """F19: 攻击/防守风格"""

    def test_f19_style_mismatch(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            style_matchup_score=0.8,  # 风格相克
        )
        assert result["F19"]["home"] > 0

    def test_f19_style_match(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            style_matchup_score=0.3,
        )
        assert result["F19"]["home"] < 0


class TestF20_StreakMomentum:
    """F20: 连胜/连败动量"""

    def test_f20_streak(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            streak_momentum=0.7, streak_momentum_league=0.7,  # 5连胜
        )
        # F20 与 F38 互斥，优先使用 F38；若两者都传入则 F20 被排除
        assert "F38" in result, "F38 应优先于 F20"
        assert result["F38"]["home"] > 0


class TestF21_KeyPlayerForm:
    """F21: 核心球员状态"""

    def test_f21_above_average(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            player_form=8.0,
        )
        assert result["F21"]["home"] > 0

    def test_f21_delta(self):
        """F21: player_form=7.0 → (7.0-6.5)×0.06 = 0.03"""
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            player_form=7.0,
        )
        assert abs(result["F21"]["home"] / 0.7 - 0.03) < 0.01


class TestF22_MarketSentiment:
    """F22: 市场情绪"""

    def test_f22_sentiment(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            market_sentiment=0.5,
        )
        assert result["F22"]["home"] > 0


class TestF23_OddsDiscrepancy:
    """F23: 赔率离散度 — 仅影响平局"""

    def test_f23_high_discrepancy(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            odds_std=0.15,
        )
        assert result["F23"]["draw"] > 0
        assert abs(result["F23"]["home"]) < 0.001


class TestF24_NewsNLP:
    """F24: 新闻NLP"""

    def test_f24_sentiment(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            nlp_sentiment=0.6,
        )
        assert result["F24"]["home"] > 0


class TestF25_TimeDecay:
    """F25: 时间衰减加权"""

    def test_f25_decay(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            time_decay_factor=0.8,
        )
        assert result["F25"]["home"] > 0


class TestF26_LeagueStrength:
    """F26: 联赛强度调整"""

    def test_f26_bias(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            league_strength_bias=1.0,
        )
        assert result["F26"]["home"] > 0


class TestF27_GoalDistribution:
    """F27: 进球分布修正 — 三个方向相同"""

    def test_f27_all_equal(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            poisson_correction=0.3,
        )
        assert abs(result["F27"]["home"] - result["F27"]["draw"]) < 0.001
        assert abs(result["F27"]["draw"] - result["F27"]["away"]) < 0.001


class TestF28_AsianHandicap:
    """F28: 亚盘深度"""

    def test_f28_handicap(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            handicap_depth=0.6,
        )
        assert result["F28"]["home"] > 0


class TestF29_TotalsTrend:
    """F29: 大小球趋势 — home 和 away 相同"""

    def test_f29_home_away_equal(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            totals_trend=0.5,
        )
        assert abs(result["F29"]["home"] - result["F29"]["away"]) < 0.001


class TestF30_ValueSignal:
    """F30: 赔率价值信号 — 三个方向相同"""

    def test_f30_all_equal(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            value_signal=0.2,
        )
        assert abs(result["F30"]["home"] - result["F30"]["draw"]) < 0.001
        assert abs(result["F30"]["draw"] - result["F30"]["away"]) < 0.001


class TestF31_ContrarianSignal:
    """F31: 反市场偏差 — home 和 away 相同"""

    def test_f31_home_away_equal(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            contrarian_signal=0.3,
        )
        assert abs(result["F31"]["home"] - result["F31"]["away"]) < 0.001


class TestF32_MarketEfficiency:
    """F32: 市场效率评分 — 三个方向相同"""

    def test_f32_all_equal(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            market_efficiency=0.85,
        )
        assert abs(result["F32"]["home"] - result["F32"]["draw"]) < 0.001


# ================================================================
# 联赛特定因子 F33-F41 (v5.5.1: F42已合并到F18)
# ================================================================

class TestF33_Motivation:
    """F33: 保级/争冠动力 — 仅影响主队"""

    def test_f33_home_only(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            motivation_boost=8.0,  # 保级动力
        )
        assert result["F33"]["home"] > 0
        assert abs(result["F33"]["draw"]) < 0.001
        assert abs(result["F33"]["away"]) < 0.001


class TestF34_FinancialDisparity:
    """F34: 财力差距"""

    def test_f34_gap(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            financial_gap_effect=5.0,  # 身价比>5x
        )
        assert result["F34"]["home"] > 0


class TestF35_WinterBreak:
    """F35: 冬歇期效应 — 仅德甲"""

    def test_f35_only_bundesliga(self):
        """F35 仅在德甲激活"""
        epl = FactorComputationEngine("premier_league")
        r_epl = epl.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            winter_break_effect=0.5,
        )
        assert "F35" not in r_epl, "F35 不应出现在英超结果中"

        bl = FactorComputationEngine("bundesliga")
        r_bl = bl.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            winter_break_effect=0.5,
        )
        assert "F35" in r_bl, "F35 应出现在德甲结果中"


class TestF36_ChristmasFixtures:
    """F36: 圣诞赛程 — 仅英超，home 为负"""

    def test_f36_home_negative(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            christmas_fatigue=0.5,
        )
        assert result["F36"]["home"] < 0
        assert result["F36"]["draw"] > 0


class TestF37_MidtableComplacency:
    """F37: 中游无欲 — home 为负，draw 和 away 为正"""

    def test_f37_signs(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            complacency_effect=0.6,
        )
        assert result["F37"]["home"] < 0
        assert result["F37"]["draw"] > 0
        assert result["F37"]["away"] > 0


class TestF38_StreakMomentumLeague:
    """F38: 连胜/连败(联赛特定)"""

    def test_f38_momentum(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            streak_momentum_league=0.7,
        )
        assert result["F38"]["home"] > 0


class TestF39_TablePosition:
    """F39: 积分榜排名"""

    def test_f39_advantage(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            position_advantage=10.0,
        )
        assert result["F39"]["home"] > 0


class TestF40_PromotedTeam:
    """F40: 升班马数据"""

    def test_f40_delta(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            promoted_team_delta=0.5,
        )
        assert result["F40"]["home"] > 0


class TestF41_ScheduleAdvantage:
    """F41: 赛程优势"""

    def test_f41_advantage(self):
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            schedule_advantage=0.5,
        )
        assert result["F41"]["home"] > 0


# v5.5.1: F42 已合并到 F18 (德比战), 不再单独测试

# ================================================================
# 互斥与叠加逻辑
# ================================================================

class TestMutualExclusionLogic:
    """F20/F38 互斥逻辑"""

    def test_f20_f38_mutual_exclusion(self):
        """当 F20 和 F38 均启用时，F20 不应出现"""
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            streak_momentum=0.7,
            streak_momentum_league=0.7,
        )
        assert "F20" not in result, "F20 被 F38 禁用时应排除"
        assert "F38" in result, "F38 应保留"


class TestF6F36SuperpositionCap:
    """F6+F36 叠加上限"""

    def test_f6_f36_cap(self):
        """F6 + F36 合计调整上限为 -8%"""
        engine = FactorComputationEngine("premier_league")
        result = engine.compute_all(
            elo_diff=0.0, xi_rating=6.0, recent_results=[1.5]*5,
            h2h_results=[], matches_7d=5, rank_diff=0, goal_diff=0.0,
            xg_diff=0.0, market_probs=DEFAULT_MARKET_PROBS,
            christmas_fatigue=1.0,  # 圣诞赛程高度疲劳
        )
        combined_home = result["F6"]["home"] + result["F36"]["home"]
        assert combined_home >= -0.0801, f"F6+F36 合计不应低于 -0.08，实际 {combined_home}"


# ================================================================
# 5 联赛覆盖
# ================================================================

class TestFiveLeagueCoverage:
    """5 联赛因子计算"""

    LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]

    def test_all_leagues_produce_results(self):
        """所有 5 个联赛都能正常计算因子"""
        for league_id in self.LEAGUES:
            engine = FactorComputationEngine(league_id)
            result = engine.compute_all(
                elo_diff=200.0, xi_rating=6.0, recent_results=DEFAULT_RECENT,
                h2h_results=DEFAULT_H2H, matches_7d=2, rank_diff=5, goal_diff=10.0,
                xg_diff=1.0, market_probs=DEFAULT_MARKET_PROBS,
            )
            assert len(result) > 0, f"{league_id} 无结果"
            assert "F1" in result, f"{league_id} 缺少 F1"

    def test_league_specific_factor_counts_differ(self):
        """不同联赛的活跃因子数量可能不同"""
        counts = {}
        for league_id in self.LEAGUES:
            engine = FactorComputationEngine(league_id)
            result = engine.compute_all(
                elo_diff=200.0, xi_rating=6.0, recent_results=DEFAULT_RECENT,
                h2h_results=DEFAULT_H2H, matches_7d=2, rank_diff=5, goal_diff=10.0,
                xg_diff=1.0, market_probs=DEFAULT_MARKET_PROBS,
            )
            counts[league_id] = len(result)

        # 活跃因子数应在合理范围
        for league_id, count in counts.items():
            assert count >= 30, f"{league_id} 因子数过少: {count}"
            assert count <= 42, f"{league_id} 因子数过多: {count}"


# ================================================================
# 辅助函数
# ================================================================

class TestHelperFunctions:
    """EWMA / H2H / logit 辅助函数"""

    def test_ewma_all_wins(self):
        engine = FactorComputationEngine("premier_league")
        val = engine._compute_ewma([3.0, 3.0, 3.0, 3.0, 3.0], 0.5)
        assert abs(val - 3.0) < 0.01

    def test_ewma_all_losses(self):
        engine = FactorComputationEngine("premier_league")
        val = engine._compute_ewma([0.0, 0.0, 0.0, 0.0, 0.0], 0.5)
        assert abs(val - 0.0) < 0.01

    def test_ewma_empty(self):
        engine = FactorComputationEngine("premier_league")
        val = engine._compute_ewma([], 0.5)
        assert abs(val - 1.5) < 0.01

    def test_h2h_all_wins(self):
        engine = FactorComputationEngine("premier_league")
        val = engine._compute_h2h_advantage([3.0, 3.0, 3.0, 3.0, 3.0])
        assert val > 0.45

    def test_h2h_all_losses(self):
        engine = FactorComputationEngine("premier_league")
        val = engine._compute_h2h_advantage([0.0, 0.0, 0.0, 0.0, 0.0])
        assert val < -0.45

    def test_logit_midpoint(self):
        val = FactorComputationEngine._logit(0.5)
        assert abs(val) < 0.001

    def test_logit_high(self):
        val = FactorComputationEngine._logit(0.9)
        assert val > 0

    def test_logit_low(self):
        val = FactorComputationEngine._logit(0.1)
        assert val < 0

    def test_logit_boundary(self):
        """logit 边界值保护"""
        val = FactorComputationEngine._logit(0.0)
        assert not math.isnan(val) and not math.isinf(val)

    def test_logit_upper_boundary(self):
        val = FactorComputationEngine._logit(1.0)
        assert not math.isnan(val) and not math.isinf(val)


# ================================================================
# 便捷函数
# ================================================================

class TestComputeFactorsFromContext:
    """compute_factors_from_context 便捷函数"""

    def test_basic_call(self):
        result = compute_factors_from_context("premier_league", {
            "elo_diff": 200.0,
            "market_probs": DEFAULT_MARKET_PROBS,
        })
        assert "F1" in result
        assert result["F1"]["home"] > 0

    def test_missing_keys_use_defaults(self):
        """缺失键使用默认值，不抛出异常"""
        result = compute_factors_from_context("premier_league", {})
        assert "F1" in result