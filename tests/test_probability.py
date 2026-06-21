"""
L2 模块测试：概率引擎 (probability.py)

测试范围:
- Stage 2: logit_accumulation
- Stage 3: sigmoid_normalization
- Stage 4: poisson_bridge + Dual-Domain fusion
- Stage 5: value_calculation (含动态 overround)
- 概率分布归一化
- 边界条件
"""
import pytest
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine.probability import ProbabilityEngine
from src.data.models import ProbabilityDistribution


# ================================================================
# Stage 2: logit_accumulation
# ================================================================

class TestLogitAccumulation:
    """logit 累加测试"""

    def test_uniform_prior_all_zeros(self):
        """均匀先验 + 无因子 → 所有 logit = 0"""
        engine = ProbabilityEngine("premier_league")
        logits = engine.logit_accumulation(
            market_probs={"home": 0.33, "draw": 0.33, "away": 0.33},
            factor_deltas={},
            uniform_prior=True,
        )
        for outcome in ("home", "draw", "away"):
            assert abs(logits[outcome]) < 0.001

    def test_market_prior(self):
        """市场先验 → logit 应反映市场概率"""
        engine = ProbabilityEngine("premier_league")
        logits = engine.logit_accumulation(
            market_probs={"home": 0.60, "draw": 0.25, "away": 0.15},
            factor_deltas={},
            uniform_prior=False,
        )
        assert logits["home"] > logits["draw"]
        assert logits["draw"] > logits["away"]

    def test_factor_delta_shifts_logit(self):
        """因子 delta 应正向移动 logit"""
        engine = ProbabilityEngine("premier_league")
        logits_no_factor = engine.logit_accumulation(
            market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
            factor_deltas={},
            uniform_prior=True,
        )
        logits_with_factor = engine.logit_accumulation(
            market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
            factor_deltas={
                "F1": {"home": 0.25, "draw": 0.0, "away": -0.25},
            },
            uniform_prior=True,
        )
        assert logits_with_factor["home"] > logits_no_factor["home"]
        assert logits_with_factor["away"] < logits_no_factor["away"]

    def test_multiple_factors_accumulate(self):
        """多个因子 delta 累加"""
        engine = ProbabilityEngine("premier_league")
        logits = engine.logit_accumulation(
            market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
            factor_deltas={
                "F1": {"home": 0.25, "draw": 0.0, "away": -0.25},
                "F4": {"home": 0.10, "draw": 0.0, "away": 0.0},
                "F9": {"home": 0.18, "draw": 0.0, "away": -0.18},
            },
            uniform_prior=True,
        )
        assert abs(logits["home"] - 0.53) < 0.01


# ================================================================
# Stage 3: sigmoid_normalization
# ================================================================

class TestSigmoidNormalization:
    """sigmoid 归一化测试"""

    def test_probs_sum_to_one(self):
        """概率总和应为 1"""
        engine = ProbabilityEngine("premier_league")
        probs = engine.sigmoid_normalization({
            "home": 0.5, "draw": 0.0, "away": -0.5,
        })
        total = probs.prob_home + probs.prob_draw + probs.prob_away
        assert abs(total - 1.0) < 0.001

    def test_higher_logit_higher_prob(self):
        """更高的 logit 应转换为更高的概率"""
        engine = ProbabilityEngine("premier_league")
        probs = engine.sigmoid_normalization({
            "home": 1.0, "draw": 0.0, "away": -1.0,
        })
        assert probs.prob_home > probs.prob_draw
        assert probs.prob_draw > probs.prob_away

    def test_equal_logits_equal_probs(self):
        """相等的 logit → 相等的概率"""
        engine = ProbabilityEngine("premier_league")
        probs = engine.sigmoid_normalization({
            "home": 0.0, "draw": 0.0, "away": 0.0,
        })
        assert abs(probs.prob_home - 0.333) < 0.001
        assert abs(probs.prob_draw - 0.333) < 0.001
        assert abs(probs.prob_away - 0.333) < 0.001

    def test_logit_to_probability_combined(self):
        """Stage 2+3 组合调用"""
        engine = ProbabilityEngine("premier_league")
        probs = engine.logit_to_probability(
            market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
            factor_deltas={"F1": {"home": 0.25, "draw": 0.0, "away": -0.25}},
        )
        assert isinstance(probs, ProbabilityDistribution)
        assert probs.prob_home > 0.45  # F1 正向调整


# ================================================================
# Stage 4: poisson_bridge
# ================================================================

class TestPoissonBridge:
    """泊松桥接测试"""

    def test_poisson_probs_sum_to_one(self):
        """泊松概率应归一化"""
        engine = ProbabilityEngine("premier_league")
        poisson_probs, score_matrix = engine.poisson_bridge(
            home_elo=1600, away_elo=1400, factor_deltas={}, max_goals=5,
        )
        total = poisson_probs.prob_home + poisson_probs.prob_draw + poisson_probs.prob_away
        assert abs(total - 1.0) < 0.001

    def test_higher_elo_higher_win_prob(self):
        """ELO 更高 → 泊松胜率更高"""
        engine = ProbabilityEngine("premier_league")
        probs_strong, _ = engine.poisson_bridge(
            home_elo=1700, away_elo=1300, factor_deltas={}, max_goals=5,
        )
        probs_even, _ = engine.poisson_bridge(
            home_elo=1500, away_elo=1500, factor_deltas={}, max_goals=5,
        )
        assert probs_strong.prob_home > probs_even.prob_home

    def test_score_matrix_contains_all_combos(self):
        """比分矩阵包含所有组合"""
        engine = ProbabilityEngine("premier_league")
        _, score_matrix = engine.poisson_bridge(
            home_elo=1500, away_elo=1500, factor_deltas={}, max_goals=5,
        )
        assert score_matrix.max_goals == 5
        assert len(score_matrix.matrix) == 36  # 6×6

    def test_score_matrix_sum_to_one(self):
        """比分矩阵概率总和为 1"""
        engine = ProbabilityEngine("premier_league")
        _, score_matrix = engine.poisson_bridge(
            home_elo=1500, away_elo=1500, factor_deltas={}, max_goals=5,
        )
        total = sum(score_matrix.matrix.values())
        assert abs(total - 1.0) < 0.001

    def test_factor_delta_affects_poisson(self):
        """因子 delta 影响泊松预期进球"""
        engine = ProbabilityEngine("premier_league")
        probs_no_factor, _ = engine.poisson_bridge(
            home_elo=1500, away_elo=1500, factor_deltas={}, max_goals=5,
        )
        probs_with_factor, _ = engine.poisson_bridge(
            home_elo=1500, away_elo=1500,
            factor_deltas={"F9": {"home": 0.18, "draw": 0.0, "away": -0.18}},
            max_goals=5,
        )
        assert probs_with_factor.prob_home > probs_no_factor.prob_home


# ================================================================
# Dual-Domain Fusion
# ================================================================

class TestDualDomainFusion:
    """Dual-Domain 融合测试"""

    def test_fusion_weights(self):
        """融合权重: 70% logit + 30% poisson"""
        engine = ProbabilityEngine("premier_league")
        logit = ProbabilityDistribution(0.60, 0.25, 0.15)
        poisson = ProbabilityDistribution(0.50, 0.30, 0.20)
        fused = engine.dual_domain_fusion(logit, poisson, fusion_weight=0.3)
        expected_home = 0.7 * 0.60 + 0.3 * 0.50  # = 0.57
        assert abs(fused.prob_home - expected_home) < 0.001

    def test_fusion_sums_to_one(self):
        """融合后概率总和为 1"""
        engine = ProbabilityEngine("premier_league")
        logit = ProbabilityDistribution(0.55, 0.25, 0.20)
        poisson = ProbabilityDistribution(0.45, 0.30, 0.25)
        fused = engine.dual_domain_fusion(logit, poisson)
        total = fused.prob_home + fused.prob_draw + fused.prob_away
        assert abs(total - 1.0) < 0.001


# ================================================================
# Stage 5: value_calculation
# ================================================================

class TestValueCalculation:
    """价值计算测试"""

    def test_dynamic_overround(self):
        """overround 应动态计算"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.50, 0.25, 0.25)
        # 1/2.0 + 1/4.0 + 1/4.0 = 0.5 + 0.25 + 0.25 = 1.0
        odds = {"home": 2.0, "draw": 4.0, "away": 4.0}
        result = engine.calculate_value(model_probs, odds)
        assert abs(result["home"]["implied_prob"] - 0.50) < 0.01

    def test_value_positive_when_model_higher(self):
        """模型概率 > 隐含概率 → 正价值"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.60, 0.20, 0.20)
        odds = {"home": 2.0, "draw": 4.0, "away": 4.0}
        result = engine.calculate_value(model_probs, odds)
        assert result["home"]["value"] > 0

    def test_value_negative_when_model_lower(self):
        """模型概率 < 隐含概率 → 负价值"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.30, 0.35, 0.35)
        odds = {"home": 2.0, "draw": 4.0, "away": 4.0}
        result = engine.calculate_value(model_probs, odds)
        assert result["home"]["value"] < 0

    def test_value_with_margin(self):
        """有庄家 margin 时的价值计算"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.50, 0.25, 0.25)
        # 含 margin 赔率: 1/1.8 + 1/3.5 + 1/3.5 = 0.556 + 0.286 + 0.286 = 1.127
        odds = {"home": 1.8, "draw": 3.5, "away": 3.5}
        result = engine.calculate_value(model_probs, odds)
        # implied = 1/1.8 / 1.127 ≈ 0.493
        assert abs(result["home"]["implied_prob"] - 0.493) < 0.01

    def test_find_value_opportunities(self):
        """筛选价值机会"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.55, 0.25, 0.20)
        odds = {"home": 1.8, "draw": 3.5, "away": 5.0}
        value_results = engine.calculate_value(model_probs, odds)
        opportunities = engine.find_value_opportunities(value_results, threshold=0.005)
        # 模型概率 0.55 > 隐含概率 0.493 → 有正价值
        assert len(opportunities) > 0

    def test_no_opportunities_below_threshold(self):
        """低于阈值不应出现在机会列表中"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.35, 0.33, 0.32)
        odds = {"home": 2.0, "draw": 3.0, "away": 3.0}
        value_results = engine.calculate_value(model_probs, odds)
        opportunities = engine.find_value_opportunities(value_results, threshold=0.10)
        assert len(opportunities) == 0


# ================================================================
# 边界条件
# ================================================================

class TestEdgeCases:
    """边界条件测试"""

    def test_zero_odds(self):
        """赔率为 0 时的防御"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.50, 0.25, 0.25)
        odds = {"home": 0.0, "draw": 4.0, "away": 4.0}
        result = engine.calculate_value(model_probs, odds)
        assert "home" in result

    def test_extreme_probabilities(self):
        """极端概率值"""
        engine = ProbabilityEngine("premier_league")
        model_probs = ProbabilityDistribution(0.99, 0.005, 0.005)
        odds = {"home": 1.01, "draw": 100.0, "away": 100.0}
        result = engine.calculate_value(model_probs, odds)
        assert result["home"]["model_prob"] > 0.98

    def test_lambda_non_negative(self):
        """泊松 λ 不能为负"""
        engine = ProbabilityEngine("premier_league")
        poisson_probs, _ = engine.poisson_bridge(
            home_elo=500, away_elo=2500, factor_deltas={}, max_goals=5,
        )
        assert poisson_probs.prob_home + poisson_probs.prob_draw + poisson_probs.prob_away > 0

    def test_logit_boundary(self):
        """logit_accumulation 不做概率边界裁剪"""
        engine = ProbabilityEngine("premier_league")
        logits = engine.logit_accumulation(
            market_probs={"home": 0.99, "draw": 0.005, "away": 0.005},
            factor_deltas={},
            uniform_prior=False,
        )
        for v in logits.values():
            assert not math.isnan(v)
            assert not math.isinf(v)