"""
L2 模块测试：资金管理与投注引擎 (bankroll.py)

测试范围:
- Kelly 公式计算
- 分数 Kelly (1/4)
- 置信度计算 (4因子公式)
- 优先级评分 (priority_score = f_actual × value × confidence)
- 资金分配评分
- 多注分配
- 硬性过滤 (value≥0.03, confidence≥0.6, odds 1.05-10.0)
- 投注建议生成
- 资金状态管理 (结算)
"""
import pytest
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine.bankroll import (
    BankrollManager, generate_bet_proposals, compute_confidence,
)
from src.data.models import BetProposal, BetSelection, BetPlacement, BetResult, BankrollState


# ================================================================
# Kelly 公式
# ================================================================

class TestKellyFormula:
    """Kelly 公式测试"""

    def test_standard_kelly(self):
        """标准 Kelly: f = (b×p−q)/b (v5.8: match_phase=0.5 无季节调整)"""
        mgr = BankrollManager(initial_bankroll=10000)
        # 赔率 2.0, 模型概率 0.55 → b=1.0, p=0.55, q=0.45
        f = mgr.compute_kelly(model_prob=0.55, odds=2.0, discount=1.0, match_phase=0.5)
        # f = (1.0 × 0.55 - 0.45) / 1.0 = 0.10
        assert abs(f - 0.10) < 0.001

    def test_half_kelly(self):
        """半 Kelly: 标准 Kelly × 0.25 (v5.8: match_phase=0.5)"""
        mgr = BankrollManager(initial_bankroll=10000)
        f = mgr.compute_kelly(model_prob=0.55, odds=2.0, match_phase=0.5)
        assert abs(f - 0.025) < 0.001  # 0.10 × 0.25

    def test_no_edge_returns_zero(self):
        """无优势时返回 0"""
        mgr = BankrollManager(initial_bankroll=10000)
        f = mgr.compute_kelly(model_prob=0.45, odds=2.0, match_phase=0.5)
        # b×p−q = 1.0×0.45−0.55 = -0.10 → f=0
        assert f == 0.0

    def test_zero_odds_returns_zero(self):
        """赔率为 0 返回 0"""
        mgr = BankrollManager(initial_bankroll=10000)
        f = mgr.compute_kelly(model_prob=0.55, odds=0.0)
        assert f == 0.0

    def test_odds_equal_one(self):
        """赔率 = 1.0 (净赔率 = 0)"""
        mgr = BankrollManager(initial_bankroll=10000)
        f = mgr.compute_kelly(model_prob=0.99, odds=1.0)
        assert f == 0.0

    def test_high_probability_high_odds(self):
        """高概率 + 高赔率 → 高 Kelly 比例 (v5.8: match_phase=0.5)"""
        mgr = BankrollManager(initial_bankroll=10000)
        f = mgr.compute_kelly(model_prob=0.70, odds=3.0, discount=1.0, match_phase=0.5)
        # b=2.0, p=0.70, q=0.30 → f = (2×0.70-0.30)/2 = 0.55
        assert abs(f - 0.55) < 0.001

    def test_kelly_with_discount(self):
        """自定义折扣 (v5.8: match_phase=0.5)"""
        mgr = BankrollManager(initial_bankroll=10000)
        f = mgr.compute_kelly(model_prob=0.55, odds=2.0, discount=1.0, match_phase=0.5)
        f_half = mgr.compute_kelly(model_prob=0.55, odds=2.0, discount=0.5, match_phase=0.5)
        assert abs(f_half - f * 0.5) < 0.001

    def test_extreme_probability(self):
        """极端概率值保护"""
        mgr = BankrollManager(initial_bankroll=10000)
        f = mgr.compute_kelly(model_prob=0.9999, odds=1.01, discount=1.0)
        assert not math.isnan(f) and not math.isinf(f)
        assert f >= 0


# ================================================================
# 置信度计算
# ================================================================

class TestConfidence:
    """置信度计算测试 — 规范第8.3b节"""

    def test_perfect_confidence(self):
        """完美数据 → 高置信度"""
        c = compute_confidence(
            data_completeness=1.0,
            factor_activation_rate=1.0,
            dispersion_penalty=1.0,
            match_phase=1.0,
        )
        # 0.4×1.0 + 0.3×1.0 + 0.2×1.0 + 0.1×1.0 = 1.0
        assert abs(c - 1.0) < 0.001

    def test_poor_confidence(self):
        """差数据 → 低置信度"""
        c = compute_confidence(
            data_completeness=0.5,
            factor_activation_rate=0.5,
            dispersion_penalty=0.5,
            match_phase=0.85,
        )
        # 0.4×0.5 + 0.3×0.5 + 0.2×0.5 + 0.1×0.85 = 0.535
        assert abs(c - 0.535) < 0.01

    def test_confidence_bounded_0_1(self):
        """置信度应在 [0, 1] 范围内"""
        c = compute_confidence(2.0, 2.0, 2.0, 2.0)
        assert 0.0 <= c <= 1.0

    def test_negative_inputs(self):
        """负输入值不应导致崩溃"""
        c = compute_confidence(-0.5, -0.5, -0.5, -0.5)
        assert 0.0 <= c <= 1.0

    def test_weights_sum_to_one(self):
        """权重 0.4+0.3+0.2+0.1 = 1.0"""
        c = compute_confidence(0.8, 0.8, 0.8, 0.8)
        assert abs(c - 0.8) < 0.001


# ================================================================
# 优先级评分
# ================================================================

class TestPriorityScore:
    """优先级评分测试 — 规范第8.3节"""

    def test_priority_score_formula(self):
        """priority_score = f_actual × value × confidence"""
        mgr = BankrollManager(initial_bankroll=10000)
        # f_actual (Kelly × 0.25) with model_prob=0.55, odds=2.0 → 0.025
        score = mgr.compute_priority_score(
            value=0.03, model_prob=0.55, implied_prob=0.50,
            confidence=0.8, odds=2.0,
        )
        # f_actual = 0.025, score = 0.025 × 0.03 × 0.8 = 0.0006
        assert abs(score - 0.0006) < 0.0001

    def test_zero_value_zero_score(self):
        """价值为 0 → 评分为 0"""
        mgr = BankrollManager(initial_bankroll=10000)
        score = mgr.compute_priority_score(
            value=0.0, model_prob=0.55, implied_prob=0.50,
            confidence=0.8, odds=2.0,
        )
        assert score == 0.0

    def test_low_confidence_low_score(self):
        """低置信度 → 低评分"""
        mgr = BankrollManager(initial_bankroll=10000)
        s_high = mgr.compute_priority_score(
            value=0.05, model_prob=0.55, implied_prob=0.50,
            confidence=0.9, odds=2.0,
        )
        s_low = mgr.compute_priority_score(
            value=0.05, model_prob=0.55, implied_prob=0.50,
            confidence=0.5, odds=2.0,
        )
        assert s_high > s_low

    def test_sort_by_priority(self):
        """按优先级降序排列"""
        mgr = BankrollManager(initial_bankroll=10000)
        p1 = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 0, 0.1)
        p2 = BetProposal("m2", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 0, 0.3)
        p3 = BetProposal("m3", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 0, 0.2)
        sorted_proposals = mgr.sort_by_priority([p1, p3, p2])
        assert sorted_proposals[0].priority_score == 0.3
        assert sorted_proposals[-1].priority_score == 0.1


# ================================================================
# 资金分配评分
# ================================================================

class TestAllocationScore:
    """资金分配评分测试 — 规范第9.4节"""

    def test_allocation_score_formula(self):
        """score_i = value_ratio × confidence × edge"""
        mgr = BankrollManager(initial_bankroll=10000)
        score = mgr.compute_allocation_score(
            model_prob=0.55, implied_prob=0.50, confidence=0.8,
        )
        # value_ratio = 0.55/0.50 - 1 = 0.10
        # edge = max(0.10, 0) = 0.10
        # score = 0.10 × 0.8 × 0.10 = 0.008
        assert abs(score - 0.008) < 0.001

    def test_no_edge_zero_score(self):
        """无优势 → 评分为 0"""
        mgr = BankrollManager(initial_bankroll=10000)
        score = mgr.compute_allocation_score(
            model_prob=0.45, implied_prob=0.50, confidence=0.8,
        )
        assert score == 0.0


# ================================================================
# 多注分配
# ================================================================

class TestAllocateStakes:
    """多注资金分配测试"""

    def test_single_stake_within_limits(self):
        mgr = BankrollManager(initial_bankroll=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 0, 0.1)
        results = mgr.allocate_stakes([p])
        assert len(results) == 1
        assert results[0].adjusted_stake > 0
        assert results[0].adjusted_stake <= 10000 * 0.05  # 单注上限 5%

    def test_multi_stake_total_within_exposure(self):
        """多注总投注不超过 20%"""
        mgr = BankrollManager(initial_bankroll=10000)
        proposals = []
        for i in range(5):
            p = BetProposal(f"m{i}", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 0, 0.1)
            proposals.append(p)
        results = mgr.allocate_stakes(proposals)
        total = sum(p.adjusted_stake for p in results)
        assert total <= 10000 * 0.20

    def test_empty_proposals(self):
        mgr = BankrollManager(initial_bankroll=10000)
        results = mgr.allocate_stakes([])
        assert results == []

    def test_zero_total_score(self):
        """总评分为 0 → 返回空"""
        mgr = BankrollManager(initial_bankroll=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 0, 0.0)
        results = mgr.allocate_stakes([p])
        assert results == []

    def test_adjusted_stake_rounded(self):
        """调整后 stake 非负"""
        mgr = BankrollManager(initial_bankroll=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 0, 0.1)
        results = mgr.allocate_stakes([p])
        assert results[0].adjusted_stake >= 0


# ================================================================
# 投注建议生成 + 硬性过滤
# ================================================================

class TestGenerateBetProposals:
    """投注建议生成 + 硬性过滤测试 — 规范第8.4节"""

    def test_value_below_threshold_filtered(self):
        """value < 0.03 应被过滤"""
        value_results = {
            "home": {"model_prob": 0.45, "implied_prob": 0.44, "value": 0.01, "odds": 2.0},
        }
        proposals = generate_bet_proposals(value_results, "m1", "premier_league")
        assert len(proposals) == 0

    def test_value_above_threshold_passes(self):
        """value ≥ 0.03 应通过"""
        value_results = {
            "home": {"model_prob": 0.55, "implied_prob": 0.50, "value": 0.05, "odds": 2.0},
        }
        proposals = generate_bet_proposals(
            value_results, "m1", "premier_league",
            factor_count=41, data_source_count=5, odds_std=0.05,
        )
        assert len(proposals) == 1

    def test_odds_below_min_filtered(self):
        """odds < 1.05 应被过滤"""
        value_results = {
            "home": {"model_prob": 0.97, "implied_prob": 0.95, "value": 0.03, "odds": 1.02},
        }
        proposals = generate_bet_proposals(value_results, "m1", "premier_league")
        assert len(proposals) == 0

    def test_odds_above_max_filtered(self):
        """odds > 10.0 应被过滤"""
        value_results = {
            "home": {"model_prob": 0.15, "implied_prob": 0.08, "value": 0.07, "odds": 15.0},
        }
        proposals = generate_bet_proposals(value_results, "m1", "premier_league")
        assert len(proposals) == 0

    def test_low_confidence_filtered(self):
        """低数据质量 → 低置信度 → 被过滤"""
        value_results = {
            "home": {"model_prob": 0.55, "implied_prob": 0.50, "value": 0.05, "odds": 2.0},
        }
        proposals = generate_bet_proposals(
            value_results, "m1", "premier_league",
            factor_count=10, data_source_count=1, odds_std=0.20,
        )
        assert len(proposals) == 0  # 低激活率 + 高离散度 → 置信度 < 0.6

    def test_high_confidence_passes(self):
        """高数据质量 → 高置信度 → 通过"""
        value_results = {
            "home": {"model_prob": 0.55, "implied_prob": 0.50, "value": 0.05, "odds": 2.0},
        }
        proposals = generate_bet_proposals(
            value_results, "m1", "premier_league",
            factor_count=41, data_source_count=5, odds_std=0.02,
        )
        assert len(proposals) == 1

    def test_multiple_outcomes(self):
        """多个结果同时评估"""
        value_results = {
            "home": {"model_prob": 0.55, "implied_prob": 0.50, "value": 0.05, "odds": 2.0},
            "draw": {"model_prob": 0.30, "implied_prob": 0.25, "value": 0.05, "odds": 4.0},
            "away": {"model_prob": 0.15, "implied_prob": 0.25, "value": -0.10, "odds": 4.0},
        }
        proposals = generate_bet_proposals(
            value_results, "m1", "premier_league",
            factor_count=41, data_source_count=5, odds_std=0.05,
        )
        assert len(proposals) == 2  # home 和 draw 通过，away 负价值


# ================================================================
# 资金状态管理
# ================================================================

class TestBankrollState:
    """资金状态管理测试"""

    def test_initial_state(self):
        mgr = BankrollManager(initial_bankroll=10000)
        assert mgr.state.balance == 10000
        assert mgr.state.total_bets == 0
        assert mgr.state.total_wins == 0

    def test_record_bet(self):
        mgr = BankrollManager(initial_bankroll=10000)
        bet = BetPlacement("b1", "m1", BetSelection.HOME_WIN, 2.0, 500, None)
        mgr.record_bet(bet)
        assert mgr.state.total_staked == 500
        assert mgr.state.total_bets == 1

    def test_settle_win(self):
        mgr = BankrollManager(initial_bankroll=10000)
        bet = BetPlacement("b1", "m1", BetSelection.HOME_WIN, 2.0, 500, None)
        mgr.record_bet(bet)
        mgr.settle_bet(bet, BetResult.WIN, 500)
        assert mgr.state.balance == 10500
        assert mgr.state.total_wins == 1
        assert mgr.state.consecutive_losses == 0

    def test_settle_loss(self):
        mgr = BankrollManager(initial_bankroll=10000)
        bet = BetPlacement("b1", "m1", BetSelection.HOME_WIN, 2.0, 500, None)
        mgr.record_bet(bet)
        mgr.settle_bet(bet, BetResult.LOSS, -500)
        assert mgr.state.balance == 9500
        assert mgr.state.consecutive_losses == 1

    def test_consecutive_losses_counter(self):
        mgr = BankrollManager(initial_bankroll=10000)
        for i in range(3):
            bet = BetPlacement(f"b{i}", f"m{i}", BetSelection.HOME_WIN, 2.0, 500, None)
            mgr.record_bet(bet)
            mgr.settle_bet(bet, BetResult.LOSS, -500)
        assert mgr.state.consecutive_losses == 3

    def test_win_resets_losses(self):
        mgr = BankrollManager(initial_bankroll=10000)
        for i in range(3):
            bet = BetPlacement(f"b{i}", f"m{i}", BetSelection.HOME_WIN, 2.0, 500, None)
            mgr.record_bet(bet)
            mgr.settle_bet(bet, BetResult.LOSS, -500)
        bet = BetPlacement("b4", "m4", BetSelection.HOME_WIN, 2.0, 500, None)
        mgr.record_bet(bet)
        mgr.settle_bet(bet, BetResult.WIN, 500)
        assert mgr.state.consecutive_losses == 0

    def test_peak_balance_and_drawdown(self):
        mgr = BankrollManager(initial_bankroll=10000)
        bet = BetPlacement("b1", "m1", BetSelection.HOME_WIN, 2.0, 500, None)
        mgr.record_bet(bet)
        mgr.settle_bet(bet, BetResult.WIN, 1000)
        assert mgr.state.peak_balance == 11000
        bet2 = BetPlacement("b2", "m2", BetSelection.HOME_WIN, 2.0, 500, None)
        mgr.record_bet(bet2)
        mgr.settle_bet(bet2, BetResult.LOSS, -500)
        assert mgr.state.max_drawdown > 0

    def test_roi(self):
        mgr = BankrollManager(initial_bankroll=10000)
        bet = BetPlacement("b1", "m1", BetSelection.HOME_WIN, 2.0, 1000, None)
        mgr.record_bet(bet)
        mgr.settle_bet(bet, BetResult.WIN, 1000)
        # ROI = (total_returned - total_staked) / total_staked = (1000 - 1000) / 1000 = 0
        assert abs(mgr.state.roi - 0.0) < 0.001

    def test_win_rate(self):
        mgr = BankrollManager(initial_bankroll=10000)
        for i in range(3):
            bet = BetPlacement(f"b{i}", f"m{i}", BetSelection.HOME_WIN, 2.0, 500, None)
            mgr.record_bet(bet)
            mgr.settle_bet(bet, BetResult.WIN if i < 2 else BetResult.LOSS, 500 if i < 2 else -500)
        assert abs(mgr.state.win_rate - 2/3) < 0.001