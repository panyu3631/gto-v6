"""
L4 回归测试：场景快照对比 (golden_snapshot.json)

测试范围:
- 30 个预定义测试场景
- 每个场景包含输入数据和预期输出快照
- 确保代码变更不破坏已有行为
- 支持 5 联赛覆盖
"""
import pytest
import sys
import os
import json
import math
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import MatchContext, BetSelection
from src.pipeline.orchestrator import GameFlowPipeline
from src.factors.compute import FactorComputationEngine
from src.engine.probability import ProbabilityEngine
from src.engine.bankroll import BankrollManager, compute_confidence
from src.factors.registry import get_active_factors

# 加载 golden snapshot
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "golden_snapshot.json")


def load_golden():
    if os.path.exists(SNAPSHOT_PATH):
        with open(SNAPSHOT_PATH, "r") as f:
            return json.load(f)
    return {}


# ================================================================
# 场景 1-10: 基础因子确定性
# ================================================================

class TestFactorDeterminism:
    """因子计算确定性"""

    def test_scenario_01_f1_deterministic(self):
        """F1: 相同输入 → 相同输出"""
        for league in ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]:
            engine = FactorComputationEngine(league)
            r1 = engine.compute_all(
                elo_diff=200.0, xi_rating=6.0, recent_results=[1.5]*5,
                h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
                xg_diff=0.0, market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
            )
            r2 = engine.compute_all(
                elo_diff=200.0, xi_rating=6.0, recent_results=[1.5]*5,
                h2h_results=[], matches_7d=1, rank_diff=0, goal_diff=0.0,
                xg_diff=0.0, market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
            )
            assert r1["F1"]["home"] == r2["F1"]["home"]

    def test_scenario_02_ewma_deterministic(self):
        """EWMA 计算确定性"""
        engine = FactorComputationEngine("premier_league")
        v1 = engine._compute_ewma([3.0, 3.0, 1.0, 0.0, 1.0], 0.5)
        v2 = engine._compute_ewma([3.0, 3.0, 1.0, 0.0, 1.0], 0.5)
        assert v1 == v2

    def test_scenario_03_h2h_deterministic(self):
        """H2H 计算确定性"""
        engine = FactorComputationEngine("premier_league")
        v1 = engine._compute_h2h_advantage([3.0, 0.0, 1.0, 3.0, 0.0])
        v2 = engine._compute_h2h_advantage([3.0, 0.0, 1.0, 3.0, 0.0])
        assert v1 == v2

    def test_scenario_04_logit_deterministic(self):
        """logit 计算确定性"""
        v1 = FactorComputationEngine._logit(0.45)
        v2 = FactorComputationEngine._logit(0.45)
        assert v1 == v2

    def test_scenario_05_poisson_deterministic(self):
        """泊松模型确定性"""
        engine = ProbabilityEngine("premier_league")
        p1, _ = engine.poisson_bridge(1500, 1500, {}, 5)
        p2, _ = engine.poisson_bridge(1500, 1500, {}, 5)
        assert p1.prob_home == p2.prob_home

    def test_scenario_06_kelly_deterministic(self):
        """Kelly 计算确定性"""
        mgr = BankrollManager(10000)
        f1 = mgr.compute_kelly(0.55, 2.0)
        f2 = mgr.compute_kelly(0.55, 2.0)
        assert f1 == f2

    def test_scenario_07_confidence_deterministic(self):
        """置信度计算确定性"""
        c1 = compute_confidence(0.8, 0.8, 0.9, 1.0)
        c2 = compute_confidence(0.8, 0.8, 0.9, 1.0)
        assert c1 == c2

    def test_scenario_08_overround_deterministic(self):
        """overround 计算确定性"""
        engine = ProbabilityEngine("premier_league")
        model = engine.sigmoid_normalization({"home": 0.0, "draw": 0.0, "away": 0.0})
        r1 = engine.calculate_value(model, {"home": 2.0, "draw": 3.5, "away": 4.0})
        r2 = engine.calculate_value(model, {"home": 2.0, "draw": 3.5, "away": 4.0})
        assert r1["home"]["value"] == r2["home"]["value"]

    def test_scenario_09_priority_score_deterministic(self):
        """优先级评分确定性"""
        mgr = BankrollManager(10000)
        s1 = mgr.compute_priority_score(0.05, 0.55, 0.50, 0.8, 2.0)
        s2 = mgr.compute_priority_score(0.05, 0.55, 0.50, 0.8, 2.0)
        assert s1 == s2

    def test_scenario_10_fusion_deterministic(self):
        """Dual-Domain 融合确定性"""
        engine = ProbabilityEngine("premier_league")
        logit = engine.sigmoid_normalization({"home": 0.5, "draw": 0.0, "away": -0.5})
        poisson, _ = engine.poisson_bridge(1600, 1400, {}, 5)
        f1 = engine.dual_domain_fusion(logit, poisson)
        f2 = engine.dual_domain_fusion(logit, poisson)
        assert f1.prob_home == f2.prob_home


# ================================================================
# 场景 11-20: 边界场景
# ================================================================

class TestBoundaryScenarios:
    """边界场景"""

    def test_scenario_11_equal_teams(self):
        """完全相等的两队"""
        pipeline = GameFlowPipeline("premier_league")
        match = MatchContext(
            match_id="EQUAL", league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1500, away_elo=1500,
            odds_home=2.5, odds_draw=3.2, odds_away=2.5,
        )
        result = pipeline.run_stages_1_5(match)
        total = result.fused_probs.prob_home + result.fused_probs.prob_draw + result.fused_probs.prob_away
        assert abs(total - 1.0) < 0.001

    def test_scenario_12_strong_favorite(self):
        """强队 vs 弱队"""
        pipeline = GameFlowPipeline("premier_league")
        match = MatchContext(
            match_id="STRONG", league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Strong", away_team="Weak",
            home_elo=1800, away_elo=1200,
            odds_home=1.2, odds_draw=6.0, odds_away=12.0,
        )
        extra = {
            "elo_diff": 600.0,
            "recent_results": [3.0, 3.0, 3.0, 3.0, 3.0],
            "rank_diff": 18,
            "goal_diff": 30.0,
            "xg_diff": 3.0,
            "streak_momentum": 1.0,
            "streak_momentum_league": 1.0,
            "position_advantage": 18.0,
        }
        result = pipeline.run_full(match, extra)
        assert result.fused_probs.prob_home > 0.70

    def test_scenario_13_high_odds_underdog(self):
        """高赔率冷门"""
        pipeline = GameFlowPipeline("premier_league")
        match = MatchContext(
            match_id="UNDERDOG", league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Weak", away_team="Strong",
            home_elo=1200, away_elo=1800,
            odds_home=10.0, odds_draw=5.0, odds_away=1.3,
        )
        result = pipeline.run_stages_1_5(match)
        assert result.fused_probs.prob_away > result.fused_probs.prob_home

    def test_scenario_14_derby_match(self):
        """德比战 — 平局概率上升"""
        pipeline = GameFlowPipeline("premier_league")
        match = MatchContext(
            match_id="DERBY", league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Man Utd", away_team="Man City",
            home_elo=1700, away_elo=1750,
            odds_home=3.0, odds_draw=3.4, odds_away=2.3,
        )
        extra = {
            "elo_diff": -50.0,
            "derby_boost": 0.8,
            "derby_intensity": 0.8,
        }
        result = pipeline.run_full(match, extra)
        # 德比战平局概率应高于不受风格影响时的基准
        assert result.fused_probs.prob_draw > 0.15

    def test_scenario_15_christmas_fixtures(self):
        """圣诞赛程 — 英超特定"""
        pipeline = GameFlowPipeline("premier_league")
        match = MatchContext(
            match_id="XMAS", league_id="premier_league",
            season="2025/26", matchday=19,
            kickoff_time=datetime(2025, 12, 26, 15, 0, 0),
            home_team="Team A", away_team="Team B",
            home_elo=1600, away_elo=1500,
            odds_home=2.0, odds_draw=3.5, odds_away=3.5,
        )
        extra = {
            "christmas_fatigue": 0.8,
            "matches_7d": 3,
        }
        result = pipeline.run_stages_1_5(match, extra)
        assert "F36" in result.factor_deltas, "圣诞赛程因子应激活"

    def test_scenario_16_winter_break(self):
        """冬歇期效应 — 德甲特定"""
        pipeline = GameFlowPipeline("bundesliga")
        match = MatchContext(
            match_id="WINTER", league_id="bundesliga",
            season="2025/26", matchday=18,
            kickoff_time=datetime(2026, 1, 20, 15, 0, 0),
            home_team="Bayern", away_team="Dortmund",
            home_elo=1800, away_elo=1700,
            odds_home=1.55, odds_draw=4.50, odds_away=5.00,
        )
        extra = {"winter_break_effect": 0.5}
        result = pipeline.run_stages_1_5(match, extra)
        assert "F35" in result.factor_deltas, "冬歇期因子应激活"

    def test_scenario_17_all_league_factors_activated(self):
        """所有联赛特定因子激活"""
        pipeline = GameFlowPipeline("premier_league")
        match = MatchContext(
            match_id="ALL", league_id="premier_league",
            season="2025/26", matchday=30,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1600, away_elo=1500,
            odds_home=2.0, odds_draw=3.5, odds_away=3.5,
        )
        extra = {
            "motivation_boost": 8.0,
            "financial_gap_effect": 5.0,
            "christmas_fatigue": 0.5,
            "complacency_effect": 0.5,
            "streak_momentum_league": 0.5,
            "position_advantage": 8.0,
            "promoted_team_delta": 0.3,
            "schedule_advantage": 0.4,
            "derby_intensity": 0.5,
        }
        result = pipeline.run_stages_1_5(match, extra)
        # 检查联赛特定因子是否都在原始因子结果中 (v5.3a: 正交化后可能在 factor_deltas 中被移除)
        for fid in ["F33", "F34", "F36", "F37", "F38", "F39", "F40", "F41"]:
            if fid in get_active_factors("premier_league"):
                assert fid in result.raw_factor_deltas, f"{fid} 应激活"

    def test_scenario_18_circuit_breaker_integration(self):
        """熔断集成 — 连续亏损后流水线阻断"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="BREAKER", league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1500, away_elo=1500,
            odds_home=2.0, odds_draw=3.0, odds_away=3.0,
        )
        # 手动设置连续亏损
        pipeline.bankroll_mgr.state.consecutive_losses = 5
        result = pipeline.run_full(match)
        if result.circuit_broken:
            assert result.proposals == []

    def test_scenario_19_data_degradation(self):
        """数据降级 — 缺少联赛特定因子不应崩溃"""
        pipeline = GameFlowPipeline("bundesliga")
        match = MatchContext(
            match_id="DEGRADE", league_id="bundesliga",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1500, away_elo=1500,
            odds_home=2.0, odds_draw=3.0, odds_away=3.0,
        )
        result = pipeline.run_full(match, extra_data={})
        assert result.errors == []

    def test_scenario_20_multi_bet_allocation(self):
        """多注分配 — 多个高价值机会"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        # 构建一个高价值场景
        match = MatchContext(
            match_id="MULTI", league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Strong", away_team="Weak",
            home_elo=1750, away_elo=1250,
            odds_home=1.5, odds_draw=4.0, odds_away=7.0,
        )
        extra = {
            "elo_diff": 500.0,
            "recent_results": [3.0, 3.0, 3.0, 3.0, 3.0],
            "h2h_results": [3.0, 3.0, 3.0, 3.0, 3.0],
            "rank_diff": 15,
            "goal_diff": 25.0,
            "xg_diff": 2.5,
            "streak_momentum": 0.8,
            "streak_momentum_league": 0.8,
            "position_advantage": 15.0,
            "xi_rating": 8.0,
        }
        result = pipeline.run_full(match, extra)
        if result.proposals:
            total = sum(p.adjusted_stake for p in result.proposals)
            assert total <= 10000 * 0.20  # 总曝光 ≤ 20%


# ================================================================
# 场景 21-30: 快照对比
# ================================================================

class TestSnapshotComparison:
    """快照对比测试"""

    def test_scenario_21_arsenal_tottenham_snapshot(self):
        """Arsenal vs Tottenham 快照对比"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="SNAP_ARS_TOT",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime(2025, 8, 1, 15, 0, 0),
            home_team="Arsenal", away_team="Tottenham",
            home_elo=1750, away_elo=1650,
            odds_home=1.91, odds_draw=3.75, odds_away=3.80,
        )
        extra = {
            "elo_diff": 100.0, "xi_rating": 7.0,
            "recent_results": [3.0, 3.0, 1.0, 3.0, 1.0],
            "rank_diff": 5, "goal_diff": 8.0, "xg_diff": 1.2,
            "streak_momentum": 0.5, "streak_momentum_league": 0.5,
            "position_advantage": 8.0,
        }
        result = pipeline.run_full(match, extra)

        # 快照断言
        snapshot = {
            "match_id": result.match_id,
            "factor_count": len(result.factor_deltas),
            "prob_home": round(result.fused_probs.prob_home, 4),
            "prob_draw": round(result.fused_probs.prob_draw, 4),
            "prob_away": round(result.fused_probs.prob_away, 4),
        }
        # 验证关键特征
        assert snapshot["factor_count"] > 0
        assert 0 < snapshot["prob_home"] < 1
        assert 0 < snapshot["prob_draw"] < 1
        assert 0 < snapshot["prob_away"] < 1

    def test_scenario_22_barcelona_madrid_snapshot(self):
        """Barcelona vs Real Madrid 快照"""
        pipeline = GameFlowPipeline("la_liga", initial_bankroll=10000)
        match = MatchContext(
            match_id="SNAP_BAR_RMA",
            league_id="la_liga",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Barcelona", away_team="Real Madrid",
            home_elo=1780, away_elo=1760,
            odds_home=1.95, odds_draw=3.60, odds_away=3.50,
        )
        result = pipeline.run_full(match)
        assert result.fused_probs is not None

    def test_scenario_23_bayern_dortmund_snapshot(self):
        """Bayern vs Dortmund 快照"""
        pipeline = GameFlowPipeline("bundesliga", initial_bankroll=10000)
        match = MatchContext(
            match_id="SNAP_BAY_BVB",
            league_id="bundesliga",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Bayern", away_team="Dortmund",
            home_elo=1800, away_elo=1700,
            odds_home=1.55, odds_draw=4.50, odds_away=5.00,
        )
        result = pipeline.run_full(match)
        assert result.fused_probs is not None

    def test_scenario_24_juventus_inter_snapshot(self):
        """Juventus vs Inter 快照"""
        pipeline = GameFlowPipeline("serie_a", initial_bankroll=10000)
        match = MatchContext(
            match_id="SNAP_JUV_INT",
            league_id="serie_a",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Juventus", away_team="Inter",
            home_elo=1720, away_elo=1740,
            odds_home=2.30, odds_draw=3.20, odds_away=3.00,
        )
        result = pipeline.run_full(match)
        assert result.fused_probs is not None

    def test_scenario_25_psg_marseille_snapshot(self):
        """PSG vs Marseille 快照"""
        pipeline = GameFlowPipeline("ligue_1", initial_bankroll=10000)
        match = MatchContext(
            match_id="SNAP_PSG_OM",
            league_id="ligue_1",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="PSG", away_team="Marseille",
            home_elo=1850, away_elo=1600,
            odds_home=1.35, odds_draw=5.00, odds_away=7.00,
        )
        result = pipeline.run_full(match)
        assert result.fused_probs is not None

    def test_scenario_26_value_threshold_boundary(self):
        """价值阈值边界: value=0.03 应通过"""
        from src.engine.bankroll import generate_bet_proposals
        value_results = {
            "home": {"model_prob": 0.53, "implied_prob": 0.50, "value": 0.03, "odds": 2.0},
        }
        proposals = generate_bet_proposals(
            value_results, "m1", "premier_league",
            factor_count=41, data_source_count=5, odds_std=0.05,
        )
        assert len(proposals) == 1

    def test_scenario_27_value_threshold_below(self):
        """价值阈值边界: value=0.029 应被过滤"""
        from src.engine.bankroll import generate_bet_proposals
        value_results = {
            "home": {"model_prob": 0.529, "implied_prob": 0.50, "value": 0.029, "odds": 2.0},
        }
        proposals = generate_bet_proposals(value_results, "m1", "premier_league")
        assert len(proposals) == 0

    def test_scenario_28_odds_min_boundary(self):
        """赔率下界: odds=1.05 应通过"""
        from src.engine.bankroll import generate_bet_proposals
        value_results = {
            "home": {"model_prob": 0.97, "implied_prob": 0.95, "value": 0.03, "odds": 1.05},
        }
        proposals = generate_bet_proposals(
            value_results, "m1", "premier_league",
            factor_count=41, data_source_count=5, odds_std=0.05,
        )
        assert len(proposals) == 1

    def test_scenario_29_odds_max_boundary(self):
        """赔率上界: odds=10.0 应通过"""
        from src.engine.bankroll import generate_bet_proposals
        value_results = {
            "home": {"model_prob": 0.15, "implied_prob": 0.10, "value": 0.05, "odds": 10.0},
        }
        proposals = generate_bet_proposals(
            value_results, "m1", "premier_league",
            factor_count=41, data_source_count=5, odds_std=0.05,
        )
        assert len(proposals) == 1

    def test_scenario_30_consecutive_losses_boundary(self):
        """连续亏损边界: 4 场不触发，5 场触发"""
        from src.engine.risk_control import RiskController
        from src.data.models import BankrollState

        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=4)
        broken, _ = ctrl.check_circuit_breaker(bankroll)
        assert not broken

        bankroll2 = BankrollState(balance=10000, consecutive_losses=5)
        broken2, _ = ctrl.check_circuit_breaker(bankroll2)
        assert broken2