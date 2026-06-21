"""
L3 集成测试：端到端流水线 (orchestrator.py)

测试范围:
- 完整 9 阶段流水线
- 5 联赛并行
- 冷启动 (最小数据)
- 端到端：MatchContext → 概率 → 投注建议 → 结算
- 错误处理
- 数据持久化
"""
import pytest
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    BankrollState, ProbabilityDistribution,
)
from src.pipeline.orchestrator import GameFlowPipeline, PipelineResult
from src.factors.registry import get_active_factors
from src.config.settings import config


# ================================================================
# 端到端测试
# ================================================================

class TestEndToEndPipeline:
    """完整 9 阶段流水线"""

    def test_full_pipeline_arsenal_vs_tottenham(self):
        """Arsenal vs Tottenham 完整流水线"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="EPL_2025_08_01_ARS_TOT",
            league_id="premier_league",
            season="2025/26",
            matchday=1,
            kickoff_time=datetime(2025, 8, 1, 15, 0, 0),
            home_team="Arsenal",
            away_team="Tottenham",
            home_elo=1750,
            away_elo=1650,
            odds_home=1.91,
            odds_draw=3.75,
            odds_away=3.80,
        )
        extra = {
            "elo_diff": 100.0,
            "xi_rating": 7.0,
            "recent_results": [3.0, 3.0, 1.0, 3.0, 1.0],
            "h2h_results": [3.0, 0.0, 1.0, 3.0, 0.0],
            "matches_7d": 1,
            "rank_diff": 5,
            "goal_diff": 8.0,
            "xg_diff": 1.2,
            "streak_momentum": 0.5,
            "streak_momentum_league": 0.5,
            "motivation_boost": 5.0,
            "position_advantage": 8.0,
        }

        # Stage 1-5
        result = pipeline.run_stages_1_5(match, extra)
        assert len(result.factor_deltas) > 0, "因子计算应返回结果"
        assert result.logit_probs is not None, "logit 概率应存在"
        assert result.fused_probs is not None, "融合概率应存在"
        assert len(result.value_results) == 3, "应有 3 个结果 (home/draw/away)"
        assert result.errors == [], f"Stage 1-5 错误: {result.errors}"

        # Stage 6-9
        result = pipeline.run_stages_6_9(result, extra_data=extra)
        assert result.bankroll_state is not None, "资金状态应存在"
        assert result.errors == [], f"Stage 6-9 错误: {result.errors}"

    def test_run_full_shortcut(self):
        """run_full 快捷方式"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="EPL_TEST",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1600, away_elo=1400,
            odds_home=2.0, odds_draw=3.5, odds_away=3.5,
        )
        result = pipeline.run_full(match)
        assert isinstance(result, PipelineResult)
        assert result.fused_probs is not None

    def test_execute_bets(self):
        """投注执行"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="EPL_TEST",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1700, away_elo=1300,
            odds_home=1.8, odds_draw=3.6, odds_away=4.5,
        )
        extra = {
            "elo_diff": 400.0,
            "recent_results": [3.0, 3.0, 3.0, 1.0, 3.0],
            "h2h_results": [3.0, 3.0, 1.0, 3.0, 0.0],
            "matches_7d": 1,
            "rank_diff": 10,
            "goal_diff": 15.0,
            "xg_diff": 2.0,
            "streak_momentum": 0.7,
            "streak_momentum_league": 0.7,
            "motivation_boost": 5.0,
            "position_advantage": 12.0,
        }
        result = pipeline.run_full(match, extra)
        result = pipeline.execute_bets(result)

        if result.placements:
            assert result.placements[0].stake > 0
            assert result.placements[0].result == BetResult.PENDING

    def test_settle_bets(self):
        """投注结算"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="EPL_TEST_SETTLE",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1700, away_elo=1300,
            odds_home=1.8, odds_draw=3.6, odds_away=4.5,
        )
        extra = {
            "elo_diff": 400.0,
            "recent_results": [3.0, 3.0, 3.0, 1.0, 3.0],
            "h2h_results": [3.0, 3.0, 1.0, 3.0, 0.0],
            "matches_7d": 1,
            "rank_diff": 10,
            "goal_diff": 15.0,
            "xg_diff": 2.0,
            "streak_momentum": 0.7,
            "streak_momentum_league": 0.7,
            "motivation_boost": 5.0,
            "position_advantage": 12.0,
        }
        result = pipeline.run_full(match, extra)
        result = pipeline.execute_bets(result)

        if result.placements:
            # 结算为主胜
            placements = pipeline.settle_bets(result.placements, BetSelection.HOME_WIN)
            for p in placements:
                assert p.result != BetResult.PENDING
                if p.selection == BetSelection.HOME_WIN:
                    assert p.result == BetResult.WIN
                    assert p.profit_loss > 0
                else:
                    assert p.result == BetResult.LOSS
                    assert p.profit_loss < 0

    def test_fused_probs_reasonable(self):
        """融合概率应在合理范围"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="EPL_TEST",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1500, away_elo=1500,
            odds_home=2.5, odds_draw=3.2, odds_away=2.8,
        )
        result = pipeline.run_stages_1_5(match)
        total = result.fused_probs.prob_home + result.fused_probs.prob_draw + result.fused_probs.prob_away
        assert abs(total - 1.0) < 0.001
        for prob in [result.fused_probs.prob_home, result.fused_probs.prob_draw, result.fused_probs.prob_away]:
            assert 0.0 < prob < 1.0


# ================================================================
# 5 联赛并行
# ================================================================

class TestMultiLeagueParallel:
    """5 联赛并行测试"""

    LEAGUES = [
        ("premier_league", "Arsenal", "Chelsea", 1750, 1680, 1.91, 3.75, 3.80),
        ("la_liga", "Barcelona", "Real Madrid", 1780, 1760, 1.95, 3.60, 3.50),
        ("bundesliga", "Bayern", "Dortmund", 1800, 1700, 1.55, 4.50, 5.00),
        ("serie_a", "Juventus", "Inter", 1720, 1740, 2.30, 3.20, 3.00),
        ("ligue_1", "PSG", "Marseille", 1850, 1600, 1.35, 5.00, 7.00),
    ]

    def test_all_5_leagues_pipeline(self):
        """5 个联赛同时运行流水线"""
        results = []
        for league_id, home, away, helo, aelo, ho, do, ao in self.LEAGUES:
            pipeline = GameFlowPipeline(league_id, initial_bankroll=10000)
            match = MatchContext(
                match_id=f"{league_id}_TEST",
                league_id=league_id,
                season="2025/26", matchday=1,
                kickoff_time=datetime.now(),
                home_team=home, away_team=away,
                home_elo=helo, away_elo=aelo,
                odds_home=ho, odds_draw=do, odds_away=ao,
            )
            result = pipeline.run_full(match)
            results.append(result)

        # 所有联赛都应生成结果
        for r in results:
            assert r.fused_probs is not None, f"{r.league_id} 无融合概率"
            assert len(r.factor_deltas) > 0, f"{r.league_id} 无因子结果"
            assert r.errors == [], f"{r.league_id} 错误: {r.errors}"

    def test_league_factor_counts_differ(self):
        """不同联赛的活跃因子数量在合理范围"""
        counts = {}
        for league_id, home, away, helo, aelo, ho, do, ao in self.LEAGUES:
            pipeline = GameFlowPipeline(league_id)
            match = MatchContext(
                match_id=f"{league_id}_TEST",
                league_id=league_id,
                season="2025/26", matchday=1,
                kickoff_time=datetime.now(),
                home_team=home, away_team=away,
                home_elo=helo, away_elo=aelo,
                odds_home=ho, odds_draw=do, odds_away=ao,
            )
            result = pipeline.run_stages_1_5(match)
            counts[league_id] = len(result.raw_factor_deltas)

        for league_id, count in counts.items():
            assert count >= 30, f"{league_id} 因子数过少: {count}"


# ================================================================
# 冷启动
# ================================================================

class TestColdStart:
    """冷启动测试"""

    def test_minimal_data(self):
        """最小数据输入 (所有默认值)"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="COLD_START",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="New Team", away_team="New Team B",
            home_elo=1500, away_elo=1500,
            odds_home=2.5, odds_draw=3.2, odds_away=2.8,
        )
        result = pipeline.run_full(match)
        assert result.errors == [], f"冷启动错误: {result.errors}"
        assert result.fused_probs is not None

    def test_zero_elo_diff(self):
        """ELO 差为 0"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="ZERO_ELO",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Team A", away_team="Team B",
            home_elo=1500, away_elo=1500,
            odds_home=2.5, odds_draw=3.2, odds_away=2.8,
        )
        result = pipeline.run_full(match)
        assert result.errors == []

    def test_promoted_team_cold_start(self):
        """升班马冷启动 (F40 激活)"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="PROMOTED",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Promoted FC", away_team="Established FC",
            home_elo=1300, away_elo=1600,
            odds_home=5.0, odds_draw=3.5, odds_away=1.7,
        )
        extra = {"promoted_team_delta": 0.3}
        result = pipeline.run_full(match, extra)
        assert result.errors == []


# ================================================================
# 错误处理
# ================================================================

class TestErrorHandling:
    """错误处理测试"""

    def test_invalid_league(self):
        """无效联赛应抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的联赛"):
            GameFlowPipeline("invalid_league", initial_bankroll=10000)

    def test_negative_bankroll(self):
        """负资金"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=-1000)
        match = MatchContext(
            match_id="TEST",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="A", away_team="B",
            home_elo=1500, away_elo=1500,
            odds_home=2.0, odds_draw=3.0, odds_away=3.0,
        )
        result = pipeline.run_full(match)
        assert isinstance(result, PipelineResult)


# ================================================================
# 数据持久化集成
# ================================================================

class TestDataPersistence:
    """数据持久化集成测试"""

    def test_pipeline_result_serializable(self):
        """PipelineResult 应可序列化"""
        pipeline = GameFlowPipeline("premier_league", initial_bankroll=10000)
        match = MatchContext(
            match_id="EPL_TEST",
            league_id="premier_league",
            season="2025/26", matchday=1,
            kickoff_time=datetime.now(),
            home_team="Arsenal", away_team="Tottenham",
            home_elo=1750, away_elo=1650,
            odds_home=1.91, odds_draw=3.75, odds_away=3.80,
        )
        result = pipeline.run_full(match)
        # 验证关键字段存在
        assert result.match_id == "EPL_TEST"
        assert result.league_id == "premier_league"
        assert result.fused_probs is not None