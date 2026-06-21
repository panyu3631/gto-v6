"""
GTO-GameFlow v5.5 — 策略扩展测试套件

测试内容:
1. AsianHandicapStrategy — 亚盘引擎 (让球线评估、结算、赔率生成)
2. OverUnderStrategy — 大小球引擎 (总进球分布、整数线走水、结算)
3. MPTPortfolioOptimizer — MPT 多策略组合 (协方差、优化、风险平价)
4. StrategyOrchestrator — 多策略编排器 (统一调度、聚合、结算记录)

用法:
    python tests/test_strategies.py
"""

import sys
import os
import math
import logging
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test_strategies")

PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition, test_name):
    """断言并统计"""
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        logger.info(f"  PASS: {test_name}")
    else:
        FAIL_COUNT += 1
        logger.error(f"  FAIL: {test_name}")


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def make_score_matrix(home_xg=1.5, away_xg=1.0, max_goals=7):
    """构建泊松比分矩阵"""
    from src.data.models import ScoreMatrix
    matrix = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = (np.exp(-home_xg) * (home_xg ** h) / max(1, math.factorial(h)) *
                 np.exp(-away_xg) * (away_xg ** a) / max(1, math.factorial(a)))
            matrix[(h, a)] = p
    total = sum(matrix.values())
    return ScoreMatrix(league_id="test", max_goals=max_goals, matrix={k: v / total for k, v in matrix.items()})


# ═══════════════════════════════════════════════════════════════
# 1. 亚盘策略测试
# ═══════════════════════════════════════════════════════════════

def test_asian_handicap():
    from src.strategies.asian_handicap import (
        AsianHandicapStrategy, HandicapLine, HandicapType,
        STANDARD_HANDICAP_LINES, AsianHandicapResult,
    )
    from src.data.models import AsianHandicapResult as AR

    logger.info("=== Asian Handicap Strategy ===")

    # 1.1 标准让球线
    check(len(STANDARD_HANDICAP_LINES) == 11, "STANDARD_HANDICAP_LINES count = 11")
    check(STANDARD_HANDICAP_LINES[0].line == 0.0, "first line = 0.0 (flat)")
    check(STANDARD_HANDICAP_LINES[0].htype == HandicapType.FLAT, "0.0 type = FLAT")
    check(STANDARD_HANDICAP_LINES[2].htype == HandicapType.HALF, "0.5 type = HALF")
    check(STANDARD_HANDICAP_LINES[4].htype == HandicapType.WHOLE, "1.0 type = WHOLE")
    check(STANDARD_HANDICAP_LINES[1].htype == HandicapType.QUARTER, "0.25 type = QUARTER")

    # 1.2 HandicapLine 属性
    hl = HandicapLine(1.25, HandicapType.QUARTER)
    check(hl.whole_part == 1, "whole_part of 1.25 = 1")
    check(hl.fraction_part == 0.25, "fraction_part of 1.25 = 0.25")

    # 1.3 策略初始化
    strategy = AsianHandicapStrategy(league_id="test", value_threshold=0.02)
    check(strategy.value_threshold == 0.02, "value_threshold = 0.02")
    check(strategy.min_odds == 1.70, "min_odds = 1.70")

    # 1.4 让球线类型判断
    check(strategy._get_handicap_type(0.0) == HandicapType.FLAT, "0.0 → FLAT")
    check(strategy._get_handicap_type(0.5) == HandicapType.HALF, "0.5 → HALF")
    check(strategy._get_handicap_type(1.0) == HandicapType.WHOLE, "1.0 → WHOLE")
    check(strategy._get_handicap_type(0.25) == HandicapType.QUARTER, "0.25 → QUARTER")
    check(strategy._get_handicap_type(0.75) == HandicapType.QUARTER, "0.75 → QUARTER")
    check(strategy._get_handicap_type(1.5) == HandicapType.HALF, "1.5 → HALF")
    check(strategy._get_handicap_type(-0.5) == HandicapType.HALF, "-0.5 → HALF")

    # 1.5 结算逻辑: 平手盘 (0.0)
    # 主队视角: 主队让0球
    check(strategy._evaluate_handicap_result(2, 1, 0.0, HandicapType.FLAT, "home") == AR.FULL_WIN,
          "FLAT home: 2-1 → FULL_WIN")
    check(strategy._evaluate_handicap_result(1, 1, 0.0, HandicapType.FLAT, "home") == AR.PUSH,
          "FLAT home: 1-1 → PUSH")
    check(strategy._evaluate_handicap_result(0, 1, 0.0, HandicapType.FLAT, "home") == AR.FULL_LOSS,
          "FLAT home: 0-1 → FULL_LOSS")

    # 1.6 半球盘 (0.5)
    check(strategy._evaluate_handicap_result(1, 0, 0.5, HandicapType.HALF, "home") == AR.FULL_WIN,
          "HALF home: 1-0 vs 0.5 → FULL_WIN")
    check(strategy._evaluate_handicap_result(0, 0, 0.5, HandicapType.HALF, "home") == AR.FULL_LOSS,
          "HALF home: 0-0 vs 0.5 → FULL_LOSS")

    # 1.7 一球盘 (1.0)
    check(strategy._evaluate_handicap_result(2, 0, 1.0, HandicapType.WHOLE, "home") == AR.FULL_WIN,
          "WHOLE home: 2-0 vs 1.0 → FULL_WIN")
    check(strategy._evaluate_handicap_result(1, 0, 1.0, HandicapType.WHOLE, "home") == AR.PUSH,
          "WHOLE home: 1-0 vs 1.0 → PUSH")
    check(strategy._evaluate_handicap_result(0, 0, 1.0, HandicapType.WHOLE, "home") == AR.FULL_LOSS,
          "WHOLE home: 0-0 vs 1.0 → FULL_LOSS")

    # 1.8 平半盘 (0.25)
    check(strategy._evaluate_handicap_result(2, 0, 0.25, HandicapType.QUARTER, "home") == AR.FULL_WIN,
          "QUARTER home: 2-0 vs 0.25 → FULL_WIN")
    check(strategy._evaluate_handicap_result(1, 0, 0.25, HandicapType.QUARTER, "home") == AR.FULL_WIN,
          "QUARTER home: 1-0 vs 0.25 → FULL_WIN (主胜=全赢)")
    check(strategy._evaluate_handicap_result(0, 0, 0.25, HandicapType.QUARTER, "home") == AR.HALF_LOSS,
          "QUARTER home: 0-0 vs 0.25 → HALF_LOSS")
    check(strategy._evaluate_handicap_result(0, 1, 0.25, HandicapType.QUARTER, "home") == AR.FULL_LOSS,
          "QUARTER home: 0-1 vs 0.25 → FULL_LOSS")

    # 1.9 半一盘 (0.75)
    check(strategy._evaluate_handicap_result(3, 0, 0.75, HandicapType.QUARTER, "home") == AR.FULL_WIN,
          "QUARTER home: 3-0 vs 0.75 → FULL_WIN")
    check(strategy._evaluate_handicap_result(1, 0, 0.75, HandicapType.QUARTER, "home") == AR.HALF_WIN,
          "QUARTER home: 1-0 vs 0.75 → HALF_WIN")
    check(strategy._evaluate_handicap_result(0, 0, 0.75, HandicapType.QUARTER, "home") == AR.FULL_LOSS,
          "QUARTER home: 0-0 vs 0.75 → FULL_LOSS")

    # 1.10 客队视角
    check(strategy._evaluate_handicap_result(0, 1, 0.0, HandicapType.FLAT, "away") == AR.FULL_WIN,
          "FLAT away: 0-1 → FULL_WIN")
    # 客队受让0.5: 主队1-0, 客队视角 => 主队让0.5, 客队投注 = 主队不赢即客赢
    # adjusted_home = 1 - 0.5 = 0.5, diff = 0.5 - 0 = 0.5 > 0 → 主赢, 客队投注输
    check(strategy._evaluate_handicap_result(1, 0, 0.5, HandicapType.HALF, "away") == AR.FULL_LOSS,
          "HALF away: 1-0 vs 0.5 → FULL_LOSS (主赢=客输)")

    # 1.11 概率计算
    sm = make_score_matrix(1.5, 1.0)
    prob = strategy._calculate_cover_probability(sm, 0.0, "home")
    check(0.4 < prob < 0.6, f"cover_prob 0.0 home ≈ {prob:.3f} (expected ~0.5)")

    prob = strategy._calculate_cover_probability(sm, 0.5, "home")
    check(0.2 < prob < 0.5, f"cover_prob 0.5 home ≈ {prob:.3f}")

    # 1.12 合成赔率生成
    odds = strategy.generate_synthetic_odds(sm)
    check(len(odds) > 0, "generate_synthetic_odds produces results")
    for line, o in odds.items():
        check(1.70 <= o["home"] <= 2.30, f"home odds {o['home']} in [1.70, 2.30]")
        check(1.70 <= o["away"] <= 2.30, f"away odds {o['away']} in [1.70, 2.30]")

    # 1.13 完整分析流程
    handicap_odds = {
        0.0: {"home": 1.90, "away": 2.00},
        0.5: {"home": 1.85, "away": 2.05},
        1.0: {"home": 1.92, "away": 1.98},
    }
    proposals = strategy.analyze(sm, handicap_odds, match_id="T1")
    check(len(proposals) > 0, f"analyze produces {len(proposals)} proposals")
    for p in proposals:
        check(p.match_id == "T1", "proposal has match_id")
        check(p.league_id == "test", "proposal has league_id")
        check(p.odds > 0, f"proposal odds = {p.odds} > 0")
        check(p.cover_prob > 0, f"cover_prob = {p.cover_prob} > 0")
        check(p.value >= strategy.value_threshold, f"value = {p.value:.4f} >= {strategy.value_threshold}")

    # 1.14 结算
    for p in proposals:
        result, pnl = strategy.settle(p, 2, 1)
        check(isinstance(result, AR), f"settle returns AsianHandicapResult: {result.value}")
        check(isinstance(pnl, float), f"settle pnl is float: {pnl}")

    # 1.15 结算: 走水
    if proposals:
        p = proposals[0]
        # 平手盘 1-1 走水
        result, pnl = strategy.settle(p, 1, 1)
        if p.handicap_line == 0.0:
            check(result == AR.PUSH, f"0.0 line 1-1 → PUSH")
            check(pnl == 0.0, "PUSH pnl = 0")

    logger.info("  Asian Handicap: ALL TESTS DONE")


# ═══════════════════════════════════════════════════════════════
# 2. 大小球策略测试
# ═══════════════════════════════════════════════════════════════

def test_over_under():
    from src.strategies.over_under import OverUnderStrategy, STANDARD_TOTALS_LINES

    logger.info("=== Over/Under Strategy ===")

    # 2.1 标准线
    check(len(STANDARD_TOTALS_LINES) == 7, "STANDARD_TOTALS_LINES count = 7")
    check(2.5 in STANDARD_TOTALS_LINES, "2.5 in standard lines")

    # 2.2 策略初始化
    strategy = OverUnderStrategy(league_id="test", value_threshold=0.02)
    check(strategy.value_threshold == 0.02, "value_threshold = 0.02")

    # 2.3 总进球分布构建
    sm = make_score_matrix(1.5, 1.0)
    dist = strategy._build_totals_distribution(sm)
    check(dist.avg_goals > 0, f"avg_goals = {dist.avg_goals:.2f} > 0")
    check(abs(dist.avg_goals - 2.5) < 1.5, f"avg_goals = {dist.avg_goals:.2f} ≈ 2.5")

    # 分布概率和应为 1
    total_p = sum(dist.distribution.values())
    check(abs(total_p - 1.0) < 0.01, f"distribution sum = {total_p:.4f} ≈ 1.0")

    # 2.4 Over/Under 概率
    over_2_5 = dist.over_prob(2.5)
    under_2_5 = dist.under_prob(2.5)
    check(0.1 < over_2_5 < 0.9, f"over 2.5 = {over_2_5:.3f}")
    check(0.1 < under_2_5 < 0.9, f"under 2.5 = {under_2_5:.3f}")

    # 2.5 整数线走水
    exact_2 = dist.exact_prob(2.0)
    check(exact_2 > 0, f"exact_prob(2.0) = {exact_2:.4f} > 0")
    exact_2_5 = dist.exact_prob(2.5)
    check(exact_2_5 == 0.0, "exact_prob(2.5) = 0 (non-integer)")

    # 2.6 合成赔率
    odds = strategy.generate_synthetic_odds(sm)
    check(len(odds) > 0, "generate_synthetic_odds produces results")
    for line, o in odds.items():
        check(1.70 <= o["over"] <= 2.30, f"over odds {o['over']} in [1.70, 2.30]")
        check(1.70 <= o["under"] <= 2.30, f"under odds {o['under']} in [1.70, 2.30]")

    # 2.7 完整分析
    totals_odds = {
        2.5: {"over": 1.90, "under": 2.00},
        3.0: {"over": 2.05, "under": 1.85},
    }
    proposals = strategy.analyze(sm, totals_odds, match_id="T1")
    check(len(proposals) > 0, f"analyze produces {len(proposals)} proposals")
    for p in proposals:
        check(p.match_id == "T1", "proposal has match_id")
        check(p.league_id == "test", "proposal has league_id")
        check(p.odds > 0, f"proposal odds = {p.odds} > 0")
        check(p.over_prob > 0, f"over_prob = {p.over_prob} > 0")
        check(p.value >= strategy.value_threshold, f"value = {p.value:.4f} >= threshold")

    # 2.8 结算
    for p in proposals:
        result, pnl = strategy.settle(p, 2, 1)
        check(result in ("win", "loss", "push"), f"settle result = {result}")
        check(isinstance(pnl, float), f"settle pnl = {pnl}")

    # 2.9 结算: Over 2.5, 总进球 3 → win
    result, pnl = strategy.settle(proposals[0], 2, 1)
    if proposals[0].side == "over" and proposals[0].totals_line == 2.5:
        check(result == "win", f"over 2.5, total=3 → win")
        check(pnl > 0, f"pnl = {pnl} > 0")

    # 2.10 结算: 整数线走水
    result, pnl = strategy.settle(proposals[0], 2, 0)
    if proposals[0].totals_line == 2.0:
        check(result == "push", "integer line push")
        check(pnl == 0.0, "push pnl = 0")

    logger.info("  Over/Under: ALL TESTS DONE")


# ═══════════════════════════════════════════════════════════════
# 3. MPT 投资组合优化器测试
# ═══════════════════════════════════════════════════════════════

def test_mpt_portfolio():
    from src.strategies.mpt_portfolio import (
        MPTPortfolioOptimizer, StrategyReturnSeries, CovarianceEstimator,
        analyze_strategy_correlation,
    )
    from src.data.models import StrategyAllocation, StrategyPortfolio

    logger.info("=== MPT Portfolio Optimizer ===")

    # 3.1 StrategyReturnSeries
    srs = StrategyReturnSeries(
        strategy_type="1x2",
        returns=[0.05, -0.10, 0.08, 0.03, -0.02, 0.06, 0.04, -0.05, 0.07, 0.02],
        total_bets=10, total_stake=5000, total_profit=250, win_rate=0.55, avg_odds=1.95,
    )
    check(abs(srs.mean_return - 0.018) < 0.02, f"mean_return = {srs.mean_return:.4f}")
    check(srs.volatility > 0, f"volatility = {srs.volatility:.4f} > 0")
    check(srs.n_bets == 10, "n_bets = 10")
    check(srs.total_bets == 10, "total_bets = 10")

    d = srs.to_dict()
    check("mean_return" in d, "to_dict has mean_return")
    check(d["strategy_type"] == "1x2", "to_dict has strategy_type")

    # 3.2 空序列
    empty = StrategyReturnSeries(strategy_type="empty")
    check(empty.mean_return == 0.0, "empty mean_return = 0")
    check(empty.volatility == 0.05, "empty volatility = 0.05 (default)")
    check(empty.sharpe == 0.0, "empty sharpe = 0")

    # 3.3 CovarianceEstimator — 单策略
    cov, names = CovarianceEstimator.estimate({"1x2": srs})
    check(cov.shape == (1, 1), "single strategy cov = 1x1")
    check(names == ["1x2"], "names = ['1x2']")

    # 3.4 CovarianceEstimator — 多策略
    asian = StrategyReturnSeries(
        strategy_type="asian_handicap",
        returns=[0.04, -0.08, 0.06, 0.02, -0.01, 0.05, 0.03, -0.04, 0.06, 0.01],
        total_bets=10,
    )
    ou = StrategyReturnSeries(
        strategy_type="over_under",
        returns=[0.06, -0.12, 0.10, 0.01, -0.03, 0.07, 0.03, -0.06, 0.08, 0.01],
        total_bets=10,
    )
    cov, names = CovarianceEstimator.estimate({"1x2": srs, "asian_handicap": asian, "over_under": ou})
    check(cov.shape == (3, 3), "three-strategy cov = 3x3")
    check(cov[0, 0] > 0, "diagonal > 0")
    # 对称性
    check(abs(cov[0, 1] - cov[1, 0]) < 1e-10, "cov is symmetric")

    # 3.5 CovarianceEstimator — 空数据
    empty_all = {
        "1x2": StrategyReturnSeries(strategy_type="1x2"),
        "asian_handicap": StrategyReturnSeries(strategy_type="asian_handicap"),
    }
    cov, names = CovarianceEstimator.estimate(empty_all)
    check(cov.shape == (2, 2), "empty data cov = 2x2")
    check(cov[0, 0] > 0, "empty data diagonal > 0")

    # 3.6 MPTPortfolioOptimizer
    opt = MPTPortfolioOptimizer(
        risk_free_rate=0.0,
        max_single_weight=0.6,
        min_single_weight=0.0,
        num_portfolios=5000,
    )
    check(opt.max_single_weight == 0.6, "max_single_weight = 0.6")
    check(opt.default_weights == {"1x2": 0.50, "asian_handicap": 0.25, "over_under": 0.25},
          "default_weights correct")

    # 3.7 优化 (充足数据)
    portfolio = opt.optimize(
        {"1x2": srs, "asian_handicap": asian, "over_under": ou},
        total_bankroll=100000,
    )
    check(isinstance(portfolio, StrategyPortfolio), "optimize returns StrategyPortfolio")
    check(len(portfolio.allocations) == 3, "3 allocations")
    check(abs(sum(a.weight for a in portfolio.allocations) - 1.0) < 0.01,
          "weights sum to 1.0")
    check(portfolio.total_bankroll == 100000, "total_bankroll = 100000")

    for a in portfolio.allocations:
        check(isinstance(a, StrategyAllocation), f"allocation is StrategyAllocation: {a.strategy_type}")
        check(0 <= a.weight <= 1, f"weight {a.weight} in [0, 1]")
        check(a.allocation > 0, f"allocation {a.allocation} > 0")

    # 3.8 优化 (数据不足, 应使用风险平价)
    short = StrategyReturnSeries(strategy_type="1x2", returns=[0.05, 0.03, -0.02])
    short2 = StrategyReturnSeries(strategy_type="asian_handicap", returns=[0.04, 0.02, -0.01])
    portfolio_small = opt.optimize(
        {"1x2": short, "asian_handicap": short2},
        total_bankroll=100000,
    )
    check(len(portfolio_small.allocations) == 2, "small data: 2 allocations")
    weight_sum = sum(a.weight for a in portfolio_small.allocations)
    check(abs(weight_sum - 1.0) < 0.01, f"small data weights sum = {weight_sum:.4f} ≈ 1.0")

    # 3.9 更新收益序列
    updated = opt.update_return_series(srs, new_return=0.10, stake=500, profit=50, odds=2.0, won=True)
    check(updated.total_bets == 11, "update: total_bets incremented")
    check(updated.total_profit == 300, "update: total_profit = 250 + 50")
    check(updated.returns[-1] == 0.10, "update: last return = 0.10")

    # 3.10 build_from_results
    results = [
        {"stake": 500, "profit_loss": 50, "odds": 2.0, "won": True},
        {"stake": 500, "profit_loss": -500, "odds": 1.9, "won": False},
    ]
    built = opt.build_from_results("1x2", results)
    check(built.total_bets == 2, "build_from_results: 2 bets")
    check(built.total_profit == -450, "build_from_results: profit = -450")
    check(built.win_rate == 0.5, "build_from_results: win_rate = 0.5")

    # 3.11 策略相关性分析
    corr = analyze_strategy_correlation({"1x2": srs, "asian_handicap": asian, "over_under": ou})
    check("1x2" in corr, "corr has 1x2")
    check(corr["1x2"]["1x2"] == 1.0, "self correlation = 1.0")
    check(-1.0 <= corr["1x2"]["asian_handicap"] <= 1.0, "corr in [-1, 1]")

    logger.info("  MPT Portfolio: ALL TESTS DONE")


# ═══════════════════════════════════════════════════════════════
# 4. 策略编排器测试
# ═══════════════════════════════════════════════════════════════

def test_strategy_orchestrator():
    from src.strategies.strategy_orchestrator import (
        StrategyOrchestrator, MultiStrategyResult, create_orchestrator,
    )
    from src.data.models import (
        MatchContext, ScoreMatrix, BetProposal, BetSelection,
        StrategyType, AsianHandicapProposal, TotalsProposal,
        StrategyPortfolio, StrategyAllocation,
    )

    logger.info("=== Strategy Orchestrator ===")

    # 4.1 创建编排器
    orch = create_orchestrator("bundesliga")
    check(orch.enable_asian, "asian enabled")
    check(orch.enable_over_under, "over_under enabled")
    check(orch.league_id == "bundesliga", "league_id = bundesliga")

    orch2 = create_orchestrator("serie_a")
    check(orch2.mpt.default_weights != orch.mpt.default_weights,
          "different leagues have different default weights")

    # 4.2 构建测试数据
    sm = make_score_matrix(1.5, 1.0)
    match = MatchContext(
        match_id="BAYvsDOR", league_id="bundesliga", season="2025-26",
        matchday=10, kickoff_time=datetime.now(),
        home_team="Bayern", away_team="Dortmund",
        odds_home=1.80, odds_draw=3.60, odds_away=4.20,
    )

    # 4.3 运行编排 (无 1X2 proposals)
    result = orch.run(match=match, score_matrix=sm, x2_proposals=[], total_bankroll=100000)
    check(isinstance(result, MultiStrategyResult), "run returns MultiStrategyResult")
    check(result.match_id == "BAYvsDOR", "match_id preserved")
    check(result.strategy_count >= 2, f"strategy_count = {result.strategy_count} >= 2")
    check(len(result.active_strategies) >= 2, f"active_strategies: {result.active_strategies}")
    check(len(result.asian_proposals) > 0, f"asian_proposals: {len(result.asian_proposals)}")
    check(len(result.totals_proposals) > 0, f"totals_proposals: {len(result.totals_proposals)}")

    # 4.4 聚合投注建议
    check(len(result.unified_proposals) > 0, f"unified_proposals: {len(result.unified_proposals)}")
    for p in result.unified_proposals:
        check(isinstance(p, BetProposal), "unified proposal is BetProposal")
        check(p.strategy_type in ("1x2", "asian_handicap", "over_under"),
              f"strategy_type: {p.strategy_type}")
        check(p.strategy_weight > 0, f"strategy_weight = {p.strategy_weight} > 0")

    # 4.5 MPT 组合
    check(result.portfolio is not None, "portfolio is not None")
    check(isinstance(result.portfolio, StrategyPortfolio), "portfolio is StrategyPortfolio")
    check(abs(sum(a.weight for a in result.portfolio.allocations) - 1.0) < 0.01,
          "portfolio weights sum to 1.0")

    # 4.6 摘要
    summary = result.summary()
    check("match_id" in summary, "summary has match_id")
    check("total_proposals" in summary, "summary has total_proposals")
    check("portfolio" in summary, "summary has portfolio")

    # 4.7 结算记录
    orch.record_settlement("asian_handicap", roi=0.05, stake=500, profit=25, odds=1.92, won=True)
    orch.record_settlement("over_under", roi=-0.10, stake=500, profit=-50, odds=1.90, won=False)
    orch.record_settlement("1x2", roi=0.08, stake=500, profit=40, odds=2.10, won=True)

    series = orch.get_all_series()
    check(series["asian_handicap"].total_bets == 1, "asian: 1 bet recorded")
    check(series["over_under"].total_bets == 1, "over_under: 1 bet recorded")
    check(series["1x2"].total_bets == 1, "1x2: 1 bet recorded")
    check(series["asian_handicap"].total_profit == 25, "asian: profit = 25")
    check(series["over_under"].total_profit == -50, "over_under: profit = -50")

    # 4.8 未知策略类型
    orch.record_settlement("unknown", roi=0.05, stake=500, profit=25, odds=2.0, won=True)
    check(True, "unknown strategy type handled gracefully")

    # 4.9 相关性矩阵
    corr = orch.get_correlation_matrix()
    check(isinstance(corr, dict), "correlation_matrix returns dict")

    # 4.10 禁用策略
    orch_disabled = StrategyOrchestrator(league_id="test", enable_asian=False, enable_over_under=False)
    result_disabled = orch_disabled.run(match=match, score_matrix=sm, x2_proposals=[], total_bankroll=100000)
    check(len(result_disabled.asian_proposals) == 0, "asian disabled → 0 proposals")
    check(len(result_disabled.totals_proposals) == 0, "over_under disabled → 0 proposals")

    # 4.11 带 1X2 proposals 的编排
    x2_proposals = [
        BetProposal(
            match_id="BAYvsDOR", selection=BetSelection.HOME_WIN,
            odds=1.80, model_prob=0.60, implied_prob=0.556,
            value=0.044, kelly_stake=500, adjusted_stake=500,
            priority_score=0.15, league_id="bundesliga",
        ),
    ]
    result_with_x2 = orch.run(
        match=match, score_matrix=sm,
        x2_proposals=x2_proposals, total_bankroll=100000,
    )
    check(len(result_with_x2.x2_proposals) == 1, "1 x2 proposal preserved")
    check("1x2" in result_with_x2.active_strategies, "1x2 in active_strategies")
    check(len(result_with_x2.unified_proposals) >= 3,
          f"unified >= 3 (1x2+asian+totals) = {len(result_with_x2.unified_proposals)}")

    logger.info("  Strategy Orchestrator: ALL TESTS DONE")


# ═══════════════════════════════════════════════════════════════
# 5. 集成测试: 端到端多策略流程
# ═══════════════════════════════════════════════════════════════

def test_integration():
    """端到端: 从 ScoreMatrix 到多策略投注建议"""
    from src.strategies.strategy_orchestrator import create_orchestrator
    from src.data.models import MatchContext, ScoreMatrix, BetProposal, BetSelection

    logger.info("=== Integration Test ===")

    sm = make_score_matrix(1.5, 1.0)
    match = MatchContext(
        match_id="INT001", league_id="bundesliga", season="2025-26",
        matchday=10, kickoff_time=datetime.now(),
        home_team="TeamA", away_team="TeamB",
        odds_home=1.80, odds_draw=3.60, odds_away=4.20,
    )

    orch = create_orchestrator("bundesliga")

    # 模拟 5 轮投注 + 结算
    for round_num in range(5):
        # 生成模拟实际比分
        h = np.random.poisson(1.5)
        a = np.random.poisson(1.0)

        result = orch.run(match=match, score_matrix=sm, x2_proposals=[], total_bankroll=100000)

        # 结算亚盘和大小球
        for p in result.asian_proposals:
            orch.asian_strategy.settle(p, h, a)
            asian_result, asian_pnl = orch.asian_strategy.settle(p, h, a)
            roi = asian_pnl / max(p.kelly_stake, 1)
            orch.record_settlement(
                "asian_handicap", roi=roi,
                stake=p.kelly_stake, profit=asian_pnl,
                odds=p.odds,
                won=(asian_result.value in ("full_win", "half_win")),
            )

        for p in result.totals_proposals:
            totals_result, totals_pnl = orch.over_under_strategy.settle(p, h, a)
            roi = totals_pnl / max(p.kelly_stake, 1)
            orch.record_settlement(
                "over_under", roi=roi,
                stake=p.kelly_stake, profit=totals_pnl,
                odds=p.odds,
                won=(totals_result == "win"),
            )

    # 验证累积
    series = orch.get_all_series()
    check(series["asian_handicap"].total_bets > 0, f"asian bets: {series['asian_handicap'].total_bets} > 0")
    check(series["over_under"].total_bets > 0, f"over_under bets: {series['over_under'].total_bets} > 0")

    logger.info("  Integration: ALL TESTS DONE")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 确保导入路径正确
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    test_asian_handicap()
    test_over_under()
    test_mpt_portfolio()
    test_strategy_orchestrator()
    test_integration()

    print()
    print(f"═══════════════════════════════════════════")
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"═══════════════════════════════════════════")

    if FAIL_COUNT > 0:
        sys.exit(1)
    else:
        sys.exit(0)