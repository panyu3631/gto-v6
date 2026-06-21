#!/usr/bin/env python3
"""
GTO-GameFlow v5.10.8 — 联赛表现深度分析

分析维度:
1. 各联赛按赔率区间的 ROI 分布
2. 各联赛按策略类型的 ROI 分布
3. 各联赛按模型概率与实际胜率的校准曲线
4. 各联赛按投注方向(主/平/客)的 ROI
5. 各联赛零值因子比例 vs 数据质量
6. 法甲专项: 找出系统性亏损根因

用法:
    python tests/analyze_league_performance.py
"""

import sys
import os
import csv
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models import MatchContext, BetSelection
from src.data.odds_provider import OddsProvider, get_odds_provider, MatchOddsBundle
from src.engine.elo_cold_start import EloColdStart
from src.engine.unified_bayesian_shrinkage import create_shrinkage_for_league
from src.engine.unified_decision_gate import (
    UnifiedDecisionGate, UnifiedProposal, proposals_to_unified,
    create_decision_gate_for_league,
)
from src.engine.unified_bankroll_manager import (
    UnifiedBankrollManager, create_bankroll_manager_for_league,
)
from src.engine.market_realism_integrator import (
    MarketRealismIntegrator, create_integrator_for_league,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.strategies.asian_handicap import AsianHandicapResult
from src.strategies.strategy_orchestrator import StrategyOrchestrator
from src.data.enhanced_data_provider import EnhancedDataProvider
from src.data.match_stats_enricher import MatchStatsEnricher

# 复用 test_phase6_walk_forward 的数据加载函数
from test_phase6_walk_forward import (
    _load_dated_matches, _extract_odds_dispersion, _extract_opening_probs,
    _get_league_dispersion, DatedMatch, EloTracker, TeamStatsTracker,
    ALL_SEASONS, LEAGUES, WINDOW_CONFIG, INITIAL_BANKROLL,
)

# ============================================================
# 详细投注记录收集
# ============================================================

def run_detailed_analysis() -> Dict[str, Any]:
    """运行带详细记录的回测"""
    odds_provider = get_odds_provider()

    csv_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "data", "historical_odds",
    )
    enhanced_provider = EnhancedDataProvider(
        csv_dir=csv_dir,
        leagues=LEAGUES,
        seasons=ALL_SEASONS,
    )
    stats_enricher = MatchStatsEnricher(
        csv_dir=csv_dir,
        leagues=LEAGUES,
        seasons=ALL_SEASONS,
    )

    # 使用所有赛季
    train_seasons = ALL_SEASONS[:7]  # 2014-15 to 2020-21
    val_season = ALL_SEASONS[7:8]    # 2021-22
    test_seasons = ALL_SEASONS[8:]   # 2022-23, 2023-24

    # OOS 窗口
    oos_seasons = test_seasons

    elo_tracker = EloTracker()
    for season in train_seasons + val_season:
        for lid in LEAGUES:
            for day_matches in _load_dated_matches(lid, season, odds_provider):
                for dm in day_matches:
                    elo_tracker.update_elo(lid, dm.home_team, dm.away_team,
                                       int(dm.row.get("FTHG", 0) or 0), int(dm.row.get("FTAG", 0) or 0))

    # 详细记录
    all_bet_records = []  # 每条投注的详细记录
    league_odds_bins = {lid: defaultdict(lambda: [0, 0, 0.0, 0.0]) for lid in LEAGUES}  # bins→[bets, wins, staked, returned]
    league_probs_bins = {lid: defaultdict(lambda: [0, 0, 0.0, 0.0]) for lid in LEAGUES}
    league_direction = {lid: {"home": [0, 0, 0.0, 0.0], "draw": [0, 0, 0.0, 0.0], "away": [0, 0, 0.0, 0.0]} for lid in LEAGUES}
    league_factor_stats = {lid: {"total_nonzero": 0, "total_bets": 0, "avg_nonzero": 0.0} for lid in LEAGUES}

    pipelines = {}
    orchestrators = {}
    decision_gates = {}
    bankroll_mgrs = {}
    market_integrators = {}

    for lid in LEAGUES:
        pipeline = GameFlowPipeline(lid, initial_bankroll=INITIAL_BANKROLL)
        pipeline.unified_shrinkage = create_shrinkage_for_league(lid)
        pipelines[lid] = pipeline
        orchestrators[lid] = StrategyOrchestrator(
            lid,
            asian_config={"value_threshold": 0.010, "confidence_threshold": 0.10},
            over_under_config={
                "value_threshold": 0.015, "confidence_threshold": 0.20,
                "min_odds": 1.85, "dispersion": _get_league_dispersion(lid),
            },
        )
        decision_gates[lid] = create_decision_gate_for_league(lid)
        bankroll_mgrs[lid] = create_bankroll_manager_for_league(lid, INITIAL_BANKROLL)
        market_integrators[lid] = create_integrator_for_league(lid)

    team_stats = TeamStatsTracker()
    matchup_count = 0

    for season in oos_seasons:
        print(f"  分析赛季: {season}...")
        all_league_dates = {}
        for lid in LEAGUES:
            all_league_dates[lid] = _load_dated_matches(lid, season, odds_provider)

        all_dates = defaultdict(list)
        for lid in LEAGUES:
            for day_matches in all_league_dates[lid]:
                if day_matches:
                    date = day_matches[0].date
                    for dm in day_matches:
                        all_dates[date].append((lid, dm))

        for date in sorted(all_dates.keys()):
            day_matches = all_dates[date]
            for lid, dm in day_matches:
                match = dm.match
                bundle = odds_provider.get_odds_for_match(
                    dm.row, f"{lid}_{dm.home_team}_{dm.away_team}_{date.strftime('%Y%m%d')}",
                    dm.home_team, dm.away_team, date,
                )
                odds_h = bundle.odds_home
                odds_d = bundle.odds_draw
                odds_a = bundle.odds_away
                if not (odds_h and odds_d and odds_a and odds_h > 0 and odds_d > 0 and odds_a > 0):
                    continue

                integrator = market_integrators[lid]
                mkt_adj = integrator.process_match(odds_h, odds_d, odds_a, season=season)
                if mkt_adj.skip:
                    continue
                odds_h, odds_d, odds_a = mkt_adj.adjusted_home, mkt_adj.adjusted_draw, mkt_adj.adjusted_away

                row = dm.row
                home_goals = int(row.get("FTHG", 0) or 0)
                away_goals = int(row.get("FTAG", 0) or 0)
                ftr = row.get("FTR", "").strip()
                if ftr == "H":
                    actual_outcome = "home_win"
                elif ftr == "D":
                    actual_outcome = "draw"
                elif ftr == "A":
                    actual_outcome = "away_win"
                else:
                    continue

                outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW, "away_win": BetSelection.AWAY_WIN}
                actual = outcome_map[actual_outcome]

                home_elo = elo_tracker.get_elo(lid, dm.home_team)
                away_elo = elo_tracker.get_elo(lid, dm.away_team)
                matchup_count += 1

                match_ctx = MatchContext(
                    match_id=f'{lid}_s_{matchup_count}',
                    league_id=lid, season=season, matchday=matchup_count,
                    kickoff_time=date,
                    home_team=dm.home_team, away_team=dm.away_team,
                    home_elo=home_elo, away_elo=away_elo,
                    odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
                )

                extra = {
                    'elo_diff': home_elo - away_elo,
                    'recent_results': team_stats.get_recent_form(lid, dm.home_team),
                    'h2h_results': [1.5, 1.5, 1.5, 1.5, 1.5],
                    'rank_diff': 0,
                    'goal_diff': team_stats.get_goal_diff(lid, dm.home_team) - team_stats.get_goal_diff(lid, dm.away_team),
                    'xg_diff': 0,
                    'streak_momentum': team_stats.get_streak_momentum(lid, dm.home_team),
                    'streak_momentum_league': team_stats.get_streak_momentum(lid, dm.away_team),
                    'match_phase': 1.0,
                    'data_source_count': _extract_odds_dispersion(row)[1],
                    'odds_std': _extract_odds_dispersion(row)[0],
                    'opening_probs': _extract_opening_probs(row),
                }

                if enhanced_provider:
                    extra = enhanced_provider.get_enhanced_data(
                        league=lid, season=season, home_team=dm.home_team,
                        away_team=dm.away_team, match_date=date,
                        existing_extra=extra, stats_enricher=stats_enricher,
                    )

                pipeline_result = pipelines[lid].run_full(match_ctx, extra_data=extra)

                # 因子统计
                if pipeline_result.factor_deltas:
                    nonzero = sum(1 for d in pipeline_result.factor_deltas.values()
                                  if isinstance(d, dict) and (
                                      abs(d.get('home', 0)) > 0.0001 or
                                      abs(d.get('draw', 0)) > 0.0001 or
                                      abs(d.get('away', 0)) > 0.0001))
                    total_factors = len(pipeline_result.factor_deltas)
                    league_factor_stats[lid]["total_nonzero"] += nonzero
                    league_factor_stats[lid]["total_bets"] += 1

                # 收集提案
                x2_unified = []
                if pipeline_result.proposals:
                    for p in pipeline_result.proposals:
                        x2_unified.append(UnifiedProposal(
                            match_id=p.match_id, strategy="1x2",
                            selection=p.selection.value if hasattr(p.selection, 'value') else str(p.selection),
                            odds=p.odds, model_prob=p.model_prob, implied_prob=p.implied_prob,
                            value=p.value, kelly_stake=p.kelly_stake,
                            confidence=getattr(p, 'confidence', 0.5),
                            priority_score=getattr(p, 'priority_score', 0.0),
                            league_id=lid, original=p,
                        ))

                asian_ou_unified = []
                try:
                    if bundle.has_real_asian or bundle.has_real_totals:
                        multi = orchestrators[lid].run(
                            match=match_ctx, score_matrix=pipeline_result.poisson_score_matrix,
                            handicap_odds=bundle.asian_odds if bundle.has_real_asian else {},
                            totals_odds=bundle.totals_odds if bundle.has_real_totals else {},
                            total_bankroll=INITIAL_BANKROLL,
                            strip_margin_asian=True, strip_margin_totals=True,
                        )
                        asian_ou_unified = proposals_to_unified(
                            asian_proposals=multi.asian_proposals,
                            totals_proposals=multi.totals_proposals,
                        )
                except Exception:
                    pass

                all_unified = x2_unified + asian_ou_unified
                if not all_unified:
                    continue

                decision = decision_gates[lid].evaluate(all_unified, 0)
                if not decision.approved:
                    continue

                allocation = bankroll_mgrs[lid].allocate(decision.approved, 0, 0)

                for sp in allocation.proposals:
                    orig = sp.original
                    if orig is None:
                        continue
                    stake = sp.final_stake

                    if sp.strategy == "1x2":
                        sel = sp.selection
                        bet_sel_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW, "away_win": BetSelection.AWAY_WIN}
                        bet_sel = bet_sel_map.get(sel)
                        if bet_sel is None:
                            continue
                        is_win = (actual == bet_sel)
                        pnl = stake * (sp.odds - 1.0) if is_win else -stake
                    elif sp.strategy == "asian_handicap":
                        ar, pnl = orchestrators[lid].asian_strategy.settle(orig, home_goals, away_goals)
                        is_win = ar in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                    elif sp.strategy == "over_under":
                        tr, pnl = orchestrators[lid].over_under_strategy.settle(orig, home_goals, away_goals)
                        is_win = (tr == "win")
                    else:
                        continue

                    can_exec, _ = integrator.check_liquidity(stake, sp.strategy)
                    if not can_exec:
                        continue

                    settlement = integrator.process_settlement(pnl, stake, sp.strategy)
                    net_pnl = settlement.net_profit

                    # 记录详情
                    record = {
                        "league": lid,
                        "season": season,
                        "home_team": dm.home_team,
                        "away_team": dm.away_team,
                        "strategy": sp.strategy,
                        "selection": sp.selection,
                        "odds": sp.odds,
                        "stake": stake,
                        "model_prob": sp.model_prob,
                        "implied_prob": getattr(orig, 'implied_prob', 1.0 / sp.odds) if orig else 1.0 / sp.odds,
                        "value": sp.value,
                        "confidence": getattr(orig, 'confidence', sp.priority_score) if orig else sp.priority_score,
                        "is_win": is_win,
                        "pnl": net_pnl,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "actual_outcome": actual_outcome,
                        "elo_diff": home_elo - away_elo,
                        "factor_nonzero_rate": nonzero / max(total_factors, 1) if total_factors > 0 else 0,
                    }
                    all_bet_records.append(record)

                    # 赔率分桶
                    if sp.odds < 1.5:
                        odds_bin = "1.0-1.5"
                    elif sp.odds < 2.0:
                        odds_bin = "1.5-2.0"
                    elif sp.odds < 2.5:
                        odds_bin = "2.0-2.5"
                    elif sp.odds < 3.0:
                        odds_bin = "2.5-3.0"
                    elif sp.odds < 4.0:
                        odds_bin = "3.0-4.0"
                    elif sp.odds < 5.0:
                        odds_bin = "4.0-5.0"
                    else:
                        odds_bin = "5.0+"

                    lb = league_odds_bins[lid][odds_bin]
                    lb[0] += 1; lb[2] += stake
                    if is_win:
                        lb[1] += 1; lb[3] += stake + net_pnl

                    # 概率分桶
                    prob_bin = f"{int(sp.model_prob * 10) / 10:.1f}-{int(sp.model_prob * 10) / 10 + 0.1:.1f}"
                    pb = league_probs_bins[lid][prob_bin]
                    pb[0] += 1; pb[2] += stake
                    if is_win:
                        pb[1] += 1; pb[3] += stake + net_pnl

                    # 方向统计
                    direction = sp.selection if sp.strategy == "1x2" else sp.selection
                    for dkey in ["home", "draw", "away"]:
                        if dkey in direction.lower():
                            ld = league_direction[lid][dkey]
                            ld[0] += 1; ld[2] += stake
                            if is_win:
                                ld[1] += 1; ld[3] += stake + net_pnl
                            break
                    else:
                        # 亚盘/大小球归属
                        if "home" in str(direction).lower() or "h" in str(direction).lower():
                            ld = league_direction[lid]["home"]
                        elif "away" in str(direction).lower() or "a" in str(direction).lower():
                            ld = league_direction[lid]["away"]
                        else:
                            ld = league_direction[lid]["draw"]
                        ld[0] += 1; ld[2] += stake
                        if is_win:
                            ld[1] += 1; ld[3] += stake + net_pnl

                # Elo 更新
                elo_tracker.update_elo(lid, dm.home_team, dm.away_team,
                                       home_goals, away_goals)
                team_stats.record_match(lid, dm.home_team, dm.away_team,
                                  home_goals, away_goals)

    return {
        "all_bet_records": all_bet_records,
        "league_odds_bins": league_odds_bins,
        "league_probs_bins": league_probs_bins,
        "league_direction": league_direction,
        "league_factor_stats": league_factor_stats,
    }


# ============================================================
# 分析函数
# ============================================================

def analyze_odds_bins(results: dict):
    print("\n" + "=" * 70)
    print("  分析1: 各联赛按赔率区间的 ROI")
    print("=" * 70)

    bins_order = ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0-5.0", "5.0+"]
    for lid in LEAGUES:
        bins_data = results["league_odds_bins"][lid]
        print(f"\n  {lid}:")
        print(f"  {'赔率区间':10s} | {'投注':>6s} | {'胜率':>7s} | {'ROI':>8s} | {'利润':>8s}")
        print(f"  {'-'*10} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*8}")
        for b in bins_order:
            if b in bins_data:
                d = bins_data[b]
                bets, wins, staked, returned = d[0], d[1], d[2], d[3]
                win_rate = wins / max(bets, 1)
                roi = (returned - staked) / max(staked, 1) if staked > 0 else 0
                profit = returned - staked
                print(f"  {b:10s} | {bets:6d} | {win_rate:6.1%} | {roi:7.1%} | {profit:8.1f}")


def analyze_probs_bins(results: dict):
    print("\n" + "=" * 70)
    print("  分析2: 各联赛模型概率校准 (实际胜率 vs 预测概率)")
    print("=" * 70)

    for lid in LEAGUES:
        bins_data = results["league_probs_bins"][lid]
        print(f"\n  {lid}:")
        print(f"  {'概率区间':12s} | {'投注':>6s} | {'实际胜率':>8s} | {'偏差':>8s} | {'ROI':>8s}")
        print(f"  {'-'*12} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")
        for b in sorted(bins_data.keys()):
            d = bins_data[b]
            bets, wins, staked, returned = d[0], d[1], d[2], d[3]
            actual_wr = wins / max(bets, 1)
            exp_prob = (float(b.split('-')[0]) + float(b.split('-')[1])) / 2
            bias = actual_wr - exp_prob
            roi = (returned - staked) / max(staked, 1) if staked > 0 else 0
            print(f"  {b:12s} | {bets:6d} | {actual_wr:7.1%} | {bias:+7.1%} | {roi:7.1%}")


def analyze_direction(results: dict):
    print("\n" + "=" * 70)
    print("  分析3: 各联赛按投注方向的 ROI")
    print("=" * 70)

    for lid in LEAGUES:
        dir_data = results["league_direction"][lid]
        print(f"\n  {lid}:")
        print(f"  {'方向':6s} | {'投注':>6s} | {'胜率':>7s} | {'ROI':>8s} | {'利润':>8s}")
        print(f"  {'-'*6} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*8}")
        for dkey in ["home", "draw", "away"]:
            d = dir_data[dkey]
            bets, wins, staked, returned = d[0], d[1], d[2], d[3]
            win_rate = wins / max(bets, 1)
            roi = (returned - staked) / max(staked, 1) if staked > 0 else 0
            profit = returned - staked
            print(f"  {dkey:6s} | {bets:6d} | {win_rate:6.1%} | {roi:7.1%} | {profit:8.1f}")


def analyze_factor_stats(results: dict):
    print("\n" + "=" * 70)
    print("  分析4: 各联赛因子激活率")
    print("=" * 70)

    for lid in LEAGUES:
        fs = results["league_factor_stats"][lid]
        if fs["total_bets"] > 0:
            avg_nonzero = fs["total_nonzero"] / fs["total_bets"]
            print(f"  {lid}: 平均非零因子数={avg_nonzero:.1f} (共{fs['total_bets']}场比赛)")


def analyze_ligue1_deep_dive(results: dict):
    print("\n" + "=" * 70)
    print("  分析5: 法甲(ligue_1) 深度诊断")
    print("=" * 70)

    bets = [r for r in results["all_bet_records"] if r["league"] == "ligue_1"]
    if not bets:
        print("  无法甲投注记录")
        return

    total = len(bets)
    wins = sum(1 for b in bets if b["is_win"])
    total_pnl = sum(b["pnl"] for b in bets)
    total_staked = sum(b["stake"] for b in bets)
    print(f"  总投注: {total}, 胜率: {wins/total:.1%}, 总盈亏: {total_pnl:.1f}, ROI: {total_pnl/total_staked:.1%}")

    # 按策略分
    print(f"\n  --- 按策略 ---")
    for strat in ["1x2", "asian_handicap", "over_under"]:
        sb = [b for b in bets if b["strategy"] == strat]
        if not sb:
            continue
        s_wins = sum(1 for b in sb if b["is_win"])
        s_staked = sum(b["stake"] for b in sb)
        s_pnl = sum(b["pnl"] for b in sb)
        print(f"  {strat}: {len(sb)}注, 胜率={s_wins/len(sb):.1%}, ROI={s_pnl/s_staked:.1%}, 盈亏={s_pnl:.1f}")

    # 按赛季分
    print(f"\n  --- 按赛季 ---")
    for season in sorted(set(b["season"] for b in bets)):
        sb = [b for b in bets if b["season"] == season]
        s_wins = sum(1 for b in sb if b["is_win"])
        s_staked = sum(b["stake"] for b in sb)
        s_pnl = sum(b["pnl"] for b in sb)
        print(f"  {season}: {len(sb)}注, 胜率={s_wins/len(sb):.1%}, ROI={s_pnl/s_staked:.1%}, 盈亏={s_pnl:.1f}")

    # 按实际结果分
    print(f"\n  --- 按实际比赛结果 ---")
    for outcome in ["home_win", "draw", "away_win"]:
        ob = [b for b in bets if b["actual_outcome"] == outcome]
        if not ob:
            continue
        o_wins = sum(1 for b in ob if b["is_win"])
        o_staked = sum(b["stake"] for b in ob)
        o_pnl = sum(b["pnl"] for b in ob)
        print(f"  实际{outcome}: {len(ob)}注, 胜率={o_wins/len(ob):.1%}, ROI={o_pnl/o_staked:.1%}, 盈亏={o_pnl:.1f}")

    # 按Elo差值分
    print(f"\n  --- 按Elo差值 ---")
    strong_fav = [b for b in bets if b["elo_diff"] > 150]  # 主队强
    close_elo = [b for b in bets if -50 <= b["elo_diff"] <= 150]  # Elo接近
    away_fav = [b for b in bets if b["elo_diff"] < -50]  # 客队强 or 接近
    for label, subset in [("主队强(>150)", strong_fav), ("Elo接近(-50~150)", close_elo), ("客队强/平(< -50)", away_fav)]:
        if not subset:
            continue
        s_wins = sum(1 for b in subset if b["is_win"])
        s_staked = sum(b["stake"] for b in subset)
        s_pnl = sum(b["pnl"] for b in subset)
        print(f"  {label}: {len(subset)}注, 胜率={s_wins/len(subset):.1%}, ROI={s_pnl/s_staked:.1%}, 盈亏={s_pnl:.1f}")

    # 按因子激活率分
    print(f"\n  --- 按因子激活率 ---")
    high_activation = [b for b in bets if b["factor_nonzero_rate"] > 0.3]
    low_activation = [b for b in bets if b["factor_nonzero_rate"] <= 0.3]
    for label, subset in [("高激活(>30%)", high_activation), ("低激活(≤30%)", low_activation)]:
        if not subset:
            continue
        s_wins = sum(1 for b in subset if b["is_win"])
        s_staked = sum(b["stake"] for b in subset)
        s_pnl = sum(b["pnl"] for b in subset)
        print(f"  {label}: {len(subset)}注, 胜率={s_wins/len(subset):.1%}, ROI={s_pnl/s_staked:.1%}, 盈亏={s_pnl:.1f}")

    # 最大亏损单
    print(f"\n  --- 最大亏损单 (Top 10) ---")
    losers = sorted([b for b in bets if not b["is_win"]], key=lambda x: x["pnl"])[:10]
    for b in losers:
        print(f"  {b['home_team']} vs {b['away_team']} | {b['strategy']} {b['selection']} | "
              f"odds={b['odds']:.2f} | stake={b['stake']:.1f} | pnl={b['pnl']:.1f} | "
              f"model={b['model_prob']:.1%} | actual={b['actual_outcome']}")

    # 置信度分布
    print(f"\n  --- 置信度分布 ---")
    conf_bins = {"0.3-0.4": [], "0.4-0.5": [], "0.5-0.6": [], "0.6-0.7": [], "0.7+": []}
    for b in bets:
        c = b["confidence"]
        if c < 0.4: conf_bins["0.3-0.4"].append(b)
        elif c < 0.5: conf_bins["0.4-0.5"].append(b)
        elif c < 0.6: conf_bins["0.5-0.6"].append(b)
        elif c < 0.7: conf_bins["0.6-0.7"].append(b)
        else: conf_bins["0.7+"].append(b)
    for label, subset in conf_bins.items():
        if not subset:
            continue
        s_wins = sum(1 for b in subset if b["is_win"])
        s_staked = sum(b["stake"] for b in subset)
        s_pnl = sum(b["pnl"] for b in subset)
        print(f"  {label}: {len(subset)}注, 胜率={s_wins/len(subset):.1%}, ROI={s_pnl/s_staked:.1%}, 盈亏={s_pnl:.1f}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  GTO-GameFlow v5.10.8 联赛表现深度分析")
    print("=" * 70)

    print("\n正在运行详细回测 (收集每条投注记录)...")
    results = run_detailed_analysis()

    analyze_odds_bins(results)
    analyze_probs_bins(results)
    analyze_direction(results)
    analyze_factor_stats(results)
    analyze_ligue1_deep_dive(results)

    # 保存详细记录
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'outputs', 'detailed_bet_analysis.json'
    )
    # 只保存摘要 (不保存all_bet_records避免文件过大)
    summary = {
        k: v for k, v in results.items() if k != "all_bet_records"
    }
    summary["total_bets"] = len(results["all_bet_records"])
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n详细记录已保存至: {output_path}")


if __name__ == '__main__':
    main()