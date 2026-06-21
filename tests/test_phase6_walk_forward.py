#!/usr/bin/env python3
"""
GTO-GameFlow v5.10 — 统一 Walk-Forward 交叉验证回测

v5.10 统一架构:
- UnifiedBayesianShrinkage: 一次完成信号分解+贝叶斯收缩 (替代 SignalDecomposer+PriorShrinkage)
- UnifiedDecisionGate: 所有策略使用统一阈值和排序
- UnifiedBankrollManager: 所有策略通过同一 Kelly 引擎
- MarketRealismIntegrator: 统一市场真实化处理
- 禁用合成赔率: 仅使用真实赔率数据

7:1:2 滚动窗口验证:
- 7 赛季训练 + 1 赛季验证 + 2 赛季测试
- 严格避免未来信息泄露
- 每个窗口独立 Elo 追踪

使用方式:
    python tests/test_phase6_walk_forward.py
"""

import sys
import os
import csv
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models import MatchContext, BetSelection, BetResult, ScoreMatrix
from src.data.odds_provider import OddsProvider, get_odds_provider, MatchOddsBundle
from src.engine.elo_cold_start import EloColdStart
from src.engine.unified_bayesian_shrinkage import (
    UnifiedBayesianShrinkage, create_shrinkage_for_league,
)
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
from src.factors.lasso_selector import (
    RollingLassoSelector, build_training_data_from_matches,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.strategies.asian_handicap import AsianHandicapResult
from src.strategies.strategy_orchestrator import StrategyOrchestrator
from src.utils.i18n import cn_league, cn_strategy
from src.data.enhanced_data_provider import EnhancedDataProvider
from src.data.match_stats_enricher import MatchStatsEnricher  # v5.10.8

# ============================================================================
# 配置
# ============================================================================

ALL_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]

WINDOW_CONFIG = {"train_size": 5, "val_size": 1, "test_size": 2}

INITIAL_BANKROLL = 100000.0


# ============================================================================
# 数据加载
# ============================================================================

class DatedMatch:
    """按日期分组的比赛数据"""
    def __init__(self, row: dict, home_team: str, away_team: str, date: datetime):
        self.row = row
        self.home_team = home_team
        self.away_team = away_team
        self.date = date
        self.match = MatchOddsBundle(
            match_id="", home_team=home_team, away_team=away_team,
            kickoff_time=date,
        )


def _load_dated_matches(league_id: str, season: str,
                         odds_provider: OddsProvider) -> List[List['DatedMatch']]:
    """加载 CSV 并按日期分组"""
    csv_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "data", "historical_odds",
    )
    filename = f"{league_id}_{season.replace('/', '-')}.csv"
    filepath = os.path.join(csv_dir, filename)

    if not os.path.exists(filepath):
        return []

    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    dated: Dict[str, List[DatedMatch]] = defaultdict(list)
    for row in rows:
        date_str = row.get("Date", "").strip()
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            try:
                dt = datetime.strptime(date_str, "%d/%m/%y")
            except ValueError:
                continue

        home = row.get("HomeTeam", "").strip()
        away = row.get("AwayTeam", "").strip()
        if not home or not away:
            continue

        dm = DatedMatch(row, home, away, dt)
        dm.match = odds_provider.build_from_csv(
            row, f"{league_id}_{home}_{away}_{date_str}",
            home, away, dt,
        )
        dated[date_str].append(dm)

    return [matches for matches in dated.values() if matches]


# ============================================================================
# Elo 管理
# ============================================================================

class EloTracker:
    """跨赛季 Elo 追踪"""

    def __init__(self):
        self.elos: Dict[str, Dict[str, float]] = {}

    def get_elo(self, league_id: str, team: str) -> float:
        if league_id not in self.elos:
            self.elos[league_id] = {}
        return self.elos[league_id].get(team, 1500.0)

    def update_elo(self, league_id: str, home_team: str, away_team: str,
                   home_goals: int, away_goals: int, k: int = 24, home_adv: float = 65):
        home_elo = self.get_elo(league_id, home_team)
        away_elo = self.get_elo(league_id, away_team)
        expected_home = 1.0 / (1.0 + 10.0 ** (-(home_elo + home_adv - away_elo) / 400.0))

        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        goal_diff = abs(home_goals - away_goals)
        margin_factor = 1.0 + min(goal_diff, 3) * 0.33

        new_home = home_elo + k * margin_factor * (actual_home - expected_home)
        new_away = away_elo + k * margin_factor * ((1.0 - actual_home) - (1.0 - expected_home))

        if league_id not in self.elos:
            self.elos[league_id] = {}
        self.elos[league_id][home_team] = new_home
        self.elos[league_id][away_team] = new_away

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {lid: dict(teams) for lid, teams in self.elos.items()}

    def restore(self, snapshot: Dict[str, Dict[str, float]]):
        self.elos = {lid: dict(teams) for lid, teams in snapshot.items()}


# ============================================================================
# v5.10.2: 球队统计追踪器 — 跨赛季追踪球队近期战绩、进球差、H2H
# ============================================================================

class TeamStatsTracker:
    """追踪球队跨赛季统计数据，用于因子计算"""

    def __init__(self):
        # {league: {team: {matches: [], goals_for: int, goals_against: int}}}
        self._stats: Dict[str, Dict[str, dict]] = {}

    def _ensure_team(self, league: str, team: str):
        if league not in self._stats:
            self._stats[league] = {}
        if team not in self._stats[league]:
            self._stats[league][team] = {
                "matches": [],  # [(goals_for, goals_against, result), ...] most recent last
                "goals_for": 0,
                "goals_against": 0,
            }

    def record_match(self, league: str, home_team: str, away_team: str,
                     home_goals: int, away_goals: int):
        """记录一场比赛结果"""
        self._ensure_team(league, home_team)
        self._ensure_team(league, away_team)

        # 确定结果
        if home_goals > away_goals:
            home_result = 3.0   # 胜
            away_result = 0.0   # 负
        elif home_goals < away_goals:
            home_result = 0.0
            away_result = 3.0
        else:
            home_result = 1.0   # 平
            away_result = 1.0

        home_stats = self._stats[league][home_team]
        away_stats = self._stats[league][away_team]

        home_stats["matches"].append((home_goals, away_goals, home_result))
        away_stats["matches"].append((away_goals, home_goals, away_result))

        # 只保留最近 10 场
        if len(home_stats["matches"]) > 10:
            home_stats["matches"] = home_stats["matches"][-10:]
        if len(away_stats["matches"]) > 10:
            away_stats["matches"] = away_stats["matches"][-10:]

        home_stats["goals_for"] += home_goals
        home_stats["goals_against"] += away_goals
        away_stats["goals_for"] += away_goals
        away_stats["goals_against"] += home_goals

    def get_recent_form(self, league: str, team: str, n: int = 5) -> List[float]:
        """获取最近 N 场战绩 (胜=3, 平=1, 负=0)"""
        self._ensure_team(league, team)
        matches = self._stats[league][team]["matches"]
        recent = matches[-n:] if len(matches) >= n else matches
        if not recent:
            return [1.5, 1.5, 1.5, 1.5, 1.5]  # 无数据时默认
        # 补齐到 n 场
        results = [1.5] * (n - len(recent)) + [m[2] for m in recent]
        return results

    def get_goal_diff(self, league: str, team: str) -> int:
        """获取赛季进球差"""
        self._ensure_team(league, team)
        s = self._stats[league][team]
        return s["goals_for"] - s["goals_against"]

    def get_streak_momentum(self, league: str, team: str) -> float:
        """获取近期势头 (连胜/连败加权)"""
        results = self.get_recent_form(league, team, 5)
        momentum = 0.0
        for i, r in enumerate(results):
            weight = (i + 1) / 15.0  # 越近权重越大
            momentum += weight * (r - 1.5) / 1.5  # 归一化到 [-1, 1]
        return momentum

    def reset_season_stats(self):
        """重置赛季统计 (进球差归零)，保留近期战绩"""
        for league in self._stats:
            for team in self._stats[league]:
                self._stats[league][team]["goals_for"] = 0
                self._stats[league][team]["goals_against"] = 0


# ============================================================================
# CSV 真实数据提取工具
# ============================================================================

def _extract_opening_probs(row: dict) -> Optional[Dict[str, float]]:
    """从 CSV 提取开盘赔率 (使用 B365 作为开盘参考)"""
    try:
        oh = float(row.get("B365H", 0) or 0)
        od = float(row.get("B365D", 0) or 0)
        oa = float(row.get("B365A", 0) or 0)
        if oh > 1.0 and od > 1.0 and oa > 1.0:
            return {"home": oh, "draw": od, "away": oa}
    except (ValueError, TypeError):
        pass
    return None


def _extract_odds_dispersion(row: dict) -> Tuple[float, float]:
    """从 CSV 提取赔率标准差和多博彩公司数量"""
    bookmaker_cols = [
        ("B365H", "B365D", "B365A"),
        ("BWH", "BWD", "BWA"),
        ("IWH", "IWD", "IWA"),
        ("PSH", "PSD", "PSA"),
        ("WHH", "WHD", "WHA"),
        ("VCH", "VCD", "VCA"),
    ]
    home_odds = []
    for h_col, d_col, a_col in bookmaker_cols:
        try:
            h = float(row.get(h_col, 0) or 0)
            if h > 1.0:
                home_odds.append(h)
        except (ValueError, TypeError):
            pass

    if len(home_odds) < 2:
        return 0.05, 1  # 默认: 低标准差, 1个来源

    mean = sum(home_odds) / len(home_odds)
    variance = sum((x - mean) ** 2 for x in home_odds) / len(home_odds)
    std = max(0.01, min(variance ** 0.5 / mean, 0.20))  # 归一化, 上限 0.20
    return round(std, 4), min(len(home_odds), 5)


# ============================================================================
# v5.10 统一回测引擎
# ============================================================================

def run_unified_window(
    seasons: List[str],
    leagues: List[str],
    elo_tracker: EloTracker,
    odds_provider: OddsProvider,
    is_training: bool = False,
    lasso_weights: Optional[Dict[str, float]] = None,
    enhanced_provider: Optional[EnhancedDataProvider] = None,
    stats_enricher: Optional[MatchStatsEnricher] = None,  # v5.10.8
) -> Dict[str, Any]:
    """
    v5.10 统一回测引擎。

    使用 UnifiedBayesianShrinkage + UnifiedDecisionGate + UnifiedBankrollManager
    替代 v5.9 的分散架构。

    参数:
        lasso_weights: LASSO 选择的因子权重 {factor_id: 1.0 or 0.0}
                       如果为 None，使用全部因子
    """
    # 训练数据收集 (用于 LASSO)
    training_data = [] if is_training else None
    # v5.10.2: 球队统计追踪器
    team_stats = TeamStatsTracker()
    # 结果容器
    result = {
        "total_bets": 0, "total_wins": 0, "total_staked": 0.0, "total_returned": 0.0,
        "profit": 0.0, "roi": 0.0, "win_rate": 0.0,
        "by_league": {}, "by_strategy": {
            "1x2": [0, 0, 0.0, 0.0],
            "asian_handicap": [0, 0, 0.0, 0.0],
            "over_under": [0, 0, 0.0, 0.0],
        },
        "seasons": seasons,
        "decisions": {"total_submitted": 0, "total_approved": 0, "total_rejected": 0},
    }
    # v5.10.8: 1X2 校准数据收集
    calibration_x2 = []

    # v5.10 模块
    pipelines = {}
    orchestrators = {}
    decision_gates = {}
    bankroll_mgrs = {}
    market_integrators = {}

    for lid in leagues:
        pipeline = GameFlowPipeline(lid, initial_bankroll=INITIAL_BANKROLL)
        pipeline.unified_shrinkage = create_shrinkage_for_league(lid)
        # v5.10.4: 应用 LASSO 权重 (如果提供)
        if lasso_weights:
            pipeline.lasso_weights = lasso_weights
        pipelines[lid] = pipeline

        orchestrators[lid] = StrategyOrchestrator(
            lid,
            asian_config={"value_threshold": 0.010, "confidence_threshold": 0.10},  # v5.10.8
            over_under_config={
                "value_threshold": 0.020,  # v5.10.9: 提高阈值抑制大小球亏损
                "confidence_threshold": 0.25,  # v5.10.9: 提高置信度阈值
                "min_odds": 1.88,  # v5.10.9: 提高最低赔率
                "dispersion": _get_league_dispersion(lid),
            },
        )
        decision_gates[lid] = create_decision_gate_for_league(lid)
        bankroll_mgrs[lid] = create_bankroll_manager_for_league(lid, INITIAL_BANKROLL)
        market_integrators[lid] = create_integrator_for_league(lid)

    matchup_count = 0
    season_elo_starts = {}
    for season in seasons:
        season_elo_starts[season] = elo_tracker.snapshot()

    for season in seasons:
        elo_tracker.restore(season_elo_starts[season])

        all_league_dates = {}
        for lid in leagues:
            all_league_dates[lid] = _load_dated_matches(lid, season, odds_provider)

        all_dates = defaultdict(list)
        for lid in leagues:
            for day_matches in all_league_dates[lid]:
                if day_matches:
                    date = day_matches[0].date
                    for dm in day_matches:
                        all_dates[date].append((lid, dm))

        sorted_dates = sorted(all_dates.keys())

        for date in sorted_dates:
            day_matches = all_dates[date]
            day_results: Dict[str, Tuple[str, Any, Tuple[int, int]]] = {}
            daily_staked = 0.0
            daily_bets = 0

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

                # v5.10: 市场真实化集成
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

                outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW,
                               "away_win": BetSelection.AWAY_WIN}
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
                    'h2h_results': [1.5, 1.5, 1.5, 1.5, 1.5],  # v5.10.7: EnhancedDataProvider 覆盖为真实H2H
                    'rank_diff': 0,  # v5.10.7: EnhancedDataProvider 覆盖
                    'goal_diff': team_stats.get_goal_diff(lid, dm.home_team) - team_stats.get_goal_diff(lid, dm.away_team),
                    'xg_diff': 0,  # v5.10.7: EnhancedDataProvider 覆盖
                    'streak_momentum': team_stats.get_streak_momentum(lid, dm.home_team),
                    'streak_momentum_league': team_stats.get_streak_momentum(lid, dm.away_team),
                    'match_phase': 1.0,
                    'data_source_count': _extract_odds_dispersion(row)[1],
                    'odds_std': _extract_odds_dispersion(row)[0],
                    'opening_probs': _extract_opening_probs(row),
                }

                # v5.10.5: 使用 EnhancedDataProvider 补充因子数据
                if enhanced_provider:
                    extra = enhanced_provider.get_enhanced_data(
                        league=lid,
                        season=season,
                        home_team=dm.home_team,
                        away_team=dm.away_team,
                        match_date=date,
                        existing_extra=extra,
                        stats_enricher=stats_enricher,  # v5.10.8
                    )

                pipeline_result = pipelines[lid].run_full(match_ctx, extra_data=extra)

                # v5.10.9: 因子激活率追踪
                if pipeline_result.factor_deltas and not hasattr(run_unified_window, '_factor_activation'):
                    run_unified_window._factor_activation = defaultdict(int)
                    run_unified_window._factor_total = 0
                if pipeline_result.factor_deltas:
                    run_unified_window._factor_total += 1
                    for fid in pipeline_result.factor_deltas:
                        d = pipeline_result.factor_deltas[fid]
                        if isinstance(d, dict) and (abs(d.get('home',0))>0.0001 or abs(d.get('away',0))>0.0001 or abs(d.get('draw',0))>0.0001):
                            run_unified_window._factor_activation[fid] += 1

                # v5.10.4: 收集训练数据 (用于 LASSO)
                if is_training and pipeline_result.factor_deltas:
                    training_data.append({
                        "factor_deltas": dict(pipeline_result.factor_deltas),
                        "actual_outcome": actual_outcome,
                        "league_id": lid,
                        "elo_diff": home_elo - away_elo,
                    })

                day_results[match_ctx.match_id] = (actual_outcome, actual, (home_goals, away_goals))

                # ================================================================
                # v5.10.3: 统一投注路由 — 所有策略经同一决策门+资金管理
                # ================================================================

                # 1. 收集 1X2 提案 (从 pipeline 的 proposals，非 placements)
                x2_unified = []
                if pipeline_result.proposals:
                    for p in pipeline_result.proposals:
                        x2_unified.append(UnifiedProposal(
                            match_id=p.match_id,
                            strategy="1x2",
                            selection=p.selection.value if hasattr(p.selection, 'value') else str(p.selection),
                            odds=p.odds,
                            model_prob=p.model_prob,
                            implied_prob=p.implied_prob,
                            value=p.value,
                            kelly_stake=p.kelly_stake,
                            confidence=getattr(p, 'confidence', 0.5),  # v5.10.5: 使用真实置信度，非 priority_score
                            priority_score=getattr(p, 'priority_score', 0.0),
                            league_id=lid,
                            original=p,
                        ))

                # 2. 收集亚盘 + 大小球 提案
                asian_ou_unified = []
                try:
                    if bundle.has_real_asian or bundle.has_real_totals:
                        multi = orchestrators[lid].run(
                            match=match_ctx, score_matrix=pipeline_result.poisson_score_matrix,
                            handicap_odds=bundle.asian_odds if bundle.has_real_asian else {},
                            totals_odds=bundle.totals_odds if bundle.has_real_totals else {},
                            total_bankroll=INITIAL_BANKROLL,
                            strip_margin_asian=True,
                            strip_margin_totals=True,
                        )
                        asian_ou_unified = proposals_to_unified(
                            asian_proposals=multi.asian_proposals,
                            totals_proposals=multi.totals_proposals,
                        )
                except Exception:
                    pass

                # 3. 合并所有提案
                all_unified = x2_unified + asian_ou_unified
                if not all_unified:
                    continue

                # 4. 统一决策门
                decision = decision_gates[lid].evaluate(all_unified, daily_bets)
                result["decisions"]["total_submitted"] += len(all_unified)
                result["decisions"]["total_approved"] += decision.approved_count
                result["decisions"]["total_rejected"] += decision.rejected_count

                if not decision.approved:
                    continue

                # 5. 统一资金管理
                allocation = bankroll_mgrs[lid].allocate(
                    decision.approved, daily_staked, daily_bets,
                )

                # 6. 统一结算
                for sp in allocation.proposals:
                    orig = sp.original
                    if orig is None:
                        continue
                    stake = sp.final_stake

                    # 根据策略类型结算
                    if sp.strategy == "1x2":
                        # 使用 pipeline 的 settle_bets 逻辑
                        sel = sp.selection
                        if sel == "home_win":
                            bet_sel = BetSelection.HOME_WIN
                        elif sel == "draw":
                            bet_sel = BetSelection.DRAW
                        elif sel == "away_win":
                            bet_sel = BetSelection.AWAY_WIN
                        else:
                            continue
                        if actual == bet_sel:
                            is_win = True
                            pnl = stake * (orig.odds - 1.0)
                        else:
                            is_win = False
                            pnl = -stake
                        # v5.10.8: 记录 1X2 校准数据
                        if sp.strategy == "1x2":
                            calibration_x2.append({
                                "model_prob": sp.model_prob,
                                "selection": sp.selection,
                                "is_win": is_win,
                            })
                    elif sp.strategy == "asian_handicap":
                        if hasattr(orig, 'adjusted_stake'):
                            orig.adjusted_stake = stake
                        ar, pnl = orchestrators[lid].asian_strategy.settle(
                            orig, home_goals, away_goals,
                        )
                        is_win = ar in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                    elif sp.strategy == "over_under":
                        if hasattr(orig, 'adjusted_stake'):
                            orig.adjusted_stake = stake
                        tr, pnl = orchestrators[lid].over_under_strategy.settle(
                            orig, home_goals, away_goals,
                        )
                        is_win = (tr == "win")
                    else:
                        continue

                    # 流动性检查
                    can_exec, _ = integrator.check_liquidity(stake, sp.strategy)
                    if not can_exec:
                        continue

                    # 交易成本
                    settlement = integrator.process_settlement(pnl, stake, sp.strategy)
                    net_pnl = settlement.net_profit

                    # 统计
                    result["total_bets"] += 1
                    result["total_staked"] += stake
                    daily_staked += stake
                    daily_bets += 1
                    if is_win:
                        result["total_wins"] += 1
                        result["total_returned"] += stake + net_pnl
                    if lid not in result["by_league"]:
                        result["by_league"][lid] = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}
                    result["by_league"][lid]["bets"] += 1
                    result["by_league"][lid]["staked"] += stake
                    if is_win:
                        result["by_league"][lid]["wins"] += 1
                        result["by_league"][lid]["returned"] += stake + net_pnl

                    strategy_key = sp.strategy
                    result["by_strategy"][strategy_key][0] += 1
                    result["by_strategy"][strategy_key][2] += stake
                    if is_win:
                        result["by_strategy"][strategy_key][1] += 1
                        result["by_strategy"][strategy_key][3] += stake + net_pnl

                # Elo 更新
                elo_tracker.update_elo(
                    lid, dm.home_team, dm.away_team,
                    home_goals, away_goals,
                    k=24, home_adv=mkt_adj.home_advantage * 100,
                )
                # v5.10.2: 记录球队统计 (用于后续比赛的因子计算)
                team_stats.record_match(lid, dm.home_team, dm.away_team,
                                        home_goals, away_goals)

        season_elo_starts[season] = elo_tracker.snapshot()

    # 计算 ROI
    roi = (result["total_returned"] - result["total_staked"]) / max(result["total_staked"], 0.01)
    result["roi"] = roi
    result["profit"] = result["total_returned"] - result["total_staked"]
    result["win_rate"] = result["total_wins"] / max(result["total_bets"], 1)
    # v5.10.4: 附加训练数据
    result["training_data"] = training_data

    # v5.10.8: 1X2 概率校准分析
    if calibration_x2 and not is_training:
        _print_calibration_analysis(calibration_x2)

    return result


def _print_calibration_analysis(calibration_x2: list):
    """v5.10.8: 打印 1X2 概率校准分析"""
    from collections import defaultdict
    print("\n  ── 1X2 概率校准分析 ──")
    
    # 按 selection 分组
    for sel in ['home_win', 'draw', 'away_win']:
        bets = [b for b in calibration_x2 if b['selection'] == sel]
        if not bets:
            continue
        
        # 按概率分桶
        buckets = defaultdict(list)
        for b in bets:
            bucket = int(b['model_prob'] * 10) / 10  # 0.1 buckets
            buckets[bucket].append(b['is_win'])
        
        actual_rate = sum(1 for b in bets if b['is_win']) / len(bets)
        mean_prob = sum(b['model_prob'] for b in bets) / len(bets)
        
        sel_name = {'home_win': '主胜', 'draw': '平局', 'away_win': '客胜'}.get(sel, sel)
        print(f"  {sel_name}: {len(bets)}注, 平均模型概率={mean_prob:.1%}, 实际胜率={actual_rate:.1%}, 偏差={mean_prob-actual_rate:+.1%}")
        for b in sorted(buckets.keys()):
            acts = buckets[b]
            print(f"    [{b:.1f}-{b+0.1:.1f}): {len(acts):4d}注, 实际={sum(acts)/len(acts):.1%}")


def _get_league_dispersion(lid: str) -> float:
    """联赛差异化 over-dispersion 参数"""
    return {
        "premier_league": 0.15,
        "bundesliga": 0.20,
        "la_liga": 0.15,
        "serie_a": 0.10,
        "ligue_1": 0.12,
    }.get(lid, 0.15)


# ============================================================================
# Walk-Forward 验证
# ============================================================================

def generate_walk_forward_windows() -> List[Tuple[List[str], str, List[str]]]:
    """生成 Walk-Forward 窗口"""
    windows = []
    idx = 0
    while idx + WINDOW_CONFIG["train_size"] + WINDOW_CONFIG["val_size"] + WINDOW_CONFIG["test_size"] <= len(ALL_SEASONS):
        train = ALL_SEASONS[idx:idx + WINDOW_CONFIG["train_size"]]
        val = ALL_SEASONS[idx + WINDOW_CONFIG["train_size"]]
        test = ALL_SEASONS[idx + WINDOW_CONFIG["train_size"] + 1:
                             idx + WINDOW_CONFIG["train_size"] + 1 + WINDOW_CONFIG["test_size"]]
        windows.append((train, val, test))
        idx += WINDOW_CONFIG["test_size"]
    return windows


def run_walk_forward_v510():
    """v5.10 统一 Walk-Forward 验证"""
    print("=" * 70)
    print("GTO-GameFlow v5.10 — 统一 Walk-Forward 交叉验证")
    print(f"窗口: {WINDOW_CONFIG['train_size']}训练 + {WINDOW_CONFIG['val_size']}验证 + {WINDOW_CONFIG['test_size']}测试")
    print(f"所有赔率: 真实数据 (禁用合成赔率)")
    print("=" * 70)

    odds_provider = get_odds_provider(use_pinnacle=True)
    windows = generate_walk_forward_windows()

    # Elo 冷启动
    csv_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "data", "historical_odds",
    )
    print("\n[Elo 冷启动] 从历史数据初始化...")
    elo_cs = EloColdStart(default_elo=1500.0, k=24, home_advantage=65)
    try:
        elo_cs.initialize_from_csv_directory(csv_dir, LEAGUES, ALL_SEASONS[:7])
        print(f"  已初始化 {sum(len(v) for v in elo_cs._elos.values())} 支球队的 Elo")
    except Exception as e:
        print(f"  Elo 冷启动失败 (使用默认值): {e}")
        elo_cs = None

    print(f"\n生成 {len(windows)} 个窗口:")
    for i, (train, val, test) in enumerate(windows):
        print(f"  窗口 {i+1}: 训练={train[0]}~{train[-1]}, 验证={val}, 测试={test[0]}~{test[-1]}")

    all_test_results = []
    oos_rois = []

    for win_idx, (train_seasons, val_season, test_seasons) in enumerate(windows):
        print(f"\n{'─' * 60}")
        print(f"窗口 {win_idx + 1}/{len(windows)}")
        print(f"  训练: {train_seasons}")
        print(f"  验证: {val_season}")
        print(f"  测试: {test_seasons}")
        print(f"{'─' * 60}")

        # 阶段 1: 训练窗口参数调优
        print("  [1/2] 训练窗口...")
        train_elo = EloTracker()
        if elo_cs:
            for lid in LEAGUES:
                for team, elo_val in elo_cs.get_all_elos(lid).items():
                    if lid not in train_elo.elos:
                        train_elo.elos[lid] = {}
                    train_elo.elos[lid][team] = elo_val

        # v5.10.5: 增强数据提供器 — 从CSV计算18个之前为零的因子
        print("    初始化增强数据提供器...")
        enhanced_provider = EnhancedDataProvider(
            csv_dir=csv_dir,
            leagues=LEAGUES,
            seasons=train_seasons + [val_season] + test_seasons,
        )

        # v5.10.8: 比赛统计增强器 — 14个高维衍生特征
        stats_enricher = MatchStatsEnricher(
            csv_dir=csv_dir,
            leagues=LEAGUES,
            seasons=train_seasons + [val_season] + test_seasons,
        )
        print(f"    初始化比赛统计增强器 — 14个新特征")

        train_result = run_unified_window(
            train_seasons, LEAGUES, train_elo, odds_provider,
            is_training=True,
            enhanced_provider=enhanced_provider,
            stats_enricher=stats_enricher,  # v5.10.8
        )
        print(f"    训练窗口: 投注={train_result['total_bets']}, ROI={train_result['roi']:+.1%}")

        # v5.10.4: LASSO 因子选择
        print("    运行 LASSO 因子选择...")
        lasso_weights = {}
        lasso_selected = {}
        lasso_selector = RollingLassoSelector(window_size=len(train_seasons) * 380)
        try:
            train_data = train_result.get("training_data", [])
            if train_data:
                X, y, factor_names = build_training_data_from_matches(train_data)
                lasso_weights = lasso_selector.select("all_leagues", X, y, factor_names, alpha=0.01)
                lasso_selected = {f: w for f, w in lasso_weights.items() if w > 0}
                print(f"    LASSO 选中 {len(lasso_selected)}/{len(factor_names)} 个因子: {sorted(lasso_selected.keys())}")
            else:
                print("    无训练数据，跳过 LASSO")
        except Exception as e:
            print(f"    LASSO 失败: {e}")

        # 阶段 2: 验证窗口 (使用训练末 Elo)
        print("  [2/2] 验证+测试窗口...")
        val_result = run_unified_window(
            [val_season] + test_seasons, LEAGUES, train_elo, odds_provider,
            is_training=False,
            lasso_weights=lasso_weights if lasso_weights else None,
            enhanced_provider=enhanced_provider,
            stats_enricher=stats_enricher,  # v5.10.8
        )

        val_roi = val_result.get("roi", 0.0)
        oos_rois.append(val_roi)
        all_test_results.append(val_result)

        print(f"  OOS 结果: 投注={val_result['total_bets']}, "
              f"胜率={val_result['win_rate']:.1%}, "
              f"ROI={val_roi:+.1%}, "
              f"利润={val_result['profit']:+,.0f}")

        # 策略分布
        for s in ["1x2", "asian_handicap", "over_under"]:
            bets, wins, staked, returned = val_result["by_strategy"][s]
            if bets > 0:
                s_roi = (returned - staked) / max(staked, 0.01)
                print(f"    {s}: {bets}注, ROI={s_roi:+.1%}")

        # 决策统计
        dec = val_result.get("decisions", {})
        if dec:
            print(f"    决策门: 提交={dec.get('total_submitted',0)}, "
                  f"通过={dec.get('total_approved',0)}, "
                  f"拒绝={dec.get('total_rejected',0)}")

    # ── 最终汇总 ──
    print(f"\n{'=' * 70}")
    print("v5.10 Walk-Forward 最终汇总")
    print(f"{'=' * 70}")

    total_bets = sum(r["total_bets"] for r in all_test_results)
    total_wins = sum(r["total_wins"] for r in all_test_results)
    total_staked = sum(r["total_staked"] for r in all_test_results)
    total_returned = sum(r["total_returned"] for r in all_test_results)
    total_profit = total_returned - total_staked
    oos_roi = total_profit / max(total_staked, 0.01)
    oos_win_rate = total_wins / max(total_bets, 1)

    print(f"\n总 OOS 投注: {total_bets}")
    print(f"总 OOS 胜率: {oos_win_rate:.1%}")
    print(f"总 OOS ROI: {oos_roi:+.1%}")
    print(f"总 OOS 利润: {total_profit:+,.0f}")

    print(f"\n各窗口 OOS ROI:")
    for i, roi in enumerate(oos_rois):
        print(f"  窗口 {i+1}: {roi:+.1%}")

    # 联赛汇总
    print(f"\n联赛汇总:")
    league_agg = defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0})
    for r in all_test_results:
        for lid, stats in r.get("by_league", {}).items():
            league_agg[lid]["bets"] += stats["bets"]
            league_agg[lid]["wins"] += stats["wins"]
            league_agg[lid]["staked"] += stats["staked"]
            league_agg[lid]["returned"] += stats["returned"]
    for lid in LEAGUES:
        la = league_agg[lid]
        if la["bets"] > 0:
            l_roi = (la["returned"] - la["staked"]) / max(la["staked"], 0.01)
            l_wr = la["wins"] / max(la["bets"], 1)
            print(f"  {cn_league(lid)}: {la['bets']}注, 胜率={l_wr:.1%}, ROI={l_roi:+.1%}")

    # 策略汇总
    print(f"\n策略汇总:")
    strategy_agg = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for r in all_test_results:
        for s, vals in r.get("by_strategy", {}).items():
            strategy_agg[s][0] += vals[0]
            strategy_agg[s][1] += vals[1]
            strategy_agg[s][2] += vals[2]
            strategy_agg[s][3] += vals[3]
    for s in ["1x2", "asian_handicap", "over_under"]:
        bets, wins, staked, returned = strategy_agg[s]
        if bets > 0:
            s_roi = (returned - staked) / max(staked, 0.01)
            s_wr = wins / max(bets, 1)
            pct = bets / max(total_bets, 1) * 100
            print(f"  {cn_strategy(s)}: {bets}注 ({pct:.0f}%), 胜率={s_wr:.1%}, ROI={s_roi:+.1%}")

    # v5.10.9: 因子激活率报告
    if hasattr(run_unified_window, '_factor_activation') and run_unified_window._factor_total > 0:
        print(f"\n因子激活率 (采样 {run_unified_window._factor_total} 场):")
        from src.factors.registry import FACTOR_REGISTRY
        act = run_unified_window._factor_activation
        total = run_unified_window._factor_total
        abandoned = []
        low = []
        for fid in sorted(FACTOR_REGISTRY.keys(), key=lambda f: act.get(f,0)/total):
            rate = act.get(fid,0)/total
            if rate < 0.05:
                abandoned.append(fid)
            elif rate < 0.20:
                low.append(fid)
        print(f"  ❌ 废弃(<5%): {len(abandoned)}个 — {', '.join(abandoned) if abandoned else '无'}")
        print(f"  ⚠ 低(<20%): {len(low)}个 — {', '.join(low) if low else '无'}")
        print(f"  ✅ 正常(≥20%): {len(FACTOR_REGISTRY)-len(abandoned)-len(low)}个")

    return {
        "total_bets": total_bets,
        "total_wins": total_wins,
        "total_staked": total_staked,
        "total_returned": total_returned,
        "total_profit": total_profit,
        "oos_roi": oos_roi,
        "oos_win_rate": oos_win_rate,
        "window_rois": oos_rois,
        "league_agg": dict(league_agg),
        "strategy_agg": dict(strategy_agg),
    }


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    results = run_walk_forward_v510()

    # 保存结果
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase6_walk_forward_results.json")

    serializable = {
        "total_bets": results["total_bets"],
        "total_wins": results["total_wins"],
        "total_staked": results["total_staked"],
        "total_returned": results["total_returned"],
        "total_profit": results["total_profit"],
        "oos_roi": results["oos_roi"],
        "oos_win_rate": results["oos_win_rate"],
        "window_rois": results["window_rois"],
        "by_league": {lid: dict(stats) for lid, stats in results["league_agg"].items()},
        "by_strategy": {s: list(vals) for s, vals in results["strategy_agg"].items()},
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存至: {output_path}")