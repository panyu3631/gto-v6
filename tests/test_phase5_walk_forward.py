#!/usr/bin/env python3
"""
Phase 5.0 Walk-Forward 交叉验证回测 (P0b)

使用 7:1:2 滚动窗口验证:
- 7 赛季训练窗口: 调优联赛参数
- 1 赛季验证窗口: 选参
- 2 赛季测试窗口: OOS 评估 (仅运行一次)

严格避免未来信息泄露:
- 训练窗口内 Elo carryover 仅限于训练窗口
- 验证窗口使用训练窗口末的 Elo
- 测试窗口使用验证窗口末的 Elo
- 每个窗口独立调优，训练窗口的参数不传递到测试窗口

集成 Phase 5.0 所有新模块:
- OddsProvider: 统一赔率源 (Pinnacle 优先)
- BetLogger: 投注决策日志
- BookmakerBehavior: 庄家行为调整
- TransactionCost: 交易成本
- LiquidityCheck: 流动性验证
- DynamicHomeAdvantage: 动态主场优势
"""

import sys
import os
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models import MatchContext, BetSelection, BetResult
from src.data.odds_provider import OddsProvider, get_odds_provider, MatchOddsBundle
from src.engine.bet_logger import BetLogger, BetDecision, create_bet_id
from src.engine.market_realism import (
    BookmakerBehavior, TransactionCost, LiquidityCheck,
    get_dynamic_home_advantage,
)
from src.engine.elo_cold_start import EloColdStart
from src.pipeline.orchestrator import GameFlowPipeline
from src.strategies.strategy_orchestrator import StrategyOrchestrator, ParlayBatchManager
from src.strategies.asian_handicap import AsianHandicapResult
from src.calibration.signal_decomposer import PriorShrinkage  # v5.9 兼容保留
from src.engine.unified_bayesian_shrinkage import UnifiedBayesianShrinkage, create_shrinkage_for_league
from src.engine.bankroll import generate_bet_proposals
from src.utils.i18n import cn_league, cn_strategy

# ============================================================================
# 配置
# ============================================================================

ALL_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]

# Walk-Forward 窗口: 7 训练 + 1 验证 + 2 测试
WINDOW_CONFIG = {
    "train_size": 7,
    "val_size": 1,
    "test_size": 2,
}

# 联赛参数 (训练窗口内调优的搜索空间)
LEAGUE_PARAM_SEARCH = {
    "premier_league": {
        "value_threshold": [0.018, 0.020, 0.022, 0.025],
        "confidence_threshold": [0.48, 0.50, 0.55],
        "shrinkage_alpha_high": [0.50, 0.55, 0.60],
    },
    "la_liga": {
        "value_threshold": [0.012, 0.015, 0.018],
        "confidence_threshold": [0.48, 0.52, 0.55],
        "shrinkage_alpha_high": [0.45, 0.50, 0.55],
    },
    "bundesliga": {
        "value_threshold": [0.015, 0.018, 0.022],
        "confidence_threshold": [0.48, 0.52, 0.55],
        "shrinkage_alpha_high": [0.45, 0.50, 0.55],
    },
    "serie_a": {
        "value_threshold": [0.010, 0.015, 0.018],
        "confidence_threshold": [0.38, 0.40, 0.45],
        "shrinkage_alpha_high": [0.45, 0.48, 0.52],
    },
    "ligue_1": {
        "value_threshold": [0.012, 0.015, 0.018],
        "confidence_threshold": [0.48, 0.50, 0.52],
        "shrinkage_alpha_high": [0.45, 0.48, 0.50],
    },
}

# 默认联赛参数
DEFAULT_LEAGUE_PARAMS = {
    "premier_league": {
        "value_threshold": 0.020, "confidence_threshold": 0.50,
        "shrinkage_alpha_high": 0.55, "shrinkage_alpha_low": 0.12,
        "elo_k": 24, "home_advantage": 65,
        "calib_multiplier_base": 0.80, "calib_multiplier_enhanced": 0.80, "calib_multiplier_league": 0.80,
    },
    "la_liga": {
        "value_threshold": 0.015, "confidence_threshold": 0.52,
        "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10,
        "elo_k": 20, "home_advantage": 50,
        "calib_multiplier_base": 0.75, "calib_multiplier_enhanced": 0.75, "calib_multiplier_league": 0.75,
    },
    "bundesliga": {
        "value_threshold": 0.018, "confidence_threshold": 0.52,
        "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10,
        "elo_k": 22, "home_advantage": 60,
        "calib_multiplier_base": 0.75, "calib_multiplier_enhanced": 0.75, "calib_multiplier_league": 0.75,
    },
    "serie_a": {
        "value_threshold": 0.015, "confidence_threshold": 0.40,
        "shrinkage_alpha_high": 0.48, "shrinkage_alpha_low": 0.10,
        "elo_k": 20, "home_advantage": 55,
        "calib_multiplier_base": 0.80, "calib_multiplier_enhanced": 0.80, "calib_multiplier_league": 0.80,
    },
    "ligue_1": {
        "value_threshold": 0.015, "confidence_threshold": 0.50,
        "shrinkage_alpha_high": 0.48, "shrinkage_alpha_low": 0.10,
        "elo_k": 20, "home_advantage": 55,
        "calib_multiplier_base": 0.70, "calib_multiplier_enhanced": 0.70, "calib_multiplier_league": 0.70,
    },
}

# ============================================================================
# 数据加载 (复用 Phase 4 逻辑)
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
                         odds_provider: OddsProvider) -> List[List[DatedMatch]]:
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
        # 使用 OddsProvider 解析赔率 (自动选择 Pinnacle)
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
    """跨赛季 Elo 追踪 (严格避免未来信息)"""

    def __init__(self):
        self.elos: Dict[str, Dict[str, float]] = {}  # {league_id: {team: elo}}

    def get_elo(self, league_id: str, team: str) -> float:
        if league_id not in self.elos:
            self.elos[league_id] = {}
        return self.elos[league_id].get(team, 1500.0)

    def update_elo(self, league_id: str, home_team: str, away_team: str,
                   home_goals: int, away_goals: int, k: int = 24, home_adv: float = 65):
        """标准 Elo 更新"""
        home_elo = self.get_elo(league_id, home_team)
        away_elo = self.get_elo(league_id, away_team)

        expected_home = 1.0 / (1.0 + 10.0 ** (-(home_elo + home_adv - away_elo) / 400.0))
        expected_away = 1.0 - expected_home

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        goal_diff = abs(home_goals - away_goals)
        margin_factor = 1.0 + min(goal_diff, 3) * 0.33

        new_home = home_elo + k * margin_factor * (actual_home - expected_home)
        new_away = away_elo + k * margin_factor * (actual_away - expected_away)

        if league_id not in self.elos:
            self.elos[league_id] = {}
        self.elos[league_id][home_team] = new_home
        self.elos[league_id][away_team] = new_away

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """获取当前 Elo 快照 (用于窗口间传递)"""
        return {lid: dict(teams) for lid, teams in self.elos.items()}

    def restore(self, snapshot: Dict[str, Dict[str, float]]):
        """恢复 Elo 快照"""
        self.elos = {lid: dict(teams) for lid, teams in snapshot.items()}


# ============================================================================
# 回测引擎
# ============================================================================


def _get_league_dispersion(lid: str) -> float:
    """联赛差异化 over-dispersion 参数"""
    return {
        "premier_league": 0.15,
        "bundesliga": 0.20,      # 德甲高进球, 方差大
        "la_liga": 0.15,
        "serie_a": 0.10,         # 意甲低进球, 方差小
        "ligue_1": 0.12,
    }.get(lid, 0.15)


def run_window_seasons(
    seasons: List[str],
    leagues: List[str],
    elo_tracker: EloTracker,
    odds_provider: OddsProvider,
    league_params: Dict[str, dict],
    bet_logger: Optional[BetLogger] = None,
    initial_bankroll: float = 10000.0,
    is_training: bool = False,
) -> Dict[str, Any]:
    """
    运行指定赛季窗口的回测。

    参数:
        seasons: 赛季列表
        leagues: 联赛列表
        elo_tracker: Elo 追踪器 (包含窗口起始状态)
        odds_provider: 赔率提供者
        league_params: 联赛参数
        bet_logger: 投注日志 (可选)
        initial_bankroll: 初始资金
        is_training: 是否为训练窗口 (训练窗口不计入最终统计)

    返回:
        回测结果字典
    """
    # 市场真实化模块
    bookmaker = BookmakerBehavior()
    tx_cost = TransactionCost()
    liquidity = LiquidityCheck()

    # 共享资金池
    class SharedBankroll:
        def __init__(self, balance):
            self.state = type('obj', (object,), {'balance': balance})()
        def _get_base_bankroll(self):
            return 10000.0

    shared_bankroll = SharedBankroll(initial_bankroll)

    # 结果容器
    result = {
        "total_bets": 0, "total_wins": 0, "total_staked": 0.0, "total_returned": 0.0,
        "by_league": {}, "by_strategy": {
            "1x2": [0, 0, 0.0, 0.0],
            "asian": [0, 0, 0.0, 0.0],
            "over_under": [0, 0, 0.0, 0.0],
        },
        "parlay_bets": 0, "parlay_wins": 0, "parlay_staked": 0.0,
        "risk_events": 0, "seasons": seasons,
    }

    # 为每个联赛创建 pipeline 和 orchestrator
    pipelines = {}
    orchestrators = {}
    for lid in leagues:
        params = league_params.get(lid, DEFAULT_LEAGUE_PARAMS[lid])
        pipeline = GameFlowPipeline(lid, initial_bankroll=initial_bankroll)
        # v5.10: 使用 UnifiedBayesianShrinkage 替代 PriorShrinkage
        pipeline.unified_shrinkage = create_shrinkage_for_league(lid)
        # v5.9 兼容: 保留 prior_shrinkage 引用
        pipeline.prior_shrinkage = PriorShrinkage(
            alpha_high=params["shrinkage_alpha_high"],
            alpha_low=params["shrinkage_alpha_low"],
        )
        pipelines[lid] = pipeline
        orchestrators[lid] = StrategyOrchestrator(
            lid,
            asian_config={"value_threshold": 0.015, "confidence_threshold": 0.35},
            over_under_config={
                "value_threshold": 0.015,
                "confidence_threshold": 0.40,
                "min_odds": 1.80,
                "dispersion": _get_league_dispersion(lid),
            },
        )

    # 匹配计数器
    matchup_count = 0

    # 赛季间 Elo 快照传递
    season_elo_starts = {}
    for season in seasons:
        season_elo_starts[season] = elo_tracker.snapshot()

    for season in seasons:
        # 赛季 Elo 起始点
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

            # 重置当天串关池
            parlay_mgr = ParlayBatchManager(
                max_legs=2, kelly_discount=0.30, max_batch_size=10,
                min_single_value=0.025, min_combined_value=0.045,
            )

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

                # 庄家行为调整
                adj = bookmaker.adjust_odds(odds_h, odds_d, odds_a)
                if adj.skip_recommendation:
                    continue
                odds_h, odds_d, odds_a = adj.adjusted_home, adj.adjusted_draw, adj.adjusted_away

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

                # 动态主场优势
                params = league_params.get(lid, DEFAULT_LEAGUE_PARAMS[lid])
                home_adv = get_dynamic_home_advantage(lid, season)
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
                    'recent_results': [], 'h2h_results': [],
                    'rank_diff': 0, 'goal_diff': 0, 'xg_diff': 0,
                    'streak_momentum': 0, 'streak_momentum_league': 0,
                    'match_phase': 1.0, 'data_source_count': 5, 'odds_std': 0.05,
                }

                pipeline_result = pipelines[lid].run_full(match_ctx, extra_data=extra)

                # 串关池
                if pipeline_result.value_results and pipeline_result.fused_probs:
                    parlay_proposals = generate_bet_proposals(
                        pipeline_result.value_results,
                        match_id=match_ctx.match_id,
                        league_id=lid,
                        factor_count=37, data_source_count=5, odds_std=0.05,
                        match_phase=1.0,
                        threshold=params.get("value_threshold", 0.015),
                        confidence_threshold=params.get("confidence_threshold", 0.50),
                    )
                    if parlay_proposals:
                        parlay_mgr.add_match_bets(match_ctx.match_id, parlay_proposals)

                day_results[match_ctx.match_id] = (actual_outcome, actual, (home_goals, away_goals))

                if pipeline_result.placements:
                    placements = pipelines[lid].settle_bets(pipeline_result.placements, actual)

                    # 从 proposals 中获取模型概率等数据
                    proposal_map = {prop.selection.value: prop for prop in pipeline_result.proposals}

                    for p in placements:
                        is_win = p.result == BetResult.WIN
                        stake = p.stake
                        pnl = p.profit_loss

                        # 流动性检查
                        can_exec, _ = liquidity.check(lid, "1x2", stake)
                        if not can_exec:
                            continue

                        # 交易成本
                        net_pnl = tx_cost.apply_costs(pnl, stake) if pnl > 0 else pnl

                        result["total_bets"] += 1
                        result["total_staked"] += stake
                        daily_staked += stake
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
                        result["by_strategy"]["1x2"][0] += 1
                        result["by_strategy"]["1x2"][2] += stake
                        if is_win:
                            result["by_strategy"]["1x2"][1] += 1
                            result["by_strategy"]["1x2"][3] += stake + net_pnl

                        # 投注日志
                        if bet_logger:
                            prop = proposal_map.get(p.selection.value)
                            model_prob = prop.model_prob if prop else 0.0
                            value = prop.value if prop else 0.0
                            implied_prob = prop.implied_prob if prop else 0.0
                            confidence = prop.priority_score if prop else 0.0
                            decision = BetDecision(
                                bet_id=create_bet_id(lid, season, matchup_count, "1x2"),
                                match_id=match_ctx.match_id,
                                match_desc=f"{dm.home_team} vs {dm.away_team}",
                                league_id=lid, season=season, strategy="1x2",
                                selection=p.selection.value,
                                odds=p.odds, odds_source=bundle.odds_source,
                                model_prob=model_prob, implied_prob=implied_prob, value=value,
                                confidence=confidence, factor_count=37, data_source_count=5,
                                kelly_stake=stake, adjusted_stake=stake, kelly_fraction=0.25,
                                bankroll_before=shared_bankroll.state.balance,
                                risk_approved=True, margin_estimate=odds_provider.estimate_margin(odds_h, odds_d, odds_a),
                                home_advantage_used=home_adv, elo_home=home_elo, elo_away=away_elo,
                            )
                            bet_logger.log_decision(decision)
                            bet_logger.log_settlement(decision.bet_id, "win" if is_win else "loss", net_pnl)

                    # 亚盘 + 大小球 (v5.10: 仅使用真实赔率)
                    try:
                        # v5.10: 禁用合成赔率 — 无真实赔率时策略不输出
                        if not bundle.has_real_asian and not bundle.has_real_totals:
                            # 两条策略都没有真实赔率，跳过
                            pass
                        else:
                            multi = orchestrators[lid].run(
                                match=match_ctx, score_matrix=pipeline_result.poisson_score_matrix,
                                handicap_odds=bundle.asian_odds if bundle.has_real_asian else {},
                                totals_odds=bundle.totals_odds if bundle.has_real_totals else {},
                                total_bankroll=shared_bankroll._get_base_bankroll(),
                                strip_margin_asian=True,
                                strip_margin_totals=True,
                            )

                        for ap in multi.asian_proposals:
                            ar, pnl = orchestrators[lid].asian_strategy.settle(ap, home_goals, away_goals)
                            is_win = ar in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                            stake = ap.kelly_stake * getattr(ap, 'strategy_weight', 1.0)
                            net_pnl = tx_cost.apply_costs(pnl, stake) if pnl > 0 else pnl
                            result["total_bets"] += 1
                            result["total_staked"] += abs(stake)
                            daily_staked += abs(stake)
                            if lid not in result["by_league"]:
                                result["by_league"][lid] = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}
                            result["by_league"][lid]["bets"] += 1
                            result["by_league"][lid]["staked"] += abs(stake)
                            if is_win:
                                result["total_wins"] += 1
                                result["total_returned"] += stake + net_pnl
                                result["by_league"][lid]["wins"] += 1
                                result["by_league"][lid]["returned"] += stake + net_pnl
                            result["by_strategy"]["asian"][0] += 1
                            result["by_strategy"]["asian"][2] += abs(stake)
                            if is_win:
                                result["by_strategy"]["asian"][1] += 1
                                result["by_strategy"]["asian"][3] += stake + net_pnl

                        for tp in multi.totals_proposals:
                            tr, pnl = orchestrators[lid].over_under_strategy.settle(tp, home_goals, away_goals)
                            is_win = (tr == "win")
                            stake = tp.kelly_stake * getattr(tp, 'strategy_weight', 1.0)
                            net_pnl = tx_cost.apply_costs(pnl, stake) if pnl > 0 else pnl
                            result["total_bets"] += 1
                            result["total_staked"] += abs(stake)
                            daily_staked += abs(stake)
                            if lid not in result["by_league"]:
                                result["by_league"][lid] = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}
                            result["by_league"][lid]["bets"] += 1
                            result["by_league"][lid]["staked"] += abs(stake)
                            if is_win:
                                result["total_wins"] += 1
                                result["total_returned"] += stake + net_pnl
                                result["by_league"][lid]["wins"] += 1
                                result["by_league"][lid]["returned"] += stake + net_pnl
                            result["by_strategy"]["over_under"][0] += 1
                            result["by_strategy"]["over_under"][2] += abs(stake)
                            if is_win:
                                result["by_strategy"]["over_under"][1] += 1
                                result["by_strategy"]["over_under"][3] += stake + net_pnl
                    except Exception:
                        pass

                # Elo 更新
                elo_tracker.update_elo(
                    lid, dm.home_team, dm.away_team,
                    home_goals, away_goals,
                    k=params.get("elo_k", 24),
                    home_adv=home_adv,
                )

            # 结算当天串关
            settlements = parlay_mgr.settle_all_ready(day_results)
            for s in settlements:
                if s.won:
                    result["parlay_wins"] += 1
                    result["total_wins"] += 1

        # 赛季结束: 保存 Elo 快照
        season_elo_starts[season] = elo_tracker.snapshot()

    # 计算 ROI
    roi = (result["total_returned"] - result["total_staked"]) / max(result["total_staked"], 0.01)
    result["roi"] = roi
    result["profit"] = result["total_returned"] - result["total_staked"]
    result["win_rate"] = result["total_wins"] / max(result["total_bets"], 1)

    return result


# ============================================================================
# Walk-Forward 验证
# ============================================================================


def param_grid_search(
    train_seasons: List[str],
    val_season: str,
    leagues: List[str],
    odds_provider: OddsProvider,
    base_params: Dict[str, dict],
    fast_mode: bool = True,
) -> Dict[str, dict]:
    """
    在训练窗口上搜索最优参数，在验证窗口上评估。

    返回:
        {league_id: best_params}
    """
    best_params = {}
    for lid in leagues:
        search = LEAGUE_PARAM_SEARCH.get(lid, {})
        if not search or fast_mode:
            best_params[lid] = base_params[lid]
            continue

        best_roi = -float("inf")
        best_cfg = base_params[lid].copy()

        for vt in search.get("value_threshold", [base_params[lid]["value_threshold"]])[:2]:
            for ct in search.get("confidence_threshold", [base_params[lid]["confidence_threshold"]])[:2]:
                for sa in search.get("shrinkage_alpha_high", [base_params[lid]["shrinkage_alpha_high"]])[:2]:
                    trial_params = {
                        **base_params[lid],
                        "value_threshold": vt,
                        "confidence_threshold": ct,
                        "shrinkage_alpha_high": sa,
                    }

                    elo = EloTracker()
                    trial_result = run_window_seasons(
                        train_seasons, [lid], elo, odds_provider,
                        {lid: trial_params}, is_training=True,
                    )

                    # 在验证窗口上评估
                    val_result = run_window_seasons(
                        [val_season], [lid], elo, odds_provider,
                        {lid: trial_params}, is_training=False,
                    )

                    val_roi = val_result.get("roi", -1.0)
                    if val_roi > best_roi:
                        best_roi = val_roi
                        best_cfg = trial_params.copy()

        best_params[lid] = best_cfg

    return best_params


def generate_walk_forward_windows() -> List[Tuple[List[str], str, List[str]]]:
    """生成 Walk-Forward 窗口: (训练, 验证, 测试)"""
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


def run_walk_forward():
    """主 Walk-Forward 验证流程"""
    print("=" * 70)
    print("Phase 5.0 Walk-Forward 交叉验证")
    print(f"窗口: {WINDOW_CONFIG['train_size']}训练 + {WINDOW_CONFIG['val_size']}验证 + {WINDOW_CONFIG['test_size']}测试")
    print("=" * 70)

    odds_provider = get_odds_provider(use_pinnacle=True)
    windows = generate_walk_forward_windows()

    # Elo 冷启动: 从历史数据初始化 Elo 评分
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

    # 日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "bet_logs")
    os.makedirs(log_dir, exist_ok=True)

    all_test_results = []
    oos_rois = []

    for win_idx, (train_seasons, val_season, test_seasons) in enumerate(windows):
        print(f"\n{'─' * 60}")
        print(f"窗口 {win_idx + 1}/{len(windows)}")
        print(f"  训练: {train_seasons}")
        print(f"  验证: {val_season}")
        print(f"  测试: {test_seasons}")
        print(f"{'─' * 60}")

        # 阶段 1: 参数搜索 (训练 → 验证)
        print("  [1/3] 参数搜索...")
        best_params = param_grid_search(
            train_seasons, val_season, LEAGUES,
            odds_provider, DEFAULT_LEAGUE_PARAMS,
        )
        for lid in LEAGUES:
            p = best_params[lid]
            print(f"    {lid}: vt={p['value_threshold']}, ct={p['confidence_threshold']}, "
                  f"sa={p['shrinkage_alpha_high']}")

        # 阶段 2: 在训练+验证上最终训练
        print("  [2/3] 最终训练 (训练+验证)...")
        final_elo = EloTracker()
        # 注入冷启动 Elo
        if elo_cs:
            for lid in LEAGUES:
                for team, elo_val in elo_cs.get_all_elos(lid).items():
                    if lid not in final_elo.elos:
                        final_elo.elos[lid] = {}
                    final_elo.elos[lid][team] = elo_val
        _ = run_window_seasons(
            train_seasons + [val_season], LEAGUES,
            final_elo, odds_provider, best_params,
            is_training=True,
        )

        # 阶段 3: OOS 测试 (仅运行一次)
        print("  [3/3] OOS 测试...")
        bet_logger = BetLogger(log_dir, session_id=f"wf_win{win_idx+1}")
        test_result = run_window_seasons(
            test_seasons, LEAGUES,
            final_elo, odds_provider, best_params,
            bet_logger=bet_logger, is_training=False,
        )

        bet_logger.flush()

        test_roi = test_result.get("roi", 0.0)
        oos_rois.append(test_roi)
        all_test_results.append(test_result)

        print(f"  OOS 结果: 投注={test_result['total_bets']}, "
              f"胜率={test_result['win_rate']:.1%}, "
              f"ROI={test_roi:+.1%}, "
              f"利润={test_result['profit']:+,.0f}")

    # ── 最终汇总 ──
    print(f"\n{'=' * 70}")
    print("Walk-Forward 最终汇总")
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

    # 各窗口 ROI
    print(f"\n各窗口 OOS ROI:")
    for i, roi in enumerate(oos_rois):
        print(f"  窗口 {i+1}: {roi:+.1%}")

    # 策略维度
    print(f"\n策略维度 (OOS):")
    for strat in ["1x2", "asian", "over_under"]:
        s_bets = sum(r["by_strategy"].get(strat, [0, 0, 0, 0])[0] for r in all_test_results)
        s_wins = sum(r["by_strategy"].get(strat, [0, 0, 0, 0])[1] for r in all_test_results)
        s_staked = sum(r["by_strategy"].get(strat, [0, 0, 0, 0])[2] for r in all_test_results)
        s_returned = sum(r["by_strategy"].get(strat, [0, 0, 0, 0])[3] for r in all_test_results)
        if s_bets > 0:
            s_roi = (s_returned - s_staked) / max(s_staked, 0.01)
            print(f"  {cn_strategy(strat)}: {s_bets}注, 胜率{s_wins/max(s_bets,1):.1%}, ROI{s_roi:+.1%}")

    # 联赛维度
    print(f"\n联赛维度 (OOS):")
    league_agg = defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0})
    for r in all_test_results:
        for lid, data in r.get("by_league", {}).items():
            league_agg[lid]["bets"] += data["bets"]
            league_agg[lid]["wins"] += data["wins"]
            league_agg[lid]["staked"] += data["staked"]
            league_agg[lid]["returned"] += data["returned"]
    for lid, data in sorted(league_agg.items()):
        if data["bets"] > 0:
            l_roi = (data["returned"] - data["staked"]) / max(data["staked"], 0.01)
            l_wr = data["wins"] / max(data["bets"], 1)
            print(f"  {cn_league(lid)}: {data['bets']}注 ({data['bets']/max(total_bets,1)*100:.1f}%), "
                  f"胜率{l_wr:.1%}, ROI{l_roi:+.1%}")

    # 保存结果
    results_path = os.path.join(log_dir, "walk_forward_results.json")
    results_data = {
        "config": WINDOW_CONFIG,
        "windows": [{"train": w[0], "val": w[1], "test": w[2]} for w in windows],
        "oos_rois": oos_rois,
        "summary": {
            "total_bets": total_bets, "total_wins": total_wins,
            "win_rate": oos_win_rate, "roi": oos_roi, "profit": total_profit,
            "total_staked": total_staked, "total_returned": total_returned,
        },
        "by_league": {
            lid: {**data, "roi": (data["returned"] - data["staked"]) / max(data["staked"], 0.01)}
            for lid, data in league_agg.items()
        },
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {results_path}")

    return oos_roi, oos_win_rate, total_profit


if __name__ == "__main__":
    run_walk_forward()