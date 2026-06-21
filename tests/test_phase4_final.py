"""
GTO-GameFlow v5.9.2 — Phase 4: 无未来信息 + 同日串关 + 多联赛均衡
- 按日期增量加载比赛，严格禁止未来信息
- 追踪真实历史近期战绩（替代合成随机数据）
- 同日比赛串关
- 修复非英超联赛投注为0的问题
- 联赛投注比例均衡
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv, random
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from src.data.models import (
    MatchContext, BetSelection, BetResult, AsianHandicapResult,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.strategies.strategy_orchestrator import ParlayBatchManager, create_orchestrator
from src.data.historical_odds_loader import (
    HistoricalMatchOdds, load_odds_for_season, DATA_DIR, TEAM_NAME_MAP,
)
from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
from tests.test_backtesting_real import _load_calibrated_weights, _build_weight_multipliers
from tests.test_phase2_optimization import DynamicKellyEngine, RiskControlLayer
from src.data.real_orthogonal_loader import RealOddsEnhancer
from src.config.settings import GlobalConfig
from src.utils.i18n import cn_league, cn_strategy
version = GlobalConfig.version

# ═══════════════════════════════════════════════════════════════
# Phase 4: 无未来信息比赛加载器
# ═══════════════════════════════════════════════════════════════

@dataclass
class DatedMatch:
    """带日期的比赛数据"""
    match: HistoricalMatchOdds
    date: datetime
    home_team: str
    away_team: str

def load_matches_by_date(league_id: str, season: str) -> List[List[DatedMatch]]:
    """
    按日期分组加载比赛，返回按日期排序的分组列表。
    每个日期可能有多场比赛。
    """
    odds_data = load_odds_for_season(league_id, season)
    if not odds_data:
        return []

    # 按日期分组
    by_date: Dict[str, List[DatedMatch]] = defaultdict(list)
    for key, match in odds_data.items():
        # 从 CSV 中重新读取日期
        # 由于 HistoricalMatchOdds 不含日期，需要从原始 CSV 读取
        pass

    # 需要从 CSV 重新读取以获取日期
    return _load_dated_matches_from_csv(league_id, season)


def _load_dated_matches_from_csv(league_id: str, season: str) -> List[List[DatedMatch]]:
    """从 CSV 按日期分组加载比赛"""
    from src.data.historical_odds_loader import DATA_DIR, TEAM_NAME_MAP
    
    filename = f"{league_id}_{season.replace('/', '-')}.csv"
    filepath = os.path.join(str(DATA_DIR), filename)
    
    if not os.path.exists(filepath):
        return []
    
    name_map = TEAM_NAME_MAP.get(league_id, {})
    
    by_date: Dict[str, List[DatedMatch]] = defaultdict(list)
    
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = (row.get("Date") or "").strip()
            home_en = (row.get("HomeTeam") or "").strip()
            away_en = (row.get("AwayTeam") or "").strip()
            if not date_str or not home_en or not away_en:
                continue
            
            # 解析日期 DD/MM/YYYY 或 DD/MM/YY
            try:
                parts = date_str.split("/")
                if len(parts) == 3:
                    d, m = int(parts[0]), int(parts[1])
                    y = int(parts[2])
                    if y < 100:  # DD/MM/YY 格式
                        y += 2000
                    match_date = datetime(y, m, d)
                else:
                    continue
            except (ValueError, IndexError):
                continue
            
            home = name_map.get(home_en, home_en)
            away = name_map.get(away_en, away_en)
            
            # 构建 HistoricalMatchOdds
            def _sf(v):
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            
            match = HistoricalMatchOdds(
                date=date_str,
                home_team=home,
                away_team=away,
                home_goals=int(row.get("FTHG", 0) or 0),
                away_goals=int(row.get("FTAG", 0) or 0),
                result=row.get("FTR", "").strip(),
                b365_h=_sf(row.get("B365H")),
                b365_d=_sf(row.get("B365D")),
                b365_a=_sf(row.get("B365A")),
                avg_h=_sf(row.get("BbAvH")),
                avg_d=_sf(row.get("BbAvD")),
                avg_a=_sf(row.get("BbAvA")),
                asian_handicap=_sf(row.get("BbAHh")),
                asian_home_odds=_sf(row.get("BbAvAHH")),
                asian_away_odds=_sf(row.get("BbAvAHA")),
                over_odds=_sf(row.get("BbAv>2.5")),
                under_odds=_sf(row.get("BbAv<2.5")),
            )
            
            dm = DatedMatch(match=match, date=match_date, home_team=home, away_team=away)
            by_date[date_str].append(dm)
    
    # 按日期排序
    def parse_date(ds):
        parts = ds.split("/")
        d, m = int(parts[0]), int(parts[1])
        y = int(parts[2])
        if y < 100:
            y += 2000
        return datetime(y, m, d)
    sorted_dates = sorted(by_date.keys(), key=parse_date)
    return [by_date[d] for d in sorted_dates]


# ═══════════════════════════════════════════════════════════════
# 历史战绩追踪器 (无未来信息)
# ═══════════════════════════════════════════════════════════════

class HistoricalTracker:
    """追踪每支球队的历史战绩，仅基于已完成的比赛"""
    
    def __init__(self, league_id: str):
        self.league_id = league_id
        self.team_results: Dict[str, List[float]] = defaultdict(list)  # 3=胜, 1=平, 0=负
        self.team_h2h: Dict[str, List[float]] = defaultdict(list)       # 历史交锋
        self.team_goals_for: Dict[str, List[int]] = defaultdict(list)
        self.team_goals_against: Dict[str, List[int]] = defaultdict(list)
    
    def get_recent_results(self, team: str, n: int = 5) -> List[float]:
        """获取最近 n 场结果 (3=胜, 1=平, 0=负)"""
        results = self.team_results.get(team, [])
        recent = results[-n:] if len(results) >= n else results
        # 不足时用 0 填充
        while len(recent) < n:
            recent.insert(0, 0.0)
        return recent
    
    def get_h2h_results(self, home: str, away: str, n: int = 5) -> List[float]:
        """获取两队历史交锋结果"""
        key = f"{home}|{away}"
        results = self.team_h2h.get(key, [])
        recent = results[-n:] if len(results) >= n else results
        while len(recent) < n:
            recent.insert(0, 0.0)
        return recent
    
    def get_goal_diff(self, team: str, n: int = 5) -> float:
        """最近 n 场进球差"""
        gf = self.team_goals_for.get(team, [])[-n:]
        ga = self.team_goals_against.get(team, [])[-n:]
        if not gf:
            return 0.0
        return (sum(gf) - sum(ga)) / len(gf)
    
    def record_result(self, home: str, away: str, home_goals: int, away_goals: int, result: str):
        """记录比赛结果 (仅在比赛结束后调用)"""
        if result == "H":
            self.team_results[home].append(3.0)
            self.team_results[away].append(0.0)
        elif result == "A":
            self.team_results[home].append(0.0)
            self.team_results[away].append(3.0)
        else:
            self.team_results[home].append(1.0)
            self.team_results[away].append(1.0)
        
        # 历史交锋
        h2h_key = f"{home}|{away}"
        h2h_rev = f"{away}|{home}"
        if result == "H":
            self.team_h2h[h2h_key].append(3.0)
            self.team_h2h[h2h_rev].append(0.0)
        elif result == "A":
            self.team_h2h[h2h_key].append(0.0)
            self.team_h2h[h2h_rev].append(3.0)
        else:
            self.team_h2h[h2h_key].append(1.0)
            self.team_h2h[h2h_rev].append(1.0)
        
        self.team_goals_for[home].append(home_goals)
        self.team_goals_for[away].append(away_goals)
        self.team_goals_against[home].append(away_goals)
        self.team_goals_against[away].append(home_goals)


# ═══════════════════════════════════════════════════════════════
# Phase 4: 主回测函数
# ═══════════════════════════════════════════════════════════════

def run_phase4_season(
    season: str,
    initial_bankroll: float = 10000.0,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """
    Phase 4 赛季回测:
    - 按日期增量加载
    - 真实历史战绩
    - 同日串关
    - 多联赛均衡阈值
    """
    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    
    # ── 联赛特定优化参数 (降低非英超阈值以激活投注) ──
    league_params = {
        "premier_league": {
            "value_threshold": 0.015, "confidence_threshold": 0.55,
            "shrinkage_alpha_high": 0.55, "shrinkage_alpha_low": 0.12,
            "elo_k": 24, "home_advantage": 65,
            "calib_multiplier_base": 0.8, "calib_multiplier_enhanced": 0.8, "calib_multiplier_league": 0.8,
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

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    # 串关管理器: 同一天的所有比赛共享
    parlay_mgr = ParlayBatchManager(
        max_legs=4, kelly_discount=0.35, max_batch_size=20,
        min_single_value=0.01, min_combined_value=0.015
    )
    kelly_engine = DynamicKellyEngine(base_discount=0.25)
    risk_control = RiskControlLayer()

    if carryover_elos is None:
        carryover_elos = {}

    result = {
        "season": season,
        "total_bets": 0, "total_wins": 0, "total_staked": 0.0, "total_returned": 0.0,
        "risk_events": 0,
        "parlay_bets": 0, "parlay_wins": 0, "parlay_staked": 0.0, "parlay_returned": 0.0,
        "by_league": {},
        "by_strategy": {"1x2": [0, 0, 0.0, 0.0], "asian": [0, 0, 0.0, 0.0],
                        "over_under": [0, 0, 0.0, 0.0], "parlay": [0, 0, 0.0, 0.0]},
    }

    # 为每个联赛创建历史追踪器
    trackers = {lid: HistoricalTracker(lid) for lid in leagues}

    # 加载所有联赛的按日期分组比赛
    all_league_dates: Dict[str, List[List[DatedMatch]]] = {}
    for lid in leagues:
        all_league_dates[lid] = _load_dated_matches_from_csv(lid, season)

    # 所有联赛的比赛日期合并排序
    all_dates: Dict[datetime, List[Tuple[str, DatedMatch]]] = defaultdict(list)
    for lid in leagues:
        for day_matches in all_league_dates[lid]:
            if day_matches:
                date = day_matches[0].date
                for dm in day_matches:
                    all_dates[date].append((lid, dm))

    sorted_dates = sorted(all_dates.keys())

    # 初始化 Elo
    team_elos: Dict[str, Dict[str, float]] = {}
    for lid in leagues:
        if lid in carryover_elos:
            team_elos[lid] = dict(carryover_elos[lid])
        else:
            team_elos[lid] = {}

    # 初始化 pipeline 缓存
    pipelines: Dict[str, GameFlowPipeline] = {}
    orchestrators: Dict[str, any] = {}
    enhancers: Dict[str, RealOddsEnhancer] = {}
    
    for lid in leagues:
        params = league_params[lid]
        calib = _load_calibrated_weights().get(lid, {})
        # 使用提升后的校准乘数
        weight_multipliers = {
            "base": params["calib_multiplier_base"],
            "enhanced": params["calib_multiplier_enhanced"],
            "league": params["calib_multiplier_league"],
        }
        pipeline = GameFlowPipeline(lid, initial_bankroll=initial_bankroll,
                                     weight_multipliers=weight_multipliers)
        pipeline.set_bankroll_manager(shared_bankroll)
        pipeline.prior_shrinkage = PriorShrinkage(
            alpha_high=params["shrinkage_alpha_high"],
            alpha_low=params["shrinkage_alpha_low"],
        )
        pipelines[lid] = pipeline
        orchestrators[lid] = create_orchestrator(lid)
        enhancers[lid] = RealOddsEnhancer(lid, season)

    # ── 按日期逐日处理所有联赛的比赛 ──
    matchup_count = 0
    day_results: Dict[str, Tuple[str, BetSelection]] = {}  # 当天比赛结果 (用于同日串关结算)

    for date in sorted_dates:
        day_matches = all_dates[date]
        day_results.clear()
        daily_staked = 0.0  # 当日已投注金额
        
        # 重置当天串关池
        parlay_mgr = ParlayBatchManager(
            max_legs=2, kelly_discount=0.30, max_batch_size=10,
            min_single_value=0.025, min_combined_value=0.045
        )

        # 处理当天所有联赛的所有比赛
        for lid, dm in day_matches:
            try:
                matchup_count += 1
                params = league_params[lid]
                pipeline = pipelines[lid]
                orchestrator = orchestrators[lid]
                tracker = trackers[lid]
                enhancer = enhancers[lid]

                # 初始化 Elo
                if dm.home_team not in team_elos[lid]:
                    team_elos[lid][dm.home_team] = 1650.0
                if dm.away_team not in team_elos[lid]:
                    team_elos[lid][dm.away_team] = 1650.0

                home_elo = team_elos[lid][dm.home_team]
                away_elo = team_elos[lid][dm.away_team]

                match = dm.match
                odds_h = match.avg_h or match.b365_h
                odds_d = match.avg_d or match.b365_d
                odds_a = match.avg_a or match.b365_a
                if not (odds_h and odds_d and odds_a):
                    continue

                if match.result == "H":
                    actual_outcome = "home_win"
                elif match.result == "D":
                    actual_outcome = "draw"
                elif match.result == "A":
                    actual_outcome = "away_win"
                else:
                    continue

                outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW,
                               "away_win": BetSelection.AWAY_WIN}
                actual = outcome_map[actual_outcome]

                match_ctx = MatchContext(
                    match_id=f"{lid}_S{season}_D{date.strftime('%Y%m%d')}_M{matchup_count:04d}",
                    league_id=lid, season=season, matchday=matchup_count,
                    kickoff_time=date,
                    home_team=dm.home_team, away_team=dm.away_team,
                    home_elo=home_elo, away_elo=away_elo,
                    odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
                )

                # ── 使用真实历史数据 (无未来信息) ──
                recent_home = tracker.get_recent_results(dm.home_team, 5)
                recent_away = tracker.get_recent_results(dm.away_team, 5)
                h2h = tracker.get_h2h_results(dm.home_team, dm.away_team, 5)
                goal_diff = tracker.get_goal_diff(dm.home_team, 5) - tracker.get_goal_diff(dm.away_team, 5)

                extra = enhancer.enhance_ortho_data(
                    dm.home_team, dm.away_team, date, odds_h, odds_d, odds_a,
                )
                extra["elo_diff"] = home_elo - away_elo
                extra["recent_results"] = recent_home
                extra["h2h_results"] = h2h
                extra["rank_diff"] = int((home_elo - away_elo) / 20)
                extra["goal_diff"] = goal_diff * 2
                extra["xg_diff"] = goal_diff / 10
                extra["streak_momentum"] = sum(1 for r in recent_home if r == 3.0) / max(len(recent_home), 1) - \
                                            sum(1 for r in recent_away if r == 3.0) / max(len(recent_away), 1)
                extra["streak_momentum_league"] = extra["streak_momentum"]
                extra["match_phase"] = 1.0

                if enhancer.get_real_odds_movement(dm.home_team, dm.away_team, odds_h, odds_d, odds_a)["opening_probs"]:
                    extra["opening_probs"] = enhancer.get_real_odds_movement(
                        dm.home_team, dm.away_team, odds_h, odds_d, odds_a)["opening_probs"]

                pipeline_result = pipeline.run_full(match_ctx, extra_data=extra)

                # ── 串关池: 从 1X2 value_results 生成低阈值提案 ──
                if pipeline_result.value_results and pipeline_result.fused_probs:
                    from src.engine.bankroll import generate_bet_proposals
                    parlay_proposals = generate_bet_proposals(
                        pipeline_result.value_results,
                        match_id=match_ctx.match_id,
                        league_id=lid,
                        factor_count=37,
                        data_source_count=extra.get("data_source_count", 5),
                        odds_std=extra.get("odds_std", 0.05),
                        match_phase=extra.get("match_phase", 1.0),
                        threshold=0.020,
                        confidence_threshold=0.50,
                    )
                    if parlay_proposals:
                        parlay_mgr.add_match_bets(match_ctx.match_id, parlay_proposals)

                # 保存当天赛果 (所有比赛都需要，用于串关结算)
                day_results[match_ctx.match_id] = (actual_outcome, actual, (match.home_goals, match.away_goals))

                if pipeline_result.placements:
                    placements = pipeline.settle_bets(pipeline_result.placements, actual)
                    for p in placements:
                        # 动态 Kelly
                        if pipeline_result.fused_probs:
                            imp = 1/odds_h + 1/odds_d + 1/odds_a
                            sel_map = {BetSelection.HOME_WIN: "home", BetSelection.DRAW: "draw",
                                       BetSelection.AWAY_WIN: "away"}
                            key = sel_map.get(p.selection, "home")
                            market_prob = {"home": (1/odds_h)/imp, "draw": (1/odds_d)/imp,
                                           "away": (1/odds_a)/imp}[key]
                            model_prob = {"home": pipeline_result.fused_probs.prob_home,
                                          "draw": pipeline_result.fused_probs.prob_draw,
                                          "away": pipeline_result.fused_probs.prob_away}[key]
                            discount = kelly_engine.compute_discount(
                                value_signal=abs(model_prob - market_prob),
                                odds_std=extra.get("odds_std", 0.05),
                                market_efficiency=extra.get("market_efficiency", 0.05),
                                model_prob=model_prob, market_prob=market_prob,
                            )
                            p.stake *= (discount / 0.25)

                        approved, adj_stake, _ = risk_control.check_bet(
                            p.stake, p.odds,
                            shared_bankroll._get_base_bankroll(),
                            daily_staked,
                            shared_bankroll.state.balance,
                        )
                        if not approved:
                            result["risk_events"] += 1
                            continue
                        p.stake = adj_stake

                        result["total_bets"] += 1
                        result["total_staked"] += p.stake
                        daily_staked += p.stake
                        if lid not in result["by_league"]:
                            result["by_league"][lid] = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}
                        result["by_league"][lid]["bets"] += 1
                        result["by_league"][lid]["staked"] += p.stake
                        if p.result == BetResult.WIN:
                            result["total_wins"] += 1
                            result["total_returned"] += p.stake + p.profit_loss
                            result["by_league"][lid]["wins"] += 1
                            result["by_league"][lid]["returned"] += p.stake + p.profit_loss
                        result["by_strategy"]["1x2"][0] += 1
                        result["by_strategy"]["1x2"][2] += p.stake
                        if p.result == BetResult.WIN:
                            result["by_strategy"]["1x2"][1] += 1
                            result["by_strategy"]["1x2"][3] += p.stake + p.profit_loss
                        kelly_engine.record_result(p.profit_loss)

                    # 亚盘 + 大小球
                    try:
                        synth_handicap = {}
                        if match.asian_handicap is not None and match.asian_home_odds and match.asian_away_odds:
                            synth_handicap[match.asian_handicap] = {"home": match.asian_home_odds, "away": match.asian_away_odds}
                        else:
                            synth_handicap = orchestrator.asian_strategy.generate_synthetic_odds(
                                pipeline_result.poisson_score_matrix)

                        synth_totals = {}
                        if match.over_odds and match.under_odds:
                            synth_totals[2.5] = {"over": match.over_odds, "under": match.under_odds}
                        else:
                            synth_totals = orchestrator.over_under_strategy.generate_synthetic_odds(
                                pipeline_result.poisson_score_matrix)

                        multi = orchestrator.run(
                            match=match_ctx, score_matrix=pipeline_result.poisson_score_matrix,
                            handicap_odds=synth_handicap, totals_odds=synth_totals,
                            total_bankroll=shared_bankroll._get_base_bankroll(),
                        )

                        # 亚盘/大小球提案不加入串关池 (它们的 value 估计偏差较大)
                        # 仅 1X2 提案入池，质量更高

                        for ap in multi.asian_proposals:
                            ar, pnl = orchestrator.asian_strategy.settle(ap, match.home_goals, match.away_goals)
                            is_win = ar in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                            stake = ap.kelly_stake * getattr(ap, 'strategy_weight', 1.0)
                            result["total_bets"] += 1
                            result["total_staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                            if lid not in result["by_league"]:
                                result["by_league"][lid] = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}
                            result["by_league"][lid]["bets"] += 1
                            result["by_league"][lid]["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                            if is_win:
                                result["total_wins"] += 1
                                result["total_returned"] += stake + pnl
                                result["by_league"][lid]["wins"] += 1
                                result["by_league"][lid]["returned"] += stake + pnl
                            result["by_strategy"]["asian"][0] += 1
                            result["by_strategy"]["asian"][2] += abs(stake) if stake > 0 else ap.kelly_stake
                            if is_win:
                                result["by_strategy"]["asian"][1] += 1
                                result["by_strategy"]["asian"][3] += stake + pnl
                        for tp in multi.totals_proposals:
                            tr, pnl = orchestrator.over_under_strategy.settle(tp, match.home_goals, match.away_goals)
                            is_win = (tr == "win")
                            stake = tp.kelly_stake * getattr(tp, 'strategy_weight', 1.0)
                            result["total_bets"] += 1
                            result["total_staked"] += abs(stake) if stake > 0 else tp.kelly_stake
                            if lid not in result["by_league"]:
                                result["by_league"][lid] = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}
                            result["by_league"][lid]["bets"] += 1
                            result["by_league"][lid]["staked"] += abs(stake) if stake > 0 else tp.kelly_stake
                            if is_win:
                                result["total_wins"] += 1
                                result["total_returned"] += stake + pnl
                                result["by_league"][lid]["wins"] += 1
                                result["by_league"][lid]["returned"] += stake + pnl
                            result["by_strategy"]["over_under"][0] += 1
                            result["by_strategy"]["over_under"][2] += abs(stake) if stake > 0 else tp.kelly_stake
                            if is_win:
                                result["by_strategy"]["over_under"][1] += 1
                                result["by_strategy"]["over_under"][3] += stake + pnl
                    except Exception:
                        pass

                # ── 记录结果 (更新 Elo 和历史追踪器) ──
                if match.result == "H":
                    actual_h, actual_a = 1.0, 0.0
                elif match.result == "A":
                    actual_h, actual_a = 0.0, 1.0
                else:
                    actual_h, actual_a = 0.5, 0.5
                exp_h = 1.0 / (1.0 + 10 ** (-(home_elo + params["home_advantage"] - away_elo) / 400.0))
                exp_a = 1.0 - exp_h
                team_elos[lid][dm.home_team] = home_elo + params["elo_k"] * (actual_h - exp_h)
                team_elos[lid][dm.away_team] = away_elo + params["elo_k"] * (actual_a - exp_a)

                # 记录到历史追踪器 (比赛结束后)
                tracker.record_result(dm.home_team, dm.away_team, match.home_goals, match.away_goals, match.result)

            except Exception:
                continue

        # ── 当天所有比赛处理完毕，生成并结算同日串关 ──
        if len(parlay_mgr._pending_pool) >= 2:
            parlays = parlay_mgr.generate_batch(shared_bankroll._get_base_bankroll())
            for parlay in parlays:
                approved, adj_stake, _ = risk_control.check_bet(
                    parlay.adjusted_stake, parlay.combined_odds,
                    shared_bankroll._get_base_bankroll(),
                    daily_staked, shared_bankroll.state.balance,
                )
                if not approved:
                    # 风控拒绝的串关从活跃列表移除，避免被结算
                    parlay_mgr._active_parlays.pop(parlay.parlay_id, None)
                    continue
                parlay.adjusted_stake = adj_stake
                shared_bankroll.state.balance -= adj_stake
                result["parlay_bets"] += 1
                result["parlay_staked"] += adj_stake
                result["total_bets"] += 1
                result["total_staked"] += adj_stake
                daily_staked += adj_stake

        # 结算当天串关
        settlements = parlay_mgr.settle_all_ready(day_results)
        for s in settlements:
            shared_bankroll.state.balance += s.returned
            result["total_returned"] += s.returned
            result["parlay_returned"] += s.returned
            if s.won:
                result["parlay_wins"] += 1
                result["total_wins"] += 1

    # 保存 Elo 供下赛季
    carryover_elos = team_elos

    result["by_strategy"]["parlay"] = [
        result["parlay_bets"], result["parlay_wins"],
        result["parlay_staked"], result["parlay_returned"],
    ]
    return result, carryover_elos


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print(f"  GTO-GameFlow {version} Phase 4: 无未来信息 + 同日串关 + 多联赛均衡")
    print("=" * 78)
    print("  配置:")
    print("  - 按日期增量加载 (禁止未来信息)")
    print("  - 真实历史战绩追踪 (替代合成随机数据)")
    print("  - 同日比赛串关 (跨联赛)")
    print("  - 多联赛均衡阈值 (非英超投注激活)")
    print("  - 动态 Kelly + 风控层")
    print("  - 固定基数 10,000 | 10赛季: 2014/15 → 2023/24")
    print()

    seasons = [f"{y}/{str(y+1)[-2:]}" for y in range(2014, 2024)]
    all_results = []
    elos = None
    totals = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}

    print(f"  {'赛季':<10} {'总投注':>6} {'单场':>6} {'串关':>6} {'胜率':>7} {'ROI':>8} {'利润':>10}")
    print(f"  {'-'*65}")

    for season in seasons:
        r, elos = run_phase4_season(season, 10000.0, elos)
        all_results.append(r)
        totals["bets"] += r["total_bets"]
        totals["wins"] += r["total_wins"]
        totals["staked"] += r["total_staked"]
        totals["returned"] += r["total_returned"]

        wr = r["total_wins"] / r["total_bets"] if r["total_bets"] > 0 else 0
        roi = (r["total_returned"] - r["total_staked"]) / r["total_staked"] if r["total_staked"] > 0 else 0
        profit = r["total_returned"] - r["total_staked"]
        single = r["total_bets"] - r["parlay_bets"]
        print(f"  {season:<10} {r['total_bets']:>6} {single:>6} {r['parlay_bets']:>6} "
              f"{wr:>6.1%} {roi:>+7.1%} {profit:+10,.0f}")

    wr = totals["wins"] / totals["bets"] if totals["bets"] > 0 else 0
    roi = (totals["returned"] - totals["staked"]) / totals["staked"] if totals["staked"] > 0 else 0
    profit = totals["returned"] - totals["staked"]
    total_parlay = sum(r["parlay_bets"] for r in all_results)

    print(f"  {'-'*65}")
    print(f"  {'合计':<10} {totals['bets']:>6} {totals['bets']-total_parlay:>6} {total_parlay:>6} "
          f"{wr:>6.1%} {roi:>+7.1%} {profit:+10,.0f}")

    # 联赛明细
    print(f"\n  {'联赛':<16} {'投注':>6} {'占比':>7} {'胜率':>7} {'ROI':>8} {'利润':>10}")
    print(f"  {'-'*55}")
    for lid in ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]:
        lb = sum(r["by_league"].get(lid, {}).get("bets", 0) for r in all_results)
        lw = sum(r["by_league"].get(lid, {}).get("wins", 0) for r in all_results)
        ls = sum(r["by_league"].get(lid, {}).get("staked", 0) for r in all_results)
        lr = sum(r["by_league"].get(lid, {}).get("returned", 0) for r in all_results)
        if lb > 0:
            lwr = lw / lb
            lroi = (lr - ls) / ls if ls > 0 else 0
            pct = lb / totals["bets"] * 100 if totals["bets"] > 0 else 0
            print(f"  {cn_league(lid):<16} {lb:>6} {pct:>6.1f}% {lwr:>6.1%} {lroi:>+7.1%} {lr-ls:+10,.0f}")

    # 策略明细
    print(f"\n  {'策略':<12} {'投注':>6} {'胜率':>7} {'ROI':>8} {'利润':>10}")
    print(f"  {'-'*50}")
    name_map = {"1x2": "胜平负", "asian": "亚盘", "over_under": "大小球", "parlay": "串关"}
    for strat in ["1x2", "asian", "over_under", "parlay"]:
        nb = sum(r["by_strategy"][strat][0] for r in all_results)
        nw = sum(r["by_strategy"][strat][1] for r in all_results)
        st = sum(r["by_strategy"][strat][2] for r in all_results)
        ret = sum(r["by_strategy"][strat][3] for r in all_results)
        if nb > 0:
            s_wr = nw / nb
            s_roi = (ret - st) / st if st > 0 else 0
            print(f"  {name_map[strat]:<12} {nb:>6} {s_wr:>6.1%} {s_roi:>+7.1%} {ret-st:+10,.0f}")

    print(f"\n{'='*78}")
    print(f"  v{version} Phase 4 完成")
    print(f"{'='*78}")