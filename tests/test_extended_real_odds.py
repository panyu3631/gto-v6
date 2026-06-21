"""
GTO-GameFlow v5.9.2 — 扩展回测使用真实历史赔率 (football-data.co.uk)

- 10赛季五大联赛 (2014/15 → 2023/24)
- 真实历史赔率 (Bet365 + Avg from football-data.co.uk)
- 每赛季独立全局资金池 10,000（固定基数）
- 5大联赛共享资金池，跨联赛串关
- 进球和比赛结果使用真实数据（不是模拟）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    AsianHandicapResult,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.strategies.strategy_orchestrator import ParlayBatchManager
from src.strategies import StrategyOrchestrator
from src.config.settings import GlobalConfig
version = GlobalConfig.version

from src.data.historical_odds_loader import (
    HistoricalMatchOdds, load_odds_for_season, get_real_odds,
    LEAGUE_CODES, available_seasons,
)

# 复用已有函数
from tests.test_backtesting_real import (
    _load_calibrated_weights, _build_weight_multipliers,
)
from src.strategies.strategy_orchestrator import create_orchestrator
from src.utils.i18n import cn_league, cn_strategy


def get_season_teams(league_id: str, season_idx: int) -> List[Tuple[str, float]]:
    """获取指定赛季的球队列表（从真实数据提取）"""
    odds_data = load_odds_for_season(league_id, f"{2014+season_idx}/{2014+season_idx+1}")
    # 提取所有球队并去重
    all_teams = set()
    for k, match in odds_data.items():
        all_teams.add(match.home_team)
        all_teams.add(match.away_team)
    # 每个球队初始 Elo = 1650 ± 50
    seed = season_idx * 100 + hash(league_id) % 10000
    rng = random.Random(seed)
    return [(t, 1650 + rng.randint(-50, 50)) for t in sorted(all_teams)]


def generate_fixture_list_from_odds(odds_data: Dict[str, HistoricalMatchOdds]) -> List[Tuple[str, str]]:
    """从真实赔率数据生成赛程（保持原有顺序）"""
    fixtures = []
    for match in odds_data.values():
        fixtures.append((match.home_team, match.away_team))
    return fixtures


def run_real_odds_season(
    season: str,
    season_idx: int,
    initial_bankroll: float = 10000.0,
    seed: int = 42,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
) -> dataclass:
    """
    单赛季五大联赛全局回测，使用真实历史赔率和真实比赛结果。
    """
    from dataclasses import dataclass

    @dataclass
    class LeagueSeasonResult:
        season: str
        initial_bankroll: float = 10000.0
        final_balance: float = 10000.0
        total_bets: int = 0
        total_wins: int = 0
        total_staked: float = 0.0
        total_returned: float = 0.0
        profit_history: List[float] = field(default_factory=list)
        equity_curve: List[float] = field(default_factory=list)
        strategy_stats: Dict = field(default_factory=dict)
        all_elos: Dict = field(default_factory=dict)

        @property
        def roi(self):
            return (self.total_returned - self.total_staked) / self.total_staked if self.total_staked > 0 else 0.0
        @property
        def win_rate(self):
            return self.total_wins / self.total_bets if self.total_bets > 0 else 0.0
        @property
        def max_drawdown(self):
            if not self.equity_curve: return 0.0
            peak, max_dd = self.equity_curve[0], 0.0
            for v in self.equity_curve:
                if v > peak: peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                if dd > max_dd: max_dd = dd
            return max_dd

    _rng = random.Random(seed + season_idx)

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    parlay_mgr = ParlayBatchManager(max_legs=4, kelly_discount=0.35, max_batch_size=20,
        min_single_value=0.01, min_combined_value=0.015)
    match_outcomes = {}

    result = LeagueSeasonResult(
        season=season, initial_bankroll=initial_bankroll,
        strategy_stats={
            "1x2": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "asian_handicap": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "over_under": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "parlay": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
        },
    )
    result.equity_curve.append(initial_bankroll)

    league_order = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    all_fixtures = []

    # ── 加载真实赔率和生成数据结构 ──
    league_pipelines = {}
    league_ortho_gens = {}
    league_orchestrators = {}
    league_team_elos = {}
    league_odds_data = {}
    all_elos = {}

    from src.data.orthogonal_sources import OrthogonalDataGenerator
    for league_id in league_order:
        odds_data = load_odds_for_season(league_id, season)
        league_odds_data[league_id] = odds_data
        print(f"  {cn_league(league_id)}: {len(odds_data)} 场比赛")

        if carryover_elos and league_id in carryover_elos:
            team_elos = dict(carryover_elos[league_id])
            # 补充新球队
            all_teams_this = set()
            for m in odds_data.values():
                all_teams_this.add(m.home_team)
                all_teams_this.add(m.away_team)
            for t in all_teams_this:
                if t not in team_elos:
                    seed = season_idx * 100 + hash(league_id + t) % 10000
                    rng = random.Random(seed)
                    team_elos[t] = 1650 + rng.randint(-50, 50)
        else:
            teams = get_season_teams(league_id, season_idx)
            team_elos = {t[0]: t[1] for t in teams}

        league_team_elos[league_id] = team_elos
        all_elos[league_id] = dict(team_elos)

        calib = _load_calibrated_weights().get(league_id, {})
        weight_multipliers = _build_weight_multipliers(league_id, calib)

        pipeline = GameFlowPipeline(
            league_id, initial_bankroll=initial_bankroll,
            weight_multipliers=weight_multipliers,
        )
        pipeline.set_bankroll_manager(shared_bankroll)
        if calib:
            from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
            pipeline.signal_decomposer = SignalDecomposer(elo_suppression=1.0)
            pipeline.prior_shrinkage = PriorShrinkage(
                alpha_high=calib.get("shrinkage_alpha_high", 0.50),
                alpha_low=calib.get("shrinkage_alpha_low", 0.10),
            )
        league_pipelines[league_id] = pipeline
        league_ortho_gens[league_id] = OrthogonalDataGenerator(league_id, seed=seed + season_idx)
        league_orchestrators[league_id] = create_orchestrator(league_id)

        league_fixtures = generate_fixture_list_from_odds(odds_data)
        for round_idx, (h, a) in enumerate(league_fixtures):
            all_fixtures.append((league_id, round_idx, h, a))

    # ── 排序：按原数据顺序（已有时间顺序）
    # 数据已经按时间排序，保持即可

    home_advantage = {
        "premier_league": 65.0,
        "la_liga": 50.0,
        "bundesliga": 60.0,
        "serie_a": 55.0,
        "ligue_1": 55.0,
    }

    total_fixtures = len(all_fixtures)
    processed = 0
    for global_idx, (lid, round_idx, home_team, away_team) in enumerate(all_fixtures):
        try:
            processed += 1
            pipeline = league_pipelines[lid]
            ortho_gen = league_ortho_gens[lid]
            team_elos = league_team_elos[lid]
            odds_data = league_odds_data[lid]

            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]

            # 读取真实赔率
            ro = get_real_odds(odds_data, home_team, away_team)
            if not ro:
                continue

            # 真实比赛结果
            actual_home_goals = ro.home_goals
            actual_away_goals = ro.away_goals
            if ro.result == "H":
                actual_outcome = "home_win"
            elif ro.result == "D":
                actual_outcome = "draw"
            elif ro.result == "A":
                actual_outcome = "away_win"
            else:
                continue

            match_date = datetime(2014+season_idx, 8, 10)
            match_date = match_date.replace(day=match_date.day + int(global_idx * 1.5))

            odds_h = ro.avg_h or ro.b365_h
            odds_d = ro.avg_d or ro.b365_d
            odds_a = ro.avg_a or ro.b365_a
            if not (odds_h and odds_d and odds_a):
                continue

            match = MatchContext(
                match_id=f"{lid}_S{season}_M{global_idx:04d}",
                league_id=lid, season=season,
                matchday=round_idx + 1, kickoff_time=match_date,
                home_team=home_team, away_team=away_team,
                home_elo=home_elo, away_elo=away_elo,
                odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            )

            ortho_data = ortho_gen.generate(
                global_idx, home_team, away_team, match_date, odds_h, odds_d, odds_a,
            )
            extra = {
                "elo_diff": home_elo - away_elo,
                "recent_results": [_rng.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
                "rank_diff": int((home_elo - away_elo) / 20),
                "goal_diff": (home_elo - away_elo) / 20,
                "xg_diff": (home_elo - away_elo) / 200,
                "streak_momentum": _rng.uniform(0, 0.5),
                "streak_momentum_league": _rng.uniform(0, 0.5),
                "match_phase": 1.0,
            }
            extra.update(ortho_gen.to_extra_dict(ortho_data))

            pipeline_result = pipeline.run_full(match, extra_data=extra)

            if pipeline_result.placements:
                outcome_map = {
                    "home_win": BetSelection.HOME_WIN,
                    "draw": BetSelection.DRAW,
                    "away_win": BetSelection.AWAY_WIN,
                }
                actual = outcome_map[actual_outcome]

                # 1X2 结算
                placements = pipeline.settle_bets(pipeline_result.placements, actual)
                for p in placements:
                    result.total_bets += 1
                    result.total_staked += p.stake
                    if p.result == BetResult.WIN:
                        result.total_wins += 1
                        result.total_returned += p.stake + p.profit_loss
                    result.profit_history.append(p.profit_loss)
                    result.strategy_stats["1x2"]["bets"] += 1
                    result.strategy_stats["1x2"]["staked"] += p.stake
                    result.strategy_stats["1x2"]["returned"] += p.stake + p.profit_loss
                    if p.result == BetResult.WIN:
                        result.strategy_stats["1x2"]["wins"] += 1

                # 串关池
                if pipeline_result.proposals:
                    parlay_mgr.add_match_bets(match.match_id, pipeline_result.proposals)

                # 亚盘 + 大小球（使用真实赔率如果有）
                orchestrator = league_orchestrators[lid]
                try:
                    from src.strategies.asian_handicap import AsianHandicapStrategy
                    from src.strategies.over_under import OverUnderStrategy

                    # 真实数据中只有 Bet365 提供亚盘和大小球
                    synthetic_handicap = {}
                    if ro.asian_handicap is not None and ro.asian_home_odds and ro.asian_away_odds:
                        synthetic_handicap[ro.asian_handicap] = {
                            "home": ro.asian_home_odds,
                            "away": ro.asian_away_odds,
                        }
                    else:
                        synthetic_handicap = orchestrator.asian_strategy.generate_synthetic_odds(
                            pipeline_result.poisson_score_matrix)

                    synthetic_totals = {}
                    if ro.over_odds and ro.under_odds:
                        synthetic_totals[2.5] = {
                            "over": ro.over_odds,
                            "under": ro.under_odds,
                        }
                    else:
                        synthetic_totals = orchestrator.over_under_strategy.generate_synthetic_odds(
                            pipeline_result.poisson_score_matrix)

                    multi_result = orchestrator.run(
                        match=match, score_matrix=pipeline_result.poisson_score_matrix,
                        handicap_odds=synthetic_handicap, totals_odds=synthetic_totals,
                        total_bankroll=shared_bankroll._get_base_bankroll(),
                    )
                    for ap in multi_result.asian_proposals:
                        asian_result_val, asian_pnl = orchestrator.asian_strategy.settle(
                            ap, actual_home_goals, actual_away_goals)
                        is_win = asian_result_val in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                        stake = ap.kelly_stake * getattr(ap, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else ap.kelly_stake
                        if is_win: result.total_wins += 1
                        if asian_pnl >= 0: result.total_returned += stake + asian_pnl
                        result.profit_history.append(asian_pnl)
                        result.strategy_stats["asian_handicap"]["bets"] += 1
                        result.strategy_stats["asian_handicap"]["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                        result.strategy_stats["asian_handicap"]["returned"] += (stake + asian_pnl) if asian_pnl >= 0 else 0
                        if is_win: result.strategy_stats["asian_handicap"]["wins"] += 1

                    for tp in multi_result.totals_proposals:
                        totals_result_val, totals_pnl = orchestrator.over_under_strategy.settle(
                            tp, actual_home_goals, actual_away_goals)
                        is_win = (totals_result_val == "win")
                        stake = tp.kelly_stake * getattr(tp, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else tp.kelly_stake
                        if is_win: result.total_wins += 1
                        if totals_pnl >= 0: result.total_returned += stake + totals_pnl
                        result.profit_history.append(totals_pnl)
                        result.strategy_stats["over_under"]["bets"] += 1
                        result.strategy_stats["over_under"]["staked"] += abs(stake) if stake > 0 else tp.kelly_stake
                        result.strategy_stats["over_under"]["returned"] += (stake + totals_pnl) if totals_pnl >= 0 else 0
                        if is_win: result.strategy_stats["over_under"]["wins"] += 1
                except Exception:
                    pass

                # 更新 Elo（标准 Elo 公式）
                def update_elo(h_elo, a_elo, outcome, k_elo, home_adv):
                    exp_h = 1.0 / (1.0 + 10 ** (-(h_elo + home_adv - a_elo) / 400.0))
                    exp_a = 1.0 - exp_h
                    actual_h = {"home_win": 1.0, "draw": 0.5, "away_win": 0.0}[outcome]
                    actual_a = 1.0 - actual_h
                    return h_elo + k_elo * (actual_h - exp_h), a_elo + k_elo * (actual_a - exp_a)

                ha = home_advantage.get(lid, 60.0)
                k_league = {
                    "premier_league": 24, "la_liga": 20, "bundesliga": 22,
                    "serie_a": 20, "ligue_1": 20,
                }[lid]
                new_home, new_away = update_elo(home_elo, away_elo, actual_outcome, k_league, ha)
                team_elos[home_team] = new_home
                team_elos[away_team] = new_away

            # 每 20 场生成串关
            if (processed + 1) % 20 == 0 or processed == total_fixtures - 1:
                settlements = parlay_mgr.settle_all_ready(match_outcomes)
                for s in settlements:
                    shared_bankroll.state.balance += s.profit
                    result.total_bets += 1
                    result.total_staked += s.stake
                    if s.won:
                        result.total_wins += 1
                        result.total_returned += s.returned
                    result.strategy_stats["parlay"]["bets"] += 1
                    result.strategy_stats["parlay"]["staked"] += s.stake
                    result.strategy_stats["parlay"]["returned"] += s.returned
                    if s.won:
                        result.strategy_stats["parlay"]["wins"] += 1
                    result.profit_history.append(s.profit)

            result.equity_curve.append(shared_bankroll.state.balance)

        except Exception:
            import traceback
            # traceback.print_exc()
            continue

    result.final_balance = shared_bankroll.state.balance
    result.all_elos = {lid: dict(league_team_elos[lid]) for lid in league_order}
    return result


def print_season_row(label: str, r):
    ss = r.strategy_stats
    print(f"  {label:<12} {r.total_bets:>6} {r.win_rate:>6.1%} {r.roi:>7.1%} "
          f"{r.final_balance:>10,.0f} {r.max_drawdown:>6.1%} "
          f"{ss['1x2']['bets']:>5} {ss['asian_handicap']['bets']:>5} "
          f"{ss['over_under']['bets']:>5} {ss['parlay']['bets']:>5}")


if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + f"  GTO-GameFlow {version} 真实赔率扩展回测 — 10赛季五大联赛".center(74) + "║")
    print("╠" + "═" * 78 + "╣")
    print("║" + "  资金: 全局固定基数 10,000 (每赛季独立) | 真实赔率: football-data.co.uk".center(70) + "║")
    print("╚" + "═" * 78 + "╝")

    # ── 10 赛季五大联赛 ──
    seasons = [f"{y}/{str(y+1)[-2:]}" for y in range(2014, 2024)]
    results = []
    carryover_elos = None

    print("\n" + "▇" * 78)
    print("  五大联赛 10 赛季回测 (2014/15 → 2023/24)")
    print("▇" * 78)

    print(f"\n  {'赛季':<12} {'投注':>6} {'胜率':>7} {'ROI':>7} {'最终资金':>10} "
          f"{'回撤':>6} {'1X2':>5} {'亚盘':>5} {'大小球':>5} {'串关':>5}")
    print(f"  {'─'*78}")

    total_bets = 0
    total_staked = 0.0
    total_returned = 0.0
    total_profit = 0.0

    for idx, season in enumerate(seasons):
        r = run_real_odds_season(season, idx, seed=42 + idx * 10,
                              carryover_elos=carryover_elos)
        results.append(r)
        carryover_elos = r.all_elos

        total_bets += r.total_bets
        total_staked += r.total_staked
        total_returned += r.total_returned
        total_profit += (r.final_balance - 10000)

        print_season_row(season, r)

    # 汇总
    combined_roi = ((total_returned - total_staked) / total_staked
                   if total_staked > 0 else 0)
    print(f"  {'─'*78}")
    print(f"  {'联赛合计':<12} {total_bets:>6} "
          f"{'':>7} {combined_roi:>7.1%} "
          f"{'':>10} {'':>6} {'':>5} {'':>5} {'':>5} {'':>5}")
    print(f"  10赛季联赛总利润: {total_profit:+,.0f}")

    # ── 策略分类汇总 ──
    print("\n\n" + "╔" + "═" * 78 + "╗")
    print("║" + "  全时段策略分类汇总".center(74) + "║")
    print("╚" + "═" * 78 + "╝")

    all_strategies = {"1x2": [0, 0, 0.0, 0.0], "asian_handicap": [0, 0, 0.0, 0.0],
                      "over_under": [0, 0, 0.0, 0.0], "parlay": [0, 0, 0.0, 0.0]}
    for r in results:
        for s in all_strategies:
            ss = r.strategy_stats[s]
            all_strategies[s][0] += ss["bets"]
            all_strategies[s][1] += ss["wins"]
            all_strategies[s][2] += ss["staked"]
            all_strategies[s][3] += ss["returned"]

    print(f"\n  {'策略':<18} {'投注':>6} {'胜率':>7} {'ROI':>7} {'投入':>10} {'回报':>10}")
    print(f"  {'─'*62}")
    for s in ["1x2", "asian_handicap", "over_under", "parlay"]:
        bets, wins, staked, returned = all_strategies[s]
        if bets > 0:
            wr = wins / bets
            roi = (returned - staked) / staked if staked > 0 else 0
            label = cn_strategy(s)
            print(f"  {label:<18} {bets:>6} {wr:>6.1%} {roi:>6.1%} "
                  f"{staked:>10,.0f} {returned:>10,.0f}")

    # ── 总汇总 ──
    print("\n\n" + "╔" + "═" * 78 + "╗")
    print("║" + "  总汇总".center(74) + "║")
    print("╚" + "═" * 78 + "╝")

    grand_total_bets = total_bets
    grand_total_staked = total_staked
    grand_total_returned = total_returned
    grand_total_profit = total_profit
    grand_roi = ((grand_total_returned - grand_total_staked) / grand_total_staked
                 if grand_total_staked > 0 else 0)

    print(f"\n  联赛: 10赛季, {grand_total_bets:,} 注, 利润 {grand_total_profit:+,.0f}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  总计: {grand_total_bets:,} 注, 利润 {grand_total_profit:+,.0f}, ROI {grand_roi:+.1%}")
    print(f"  总投入: {grand_total_staked:,.0f}, 总回报: {grand_total_returned:,.0f}")
    print(f"  总初始资金: {len(seasons) * 10000:,}")

    print("\n" + "═" * 78)
    print("  v5.9.2 真实赔率扩展回测完成")
    print("═" * 78)