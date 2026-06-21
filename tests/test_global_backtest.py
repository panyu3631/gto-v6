"""
GTO-GameFlow v5.9 全局回测 — 5大联赛共享资金池 + 跨联赛串关

- 全局资金池: 10,000 (固定基数)
- 5大联赛按比赛日交叉调度
- 串关: 跨联赛组合
- 每赛季独立资金
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random as _random

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    AsianHandicapResult, AsianHandicapProposal, TotalsProposal,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.strategies.strategy_orchestrator import ParlayBatchManager
from src.strategies import StrategyOrchestrator
from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
from src.utils.i18n import cn_league, cn_strategy

# ── 联赛中文映射 ──
LEAGUE_CN = {
    "premier_league": "英超", "la_liga": "西甲",
    "bundesliga": "德甲", "serie_a": "意甲", "ligue_1": "法甲",
}
def cn(lid): return LEAGUE_CN.get(lid, lid)

# ── 球队数据 ──
from tests.test_backtesting_real import (
    LEAGUE_CONFIG, generate_fixture_list, generate_realistic_odds,
    simulate_match_result, update_elo, _simulate_goals,
    OrthogonalDataGenerator, _load_calibrated_weights, _build_weight_multipliers,
    create_orchestrator,
)


@dataclass
class GlobalBacktestResult:
    season: str
    initial_bankroll: float = 10000.0
    final_balance: float = 10000.0
    total_bets: int = 0
    total_wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0
    profit_history: List[float] = field(default_factory=list)
    parlay_stats: Dict[str, int] = field(default_factory=lambda: {"bets": 0, "wins": 0})
    parlay_profit: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    strategy_stats: Dict[str, Dict] = field(default_factory=dict)
    all_elos: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def roi(self):
        return ((self.total_returned - self.total_staked) / self.total_staked
                if self.total_staked > 0 else 0.0)

    @property
    def win_rate(self):
        return self.total_wins / self.total_bets if self.total_bets > 0 else 0.0

    @property
    def sharpe_ratio(self):
        if len(self.profit_history) < 2: return 0.0
        arr = np.array(self.profit_history)
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        return mu / sigma * math.sqrt(252) if sigma > 1e-10 else 0.0

    @property
    def max_drawdown(self):
        if not self.equity_curve: return 0.0
        peak, max_dd = self.equity_curve[0], 0.0
        for v in self.equity_curve:
            if v > peak: peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
        return max_dd

    @property
    def profit_factor(self):
        gp = sum(p for p in self.profit_history if p > 0)
        gl = abs(sum(p for p in self.profit_history if p < 0))
        return gp / gl if gl > 0 else 999.0


def run_global_backtest(
    season: str = "2023/24",
    initial_bankroll: float = 10000.0,
    seed: int = 42,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
) -> GlobalBacktestResult:
    """
    v5.9: 全局回测 — 5大联赛共享一个资金池 10,000。
    """
    _rng = _random.Random(seed)

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    parlay_mgr = ParlayBatchManager(max_legs=4, kelly_discount=0.35, max_batch_size=20,
        min_single_value=0.01, min_combined_value=0.015)
    match_outcomes = {}

    result = GlobalBacktestResult(
        season=season, initial_bankroll=initial_bankroll,
        strategy_stats={
            "1x2": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "asian_handicap": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "over_under": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "parlay": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
        },
    )
    result.equity_curve.append(initial_bankroll)

    # ── 为每个联赛准备基础设施 ──
    league_pipelines = {}
    league_ortho_gens = {}
    league_orchestrators = {}
    league_team_elos = {}
    league_fixtures = {}
    all_elos = {}

    for league_id, config in LEAGUE_CONFIG.items():
        teams = config["teams_2324"] if season == "2023/24" else config["teams_2425"]

        if carryover_elos and league_id in carryover_elos:
            team_elos = dict(carryover_elos[league_id])
            for name, default_elo in teams:
                if name not in team_elos:
                    team_elos[name] = default_elo
        else:
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
            pipeline.signal_decomposer = SignalDecomposer(elo_suppression=1.0)
            pipeline.prior_shrinkage = PriorShrinkage(
                alpha_high=calib.get("shrinkage_alpha_high", 0.50),
                alpha_low=calib.get("shrinkage_alpha_low", 0.10),
            )
        league_pipelines[league_id] = pipeline
        league_ortho_gens[league_id] = OrthogonalDataGenerator(league_id, seed=seed)
        league_orchestrators[league_id] = create_orchestrator(league_id)

        league_fixtures[league_id] = generate_fixture_list(teams)

    # ── 按比赛日交叉排序 ──
    league_order = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    max_rounds = max(len(league_fixtures[lid]) for lid in league_order)
    all_fixtures = []
    for rnd in range(max_rounds):
        for lid in league_order:
            if lid in league_fixtures and rnd < len(league_fixtures[lid]):
                h, a = league_fixtures[lid][rnd]
                all_fixtures.append((lid, rnd, h, a))

    if season == "2023/24":
        base_date = datetime(2023, 8, 11)
    else:
        base_date = datetime(2024, 8, 16)

    home_advantage = 65.0
    total_fixtures = len(all_fixtures)
    days_between = 3.5 * 380 / total_fixtures * 5

    daily_staked = 0.0
    weekly_staked = 0.0
    current_day = base_date.date()
    current_week = base_date.isocalendar()[1]

    # ── 主力循环 ──
    for global_idx, (lid, round_idx, home_team, away_team) in enumerate(all_fixtures):
        try:
            pipeline = league_pipelines[lid]
            ortho_gen = league_ortho_gens[lid]
            team_elos = league_team_elos[lid]
            config = LEAGUE_CONFIG[lid]

            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]

            odds_h, odds_d, odds_a = generate_realistic_odds(
                home_elo, away_elo, home_advantage, seed=seed + global_idx,
            )

            match_date = base_date + timedelta(days=int(global_idx * days_between))
            match_date += timedelta(hours=_rng.randint(12, 21))

            if match_date.date() != current_day:
                daily_staked = 0.0
                shared_bankroll.reset_daily_loss()
                current_day = match_date.date()
            if match_date.isocalendar()[1] != current_week:
                weekly_staked = 0.0
                shared_bankroll.reset_weekly_loss()
                current_week = match_date.isocalendar()[1]

            match = MatchContext(
                match_id=f"{lid}_S{season.replace('/', '')}_M{global_idx:04d}",
                league_id=lid, season=season,
                matchday=round_idx + 1, kickoff_time=match_date,
                home_team=home_team, away_team=away_team,
                home_elo=home_elo, away_elo=away_elo,
                odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            )

            ortho_data = ortho_gen.generate(
                global_idx, home_team, away_team, match_date,
                odds_h, odds_d, odds_a,
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
                actual_outcome = simulate_match_result(
                    home_elo, away_elo, odds_h, odds_d, odds_a,
                    home_advantage, seed=seed + global_idx,
                )
                actual_home_goals, actual_away_goals = _simulate_goals(
                    actual_outcome, lid, home_elo, away_elo, seed=seed + global_idx,
                )
                match_outcomes[match.match_id] = (actual_outcome,
                    actual_outcome.replace("_win", ""))

                outcome_map = {
                    "home_win": BetSelection.HOME_WIN,
                    "draw": BetSelection.DRAW,
                    "away_win": BetSelection.AWAY_WIN,
                }
                actual = outcome_map[actual_outcome]

                # ── 1X2 结算 ──
                placements = pipeline.settle_bets(pipeline_result.placements, actual)
                for p in placements:
                    daily_staked += p.stake
                    weekly_staked += p.stake
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

                # ── 串关池收集 ──
                if pipeline_result.proposals:
                    parlay_mgr.add_match_bets(match.match_id, pipeline_result.proposals)

                # ── 亚盘 + 大小球 (多策略) ──
                orchestrator = league_orchestrators[lid]
                try:
                    # v5.9.1: 使用合成赔率 (覆盖全部标准线)
                    synthetic_handicap = orchestrator.asian_strategy.generate_synthetic_odds(
                        pipeline_result.poisson_score_matrix,
                    )
                    synthetic_totals = orchestrator.over_under_strategy.generate_synthetic_odds(
                        pipeline_result.poisson_score_matrix,
                    )
                    multi_result = orchestrator.run(
                        match=match,
                        score_matrix=pipeline_result.poisson_score_matrix,
                        handicap_odds=synthetic_handicap,
                        totals_odds=synthetic_totals,
                        total_bankroll=shared_bankroll._get_base_bankroll(),
                    )

                    for ap in multi_result.asian_proposals:
                        asian_result_val, asian_pnl = orchestrator.asian_strategy.settle(
                            ap, actual_home_goals, actual_away_goals,
                        )
                        is_win = asian_result_val in (
                            AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN,
                        )
                        stake = ap.kelly_stake * getattr(ap, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else ap.kelly_stake
                        if is_win:
                            result.total_wins += 1
                        if asian_pnl >= 0:
                            result.total_returned += stake + asian_pnl
                        result.profit_history.append(asian_pnl)
                        result.strategy_stats["asian_handicap"]["bets"] += 1
                        result.strategy_stats["asian_handicap"]["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                        result.strategy_stats["asian_handicap"]["returned"] += (stake + asian_pnl) if asian_pnl >= 0 else 0
                        if is_win:
                            result.strategy_stats["asian_handicap"]["wins"] += 1

                    for tp in multi_result.totals_proposals:
                        totals_result_val, totals_pnl = orchestrator.over_under_strategy.settle(
                            tp, actual_home_goals, actual_away_goals,
                        )
                        is_win = (totals_result_val == "win")
                        stake = tp.kelly_stake * getattr(tp, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else tp.kelly_stake
                        if is_win:
                            result.total_wins += 1
                        if totals_pnl >= 0:
                            result.total_returned += stake + totals_pnl
                        result.profit_history.append(totals_pnl)
                        result.strategy_stats["over_under"]["bets"] += 1
                        result.strategy_stats["over_under"]["staked"] += abs(stake) if stake > 0 else tp.kelly_stake
                        result.strategy_stats["over_under"]["returned"] += (stake + totals_pnl) if totals_pnl >= 0 else 0
                        if is_win:
                            result.strategy_stats["over_under"]["wins"] += 1
                except Exception:
                    pass

                # ── 更新 Elo ──
                new_home, new_away = update_elo(
                    home_elo, away_elo, actual_outcome,
                    config["k_elo"], home_advantage,
                )
                team_elos[home_team] = new_home
                team_elos[away_team] = new_away

            # ── 每 20 场生成串关 ──
            if (global_idx + 1) % 20 == 0 or global_idx == total_fixtures - 1:
                parlay_proposals = parlay_mgr.generate_batch(
                    shared_bankroll._get_base_bankroll(),
                )
                settlements = parlay_mgr.settle_all_ready(match_outcomes)
                for s in settlements:
                    shared_bankroll.state.balance += s.profit
                    result.total_bets += 1
                    result.total_staked += s.stake
                    result.parlay_stats["bets"] += 1
                    result.parlay_profit += s.profit
                    if s.won:
                        result.total_wins += 1
                        result.total_returned += s.returned
                        result.parlay_stats["wins"] += 1
                    result.strategy_stats["parlay"]["bets"] += 1
                    result.strategy_stats["parlay"]["staked"] += s.stake
                    result.strategy_stats["parlay"]["returned"] += s.returned
                    if s.won:
                        result.strategy_stats["parlay"]["wins"] += 1
                    result.profit_history.append(s.profit)

            result.equity_curve.append(shared_bankroll.state.balance)

        except Exception as e:
            if global_idx < 10:
                print(f"  错误 ({home_team} vs {away_team}): {e}")

    result.final_balance = shared_bankroll.state.balance
    result.all_elos = {lid: dict(league_team_elos[lid]) for lid in LEAGUE_CONFIG}
    return result


def print_global_report(result: GlobalBacktestResult):
    print(f"─" * 60)
    print(f"  赛季: {result.season}  |  全局资金池: {result.initial_bankroll:,.0f}")
    print(f"─" * 60)
    print(f"  总投注:    {result.total_bets:>8}")
    print(f"  总胜场:    {result.total_wins:>8}")
    print(f"  胜率:      {result.win_rate:>7.1%}")
    print(f"  总投注额:  {result.total_staked:>8,.0f}")
    print(f"  总回报:    {result.total_returned:>8,.0f}")
    print(f"  ROI:       {result.roi:>7.1%}")
    print(f"  最终资金:  {result.final_balance:>8,.0f}")
    print(f"  最大回撤:  {result.max_drawdown:>7.1%}")
    print(f"  夏普比率:  {result.sharpe_ratio:>7.2f}")
    print(f"  利润因子:  {result.profit_factor:>7.2f}")

    ss = result.strategy_stats
    if any(ss[s]["bets"] > 0 for s in ss):
        print("  ── 策略分类 ──")
        print(f"  {'策略':<18} {'投注':>5} {'胜率':>7} {'ROI':>7} {'投入':>10} {'回报':>10}")
        for s in ["1x2", "asian_handicap", "over_under", "parlay"]:
            st = ss[s]
            if st["bets"] > 0:
                wr = st["wins"] / st["bets"]
                roi_s = ((st["returned"] - st["staked"]) / st["staked"]
                         if st["staked"] > 0 else 0)
                label = cn_strategy(s)
                print(f"  {label:<18} {st['bets']:>5} {wr:>6.1%} {roi_s:>6.1%} "
                      f"{st['staked']:>10,.0f} {st['returned']:>10,.0f}")

    if result.parlay_stats["bets"] > 0:
        ps = result.parlay_stats
        pw = ps["wins"] / ps["bets"] if ps["bets"] > 0 else 0
        print(f"  ── 串关详情 ──")
        print(f"  串关投注: {ps['bets']:>5}  胜率: {pw:>6.1%}  "
              f"利润: {result.parlay_profit:>+,.0f}")
    print(f"─" * 60)


# ================================================================
# 主流程
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + "  GTO-GameFlow v5.9 全局双赛季回测".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + "  资金: 全局固定基数 10,000 (5大联赛共享)".center(64) + "║")
    print("║" + "  赛程: 5大联赛按比赛日交叉调度".center(58) + "║")
    print("║" + "  串关: 跨联赛组合".center(42) + "║")
    print("╚" + "═" * 68 + "╝")

    # ── 第一季 ──
    print("\n" + "▇" * 70)
    print(" 第一季: 2023/24 (全局资金池 10,000)")
    print("▇" * 70)

    r1 = run_global_backtest(season="2023/24", seed=42)
    print_global_report(r1)

    # ── 第二季 (Elo 延续，资金独立) ──
    print("\n\n" + "▇" * 70)
    print(" 第二季: 2024/25 (Elo 延续，资金独立，全局资金池 10,000)")
    print("▇" * 70)

    r2 = run_global_backtest(
        season="2024/25", seed=142, carryover_elos=r1.all_elos,
    )
    print_global_report(r2)

    # ── 双赛季汇总 ──
    print("\n\n╔" + "═" * 70 + "╗")
    print("║" + "  双赛季总汇总 (全局资金池，每赛季独立)".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    print(f"\n  每赛季初始资金: 10,000 (5大联赛共享)")
    print(f"  23/24: {r1.final_balance:,.0f}  ROI: {r1.roi:+.1%}  胜率: {r1.win_rate:.1%}  "
          f"回撤: {r1.max_drawdown:.1%}")
    print(f"  24/25: {r2.final_balance:,.0f}  ROI: {r2.roi:+.1%}  胜率: {r2.win_rate:.1%}  "
          f"回撤: {r2.max_drawdown:.1%}")

    total_bets = r1.total_bets + r2.total_bets
    total_wins = r1.total_wins + r2.total_wins
    total_staked = r1.total_staked + r2.total_staked
    total_returned = r1.total_returned + r2.total_returned
    combined_roi = ((total_returned - total_staked) / total_staked) if total_staked > 0 else 0
    combined_wr = total_wins / total_bets if total_bets > 0 else 0

    print(f"\n  {'策略':<18} {'投注':>6} {'胜率':>7} {'ROI':>7} {'投入':>10} {'回报':>10}")
    print(f"  {'─'*62}")
    for s in ["1x2", "asian_handicap", "over_under", "parlay"]:
        s1 = r1.strategy_stats[s]
        s2 = r2.strategy_stats[s]
        bets = s1["bets"] + s2["bets"]
        wins = s1["wins"] + s2["wins"]
        staked = s1["staked"] + s2["staked"]
        returned = s1["returned"] + s2["returned"]
        if bets > 0:
            wr = wins / bets
            roi = (returned - staked) / staked if staked > 0 else 0
            label = {"1x2": "1X2", "asian_handicap": "亚盘",
                     "over_under": "大小球", "parlay": "串关"}[s]
            print(f"  {label:<18} {bets:>6} {wr:>6.1%} {roi:>6.1%} "
                  f"{staked:>10,.0f} {returned:>10,.0f}")

    # 串关详情
    p1 = r1.parlay_stats
    p2 = r2.parlay_stats
    total_parlay_bets = p1["bets"] + p2["bets"]
    total_parlay_wins = p1["wins"] + p2["wins"]
    total_parlay_profit = r1.parlay_profit + r2.parlay_profit
    if total_parlay_bets > 0:
        print(f"\n  串关总计: {total_parlay_bets} 注, "
              f"胜率 {total_parlay_wins/total_parlay_bets:.1%}, "
              f"利润 {total_parlay_profit:+,.0f}")

    print(f"  {'─'*62}")
    print(f"  {'全策略合计':<18} {total_bets:>6} {combined_wr:>6.1%} {combined_roi:>6.1%} "
          f"{total_staked:>10,.0f} {total_returned:>10,.0f}")

    two_season_profit = (r1.final_balance - 10000) + (r2.final_balance - 10000)
    print(f"\n  两赛季合计利润: {two_season_profit:+,.0f}")
    print("\n" + "═" * 70)
    print("  v5.9 全局回测完成")
    print("═" * 70)