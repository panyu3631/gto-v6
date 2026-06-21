"""
GTO-GameFlow v5.0 回测框架

功能:
- 历史比赛回测
- 胜率 / ROI / 夏普比率 / 最大回撤
- 按联赛/赛季/因子分组统计
- 资金曲线
"""
import sys
import os
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    BankrollState, ProbabilityDistribution,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.config.settings import config as global_config
from src.utils.i18n import cn_league


@dataclass
class BacktestMatch:
    """回测比赛记录"""
    match_id: str
    league_id: str
    season: str
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    odds_home: float
    odds_draw: float
    odds_away: float
    actual_outcome: str  # "home", "draw", "away"
    extra_data: Dict = field(default_factory=dict)
    kickoff_time: str = ""  # ISO format


@dataclass
class BacktestResult:
    """回测结果"""
    total_matches: int = 0
    total_bets: int = 0
    total_wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0
    final_balance: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    # 按联赛
    by_league: Dict[str, dict] = field(default_factory=dict)
    # 按赛季
    by_season: Dict[str, dict] = field(default_factory=dict)
    # 资金曲线
    equity_curve: List[float] = field(default_factory=list)
    # 每笔投注详情
    bet_history: List[dict] = field(default_factory=list)


class Backtester:
    """
    回测引擎。

    使用方式:
        backtester = Backtester("premier_league", initial_bankroll=10000)
        matches = load_matches("matches.json")
        result = backtester.run(matches)
        print(result.summary())
    """

    def __init__(self, league_id: str, initial_bankroll: float = 10000):
        self.league_id = league_id
        self.pipeline = GameFlowPipeline(league_id, initial_bankroll=initial_bankroll)
        self.initial_bankroll = initial_bankroll

    def run(self, matches: List[BacktestMatch]) -> BacktestResult:
        """运行回测"""
        result = BacktestResult()
        result.total_matches = len(matches)
        result.equity_curve.append(self.initial_bankroll)

        # 按联赛/赛季分组统计
        daily_staked = 0.0
        weekly_staked = 0.0
        current_date = None
        current_week = None

        profit_history = []
        balance = self.initial_bankroll

        for i, match in enumerate(matches):
            try:
                # 构建 MatchContext
                kickoff = (datetime.fromisoformat(match.kickoff_time)
                          if match.kickoff_time else datetime.now() + timedelta(days=i))

                ctx = MatchContext(
                    match_id=match.match_id,
                    league_id=match.league_id,
                    season=match.season,
                    matchday=(i % 38) + 1,
                    kickoff_time=kickoff,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    home_elo=match.home_elo,
                    away_elo=match.away_elo,
                    odds_home=match.odds_home,
                    odds_draw=match.odds_draw,
                    odds_away=match.odds_away,
                )

                # 重置日/周累计 (简化: 按比赛数)
                date_key = kickoff.date()
                week_key = kickoff.isocalendar()[1]

                if date_key != current_date:
                    daily_staked = 0.0
                    current_date = date_key
                if week_key != current_week:
                    weekly_staked = 0.0
                    current_week = week_key

                # 运行流水线
                pipeline_result = self.pipeline.run_full(
                    ctx, extra_data=match.extra_data,
                    daily_staked=daily_staked, weekly_staked=weekly_staked,
                )

                # 执行投注
                pipeline_result = self.pipeline.execute_bets(pipeline_result)

                if pipeline_result.placements:
                    # 结算
                    outcome_map = {
                        "home": BetSelection.HOME_WIN,
                        "draw": BetSelection.DRAW,
                        "away": BetSelection.AWAY_WIN,
                    }
                    actual = outcome_map.get(match.actual_outcome, BetSelection.DRAW)
                    placements = self.pipeline.settle_bets(
                        pipeline_result.placements, actual
                    )

                    for p in placements:
                        daily_staked += p.stake
                        weekly_staked += p.stake

                        result.total_bets += 1
                        result.total_staked += p.stake

                        if p.result == BetResult.WIN:
                            result.total_wins += 1
                            result.total_returned += p.stake + p.profit_loss
                            profit_history.append(p.profit_loss)
                        elif p.result == BetResult.LOSS:
                            result.total_returned += 0
                            profit_history.append(p.profit_loss)

                        result.bet_history.append({
                            "match_id": match.match_id,
                            "selection": p.selection.value,
                            "odds": p.odds,
                            "stake": p.stake,
                            "result": p.result.value,
                            "profit_loss": p.profit_loss,
                            "league": match.league_id,
                            "season": match.season,
                        })

                # 更新资金曲线
                balance = self.pipeline.bankroll_mgr.state.balance
                result.equity_curve.append(balance)

                # 按联赛统计
                lid = match.league_id
                if lid not in result.by_league:
                    result.by_league[lid] = {"bets": 0, "wins": 0, "staked": 0, "returned": 0}
                if pipeline_result.placements:
                    for p in pipeline_result.placements:
                        result.by_league[lid]["bets"] += 1
                        result.by_league[lid]["staked"] += p.stake
                        if p.result == BetResult.WIN:
                            result.by_league[lid]["wins"] += 1
                            result.by_league[lid]["returned"] += p.stake + p.profit_loss
                        elif p.result == BetResult.LOSS:
                            result.by_league[lid]["returned"] += 0

                # 按赛季统计
                sid = match.season
                if sid not in result.by_season:
                    result.by_season[sid] = {"bets": 0, "wins": 0, "staked": 0, "returned": 0}
                if pipeline_result.placements:
                    for p in pipeline_result.placements:
                        result.by_season[sid]["bets"] += 1
                        result.by_season[sid]["staked"] += p.stake
                        if p.result == BetResult.WIN:
                            result.by_season[sid]["wins"] += 1
                            result.by_season[sid]["returned"] += p.stake + p.profit_loss
                        elif p.result == BetResult.LOSS:
                            result.by_season[sid]["returned"] += 0

            except Exception as e:
                print(f"回测错误 {match.match_id}: {e}")
                continue

        # 计算结果指标
        result.final_balance = balance
        if result.total_staked > 0:
            result.roi = (result.total_returned - result.total_staked) / result.total_staked
        if result.total_bets > 0:
            result.win_rate = result.total_wins / result.total_bets

        # 最大回撤
        peak = self.initial_bankroll
        for eq in result.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            result.max_drawdown = max(result.max_drawdown, dd)

        # 夏普比率 (简化)
        if len(profit_history) > 1:
            mean_return = sum(profit_history) / len(profit_history)
            variance = sum((r - mean_return) ** 2 for r in profit_history) / len(profit_history)
            if variance > 0:
                result.sharpe_ratio = mean_return / math.sqrt(variance)

        # 利润因子
        total_profit = sum(r for r in profit_history if r > 0)
        total_loss = abs(sum(r for r in profit_history if r < 0))
        if total_loss > 0:
            result.profit_factor = total_profit / total_loss

        return result

    def summary(self, result: BacktestResult) -> str:
        """生成回测摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append("GTO-GameFlow v5.0 回测报告")
        lines.append("=" * 60)
        lines.append(f"总比赛: {result.total_matches}")
        lines.append(f"总投注: {result.total_bets}")
        lines.append(f"总胜场: {result.total_wins}")
        lines.append(f"总投注金额: {result.total_staked:.2f}")
        lines.append(f"总回报: {result.total_returned:.2f}")
        lines.append(f"最终资金: {result.final_balance:.2f}")
        lines.append(f"ROI: {result.roi:.2%}")
        lines.append(f"胜率: {result.win_rate:.2%}")
        lines.append(f"最大回撤: {result.max_drawdown:.2%}")
        lines.append(f"夏普比率: {result.sharpe_ratio:.2f}")
        lines.append(f"利润因子: {result.profit_factor:.2f}")

        if result.by_league:
            lines.append("-" * 40)
            lines.append("按联赛:")
            for lid, stats in result.by_league.items():
                if stats["bets"] > 0:
                    l_roi = (stats["returned"] - stats["staked"]) / stats["staked"]
                    l_wr = stats["wins"] / stats["bets"]
                    lines.append(f"  {cn_league(lid)}: {stats['bets']}注, 胜率={l_wr:.1%}, ROI={l_roi:.1%}")

        if result.by_season:
            lines.append("-" * 40)
            lines.append("按赛季:")
            for sid, stats in result.by_season.items():
                if stats["bets"] > 0:
                    s_roi = (stats["returned"] - stats["staked"]) / stats["staked"]
                    s_wr = stats["wins"] / stats["bets"]
                    lines.append(f"  {sid}: {stats['bets']}注, 胜率={s_wr:.1%}, ROI={s_roi:.1%}")

        lines.append("=" * 60)
        return "\n".join(lines)


def generate_sample_matches(league_id: str, n_matches: int = 100) -> List[BacktestMatch]:
    """生成样本回测数据"""
    import random
    random.seed(42)

    matches = []
    base_date = datetime(2025, 8, 1)
    for i in range(n_matches):
        elo_home = random.gauss(1500, 200)
        elo_away = random.gauss(1500, 200)
        odds_home = round(1.0 / (1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400))), 2)
        odds_home = max(1.05, min(10.0, odds_home))
        odds_draw = round(3.0 + random.random() * 2.0, 2)
        odds_away = round(1.0 / (1.0 - 1.0 / odds_home - 1.0 / odds_draw), 2)
        odds_away = max(1.05, min(10.0, odds_away))

        # 实际结果 (按概率采样)
        prob_home = 1.0 / odds_home
        prob_away = 1.0 / odds_away
        prob_draw = 1.0 - prob_home - prob_away
        r = random.random()
        if r < prob_home:
            outcome = "home"
        elif r < prob_home + prob_draw:
            outcome = "draw"
        else:
            outcome = "away"

        matches.append(BacktestMatch(
            match_id=f"{league_id}_{i:04d}",
            league_id=league_id,
            season=f"2025/26",
            home_team=f"Team_{i*2}",
            away_team=f"Team_{i*2+1}",
            home_elo=elo_home,
            away_elo=elo_away,
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            actual_outcome=outcome,
            kickoff_time=(base_date + timedelta(days=i)).isoformat(),
            extra_data={
                "elo_diff": elo_home - elo_away,
                "recent_results": [random.choice([3, 1, 0]) for _ in range(5)],
                "rank_diff": int((elo_home - elo_away) / 20),
            },
        ))
    return matches


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    print("GTO-GameFlow v5.0 回测框架")
    print("=" * 60)

    # 生成样本数据运行回测
    for league in ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]:
        print(f"\n回测 {league}...")
        backtester = Backtester(league, initial_bankroll=10000)
        matches = generate_sample_matches(league, n_matches=20)
        result = backtester.run(matches)
        print(backtester.summary(result))