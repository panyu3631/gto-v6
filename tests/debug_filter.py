"""
快速调试: 追踪第一个联赛前30场比赛的过滤情况
"""
import sys
import os
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import MatchContext, BetSelection, BetResult
from src.pipeline.orchestrator import GameFlowPipeline

from tests.test_backtesting_real import (
    LEAGUE_CONFIG, generate_fixture_list,
    generate_realistic_odds, simulate_match_result,
)

league_id = "premier_league"
config = LEAGUE_CONFIG[league_id]
teams = config["teams_2324"]
team_elos = {t[0]: t[1] for t in teams}
fixtures = generate_fixture_list(teams)
random.seed(42)

pipeline = GameFlowPipeline(league_id, initial_bankroll=10000.0)
base_date = datetime(2023, 8, 11)
days_between = 3.5 * 380 / len(fixtures)

total_bets = 0
total_wins = 0
no_proposals = 0
circuit_breaks = 0
risk_blocks = 0
stake_zero = 0

for i in range(min(30, len(fixtures))):
    home_team, away_team = fixtures[i]
    home_elo = team_elos[home_team]
    away_elo = team_elos[away_team]

    odds_h, odds_d, odds_a = generate_realistic_odds(home_elo, away_elo, seed=42 + i)

    match_date = base_date + timedelta(days=int(i * days_between))
    match_date += timedelta(hours=random.randint(12, 21))

    match = MatchContext(
        match_id=f"debug_{i:04d}",
        league_id=league_id,
        season="2023/24",
        matchday=(i % 38) + 1,
        kickoff_time=match_date,
        home_team=home_team,
        away_team=away_team,
        home_elo=home_elo,
        away_elo=away_elo,
        odds_home=odds_h,
        odds_draw=odds_d,
        odds_away=odds_a,
    )

    extra = {
        "elo_diff": home_elo - away_elo,
        "recent_results": [random.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
        "rank_diff": int((home_elo - away_elo) / 20),
        "goal_diff": (home_elo - away_elo) / 20,
        "xg_diff": (home_elo - away_elo) / 200,
        "streak_momentum": random.uniform(0, 0.5),
        "streak_momentum_league": random.uniform(0, 0.5),
        "data_source_count": 5,
        "match_phase": 1.0,
    }

    pipeline_result = pipeline.run_full(match, extra_data=extra)

    n_proposals = len(pipeline_result.proposals)
    n_placements = len(pipeline_result.placements)

    breaker_state = pipeline.risk_ctrl.breaker_state
    brkr = f"BREAKER:{breaker_state.trigger_reason}" if breaker_state.is_active else ""

    if n_placements == 0:
        if n_proposals == 0:
            no_proposals += 1
        elif pipeline_result.circuit_broken:
            circuit_breaks += 1
        else:
            risk_blocks += 1
            # Check which proposals were filtered
            stake_zero += sum(1 for p in pipeline_result.proposals if p.adjusted_stake <= 0)

        if i < 10:
            status = f"提案={n_proposals} 投注={n_placements} {brkr}"
            if pipeline_result.circuit_broken:
                status += f" [{pipeline_result.circuit_reason}]"
            if pipeline_result.warnings:
                status += f" WARN:{pipeline_result.warnings}"
            print(f"  [{i:3d}] {home_team[:12]:12s} vs {away_team[:12]:12s} | {status}")

    if n_placements > 0:
        actual_outcome = simulate_match_result(
            home_elo, away_elo, odds_h, odds_d, odds_a, seed=42 + i,
        )
        outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW, "away_win": BetSelection.AWAY_WIN}
        actual = outcome_map[actual_outcome]
        placements = pipeline.settle_bets(pipeline_result.placements, actual)

        total_bets += len(placements)
        for p in placements:
            if p.result == BetResult.WIN:
                total_wins += 1

        from tests.test_backtesting_real import update_elo
        new_home, new_away = update_elo(home_elo, away_elo, actual_outcome, config["k_elo"])
        team_elos[home_team] = new_home
        team_elos[away_team] = new_away

        bal = pipeline.bankroll_mgr.state.balance
        print(f"  [{i:3d}] {home_team[:12]:12s} vs {away_team[:12]:12s} | "
              f"BET: {p.selection.value} odds={p.odds:.2f} stake={p.stake:.0f} "
              f"result={p.result.value} balance={bal:.0f}")

print(f"\n总结 (前30场):")
print(f"  总投注: {total_bets}")
print(f"  无提案: {no_proposals}")
print(f"  熔断:   {circuit_breaks}")
print(f"  风控拦截: {risk_blocks}")
print(f"  stake=0: {stake_zero}")
print(f"  最终资金: {pipeline.bankroll_mgr.state.balance:.0f}")
print(f"  连续亏损: {pipeline.bankroll_mgr.state.consecutive_losses}")
print(f"  月亏损:   {pipeline.bankroll_mgr.state.monthly_loss:.0f}")
print(f"  周亏损:   {pipeline.bankroll_mgr.state.weekly_loss:.0f}")
print(f"  日亏损:   {pipeline.bankroll_mgr.state.daily_loss:.0f}")