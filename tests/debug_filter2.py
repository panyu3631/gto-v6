"""
快速调试2: 追踪第11-20场比赛为什么不生成提案
"""
import sys
import os
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import MatchContext, BetSelection, BetResult
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import generate_bet_proposals, compute_confidence
from src.config.league_params import get_league_params

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

# 先跑前11场 (与之前一样)
for i in range(11):
    home_team, away_team = fixtures[i]
    home_elo = team_elos[home_team]
    away_elo = team_elos[away_team]

    odds_h, odds_d, odds_a = generate_realistic_odds(home_elo, away_elo, seed=42 + i)

    match_date = base_date + timedelta(days=int(i * days_between))
    match = MatchContext(
        match_id=f"debug_{i:04d}",
        league_id=league_id, season="2023/24",
        matchday=(i % 38) + 1, kickoff_time=match_date,
        home_team=home_team, away_team=away_team,
        home_elo=home_elo, away_elo=away_elo,
        odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
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

    if pipeline_result.placements:
        actual_outcome = simulate_match_result(home_elo, away_elo, odds_h, odds_d, odds_a, seed=42 + i)
        outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW, "away_win": BetSelection.AWAY_WIN}
        actual = outcome_map[actual_outcome]
        pipeline.settle_bets(pipeline_result.placements, actual)

        from tests.test_backtesting_real import update_elo
        new_home, new_away = update_elo(home_elo, away_elo, actual_outcome, config["k_elo"])
        team_elos[home_team] = new_home
        team_elos[away_team] = new_away

print(f"前11场结算完成, balance={pipeline.bankroll_mgr.state.balance:.0f}")
print(f"consecutive_losses={pipeline.bankroll_mgr.state.consecutive_losses}")
print(f"breaker_active={pipeline.risk_ctrl.breaker_state.is_active}")
print()

# 诊断第12-20场: 手动计算 value 和 confidence
print("=" * 80)
print("第12-20场详细诊断")
print("=" * 80)

lp = get_league_params(league_id)
conf_weights = (lp.confidence_w_data, lp.confidence_w_factor, lp.confidence_w_dispersion, lp.confidence_w_phase)

for i in range(11, 20):
    home_team, away_team = fixtures[i]
    home_elo = team_elos[home_team]
    away_elo = team_elos[away_team]
    elo_diff = home_elo - away_elo

    odds_h, odds_d, odds_a = generate_realistic_odds(home_elo, away_elo, seed=42 + i)

    impl_h = 1.0 / odds_h
    impl_d = 1.0 / odds_d
    impl_a = 1.0 / odds_a
    total = impl_h + impl_d + impl_a
    impl_h /= total
    impl_d /= total
    impl_a /= total

    match_date = base_date + timedelta(days=int(i * days_between))
    match = MatchContext(
        match_id=f"debug_{i:04d}",
        league_id=league_id, season="2023/24",
        matchday=(i % 38) + 1, kickoff_time=match_date,
        home_team=home_team, away_team=away_team,
        home_elo=home_elo, away_elo=away_elo,
        odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
    )

    extra = {
        "elo_diff": elo_diff,
        "recent_results": [random.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
        "rank_diff": int(elo_diff / 20),
        "goal_diff": elo_diff / 20,
        "xg_diff": elo_diff / 200,
        "streak_momentum": random.uniform(0, 0.5),
        "streak_momentum_league": random.uniform(0, 0.5),
        "data_source_count": 5,
        "match_phase": 1.0,
    }

    pipeline_result = pipeline.run_full(match, extra_data=extra)

    if pipeline_result.fused_probs:
        mp = pipeline_result.fused_probs
        vh = mp.prob_home - impl_h
        vd = mp.prob_draw - impl_d
        va = mp.prob_away - impl_a

        # 手动计算 confidence
        active_factor_count = len(pipeline_result.factor_deltas)
        confidence = compute_confidence(
            data_completeness=0.8,
            factor_activation_rate=active_factor_count / 41.0,
            dispersion_penalty=0.9,
            match_phase=1.0,
            weights=conf_weights,
        )

        n_proposals = len(pipeline_result.proposals)
        n_placements = len(pipeline_result.placements)

        print(f"\n[{i:3d}] {home_team[:14]:14s}({home_elo:.0f}) vs {away_team[:14]:14s}({away_elo:.0f}) "
              f"| diff={elo_diff:+4.0f}")
        print(f"      odds: {odds_h:.2f}/{odds_d:.2f}/{odds_a:.2f}")
        print(f"      model: {mp.prob_home:.4f}/{mp.prob_draw:.4f}/{mp.prob_away:.4f}")
        print(f"      value: {vh:+.4f}/{vd:+.4f}/{va:+.4f}")
        print(f"      factors={active_factor_count} conf={confidence:.4f} "
              f"proposals={n_proposals} placements={n_placements}")
        if pipeline_result.circuit_broken:
            print(f"      *** CIRCUIT BREAKER: {pipeline_result.circuit_reason}")
        if pipeline_result.warnings:
            print(f"      warnings: {pipeline_result.warnings}")
        if pipeline_result.errors:
            print(f"      errors: {pipeline_result.errors}")
    else:
        print(f"\n[{i:3d}] {home_team} vs {away_team} | NO fused_probs! errors={pipeline_result.errors}")

    # 如果有投注，结算
    if pipeline_result.placements:
        actual_outcome = simulate_match_result(home_elo, away_elo, odds_h, odds_d, odds_a, seed=42 + i)
        outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW, "away_win": BetSelection.AWAY_WIN}
        actual = outcome_map[actual_outcome]
        pipeline.settle_bets(pipeline_result.placements, actual)

        from tests.test_backtesting_real import update_elo
        new_home, new_away = update_elo(home_elo, away_elo, actual_outcome, config["k_elo"])
        team_elos[home_team] = new_home
        team_elos[away_team] = new_away