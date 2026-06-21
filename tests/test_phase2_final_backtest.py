"""
GTO-GameFlow v5.9.2 — Phase 2 最终验证: 10赛季完整回测
- 动态 Kelly 分数
- 联赛特定优化参数
- 风控层
- 真实赔率数据
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random, json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.data.models import (
    MatchContext, BetSelection, BetResult, AsianHandicapResult,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.strategies.strategy_orchestrator import ParlayBatchManager, create_orchestrator
from src.data.historical_odds_loader import load_odds_for_season
from src.data.orthogonal_sources import OrthogonalDataGenerator
from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
from tests.test_backtesting_real import _load_calibrated_weights, _build_weight_multipliers
from tests.test_phase2_optimization import (
    DynamicKellyEngine, RiskControlLayer, LeagueOptimizedParams,
)
from src.config.settings import GlobalConfig
version = GlobalConfig.version


def run_phase2_season(
    season: str,
    initial_bankroll: float = 10000.0,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """Phase 2 优化版单赛季回测"""
    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    opt_params = LeagueOptimizedParams.optimized_defaults()

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    parlay_mgr = ParlayBatchManager(max_legs=4, kelly_discount=0.35, max_batch_size=20,
        min_single_value=0.01, min_combined_value=0.015)
    kelly_engine = DynamicKellyEngine(base_discount=0.25)
    risk_control = RiskControlLayer()

    if carryover_elos is None:
        carryover_elos = {}

    result = {
        "season": season,
        "total_bets": 0, "total_wins": 0, "total_staked": 0.0, "total_returned": 0.0,
        "risk_events": 0,
        "by_league": {},
        "by_strategy": {"1x2": [0, 0, 0.0, 0.0], "asian": [0, 0, 0.0, 0.0],
                        "over_under": [0, 0, 0.0, 0.0], "parlay": [0, 0, 0.0, 0.0]},
    }

    for lid in leagues:
        odds_data = load_odds_for_season(lid, season)
        if not odds_data:
            continue

        params = opt_params[lid]
        league_result = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}

        # Elo
        if lid in carryover_elos:
            team_elos = dict(carryover_elos[lid])
        else:
            team_elos = {}
        for m in odds_data.values():
            if m.home_team not in team_elos:
                team_elos[m.home_team] = 1650.0
            if m.away_team not in team_elos:
                team_elos[m.away_team] = 1650.0

        calib = _load_calibrated_weights().get(lid, {})
        weight_multipliers = _build_weight_multipliers(lid, calib)

        pipeline = GameFlowPipeline(lid, initial_bankroll=initial_bankroll,
                                     weight_multipliers=weight_multipliers)
        pipeline.set_bankroll_manager(shared_bankroll)
        pipeline.prior_shrinkage = PriorShrinkage(
            alpha_high=params.shrinkage_alpha_high,
            alpha_low=params.shrinkage_alpha_low,
        )

        ortho_gen = OrthogonalDataGenerator(lid, seed=hash(season + lid) % 10000)
        orchestrator = create_orchestrator(lid)
        match_outcomes = {}

        ha = {"premier_league": 65, "la_liga": 50, "bundesliga": 60, "serie_a": 55, "ligue_1": 55}.get(lid, 60)
        k_elo = params.elo_k

        for round_idx, match in enumerate(odds_data.values()):
            try:
                home_elo = team_elos[match.home_team]
                away_elo = team_elos[match.away_team]

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

                from datetime import datetime
                match_ctx = MatchContext(
                    match_id=f"{lid}_S{season}_M{round_idx:04d}",
                    league_id=lid, season=season, matchday=round_idx+1,
                    kickoff_time=datetime(2000, 1, 1),
                    home_team=match.home_team, away_team=match.away_team,
                    home_elo=home_elo, away_elo=away_elo,
                    odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
                )

                ortho_data = ortho_gen.generate(round_idx, match.home_team, match.away_team,
                    datetime(2000, 1, 1), odds_h, odds_d, odds_a)
                extra = {
                    "elo_diff": home_elo - away_elo,
                    "recent_results": [random.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
                    "rank_diff": int((home_elo - away_elo) / 20),
                    "goal_diff": (home_elo - away_elo) / 20,
                    "xg_diff": (home_elo - away_elo) / 200,
                    "streak_momentum": random.uniform(0, 0.5),
                    "streak_momentum_league": random.uniform(0, 0.5),
                    "match_phase": 1.0,
                }
                extra.update(ortho_gen.to_extra_dict(ortho_data))

                pipeline_result = pipeline.run_full(match_ctx, extra_data=extra)

                if pipeline_result.placements:
                    outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW,
                                   "away_win": BetSelection.AWAY_WIN}
                    actual = outcome_map[actual_outcome]

                    placements = pipeline.settle_bets(pipeline_result.placements, actual)
                    for p in placements:
                        # 动态 Kelly
                        if pipeline_result.fused_probs:
                            imp = 1/odds_h + 1/odds_d + 1/odds_a
                            sel_map = {
                                BetSelection.HOME_WIN: "home",
                                BetSelection.DRAW: "draw",
                                BetSelection.AWAY_WIN: "away",
                            }
                            key = sel_map.get(p.selection, "home")
                            market_prob = {"home": (1/odds_h)/imp, "draw": (1/odds_d)/imp,
                                           "away": (1/odds_a)/imp}[key]
                            model_prob = {
                                "home": pipeline_result.fused_probs.prob_home,
                                "draw": pipeline_result.fused_probs.prob_draw,
                                "away": pipeline_result.fused_probs.prob_away,
                            }[key]
                            discount = kelly_engine.compute_discount(
                                value_signal=abs(model_prob - market_prob),
                                odds_std=extra.get("odds_std", 0.05),
                                market_efficiency=extra.get("market_efficiency", 0.05),
                                model_prob=model_prob,
                                market_prob=market_prob,
                            )
                            p.stake *= (discount / 0.25)

                        # 风控
                        approved, adj_stake, reason = risk_control.check_bet(
                            p.stake, p.odds,
                            shared_bankroll._get_base_bankroll(),
                            result["total_staked"],
                            shared_bankroll.state.balance,
                        )
                        if not approved:
                            result["risk_events"] += 1
                            continue
                        p.stake = adj_stake

                        result["total_bets"] += 1
                        result["total_staked"] += p.stake
                        league_result["bets"] += 1
                        league_result["staked"] += p.stake

                        if p.result == BetResult.WIN:
                            result["total_wins"] += 1
                            result["total_returned"] += p.stake + p.profit_loss
                            league_result["wins"] += 1
                            league_result["returned"] += p.stake + p.profit_loss

                        result["by_strategy"]["1x2"][0] += 1
                        result["by_strategy"]["1x2"][2] += p.stake
                        if p.result == BetResult.WIN:
                            result["by_strategy"]["1x2"][1] += 1
                            result["by_strategy"]["1x2"][3] += p.stake + p.profit_loss

                        kelly_engine.record_result(p.profit_loss)

                    if pipeline_result.proposals:
                        parlay_mgr.add_match_bets(match_ctx.match_id, pipeline_result.proposals)
                    match_outcomes[match_ctx.match_id] = actual

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

                        for ap in multi.asian_proposals:
                            ar, pnl = orchestrator.asian_strategy.settle(ap, match.home_goals, match.away_goals)
                            is_win = ar in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                            stake = ap.kelly_stake * getattr(ap, 'strategy_weight', 1.0)
                            result["total_bets"] += 1
                            result["total_staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                            league_result["bets"] += 1
                            league_result["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                            if is_win:
                                result["total_wins"] += 1
                                result["total_returned"] += stake + pnl
                                league_result["wins"] += 1
                                league_result["returned"] += stake + pnl
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
                            league_result["bets"] += 1
                            league_result["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                            if is_win:
                                result["total_wins"] += 1
                                result["total_returned"] += stake + pnl
                                league_result["wins"] += 1
                                league_result["returned"] += stake + pnl
                            result["by_strategy"]["over_under"][0] += 1
                            result["by_strategy"]["over_under"][2] += abs(stake) if stake > 0 else tp.kelly_stake
                            if is_win:
                                result["by_strategy"]["over_under"][1] += 1
                                result["by_strategy"]["over_under"][3] += stake + pnl
                    except Exception:
                        pass

                    # 更新 Elo
                    if match.result == "H":
                        actual_h, actual_a = 1.0, 0.0
                    elif match.result == "A":
                        actual_h, actual_a = 0.0, 1.0
                    else:
                        actual_h, actual_a = 0.5, 0.5
                    exp_h = 1.0 / (1.0 + 10 ** (-(home_elo + ha - away_elo) / 400.0))
                    exp_a = 1.0 - exp_h
                    team_elos[match.home_team] = home_elo + k_elo * (actual_h - exp_h)
                    team_elos[match.away_team] = away_elo + k_elo * (actual_a - exp_a)

            except Exception:
                continue

        carryover_elos[lid] = team_elos
        result["by_league"][lid] = league_result

    return result, carryover_elos


if __name__ == "__main__":
    print("=" * 78)
    print(f"  GTO-GameFlow {version} Phase 2: 10赛季完整回测 (动态 Kelly + 风控)")
    print("=" * 78)

    seasons = [f"{y}/{str(y+1)[-2:]}" for y in range(2014, 2024)]
    print(f"\n  赛季: {seasons[0]} → {seasons[-1]} (共 {len(seasons)} 季)")
    print(f"  配置: 动态 Kelly + 联赛优化 + 风控层 + 固定基数 10,000")
    print()

    all_results = []
    elos = None
    total_bets = 0
    total_wins = 0
    total_staked = 0.0
    total_returned = 0.0

    print(f"  {'赛季':<12} {'投注':>5} {'胜率':>7} {'ROI':>8} {'投入':>10} {'回报':>10} {'利润':>10} {'风控':>5}")
    print(f"  {'-'*70}")

    for season in seasons:
        r, elos = run_phase2_season(season, 10000.0, elos)
        all_results.append(r)

        total_bets += r["total_bets"]
        total_wins += r["total_wins"]
        total_staked += r["total_staked"]
        total_returned += r["total_returned"]

        wr = r["total_wins"] / r["total_bets"] if r["total_bets"] > 0 else 0
        roi = (r["total_returned"] - r["total_staked"]) / r["total_staked"] if r["total_staked"] > 0 else 0
        profit = r["total_returned"] - r["total_staked"]

        print(f"  {season:<12} {r['total_bets']:>5} {wr:>6.1%} {roi:>+7.1%} "
              f"{r['total_staked']:>10,.0f} {r['total_returned']:>10,.0f} {profit:>+10,.0f} "
              f"{r['risk_events']:>5}")

    overall_wr = total_wins / total_bets if total_bets > 0 else 0
    overall_roi = (total_returned - total_staked) / total_staked if total_staked > 0 else 0
    overall_profit = total_returned - total_staked

    print(f"  {'-'*70}")
    print(f"  {'合计':<12} {total_bets:>5} {overall_wr:>6.1%} {overall_roi:>+7.1%} "
          f"{total_staked:>10,.0f} {total_returned:>10,.0f} {overall_profit:>+10,.0f}")

    # 联赛统计
    print(f"\n  {'联赛':<20} {'投注':>5} {'胜率':>7} {'ROI':>8} {'利润':>10}")
    print(f"  {'-'*55}")
    for lid in ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]:
        lb = sum(r["by_league"].get(lid, {}).get("bets", 0) for r in all_results)
        lw = sum(r["by_league"].get(lid, {}).get("wins", 0) for r in all_results)
        ls = sum(r["by_league"].get(lid, {}).get("staked", 0) for r in all_results)
        lr = sum(r["by_league"].get(lid, {}).get("returned", 0) for r in all_results)
        lwr = lw / lb if lb > 0 else 0
        lroi = (lr - ls) / ls if ls > 0 else 0
        print(f"  {lid:<20} {lb:>5} {lwr:>6.1%} {lroi:>+7.1%} {lr-ls:>+10,.0f}")

    print(f"\n{'='*78}")
    print(f"  v{version} Phase 2 最终验证完成")
    print(f"{'='*78}")