"""
GTO-GameFlow v5.9.2 — Phase 3: 修复串关 + 激活多联赛 + 动态 Kelly
- 修复串关生成逻辑：每 20 场调用 generate_batch()
- 修复 match_outcomes 格式
- 降低非英超价值阈值以激活投注
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
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


def run_phase3_season(
    season: str,
    initial_bankroll: float = 10000.0,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
    enable_parlay: bool = True,
    lower_threshold: bool = True,
) -> Dict:
    """Phase 3: 修复串关 + 激活多联赛"""
    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    opt_params = LeagueOptimizedParams.optimized_defaults()

    # 降低阈值以激活更多投注
    if lower_threshold:
        opt_params["la_liga"].value_threshold = 0.018
        opt_params["bundesliga"].value_threshold = 0.018
        opt_params["serie_a"].value_threshold = 0.020
        opt_params["ligue_1"].value_threshold = 0.020

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    parlay_mgr = ParlayBatchManager(
        max_legs=4, kelly_discount=0.35, max_batch_size=20,
        min_single_value=0.01, min_combined_value=0.015
    )
    kelly_engine = DynamicKellyEngine(base_discount=0.25)
    risk_control = RiskControlLayer()

    if carryover_elos is None:
        carryover_elos = {}

    # 跨联赛共享的 match_outcomes（串关结算需要）
    all_match_outcomes: Dict[str, Tuple[str, BetSelection]] = {}

    result = {
        "season": season,
        "total_bets": 0, "total_wins": 0, "total_staked": 0.0, "total_returned": 0.0,
        "risk_events": 0,
        "parlay_bets": 0, "parlay_wins": 0, "parlay_staked": 0.0, "parlay_returned": 0.0,
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

        # 初始化 Elo
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
        processed = 0
        parlays_added = 0

        ha = {"premier_league": 65, "la_liga": 50, "bundesliga": 60, "serie_a": 55, "ligue_1": 55}.get(lid, 60)
        k_elo = params.elo_k

        for round_idx, match in enumerate(odds_data.values()):
            try:
                processed += 1
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

                        # 风控检查
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
                        parlays_added += len(pipeline_result.proposals)

                    # 保存赛果供串关结算 (正确格式: (actual_outcome, actual_selection))
                    all_match_outcomes[match_ctx.match_id] = (actual_outcome, actual)

                    # ── 关键修复: 每 20 场生成串关 ──
                    if enable_parlay and ((processed + 1) % 20 == 0 or processed == len(odds_data) - 1):
                        if len(parlay_mgr._pending_pool) >= 2:
                            parlays = parlay_mgr.generate_batch(shared_bankroll._get_base_bankroll())
                            # 立即放置到资金池
                            for parlay in parlays:
                                p_stake = parlay.adjusted_stake
                                # 风控
                                approved, adj_stake, _ = risk_control.check_bet(
                                    p_stake, parlay.combined_odds,
                                    shared_bankroll._get_base_bankroll(),
                                    result["total_staked"],
                                    shared_bankroll.state.balance,
                                )
                                if not approved:
                                    continue
                                parlay.adjusted_stake = adj_stake
                                shared_bankroll.state.balance -= adj_stake
                                result["parlay_bets"] += 1
                                result["parlay_staked"] += adj_stake
                                result["total_bets"] += 1
                                result["total_staked"] += adj_stake

                    # 结算已完成串关
                    if enable_parlay:
                        settlements = parlay_mgr.settle_all_ready(all_match_outcomes)
                        for s in settlements:
                            shared_bankroll.state.balance += s.returned
                            result["total_returned"] += s.returned
                            result["parlay_returned"] += s.returned
                            result["parlay_bets"] += 1
                            result["total_bets"] += 1
                            result["total_staked"] += s.stake
                            if s.won:
                                result["parlay_wins"] += 1
                                result["parlay_returned"] += s.returned
                                result["total_wins"] += 1
                                result["total_returned"] += s.returned

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
                            league_result["staked"] += abs(stake) if stake > 0 else tp.kelly_stake
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

    # 赛后最终结算所有串关
    if enable_parlay:
        settlements = parlay_mgr.settle_all_ready(all_match_outcomes)
        for s in settlements:
            result["parlay_bets"] += 1
            result["parlay_staked"] += s.stake
            result["total_bets"] += 1
            result["total_staked"] += s.stake
            if s.won:
                result["parlay_wins"] += 1
                result["total_wins"] += 1
                result["total_returned"] += s.returned

    result["by_strategy"]["parlay"] = [
        result["parlay_bets"], result["parlay_wins"],
        result["parlay_staked"], result["parlay_returned"],
    ]

    return result, carryover_elos


if __name__ == "__main__":
    print("=" * 78)
    print(f"  GTO-GameFlow {version} Phase 3: 修复串关 + 激活五大联赛")
    print("=" * 78)

    print("\n  配置:")
    print("  - 动态 Kelly 分数")
    print("  - 风控层")
    print("  - 修复串关生成逻辑")
    print("  - 降低非英超价值阈值")
    print("  - 固定基数 10,000 (10赛季: 2014/15 → 2023/24)")

    seasons = [f"{y}/{str(y+1)[-2:]}" for y in range(2014, 2024)]
    print()

    all_results = []
    elos = None
    total_bets = 0
    total_wins = 0
    total_staked = 0.0
    total_returned = 0.0

    print(f"  {'赛季':<10} {'总投注':>6} {'单场':>6} {'串关':>6} {'胜率':>7} {'ROI':>8} {'利润':>10}")
    print(f"  {'-'*60}")

    for season in seasons:
        r, elos = run_phase3_season(season, 10000.0, elos,
            enable_parlay=True, lower_threshold=True)
        all_results.append(r)

        total_bets += r["total_bets"]
        total_wins += r["total_wins"]
        total_staked += r["total_staked"]
        total_returned += r["total_returned"]

        wr = r["total_wins"] / r["total_bets"] if r["total_bets"] > 0 else 0
        roi = (r["total_returned"] - r["total_staked"]) / r["total_staked"] if r["total_staked"] > 0 else 0
        profit = r["total_returned"] - r["total_staked"]
        single = r["total_bets"] - r["parlay_bets"]
        print(f"  {season:<10} {r['total_bets']:>6} {single:>6} {r['parlay_bets']:>6} "
              f"{wr:>6.1%} {roi:>+7.1%} {profit:+10,.0f}")

    overall_wr = total_wins / total_bets if total_bets > 0 else 0
    overall_roi = (total_returned - total_staked) / total_staked if total_staked > 0 else 0
    overall_profit = total_returned - total_staked

    print(f"  {'-'*60}")
    print(f"  {'合计':<10} {total_bets:>6} {total_bets - sum(r['parlay_bets'] for r in all_results):>6} "
          f"{sum(r['parlay_bets'] for r in all_results):>6} "
          f"{overall_wr:>6.1%} {overall_roi:>+7.1%} {overall_profit:+10,.0f}")

    print(f"\n  联赛明细:")
    print(f"  {'联赛':<16} {'投注':>6} {'胜率':>7} {'ROI':>8} {'利润':>10}")
    print(f"  {'-'*50}")
    league_totals: Dict[str, Dict] = defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0})
    for r in all_results:
        for lid, lr in r["by_league"].items():
            for k in lr:
                league_totals[lid][k] += lr[k]
    for lid, lt in league_totals.items():
        if lt["bets"] > 0:
            wr = lt["wins"] / lt["bets"] if lt["bets"] > 0 else 0
            roi = (lt["returned"] - lt["staked"]) / lt["staked"] if lt["staked"] > 0 else 0
            print(f"  {lid:<16} {lt['bets']:>6} {wr:>6.1%} {roi:>+7.1%} {(lt['returned'] - lt['staked']):+10,.0f}")

    print(f"\n  策略明细:")
    print(f"  {'策略':<12} {'投注':>6} {'胜率':>7} {'ROI':>8} {'利润':>10}")
    print(f"  {'-'*50}")
    for strat, [nb, nw, st, ret] in all_results[-1]["by_strategy"].items():
        nb_total = sum(r["by_strategy"][strat][0] for r in all_results)
        nw_total = sum(r["by_strategy"][strat][1] for r in all_results)
        st_total = sum(r["by_strategy"][strat][2] for r in all_results)
        ret_total = sum(r["by_strategy"][strat][3] for r in all_results)
        if nb_total > 0:
            wr = nw_total / nb_total
            roi = (ret_total - st_total) / st_total if st_total > 0 else 0
            name_map = {
                "1x2": "胜平负", "asian": "亚盘", "over_under": "大小球", "parlay": "串关",
            }
            print(f"  {name_map[strat]:<12} {nb_total:>6} {wr:>6.1%} {roi:>+7.1%} {(ret_total - st_total):+10,.0f}")

    print(f"\n{'='*78}")
    print(f"  v{version} Phase 3 验证完成")
    print(f"{'='*78}")