"""
意甲深度复盘: 追踪每笔投注的盈亏模式
"""
import sys, os, random, json
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import MatchContext, BetSelection, BetResult
from src.pipeline.orchestrator import GameFlowPipeline
from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
from src.factors.registry import FACTOR_REGISTRY, LEAGUE_FACTOR_WEIGHTS as LFW, FactorCategory
from src.utils.i18n import cn_league, cn_selection

from tests.test_backtesting_real import (
    LEAGUE_CONFIG, generate_fixture_list,
    generate_realistic_odds, simulate_match_result, update_elo,
)


def diagnose_serie_a(season: str, seed: int):
    """深度诊断意甲回测"""
    league_id = "serie_a"
    config = LEAGUE_CONFIG[league_id]

    if season == "2023/24":
        teams = config["teams_2324"]
        base_date = datetime(2023, 8, 18)
    else:
        teams = config["teams_2425"]
        base_date = datetime(2024, 8, 17)

    team_elos = {t[0]: t[1] for t in teams}
    fixtures = generate_fixture_list(teams)
    random.seed(seed)
    days_between = 3.5 * 380 / len(fixtures)

    # 加载校准权重
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'reports',
                              'calibrated_weights_v53b.json')
    calib = {}
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib = json.load(f).get(league_id, {})

    weight_multipliers = {}
    if calib:
        base_weights = LFW.get(league_id, {})
        for fid in FACTOR_REGISTRY:
            if fid == "F14":
                continue
            base_w = base_weights.get(fid, FACTOR_REGISTRY[fid].default_weight)
            if base_w == 0:
                continue
            factor = FACTOR_REGISTRY[fid]
            if factor.category == FactorCategory.BASE:
                mult = calib.get("base_weight_mult", 1.0)
            elif factor.category == FactorCategory.ENHANCED:
                mult = calib.get("enhanced_weight_mult", 1.0)
            elif factor.category == FactorCategory.LEAGUE_SPECIFIC:
                mult = calib.get("league_weight_mult", 1.0)
            else:
                mult = 1.0
            if mult != 1.0:
                weight_multipliers[fid] = mult

    pipeline = GameFlowPipeline(
        league_id, initial_bankroll=10000.0,
        weight_multipliers=weight_multipliers,
    )
    if calib:
        pipeline.signal_decomposer = SignalDecomposer(elo_suppression=1.0)
        pipeline.prior_shrinkage = PriorShrinkage(
            alpha_high=calib.get("shrinkage_alpha_high", 0.50),
            alpha_low=calib.get("shrinkage_alpha_low", 0.10),
        )

    daily_staked = 0.0
    weekly_staked = 0.0
    current_day = base_date.date()
    current_week = base_date.isocalendar()[1]
    current_month = base_date.month

    bet_details = []
    stats = {
        "total_bets": 0, "total_wins": 0, "total_staked": 0.0, "total_returned": 0.0,
        "by_elo_diff": defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}),
        "by_odds_range": defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}),
        "by_selection": defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}),
        "by_matchday": defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}),
        "wins_detail": [], "losses_detail": [],
    }

    for i, (home_team, away_team) in enumerate(fixtures):
        try:
            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]
            elo_diff = home_elo - away_elo

            odds_h, odds_d, odds_a = generate_realistic_odds(home_elo, away_elo, seed=seed + i)

            match_date = base_date + timedelta(days=int(i * days_between))
            match_date += timedelta(hours=random.randint(12, 21))

            if match_date.date() != current_day:
                daily_staked = 0.0
                pipeline.bankroll_mgr.reset_daily_loss()
                current_day = match_date.date()
            if match_date.isocalendar()[1] != current_week:
                weekly_staked = 0.0
                pipeline.bankroll_mgr.reset_weekly_loss()
                current_week = match_date.isocalendar()[1]
            if match_date.month != current_month:
                pipeline.bankroll_mgr.reset_monthly_loss()
                current_month = match_date.month

            match = MatchContext(
                match_id=f"serie_a_{season}_{i:04d}",
                league_id=league_id, season=season,
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

            if pipeline_result.placements:
                actual_outcome = simulate_match_result(
                    home_elo, away_elo, odds_h, odds_d, odds_a, seed=seed + i,
                )
                outcome_map = {"home_win": BetSelection.HOME_WIN, "draw": BetSelection.DRAW, "away_win": BetSelection.AWAY_WIN}
                actual = outcome_map[actual_outcome]
                placements = pipeline.settle_bets(pipeline_result.placements, actual)

                for p in placements:
                    daily_staked += p.stake
                    weekly_staked += p.stake
                    stats["total_bets"] += 1
                    stats["total_staked"] += p.stake

                    # 按 Elo 差异分组
                    abs_elo = abs(elo_diff)
                    if abs_elo < 50:
                        bucket = "0-50"
                    elif abs_elo < 100:
                        bucket = "50-100"
                    elif abs_elo < 200:
                        bucket = "100-200"
                    else:
                        bucket = "200+"
                    stats["by_elo_diff"][bucket]["bets"] += 1
                    stats["by_elo_diff"][bucket]["staked"] += p.stake

                    # 按赔率分组
                    if p.odds < 1.30:
                        odds_bucket = "<1.30"
                    elif p.odds < 1.60:
                        odds_bucket = "1.30-1.60"
                    elif p.odds < 2.00:
                        odds_bucket = "1.60-2.00"
                    elif p.odds < 3.00:
                        odds_bucket = "2.00-3.00"
                    else:
                        odds_bucket = "3.00+"
                    stats["by_odds_range"][odds_bucket]["bets"] += 1
                    stats["by_odds_range"][odds_bucket]["staked"] += p.stake

                    # 按投注选择
                    sel_name = p.selection.value
                    stats["by_selection"][sel_name]["bets"] += 1
                    stats["by_selection"][sel_name]["staked"] += p.stake

                    # 按比赛日
                    md = (i % 38) + 1
                    if md <= 10:
                        md_bucket = "1-10"
                    elif md <= 20:
                        md_bucket = "11-20"
                    elif md <= 30:
                        md_bucket = "21-30"
                    else:
                        md_bucket = "31-38"
                    stats["by_matchday"][md_bucket]["bets"] += 1
                    stats["by_matchday"][md_bucket]["staked"] += p.stake

                    if p.result == BetResult.WIN:
                        stats["total_wins"] += 1
                        stats["total_returned"] += p.stake + p.profit_loss
                        stats["by_elo_diff"][bucket]["wins"] += 1
                        stats["by_elo_diff"][bucket]["returned"] += p.stake + p.profit_loss
                        stats["by_odds_range"][odds_bucket]["wins"] += 1
                        stats["by_odds_range"][odds_bucket]["returned"] += p.stake + p.profit_loss
                        stats["by_selection"][sel_name]["wins"] += 1
                        stats["by_selection"][sel_name]["returned"] += p.stake + p.profit_loss
                        stats["by_matchday"][md_bucket]["wins"] += 1
                        stats["by_matchday"][md_bucket]["returned"] += p.stake + p.profit_loss
                        stats["wins_detail"].append({
                            "elo_diff": elo_diff, "odds": p.odds,
                            "selection": sel_name, "stake": p.stake,
                            "profit": p.profit_loss, "actual": actual_outcome,
                        })
                    elif p.result == BetResult.LOSS:
                        stats["losses_detail"].append({
                            "elo_diff": elo_diff, "odds": p.odds,
                            "selection": sel_name, "stake": p.stake,
                            "loss": p.profit_loss, "actual": actual_outcome,
                        })

                new_home, new_away = update_elo(
                    home_elo, away_elo, actual_outcome, config["k_elo"],
                )
                team_elos[home_team] = new_home
                team_elos[away_team] = new_away

        except Exception as e:
            pass

    return stats


def print_breakdown(stats, season):
    print(f"\n{'='*70}")
    print(f" 意甲 {season} 投注结构分析")
    print(f"{'='*70}")
    print(f"  总投注: {stats['total_bets']} | 胜率: {stats['total_wins']/stats['total_bets']*100:.1f}%"
          if stats['total_bets'] > 0 else "  无投注")
    if stats['total_staked'] > 0:
        roi = (stats['total_returned'] - stats['total_staked']) / stats['total_staked']
        print(f"  ROI: {roi:+.1%} | 投注额: {stats['total_staked']:.0f} | 回报: {stats['total_returned']:.0f}")

    for section, title in [
        ("by_elo_diff", "按 Elo 差异"),
        ("by_odds_range", "按赔率区间"),
        ("by_selection", "按投注方向"),
        ("by_matchday", "按赛季阶段"),
    ]:
        print(f"\n  {title}:")
        print(f"  {'分类':<12} {'投注':>6} {'胜率':>7} {'ROI':>7} {'盈亏':>8}")
        print(f"  {'─'*40}")
        for bucket in sorted(stats[section].keys()):
            d = stats[section][bucket]
            if d["bets"] == 0:
                continue
            wr = d["wins"] / d["bets"] * 100 if d["bets"] > 0 else 0
            roi = (d["returned"] - d["staked"]) / d["staked"] * 100 if d["staked"] > 0 else 0
            pnl = d["returned"] - d["staked"]
            print(f"  {bucket:<12} {d['bets']:>6} {wr:>6.1f}% {roi:>+6.1f}% {pnl:>+8.0f}")

    # 亏损分析
    if stats["losses_detail"]:
        losses = stats["losses_detail"]
        print(f"\n  {'─'*50}")
        print(f"  亏损分析 ({len(losses)} 笔亏损):")
        avg_loss_odds = sum(l["odds"] for l in losses) / len(losses)
        avg_loss_stake = sum(abs(l["loss"]) for l in losses) / len(losses)
        # 亏损时的实际结果
        loss_by_actual = defaultdict(int)
        for l in losses:
            loss_by_actual[l["actual"]] += 1
        print(f"    平均赔率: {avg_loss_odds:.2f} | 平均损失: {avg_loss_stake:.0f}")
        print(f"    实际结果: 主胜={loss_by_actual.get('home_win',0)} "
              f"平局={loss_by_actual.get('draw',0)} 客胜={loss_by_actual.get('away_win',0)}")

        # 亏损时的 Elo 差异分布
        loss_elo = [l["elo_diff"] for l in losses]
        print(f"    Elo差异: 均值={sum(loss_elo)/len(loss_elo):.0f} "
              f"中位数={sorted(loss_elo)[len(loss_elo)//2]:.0f}")

        # 亏损时的投注方向
        loss_by_sel = defaultdict(int)
        for l in losses:
            loss_by_sel[l["selection"]] += 1
        print(f"    投注方向: {dict(loss_by_sel)}")


if __name__ == "__main__":
    print("=" * 70)
    print("意甲深度复盘")
    print("=" * 70)

    stats_2324 = diagnose_serie_a("2023/24", seed=42)
    print_breakdown(stats_2324, "2023/24")

    stats_2425 = diagnose_serie_a("2024/25", seed=142)
    print_breakdown(stats_2425, "2024/25")

    # 跨赛季对比
    print(f"\n\n{'='*70}")
    print(" 跨赛季对比")
    print(f"{'='*70}")
    for label, s in [("2324", stats_2324), ("2425", stats_2425)]:
        if s["total_bets"] > 0:
            roi = (s["total_returned"] - s["total_staked"]) / s["total_staked"]
            wr = s["total_wins"] / s["total_bets"]
            print(f"  {label}: {s['total_bets']}注 | WR={wr:.1%} | ROI={roi:+.1%}")