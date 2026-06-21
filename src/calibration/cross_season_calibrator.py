"""
GTO-GameFlow v5.3b 跨赛季因子权重校准框架

使用 Out-of-Sample 方法:
- 训练集: 2022/23 赛季
- 验证集: 2023/24 赛季
- 测试集: 2024/25 赛季

校准参数:
1. 因子类别权重乘数 (BASE / ENHANCED / LEAGUE_SPECIFIC)
2. 关键阈值 (value_threshold / confidence_threshold / kelly_fraction)
3. 单个因子权重微调

目标函数: 最大化 Sharpe Ratio (兼顾收益与风险)
"""
import sys
import os
import json
import math
import random
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.models import MatchContext, BetSelection, BetResult
from src.pipeline.orchestrator import GameFlowPipeline
from src.factors.registry import (
    FACTOR_REGISTRY, LEAGUE_FACTOR_WEIGHTS, FactorCategory,
    get_active_factors, get_factor_weight,
)
from src.calibration.signal_decomposer import (
    SignalDecomposer, PriorShrinkage,
    ELO_DIRECT_FACTORS, ELO_DERIVED_FACTORS, ELO_CORRELATED_FACTORS,
    INDEPENDENT_FACTORS,
)

from tests.test_backtesting_real import (
    LEAGUE_CONFIG, generate_fixture_list,
    generate_realistic_odds, simulate_match_result, update_elo,
)


# ================================================================
# 2022/23 赛季球队数据
# ================================================================

PREMIER_LEAGUE_TEAMS_2223 = [
    ("Manchester City", 1880), ("Arsenal", 1810), ("Liverpool", 1830),
    ("Chelsea", 1780), ("Tottenham", 1730), ("Newcastle", 1680),
    ("Manchester United", 1740), ("Aston Villa", 1630), ("West Ham", 1650),
    ("Brighton", 1610), ("Fulham", 1560), ("Crystal Palace", 1590),
    ("Bournemouth", 1530), ("Brentford", 1550), ("Nottingham Forest", 1500),
    ("Everton", 1550), ("Wolves", 1540), ("Leeds United", 1510),
    ("Leicester City", 1570), ("Southampton", 1520),
]

LA_LIGA_TEAMS_2223 = [
    ("Real Madrid", 1860), ("Barcelona", 1850), ("Atletico Madrid", 1780),
    ("Real Sociedad", 1670), ("Athletic Bilbao", 1660), ("Villarreal", 1670),
    ("Real Betis", 1640), ("Girona", 1580), ("Sevilla", 1680),
    ("Valencia", 1620), ("Osasuna", 1560), ("Getafe", 1560),
    ("Celta Vigo", 1550), ("Mallorca", 1510), ("Rayo Vallecano", 1500),
    ("Alaves", 1480), ("Las Palmas", 1460), ("Granada", 1490),
    ("Cadiz", 1470), ("Almeria", 1450),
]

BUNDESLIGA_TEAMS_2223 = [
    ("Bayern Munich", 1900), ("Bayer Leverkusen", 1800), ("RB Leipzig", 1780),
    ("Borussia Dortmund", 1790), ("Eintracht Frankfurt", 1670), ("Stuttgart", 1620),
    ("Wolfsburg", 1640), ("Freiburg", 1630), ("Hoffenheim", 1610),
    ("Borussia Monchengladbach", 1620), ("Union Berlin", 1580), ("Werder Bremen", 1570),
    ("Augsburg", 1550), ("Mainz 05", 1530), ("Bochum", 1510),
    ("Heidenheim", 1460), ("Darmstadt", 1440), ("FC Koln", 1540),
]

SERIE_A_TEAMS_2223 = [
    ("Inter Milan", 1810), ("AC Milan", 1780), ("Juventus", 1760),
    ("Napoli", 1770), ("Atalanta", 1690), ("Roma", 1700),
    ("Lazio", 1680), ("Fiorentina", 1630), ("Bologna", 1600),
    ("Torino", 1580), ("Monza", 1530), ("Genoa", 1520),
    ("Udinese", 1530), ("Lecce", 1480), ("Empoli", 1490),
    ("Cagliari", 1460), ("Verona", 1470), ("Frosinone", 1460),
    ("Sassuolo", 1510), ("Salernitana", 1440),
]

LIGUE_1_TEAMS_2223 = [
    ("Paris Saint-Germain", 1900), ("Marseille", 1700), ("Monaco", 1690),
    ("Lyon", 1690), ("Lille", 1680), ("Nice", 1640),
    ("Lens", 1630), ("Rennes", 1640), ("Reims", 1560),
    ("Strasbourg", 1570), ("Brest", 1530), ("Toulouse", 1530),
    ("Montpellier", 1540), ("Nantes", 1530), ("Le Havre", 1480),
    ("Metz", 1470), ("Lorient", 1510), ("Clermont", 1470),
]

# 将 2223 数据加入 LEAGUE_CONFIG
_LEAGUE_2223 = {
    "premier_league": PREMIER_LEAGUE_TEAMS_2223,
    "la_liga": LA_LIGA_TEAMS_2223,
    "bundesliga": BUNDESLIGA_TEAMS_2223,
    "serie_a": SERIE_A_TEAMS_2223,
    "ligue_1": LIGUE_1_TEAMS_2223,
}


@dataclass
class CalibrationParams:
    """校准参数"""
    # 因子类别权重乘数
    base_weight_mult: float = 1.0
    enhanced_weight_mult: float = 1.0
    league_weight_mult: float = 1.0

    # 关键阈值
    value_threshold: float = 0.03
    confidence_threshold: float = 0.6
    kelly_fraction: float = 0.25

    # 信号分解
    elo_suppression: float = 1.0

    # 先验收缩
    shrinkage_alpha_high: float = 0.50
    shrinkage_alpha_low: float = 0.10


@dataclass
class CalibrationResult:
    """校准结果"""
    params: CalibrationParams = field(default_factory=CalibrationParams)
    train_roi: float = 0.0
    train_sharpe: float = 0.0
    train_bets: int = 0
    train_win_rate: float = 0.0
    train_max_dd: float = 0.0
    val_roi: float = 0.0
    val_sharpe: float = 0.0
    val_bets: int = 0
    val_win_rate: float = 0.0
    val_max_dd: float = 0.0
    # 综合评分 (越高越好, 优先考虑验证集 Sharpe)
    score: float = 0.0


def compute_calibration_score(val_sharpe: float, val_roi: float,
                               train_sharpe: float, train_roi: float,
                               val_bets: int, val_max_dd: float) -> float:
    """
    校准评分函数。

    优先考虑:
    1. 验证集 Sharpe Ratio (权重 40%)
    2. 验证集 ROI (权重 25%)
    3. 训练-验证 Sharpe 一致性 (权重 15%): 惩罚过拟合, 但要求绝对水平 > 0
    4. 投注频率 (权重 10%): 投注过少说明模型太保守
    5. 最大回撤惩罚 (权重 10%)
    """
    score = 0.0

    # 1. Sharpe Ratio
    score += val_sharpe * 0.40

    # 2. ROI
    score += val_roi * 0.25

    # 3. 一致性: 只对正 Sharpe 计算一致性, 负 Sharpe 直接惩罚
    if val_sharpe > 0 or train_sharpe > 0:
        sharpe_avg = max(0.01, (abs(train_sharpe) + abs(val_sharpe)) / 2)
        sharpe_gap = abs(train_sharpe - val_sharpe)
        consistency = 1.0 - min(1.0, sharpe_gap / sharpe_avg)
        # 一致性用绝对 Sharpe 水平加权, 避免偏好零信号
        score += consistency * min(1.0, sharpe_avg * 5) * 0.15
    else:
        # 两者都负: 低一致性分
        score += 0.0

    # 4. 投注频率 (理想: 每场 0.5-2 注, 基于 380 场)
    bet_rate = val_bets / 380.0
    if bet_rate < 0.1:
        bet_score = bet_rate / 0.1  # 0 → 1
    elif bet_rate <= 0.5:
        bet_score = 1.0
    elif bet_rate <= 2.0:
        bet_score = 1.0 - (bet_rate - 0.5) / 1.5 * 0.3  # 轻微惩罚高频率
    else:
        bet_score = max(0.0, 2.0 - bet_rate) * 0.5
    score += bet_score * 0.10

    # 5. 回撤惩罚
    dd_score = max(0.0, 1.0 - val_max_dd / 0.5)
    score += dd_score * 0.10

    return score


class CrossSeasonCalibrator:
    """
    跨赛季因子权重校准器。

    使用三季数据:
    - 2022/23: 训练
    - 2023/24: 验证
    - 2024/25: 测试 (可选)
    """

    def __init__(self, league_id: str, seed: int = 42):
        self.league_id = league_id
        self.seed = seed
        self.config = LEAGUE_CONFIG[league_id]
        self.teams_2223 = _LEAGUE_2223[league_id]

    def _build_weight_map(self, params: CalibrationParams) -> Dict[str, float]:
        """根据校准参数构建因子权重映射"""
        base_weights = LEAGUE_FACTOR_WEIGHTS[self.league_id]
        weight_map = {}

        for fid, factor in FACTOR_REGISTRY.items():
            if fid == "F14":
                weight_map[fid] = 0.0
                continue

            base_w = base_weights.get(fid, factor.default_weight)

            if factor.category == FactorCategory.BASE:
                multiplier = params.base_weight_mult
            elif factor.category == FactorCategory.ENHANCED:
                multiplier = params.enhanced_weight_mult
            elif factor.category == FactorCategory.LEAGUE_SPECIFIC:
                multiplier = params.league_weight_mult
            else:
                multiplier = 1.0

            weight_map[fid] = base_w * multiplier

        return weight_map

    def run_season_with_params(
        self,
        params: CalibrationParams,
        season: str,
        initial_elos: Optional[Dict[str, float]] = None,
        initial_balance: float = 10000.0,
    ) -> Tuple[float, float, int, float, float, float]:
        """
        使用指定参数运行一个赛季的回测。

        Returns:
            (roi, sharpe, total_bets, win_rate, max_drawdown, final_balance)
        """
        if season == "2022/23":
            teams = self.teams_2223
            base_date = datetime(2022, 8, 5)
            n_teams = len(teams)
        elif season == "2023/24":
            teams = self.config["teams_2324"]
            base_date = datetime(2023, 8, 11)
            n_teams = self.config["n_teams_2324"]
        else:
            teams = self.config["teams_2425"]
            base_date = datetime(2024, 8, 16)
            n_teams = self.config["n_teams_2425"]

        if initial_elos is not None:
            team_elos = dict(initial_elos)
            for name, default_elo in teams:
                if name not in team_elos:
                    team_elos[name] = default_elo
        else:
            team_elos = {t[0]: t[1] for t in teams}

        fixtures = generate_fixture_list(teams)
        days_between = 3.5 * 380 / len(fixtures)

        daily_staked = 0.0
        weekly_staked = 0.0
        current_day = base_date.date()
        current_week = base_date.isocalendar()[1]
        current_month = base_date.month

        # 构建自定义权重
        weight_map = self._build_weight_map(params)

        # 构建权重乘数: 将 weight_map 转为 multiplier (relative to base)
        # 由于 _apply_weights 已经在 compute.py 中应用了基础权重,
        # 我们需要通过 multiplier 来调整
        from src.factors.registry import LEAGUE_FACTOR_WEIGHTS as LFW, FactorCategory
        base_weights = LFW[self.league_id]
        weight_multipliers = {}
        for fid in FACTOR_REGISTRY:
            if fid == "F14":
                continue
            base_w = base_weights.get(fid, FACTOR_REGISTRY[fid].default_weight)
            if base_w == 0:
                weight_multipliers[fid] = 0.0
                continue

            factor = FACTOR_REGISTRY[fid]
            if factor.category == FactorCategory.BASE:
                mult = params.base_weight_mult
            elif factor.category == FactorCategory.ENHANCED:
                mult = params.enhanced_weight_mult
            elif factor.category == FactorCategory.LEAGUE_SPECIFIC:
                mult = params.league_weight_mult
            else:
                mult = 1.0

            if mult != 1.0:
                weight_multipliers[fid] = mult

        pipeline = GameFlowPipeline(
            self.league_id,
            initial_bankroll=initial_balance,
            weight_multipliers=weight_multipliers,
        )
        pipeline.signal_decomposer = SignalDecomposer(elo_suppression=params.elo_suppression)
        pipeline.prior_shrinkage = PriorShrinkage(
            alpha_high=params.shrinkage_alpha_high,
            alpha_low=params.shrinkage_alpha_low,
        )

        total_bets = 0
        total_wins = 0
        total_staked = 0.0
        total_returned = 0.0
        profit_history = []
        equity_curve = [initial_balance]

        for i, (home_team, away_team) in enumerate(fixtures):
            try:
                home_elo = team_elos[home_team]
                away_elo = team_elos[away_team]

                odds_h, odds_d, odds_a = generate_realistic_odds(
                    home_elo, away_elo, seed=self.seed + i,
                )

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
                    match_id=f"{self.league_id}_{season}_{i:04d}",
                    league_id=self.league_id,
                    season=season,
                    matchday=(i % 38) + 1,
                    kickoff_time=match_date,
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
                    actual_outcome = simulate_match_result(
                        home_elo, away_elo, odds_h, odds_d, odds_a,
                        seed=self.seed + i,
                    )
                    outcome_map = {
                        "home_win": BetSelection.HOME_WIN,
                        "draw": BetSelection.DRAW,
                        "away_win": BetSelection.AWAY_WIN,
                    }
                    actual = outcome_map[actual_outcome]
                    placements = pipeline.settle_bets(pipeline_result.placements, actual)

                    for p in placements:
                        daily_staked += p.stake
                        weekly_staked += p.stake
                        total_bets += 1
                        total_staked += p.stake

                        if p.result == BetResult.WIN:
                            total_wins += 1
                            total_returned += p.stake + p.profit_loss
                            profit_history.append(p.profit_loss)
                        elif p.result == BetResult.LOSS:
                            profit_history.append(p.profit_loss)

                    new_home, new_away = update_elo(
                        home_elo, away_elo, actual_outcome,
                        self.config["k_elo"],
                    )
                    team_elos[home_team] = new_home
                    team_elos[away_team] = new_away

                equity_curve.append(pipeline.bankroll_mgr.state.balance)

            except Exception:
                pass

        final_balance = pipeline.bankroll_mgr.state.balance
        roi = ((total_returned - total_staked) / total_staked) if total_staked > 0 else 0.0
        win_rate = total_wins / total_bets if total_bets > 0 else 0.0

        # 最大回撤
        peak = initial_balance
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # 夏普比率
        sharpe = 0.0
        if len(profit_history) > 1:
            mean_return = sum(profit_history) / len(profit_history)
            variance = sum((r - mean_return) ** 2 for r in profit_history) / len(profit_history)
            if variance > 0:
                sharpe = mean_return / math.sqrt(variance)

        return roi, sharpe, total_bets, win_rate, max_dd, final_balance

    def calibrate(self, param_grid: List[Dict]) -> CalibrationResult:
        """
        执行网格搜索校准。

        Args:
            param_grid: 参数网格列表，每个元素是 {param_name: value}

        Returns:
            最佳校准结果
        """
        best_result = None
        best_score = -float('inf')

        print(f"\n  {'─'*50}")
        print(f"  校准 {self.league_id}: {len(param_grid)} 组参数")
        print(f"  {'─'*50}")

        for idx, grid_point in enumerate(param_grid):
            params = CalibrationParams(**grid_point)

            # 训练: 2022/23
            train_roi, train_sharpe, train_bets, train_wr, train_dd, _ = \
                self.run_season_with_params(params, "2022/23")

            # 验证: 2023/24
            val_roi, val_sharpe, val_bets, val_wr, val_dd, _ = \
                self.run_season_with_params(params, "2023/24")

            score = compute_calibration_score(
                val_sharpe, val_roi, train_sharpe, train_roi, val_bets, val_dd
            )

            result = CalibrationResult(
                params=params,
                train_roi=train_roi, train_sharpe=train_sharpe,
                train_bets=train_bets, train_win_rate=train_wr, train_max_dd=train_dd,
                val_roi=val_roi, val_sharpe=val_sharpe,
                val_bets=val_bets, val_win_rate=val_wr, val_max_dd=val_dd,
                score=score,
            )

            if idx % 10 == 0 or idx == len(param_grid) - 1:
                print(f"  [{idx+1}/{len(param_grid)}] "
                      f"train={train_roi:+.2%}/{train_sharpe:+.2f} "
                      f"val={val_roi:+.2%}/{val_sharpe:+.2f} "
                      f"score={score:.3f}")

            if score > best_score:
                best_score = score
                best_result = result

        print(f"\n  最佳: score={best_score:.3f}")
        print(f"    train: ROI={best_result.train_roi:+.2%} "
              f"Sharpe={best_result.train_sharpe:+.2f} "
              f"投注={best_result.train_bets}")
        print(f"    val:   ROI={best_result.val_roi:+.2%} "
              f"Sharpe={best_result.val_sharpe:+.2f} "
              f"投注={best_result.val_bets}")
        print(f"    params: {best_result.params}")

        return best_result


def build_default_param_grid() -> List[Dict]:
    """构建默认参数搜索网格 (精简版, 每组聚焦关键参数)"""
    grid = []

    # 核心参数: 因子权重乘数
    weight_mults = [0.6, 0.8, 1.0, 1.2]
    # 阈值
    value_thresholds = [0.02, 0.03, 0.04]
    confidence_thresholds = [0.50, 0.55, 0.60]
    kelly_fractions = [0.20, 0.25]
    # 先验收缩
    alpha_pairs = [(0.50, 0.10), (0.60, 0.15)]

    random.seed(42)
    sampled = set()

    for _ in range(50):
        wm = random.choice(weight_mults)
        vt = random.choice(value_thresholds)
        ct = random.choice(confidence_thresholds)
        kf = random.choice(kelly_fractions)
        ah, al = random.choice(alpha_pairs)

        key = (wm, vt, ct, kf, ah, al)
        if key in sampled:
            continue
        sampled.add(key)

        grid.append({
            "base_weight_mult": wm,
            "enhanced_weight_mult": wm,
            "league_weight_mult": wm,
            "value_threshold": vt,
            "confidence_threshold": ct,
            "kelly_fraction": kf,
            "shrinkage_alpha_high": ah,
            "shrinkage_alpha_low": al,
        })

    return grid


def calibrate_all_leagues(output_dir: str = None) -> Dict[str, CalibrationResult]:
    """对所有联赛运行校准"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')

    os.makedirs(output_dir, exist_ok=True)

    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    param_grid = build_default_param_grid()

    print("=" * 70)
    print("GTO-GameFlow v5.3b 跨赛季因子权重校准")
    print("=" * 70)
    print(f"  参数网格: {len(param_grid)} 组")
    print(f"  训练集: 2022/23 | 验证集: 2023/24")

    results = {}

    for league_id in leagues:
        calibrator = CrossSeasonCalibrator(league_id)
        result = calibrator.calibrate(param_grid)
        results[league_id] = result

    # 汇总
    print("\n\n" + "=" * 70)
    print("校准结果汇总")
    print("=" * 70)
    print(f"  {'联赛':<20} {'训练ROI':>8} {'训练Sharpe':>10} {'验证ROI':>8} "
          f"{'验证Sharpe':>10} {'投注':>5} {'最佳权重':>6}")
    print(f"  {'─'*70}")

    for lid, r in results.items():
        print(f"  {lid:<20} {r.train_roi:>+7.1%} {r.train_sharpe:>+9.2f} "
              f"{r.val_roi:>+7.1%} {r.val_sharpe:>+9.2f} "
              f"{r.val_bets:>5} {r.params.base_weight_mult:>5.1f}x")

    # 保存结果
    report_path = os.path.join(output_dir, "calibration_v53b.json")
    report_data = {}

    for lid, r in results.items():
        report_data[lid] = {
            "params": {
                "base_weight_mult": r.params.base_weight_mult,
                "enhanced_weight_mult": r.params.enhanced_weight_mult,
                "league_weight_mult": r.params.league_weight_mult,
                "value_threshold": r.params.value_threshold,
                "confidence_threshold": r.params.confidence_threshold,
                "kelly_fraction": r.params.kelly_fraction,
                "shrinkage_alpha_high": r.params.shrinkage_alpha_high,
                "shrinkage_alpha_low": r.params.shrinkage_alpha_low,
            },
            "train": {
                "roi": round(r.train_roi, 4),
                "sharpe": round(r.train_sharpe, 4),
                "bets": r.train_bets,
                "win_rate": round(r.train_win_rate, 4),
                "max_drawdown": round(r.train_max_dd, 4),
            },
            "val": {
                "roi": round(r.val_roi, 4),
                "sharpe": round(r.val_sharpe, 4),
                "bets": r.val_bets,
                "win_rate": round(r.val_win_rate, 4),
                "max_drawdown": round(r.val_max_dd, 4),
            },
            "score": round(r.score, 4),
        }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n  报告已保存: {report_path}")

    return results


if __name__ == "__main__":
    calibrate_all_leagues()