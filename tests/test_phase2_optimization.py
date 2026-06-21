"""
GTO-GameFlow v5.9.2 — Phase 2: 动态 Kelly 分数 + 联赛特定参数优化
- 根据市场效率动态调整 Kelly 折扣
- 基于 Walk-Forward 验证结果优化联赛级参数
- 整合风控层
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random, json, math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.data.models import (
    MatchContext, BetSelection, BetResult, AsianHandicapResult,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.strategies.strategy_orchestrator import ParlayBatchManager, create_orchestrator
from src.data.historical_odds_loader import load_odds_for_season, get_real_odds
from src.data.orthogonal_sources import OrthogonalDataGenerator
from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
from tests.test_backtesting_real import _load_calibrated_weights, _build_weight_multipliers
from src.config.settings import GlobalConfig
version = GlobalConfig.version


# ═══════════════════════════════════════════════════════════════
# 动态 Kelly 分数引擎
# ═══════════════════════════════════════════════════════════════

class DynamicKellyEngine:
    """
    动态 Kelly 分数计算器。

    根据多个信号动态调整 Kelly 折扣：
    - value_signal: 模型发现的价值大小 → 越大越激进
    - odds_std: 赔率离散度 (市场分歧) → 越大越保守
    - market_efficiency: 市场效率评分 → 越高越保守
    - recent_performance: 近期表现 (回撤/连胜) → 动态调整

    基准折扣: 25% (1/4 Kelly)
    动态范围: 15% ~ 40%
    """

    def __init__(
        self,
        base_discount: float = 0.25,
        min_discount: float = 0.10,
        max_discount: float = 0.40,
        lookback_window: int = 20,
    ):
        self.base_discount = base_discount
        self.min_discount = min_discount
        self.max_discount = max_discount
        self.lookback_window = lookback_window

        # 近期结果追踪
        self.recent_results: List[float] = []  # 最近 20 注的盈亏
        self.running_sharpe: float = 0.0
        self.running_win_rate: float = 0.5

    def compute_discount(
        self,
        value_signal: float,
        odds_std: float,
        market_efficiency: float,
        model_prob: float,
        market_prob: float,
    ) -> float:
        """
        动态计算 Kelly 折扣。

        discount = base × value_adj × disagreement_adj × efficiency_adj × performance_adj

        Args:
            value_signal: 模型发现的价值 (model_prob - market_prob 的标准化)
            odds_std: 赔率离散度
            market_efficiency: 市场效率评分
            model_prob: 模型概率
            market_prob: 市场隐含概率
        """
        discount = self.base_discount

        # 1. 价值信号调整 (+30% / -20%)
        value = abs(model_prob - market_prob) / max(market_prob, 0.01)
        if value > 0.15:
            value_adj = 1.30  # 强价值信号 → 更激进
        elif value > 0.08:
            value_adj = 1.15
        elif value > 0.03:
            value_adj = 1.0
        else:
            value_adj = 0.80  # 弱价值信号 → 更保守

        # 2. 市场分歧调整: 高分歧 → 保守
        if odds_std > 0.10:
            disagreement_adj = 0.75
        elif odds_std > 0.05:
            disagreement_adj = 0.90
        else:
            disagreement_adj = 1.05

        # 3. 市场效率调整: 高效市场 → 保守
        if market_efficiency > 0.08:
            efficiency_adj = 0.75
        elif market_efficiency > 0.04:
            efficiency_adj = 0.90
        else:
            efficiency_adj = 1.05

        # 4. 近期表现调整
        performance_adj = self._compute_performance_adj()

        discount = (self.base_discount * value_adj * disagreement_adj *
                    efficiency_adj * performance_adj)

        # 约束在合理范围内
        return max(self.min_discount, min(self.max_discount, discount))

    def _compute_performance_adj(self) -> float:
        """基于近期表现动态调整"""
        if len(self.recent_results) < 5:
            return 1.0  # 数据不足，使用基准

        n = min(len(self.recent_results), self.lookback_window)
        recent = self.recent_results[-n:]

        # 计算近期胜率
        wins = sum(1 for r in recent if r > 0)
        self.running_win_rate = wins / n if n > 0 else 0.5

        # 近期盈亏 (sharpe-like)
        if n >= 5:
            mean_r = np.mean(recent)
            std_r = np.std(recent) if len(recent) > 1 else 1.0
            self.running_sharpe = mean_r / std_r if std_r > 0 else 0.0

        # 调整逻辑
        if self.running_win_rate > 0.6 and self.running_sharpe > 0.3:
            return 1.20  # 手风顺 → 适度激进
        elif self.running_win_rate < 0.35 or self.running_sharpe < -0.3:
            return 0.60  # 回撤严重 → 极度保守
        elif self.running_win_rate < 0.45:
            return 0.80  # 略低于预期 → 偏保守
        else:
            return 1.0

    def record_result(self, profit: float):
        """记录投注结果"""
        self.recent_results.append(profit)
        if len(self.recent_results) > self.lookback_window * 3:
            self.recent_results = self.recent_results[-self.lookback_window:]


# ═══════════════════════════════════════════════════════════════
# 联赛特定参数优化
# ═══════════════════════════════════════════════════════════════

@dataclass
class LeagueOptimizedParams:
    """联赛最优参数"""
    league_id: str
    kelly_discount: float = 0.25
    value_threshold: float = 0.02
    elo_k: float = 24
    home_advantage: float = 65
    shrinkage_alpha_high: float = 0.50
    shrinkage_alpha_low: float = 0.10

    # 经 Walk-Forward 验证的优化参数
    @classmethod
    def optimized_defaults(cls) -> Dict[str, "LeagueOptimizedParams"]:
        return {
            "premier_league": cls("premier_league", kelly_discount=0.28, value_threshold=0.015,
                elo_k=24, home_advantage=65, shrinkage_alpha_high=0.55, shrinkage_alpha_low=0.12),
            "la_liga": cls("la_liga", kelly_discount=0.22, value_threshold=0.025,
                elo_k=20, home_advantage=50, shrinkage_alpha_high=0.45, shrinkage_alpha_low=0.08),
            "bundesliga": cls("bundesliga", kelly_discount=0.25, value_threshold=0.020,
                elo_k=22, home_advantage=60, shrinkage_alpha_high=0.50, shrinkage_alpha_low=0.10),
            "serie_a": cls("serie_a", kelly_discount=0.20, value_threshold=0.025,
                elo_k=20, home_advantage=55, shrinkage_alpha_high=0.45, shrinkage_alpha_low=0.08),
            "ligue_1": cls("ligue_1", kelly_discount=0.20, value_threshold=0.030,
                elo_k=20, home_advantage=55, shrinkage_alpha_high=0.40, shrinkage_alpha_low=0.08),
        }


# ═══════════════════════════════════════════════════════════════
# 风控层
# ═══════════════════════════════════════════════════════════════

class RiskControlLayer:
    """
    简化风控层。

    规则:
    1. 单注上限: 不超过总资金的 5%
    2. 单日上限: 不超过总资金的 15%
    3. 回撤熔断: 日回撤 > 8% 或 赛季回撤 > 20% → 暂停
    4. 赔率上下限: [1.30, 6.00]
    """

    def __init__(self):
        self.daily_staked = 0.0
        self.daily_pnl = 0.0
        self.season_peak = 0.0
        self.season_drawdown = 0.0
        self.is_paused = False
        self.pause_reason = ""

    def check_bet(
        self,
        stake: float,
        odds: float,
        bankroll: float,
        daily_staked: float,
        current_balance: float,
    ) -> Tuple[bool, float, str]:
        """
        检查投注是否通过风控。
        返回 (approved, adjusted_stake, reason)
        """
        # 1. 赔率检查
        if odds < 1.30 or odds > 6.00:
            return False, 0.0, f"赔率 {odds:.2f} 超出 [1.30, 6.00]"

        # 2. 单注上限
        max_single = bankroll * 0.05
        if stake > max_single:
            return True, max_single, f"单注上限 {max_single:.0f}"

        # 3. 单日上限
        if daily_staked + stake > bankroll * 0.15:
            return False, 0.0, f"单日上限 {bankroll*0.15:.0f}"

        # 4. 回撤熔断
        if self.season_peak > 0:
            dd = (self.season_peak - current_balance) / self.season_peak
            if dd > 0.20:
                self.is_paused = True
                self.pause_reason = f"赛季回撤 {dd:.1%} > 20%"
                return False, 0.0, self.pause_reason

        return True, stake, ""

    def update(self, balance: float, daily_pnl: float):
        """更新风控状态"""
        if balance > self.season_peak:
            self.season_peak = balance
        if self.season_peak > 0:
            self.season_drawdown = (self.season_peak - balance) / self.season_peak

        if abs(daily_pnl) / max(self.season_peak, 1) > 0.08:
            self.is_paused = True
            self.pause_reason = f"日回撤 {abs(daily_pnl)/self.season_peak:.1%} > 8%"


# ═══════════════════════════════════════════════════════════════
# 整合回测
# ═══════════════════════════════════════════════════════════════

def run_optimized_season(
    season: str,
    initial_bankroll: float = 10000.0,
    use_dynamic_kelly: bool = True,
    use_calibrated_decomposer: bool = True,
    use_risk_control: bool = True,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """运行整合了 Phase 2 优化的单赛季五大联赛回测"""
    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    opt_params = LeagueOptimizedParams.optimized_defaults()

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    parlay_mgr = ParlayBatchManager(max_legs=4, kelly_discount=0.35, max_batch_size=20,
        min_single_value=0.01, min_combined_value=0.015)
    kelly_engine = DynamicKellyEngine(base_discount=0.25) if use_dynamic_kelly else None
    risk_control = RiskControlLayer() if use_risk_control else None

    if carryover_elos is None:
        carryover_elos = {}

    result = {
        "season": season,
        "total_bets": 0, "total_wins": 0, "total_staked": 0.0, "total_returned": 0.0,
        "daily_staked": 0.0, "daily_pnl": 0.0,
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

        # 校准权重
        calib = _load_calibrated_weights().get(lid, {})
        weight_multipliers = _build_weight_multipliers(lid, calib)

        pipeline = GameFlowPipeline(lid, initial_bankroll=initial_bankroll,
                                     weight_multipliers=weight_multipliers)
        pipeline.set_bankroll_manager(shared_bankroll)

        # SignalDecomposer
        if use_calibrated_decomposer:
            # 加载校准数据
            calib_path = os.path.join(os.path.dirname(__file__), "..",
                                      f"src/data/calibrated_decomposer_{lid}.json")
            if os.path.exists(calib_path):
                with open(calib_path) as f:
                    calib_data = json.load(f)
                decomposer = SignalDecomposer(elo_suppression=1.0)
                # 用校准的 R² 更新比率
                for cls_name, cls_set in [
                    ("elo_direct", {"F1"}),
                    ("elo_derived", {"F7", "F8", "F19", "F27", "F29", "F33", "F37", "F39", "F40"}),
                    ("elo_correlated", {"F3", "F5", "F20", "F38"}),
                ]:
                    cls_factors = [f for f in cls_set if f in calib_data]
                    if cls_factors:
                        avg_r2 = np.mean([calib_data[f]["avg_r2"] for f in cls_factors])
                        decomposer.elo_explained_ratios[cls_name] = avg_r2
                pipeline.signal_decomposer = decomposer
                pipeline.prior_shrinkage = PriorShrinkage(
                    alpha_high=params.shrinkage_alpha_high,
                    alpha_low=params.shrinkage_alpha_low,
                )

        ortho_gen = OrthogonalDataGenerator(lid, seed=hash(season + lid) % 10000)
        orchestrator = create_orchestrator(lid)
        match_outcomes = {}
        processed = 0

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

                    # 1X2 结算
                    placements = pipeline.settle_bets(pipeline_result.placements, actual)
                    for p in placements:
                        # 动态 Kelly 调整
                        if use_dynamic_kelly and kelly_engine and pipeline_result.fused_probs:
                            imp = 1/odds_h + 1/odds_d + 1/odds_a
                            sel_map = {
                                BetSelection.HOME_WIN: "home",
                                BetSelection.DRAW: "draw",
                                BetSelection.AWAY_WIN: "away",
                            }
                            selection_key = sel_map.get(p.selection, "home")
                            market_prob = {"home": (1/odds_h)/imp, "draw": (1/odds_d)/imp,
                                           "away": (1/odds_a)/imp}[selection_key]
                            model_prob = {
                                "home": pipeline_result.fused_probs.prob_home,
                                "draw": pipeline_result.fused_probs.prob_draw,
                                "away": pipeline_result.fused_probs.prob_away,
                            }[selection_key]
                            discount = kelly_engine.compute_discount(
                                value_signal=abs(model_prob - market_prob),
                                odds_std=extra.get("odds_std", 0.05),
                                market_efficiency=extra.get("market_efficiency", 0.05),
                                model_prob=model_prob,
                                market_prob=market_prob,
                            )
                            # 应用动态折扣
                            p.stake *= (discount / 0.25)  # 相对于基准 25%

                        # 风控检查
                        if use_risk_control and risk_control:
                            approved, adjusted_stake, reason = risk_control.check_bet(
                                p.stake, p.odds if hasattr(p, 'odds') else 2.0,
                                shared_bankroll._get_base_bankroll(),
                                result["daily_staked"],
                                shared_bankroll.state.balance,
                            )
                            if not approved:
                                result["risk_events"] += 1
                                continue
                            p.stake = adjusted_stake

                        result["total_bets"] += 1
                        result["total_staked"] += p.stake
                        result["daily_staked"] += p.stake
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

                        if kelly_engine:
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
                            result["daily_staked"] += abs(stake) if stake > 0 else ap.kelly_stake
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
                            result["daily_staked"] += abs(stake) if stake > 0 else tp.kelly_stake
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
                    exp_h = 1.0 / (1.0 + 10 ** (-(home_elo + params.home_advantage - away_elo) / 400.0))
                    exp_a = 1.0 - exp_h
                    team_elos[match.home_team] = home_elo + params.elo_k * (actual_h - exp_h)
                    team_elos[match.away_team] = away_elo + params.elo_k * (actual_a - exp_a)

            except Exception:
                continue

        carryover_elos[lid] = team_elos
        result["by_league"][lid] = league_result

    return result, carryover_elos


if __name__ == "__main__":
    print("=" * 78)
    print(f"  GTO-GameFlow {version} Phase 2: 动态 Kelly + 联赛优化 + 风控")
    print("=" * 78)

    test_seasons = ["2022/23", "2023/24"]

    # 运行基准 (无优化)
    print("\n  [1] 基准回测 (无优化, 固定 Kelly 25%)")
    print("  " + "-" * 50)
    elos_bench = None
    for season in test_seasons:
        r, elos_bench = run_optimized_season(season, 10000.0,
            use_dynamic_kelly=False, use_calibrated_decomposer=False,
            use_risk_control=False, carryover_elos=elos_bench)
        roi = (r["total_returned"] - r["total_staked"]) / r["total_staked"] if r["total_staked"] > 0 else 0
        print(f"    {season}: {r['total_bets']:>4}注, ROI={roi:>+7.1%}, "
              f"利润={r['total_returned']-r['total_staked']:+,.0f}")

    # 运行动态 Kelly
    print("\n  [2] 动态 Kelly 回测")
    print("  " + "-" * 50)
    elos_dyn = None
    for season in test_seasons:
        r, elos_dyn = run_optimized_season(season, 10000.0,
            use_dynamic_kelly=True, use_calibrated_decomposer=False,
            use_risk_control=False, carryover_elos=elos_dyn)
        roi = (r["total_returned"] - r["total_staked"]) / r["total_staked"] if r["total_staked"] > 0 else 0
        print(f"    {season}: {r['total_bets']:>4}注, ROI={roi:>+7.1%}, "
              f"利润={r['total_returned']-r['total_staked']:+,.0f}")

    # 运行完整优化 (校准 + 动态 Kelly + 风控)
    print("\n  [3] 完整优化回测 (校准SignalDecomposer + 动态Kelly + 风控)")
    print("  " + "-" * 50)
    elos_full = None
    for season in test_seasons:
        r, elos_full = run_optimized_season(season, 10000.0,
            use_dynamic_kelly=True, use_calibrated_decomposer=True,
            use_risk_control=True, carryover_elos=elos_full)
        roi = (r["total_returned"] - r["total_staked"]) / r["total_staked"] if r["total_staked"] > 0 else 0
        wr = r["total_wins"] / r["total_bets"] if r["total_bets"] > 0 else 0
        print(f"    {season}: {r['total_bets']:>4}注, 胜率={wr:>6.1%}, ROI={roi:>+7.1%}, "
              f"利润={r['total_returned']-r['total_staked']:+,.0f}, "
              f"风控拦截={r['risk_events']}")

        # 各联赛明细
        for lid, lr in r["by_league"].items():
            if lr["bets"] > 0:
                lroi = (lr["returned"] - lr["staked"]) / lr["staked"] if lr["staked"] > 0 else 0
                print(f"      {lid}: {lr['bets']:>4}注, ROI={lroi:>+7.1%}")

    print(f"\n{'='*78}")
    print(f"  v{version} Phase 2 优化完成")
    print(f"{'='*78}")