"""
GTO-GameFlow v5.2 因子权重校准框架

基于历史比赛数据的因子权重自动化校准。

方法:
1. 网格搜索 (Grid Search): 遍历权重组合，最大化 ROI/胜率
2. 对数回归 (Logistic Regression): 用比赛结果反推最优权重
3. 联赛特定校准 (League-Specific): 每个联赛独立校准

输出: 校准后的权重配置，可直接写入 league_params.py
"""
import sys
import os
import math
import json
import random
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from itertools import product
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.factors.registry import (
    FACTOR_REGISTRY, LEAGUE_FACTOR_WEIGHTS,
    get_active_factors, get_factor_weight,
    FactorCategory,
)
from src.factors.compute import FactorComputationEngine
from src.engine.probability import ProbabilityEngine
from src.engine.bankroll import BankrollManager, compute_confidence
from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    ProbabilityDistribution,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.config.league_params import get_league_params, LEAGUE_PARAMS


# ================================================================
# 校准数据
# ================================================================

@dataclass
class CalibrationMatch:
    """校准用比赛数据"""
    match_id: str
    league_id: str
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    odds_home: float
    odds_draw: float
    odds_away: float
    actual_outcome: str  # "home_win", "draw", "away_win"
    extra_data: Dict = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """校准结果"""
    league_id: str
    method: str
    n_matches: int
    best_weights: Dict[str, float]  # 因子ID → 权重
    best_roi: float
    best_win_rate: float
    search_iterations: int
    weight_history: List[Dict] = field(default_factory=list)


# ================================================================
# 方法 1: 网格搜索
# ================================================================

class GridSearchCalibrator:
    """
    网格搜索校准器。

    对每个因子在 [0.0, 2.0] 范围内以步长 step 搜索最优权重。
    使用历史比赛数据回测，以 ROI 为目标函数。
    """

    def __init__(
        self,
        league_id: str,
        step: float = 0.25,
        n_iterations: int = 200,
        initial_bankroll: float = 10000,
    ):
        self.league_id = league_id
        self.step = step
        self.n_iterations = n_iterations
        self.initial_bankroll = initial_bankroll

    def generate_weight_candidates(self) -> List[Dict[str, float]]:
        """
        生成候选权重组合。

        策略: 对每个基础因子，生成 [0.5, 0.75, 1.0, 1.25, 1.5] 倍默认权重
        """
        active_factors = get_active_factors(self.league_id)
        base_weights = {
            fid: FACTOR_REGISTRY[fid].default_weight
            for fid in active_factors
        }

        candidates = []
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]

        # 为每个因子生成独立变化
        for fid in active_factors:
            for mult in multipliers:
                w = base_weights.copy()
                w[fid] = base_weights[fid] * mult
                candidates.append(w)

        # 随机采样组合 (避免组合爆炸)
        random.seed(42)
        for _ in range(self.n_iterations):
            w = base_weights.copy()
            # 随机调整 3-5 个因子
            n_adjust = random.randint(3, 5)
            selected = random.sample(active_factors, n_adjust)
            for fid in selected:
                mult = random.choice(multipliers)
                w[fid] = base_weights[fid] * mult
            candidates.append(w)

        return candidates

    def evaluate_weights(
        self,
        weights: Dict[str, float],
        matches: List[CalibrationMatch],
    ) -> Tuple[float, float, float]:
        """
        评估一组权重 — 使用历史数据回测。

        返回: (ROI, 胜率, 总投注数)
        """
        total_bets = 0
        total_wins = 0
        total_staked = 0.0
        total_returned = 0.0

        # 临时覆盖权重
        original_weights = LEAGUE_FACTOR_WEIGHTS.get(self.league_id, {}).copy()
        LEAGUE_FACTOR_WEIGHTS[self.league_id] = weights

        try:
            for match in matches:
                pipeline = GameFlowPipeline(self.league_id, initial_bankroll=self.initial_bankroll)
                ctx = MatchContext(
                    match_id=match.match_id,
                    league_id=match.league_id,
                    season="calibration",
                    matchday=1,
                    kickoff_time=match.extra_data.get("kickoff_time", None) or __import__("datetime").datetime.now(),
                    home_team=match.home_team,
                    away_team=match.away_team,
                    home_elo=match.home_elo,
                    away_elo=match.away_elo,
                    odds_home=match.odds_home,
                    odds_draw=match.odds_draw,
                    odds_away=match.odds_away,
                )

                result = pipeline.run_full(ctx, extra_data=match.extra_data)

                if result.placements:
                    outcome_map = {
                        "home_win": BetSelection.HOME_WIN,
                        "draw": BetSelection.DRAW,
                        "away_win": BetSelection.AWAY_WIN,
                    }
                    actual = outcome_map.get(match.actual_outcome, BetSelection.DRAW)
                    placements = pipeline.settle_bets(result.placements, actual)

                    for p in placements:
                        total_bets += 1
                        total_staked += p.stake
                        if p.result == BetResult.WIN:
                            total_wins += 1
                            total_returned += p.stake + p.profit_loss
                        elif p.result == BetResult.LOSS:
                            pass  # loss → returned = 0

        finally:
            # 恢复原始权重
            LEAGUE_FACTOR_WEIGHTS[self.league_id] = original_weights

        roi = (total_returned - total_staked) / total_staked if total_staked > 0 else -1.0
        win_rate = total_wins / total_bets if total_bets > 0 else 0.0

        return roi, win_rate, total_bets

    def calibrate(
        self,
        matches: List[CalibrationMatch],
    ) -> CalibrationResult:
        """
        运行网格搜索校准。

        参数:
            matches: 历史比赛数据 (至少 100 场)
        """
        candidates = self.generate_weight_candidates()
        result = CalibrationResult(
            league_id=self.league_id,
            method="grid_search",
            n_matches=len(matches),
            best_weights={},
            best_roi=-float("inf"),
            best_win_rate=0.0,
            search_iterations=len(candidates),
        )

        for i, weights in enumerate(candidates):
            if i % 50 == 0:
                print(f"  网格搜索进度: {i}/{len(candidates)}")

            roi, wr, bets = self.evaluate_weights(weights, matches)

            result.weight_history.append({
                "iteration": i,
                "roi": roi,
                "win_rate": wr,
                "bets": bets,
            })

            # 评估: ROI 为主，胜率为辅
            if bets >= 5:  # 至少 5 注才有统计意义
                score = roi * 0.7 + wr * 0.3
                if score > result.best_roi * 0.7 + result.best_win_rate * 0.3:
                    result.best_roi = roi
                    result.best_win_rate = wr
                    result.best_weights = weights.copy()

        return result


# ================================================================
# 方法 2: 对数回归 (简化)
# ================================================================

class LogisticRegressionCalibrator:
    """
    对数回归校准器 (简化版)。

    使用梯度下降优化因子权重，使模型预测概率与实际结果的对数似然最大化。
    """

    def __init__(
        self,
        league_id: str,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        l2_reg: float = 0.001,
    ):
        self.league_id = league_id
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg

    def calculate_log_likelihood(
        self,
        weights: Dict[str, float],
        matches: List[CalibrationMatch],
    ) -> float:
        """
        计算对数似然: Σ log(P_actual)。

        使用当前权重计算每场比赛的模型概率，评估预测精度。
        """
        ll = 0.0
        outcome_map = {"home_win": "home", "draw": "draw", "away_win": "away"}

        # 临时覆盖权重
        original = LEAGUE_FACTOR_WEIGHTS.get(self.league_id, {}).copy()
        LEAGUE_FACTOR_WEIGHTS[self.league_id] = weights

        try:
            prob_engine = ProbabilityEngine(self.league_id)
            factor_engine = FactorComputationEngine(self.league_id)

            for match in matches:
                # 计算因子
                deltas = factor_engine.compute_all(
                    elo_diff=match.home_elo - match.away_elo,
                    xi_rating=match.extra_data.get("xi_rating", 6.0),
                    recent_results=match.extra_data.get("recent_results", [1.5]*5),
                    h2h_results=match.extra_data.get("h2h_results", []),
                    matches_7d=match.extra_data.get("matches_7d", 1),
                    rank_diff=match.extra_data.get("rank_diff", 0),
                    goal_diff=match.extra_data.get("goal_diff", 0.0),
                    xg_diff=match.extra_data.get("xg_diff", 0.0),
                    market_probs={
                        "home": 1.0/match.odds_home,
                        "draw": 1.0/match.odds_draw,
                        "away": 1.0/match.odds_away,
                    },
                )

                # 计算概率
                logits = prob_engine.logit_accumulation(
                    {"home": 1.0/match.odds_home, "draw": 1.0/match.odds_draw, "away": 1.0/match.odds_away},
                    deltas, uniform_prior=False,
                )
                probs = prob_engine.sigmoid_normalization(logits)

                # 对数似然
                actual = outcome_map.get(match.actual_outcome, "draw")
                p_actual = getattr(probs, f"prob_{actual}")
                p_actual = max(1e-10, min(1.0 - 1e-10, p_actual))
                ll += math.log(p_actual)

        finally:
            LEAGUE_FACTOR_WEIGHTS[self.league_id] = original

        return ll

    def gradient_descent_step(
        self,
        weights: Dict[str, float],
        matches: List[CalibrationMatch],
    ) -> Dict[str, float]:
        """
        单步梯度下降 (数值梯度近似)。

        ∂LL/∂w_i ≈ (LL(w_i + ε) - LL(w_i - ε)) / (2ε)
        """
        epsilon = 0.01
        gradients = {}

        # 计算当前 LL
        current_ll = self.calculate_log_likelihood(weights, matches)

        for fid in weights:
            # 前向差分
            w_plus = weights.copy()
            w_plus[fid] += epsilon
            ll_plus = self.calculate_log_likelihood(w_plus, matches)

            # 后向差分
            w_minus = weights.copy()
            w_minus[fid] -= epsilon
            ll_minus = self.calculate_log_likelihood(w_minus, matches)

            # 数值梯度
            grad = (ll_plus - ll_minus) / (2.0 * epsilon)

            # L2 正则化
            grad -= self.l2_reg * weights[fid]

            gradients[fid] = grad

        # 更新权重
        new_weights = weights.copy()
        for fid, grad in gradients.items():
            new_weights[fid] += self.lr * grad
            new_weights[fid] = max(0.0, min(2.0, new_weights[fid]))  # 范围 [0, 2]

        return new_weights

    def calibrate(
        self,
        matches: List[CalibrationMatch],
    ) -> CalibrationResult:
        """运行对数回归校准"""
        result = CalibrationResult(
            league_id=self.league_id,
            method="logistic_regression",
            n_matches=len(matches),
            best_weights={},
            best_roi=0.0,
            best_win_rate=0.0,
            search_iterations=self.n_epochs,
        )

        # 初始化权重
        active_factors = get_active_factors(self.league_id)
        weights = {
            fid: FACTOR_REGISTRY[fid].default_weight
            for fid in active_factors
        }

        best_ll = -float("inf")

        for epoch in range(self.n_epochs):
            if epoch % 10 == 0:
                print(f"  对数回归 epoch {epoch}/{self.n_epochs}")

            weights = self.gradient_descent_step(weights, matches)

            # 评估
            ll = self.calculate_log_likelihood(weights, matches)
            result.weight_history.append({
                "iteration": epoch,
                "log_likelihood": ll,
            })

            if ll > best_ll:
                best_ll = ll
                result.best_weights = weights.copy()

        return result


# ================================================================
# 校准数据生成
# ================================================================

def generate_calibration_data(
    league_id: str,
    n_matches: int = 200,
    seed: int = 42,
) -> List[CalibrationMatch]:
    """
    生成校准用的模拟历史数据。

    实际使用时替换为真实历史数据。
    """
    random.seed(seed)
    matches = []

    lp = get_league_params(league_id)
    import datetime as dt
    base_date = dt.datetime(2024, 8, 16)

    for i in range(n_matches):
        home_elo = random.gauss(1500, 200)
        away_elo = random.gauss(1500, 200)

        # 生成赔率
        fair_home = 1.0 / (1.0 + 10 ** (-(home_elo + 65 - away_elo) / 400))
        fair_draw = 0.25
        fair_away = 1.0 - fair_home - fair_draw
        total = fair_home + fair_draw + fair_away
        fair_home /= total
        fair_draw /= total
        fair_away /= total

        margin = random.uniform(0.05, 0.08)
        odds_h = round(1.0 / (fair_home * (1.0 + margin)), 2)
        odds_d = round(1.0 / (fair_draw * (1.0 + margin)), 2)
        odds_a = round(1.0 / (fair_away * (1.0 + margin)), 2)

        # 实际结果
        r = random.random()
        if r < fair_home:
            outcome = "home_win"
        elif r < fair_home + fair_draw:
            outcome = "draw"
        else:
            outcome = "away_win"

        matches.append(CalibrationMatch(
            match_id=f"{league_id}_cal_{i:04d}",
            league_id=league_id,
            home_team=f"Team_{i*2}",
            away_team=f"Team_{i*2+1}",
            home_elo=home_elo,
            away_elo=away_elo,
            odds_home=odds_h,
            odds_draw=odds_d,
            odds_away=odds_a,
            actual_outcome=outcome,
            extra_data={
                "elo_diff": home_elo - away_elo,
                "recent_results": [random.choice([3, 1, 0]) for _ in range(5)],
                "rank_diff": int((home_elo - away_elo) / 20),
                "goal_diff": (home_elo - away_elo) / 20,
                "xg_diff": (home_elo - away_elo) / 200,
                "data_source_count": 5,
                "match_phase": 1.0,
                "kickoff_time": (base_date + dt.timedelta(days=i)).isoformat(),
            },
        ))

    return matches


# ================================================================
# 便捷函数
# ================================================================

def calibrate_all_leagues(
    n_matches: int = 200,
    method: str = "logistic",
    step: float = 0.25,
    learning_rate: float = 0.01,
    n_epochs: int = 50,
) -> Dict[str, CalibrationResult]:
    """
    对所有 5 个联赛进行校准。

    返回: {league_id: CalibrationResult}
    """
    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    all_results = {}

    for league_id in leagues:
        print(f"\n{'='*60}")
        print(f"校准 {league_id}...")
        print(f"{'='*60}")

        # 生成校准数据
        matches = generate_calibration_data(league_id, n_matches=n_matches)
        print(f"  校准数据: {len(matches)} 场比赛")

        if method == "grid":
            calibrator = GridSearchCalibrator(league_id, step=step)
        else:
            calibrator = LogisticRegressionCalibrator(
                league_id, learning_rate=learning_rate, n_epochs=n_epochs,
            )

        result = calibrator.calibrate(matches)
        all_results[league_id] = result

        print(f"  校准完成: {result.search_iterations} 次迭代")
        if result.best_weights:
            # 显示变化最大的 5 个因子
            changes = []
            for fid, w in result.best_weights.items():
                original = FACTOR_REGISTRY[fid].default_weight
                if abs(w - original) > 0.01:
                    changes.append((fid, original, w, w - original))
            changes.sort(key=lambda x: abs(x[3]), reverse=True)
            print(f"  权重变化最大的 {min(5, len(changes))} 个因子:")
            for fid, orig, new, diff in changes[:5]:
                print(f"    {fid}: {orig:.2f} → {new:.2f} (Δ={diff:+.2f})")

    return all_results


# ================================================================
# 权重导出
# ================================================================

def export_calibrated_weights(
    results: Dict[str, CalibrationResult],
    output_path: str,
):
    """导出校准后的权重为 JSON"""
    output = {}
    for league_id, result in results.items():
        output[league_id] = {
            "weights": result.best_weights,
            "roi": result.best_roi,
            "win_rate": result.best_win_rate,
            "method": result.method,
            "n_matches": result.n_matches,
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n校准权重已导出至: {output_path}")


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    print("GTO-GameFlow v5.2 因子权重校准")
    print("=" * 60)

    # 对数回归校准
    results = calibrate_all_leagues(
        n_matches=200,
        method="logistic",
        n_epochs=30,
    )

    # 导出
    export_calibrated_weights(results, "/workspace/calibrated_weights_v5.2.json")

    print("\n校准完成!")